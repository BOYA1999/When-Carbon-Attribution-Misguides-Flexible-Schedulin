from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pedf import load_config, load_days, run_day


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def paired_bootstrap(values: np.ndarray, seed: int, replicates: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True)
    estimates = np.median(draws, axis=1)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def summarize(stage: str, day_summaries: list[dict], failures: list[dict], seed: int) -> dict:
    p1_go = bool(
        len(day_summaries) >= 3
        and not failures
        and all(item["mci"]["stable"] for item in day_summaries[:3])
        and all(not item["p_b3_same_solution"] for item in day_summaries[:3])
    )
    gate_p1 = "NOT_EVALUATED" if len(day_summaries) < 3 else ("GO" if p1_go else "STOP")
    result = {
        "stage": stage,
        "complete_days": len(day_summaries),
        "failed_days": len(failures),
        "failures": failures,
        "gate_p1": gate_p1,
    }
    effects = np.array(
        [item["primary_improvement_pctpoint"] for item in day_summaries], float
    )
    if len(effects):
        result["all_available_effect"] = {
            "median_pctpoint": float(np.median(effects)),
            "mean_pctpoint": float(np.mean(effects)),
            "iqr_pctpoint": [float(x) for x in np.quantile(effects, [0.25, 0.75])],
            "positive_days": int(np.sum(effects > 0)),
            "days": int(len(effects)),
        }
    if stage == "main" and len(day_summaries) >= 28 and not failures:
        primary = effects[:21]
        confirmation = effects[21:28]
        interval = paired_bootstrap(primary, seed)
        h1 = bool(np.median(primary) >= 1.0 and interval[0] > 0 and np.median(confirmation) > 0)
        inversion_days = np.array(
            [
                item["aci_mci_mismatch"]["rank_inversion_rate"] > 0.25
                for item in day_summaries
            ]
        )
        result["primary_21_day"] = {
            "median_pctpoint": float(np.median(primary)),
            "mean_pctpoint": float(np.mean(primary)),
            "iqr_pctpoint": [float(x) for x in np.quantile(primary, [0.25, 0.75])],
            "bootstrap_95_ci_median": interval,
            "positive_days": int(np.sum(primary > 0)),
        }
        result["confirmation_7_day"] = {
            "median_pctpoint": float(np.median(confirmation)),
            "positive_days": int(np.sum(confirmation > 0)),
        }
        result["gate_p2"] = "PASS" if h1 else "FAIL"
        result["gate_d1_available"] = bool(np.mean(inversion_days) >= 0.20)
    return result


def write_markdown(output: Path, run_summary: dict, day_summaries: list[dict], metrics: pd.DataFrame) -> None:
    effect = run_summary.get("all_available_effect", {})
    lines = [
        f"# {output.name}",
        "",
        f"- Stage: `{run_summary['stage']}`",
        f"- Complete days: {run_summary['complete_days']}; failed days: {run_summary['failed_days']}",
        f"- Gate P1: `{run_summary['gate_p1']}`",
    ]
    if effect:
        lines += [
            f"- Median P improvement over B3: {effect['median_pctpoint']:.4f} B0 percentage points",
            f"- Positive days: {effect['positive_days']}/{effect['days']}",
        ]
    if "gate_p2" in run_summary:
        primary = run_summary["primary_21_day"]
        confirm = run_summary["confirmation_7_day"]
        lines += [
            f"- Gate P2: `{run_summary['gate_p2']}`",
            f"- Primary median and 95% CI: {primary['median_pctpoint']:.4f} "
            f"[{primary['bootstrap_95_ci_median'][0]:.4f}, {primary['bootstrap_95_ci_median'][1]:.4f}]",
            f"- Temporal confirmation median: {confirm['median_pctpoint']:.4f}",
        ]
    lines += ["", "## Daily primary effect", "", "| Date | Improvement, percentage points | Stable MCI | ACI versus MCI rho | Rank inversion rate |", "|---|---:|---|---:|---:|"]
    for item in day_summaries:
        lines.append(
            f"| {item['date']} | {item['primary_improvement_pctpoint']:.4f} | "
            f"{item['mci']['stable']} | {item['aci_mci_mismatch']['spearman_rho']:.4f} | "
            f"{item['aci_mci_mismatch']['rank_inversion_rate']:.4f} |"
        )
    lines += [
        "",
        "## Comparability",
        "",
        "All methods use the same days, network, devices, feasible set, 2% cost cap, CLARABEL tolerances and ex post carbon accounting. "
        "The memory free B2 trace generates only its action signal. Final B2 metrics retain memory aware accounting for a common source emissions comparison.",
        "",
        "## Failure denominator",
        "",
        f"Requested days: {run_summary['complete_days'] + run_summary['failed_days']}; complete days: {run_summary['complete_days']}; failed days: {run_summary['failed_days']}.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if metrics.empty:
        metrics_text = "# Mean method metrics\n\nNo complete method rows.\n"
    else:
        method_table = metrics.groupby("method")[["cost_gbp", "source_emissions_kg", "carbon_closure_relative", "runtime_s"]].mean()
        metrics_text = "# Mean method metrics\n\n" + method_table.to_markdown(floatfmt=".6g") + "\n"
    (output / "metrics.md").write_text(metrics_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "pilot", "main"), required=True)
    parser.add_argument("--days", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    default_days = {"smoke": 1, "pilot": 3, "main": 28}
    requested_days = args.days or default_days[args.stage]
    run_id = f"pedf8_{args.stage}_20260802_v1"
    output = ROOT / (args.output or f"artifacts/experiment/{run_id}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "arrays").mkdir(exist_ok=True)
    start = datetime.now(timezone.utc)
    config_path = ROOT / "configs" / "pedf8.json"
    data_path = ROOT / "data" / "processed" / "timeseries.csv"
    config = load_config(config_path)
    days = load_days(data_path)[:requested_days]
    metrics_rows = []
    day_summaries = []
    failures = []
    for index, day in enumerate(days, 1):
        print(f"RUN {index}/{len(days)} {day['date']}", flush=True)
        try:
            outcome = run_day(config, day, progress=lambda value: print(value, flush=True))
            metrics_rows.extend(outcome["metrics"])
            day_summaries.append(outcome["summary"])
            np.savez_compressed(output / "arrays" / f"{day['date']}.npz", **outcome["arrays"])
            print(
                f"DONE {day['date']} effect={outcome['summary']['primary_improvement_pctpoint']:.6f} "
                f"mci_stable={outcome['summary']['mci']['stable']}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"date": day["date"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {day['date']} {type(exc).__name__}: {exc}", flush=True)
            if args.stage in {"smoke", "pilot"}:
                break

    metrics = pd.DataFrame(metrics_rows)
    run_summary = summarize(args.stage, day_summaries, failures, config["seed"])
    end = datetime.now(timezone.utc)
    metrics.to_csv(output / "metrics.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(serializable(metrics_rows), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "day_summaries.json").write_text(
        json.dumps(serializable(day_summaries), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "evaluation_summary.json").write_text(
        json.dumps(serializable(run_summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cvxpy": cp.__version__,
        "installed_solvers": cp.installed_solvers(),
        "cpu_count": os.cpu_count(),
    }
    (output / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_manifest = {
        "run_id": run_id,
        "stage": args.stage,
        "status": "success" if not failures else "partial",
        "command": " ".join(sys.argv),
        "dataset": "data/processed/timeseries.csv",
        "dataset_sha256": sha256(data_path),
        "config": "configs/pedf8.json",
        "config_sha256": sha256(config_path),
        "seed": config["seed"],
        "requested_days": requested_days,
        "complete_days": len(day_summaries),
        "failed_days": len(failures),
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "elapsed_s": time.time() - start.timestamp(),
        "changed_files": [
            "configs/pedf8.json",
            "src/pedf/model.py",
            "src/pedf/carbon.py",
            "src/pedf/experiment.py",
            "scripts/run_experiment.py",
        ],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(output, run_summary, day_summaries, metrics)
    claim = {
        "claim": "At a 2% cost budget, P lowers source emissions relative to B3",
        "metric": "100*(E_B3-E_P)/E_B0",
        "expected_direction": "positive; main median >=1 and bootstrap lower >0",
        "observed": run_summary.get("primary_21_day", run_summary.get("all_available_effect")),
        "verdict": "supported"
        if run_summary.get("gate_p2") == "PASS"
        else ("inconclusive" if args.stage != "main" else "refuted"),
    }
    (output / "claim_validation.json").write_text(
        json.dumps(claim, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files = [path for path in output.rglob("*") if path.is_file()]
    artifact_manifest = {
        "run_id": run_id,
        "files": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files)
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(serializable(run_summary), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
