from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pedf import DayOptimizer, load_config, load_days, run_day, signals_from_trace, trace_carbon


EXPECTED_DATA_SHA256 = "a2e0993423c2f01417f43172df171b4cb603605e1bf76d558aa1647bbede8608"
ACTIONS = ("hvac_kw", "ev_kw", "task_kw", "ess_charge_kw", "ess_discharge_kw")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2), encoding="utf-8")


def validate(config: dict, result: dict, trace: dict, cap: float | None = None) -> None:
    tolerance = config["tolerance"]
    checks = {
        "energy balance": result["energy_balance_max_kw"] <= tolerance["energy_balance_kw"],
        "SOCP recovery": result["socp_gap_max"] <= tolerance["socp_gap"],
        "branch capacity": result["branch_capacity_violation_kw"] <= 2e-5,
        "voltage": result["voltage_violation_pu"] <= tolerance["constraint"],
        "ESS complementarity": result["simultaneous_charge_discharge_kw"] <= tolerance["constraint"],
        "carbon closure": trace["carbon_closure_relative"] <= tolerance["carbon_closure_relative"],
    }
    if cap is not None:
        checks["cost cap"] = result["cost_gbp"] - cap <= 1e-4
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(", ".join(failed))


def validate_saved(config: dict, row: dict) -> None:
    tolerance = config["tolerance"]
    failed = (
        row["energy_balance_max_kw"] > tolerance["energy_balance_kw"]
        or row["socp_gap_max"] > tolerance["socp_gap"]
        or row["branch_capacity_violation_kw"] > 2e-5
        or row["voltage_violation_pu"] > tolerance["constraint"]
        or row["carbon_closure_relative"] > tolerance["carbon_closure_relative"]
    )
    if failed:
        raise RuntimeError(f"{row['method']} saved residual gate failed")


def signal_sets(day: dict, config: dict, b0: dict, mci: np.ndarray) -> dict:
    reference = trace_carbon(
        b0, config, day["carbon_g_per_kwh"], day["load_kw"], memory=True
    )
    nodes = config["device_nodes"]
    return {
        "B1_BULK_ACI": signals_from_trace(reference, day["carbon_g_per_kwh"], "bulk"),
        "B3_NODAL_ACI_MEM": signals_from_trace(reference, day["carbon_g_per_kwh"], "memory"),
        "P_RESPONSE_MCI": {
            "hvac": mci[nodes["hvac"]],
            "charge": mci[nodes["ess"]],
            "discharge": mci[nodes["ess"]],
            "ev": mci[nodes["ev"]],
            "task": mci[nodes["task"]],
        },
    }


def result_row(method: str, delta: float, result: dict, trace: dict) -> dict:
    return {
        "date": result["date"],
        "method": method,
        "cost_budget": delta,
        "cost_gbp": result["cost_gbp"],
        "source_emissions_kg": result["source_emissions_kg"],
        "runtime_s": result["runtime_s"],
        "energy_balance_max_kw": result["energy_balance_max_kw"],
        "socp_gap_max": result["socp_gap_max"],
        "branch_capacity_violation_kw": result["branch_capacity_violation_kw"],
        "voltage_violation_pu": result["voltage_violation_pu"],
        "carbon_closure_relative": trace["carbon_closure_relative"],
    }


def summarize(frame: pd.DataFrame, seed: int) -> dict:
    pivot = frame.pivot(index="date", columns=["cost_budget", "method"], values="source_emissions_kg")
    summary = {"complete_days": int(len(pivot)), "cost_budgets": {}}
    for delta in (0.02, 0.10):
        b0 = pivot[(delta, "B0_ECON")]
        b4 = pivot[(delta, "B4_DIRECT_SOURCE")]
        block = {}
        for method in ("B1_BULK_ACI", "B3_NODAL_ACI_MEM", "P_RESPONSE_MCI", "B4_DIRECT_SOURCE"):
            values = pivot[(delta, method)]
            change = 100 * (values - b0) / b0
            regret = 100 * (values - b4) / b0
            valid = values.notna()
            block[method] = {
                "days": int(valid.sum()),
                "median_change_vs_b0_pctpoint": float(np.median(change[valid])),
                "median_regret_vs_b4_pctpoint": float(np.median(regret[valid])),
                "emission_reduction_days": int((change < 0).sum()),
            }
        summary["cost_budgets"][str(delta)] = block
    effect = 100 * (
        pivot[(0.02, "B3_NODAL_ACI_MEM")] - pivot[(0.02, "P_RESPONSE_MCI")]
    ) / pivot[(0.02, "B0_ECON")]
    rng = np.random.default_rng(seed)
    draws = rng.choice(effect.to_numpy(), size=(10000, len(effect)), replace=True)
    rebound = 100 * (
        pivot[(0.10, "P_RESPONSE_MCI")] - pivot[(0.10, "B0_ECON")]
    ) / pivot[(0.10, "B0_ECON")]
    regrets = pd.DataFrame(
        {
            method: 100 * (pivot[(0.02, method)] - pivot[(0.02, "B4_DIRECT_SOURCE")]) / pivot[(0.02, "B0_ECON")]
            for method in ("B1_BULK_ACI", "B3_NODAL_ACI_MEM", "P_RESPONSE_MCI")
        }
    )
    summary["primary_p_vs_b3_2pct"] = {
        "median_pctpoint": float(np.median(effect)),
        "bootstrap_95_ci_median": [float(x) for x in np.quantile(np.median(draws, axis=1), [0.025, 0.975])],
        "positive_days": int((effect > 0).sum()),
    }
    summary["p_rebound_10pct"] = {
        "median_pctpoint": float(np.median(rebound)),
        "rebound_days": int((rebound > 0).sum()),
    }
    summary["closest_nonreference_to_b4_2pct"] = regrets.idxmin(axis=1).value_counts().to_dict()
    summary["max_physical_residuals"] = {
        key: float(frame[key].max())
        for key in (
            "energy_balance_max_kw",
            "socp_gap_max",
            "branch_capacity_violation_kw",
            "voltage_violation_pu",
            "carbon_closure_relative",
        )
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=28)
    args = parser.parse_args()
    data_path = Path(os.environ.get("PEDF_DATA_PATH", ROOT / "data/processed/timeseries.csv"))
    output = Path(os.environ.get("PEDF_OUTPUT_PATH", ROOT / "artifacts/ieee33dc_main"))
    config_path = ROOT / "configs/pedf33dc.json"
    if sha256(data_path) != EXPECTED_DATA_SHA256:
        raise RuntimeError("joined input SHA-256 does not match the frozen 1,344-row input")
    config = load_config(config_path)
    days = load_days(data_path)[: args.days]
    output.mkdir(parents=True, exist_ok=True)
    (output / "arrays").mkdir(exist_ok=True)
    rows, diagnostics, failures, method_failures = [], [], [], []
    started = time.time()
    for index, day in enumerate(days, 1):
        print(f"IEEE33DC {index}/{len(days)} {day['date']}", flush=True)
        try:
            outcome = run_day(
                config,
                day,
                cost_budget=0.02,
                progress=lambda _: None,
                allow_method_failures=True,
            )
            if not outcome["summary"]["mci"]["stable"]:
                raise RuntimeError("MCI perturbation stability gate failed")
            metrics = {row["method"]: row for row in outcome["metrics"]}
            model = DayOptimizer(config, day)
            b0 = model.solve_econ()
            if b0 is None:
                raise RuntimeError("B0 failed")
            mci = outcome["arrays"]["mci_g_per_kwh"]
            sets = signal_sets(day, config, b0, mci)
            for failure in outcome["method_failures"]:
                method_failures.append({"date": day["date"], "cost_budget": 0.02, **failure})
            for method in ("B0_ECON", "B1_BULK_ACI", "B3_NODAL_ACI_MEM"):
                if method not in metrics:
                    continue
                row = metrics[method].copy()
                row["cost_budget"] = 0.02
                validate_saved(config, row)
                rows.append(result_row(method, 0.02, row, row))
            p_row = metrics["P_DUAL_MCI_ACI"].copy()
            p_row["method"] = "P_RESPONSE_MCI"
            p_row["cost_budget"] = 0.02
            validate_saved(config, p_row)
            rows.append(result_row("P_RESPONSE_MCI", 0.02, p_row, p_row))
            for delta in (0.02, 0.10):
                cap = b0["cost_gbp"] * (1 + delta)
                selected = {"B4_DIRECT_SOURCE": model.solve_emissions(cap)}
                if delta == 0.10:
                    selected["B0_ECON"] = b0
                    selected.update({method: model.solve_signal(signal, cap) for method, signal in sets.items()})
                for method, result in selected.items():
                    if result is None:
                        if method == "B1_BULK_ACI":
                            method_failures.append({"date": day["date"], "cost_budget": delta, "method": method, "reason": "solve failed"})
                            continue
                        raise RuntimeError(f"{method} failed at {delta}")
                    trace = trace_carbon(result, config, day["carbon_g_per_kwh"], day["load_kw"], memory=True)
                    try:
                        validate(config, result, trace, cap)
                    except RuntimeError as exc:
                        if method == "B1_BULK_ACI":
                            method_failures.append({"date": day["date"], "cost_budget": delta, "method": method, "reason": str(exc)})
                            continue
                        raise
                    rows.append(result_row(method, delta, result, trace))
            diagnostics.append({"date": day["date"], **outcome["summary"]})
            np.savez_compressed(output / "arrays" / f"{day['date']}.npz", **outcome["arrays"])
            pd.DataFrame(rows).to_csv(output / "daily_metrics.csv", index=False)
            write_json(output / "daily_diagnostics.json", diagnostics)
            write_json(output / "method_failures.json", method_failures)
        except Exception as exc:
            failures.append({"date": day["date"], "error": f"{type(exc).__name__}: {exc}"})
            write_json(output / "failures.json", failures)
    frame = pd.DataFrame(rows)
    write_json(output / "failures.json", failures)
    write_json(output / "method_failures.json", method_failures)
    if failures or frame["date"].nunique() != len(days):
        raise RuntimeError(f"incomplete run: {len(failures)} failed days")
    summary = summarize(frame, config["seed"])
    write_json(output / "summary.json", summary)
    write_json(output / "run_manifest.json", {
        "status": "complete",
        "reproduction_command": "python scripts/run_ieee33_extension.py --days 28",
        "requested_days": args.days,
        "method_failures": method_failures,
        "elapsed_s": time.time() - started,
        "seed": config["seed"],
        "config_sha256": sha256(config_path),
        "dataset_sha256": sha256(data_path),
        "source_code_sha256": {path.name: sha256(path) for path in sorted((ROOT / "src/pedf").glob("*.py"))},
        "environment": {"python": sys.version, "platform": platform.platform(), "cvxpy": cp.__version__, "numpy": np.__version__, "pandas": pd.__version__, "solvers": cp.installed_solvers()},
    })
    print(json.dumps(plain(summary), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
