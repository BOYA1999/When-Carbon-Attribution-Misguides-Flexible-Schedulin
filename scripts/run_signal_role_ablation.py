from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pedf import DayOptimizer, load_config, load_days, trace_carbon


MAIN = ROOT / "artifacts" / "experiment" / "pedf8_main_20260802_v1"
OUT = ROOT / "artifacts" / "analysis" / "campaign_20260802_signal_role_adversarial"
CONTRACT = ROOT / "configs" / "signal_role_contract.json"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ACTION_KEYS = ("hvac_kw", "ev_kw", "task_kw", "ess_charge_kw", "ess_discharge_kw")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(config: dict, result: dict, cost_cap: float) -> None:
    tol = config["tolerance"]
    if result["energy_balance_max_kw"] > tol["energy_balance_kw"]:
        raise RuntimeError("energy balance failed")
    if result["branch_capacity_violation_kw"] > max(tol["constraint"], 2e-5):
        raise RuntimeError("branch capacity failed")
    if result["voltage_violation_pu"] > tol["constraint"]:
        raise RuntimeError("voltage failed")
    if result["simultaneous_charge_discharge_kw"] > tol["constraint"]:
        raise RuntimeError("ESS complementarity failed")
    if result["negative_price_export_kw"] > tol["constraint"]:
        raise RuntimeError("negative-price export failed")
    if result["cost_gbp"] - cost_cap > 1e-4:
        raise RuntimeError("cost cap failed")


def signals(arrays: np.lib.npyio.NpzFile, alpha: float) -> dict[str, np.ndarray]:
    nodal = arrays["reference_nodal_aci_g_per_kwh"]
    storage = arrays["reference_storage_ci_g_per_kwh"]
    mci = arrays["mci_g_per_kwh"]
    aci = {"hvac": nodal[2], "charge": nodal[3], "discharge": storage, "ev": nodal[6], "task": nodal[7]}
    marginal = {"hvac": mci[2], "charge": mci[3], "discharge": mci[3], "ev": mci[6], "task": mci[7]}
    return {key: (1 - alpha) * aci[key] + alpha * marginal[key] for key in aci}


def shuffled_signals(arrays: np.lib.npyio.NpzFile, seed: int, date: str) -> dict[str, np.ndarray]:
    base = signals(arrays, 1.0)
    day_seed = seed + int(date.replace("-", ""))
    permutation = np.random.default_rng(day_seed).permutation(len(base["hvac"]))
    return {key: value[permutation] for key, value in base.items()}


def bootstrap_median(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.median(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def summarize(frame: pd.DataFrame, failures: list[dict], seed: int, expected_days: int, elapsed_s: float) -> dict:
    complete_dates = [date for date, group in frame.groupby("date") if len(group) == len(ALPHAS) + 1]
    complete = frame[frame["date"].isin(complete_dates)].copy()
    auditable = complete if len(complete) else frame
    alpha_rows = auditable[auditable["variant"].str.startswith("alpha_")]
    curve = []
    for alpha, group in alpha_rows.groupby("alpha", sort=True):
        effect = group["effect_vs_alpha0_pctpoint"].to_numpy()
        curve.append({
            "alpha": float(alpha),
            "median_effect_vs_alpha0_pctpoint": float(np.median(effect)),
            "iqr_pctpoint": [float(x) for x in np.quantile(effect, [0.25, 0.75])],
            "positive_days": int(np.sum(effect > 0)),
            "median_reduction_vs_b0_pct": float(group["reduction_vs_b0_pct"].median()),
            "median_cost_increase_pct": float(group["cost_increase_pct"].median()),
            "median_schedule_runtime_s": float(group["runtime_s"].median()),
        })
    shuffled = auditable[auditable["variant"] == "mci_time_shuffled"]
    actual = auditable[auditable["variant"] == "alpha_1.00"]
    paired = actual[["date", "source_emissions_kg", "b0_emissions_kg"]].merge(
        shuffled[["date", "source_emissions_kg"]], on="date", suffixes=("_actual", "_shuffled")
    )
    order_gain = 100 * (paired["source_emissions_kg_shuffled"] - paired["source_emissions_kg_actual"]) / paired["b0_emissions_kg"]
    medians = [item["median_effect_vs_alpha0_pctpoint"] for item in curve]
    return {
        "campaign_id": "pedf8_signal_role_adversarial_20260802_v1",
        "status": "complete" if len(complete_dates) == expected_days and not failures else "partial",
        "expected_days": expected_days,
        "complete_days": len(complete_dates),
        "failed_cases": len(failures),
        "failures": failures,
        "curve": curve,
        "curve_monotone_non_decreasing": bool(medians and np.all(np.diff(medians) >= -1e-9)),
        "best_alpha_by_median": float(curve[int(np.argmax(medians))]["alpha"]) if medians else None,
        "actual_mci_vs_shuffled": {
            "median_actual_order_gain_pctpoint": float(np.median(order_gain)) if len(order_gain) else None,
            "bootstrap_95ci": bootstrap_median(order_gain.to_numpy(), seed) if len(order_gain) else None,
            "actual_order_better_days": int(np.sum(order_gain > 0)),
            "paired_days": int(len(order_gain)),
        },
        "endpoint_reproduction": {
            "alpha0_max_abs_emissions_difference_kg": float(auditable.loc[auditable["variant"] == "alpha_0.00", "parent_emissions_abs_difference_kg"].max()),
            "alpha1_max_abs_emissions_difference_kg": float(auditable.loc[auditable["variant"] == "alpha_1.00", "parent_emissions_abs_difference_kg"].max()),
        },
        "maximum_checks": {
            "cost_cap_violation_gbp": float(auditable["cost_cap_violation_gbp"].max()),
            "branch_capacity_violation_kw": float(auditable["branch_capacity_violation_kw"].max()),
            "voltage_violation_pu": float(auditable["voltage_violation_pu"].max()),
            "carbon_closure_relative": float(auditable["carbon_closure_relative"].max()),
        },
        "elapsed_s": elapsed_s,
        "comparability": "same dataset, topology, feasible set, cost cap, solver, flexible degrees of freedom and one scheduling solve per candidate; MCI construction overhead is not equalized",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = OUT / "smoke" if args.smoke else OUT
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs" / "pedf8.json")
    days = load_days(ROOT / "data" / "processed" / "timeseries.csv")
    if args.smoke:
        days = days[:1]
    parent = pd.read_csv(MAIN / "metrics.csv").set_index(["date", "method"])
    rows, failures = [], []
    started = time.time()
    for day in days:
        date = day["date"]
        print(f"SIGNAL_ROLE {date}", flush=True)
        arrays = np.load(MAIN / "arrays" / f"{date}.npz")
        b0 = parent.loc[(date, "B0_ECON")]
        cost_cap = float(b0["cost_gbp"] * 1.02)
        candidates = [(f"alpha_{alpha:.2f}", alpha, signals(arrays, alpha)) for alpha in ALPHAS]
        candidates.append(("mci_time_shuffled", np.nan, shuffled_signals(arrays, config["seed"], date)))
        results = {}
        model = DayOptimizer(config, day)
        for variant, alpha, signal in candidates:
            try:
                result = model.solve_signal(signal, cost_cap)
                if result is None:
                    raise RuntimeError("signal solve failed")
                validate(config, result, cost_cap)
                results[variant] = (alpha, result)
            except Exception as exc:
                failures.append({"date": date, "variant": variant, "error": f"{type(exc).__name__}: {exc}"})
        if "alpha_0.00" not in results:
            continue
        alpha0_emissions = results["alpha_0.00"][1]["source_emissions_kg"]
        alpha0_actions = results["alpha_0.00"][1]
        for variant, (alpha, result) in results.items():
            trace = trace_carbon(result, config, day["carbon_g_per_kwh"], day["load_kw"], memory=True)
            parent_method = "B3_NODAL_ACI_MEM" if variant == "alpha_0.00" else "P_DUAL_MCI_ACI" if variant == "alpha_1.00" else None
            parent_difference = abs(result["source_emissions_kg"] - float(parent.loc[(date, parent_method), "source_emissions_kg"])) if parent_method else np.nan
            rows.append({
                "date": date,
                "variant": variant,
                "alpha": alpha,
                "b0_emissions_kg": b0["source_emissions_kg"],
                "source_emissions_kg": result["source_emissions_kg"],
                "effect_vs_alpha0_pctpoint": 100 * (alpha0_emissions - result["source_emissions_kg"]) / b0["source_emissions_kg"],
                "reduction_vs_b0_pct": 100 * (b0["source_emissions_kg"] - result["source_emissions_kg"]) / b0["source_emissions_kg"],
                "cost_gbp": result["cost_gbp"],
                "cost_increase_pct": 100 * (result["cost_gbp"] / b0["cost_gbp"] - 1),
                "runtime_s": result["runtime_s"],
                "max_action_difference_vs_alpha0_kw": max(float(np.max(np.abs(result[key] - alpha0_actions[key]))) for key in ACTION_KEYS),
                "cost_cap_violation_gbp": result["cost_cap_violation_gbp"],
                "branch_capacity_violation_kw": result["branch_capacity_violation_kw"],
                "voltage_violation_pu": result["voltage_violation_pu"],
                "carbon_closure_relative": trace["carbon_closure_relative"],
                "parent_emissions_abs_difference_kg": parent_difference,
                "scheduling_solves": 1,
            })
        pd.DataFrame(rows).to_csv(output / "partial.csv", index=False)
        (output / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "results.csv", index=False)
    summary = summarize(frame, failures, config["seed"], len(days), time.time() - started)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "contract_sha256": sha256(CONTRACT),
        "script_sha256": sha256(ROOT / "scripts" / "run_signal_role_ablation.py"),
        "config_sha256": sha256(ROOT / "configs" / "pedf8.json"),
        "dataset_sha256": sha256(ROOT / "data" / "processed" / "timeseries.csv"),
        "parent_metrics_sha256": sha256(MAIN / "metrics.csv"),
        "seed": config["seed"],
        "mode": "smoke" if args.smoke else "full",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
