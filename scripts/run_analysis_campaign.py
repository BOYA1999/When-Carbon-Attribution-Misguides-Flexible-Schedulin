from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pedf import DayOptimizer, compute_mci, load_config, load_days, signals_from_trace, trace_carbon


CAMPAIGN = ROOT / "artifacts" / "analysis" / "campaign_20260802"
MAIN = ROOT / "artifacts" / "experiment" / "pedf8_main_20260802_v1"
REPRESENTATIVE = (0, 4, 8, 12, 16, 20, 24)


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def builtin(value):
    if isinstance(value, dict):
        return {key: builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [builtin(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def validate(config: dict, result: dict, cost_cap: float | None = None) -> None:
    tol = config["tolerance"]
    if result["energy_balance_max_kw"] > tol["energy_balance_kw"]:
        raise RuntimeError("energy balance failed")
    if result["branch_capacity_violation_kw"] > max(tol["constraint"], 2e-5):
        raise RuntimeError("branch capacity failed")
    if result["voltage_violation_pu"] > tol["constraint"]:
        raise RuntimeError("voltage failed")
    if result["simultaneous_charge_discharge_kw"] > tol["constraint"]:
        raise RuntimeError("ESS complementarity failed")
    if cost_cap is not None and result["cost_gbp"] - cost_cap > 1e-4:
        raise RuntimeError("cost cap failed")


def run_pair(config: dict, day: dict, delta: float = 0.02) -> dict:
    model = DayOptimizer(config, day)
    b0 = model.solve_econ()
    if b0 is None:
        raise RuntimeError("B0 failed")
    validate(config, b0)
    reference = trace_carbon(
        b0, config, day["carbon_g_per_kwh"], day["load_kw"], memory=True
    )
    mci, mci_diagnostics = compute_mci(model, b0, progress=lambda value: print(value, flush=True))
    if not mci_diagnostics["stable"]:
        raise RuntimeError("MCI unstable")
    b3_signals = signals_from_trace(reference, day["carbon_g_per_kwh"], "memory")
    p_signals = {
        "hvac": mci[2],
        "charge": mci[3],
        "discharge": mci[3],
        "ev": mci[6],
        "task": mci[7],
    }
    cap = b0["cost_gbp"] * (1 + delta)
    b3 = model.solve_signal(b3_signals, cap)
    proposed = model.solve_signal(p_signals, cap)
    if b3 is None or proposed is None:
        raise RuntimeError("B3 or P failed")
    validate(config, b3, cap)
    validate(config, proposed, cap)
    b3_trace = trace_carbon(
        b3, config, day["carbon_g_per_kwh"], day["load_kw"], memory=True
    )
    p_trace = trace_carbon(
        proposed, config, day["carbon_g_per_kwh"], day["load_kw"], memory=True
    )
    return {
        "date": day["date"],
        "delta": delta,
        "effect_pctpoint": 100
        * (b3["source_emissions_kg"] - proposed["source_emissions_kg"])
        / b0["source_emissions_kg"],
        "b0_cost_gbp": b0["cost_gbp"],
        "b0_emissions_kg": b0["source_emissions_kg"],
        "b3_cost_gbp": b3["cost_gbp"],
        "b3_emissions_kg": b3["source_emissions_kg"],
        "p_cost_gbp": proposed["cost_gbp"],
        "p_emissions_kg": proposed["source_emissions_kg"],
        "b3_carbon_closure": b3_trace["carbon_closure_relative"],
        "p_carbon_closure": p_trace["carbon_closure_relative"],
        "max_constraint_violation": max(
            b3["branch_capacity_violation_kw"],
            proposed["branch_capacity_violation_kw"],
            b3["voltage_violation_pu"],
            proposed["voltage_violation_pu"],
        ),
        "mci_stable": mci_diagnostics["stable"],
        "runtime_s": b0["runtime_s"] + b3["runtime_s"] + proposed["runtime_s"],
    }


def save_partial(output: Path, rows: list[dict], failures: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "partial.json").write_text(
        json.dumps(builtin({"rows": rows, "failures": failures}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def finalize(output: Path, slice_id: str, rows: list[dict], failures: list[dict], config: dict, started: float) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "results.csv", index=False)
    summary = {
        "slice_id": slice_id,
        "status": "complete" if not failures else "partial",
        "complete_cases": len(rows),
        "failed_cases": len(failures),
        "failures": failures,
        "elapsed_s": time.time() - started,
    }
    if "effect_pctpoint" in frame:
        summary["effect"] = {
            "median_pctpoint": float(frame["effect_pctpoint"].median()),
            "iqr_pctpoint": [float(x) for x in frame["effect_pctpoint"].quantile([0.25, 0.75])],
            "positive_cases": int((frame["effect_pctpoint"] > 0).sum()),
        }
    (output / "summary.json").write_text(
        json.dumps(builtin(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "slice_id": slice_id,
        "config_sha256": sha256(ROOT / "configs" / "pedf8.json"),
        "dataset_sha256": sha256(ROOT / "data" / "processed" / "timeseries.csv"),
        "seed": config["seed"],
        "output": str(output.relative_to(ROOT)).replace("\\", "/"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(builtin(summary), ensure_ascii=False), flush=True)


def pareto(config: dict, days: list[dict], output: Path) -> None:
    rows, failures = [], []
    budgets = config["cost_budgets"]
    for day in days:
        print(f"PARETO {day['date']}", flush=True)
        try:
            cached = np.load(MAIN / "arrays" / f"{day['date']}.npz")
            model = DayOptimizer(config, day)
            b0 = model.solve_econ()
            nodal = cached["reference_nodal_aci_g_per_kwh"]
            storage = cached["reference_storage_ci_g_per_kwh"]
            mci = cached["mci_g_per_kwh"]
            signals = {
                "B3": {
                    "hvac": nodal[2], "charge": nodal[3], "discharge": storage,
                    "ev": nodal[6], "task": nodal[7],
                },
                "P": {
                    "hvac": mci[2], "charge": mci[3], "discharge": mci[3],
                    "ev": mci[6], "task": mci[7],
                },
            }
            for delta in budgets:
                cap = b0["cost_gbp"] * (1 + delta)
                for method, method_signal in signals.items():
                    if delta == 0:
                        result = b0
                    else:
                        result = model.solve_signal(method_signal, cap)
                        if result is None:
                            raise RuntimeError(f"{method} delta={delta} failed")
                        validate(config, result, cap)
                    rows.append(
                        {
                            "date": day["date"], "method": method, "delta": delta,
                            "cost_gbp": result["cost_gbp"],
                            "cost_increase_pct": 100 * (result["cost_gbp"] / b0["cost_gbp"] - 1),
                            "source_emissions_kg": result["source_emissions_kg"],
                            "emissions_change_vs_b0_pct": 100 * (result["source_emissions_kg"] / b0["source_emissions_kg"] - 1),
                            "b0_emissions_kg": b0["source_emissions_kg"],
                        }
                    )
            save_partial(output, rows, failures)
        except Exception as exc:
            failures.append({"date": day["date"], "error": f"{type(exc).__name__}: {exc}"})
            save_partial(output, rows, failures)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "method_results.csv", index=False)
    comparison = frame.pivot_table(index=["date", "delta"], columns="method", values="source_emissions_kg").reset_index()
    b0_emissions = frame.groupby(["date", "delta"])["b0_emissions_kg"].first().reset_index()
    comparison = comparison.merge(b0_emissions, on=["date", "delta"])
    comparison["effect_pctpoint"] = 100 * (comparison["B3"] - comparison["P"]) / comparison["b0_emissions_kg"]
    comparison.to_csv(output / "paired_effects.csv", index=False)
    finalize(output, "S3_COST_PARETO", comparison.to_dict("records"), failures, config, STARTED)


def variant_runs(config: dict, days: list[dict], output: Path, slice_id: str, variants: dict[str, dict]) -> None:
    rows, failures = [], []
    for variant, variant_config in variants.items():
        for index in REPRESENTATIVE:
            day = days[index]
            print(f"{slice_id} {variant} {day['date']}", flush=True)
            try:
                result = run_pair(variant_config, day)
                result["variant"] = variant
                rows.append(result)
            except Exception as exc:
                failures.append({"variant": variant, "date": day["date"], "error": f"{type(exc).__name__}: {exc}"})
            save_partial(output, rows, failures)
    finalize(output, slice_id, rows, failures, config, STARTED)


def stress(config: dict, days: list[dict], output: Path) -> None:
    features = np.array(
        [
            [np.sum(day["load_kw"]), np.sum(day["shortwave_w_m2"]), np.mean(day["carbon_g_per_kwh"]), np.mean(day["price_gbp_per_kwh"])]
            for day in days
        ]
    )
    scaled = (features - np.median(features, axis=0)) / np.maximum(np.std(features, axis=0), 1e-12)
    selected = int(np.argmin(np.sum(scaled**2, axis=1)))
    base = days[selected]
    rng = np.random.default_rng(config["seed"])
    rows, failures = [], []
    rho = 0.7
    sigmas = {"load_kw": 0.05, "shortwave_w_m2": 0.10, "carbon_g_per_kwh": 0.05, "price_gbp_per_kwh": 0.08}
    for scenario in range(30):
        perturbed = copy.deepcopy(base)
        for key, sigma in sigmas.items():
            innovation = rng.normal(size=48)
            process = np.zeros(48)
            process[0] = innovation[0]
            for t in range(1, 48):
                process[t] = rho * process[t - 1] + np.sqrt(1 - rho**2) * innovation[t]
            factor = np.clip(1 + sigma * process, 0.5, 1.5)
            perturbed[key] = np.asarray(base[key]) * factor
        perturbed["load_kw"] = np.maximum(perturbed["load_kw"], 1.0)
        perturbed["shortwave_w_m2"] = np.maximum(perturbed["shortwave_w_m2"], 0.0)
        perturbed["carbon_g_per_kwh"] = np.maximum(perturbed["carbon_g_per_kwh"], 1.0)
        perturbed["date"] = f"{base['date']}__scenario_{scenario:02d}"
        print(f"S4 scenario={scenario:02d}", flush=True)
        try:
            result = run_pair(config, perturbed)
            result["scenario"] = scenario
            result["base_date"] = base["date"]
            rows.append(result)
        except Exception as exc:
            failures.append({"scenario": scenario, "error": f"{type(exc).__name__}: {exc}"})
        save_partial(output, rows, failures)
    finalize(output, "S4_AR1_STRESS", rows, failures, config, STARTED)


def scale(config: dict, days: list[dict], output: Path) -> None:
    day = days[REPRESENTATIVE[3]]
    rows, failures = [], []
    try:
        start = time.perf_counter()
        base = run_pair(config, day)
        base_elapsed = time.perf_counter() - start
        start = time.perf_counter()
        copies = [run_pair(config, day) for _ in range(3)]
        copied_elapsed = time.perf_counter() - start
        rows.append(
            {
                "date": day["date"],
                "pedf8_elapsed_s": base_elapsed,
                "pedf24_block_separable_elapsed_s": copied_elapsed,
                "runtime_ratio": copied_elapsed / base_elapsed,
                "max_effect_difference_pctpoint": max(abs(x["effect_pctpoint"] - base["effect_pctpoint"]) for x in copies),
                "maximum_closure": max(max(x["b3_carbon_closure"], x["p_carbon_closure"]) for x in copies),
            }
        )
    except Exception as exc:
        failures.append({"date": day["date"], "error": f"{type(exc).__name__}: {exc}"})
    save_partial(output, rows, failures)
    finalize(output, "S0_PEDF24", rows, failures, config, STARTED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=("pareto", "loss", "flexibility", "capacity", "stress", "scale"), required=True)
    args = parser.parse_args()
    config = load_config(ROOT / "configs" / "pedf8.json")
    days = load_days(ROOT / "data" / "processed" / "timeseries.csv")
    if args.slice == "pareto":
        pareto(config, days, CAMPAIGN / "S3_COST_PARETO")
    elif args.slice == "loss":
        no_loss = copy.deepcopy(config)
        no_loss["branches"] = [[i, j, 0.0, pmax] for i, j, _, pmax in no_loss["branches"]]
        no_loss["efficiency"] = {key: 1.0 for key in no_loss["efficiency"]}
        variant_runs(config, days, CAMPAIGN / "A1_NO_LOSS", "A1_NO_LOSS", {"no_loss": no_loss})
    elif args.slice == "flexibility":
        variants = {}
        for removed in ("hvac", "ev", "task"):
            variant = copy.deepcopy(config)
            variant["flexibility"] = {"hvac": True, "ev": True, "task": True}
            variant["flexibility"][removed] = False
            variants[f"fixed_{removed}"] = variant
        variant_runs(config, days, CAMPAIGN / "A4_FLEXIBILITY", "A4_FLEXIBILITY", variants)
    elif args.slice == "capacity":
        variants = {}
        for capacity in (40.0, 120.0):
            variant = copy.deepcopy(config)
            variant["pv"]["capacity_kwp"] = capacity
            variants[f"pv_{int(capacity)}_kwp"] = variant
        for capacity in (80.0, 240.0):
            variant = copy.deepcopy(config)
            variant["ess"]["capacity_kwh"] = capacity
            variants[f"ess_{int(capacity)}_kwh"] = variant
        variant_runs(config, days, CAMPAIGN / "S1_S2_CAPACITY", "S1_S2_CAPACITY", variants)
    elif args.slice == "stress":
        stress(config, days, CAMPAIGN / "S4_AR1_STRESS")
    else:
        scale(config, days, CAMPAIGN / "S0_PEDF24")


if __name__ == "__main__":
    STARTED = time.time()
    main()
