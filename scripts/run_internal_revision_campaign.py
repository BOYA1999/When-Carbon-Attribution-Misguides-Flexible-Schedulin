from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pedf import DayOptimizer, compute_mci, load_config, load_days, signals_from_trace, trace_carbon


OUT = ROOT / "artifacts" / "analysis" / "campaign_20260802_internal_revision"
MAIN = ROOT / "artifacts" / "experiment" / "pedf8_main_20260802_v1"
ACTIONS = ("hvac_kw", "ev_kw", "task_kw", "ess_charge_kw", "ess_discharge_kw")


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2), encoding="utf-8")


def save_rows(folder: Path, rows: list[dict], failures: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "results.csv", index=False)
    write_json(folder / "failures.json", failures)


def validate(config: dict, result: dict, cap: float | None = None) -> None:
    tolerance = config["tolerance"]
    if result["energy_balance_max_kw"] > tolerance["energy_balance_kw"]:
        raise RuntimeError("energy-balance tolerance exceeded")
    if result["socp_gap_max"] > tolerance["socp_gap"]:
        raise RuntimeError("SOCP recovery tolerance exceeded")
    if result["branch_capacity_violation_kw"] > max(tolerance["constraint"], 2e-5):
        raise RuntimeError("branch-capacity tolerance exceeded")
    if result["voltage_violation_pu"] > tolerance["constraint"]:
        raise RuntimeError("voltage tolerance exceeded")
    if result["simultaneous_charge_discharge_kw"] > tolerance["constraint"]:
        raise RuntimeError("ESS mode complementarity failed")
    if cap is not None and result["cost_gbp"] - cap > 1e-4:
        raise RuntimeError("cost cap exceeded")


def signals(day: dict, b0: dict, cached) -> dict[str, dict[str, np.ndarray]]:
    naive = trace_carbon(
        b0, load_config(ROOT / "configs" / "pedf8.json"),
        day["carbon_g_per_kwh"], day["load_kw"], memory=False,
    )
    nodal = cached["reference_nodal_aci_g_per_kwh"]
    storage = cached["reference_storage_ci_g_per_kwh"]
    mci = cached["mci_g_per_kwh"]
    return {
        "B1": {key: day["carbon_g_per_kwh"].copy() for key in ("hvac", "ev", "task", "charge", "discharge")},
        "B2": signals_from_trace(naive, day["carbon_g_per_kwh"], "nodal"),
        "B3": {
            "hvac": nodal[2], "charge": nodal[3], "discharge": storage,
            "ev": nodal[6], "task": nodal[7],
        },
        "P": {
            "hvac": mci[2], "charge": mci[3], "discharge": mci[3],
            "ev": mci[6], "task": mci[7],
        },
    }


def direction_agreement(candidate: dict, oracle: dict, baseline: dict) -> tuple[float, int]:
    same = 0
    total = 0
    for key in ACTIONS:
        candidate_change = candidate[key] - baseline[key]
        oracle_change = oracle[key] - baseline[key]
        active = np.maximum(np.abs(candidate_change), np.abs(oracle_change)) > 1e-4
        same += int(np.sum(np.sign(candidate_change[active]) == np.sign(oracle_change[active])))
        total += int(np.sum(active))
    return same / max(total, 1), total


def action_distance(first: dict, second: dict) -> tuple[float, float]:
    difference = np.concatenate([first[key] - second[key] for key in ACTIONS])
    return float(np.sqrt(np.mean(difference**2))), float(np.max(np.abs(difference)))


@contextlib.contextmanager
def native_stdout(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = os.dup(1)
    with path.open("w", encoding="utf-8") as handle:
        os.dup2(handle.fileno(), 1)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)


def scip_log_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    primal_matches = re.findall(r"Primal Bound\s*:\s*([+\-0-9.eE]+)", text)
    dual_matches = re.findall(r"Dual Bound\s*:\s*([+\-0-9.eE]+)", text)
    status_matches = re.findall(r"SCIP Status\s*:\s*(.+)", text)
    primal = float(primal_matches[-1]) if primal_matches else np.nan
    dual = float(dual_matches[-1]) if dual_matches else np.nan
    gap = abs(primal - dual) / max(abs(primal), 1e-12) if np.isfinite(primal + dual) else np.nan
    return {
        "scip_primal_bound": primal,
        "scip_dual_bound": dual,
        "scip_realized_relative_gap": gap,
        "scip_termination": status_matches[-1].strip() if status_matches else "not_exposed",
    }


def finish(folder: Path, slice_id: str, rows: list[dict], failures: list[dict], started: float, extra: dict) -> None:
    save_rows(folder, rows, failures)
    summary = {
        "slice_id": slice_id,
        "status": "complete" if not failures else "partial",
        "complete_rows": len(rows),
        "failed_cases": len(failures),
        "elapsed_s": time.time() - started,
        **extra,
    }
    write_json(folder / "summary.json", summary)
    write_json(
        folder / "manifest.json",
        {
            "slice_id": slice_id,
            "seed": 20260802,
            "config_sha256": digest(ROOT / "configs" / "pedf8.json"),
            "dataset_sha256": digest(ROOT / "data" / "processed" / "timeseries.csv"),
            "output": str(folder.relative_to(ROOT)).replace("\\", "/"),
        },
    )
    print(json.dumps(plain(summary), ensure_ascii=False), flush=True)


def oracle(config: dict, days: list[dict]) -> None:
    folder = OUT / "REV-B4"
    started = time.time()
    rows, failures = [], []
    for day in days:
        print(f"REV-B4 {day['date']}", flush=True)
        try:
            model = DayOptimizer(config, day)
            b0 = model.solve_econ()
            if b0 is None:
                raise RuntimeError("B0 failed")
            validate(config, b0)
            cached = np.load(MAIN / "arrays" / f"{day['date']}.npz")
            method_signals = signals(day, b0, cached)
            for delta in config["cost_budgets"]:
                cap = b0["cost_gbp"] * (1 + delta)
                b4 = b0 if delta == 0 else model.solve_emissions(cap)
                if b4 is None:
                    raise RuntimeError(f"B4 failed at delta={delta}")
                validate(config, b4, cap)
                results = {"B0": b0, "B4": b4}
                for method, method_signal in method_signals.items():
                    result = b0 if delta == 0 else model.solve_signal(method_signal, cap)
                    if result is None:
                        raise RuntimeError(f"{method} failed at delta={delta}")
                    validate(config, result, cap)
                    results[method] = result
                for method, result in results.items():
                    agreement, denominator = direction_agreement(result, b4, b0)
                    rows.append(
                        {
                            "date": day["date"], "delta": delta, "method": method,
                            "cost_gbp": result["cost_gbp"],
                            "source_emissions_kg": result["source_emissions_kg"],
                            "emissions_change_vs_b0_pct": 100 * (result["source_emissions_kg"] / b0["source_emissions_kg"] - 1),
                            "regret_vs_b4_kg": result["source_emissions_kg"] - b4["source_emissions_kg"],
                            "regret_vs_b4_pctpoint": 100 * (result["source_emissions_kg"] - b4["source_emissions_kg"]) / b0["source_emissions_kg"],
                            "action_direction_agreement": agreement,
                            "action_direction_denominator": denominator,
                            "cost_cap_violation_gbp": max(result["cost_gbp"] - cap, 0),
                        }
                    )
            save_rows(folder, rows, failures)
        except Exception as exc:
            failures.append({"date": day["date"], "error": f"{type(exc).__name__}: {exc}"})
            save_rows(folder, rows, failures)
    frame = pd.DataFrame(rows)
    aggregate = []
    for (method, delta), group in frame.groupby(["method", "delta"]):
        aggregate.append(
            {
                "method": method, "delta": delta, "days": len(group),
                "median_regret_pctpoint": group["regret_vs_b4_pctpoint"].median(),
                "median_direction_agreement": group["action_direction_agreement"].median(),
                "emission_reduction_days": int((group["emissions_change_vs_b0_pct"] < 0).sum()),
            }
        )
    pd.DataFrame(aggregate).to_csv(folder / "aggregate.csv", index=False)
    b4_count = int(((frame["method"] == "B4") & frame["delta"].isin(config["cost_budgets"])).sum())
    finish(folder, "REV-B4", rows, failures, started, {"planned_b4_cases": 140, "completed_b4_cases": b4_count})


def mci_epsilon(config: dict, days: list[dict]) -> None:
    folder = OUT / "REV-MCI-EPS"
    array_folder = folder / "arrays"
    array_folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows, failures = [], []
    for day in days:
        print(f"REV-MCI-EPS {day['date']}", flush=True)
        try:
            model = DayOptimizer(config, day)
            b0 = model.solve_econ()
            if b0 is None:
                raise RuntimeError("B0 failed")
            validate(config, b0)
            cached = np.load(MAIN / "arrays" / f"{day['date']}.npz")
            reference_signal = signals(day, b0, cached)["B3"]
            cap = b0["cost_gbp"] * 1.02
            b3 = model.solve_signal(reference_signal, cap)
            if b3 is None:
                raise RuntimeError("B3 failed")
            daily = {}
            diagnostics = {}
            for scale in (0.5, 1.0, 2.0):
                print(f"REV-MCI-EPS {day['date']} scale={scale}", flush=True)
                mci, diagnostic = compute_mci(model, b0, progress=lambda _: None, epsilon_scale=scale)
                p_signal = {
                    "hvac": mci[2], "charge": mci[3], "discharge": mci[3],
                    "ev": mci[6], "task": mci[7],
                }
                proposed = model.solve_signal(p_signal, cap)
                if proposed is None:
                    raise RuntimeError(f"P failed at scale={scale}")
                validate(config, proposed, cap)
                daily[scale] = mci
                diagnostics[scale] = (diagnostic, proposed)
                np.savez_compressed(array_folder / f"{day['date']}_scale_{scale:g}.npz", mci_g_per_kwh=mci)
            reference = daily[1.0][[2, 3, 6, 7]].ravel()
            for scale in (0.5, 1.0, 2.0):
                mci = daily[scale][[2, 3, 6, 7]].ravel()
                diagnostic, proposed = diagnostics[scale]
                rows.append(
                    {
                        "date": day["date"], "epsilon_scale": scale,
                        "finite_fraction": diagnostic["finite_fraction"],
                        "stable": diagnostic["stable"],
                        "mci_min_g_per_kwh": diagnostic["min_g_per_kwh"],
                        "mci_median_g_per_kwh": diagnostic["median_g_per_kwh"],
                        "mci_max_g_per_kwh": diagnostic["max_g_per_kwh"],
                        "negative_fraction": float(np.mean(mci < 0)),
                        "rank_rho_vs_scale_1": float(spearmanr(reference, mci).statistic),
                        "p95_forward_backward_disagreement": diagnostic["normalized_forward_backward_disagreement_p95"],
                        "p_source_emissions_kg": proposed["source_emissions_kg"],
                        "p_minus_b3_pctpoint": 100 * (b3["source_emissions_kg"] - proposed["source_emissions_kg"]) / b0["source_emissions_kg"],
                    }
                )
            save_rows(folder, rows, failures)
        except Exception as exc:
            failures.append({"date": day["date"], "error": f"{type(exc).__name__}: {exc}"})
            save_rows(folder, rows, failures)
    frame = pd.DataFrame(rows)
    aggregate = frame.groupby("epsilon_scale").agg(
        days=("date", "count"), median_rank_rho=("rank_rho_vs_scale_1", "median"),
        median_effect_pctpoint=("p_minus_b3_pctpoint", "median"),
        positive_days=("p_minus_b3_pctpoint", lambda values: int((values > 0).sum())),
        stable_days=("stable", "sum"),
    ).reset_index()
    aggregate.to_csv(folder / "aggregate.csv", index=False)
    finish(folder, "REV-MCI-EPS", rows, failures, started, {"planned_day_scales": 84, "completed_day_scales": len(rows)})


def storage_boundary(config: dict, days: list[dict]) -> None:
    folder = OUT / "REV-Q0"
    started = time.time()
    rows, failures = [], []
    previous_ci = None
    main_metrics = pd.read_csv(MAIN / "metrics.csv")
    for day in days:
        print(f"REV-Q0 {day['date']}", flush=True)
        try:
            model = DayOptimizer(config, day)
            b0 = model.solve_econ()
            if b0 is None:
                raise RuntimeError("B0 failed")
            validate(config, b0)
            cap = b0["cost_gbp"] * 1.02
            p_emissions = float(main_metrics.loc[
                (main_metrics["date"] == day["date"]) & (main_metrics["method"] == "P_DUAL_MCI_ACI"),
                "source_emissions_kg",
            ].iloc[0])
            cases = {
                "zero": 0.0,
                "daily_first_gb_aci": float(day["carbon_g_per_kwh"][0]),
                "continuous_cross_day": float(day["carbon_g_per_kwh"][0] if previous_ci is None else previous_ci),
            }
            continuous_trace = None
            for boundary, initial_ci in cases.items():
                reference = trace_carbon(
                    b0, config, day["carbon_g_per_kwh"], day["load_kw"],
                    memory=True, initial_storage_ci=initial_ci,
                )
                b3 = model.solve_signal(
                    signals_from_trace(reference, day["carbon_g_per_kwh"], "memory"), cap
                )
                if b3 is None:
                    raise RuntimeError(f"B3 failed for {boundary}")
                validate(config, b3, cap)
                b3_trace = trace_carbon(
                    b3, config, day["carbon_g_per_kwh"], day["load_kw"],
                    memory=True, initial_storage_ci=initial_ci,
                )
                rows.append(
                    {
                        "date": day["date"], "boundary": boundary,
                        "initial_storage_ci_g_per_kwh": initial_ci,
                        "terminal_storage_ci_g_per_kwh": reference["storage_carbon_stock_g"][-1] / b0["ess_energy_kwh"][-1],
                        "b3_source_emissions_kg": b3["source_emissions_kg"],
                        "p_source_emissions_kg": p_emissions,
                        "p_minus_b3_pctpoint": 100 * (b3["source_emissions_kg"] - p_emissions) / b0["source_emissions_kg"],
                        "b3_carbon_closure_relative": b3_trace["carbon_closure_relative"],
                    }
                )
                if boundary == "continuous_cross_day":
                    continuous_trace = reference
            previous_ci = continuous_trace["storage_carbon_stock_g"][-1] / b0["ess_energy_kwh"][-1]
            save_rows(folder, rows, failures)
        except Exception as exc:
            failures.append({"date": day["date"], "error": f"{type(exc).__name__}: {exc}"})
            save_rows(folder, rows, failures)
    frame = pd.DataFrame(rows)
    aggregate = frame.groupby("boundary").agg(
        days=("date", "count"), median_effect_pctpoint=("p_minus_b3_pctpoint", "median"),
        positive_days=("p_minus_b3_pctpoint", lambda values: int((values > 0).sum())),
        max_closure=("b3_carbon_closure_relative", "max"),
    ).reset_index()
    aggregate.to_csv(folder / "aggregate.csv", index=False)
    finish(folder, "REV-Q0", rows, failures, started, {"planned_day_boundaries": 84, "completed_day_boundaries": len(rows)})


def permute_signal(mci: np.ndarray, order: np.ndarray) -> dict[str, np.ndarray]:
    shifted = mci[:, order]
    return {
        "hvac": shifted[2], "charge": shifted[3], "discharge": shifted[3],
        "ev": shifted[6], "task": shifted[7],
    }


def temporal_null(config: dict, days: list[dict]) -> None:
    folder = OUT / "REV-TIME-NULL"
    started = time.time()
    rows, failures = [], []
    metrics = pd.read_csv(MAIN / "metrics.csv")
    mci_by_day = [np.load(MAIN / "arrays" / f"{day['date']}.npz")["mci_g_per_kwh"] for day in days]
    for day_index, day in enumerate(days):
        print(f"REV-TIME-NULL within-day {day['date']}", flush=True)
        try:
            model = DayOptimizer(config, day)
            b0 = model.solve_econ()
            if b0 is None:
                raise RuntimeError("B0 failed")
            cap = b0["cost_gbp"] * 1.02
            actual = float(metrics.loc[
                (metrics["date"] == day["date"]) & (metrics["method"] == "P_DUAL_MCI_ACI"),
                "source_emissions_kg",
            ].iloc[0])
            for shift in range(1, 48):
                order = np.roll(np.arange(48), shift)
                result = model.solve_signal(permute_signal(mci_by_day[day_index], order), cap)
                if result is None:
                    raise RuntimeError(f"circular shift {shift} failed")
                validate(config, result, cap)
                rows.append({"date": day["date"], "null_type": "circular_shift", "variant": shift, "null_emissions_kg": result["source_emissions_kg"], "actual_emissions_kg": actual, "actual_gain_pctpoint": 100 * (result["source_emissions_kg"] - actual) / b0["source_emissions_kg"]})
            blocks = np.arange(48).reshape(12, 4)
            for variant in range(24):
                rng = np.random.default_rng(config["seed"] + 1000 * day_index + variant)
                order = blocks[rng.permutation(12)].ravel()
                result = model.solve_signal(permute_signal(mci_by_day[day_index], order), cap)
                if result is None:
                    raise RuntimeError(f"block permutation {variant} failed")
                validate(config, result, cap)
                rows.append({"date": day["date"], "null_type": "four_step_block", "variant": variant, "null_emissions_kg": result["source_emissions_kg"], "actual_emissions_kg": actual, "actual_gain_pctpoint": 100 * (result["source_emissions_kg"] - actual) / b0["source_emissions_kg"]})
            save_rows(folder, rows, failures)
        except Exception as exc:
            failures.append({"date": day["date"], "phase": "within_day", "error": f"{type(exc).__name__}: {exc}"})
            save_rows(folder, rows, failures)
    cross_day_sources = []
    for variant in range(24):
        rng = np.random.default_rng(config["seed"] + 100000 + variant)
        cross_day_sources.append(
            np.column_stack([rng.permutation(len(days)) for _ in range(48)])
        )
    for day_index, day in enumerate(days):
        print(f"REV-TIME-NULL cross-day {day['date']}", flush=True)
        try:
            model = DayOptimizer(config, day)
            b0 = model.solve_econ()
            if b0 is None:
                raise RuntimeError("B0 failed")
            cap = b0["cost_gbp"] * 1.02
            actual = float(metrics.loc[
                (metrics["date"] == day["date"]) & (metrics["method"] == "P_DUAL_MCI_ACI"),
                "source_emissions_kg",
            ].iloc[0])
            for variant, source_days in enumerate(cross_day_sources):
                mixed = np.column_stack([
                    mci_by_day[source_days[day_index, step]][:, step] for step in range(48)
                ])
                result = model.solve_signal(permute_signal(mixed, np.arange(48)), cap)
                if result is None:
                    raise RuntimeError(f"cross-day permutation {variant} failed")
                validate(config, result, cap)
                rows.append({"date": day["date"], "null_type": "cross_day_same_slot", "variant": variant, "null_emissions_kg": result["source_emissions_kg"], "actual_emissions_kg": actual, "actual_gain_pctpoint": 100 * (result["source_emissions_kg"] - actual) / b0["source_emissions_kg"]})
            save_rows(folder, rows, failures)
        except Exception as exc:
            failures.append({"date": day["date"], "phase": "cross_day", "error": f"{type(exc).__name__}: {exc}"})
            save_rows(folder, rows, failures)
    frame = pd.DataFrame(rows)
    daily = []
    for (date, null_type), group in frame.groupby(["date", "null_type"]):
        actual = group["actual_emissions_kg"].iloc[0]
        daily.append({
            "date": date, "null_type": null_type, "null_count": len(group),
            "actual_gain_median_pctpoint": group["actual_gain_pctpoint"].median(),
            "actual_better_fraction": float(np.mean(group["null_emissions_kg"] > actual)),
            "empirical_p_lower_emissions": (1 + int(np.sum(group["null_emissions_kg"] <= actual))) / (len(group) + 1),
        })
    daily_frame = pd.DataFrame(daily)
    daily_frame.to_csv(folder / "daily_randomization_summary.csv", index=False)
    aggregate = daily_frame.groupby("null_type").agg(
        days=("date", "count"), median_actual_gain_pctpoint=("actual_gain_median_pctpoint", "median"),
        median_actual_better_fraction=("actual_better_fraction", "median"),
        median_empirical_p=("empirical_p_lower_emissions", "median"),
    ).reset_index()
    aggregate.to_csv(folder / "aggregate.csv", index=False)
    finish(folder, "REV-TIME-NULL", rows, failures, started, {"planned_schedules": 2660, "completed_schedules": len(rows)})


def autocorrelation(values: np.ndarray, lag: int) -> float:
    centered = values - np.mean(values)
    denominator = np.sum(centered**2)
    return float(np.sum(centered[:-lag] * centered[lag:]) / denominator)


def temporal_statistics(config: dict) -> None:
    folder = OUT / "REV-TIME-STAT"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metrics = pd.read_csv(MAIN / "metrics.csv")
    emissions = metrics.pivot(index="date", columns="method", values="source_emissions_kg").sort_index()
    primary = (
        100
        * (emissions["B3_NODAL_ACI_MEM"] - emissions["P_DUAL_MCI_ACI"])
        / emissions["B0_ECON"]
    ).iloc[:21].to_numpy(float)
    acf = [{"lag": lag, "acf": autocorrelation(primary, lag)} for lag in range(1, 8)]
    pd.DataFrame(acf).to_csv(folder / "acf.csv", index=False)
    rng = np.random.default_rng(config["seed"])
    intervals = []
    distributions = {}
    n = len(primary)
    for block_length in (2, 3, 4):
        blocks_needed = int(np.ceil(n / block_length))
        medians = np.empty(10000)
        for replicate in range(10000):
            starts = rng.integers(0, n, size=blocks_needed)
            sample = np.concatenate([
                primary[(start + np.arange(block_length)) % n] for start in starts
            ])[:n]
            medians[replicate] = np.median(sample)
        distributions[f"block_{block_length}"] = medians
        intervals.append({
            "block_length_days": block_length, "replicates": 10000,
            "observed_median_pctpoint": float(np.median(primary)),
            "ci_low_pctpoint": float(np.quantile(medians, 0.025)),
            "ci_high_pctpoint": float(np.quantile(medians, 0.975)),
            "positive_bootstrap_fraction": float(np.mean(medians > 0)),
        })
    pd.DataFrame(intervals).to_csv(folder / "moving_block_bootstrap.csv", index=False)
    np.savez_compressed(folder / "bootstrap_distributions.npz", **distributions)
    finish(folder, "REV-TIME-STAT", intervals, [], started, {"primary_days": n, "bootstrap_replicates_per_block": 10000, "acf": acf})


def mip_validation(config: dict, days: list[dict]) -> None:
    folder = OUT / "REV-MIP"
    log_folder = folder / "logs"
    started = time.time()
    rows, failures = [], []
    for day_index in (0, 6, 12):
        day = days[day_index]
        print(f"REV-MIP {day['date']}", flush=True)
        try:
            model = DayOptimizer(config, day)
            b0 = model.solve_econ()
            if b0 is None:
                raise RuntimeError("B0 failed")
            cached = np.load(MAIN / "arrays" / f"{day['date']}.npz")
            method_signals = signals(day, b0, cached)
            for delta in (0.02, 0.10):
                cap = b0["cost_gbp"] * (1 + delta)
                for method in ("B3", "P"):
                    heuristic = model.solve_signal(method_signals[method], cap)
                    log_path = log_folder / f"{day['date']}_delta_{delta:g}_{method}.log"
                    with native_stdout(log_path):
                        exact = model.solve_signal_mip(
                            method_signals[method], cap, solver="SCIP", verbose=True
                        )
                    log_metrics = scip_log_metrics(log_path)
                    if heuristic is None or exact is None:
                        raise RuntimeError(f"{method} failed at delta={delta}")
                    validate(config, heuristic, cap)
                    validate(config, exact, cap)
                    agreement, denominator = direction_agreement(heuristic, exact, b0)
                    rmse, maximum = action_distance(heuristic, exact)
                    rows.append({
                        "date": day["date"], "delta": delta, "method": method,
                        "solver": "SCIP", "solver_status": exact["solver_status"],
                        "configured_relative_gap_limit": 1e-6,
                        "heuristic_emissions_kg": heuristic["source_emissions_kg"],
                        "mip_emissions_kg": exact["source_emissions_kg"],
                        "heuristic_objective": heuristic["action_objective"],
                        "mip_objective": exact["action_objective"],
                        "heuristic_minus_mip_objective": heuristic["action_objective"] - exact["action_objective"],
                        "action_direction_agreement": agreement,
                        "action_direction_denominator": denominator,
                        "action_rmse_kw": rmse,
                        "action_max_absolute_difference_kw": maximum,
                        "mip_runtime_s": exact["runtime_s"],
                        **log_metrics,
                    })
                heuristic = model.solve_emissions(cap)
                log_path = log_folder / f"{day['date']}_delta_{delta:g}_B4.log"
                with native_stdout(log_path):
                    exact = model.solve_emissions_mip(
                        cap, solver="SCIP", verbose=True
                    )
                log_metrics = scip_log_metrics(log_path)
                if heuristic is None or exact is None:
                    raise RuntimeError(f"B4 failed at delta={delta}")
                validate(config, heuristic, cap)
                validate(config, exact, cap)
                agreement, denominator = direction_agreement(heuristic, exact, b0)
                rmse, maximum = action_distance(heuristic, exact)
                rows.append({
                    "date": day["date"], "delta": delta, "method": "B4",
                    "solver": "SCIP", "solver_status": exact["solver_status"],
                    "configured_relative_gap_limit": 1e-6,
                    "heuristic_emissions_kg": heuristic["source_emissions_kg"],
                    "mip_emissions_kg": exact["source_emissions_kg"],
                    "heuristic_objective": heuristic["source_emissions_kg"],
                    "mip_objective": exact["source_emissions_kg"],
                    "heuristic_minus_mip_objective": heuristic["source_emissions_kg"] - exact["source_emissions_kg"],
                    "action_direction_agreement": agreement,
                    "action_direction_denominator": denominator,
                    "action_rmse_kw": rmse,
                    "action_max_absolute_difference_kw": maximum,
                    "mip_runtime_s": exact["runtime_s"],
                    **log_metrics,
                })
            save_rows(folder, rows, failures)
        except Exception as exc:
            failures.append({"date": day["date"], "error": f"{type(exc).__name__}: {exc}"})
            save_rows(folder, rows, failures)
    frame = pd.DataFrame(rows)
    aggregate = frame.groupby("method").agg(
        cases=("date", "count"), median_objective_gap=("heuristic_minus_mip_objective", "median"),
        max_absolute_objective_gap=("heuristic_minus_mip_objective", lambda values: float(np.max(np.abs(values)))),
        median_direction_agreement=("action_direction_agreement", "median"),
        median_action_rmse_kw=("action_rmse_kw", "median"),
        max_action_difference_kw=("action_max_absolute_difference_kw", "max"),
        max_realized_relative_gap=("scip_realized_relative_gap", "max"),
        median_mip_runtime_s=("mip_runtime_s", "median"),
    ).reset_index()
    aggregate.to_csv(folder / "aggregate.csv", index=False)
    finish(folder, "REV-MIP", rows, failures, started, {"planned_cases": 18, "completed_cases": len(rows), "bound_reporting": "SCIP native log primal and dual bounds"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("slice", choices=("oracle", "mci-eps", "q0", "time-null", "time-stat", "mip", "all"))
    args = parser.parse_args()
    config = load_config(ROOT / "configs" / "pedf8.json")
    days = load_days(ROOT / "data" / "processed" / "timeseries.csv")
    runners = {
        "oracle": lambda: oracle(config, days),
        "mci-eps": lambda: mci_epsilon(config, days),
        "q0": lambda: storage_boundary(config, days),
        "time-null": lambda: temporal_null(config, days),
        "time-stat": lambda: temporal_statistics(config),
        "mip": lambda: mip_validation(config, days),
    }
    if args.slice == "all":
        for runner in runners.values():
            runner()
    else:
        runners[args.slice]()


if __name__ == "__main__":
    main()
