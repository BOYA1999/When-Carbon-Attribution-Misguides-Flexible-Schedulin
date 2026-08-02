from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .carbon import signals_from_trace, trace_carbon
from .model import DayOptimizer


FLEXIBLE_NODES = (2, 3, 6, 7)


def compute_mci(
    model: DayOptimizer,
    baseline: dict,
    progress=print,
    epsilon_scale: float = 1.0,
) -> tuple[np.ndarray, dict]:
    mci = np.full((model.n, model.t), np.nan)
    forward = np.full_like(mci, np.nan)
    backward = np.full_like(mci, np.nan)
    schemes: dict[str, int] = {"central": 0, "forward": 0, "failed": 0}
    max_gap = baseline["socp_gap_max"]
    failures = []
    base_emissions = baseline["source_emissions_kg"]
    for node in FLEXIBLE_NODES:
        progress(f"MCI {model.day['date']} node={node}")
        for t in range(model.t):
            epsilon = epsilon_scale * max(0.1, 0.01 * model.day["load_kw"][t])
            probe = np.zeros((model.n, model.t))
            probe[node, t] = epsilon
            plus = model.solve_econ(probe)
            probe[node, t] = -epsilon
            minus = model.solve_econ(probe)
            if plus is None:
                schemes["failed"] += 1
                failures.append({"node": node, "time": t, "side": "plus"})
                continue
            max_gap = max(max_gap, plus["socp_gap_max"])
            forward[node, t] = (
                plus["source_emissions_kg"] - base_emissions
            ) * 1000 / (epsilon * model.dt)
            if minus is None:
                mci[node, t] = forward[node, t]
                schemes["forward"] += 1
                continue
            max_gap = max(max_gap, minus["socp_gap_max"])
            backward[node, t] = (
                base_emissions - minus["source_emissions_kg"]
            ) * 1000 / (epsilon * model.dt)
            mci[node, t] = 0.5 * (forward[node, t] + backward[node, t])
            schemes["central"] += 1
    selected = mci[list(FLEXIBLE_NODES)].ravel()
    paired = np.isfinite(forward) & np.isfinite(backward)
    normalized_disagreement = np.abs(forward[paired] - backward[paired]) / np.maximum(
        np.abs(mci[paired]), 25.0
    )
    diagnostics = {
        "epsilon_scale": float(epsilon_scale),
        "schemes": schemes,
        "failures": failures,
        "finite_fraction": float(np.isfinite(selected).mean()),
        "min_g_per_kwh": float(np.nanmin(selected)),
        "median_g_per_kwh": float(np.nanmedian(selected)),
        "max_g_per_kwh": float(np.nanmax(selected)),
        "normalized_forward_backward_disagreement_p50": float(
            np.nanmedian(normalized_disagreement)
        ),
        "normalized_forward_backward_disagreement_p95": float(
            np.nanquantile(normalized_disagreement, 0.95)
        ),
        "maximum_perturbation_socp_gap": float(max_gap),
    }
    diagnostics["stable"] = bool(
        schemes["failed"] == 0
        and diagnostics["finite_fraction"] == 1.0
        and max(abs(diagnostics["min_g_per_kwh"]), abs(diagnostics["max_g_per_kwh"]))
        <= 5000
        and diagnostics["normalized_forward_backward_disagreement_p95"] <= 2.0
        and max_gap <= model.cfg["tolerance"]["socp_gap"]
    )
    return mci, diagnostics


def mismatch_metrics(aci: np.ndarray, mci: np.ndarray) -> dict:
    a = aci[list(FLEXIBLE_NODES)].ravel()
    m = mci[list(FLEXIBLE_NODES)].ravel()
    valid = np.isfinite(a) & np.isfinite(m)
    a = a[valid]
    m = m[valid]
    rho = float(spearmanr(a, m).statistic)
    a_low, a_high = np.quantile(a, [0.25, 0.75])
    m_low, m_high = np.quantile(m, [0.25, 0.75])
    low_a = a <= a_low
    high_a = a >= a_high
    low_overlap = float(np.sum(low_a & (m <= m_low)) / max(np.sum(low_a), 1))
    high_overlap = float(np.sum(high_a & (m >= m_high)) / max(np.sum(high_a), 1))
    discordant = 0
    comparable = 0
    for i in range(len(a) - 1):
        products = (a[i] - a[i + 1 :]) * (m[i] - m[i + 1 :])
        non_tie = products != 0
        comparable += int(np.sum(non_tie))
        discordant += int(np.sum(products[non_tie] < 0))
    return {
        "spearman_rho": rho,
        "low_quartile_overlap": low_overlap,
        "high_quartile_overlap": high_overlap,
        "rank_inversion_rate": float(discordant / max(comparable, 1)),
        "points": int(len(a)),
    }


def canonical_metrics(method: str, result: dict, trace: dict) -> dict:
    keys = (
        "date",
        "solver_status",
        "runtime_s",
        "cost_gbp",
        "source_emissions_kg",
        "energy_balance_max_kw",
        "socp_gap_max",
        "line_loss_kwh",
        "conversion_loss_kwh",
        "pv_curtailment_kwh",
        "peak_grid_import_kw",
        "ramp_kw",
        "comfort_violation_c_h",
        "ev_energy_shortfall_kwh",
        "task_energy_shortfall_kwh",
        "voltage_violation_pu",
        "terminal_soc_error_kwh",
        "simultaneous_charge_discharge_kw",
        "simultaneous_import_export_kw",
        "relaxation_socp_gap_max",
        "power_flow_iterations_max",
        "branch_capacity_violation_kw",
        "negative_price_export_kw",
    )
    row = {key: result[key] for key in keys}
    row["mode_relaxation_simultaneous_kw"] = float(
        result.get("mode_relaxation_simultaneous_kw", 0.0)
    )
    row["mode_restoration_action_gap"] = float(
        result.get("mode_restoration_action_gap", 0.0)
    )
    row["method"] = method
    row["attributed_emissions_kg"] = float(trace["attributed_emissions_kg"])
    row["carbon_closure_relative"] = float(trace["carbon_closure_relative"])
    return row


def run_day(config: dict, day: dict, cost_budget: float = 0.02, progress=print) -> dict:
    model = DayOptimizer(config, day)
    progress(f"DAY {day['date']} B0")
    b0 = model.solve_econ()
    if b0 is None:
        raise RuntimeError(f"B0 infeasible on {day['date']}")
    memory_reference = trace_carbon(
        b0, config, day["carbon_g_per_kwh"], day["load_kw"], memory=True
    )
    naive_reference = trace_carbon(
        b0, config, day["carbon_g_per_kwh"], day["load_kw"], memory=False
    )
    cost_cap = b0["cost_gbp"] * (1 + cost_budget)
    signal_sets = {
        "B1_BULK_ACI": signals_from_trace(
            memory_reference, day["carbon_g_per_kwh"], "bulk"
        ),
        "B2_NODAL_ACI": signals_from_trace(
            naive_reference, day["carbon_g_per_kwh"], "naive"
        ),
        "B3_NODAL_ACI_MEM": signals_from_trace(
            memory_reference, day["carbon_g_per_kwh"], "memory"
        ),
    }
    results = {"B0_ECON": b0}
    for method, signals in signal_sets.items():
        progress(f"DAY {day['date']} {method}")
        result = model.solve_signal(signals, cost_cap)
        if result is None:
            raise RuntimeError(f"{method} failed on {day['date']}")
        if (
            result["cost_cap_violation_gbp"] > 1e-4
            or result["branch_capacity_violation_kw"] > config["tolerance"]["constraint"]
            or result["simultaneous_charge_discharge_kw"] > config["tolerance"]["constraint"]
            or result["negative_price_export_kw"] > config["tolerance"]["constraint"]
        ):
            raise RuntimeError(f"{method} physical recovery failed on {day['date']}")
        results[method] = result
    mci, mci_diagnostics = compute_mci(model, b0, progress=progress)
    p_signals = {
        "hvac": mci[2].copy(),
        "charge": mci[3].copy(),
        "discharge": mci[3].copy(),
        "ev": mci[6].copy(),
        "task": mci[7].copy(),
    }
    progress(f"DAY {day['date']} P_DUAL_MCI_ACI")
    proposed = model.solve_signal(p_signals, cost_cap)
    if proposed is None:
        raise RuntimeError(f"P failed on {day['date']}")
    if (
        proposed["cost_cap_violation_gbp"] > 1e-4
        or proposed["branch_capacity_violation_kw"] > config["tolerance"]["constraint"]
        or proposed["simultaneous_charge_discharge_kw"] > config["tolerance"]["constraint"]
        or proposed["negative_price_export_kw"] > config["tolerance"]["constraint"]
    ):
        raise RuntimeError(f"P physical recovery failed on {day['date']}")
    results["P_DUAL_MCI_ACI"] = proposed

    traces = {
        method: trace_carbon(
            result,
            config,
            day["carbon_g_per_kwh"],
            day["load_kw"],
            memory=True,
        )
        for method, result in results.items()
    }
    naive_b2 = trace_carbon(
        results["B2_NODAL_ACI"],
        config,
        day["carbon_g_per_kwh"],
        day["load_kw"],
        memory=False,
    )
    rows = [canonical_metrics(method, results[method], traces[method]) for method in results]
    b3 = results["B3_NODAL_ACI_MEM"]
    p = results["P_DUAL_MCI_ACI"]
    action_keys = ("hvac_kw", "ev_kw", "task_kw", "ess_charge_kw", "ess_discharge_kw")
    max_action_difference = max(
        float(np.max(np.abs(b3[key] - p[key]))) for key in action_keys
    )
    summary = {
        "date": day["date"],
        "cost_cap_gbp": float(cost_cap),
        "primary_improvement_pctpoint": float(
            100
            * (b3["source_emissions_kg"] - p["source_emissions_kg"])
            / b0["source_emissions_kg"]
        ),
        "p_minus_b3_cost_gbp": float(p["cost_gbp"] - b3["cost_gbp"]),
        "p_minus_b3_source_emissions_kg": float(
            p["source_emissions_kg"] - b3["source_emissions_kg"]
        ),
        "p_b3_max_action_difference_kw": max_action_difference,
        "p_b3_same_solution": bool(max_action_difference < 1e-5),
        "b2_naive_minus_memory_attribution_kg": float(
            naive_b2["attributed_emissions_kg"]
            - traces["B2_NODAL_ACI"]["attributed_emissions_kg"]
        ),
        "mci": mci_diagnostics,
        "aci_mci_mismatch": mismatch_metrics(
            memory_reference["nodal_ci_g_per_kwh"], mci
        ),
    }
    arrays = {
        "mci_g_per_kwh": mci,
        "reference_nodal_aci_g_per_kwh": memory_reference[
            "nodal_ci_g_per_kwh"
        ],
        "reference_storage_ci_g_per_kwh": memory_reference[
            "storage_ci_g_per_kwh"
        ],
    }
    for method, result in results.items():
        for key in action_keys + ("grid_import_kw", "grid_export_kw", "pv_array_kw"):
            arrays[f"{method}__{key}"] = result[key]
    return {"metrics": rows, "summary": summary, "arrays": arrays}
