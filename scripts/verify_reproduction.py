from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "expected_results.json").read_text(encoding="utf-8"))
MAIN = ROOT / "artifacts" / "experiment" / "pedf8_main_20260802_v1"
SIGNAL = ROOT / "artifacts" / "analysis" / "campaign_20260802_signal_role_adversarial"
PARETO = ROOT / "artifacts" / "analysis" / "campaign_20260802" / "S3_COST_PARETO"
REV = ROOT / "artifacts" / "analysis" / "campaign_20260802_internal_revision"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float) -> bool:
    tolerance = EXPECTED["numeric_tolerance"]["reported_metric_absolute"]
    return abs(float(actual) - float(expected)) <= tolerance


def main() -> None:
    failures: list[str] = []

    def scalar(name: str, actual, expected) -> None:
        if not close(actual, expected):
            failures.append(name)

    def exact(name: str, actual, expected) -> None:
        if actual != expected:
            failures.append(name)

    data_path = ROOT / "data" / "processed" / "timeseries.csv"
    if not data_path.exists():
        failures.append("missing processed time series")
    else:
        exact("processed row count", len(pd.read_csv(data_path)), EXPECTED["input"]["selected_rows"])
        exact("processed SHA-256", sha256(data_path), EXPECTED["input"]["processed_sha256"])

    required = [
        MAIN / "evaluation_summary.json",
        MAIN / "metrics.csv",
        SIGNAL / "summary.json",
        SIGNAL / "results.csv",
        PARETO / "method_results.csv",
        PARETO / "paired_effects.csv",
        REV / "REV-B4" / "results.csv",
        REV / "REV-B4" / "summary.json",
        REV / "REV-MCI-EPS" / "aggregate.csv",
        REV / "REV-Q0" / "aggregate.csv",
        REV / "REV-TIME-NULL" / "results.csv",
        REV / "REV-TIME-NULL" / "daily_randomization_summary.csv",
        REV / "REV-TIME-STAT" / "moving_block_bootstrap.csv",
        REV / "REV-MIP" / "results.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        failures.extend(f"missing {path}" for path in missing)
    else:
        main_summary = json.loads((MAIN / "evaluation_summary.json").read_text(encoding="utf-8"))
        expected_main = EXPECTED["main"]
        exact("main complete days", main_summary["complete_days"], expected_main["complete_days"])
        exact("main failed days", main_summary["failed_days"], expected_main["failed_days"])
        scalar("primary median", main_summary["primary_21_day"]["median_pctpoint"], expected_main["primary_median_pctpoint"])
        exact("primary positive days", main_summary["primary_21_day"]["positive_days"], expected_main["primary_positive_days"])
        scalar("later window median", main_summary["confirmation_7_day"]["median_pctpoint"], expected_main["later_window_median_pctpoint"])
        exact("later window positive days", main_summary["confirmation_7_day"]["positive_days"], expected_main["later_window_positive_days"])
        scalar("all day median", main_summary["all_available_effect"]["median_pctpoint"], expected_main["all_days_median_pctpoint"])
        exact("all day positive days", main_summary["all_available_effect"]["positive_days"], expected_main["all_days_positive_days"])

        metrics = pd.read_csv(MAIN / "metrics.csv")
        methods = ["B1_BULK_ACI", "B2_NODAL_ACI", "B3_NODAL_ACI_MEM", "P_DUAL_MCI_ACI"]
        winners = metrics[metrics["method"].isin(methods)].pivot(index="date", columns="method", values="source_emissions_kg").idxmin(axis=1).value_counts().to_dict()
        exact("surrogate winner counts", winners, expected_main["surrogate_winner_counts"])
        if metrics["energy_balance_max_kw"].max() > EXPECTED["numeric_tolerance"]["energy_balance_kw"]:
            failures.append("energy balance tolerance")
        if metrics["carbon_closure_relative"].max() > EXPECTED["numeric_tolerance"]["carbon_closure_relative"]:
            failures.append("carbon closure tolerance")

        signal_summary = json.loads((SIGNAL / "summary.json").read_text(encoding="utf-8"))
        signal_results = pd.read_csv(SIGNAL / "results.csv")
        expected_signal = EXPECTED["signal_role"]
        exact("signal role schedule count", len(signal_results), expected_signal["complete_schedules"])
        for row in signal_summary["curve"]:
            scalar(f"alpha {row['alpha']} median", row["median_effect_vs_alpha0_pctpoint"], expected_signal["alpha_median_effect_pctpoint"][str(row["alpha"])])
        order = signal_summary["actual_mci_vs_shuffled"]
        scalar("fixed shuffle actual order median", order["median_actual_order_gain_pctpoint"], expected_signal["fixed_shuffle_actual_order_gain_median_pctpoint"])
        exact("fixed shuffle better days", order["actual_order_better_days"], expected_signal["fixed_shuffle_actual_order_better_days"])
        if signal_results["branch_capacity_violation_kw"].max() > EXPECTED["numeric_tolerance"]["analysis_branch_capacity_kw"]:
            failures.append("analysis branch capacity tolerance")

        pareto = pd.read_csv(PARETO / "method_results.csv")
        paired = pd.read_csv(PARETO / "paired_effects.csv")
        p10 = pareto[(pareto["method"] == "P") & np.isclose(pareto["delta"], 0.10)]
        paired10 = paired[np.isclose(paired["delta"], 0.10)]
        expected_10 = EXPECTED["cost_budget_10_percent"]
        scalar("10 percent P rebound median", p10["emissions_change_vs_b0_pct"].median(), expected_10["p_median_change_vs_b0_pct"])
        exact("10 percent P rebound days", int((p10["emissions_change_vs_b0_pct"] > 0).sum()), expected_10["p_rebound_days"])
        exact("10 percent P reduction days", int((p10["emissions_change_vs_b0_pct"] < 0).sum()), expected_10["p_reduction_days"])
        scalar("10 percent P versus B3 median", paired10["effect_pctpoint"].median(), expected_10["p_vs_b3_median_pctpoint"])

        b4_summary = json.loads((REV / "REV-B4" / "summary.json").read_text(encoding="utf-8"))
        b4 = pd.read_csv(REV / "REV-B4" / "results.csv")
        expected_b4 = EXPECTED["direct_source_objective_2_percent"]
        exact("B4 day budget case count", b4_summary["completed_b4_cases"], expected_b4["complete_day_budget_cases"])
        at_two = b4[np.isclose(b4["delta"], 0.02)]
        for method, expected in expected_b4["median_regret_pctpoint"].items():
            scalar(f"{method} median regret", at_two.loc[at_two["method"] == method, "regret_vs_b4_pctpoint"].median(), expected)
        for method, expected in expected_b4["median_direction_agreement"].items():
            scalar(f"{method} direction agreement", at_two.loc[at_two["method"] == method, "action_direction_agreement"].median(), expected)
        candidates = at_two[at_two["method"].isin(["B1", "B2", "B3", "P"])]
        closest = candidates.loc[candidates.groupby("date")["regret_vs_b4_kg"].idxmin(), "method"].value_counts().to_dict()
        exact("closest nonreference days", closest, expected_b4["closest_nonreference_days"])
        lowest = at_two.pivot(index="date", columns="method", values="source_emissions_kg").idxmin(axis=1)
        exact("B4 lowest days", int(lowest.eq("B4").sum()), expected_b4["b4_lowest_days"])

        block = pd.read_csv(REV / "REV-TIME-STAT" / "moving_block_bootstrap.csv")
        for length, interval in EXPECTED["moving_block_bootstrap"].items():
            row = block[block["block_length_days"] == int(length)].iloc[0]
            scalar(f"block {length} low", row["ci_low_pctpoint"], interval[0])
            scalar(f"block {length} high", row["ci_high_pctpoint"], interval[1])

        temporal_results = pd.read_csv(REV / "REV-TIME-NULL" / "results.csv")
        temporal_daily = pd.read_csv(REV / "REV-TIME-NULL" / "daily_randomization_summary.csv")
        expected_temporal = EXPECTED["temporal_null"]
        exact("temporal null schedule count", len(temporal_results), expected_temporal["complete_schedules"])
        for name, expected in expected_temporal["median_actual_order_gain_pctpoint"].items():
            rows = temporal_daily[temporal_daily["null_type"] == name]
            scalar(f"{name} actual order median", rows["actual_gain_median_pctpoint"].median(), expected)
            exact(f"{name} significant days", int((rows["empirical_p_lower_emissions"] <= 0.05).sum()), expected_temporal["days_empirical_p_at_most_0_05"][name])

        epsilon = pd.read_csv(REV / "REV-MCI-EPS" / "aggregate.csv")
        for scale, expected in EXPECTED["mci_probe_scale"].items():
            row = epsilon[np.isclose(epsilon["epsilon_scale"], float(scale))].iloc[0]
            scalar(f"epsilon {scale} rank rho", row["median_rank_rho"], expected["median_rank_rho"])
            scalar(f"epsilon {scale} effect", row["median_effect_pctpoint"], expected["median_effect_pctpoint"])
            exact(f"epsilon {scale} stable days", int(row["stable_days"]), expected["stable_days"])

        boundary = pd.read_csv(REV / "REV-Q0" / "aggregate.csv")
        for name, expected in EXPECTED["storage_carbon_boundary"].items():
            row = boundary[boundary["boundary"] == name].iloc[0]
            scalar(f"{name} boundary effect", row["median_effect_pctpoint"], expected["median_effect_pctpoint"])
            exact(f"{name} boundary positive days", int(row["positive_days"]), expected["positive_days"])

        mip = pd.read_csv(REV / "REV-MIP" / "results.csv")
        expected_mip = EXPECTED["mip"]
        exact("MIP complete cases", len(mip), expected_mip["complete_cases"])
        scalar("MIP realized relative gap", mip["scip_realized_relative_gap"].max(), expected_mip["max_realized_relative_gap"])
        mip_two = mip[np.isclose(mip["delta"], 0.02)]
        for method, expected in expected_mip["two_percent_max_action_difference_kw"].items():
            scalar(f"{method} two percent action difference", mip_two.loc[mip_two["method"] == method, "action_max_absolute_difference_kw"].max(), expected)
        mip_b3_ten = mip[np.isclose(mip["delta"], 0.10) & mip["method"].eq("B3")]
        scalar("B3 ten percent action difference", mip_b3_ten["action_max_absolute_difference_kw"].max(), expected_mip["ten_percent_b3_max_action_difference_kw"])
        scalar("B3 ten percent objective difference", mip_b3_ten["heuristic_minus_mip_objective"].abs().max(), expected_mip["ten_percent_b3_max_absolute_objective_difference"])

    if failures:
        print(json.dumps({"status": "FAIL", "checks_failed": failures}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"status": "PASS", "contract": EXPECTED["contract_version"]}, indent=2))


if __name__ == "__main__":
    main()
