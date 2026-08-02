import numpy as np

from pedf import DayOptimizer, load_config, load_days, signals_from_trace, trace_carbon


def test_real_day_dispatch_and_carbon_closure():
    config = load_config("configs/pedf8.json")
    day = load_days("data/processed/timeseries.csv")[0]
    model = DayOptimizer(config, day)
    result = model.solve_econ()
    assert result is not None
    assert result["energy_balance_max_kw"] < config["tolerance"]["energy_balance_kw"]
    assert result["socp_gap_max"] < config["tolerance"]["socp_gap"]
    assert result["comfort_violation_c_h"] < 1e-8
    assert result["voltage_violation_pu"] < 1e-8
    assert result["terminal_soc_error_kwh"] < 1e-8
    assert result["simultaneous_charge_discharge_kw"] < 1e-6
    trace = trace_carbon(
        result,
        config,
        day["carbon_g_per_kwh"],
        day["load_kw"],
        memory=True,
    )
    assert trace["carbon_closure_relative"] < config["tolerance"]["carbon_unit_test_relative"]

    b3 = model.solve_signal(
        signals_from_trace(trace, day["carbon_g_per_kwh"], "memory"),
        result["cost_gbp"] * 1.02,
    )
    assert b3 is not None
    assert b3["socp_gap_max"] < 1e-12
    assert b3["branch_capacity_violation_kw"] < 1e-6
    assert b3["simultaneous_charge_discharge_kw"] < 1e-6
    assert b3["cost_cap_violation_gbp"] < 1e-4

    b4 = model.solve_emissions(result["cost_gbp"] * 1.02)
    assert b4 is not None
    assert b4["source_emissions_kg"] <= result["source_emissions_kg"] + 1e-6
    assert b4["socp_gap_max"] < 1e-12
    assert b4["branch_capacity_violation_kw"] < 1e-6
    assert b4["simultaneous_charge_discharge_kw"] < 1e-6
    assert b4["cost_cap_violation_gbp"] < 1e-4


def test_handcrafted_grid_only_carbon_balance():
    config = load_config("configs/pedf8.json")
    t = config["steps_per_day"]
    eta = config["efficiency"]["pcc_import"]
    p = np.zeros((len(config["branches"]), t))
    p[:, :] = np.array([10.0, 3.0, 1.5, 2.0, 0.0, 1.0, 1.5])[:, None] / 100
    result = {
        "branch_p_pu": p,
        "branch_loss_kw": np.zeros_like(p),
        "grid_import_kw": np.full(t, 10.0 / eta),
        "grid_export_kw": np.zeros(t),
        "pv_array_kw": np.zeros(t),
        "ess_charge_kw": np.zeros(t),
        "ess_discharge_kw": np.zeros(t),
        "ess_energy_kwh": np.full(t + 1, 80.0),
        "hvac_kw": np.full(t, 2.0),
        "ev_kw": np.full(t, 1.0),
        "task_kw": np.full(t, 1.5),
    }
    trace = trace_carbon(
        result,
        config,
        np.full(t, 100.0),
        np.full(t, 10.0),
        memory=True,
        initial_storage_ci=100.0,
    )
    assert np.max(np.abs(trace["nodal_ci_g_per_kwh"] - 100.0)) < 1e-10
    assert trace["carbon_closure_relative"] < 1e-12
