from __future__ import annotations

import numpy as np


def trace_carbon(
    result: dict,
    config: dict,
    bulk_ci: np.ndarray,
    load_kw: np.ndarray,
    memory: bool = True,
    initial_storage_ci: float | None = None,
) -> dict:
    n = config["nodes"]
    t_count = config["steps_per_day"]
    dt = config["dt_h"]
    sbase = config["s_base_kw"]
    edges = [tuple(x) for x in config["branches"]]
    eff = config["efficiency"]
    shares = config["load_share"]
    nodes = config.get(
        "device_nodes",
        {"hvac": 2, "ess": 3, "pv": 5, "ev": 6, "task": 7},
    )
    fixed_weights = None
    if "fixed_load_weights" in config:
        fixed_weights = np.asarray(config["fixed_load_weights"], float)
        fixed_weights = fixed_weights / fixed_weights.sum()
    initial_ci = float(bulk_ci[0] if initial_storage_ci is None else initial_storage_ci)
    energy = result["ess_energy_kwh"]
    carbon_stock = np.zeros(t_count + 1)
    carbon_stock[0] = energy[0] * initial_ci
    intensity = np.zeros((n, t_count))
    storage_ci = np.zeros(t_count)
    line_loss_carbon = 0.0
    pcc_loss_carbon = 0.0
    ess_charge_loss_carbon = 0.0
    ess_discharge_loss_carbon = 0.0
    export_carbon = 0.0
    export_loss_carbon = 0.0
    fixed_carbon = 0.0
    hvac_carbon = 0.0
    ev_carbon = 0.0
    task_carbon = 0.0

    p_branch = result["branch_p_pu"]
    branch_loss = result["branch_loss_kw"]
    grid_import = result["grid_import_kw"]
    grid_export = result["grid_export_kw"]
    pv = result["pv_array_kw"]
    charge = result["ess_charge_kw"]
    discharge = result["ess_discharge_kw"]
    hvac = result["hvac_kw"]
    ev = result["ev_kw"]
    task = result["task_kw"]

    for t in range(t_count):
        q_storage = carbon_stock[t] / max(energy[t], 1e-12)
        storage_ci[t] = q_storage
        incoming: list[tuple[int, int, float]] = []
        losses: list[tuple[int, float]] = []
        for e, (i, j, _, _) in enumerate(edges):
            p_kw = p_branch[e, t] * sbase
            loss_kw = max(branch_loss[e, t], 0.0)
            if p_kw >= 0:
                received = p_kw - loss_kw
                if received < -1e-5:
                    raise ValueError("forward branch loss exceeds sent power")
                incoming.append((i, j, max(received, 0.0)))
                losses.append((i, loss_kw))
            else:
                incoming.append((j, i, -p_kw))
                losses.append((j, loss_kw))

        source_power = np.zeros(n)
        source_carbon_rate = np.zeros(n)
        source_power[0] += eff["pcc_import"] * grid_import[t]
        source_carbon_rate[0] += (
            eff["pcc_import"] * grid_import[t] * bulk_ci[t]
        )
        source_power[nodes["pv"]] += eff["pv"] * pv[t]
        source_power[nodes["ess"]] += discharge[t]
        if memory:
            source_carbon_rate[nodes["ess"]] += discharge[t] * q_storage

        matrix = np.zeros((n, n))
        rhs = source_carbon_rate.copy()
        total_in = source_power.copy()
        for source, destination, received in incoming:
            total_in[destination] += received
            matrix[destination, source] -= received
        for node in range(n):
            matrix[node, node] += total_in[node]
        if not memory:
            matrix[nodes["ess"], nodes["ess"]] -= discharge[t]
        for node in range(n):
            if np.sum(np.abs(matrix[node])) < 1e-10:
                matrix[node, node] = 1.0
                rhs[node] = bulk_ci[t]
        try:
            node_ci = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            node_ci = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        if not np.all(np.isfinite(node_ci)):
            raise ValueError("non-finite nodal carbon intensity")
        intensity[:, t] = node_ci

        line_loss_carbon += dt * sum(node_ci[node] * loss for node, loss in losses)
        pcc_loss_carbon += (
            dt * bulk_ci[t] * (1 - eff["pcc_import"]) * grid_import[t]
        )
        if fixed_weights is None:
            fixed_carbon += dt * (
                node_ci[1] * shares["fixed_node_1"] * load_kw[t]
                + node_ci[4] * shares["fixed_node_4"] * load_kw[t]
            )
        else:
            fixed_share = shares["fixed_node_1"] + shares["fixed_node_4"]
            fixed_carbon += dt * fixed_share * load_kw[t] * np.dot(node_ci, fixed_weights)
        hvac_carbon += dt * node_ci[nodes["hvac"]] * hvac[t]
        ev_carbon += dt * node_ci[nodes["ev"]] * ev[t]
        task_carbon += dt * node_ci[nodes["task"]] * task[t]
        ess_charge_loss_carbon += (
            dt * node_ci[nodes["ess"]] * (1 - eff["ess_charge"]) * charge[t]
        )
        ess_discharge_loss_carbon += (
            dt * q_storage * (1 / eff["ess_discharge"] - 1) * discharge[t]
        )
        export_carbon += dt * node_ci[0] * grid_export[t]
        export_loss_carbon += (
            dt
            * node_ci[0]
            * (1 / eff["pcc_export"] - 1)
            * grid_export[t]
        )
        if memory:
            carbon_stock[t + 1] = carbon_stock[t] + dt * (
                node_ci[nodes["ess"]] * eff["ess_charge"] * charge[t]
                - q_storage * discharge[t] / eff["ess_discharge"]
            )
            if carbon_stock[t + 1] < -1e-6:
                raise ValueError("negative storage carbon stock")
            carbon_stock[t + 1] = max(carbon_stock[t + 1], 0.0)
        else:
            carbon_stock[t + 1] = carbon_stock[t]

    source_grid_carbon = float(dt * np.sum(grid_import * bulk_ci))
    end_use_carbon = fixed_carbon + hvac_carbon + ev_carbon + task_carbon
    loss_carbon = (
        line_loss_carbon
        + pcc_loss_carbon
        + ess_charge_loss_carbon
        + ess_discharge_loss_carbon
        + export_loss_carbon
    )
    lhs = carbon_stock[0] + source_grid_carbon
    rhs_total = carbon_stock[-1] + end_use_carbon + loss_carbon + export_carbon
    closure = abs(lhs - rhs_total) / max(abs(lhs), 1.0) if memory else np.nan
    return {
        "nodal_ci_g_per_kwh": intensity,
        "storage_ci_g_per_kwh": storage_ci,
        "storage_carbon_stock_g": carbon_stock,
        "source_grid_carbon_kg": source_grid_carbon / 1000,
        "attributed_emissions_kg": (end_use_carbon + loss_carbon) / 1000,
        "end_use_carbon_kg": end_use_carbon / 1000,
        "loss_carbon_kg": loss_carbon / 1000,
        "export_carbon_kg": export_carbon / 1000,
        "carbon_closure_relative": float(closure),
        "device_nodes": nodes,
        "carbon_components_kg": {
            "fixed": fixed_carbon / 1000,
            "hvac": hvac_carbon / 1000,
            "ev_including_converter": ev_carbon / 1000,
            "task": task_carbon / 1000,
            "line_loss": line_loss_carbon / 1000,
            "pcc_import_loss": pcc_loss_carbon / 1000,
            "ess_charge_loss": ess_charge_loss_carbon / 1000,
            "ess_discharge_loss": ess_discharge_loss_carbon / 1000,
            "export_delivered": export_carbon / 1000,
            "export_converter_loss": export_loss_carbon / 1000,
        },
    }


def signals_from_trace(trace: dict, bulk_ci: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    if mode == "bulk":
        return {key: np.asarray(bulk_ci, float).copy() for key in ("hvac", "ev", "task", "charge", "discharge")}
    ci = trace["nodal_ci_g_per_kwh"]
    nodes = trace.get(
        "device_nodes",
        {"hvac": 2, "ess": 3, "pv": 5, "ev": 6, "task": 7},
    )
    signals = {
        "hvac": ci[nodes["hvac"]].copy(),
        "ev": ci[nodes["ev"]].copy(),
        "task": ci[nodes["task"]].copy(),
        "charge": ci[nodes["ess"]].copy(),
        "discharge": ci[nodes["ess"]].copy(),
    }
    if mode == "memory":
        signals["discharge"] = trace["storage_ci_g_per_kwh"].copy()
    return signals
