from __future__ import annotations

import json
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


OPTIMAL = {"optimal", "optimal_inaccurate"}


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_days(path: str | Path) -> list[dict]:
    frame = pd.read_csv(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["timestamp_london"] = frame["timestamp_utc"].dt.tz_convert("Europe/London")
    days = []
    for date, day in frame.groupby(frame["timestamp_london"].dt.date, sort=True):
        day = day.reset_index(drop=True)
        if len(day) != 48:
            raise ValueError(f"{date} has {len(day)} rows")
        days.append(
            {
                "date": str(date),
                "timestamp_utc": day["timestamp_utc"],
                "timestamp_london": day["timestamp_london"],
                "load_kw": day["load_kw"].to_numpy(float),
                "carbon_g_per_kwh": day["carbon_g_per_kwh"].to_numpy(float),
                "price_gbp_per_kwh": day["price_gbp_per_kwh"].to_numpy(float),
                "temperature_c": day["temperature_c"].to_numpy(float),
                "shortwave_w_m2": day["shortwave_w_m2"].to_numpy(float),
            }
        )
    return days


class DayOptimizer:
    def __init__(self, config: dict, day: dict):
        self.cfg = config
        self.day = day
        self.n = config["nodes"]
        self.t = config["steps_per_day"]
        self.dt = config["dt_h"]
        self.sbase = config["s_base_kw"]
        self.edges = [tuple(x) for x in config["branches"]]
        self._build()

    def _build(self) -> None:
        c = self.cfg
        t = self.t
        eff = c["efficiency"]
        shares = c["load_share"]
        flexibility = c.get(
            "flexibility", {"hvac": True, "ev": True, "task": True}
        )
        load = self.day["load_kw"]
        price = self.day["price_gbp_per_kwh"]
        outside = self.day["temperature_c"]
        hours = self.day["timestamp_london"].dt.hour.to_numpy()

        temp_factor = 1 + c["pv"]["temperature_coefficient_per_c"] * (
            outside - c["pv"]["reference_temperature_c"]
        )
        self.pv_available = np.maximum(
            0,
            c["pv"]["capacity_kwp"]
            * self.day["shortwave_w_m2"]
            / 1000
            * temp_factor,
        )
        self.task_base = shares["task"] * load
        self.probe = cp.Parameter((self.n, t), value=np.zeros((self.n, t)))
        self.cost_cap = cp.Parameter(nonneg=True, value=1e6)
        self.charge_gate = cp.Parameter(t, nonneg=True, value=np.ones(t))
        self.discharge_gate = cp.Parameter(t, nonneg=True, value=np.ones(t))
        self.signal = {
            key: cp.Parameter(t, value=np.zeros(t))
            for key in ("hvac", "ev", "task", "charge", "discharge")
        }

        p = cp.Variable((len(self.edges), t))
        ell = cp.Variable((len(self.edges), t), nonneg=True)
        voltage = cp.Variable((self.n, t))
        grid_import = cp.Variable(t, nonneg=True)
        grid_export = cp.Variable(t, nonneg=True)
        pv = cp.Variable(t, nonneg=True)
        charge = cp.Variable(t, nonneg=True)
        discharge = cp.Variable(t, nonneg=True)
        energy = cp.Variable(t + 1)
        hvac = cp.Variable(t, nonneg=True)
        indoor = cp.Variable(t + 1)
        ev = cp.Variable(t, nonneg=True)
        task = cp.Variable(t, nonneg=True)
        ess_mode = cp.Variable(t, boolean=True)
        self.var = {
            "branch_p_pu": p,
            "branch_l_pu": ell,
            "voltage_sq_pu": voltage,
            "grid_import_kw": grid_import,
            "grid_export_kw": grid_export,
            "pv_array_kw": pv,
            "ess_charge_kw": charge,
            "ess_discharge_kw": discharge,
            "ess_energy_kwh": energy,
            "hvac_kw": hvac,
            "indoor_temperature_c": indoor,
            "ev_kw": ev,
            "task_kw": task,
        }

        constraints = [voltage[0, :] == 1]
        vmin = c["voltage_min_pu"] ** 2
        vmax = c["voltage_max_pu"] ** 2
        constraints += [voltage >= vmin, voltage <= vmax]

        children: dict[int, list[int]] = {i: [] for i in range(self.n)}
        incoming: dict[int, int] = {}
        for e, (i, j, r, pmax_kw) in enumerate(self.edges):
            children[i].append(e)
            incoming[j] = e
            pmax = pmax_kw / self.sbase
            constraints += [p[e, :] <= pmax, p[e, :] >= -pmax]
            constraints += [
                p[e, :] - r * ell[e, :] <= pmax,
                p[e, :] - r * ell[e, :] >= -pmax,
            ]
            constraints += [ell[e, :] <= (pmax / c["voltage_min_pu"]) ** 2]
            constraints += [
                voltage[j, :]
                == voltage[i, :] - 2 * r * p[e, :] + r**2 * ell[e, :]
            ]
            for k in range(t):
                constraints.append(
                    cp.SOC(
                        voltage[i, k] + ell[e, k],
                        cp.hstack(
                            [2 * p[e, k], voltage[i, k] - ell[e, k]]
                        ),
                    )
                )

        constraints += [pv <= self.pv_available]
        ess = c["ess"]
        constraints += [charge <= ess["power_charge_kw"], discharge <= ess["power_discharge_kw"]]
        e0 = ess["capacity_kwh"] * ess["soc_initial"]
        constraints += [energy[0] == e0, energy[-1] == e0]
        constraints += [
            energy >= ess["capacity_kwh"] * ess["soc_min"],
            energy <= ess["capacity_kwh"] * ess["soc_max"],
        ]
        constraints.append(
            energy[1:]
            == energy[:-1]
            + self.dt
            * (
                eff["ess_charge"] * charge
                - discharge / eff["ess_discharge"]
            )
        )

        hvac_cfg = c["hvac"]
        constraints += [hvac <= hvac_cfg["power_max_kw"]]
        constraints += [indoor[0] == hvac_cfg["temperature_initial_c"]]
        constraints.append(
            indoor[1:]
            == hvac_cfg["thermal_a"] * indoor[:-1]
            + (1 - hvac_cfg["thermal_a"]) * outside
            + hvac_cfg["thermal_beta_c_per_kwh"] * hvac * self.dt
        )
        occupied = (hours >= hvac_cfg["occupied_start_hour"]) & (
            hours < hvac_cfg["occupied_end_hour"]
        )
        lower = np.where(
            occupied,
            hvac_cfg["occupied_min_c"],
            hvac_cfg["unoccupied_min_c"],
        )
        upper = np.where(
            occupied,
            hvac_cfg["occupied_max_c"],
            hvac_cfg["unoccupied_max_c"],
        )
        self.temperature_lower = lower
        self.temperature_upper = upper
        if flexibility["hvac"]:
            constraints += [indoor[:-1] >= lower, indoor[:-1] <= upper]
            constraints += [
                indoor[-1] >= hvac_cfg["unoccupied_min_c"],
                indoor[-1] <= hvac_cfg["unoccupied_max_c"],
            ]
            constraints += [
                cp.sum(hvac) * self.dt == shares["hvac"] * np.sum(load) * self.dt
            ]
        else:
            constraints += [hvac == shares["hvac"] * load]

        ev_cfg = c["ev"]
        available = (hours >= ev_cfg["available_start_hour"]) | (
            hours < ev_cfg["available_end_hour"]
        )
        self.ev_available = available
        constraints += [ev <= ev_cfg["power_max_kw"] * available.astype(float)]
        self.ev_target_kwh = shares["ev"] * np.sum(load) * self.dt
        if flexibility["ev"]:
            constraints += [
                eff["ev"] * cp.sum(ev) * self.dt == self.ev_target_kwh
            ]
        else:
            weights = load * available.astype(float)
            fixed_ev = (
                self.ev_target_kwh
                / (eff["ev"] * self.dt)
                * weights
                / np.sum(weights)
            )
            constraints += [ev == fixed_ev]

        task_cfg = c["task"]
        if flexibility["task"]:
            constraints += [task <= task_cfg["power_multiplier_max"] * self.task_base]
            constraints += [cp.sum(task) == np.sum(self.task_base)]
            served = cp.cumsum(task) * self.dt
            arrived = np.cumsum(self.task_base) * self.dt
            constraints += [served <= arrived]
            delay = task_cfg["maximum_delay_steps"]
            for k in range(delay, t):
                constraints.append(served[k] >= arrived[k - delay])
        else:
            constraints += [task == self.task_base]

        node_demand = [self.probe[i, :] for i in range(self.n)]
        node_demand[1] = node_demand[1] + shares["fixed_node_1"] * load
        node_demand[2] = node_demand[2] + hvac
        node_demand[3] = node_demand[3] + charge - discharge
        node_demand[4] = node_demand[4] + shares["fixed_node_4"] * load
        node_demand[5] = node_demand[5] - eff["pv"] * pv
        node_demand[6] = node_demand[6] + ev
        node_demand[7] = node_demand[7] + task
        self.node_demand = node_demand

        for node in range(1, self.n):
            e = incoming[node]
            child_flow = sum((p[k, :] for k in children[node]), 0)
            r = self.edges[e][2]
            constraints.append(
                p[e, :] - r * ell[e, :]
                == node_demand[node] / self.sbase + child_flow
            )

        eta_in = eff["pcc_import"]
        eta_out = eff["pcc_export"]
        constraints += [
            p[0, :] == (eta_in * grid_import - grid_export / eta_out) / self.sbase,
            grid_import <= c["grid"]["import_max_kw"],
            grid_export <= c["grid"]["export_max_kw"],
        ]
        negative_price = price < 0
        if negative_price.any():
            constraints.append(grid_export[negative_price] == 0)

        degradation = ess["degradation_gbp_per_kwh_throughput"]
        export_revenue = c["grid"]["export_revenue_gbp_per_kwh"]
        cost = self.dt * (
            cp.sum(cp.multiply(price, grid_import))
            - export_revenue * cp.sum(grid_export)
            + degradation * cp.sum(charge + discharge)
        )
        action = self.dt * (
            self.signal["hvac"] @ hvac
            + self.signal["ev"] @ ev
            + self.signal["task"] @ task
            + self.signal["charge"] @ charge
            - self.signal["discharge"] @ discharge
        )
        source_emissions = (
            self.dt
            * cp.sum(cp.multiply(self.day["carbon_g_per_kwh"], grid_import))
            / 1000
        )
        regularizer = 1e-9 * (
            cp.sum_squares(grid_import)
            + cp.sum_squares(grid_export)
            + cp.sum_squares(charge)
            + cp.sum_squares(discharge)
            + cp.sum_squares(hvac)
            + cp.sum_squares(ev)
            + cp.sum_squares(task)
        ) + 1e-5 * cp.sum(ell)
        self.cost = cost
        self.action = action
        self.source_emissions = source_emissions
        self.econ_problem = cp.Problem(cp.Minimize(cost + regularizer), constraints)
        signal_constraints = constraints + [cost <= self.cost_cap]
        signal_objective = cp.Minimize(action + 1e-3 * cost + 1e-6 * cp.sum(ell))
        emission_objective = cp.Minimize(
            source_emissions + 1e-9 * cost + 1e-10 * cp.sum(ell)
        )
        self.relaxed_signal_problem = cp.Problem(signal_objective, signal_constraints)
        self.signal_problem = cp.Problem(
            signal_objective,
            signal_constraints
            + [
                charge <= ess["power_charge_kw"] * self.charge_gate,
                discharge <= ess["power_discharge_kw"] * self.discharge_gate,
            ],
        )
        self.mip_signal_problem = cp.Problem(
            signal_objective,
            signal_constraints
            + [
                charge <= ess["power_charge_kw"] * ess_mode,
                discharge <= ess["power_discharge_kw"] * (1 - ess_mode),
            ],
        )
        self.relaxed_emission_problem = cp.Problem(
            emission_objective, signal_constraints
        )
        self.emission_problem = cp.Problem(
            emission_objective,
            signal_constraints
            + [
                charge <= ess["power_charge_kw"] * self.charge_gate,
                discharge <= ess["power_discharge_kw"] * self.discharge_gate,
            ],
        )
        self.mip_emission_problem = cp.Problem(
            emission_objective,
            signal_constraints
            + [
                charge <= ess["power_charge_kw"] * ess_mode,
                discharge <= ess["power_discharge_kw"] * (1 - ess_mode),
            ],
        )

    def _solve(
        self,
        problem: cp.Problem,
        solver: str | None = None,
        verbose: bool = False,
    ) -> tuple[str, float]:
        solver_cfg = self.cfg["solver"]
        solver = solver or solver_cfg["name"]
        start = time.perf_counter()
        try:
            if solver == "ECOS_BB":
                problem.solve(
                    solver=solver,
                    verbose=verbose,
                    mi_max_iters=100000,
                    abstol=1e-8,
                    reltol=1e-8,
                    feastol=1e-8,
                )
            elif solver == "SCIP":
                problem.solve(
                    solver=solver,
                    verbose=verbose,
                    scip_params={
                        "limits/time": 300.0,
                        "limits/gap": 1e-6,
                        "numerics/feastol": 1e-8,
                    },
                )
            else:
                problem.solve(
                    solver=solver,
                    verbose=verbose,
                    max_iter=solver_cfg["max_iter"],
                    tol_gap_abs=solver_cfg["tol_gap_abs"],
                    tol_feas=solver_cfg["tol_feas"],
                    warm_start=True,
                )
        except cp.error.SolverError:
            return "solver_error", time.perf_counter() - start
        return problem.status, time.perf_counter() - start

    def solve_econ(self, probe_kw: np.ndarray | None = None) -> dict | None:
        self.probe.value = np.zeros((self.n, self.t)) if probe_kw is None else probe_kw
        status, runtime = self._solve(self.econ_problem)
        return self._collect(status, runtime) if status in OPTIMAL else None

    def solve_signal(self, signals: dict[str, np.ndarray], cost_cap: float) -> dict | None:
        self.probe.value = np.zeros((self.n, self.t))
        for key, parameter in self.signal.items():
            parameter.value = np.asarray(signals[key], dtype=float)
        internal_cap = float(cost_cap)
        self.cost_cap.value = internal_cap
        relaxed_status, runtime = self._solve(self.relaxed_signal_problem)
        if relaxed_status not in OPTIMAL:
            return None
        relaxed_charge = np.asarray(self.var["ess_charge_kw"].value, float).copy()
        relaxed_discharge = np.asarray(self.var["ess_discharge_kw"].value, float).copy()
        relaxed_action = float(self.action.value)
        active = np.maximum(relaxed_charge, relaxed_discharge) > 1e-7
        charge_gate = (relaxed_charge >= relaxed_discharge) & active
        discharge_gate = (relaxed_discharge > relaxed_charge) & active
        self.charge_gate.value = charge_gate.astype(float)
        self.discharge_gate.value = discharge_gate.astype(float)
        result = None
        for _ in range(5):
            self.cost_cap.value = internal_cap
            status, solve_runtime = self._solve(self.signal_problem)
            runtime += solve_runtime
            if status not in OPTIMAL:
                return None
            result = self._collect(status, runtime)
            violation = max(result["cost_gbp"] - cost_cap, 0.0)
            if violation <= 1e-4:
                break
            internal_cap -= violation + 1e-4
        result["action_objective"] = float(self.action.value)
        result["cost_cap_violation_gbp"] = max(result["cost_gbp"] - cost_cap, 0.0)
        result["internal_cost_cap_adjustment_gbp"] = float(cost_cap - internal_cap)
        result["mode_relaxation_simultaneous_kw"] = float(
            np.max(np.minimum(relaxed_charge, relaxed_discharge))
        )
        result["mode_restoration_action_gap"] = float(self.action.value - relaxed_action)
        return result

    def solve_signal_mip(
        self,
        signals: dict[str, np.ndarray],
        cost_cap: float,
        solver: str = "SCIP",
        verbose: bool = False,
    ) -> dict | None:
        self.probe.value = np.zeros((self.n, self.t))
        self.cost_cap.value = float(cost_cap)
        for key, parameter in self.signal.items():
            parameter.value = np.asarray(signals[key], dtype=float)
        status, runtime = self._solve(
            self.mip_signal_problem, solver=solver, verbose=verbose
        )
        if status not in OPTIMAL:
            return None
        result = self._collect(status, runtime)
        result["action_objective"] = float(self.action.value)
        result["mip_solver"] = solver
        return result

    def solve_emissions(self, cost_cap: float) -> dict | None:
        self.probe.value = np.zeros((self.n, self.t))
        internal_cap = float(cost_cap)
        self.cost_cap.value = internal_cap
        relaxed_status, runtime = self._solve(self.relaxed_emission_problem)
        if relaxed_status not in OPTIMAL:
            return None
        relaxed_charge = np.asarray(self.var["ess_charge_kw"].value, float).copy()
        relaxed_discharge = np.asarray(self.var["ess_discharge_kw"].value, float).copy()
        relaxed_emissions = float(self.source_emissions.value)
        active = np.maximum(relaxed_charge, relaxed_discharge) > 1e-7
        self.charge_gate.value = ((relaxed_charge >= relaxed_discharge) & active).astype(float)
        self.discharge_gate.value = ((relaxed_discharge > relaxed_charge) & active).astype(float)
        result = None
        for _ in range(5):
            self.cost_cap.value = internal_cap
            status, solve_runtime = self._solve(self.emission_problem)
            runtime += solve_runtime
            if status not in OPTIMAL:
                return None
            result = self._collect(status, runtime)
            violation = max(result["cost_gbp"] - cost_cap, 0.0)
            if violation <= 1e-4:
                break
            internal_cap -= violation + 1e-4
        result["emission_objective_kg"] = float(self.source_emissions.value)
        result["cost_cap_violation_gbp"] = max(result["cost_gbp"] - cost_cap, 0.0)
        result["internal_cost_cap_adjustment_gbp"] = float(cost_cap - internal_cap)
        result["mode_relaxation_simultaneous_kw"] = float(
            np.max(np.minimum(relaxed_charge, relaxed_discharge))
        )
        result["mode_restoration_emissions_gap_kg"] = float(
            self.source_emissions.value - relaxed_emissions
        )
        return result

    def solve_emissions_mip(
        self,
        cost_cap: float,
        solver: str = "SCIP",
        verbose: bool = False,
    ) -> dict | None:
        self.probe.value = np.zeros((self.n, self.t))
        self.cost_cap.value = float(cost_cap)
        status, runtime = self._solve(
            self.mip_emission_problem, solver=solver, verbose=verbose
        )
        if status not in OPTIMAL:
            return None
        result = self._collect(status, runtime)
        result["emission_objective_kg"] = float(self.source_emissions.value)
        result["mip_solver"] = solver
        return result

    def _recover_physical_flow(self, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        p = np.zeros((len(self.edges), self.t))
        ell = np.zeros_like(p)
        voltage = np.ones((self.n, self.t))
        children: dict[int, list[int]] = {i: [] for i in range(self.n)}
        for e, (i, _, _, _) in enumerate(self.edges):
            children[i].append(e)
        max_iterations = 0
        for t in range(self.t):
            for iteration in range(200):
                previous = ell[:, t].copy()
                for e in range(len(self.edges) - 1, -1, -1):
                    _, j, r, _ = self.edges[e]
                    p[e, t] = demand[j, t] / self.sbase + sum(
                        p[child, t] for child in children[j]
                    ) + r * ell[e, t]
                voltage[0, t] = 1.0
                for e, (i, j, r, _) in enumerate(self.edges):
                    voltage[j, t] = (
                        voltage[i, t] - 2 * r * p[e, t] + r**2 * ell[e, t]
                    )
                    ell[e, t] = p[e, t] ** 2 / max(voltage[i, t], 1e-8)
                if np.max(np.abs(ell[:, t] - previous)) < 1e-12:
                    break
            max_iterations = max(max_iterations, iteration + 1)
        net_root_kw = p[0] * self.sbase
        eta_in = self.cfg["efficiency"]["pcc_import"]
        eta_out = self.cfg["efficiency"]["pcc_export"]
        grid_import = np.maximum(net_root_kw, 0) / eta_in
        grid_export = np.maximum(-net_root_kw, 0) * eta_out
        return p, ell, voltage, grid_import, grid_export, max_iterations

    def _collect(self, status: str, runtime: float) -> dict:
        c = self.cfg
        eff = c["efficiency"]
        values = {key: np.asarray(var.value, dtype=float) for key, var in self.var.items()}
        demand = np.vstack([np.asarray(x.value, dtype=float) for x in self.node_demand])
        relaxed_p = values["branch_p_pu"].copy()
        relaxed_l = np.maximum(values["branch_l_pu"], 0)
        relaxed_voltage = values["voltage_sq_pu"].copy()
        relaxation_gaps = []
        for e, (i, _, _, _) in enumerate(self.edges):
            relaxation_gaps.append(
                relaxed_voltage[i] * relaxed_l[e] - relaxed_p[e] ** 2
            )
        p, ell, voltage, recovered_import, recovered_export, recovery_iterations = self._recover_physical_flow(demand)
        values["branch_p_pu"] = p
        values["branch_l_pu"] = ell
        values["voltage_sq_pu"] = voltage
        values["grid_import_kw"] = recovered_import
        values["grid_export_kw"] = recovered_export
        children: dict[int, list[int]] = {i: [] for i in range(self.n)}
        incoming: dict[int, int] = {}
        for e, (i, j, _, _) in enumerate(self.edges):
            children[i].append(e)
            incoming[j] = e
        residuals = []
        for node in range(1, self.n):
            e = incoming[node]
            r = self.edges[e][2]
            rhs = demand[node] / self.sbase + sum((p[k] for k in children[node]), 0)
            residuals.append((p[e] - r * ell[e] - rhs) * self.sbase)
        residuals.append(
            p[0] * self.sbase
            - (
                eff["pcc_import"] * values["grid_import_kw"]
                - values["grid_export_kw"] / eff["pcc_export"]
            )
        )
        gaps = []
        line_loss_kw = np.zeros_like(ell)
        for e, (i, _, r, _) in enumerate(self.edges):
            gaps.append(voltage[i] * ell[e] - p[e] ** 2)
            line_loss_kw[e] = r * ell[e] * self.sbase
        import_kw = values["grid_import_kw"]
        export_kw = values["grid_export_kw"]
        charge = values["ess_charge_kw"]
        discharge = values["ess_discharge_kw"]
        ev = values["ev_kw"]
        pv = values["pv_array_kw"]
        price = self.day["price_gbp_per_kwh"]
        cost_gbp = self.dt * (
            np.sum(price * import_kw)
            - c["grid"]["export_revenue_gbp_per_kwh"] * np.sum(export_kw)
            + c["ess"]["degradation_gbp_per_kwh_throughput"]
            * np.sum(charge + discharge)
        )
        conversion_loss_kw = (
            (1 - eff["pcc_import"]) * import_kw
            + (1 / eff["pcc_export"] - 1) * export_kw
            + (1 - eff["pv"]) * pv
            + (1 - eff["ess_charge"]) * charge
            + (1 / eff["ess_discharge"] - 1) * discharge
            + (1 - eff["ev"]) * ev
        )
        indoor = values["indoor_temperature_c"][:-1]
        temp_violation = np.maximum(self.temperature_lower - indoor, 0) + np.maximum(
            indoor - self.temperature_upper, 0
        )
        vmag = np.sqrt(np.maximum(voltage, 0))
        voltage_violation = np.maximum(c["voltage_min_pu"] - vmag, 0) + np.maximum(
            vmag - c["voltage_max_pu"], 0
        )
        task = values["task_kw"]
        return {
            "date": self.day["date"],
            "solver_status": status,
            "runtime_s": runtime,
            "cost_gbp": float(cost_gbp),
            "source_emissions_kg": float(
                self.dt
                * np.sum(import_kw * self.day["carbon_g_per_kwh"])
                / 1000
            ),
            "energy_balance_max_kw": float(np.max(np.abs(np.vstack(residuals)))),
            "socp_gap_max": float(np.max(np.abs(np.vstack(gaps)))),
            "relaxation_socp_gap_max": float(np.max(np.abs(np.vstack(relaxation_gaps)))),
            "power_flow_iterations_max": int(recovery_iterations),
            "line_loss_kwh": float(np.sum(line_loss_kw) * self.dt),
            "conversion_loss_kwh": float(np.sum(conversion_loss_kw) * self.dt),
            "pv_curtailment_kwh": float(np.sum(self.pv_available - pv) * self.dt),
            "peak_grid_import_kw": float(np.max(import_kw)),
            "ramp_kw": float(np.max(np.abs(np.diff(import_kw)))),
            "comfort_violation_c_h": float(np.sum(temp_violation) * self.dt),
            "ev_energy_shortfall_kwh": float(
                abs(self.ev_target_kwh - eff["ev"] * np.sum(ev) * self.dt)
            ),
            "task_energy_shortfall_kwh": float(
                abs(np.sum(self.task_base - task) * self.dt)
            ),
            "voltage_violation_pu": float(np.max(voltage_violation)),
            "terminal_soc_error_kwh": float(
                abs(
                    values["ess_energy_kwh"][-1]
                    - c["ess"]["capacity_kwh"] * c["ess"]["soc_initial"]
                )
            ),
            "simultaneous_charge_discharge_kw": float(np.max(np.minimum(charge, discharge))),
            "simultaneous_import_export_kw": float(np.max(np.minimum(import_kw, export_kw))),
            "branch_capacity_violation_kw": float(
                max(
                    np.max(
                        np.maximum(
                            np.abs(p[e]),
                            np.abs(p[e] - edge[2] * ell[e]),
                        )
                        * self.sbase
                        - edge[3]
                    )
                    for e, edge in enumerate(self.edges)
                ).clip(min=0)
            ),
            "negative_price_export_kw": float(
                np.max(export_kw[self.day["price_gbp_per_kwh"] < 0])
                if np.any(self.day["price_gbp_per_kwh"] < 0)
                else 0.0
            ),
            "node_demand_kw": demand,
            "branch_loss_kw": line_loss_kw,
            **values,
        }
