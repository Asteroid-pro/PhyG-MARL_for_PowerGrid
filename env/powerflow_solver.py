from __future__ import annotations

import copy

import numpy as np
import pandapower as pp
import pandapower.networks as pn


class PowerFlowSolver:
    def __init__(
        self,
        load_scaling_by_action: dict[int, float] | None = None,
        generator_voltage_delta_by_action: dict[int, float] | None = None,
        generator_voltage_limits: tuple[float, float] = (0.95, 1.10),
        enforce_q_lims: bool = True,
        load_control_enabled: bool = True,
        generator_control_enabled: bool = True,
    ):
        self.load_scaling_by_action = load_scaling_by_action or {0: 1.0, 1: 0.95, 2: 0.90}
        self.generator_voltage_delta_by_action = generator_voltage_delta_by_action or {
            0: 0.0,
            1: 0.01,
            2: 0.02,
        }
        if set(self.load_scaling_by_action) != {0, 1, 2}:
            raise ValueError("load_scaling_by_action must define actions 0, 1, and 2")
        if set(self.generator_voltage_delta_by_action) != {0, 1, 2}:
            raise ValueError("generator_voltage_delta_by_action must define actions 0, 1, and 2")
        if any(not 0.0 <= float(scale) <= 1.0 for scale in self.load_scaling_by_action.values()):
            raise ValueError("Load scaling values must be between 0 and 1")
        voltage_min, voltage_max = generator_voltage_limits
        if voltage_min <= 0 or voltage_max < voltage_min:
            raise ValueError("generator_voltage_limits must be positive ordered values")
        self.generator_voltage_limits = (float(voltage_min), float(voltage_max))
        self.enforce_q_lims = enforce_q_lims
        self.load_control_enabled = load_control_enabled
        self.generator_control_enabled = generator_control_enabled
        self.template_net = pn.case14()
        self.bus_load_indices = {
            int(bus): group.index.to_list()
            for bus, group in self.template_net.load.groupby("bus")
        }
        self.base_load_p = self.template_net.load["p_mw"].to_numpy(copy=True)
        self.base_load_q = self.template_net.load["q_mvar"].to_numpy(copy=True)
        self.base_total_load = float(self.base_load_p.sum())
        self.generator_buses = set(self.template_net.gen["bus"].astype(int).tolist())
        self.slack_buses = set(self.template_net.ext_grid["bus"].astype(int).tolist())
        self.support_buses = self.generator_buses | self.slack_buses

    def _validate_actions(self, defense_actions) -> np.ndarray:
        actions = np.asarray(defense_actions if defense_actions is not None else np.zeros(14), dtype=int)
        if actions.shape != (14,):
            raise ValueError(f"Expected 14 defense actions, got shape {actions.shape}")
        if np.any((actions < 0) | (actions > 2)):
            raise ValueError("Defense actions must be in {0, 1, 2}")
        return actions

    def _apply_defense_actions(self, net, actions: np.ndarray) -> np.ndarray:
        load_scaling = np.ones(len(net.load), dtype=np.float64)
        if not self.load_control_enabled:
            net.load.loc[:, "scaling"] = load_scaling
            return load_scaling
        for bus, action in enumerate(actions):
            for load_idx in self.bus_load_indices.get(bus, []):
                load_scaling[load_idx] = self.load_scaling_by_action[int(action)]
        net.load.loc[:, "scaling"] = load_scaling
        return load_scaling

    def _apply_generator_support(self, net, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        voltage_min, voltage_max = self.generator_voltage_limits
        gen_deltas = np.zeros(len(net.gen), dtype=np.float64)
        ext_grid_deltas = np.zeros(len(net.ext_grid), dtype=np.float64)
        if not self.generator_control_enabled:
            return gen_deltas, ext_grid_deltas
        for position, (idx, row) in enumerate(net.gen.iterrows()):
            bus = int(row["bus"])
            delta = float(self.generator_voltage_delta_by_action[int(actions[bus])])
            base_voltage = float(self.template_net.gen.loc[idx, "vm_pu"])
            net.gen.loc[idx, "vm_pu"] = np.clip(base_voltage + delta, voltage_min, voltage_max)
            gen_deltas[position] = float(net.gen.loc[idx, "vm_pu"] - base_voltage)
        for position, (idx, row) in enumerate(net.ext_grid.iterrows()):
            bus = int(row["bus"])
            delta = float(self.generator_voltage_delta_by_action[int(actions[bus])])
            base_voltage = float(self.template_net.ext_grid.loc[idx, "vm_pu"])
            net.ext_grid.loc[idx, "vm_pu"] = np.clip(base_voltage + delta, voltage_min, voltage_max)
            ext_grid_deltas[position] = float(net.ext_grid.loc[idx, "vm_pu"] - base_voltage)
        return gen_deltas, ext_grid_deltas

    def solve(self, disconnected_lines=None, defense_actions=None):
        net = copy.deepcopy(self.template_net)
        if disconnected_lines:
            for line_id in disconnected_lines:
                line_id = int(line_id)
                if 0 <= line_id < len(net.line):
                    net.line.loc[line_id, "in_service"] = False
        actions = self._validate_actions(defense_actions)
        load_scaling = self._apply_defense_actions(net, actions)
        gen_voltage_deltas, ext_grid_voltage_deltas = self._apply_generator_support(net, actions)
        try:
            pp.runpp(net, enforce_q_lims=self.enforce_q_lims)
        except (pp.powerflow.LoadflowNotConverged, UserWarning):
            return None

        served_load = float(np.sum(self.base_load_p * load_scaling))
        bus_p_load = np.zeros(14, dtype=np.float64)
        bus_q_load = np.zeros(14, dtype=np.float64)
        for load_idx, row in net.load.iterrows():
            bus = int(row["bus"])
            scale = float(row.get("scaling", 1.0))
            bus_p_load[bus] += float(row["p_mw"]) * scale
            bus_q_load[bus] += float(row["q_mvar"]) * scale

        bus_p_gen = np.zeros(14, dtype=np.float64)
        if len(net.ext_grid):
            for idx, row in net.ext_grid.iterrows():
                bus_p_gen[int(row["bus"])] += float(net.res_ext_grid.loc[idx, "p_mw"])
        if len(net.gen):
            for idx, row in net.gen.iterrows():
                bus_p_gen[int(row["bus"])] += float(net.res_gen.loc[idx, "p_mw"])

        gen_q = net.res_gen["q_mvar"].to_numpy(copy=True) if len(net.gen) else np.empty(0)
        ext_grid_q = net.res_ext_grid["q_mvar"].to_numpy(copy=True) if len(net.ext_grid) else np.empty(0)
        voltage = net.res_bus["vm_pu"].to_numpy(copy=True)
        low_voltage = voltage < 0.95
        high_voltage = voltage > 1.05
        voltage_violation = low_voltage | high_voltage
        line_loading = np.nan_to_num(
            net.res_line["loading_percent"].to_numpy(copy=True), nan=0.0, posinf=1e6
        )
        overloaded_lines = line_loading > 100.0
        q_tolerance = 1e-3
        if len(net.gen):
            min_q = net.gen["min_q_mvar"].to_numpy(dtype=float, copy=True)
            max_q = net.gen["max_q_mvar"].to_numpy(dtype=float, copy=True)
            min_hits = np.isfinite(min_q) & (gen_q <= min_q + q_tolerance)
            max_hits = np.isfinite(max_q) & (gen_q >= max_q - q_tolerance)
        else:
            min_hits = np.empty(0, dtype=bool)
            max_hits = np.empty(0, dtype=bool)
        q_limit_hits = min_hits | max_hits

        return {
            "voltage_pu": voltage,
            "angle_deg": net.res_bus["va_degree"].values,
            "line_flow_p": net.res_line["p_from_mw"].values,
            "line_flow_q": net.res_line["q_from_mvar"].values,
            "bus_p_load": bus_p_load,
            "bus_q_load": bus_q_load,
            "bus_p_injection": bus_p_gen - bus_p_load,
            "served_load": served_load,
            "base_total_load": self.base_total_load,
            "load_ratio": served_load / max(self.base_total_load, 1e-8),
            "gen_q_mvar": gen_q,
            "ext_grid_q_mvar": ext_grid_q,
            "gen_q_total_mvar": float(gen_q.sum()),
            "gen_q_abs_total_mvar": float(np.abs(gen_q).sum()),
            "ext_grid_q_total_mvar": float(ext_grid_q.sum()),
            "ext_grid_q_abs_total_mvar": float(np.abs(ext_grid_q).sum()),
            "all_source_q_total_mvar": float(gen_q.sum() + ext_grid_q.sum()),
            "all_source_q_abs_total_mvar": float(np.abs(gen_q).sum() + np.abs(ext_grid_q).sum()),
            "total_gen_q_mvar": float(gen_q.sum() + ext_grid_q.sum()),
            "total_abs_gen_q_mvar": float(np.abs(gen_q).sum() + np.abs(ext_grid_q).sum()),
            "gen_q_min_limit_hit_count": int(min_hits.sum()),
            "gen_q_max_limit_hit_count": int(max_hits.sum()),
            "gen_q_limit_hit_count": int(q_limit_hits.sum()),
            "gen_q_limit_hit_rate": float(q_limit_hits.mean()) if len(q_limit_hits) else 0.0,
            "gen_voltage_setpoints": net.gen["vm_pu"].to_numpy(copy=True),
            "ext_grid_voltage_setpoints": net.ext_grid["vm_pu"].to_numpy(copy=True),
            "gen_voltage_deltas": gen_voltage_deltas,
            "ext_grid_voltage_deltas": ext_grid_voltage_deltas,
            "mean_voltage_support_delta": float(
                np.mean(np.concatenate([gen_voltage_deltas, ext_grid_voltage_deltas]))
            ),
            "min_voltage": float(np.min(voltage)),
            "max_voltage": float(np.max(voltage)),
            "mean_voltage": float(np.mean(voltage)),
            "low_voltage_bus_count": int(low_voltage.sum()),
            "high_voltage_bus_count": int(high_voltage.sum()),
            "low_voltage_rate": float(np.mean(low_voltage)),
            "high_voltage_rate": float(np.mean(high_voltage)),
            "voltage_violation_count": int(voltage_violation.sum()),
            "voltage_violation_rate": float(np.mean(voltage_violation)),
            "line_loading_percent": line_loading,
            "max_line_loading_percent": float(np.max(line_loading)) if line_loading.size else 0.0,
            "line_overload_rate": float(np.mean(overloaded_lines)) if line_loading.size else 0.0,
            "line_overload_excess_mean": float(
                np.mean(np.maximum(line_loading / 100.0 - 1.0, 0.0) ** 2)
            ) if line_loading.size else 0.0,
        }


if __name__ == "__main__":
    solver = PowerFlowSolver()
    result = solver.solve(disconnected_lines=[], defense_actions=np.zeros(14, dtype=int))
    assert result is None or len(result["voltage_pu"]) == 14
    print("powerflow_solver ok", None if result is None else result["voltage_pu"].shape)
