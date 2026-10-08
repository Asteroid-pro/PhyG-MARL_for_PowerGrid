from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .topology_manager import IEEE14_LINE_DATA


class ObservationBuilder:
    def __init__(
        self,
        load_buses: set[int] | None = None,
        generator_buses: set[int] | None = None,
        ext_grid_buses: set[int] | None = None,
    ):
        self.load_buses = load_buses or set()
        self.generator_buses = generator_buses or set()
        self.ext_grid_buses = ext_grid_buses or set()

    def build(self, topology: Any, pf_result: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pf_result is None:
            return (
                torch.zeros(14, 4, dtype=torch.float32),
                torch.empty((0, 3), dtype=torch.float32),
                torch.empty((2, 0), dtype=torch.long),
            )

        voltages = np.nan_to_num(np.asarray(pf_result["voltage_pu"]), nan=0.0)
        angles = np.nan_to_num(np.asarray(pf_result["angle_deg"]), nan=0.0)
        p_load = np.nan_to_num(np.asarray(pf_result.get("bus_p_load", np.zeros(14))), nan=0.0)
        q_load = np.nan_to_num(np.asarray(pf_result.get("bus_q_load", np.zeros(14))), nan=0.0)
        node_feats = torch.as_tensor(
            np.stack([voltages, angles, p_load, q_load], axis=-1), dtype=torch.float32
        )

        p_flow = np.nan_to_num(np.asarray(pf_result["line_flow_p"]), nan=0.0)
        q_flow = np.nan_to_num(np.asarray(pf_result["line_flow_q"]), nan=0.0)
        edge_feats: list[list[float]] = []
        edge_index: list[list[int]] = []
        for idx, (src, dst, _, _) in enumerate(IEEE14_LINE_DATA):
            active = topology.lines[idx].active if topology is not None else True
            if not active:
                continue
            p = float(p_flow[idx]) if idx < len(p_flow) else 0.0
            q = float(q_flow[idx]) if idx < len(q_flow) else 0.0
            edge_index.extend(([src, dst], [dst, src]))
            edge_feats.extend(([p, q, 1.0], [-p, -q, 1.0]))

        return (
            torch.nan_to_num(node_feats),
            torch.tensor(edge_feats, dtype=torch.float32),
            torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        )

    def build_agent_observations(
        self,
        topology: Any,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        previous_actions: np.ndarray,
        breaker_state: dict[str, Any] | None = None,
        max_active_breakers: int = 2,
    ) -> dict[str, torch.Tensor]:
        observations: dict[str, torch.Tensor] = {}
        breaker_state = breaker_state or {}
        active_lines = breaker_state.get("active_breaker_lines", [])
        remaining_steps = breaker_state.get("active_breaker_remaining_steps", {})
        active_ratio = float(breaker_state.get("active_breaker_count", 0)) / max(max_active_breakers, 1)
        for bus in range(node_feats.size(0)):
            outgoing = edge_index[0] == bus
            neighbors = edge_index[1, outgoing]
            neighbor_feats = node_feats[neighbors] if neighbors.numel() else torch.zeros((1, 4))
            local_edges = edge_feats[outgoing] if outgoing.any() else torch.zeros((1, 3))
            disconnected = sum(
                1
                for line in topology.lines
                if not line.active and (line.src == bus or line.dst == bus)
            )
            has_load = float(bus in self.load_buses)
            has_generator = float(bus in self.generator_buses)
            is_ext_grid = float(bus in self.ext_grid_buses)
            incident_remaining = [
                float(remaining_steps.get(line_id, remaining_steps.get(str(line_id), 0)))
                for line_id in active_lines
                if topology.lines[int(line_id)].src == bus or topology.lines[int(line_id)].dst == bus
            ]
            incident_remaining_ratio = max(incident_remaining, default=0.0) / 20.0
            features = torch.cat(
                [
                    node_feats[bus],
                    neighbor_feats.mean(dim=0),
                    neighbor_feats[:, 0].min().view(1),
                    local_edges[:, :2].abs().max(dim=0).values,
                    torch.tensor(
                        [
                            float(neighbors.numel()),
                            float(disconnected),
                            float(previous_actions[bus]),
                            1.0,
                            has_load,
                            has_generator,
                            is_ext_grid,
                            float(bool(active_lines)),
                            active_ratio,
                            incident_remaining_ratio,
                        ],
                        dtype=torch.float32,
                    ),
                ]
            )
            observations[f"agent_{bus}"] = torch.nan_to_num(features)
        return observations


if __name__ == "__main__":
    from .powerflow_solver import PowerFlowSolver
    from .topology_manager import TopologyManager

    topology = TopologyManager()
    pf = PowerFlowSolver().solve([], np.zeros(14, dtype=int))
    node, edge, index = ObservationBuilder().build(topology, pf)
    obs = ObservationBuilder(
        load_buses=set(PowerFlowSolver().bus_load_indices),
        generator_buses=PowerFlowSolver().generator_buses,
        ext_grid_buses=PowerFlowSolver().slack_buses,
    ).build_agent_observations(topology, node, edge, index, np.zeros(14))
    assert len(obs) == 14
    print("observation_builder ok", node.shape, edge.shape, obs["agent_0"].shape)
