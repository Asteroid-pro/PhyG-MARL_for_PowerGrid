"""
IEEE14 多智能体电网攻防环境：连接攻击模拟、拓扑管理、潮流求解、局部观测和 Reward 闭环。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from .attack_simulator import AttackSimulator
from .observation_builder import ObservationBuilder
from .powerflow_solver import PowerFlowSolver
from .reward_calculator import RewardCalculator
from .termination import Termination
from .topology_manager import TopologyManager


class CPGridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_bus: int = 14,
        max_steps: int = 200,
        attack_probability: float = 0.35,
        attack_weights: dict[str, float] | None = None,
        max_breaker_attacks: int = 6,
        exclude_islanding_breakers: bool = True,
        breaker_transient: bool = True,
        breaker_duration_range: tuple[int, int] = (10, 20),
        max_active_breakers: int = 2,
        load_scaling_by_action: dict[int, float] | None = None,
        generator_voltage_delta_by_action: dict[int, float] | None = None,
        generator_voltage_limits: tuple[float, float] = (0.95, 1.10),
        enforce_q_lims: bool = True,
        load_control_enabled: bool = True,
        generator_control_enabled: bool = True,
    ):
        super().__init__()
        self.n_bus = n_bus
        self.max_steps = max_steps
        self.max_active_breakers = max_active_breakers
        self.topology_manager = TopologyManager(n_bus)
        self.powerflow_solver = PowerFlowSolver(
            load_scaling_by_action=load_scaling_by_action,
            generator_voltage_delta_by_action=generator_voltage_delta_by_action,
            generator_voltage_limits=generator_voltage_limits,
            enforce_q_lims=enforce_q_lims,
            load_control_enabled=load_control_enabled,
            generator_control_enabled=generator_control_enabled,
        )
        self.attack_simulator = AttackSimulator(
            self.topology_manager,
            attack_probability=attack_probability,
            attack_weights=attack_weights,
            max_breaker_attacks=max_breaker_attacks,
            exclude_islanding_breakers=exclude_islanding_breakers,
            breaker_transient=breaker_transient,
            breaker_duration_range=breaker_duration_range,
            max_active_breakers=max_active_breakers,
        )
        self.observation_builder = ObservationBuilder(
            load_buses=set(self.powerflow_solver.bus_load_indices),
            generator_buses=self.powerflow_solver.generator_buses,
            ext_grid_buses=self.powerflow_solver.slack_buses,
        )
        self.reward_calculator = RewardCalculator()
        self.termination = Termination(max_steps=max_steps)
        self.step_count = 0
        self.current_graph_data = None
        self.current_injection = None
        self.current_obs = None
        self.previous_actions = np.zeros(self.n_bus, dtype=int)
        self.last_attack_info: dict[str, Any] = {"type": "none", "success": False, "targets": []}

        sample_obs_dim = 21
        self.action_space = spaces.MultiDiscrete(np.full(self.n_bus, 3, dtype=np.int64))
        self.observation_space = spaces.Dict(
            {
                f"agent_{i}": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(sample_obs_dim,), dtype=np.float32
                )
                for i in range(self.n_bus)
            }
        )

    def _disconnected_lines(self) -> list[int]:
        return self.topology_manager.get_disconnected_lines()

    def _zero_obs(self) -> dict[str, torch.Tensor]:
        obs_dim = self.observation_space["agent_0"].shape[0]
        zero = torch.zeros(obs_dim, dtype=torch.float32)
        return {f"agent_{i}": zero.clone() for i in range(self.n_bus)}

    def _update_graph_state(self, pf_result) -> dict[str, torch.Tensor]:
        if pf_result is None:
            self.current_graph_data = None
            self.current_injection = None
            self.current_obs = self._zero_obs()
            return self.current_obs
        node_feats, edge_feats, edge_index = self.observation_builder.build(
            self.topology_manager, pf_result
        )
        self.current_graph_data = (node_feats, edge_feats, edge_index)
        self.current_injection = torch.as_tensor(
            pf_result.get("bus_p_injection", np.zeros(self.n_bus)), dtype=torch.float32
        )
        self.current_obs = self.observation_builder.build_agent_observations(
            self.topology_manager,
            node_feats,
            edge_feats,
            edge_index,
            self.previous_actions,
            self.attack_simulator.get_breaker_state(),
            self.max_active_breakers,
        )
        return self.current_obs

    def get_graph_data(self):
        if self.current_graph_data is None:
            return None, None, None, None
        node_feats, edge_feats, edge_index = self.current_graph_data
        return node_feats, edge_feats, edge_index, self.current_injection

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.attack_simulator.seed(seed)
        self.attack_simulator.reset_episode()
        self.step_count = 0
        self.previous_actions.fill(0)
        self.last_attack_info = {"type": "none", "success": False, "targets": []}
        for idx in range(len(self.topology_manager.lines)):
            self.topology_manager.restore_edge(idx)
        pf_result = self.powerflow_solver.solve(
            disconnected_lines=self._disconnected_lines(), defense_actions=self.previous_actions
        )
        obs = self._update_graph_state(pf_result)
        return obs, {"attack_info": self.last_attack_info}

    def step(self, actions):
        action_array = np.asarray(actions, dtype=int)
        if action_array.shape != (self.n_bus,):
            raise ValueError(f"Expected joint action shape ({self.n_bus},), got {action_array.shape}")
        if np.any((action_array < 0) | (action_array > 2)):
            raise ValueError("Defense actions must be in {0, 1, 2}")

        restored_lines = self.attack_simulator.advance_breakers()
        attack_info = self.attack_simulator.generate_attack()
        attack_info = self.attack_simulator.apply_physical_attack(attack_info)
        self.attack_simulator.record_breaker_exposure()
        pf_result = self.powerflow_solver.solve(
            disconnected_lines=self._disconnected_lines(), defense_actions=action_array
        )
        self.step_count += 1

        if pf_result is None:
            self._update_graph_state(None)
            terminated, truncated, reason = self.termination.check(None, self.step_count)
            attack_counts = self.attack_simulator.get_episode_counts()
            reward, reward_components = self.reward_calculator.calculate(
                None, attack_info, action_array, termination_reason=reason
            )
            return self.current_obs, reward, terminated, truncated, {
                "attack_info": attack_info,
                "reward_components": reward_components,
                "termination_reason": reason,
                "attack_counts": attack_counts,
                "restored_line_ids": restored_lines,
                "breaker_state": self.attack_simulator.get_breaker_state(),
                "breaker_active": len(self.attack_simulator.active_breakers) > 0,
                "powerflow_metrics": None,
                "disconnected_line_count": len(self._disconnected_lines()),
            }

        self.previous_actions = action_array.copy()
        clean_obs = self._update_graph_state(pf_result)
        attacked_obs, attack_info = self.attack_simulator.apply_observation_attack(
            clean_obs, attack_info
        )
        terminated, truncated, reason = self.termination.check(pf_result, self.step_count)
        reward, reward_components = self.reward_calculator.calculate(
            pf_result, attack_info, action_array, termination_reason=reason
        )
        self.last_attack_info = attack_info
        self.current_obs = attacked_obs
        metric_names = (
            "load_ratio", "min_voltage", "max_voltage", "mean_voltage",
            "low_voltage_rate", "high_voltage_rate", "voltage_violation_rate",
            "max_line_loading_percent", "line_overload_rate", "line_overload_excess_mean",
            "max_line_loading_percent", "line_overload_rate", "line_overload_excess_mean",
            "gen_q_total_mvar", "gen_q_abs_total_mvar",
            "ext_grid_q_total_mvar", "ext_grid_q_abs_total_mvar",
            "all_source_q_total_mvar", "all_source_q_abs_total_mvar",
            "gen_q_limit_hit_count", "gen_q_limit_hit_rate",
            "mean_voltage_support_delta",
        )
        powerflow_metrics = {name: pf_result[name] for name in metric_names}
        breaker_state = self.attack_simulator.get_breaker_state()
        return attacked_obs, reward, terminated, truncated, {
            "attack_info": attack_info,
            "termination_reason": reason,
            "reward_components": reward_components,
            "load_ratio": pf_result["load_ratio"],
            "powerflow_metrics": powerflow_metrics,
            "attack_counts": self.attack_simulator.get_episode_counts(),
            "restored_line_ids": restored_lines,
            "breaker_state": breaker_state,
            "breaker_active": breaker_state["active_breaker_count"] > 0,
            "disconnected_line_count": len(self._disconnected_lines()),
        }


if __name__ == "__main__":
    env = CPGridEnv()
    obs, info = env.reset(seed=42)
    out = env.step(np.zeros(14, dtype=int))
    assert len(obs) == 14 and len(out) == 5
    print("cpgrid_env ok", out[1], out[2], out[3])
