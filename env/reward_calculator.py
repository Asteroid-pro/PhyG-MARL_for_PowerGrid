"""
电网防御 Reward 计算器：将生存、负荷保持、电压安全、线路过载和控制代价合成为单步奖励。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RewardCalculator:
    def __init__(
        self,
        alive_weight: float = 0.2,
        load_weight: float = 0.8,
        voltage_deviation_weight: float = 2.0,
        low_voltage_weight: float = 20.0,
        high_voltage_weight: float = 30.0,
        low_voltage_rate_weight: float = 0.3,
        high_voltage_rate_weight: float = 0.5,
        load_shedding_cost_weight: float = 0.4,
        generator_support_cost_weight: float = 0.05,
        line_overload_weight: float = 0.5,
    ):
        self.alive_weight = alive_weight
        self.load_weight = load_weight
        self.voltage_deviation_weight = voltage_deviation_weight
        self.low_voltage_weight = low_voltage_weight
        self.high_voltage_weight = high_voltage_weight
        self.low_voltage_rate_weight = low_voltage_rate_weight
        self.high_voltage_rate_weight = high_voltage_rate_weight
        self.load_shedding_cost_weight = load_shedding_cost_weight
        self.generator_support_cost_weight = generator_support_cost_weight
        self.line_overload_weight = line_overload_weight

    def calculate(
        self,
        pf_result: Any,
        attack_info: Any = None,
        actions=None,
        termination_reason: str | None = None,
    ) -> tuple[float, dict[str, float]]:
        terminal_penalties = {
            "powerflow_not_converged": -10.0,
            "invalid_voltage": -10.0,
            "voltage_collapse": -15.0,
        }
        terminal_penalty = terminal_penalties.get(termination_reason, 0.0)
        if pf_result is None or not isinstance(pf_result, dict):
            components = self._empty_components(terminal_penalty or -5.0)
            return components["total_reward"], components

        voltage = np.asarray(pf_result.get("voltage_pu", []), dtype=float)
        if voltage.size == 0 or not np.all(np.isfinite(voltage)):
            components = self._empty_components(terminal_penalty or -5.0)
            return components["total_reward"], components

        load_ratio = float(np.clip(pf_result.get("load_ratio", 1.0), 0.0, 1.0))
        low_excess = np.maximum(0.95 - voltage, 0.0)
        high_excess = np.maximum(voltage - 1.05, 0.0)
        voltage_penalty = (
            self.voltage_deviation_weight * float(np.mean((voltage - 1.0) ** 2))
            + self.low_voltage_weight * float(np.mean(low_excess ** 2))
            + self.high_voltage_weight * float(np.mean(high_excess ** 2))
            + self.low_voltage_rate_weight * float(np.mean(voltage < 0.95))
            + self.high_voltage_rate_weight * float(np.mean(voltage > 1.05))
        )
        voltage_deltas = np.concatenate([
            np.asarray(pf_result.get("gen_voltage_deltas", []), dtype=float),
            np.asarray(pf_result.get("ext_grid_voltage_deltas", []), dtype=float),
        ])
        support_intensity = float(np.mean(np.abs(voltage_deltas) / 0.02)) if voltage_deltas.size else 0.0
        components = {
            "alive_reward": self.alive_weight,
            "load_reward": self.load_weight * load_ratio,
            "voltage_penalty": voltage_penalty,
            "line_overload_penalty": self.line_overload_weight * float(
                pf_result.get("line_overload_excess_mean", 0.0)
            ),
            "load_shedding_cost": self.load_shedding_cost_weight * (1.0 - load_ratio),
            "generator_support_cost": self.generator_support_cost_weight * support_intensity,
            "terminal_penalty": terminal_penalty,
        }
        components["total_reward"] = float(
            components["alive_reward"] + components["load_reward"]
            - components["voltage_penalty"] - components["line_overload_penalty"]
            - components["load_shedding_cost"]
            - components["generator_support_cost"] + components["terminal_penalty"]
        )
        return components["total_reward"], components

    @staticmethod
    def _empty_components(terminal_penalty: float) -> dict[str, float]:
        return {
            "alive_reward": 0.0,
            "load_reward": 0.0,
            "voltage_penalty": 0.0,
            "line_overload_penalty": 0.0,
            "load_shedding_cost": 0.0,
            "generator_support_cost": 0.0,
            "line_overload_penalty": 0.0,
            "terminal_penalty": float(terminal_penalty),
            "total_reward": float(terminal_penalty),
        }


if __name__ == "__main__":
    reward, components = RewardCalculator().calculate(
        {"voltage_pu": np.ones(14), "load_ratio": 0.95}, actions=[1, 0, 2]
    )
    assert isinstance(reward, float) and components["load_reward"] > 0
    print("reward_calculator ok", reward, components)
