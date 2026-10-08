"""
网络—物理攻击模拟器：生成 FDI、DoS 和暂态 Breaker 攻击，并管理攻击持续状态。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class AttackSimulator:
    ATTACK_TYPES = ("none", "fdi", "breaker", "dos")

    def __init__(
        self,
        topology_manager: Any | None = None,
        attack_probability: float = 0.35,
        attack_weights: dict[str, float] | None = None,
        max_breaker_attacks: int = 6,
        exclude_islanding_breakers: bool = True,
        breaker_transient: bool = True,
        breaker_duration_range: tuple[int, int] = (10, 20),
        max_active_breakers: int = 2,
        fdi_scale: float = 0.05,
        dos_probability: float = 0.8,
        observation_attack_duration_range: tuple[int, int] = (5, 10),
        observation_attack_target_count: int = 2,
    ):
        self.topology_manager = topology_manager
        self.attack_probability = attack_probability
        self.attack_weights = attack_weights or {"fdi": 0.45, "dos": 0.45, "breaker": 0.10}
        if set(self.attack_weights) != {"fdi", "dos", "breaker"}:
            raise ValueError("attack_weights must define fdi, dos, and breaker")
        if any(weight < 0 for weight in self.attack_weights.values()) or sum(self.attack_weights.values()) <= 0:
            raise ValueError("attack_weights must be non-negative with a positive sum")
        duration_min, duration_max = breaker_duration_range
        if duration_min < 1 or duration_max < duration_min:
            raise ValueError("breaker_duration_range must contain positive ordered values")
        if max_active_breakers < 1:
            raise ValueError("max_active_breakers must be at least 1")
        self.max_breaker_attacks = max_breaker_attacks
        self.exclude_islanding_breakers = exclude_islanding_breakers
        self.breaker_transient = breaker_transient
        self.breaker_duration_range = (int(duration_min), int(duration_max))
        self.max_active_breakers = max_active_breakers
        observation_min, observation_max = observation_attack_duration_range
        if observation_min < 1 or observation_max < observation_min:
            raise ValueError("observation_attack_duration_range must be positive and ordered")
        self.fdi_scale = fdi_scale
        self.dos_probability = dos_probability
        self.observation_attack_duration_range = (int(observation_min), int(observation_max))
        self.observation_attack_target_count = max(1, min(int(observation_attack_target_count), 14))
        self.attack_success_count = 0
        self.rng = np.random.default_rng()
        self.reset_episode()

    def seed(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset_episode(self) -> None:
        self.episode_counts = {"fdi": 0, "dos": 0, "breaker": 0}
        self.active_breakers: dict[int, int] = {}
        self.active_observation_attacks: dict[str, dict[str, Any]] = {}
        self.last_clean_observations: dict[str, torch.Tensor] = {}
        self.observation_attack_active_steps = {"fdi": 0, "dos": 0}
        self.breaker_durations: list[int] = []
        self.restored_line_count = 0
        self.max_concurrent_breakers = 0
        self.breaker_active_steps = 0
        self.steps_under_breaker = 0

    def attach_topology_manager(self, topology_manager: Any) -> None:
        self.topology_manager = topology_manager

    def get_episode_counts(self) -> dict[str, int]:
        return dict(self.episode_counts)

    def get_breaker_state(self) -> dict[str, Any]:
        return {
            "breaker_transient": self.breaker_transient,
            "active_breaker_count": len(self.active_breakers),
            "active_breaker_lines": sorted(self.active_breakers),
            "active_breaker_remaining_steps": dict(self.active_breakers),
            "restored_line_count": self.restored_line_count,
            "max_concurrent_breakers": self.max_concurrent_breakers,
            "breaker_active_steps": self.breaker_active_steps,
            "steps_under_breaker": self.steps_under_breaker,
            "mean_breaker_duration": (
                float(np.mean(self.breaker_durations)) if self.breaker_durations else 0.0
            ),
        }

    def advance_breakers(self) -> list[int]:
        if not self.breaker_transient:
            return []
        restored_lines = []
        for line_id in list(self.active_breakers):
            remaining = self.active_breakers[line_id] - 1
            if remaining <= 0:
                if self.topology_manager is not None:
                    self.topology_manager.restore_edge(line_id)
                del self.active_breakers[line_id]
                restored_lines.append(line_id)
                self.restored_line_count += 1
            else:
                self.active_breakers[line_id] = remaining
        return restored_lines

    def record_breaker_exposure(self) -> None:
        active_count = len(self.active_breakers)
        self.breaker_active_steps += active_count
        if active_count > 0:
            self.steps_under_breaker += 1
        self.max_concurrent_breakers = max(self.max_concurrent_breakers, active_count)

    def _breaker_available(self) -> bool:
        if self.episode_counts["breaker"] >= self.max_breaker_attacks:
            return False
        if self.breaker_transient and len(self.active_breakers) >= self.max_active_breakers:
            return False
        return True

    def _available_attack_distribution(self) -> tuple[list[str], np.ndarray]:
        weights = dict(self.attack_weights)
        if not self._breaker_available():
            weights["breaker"] = 0.0
        names = [name for name, weight in weights.items() if weight > 0]
        if not names:
            return [], np.empty(0, dtype=float)
        probabilities = np.asarray([weights[name] for name in names], dtype=float)
        probabilities /= probabilities.sum()
        return names, probabilities

    def generate_attack(self) -> dict[str, Any]:
        if self.rng.random() >= self.attack_probability:
            return {"type": "none", "success": False, "targets": []}
        names, probabilities = self._available_attack_distribution()
        if not names:
            return {
                "type": "none",
                "success": False,
                "targets": [],
                "blocked_reason": "no_available_attack_type",
            }
        attack_type = str(self.rng.choice(names, p=probabilities))
        target_count = 1 if attack_type == "breaker" else self.observation_attack_target_count
        targets = self.rng.choice(14, size=target_count, replace=False).astype(int).tolist()
        return {"type": attack_type, "success": True, "targets": targets}

    def apply_physical_attack(self, attack_info: dict[str, Any]) -> dict[str, Any]:
        if attack_info["type"] != "breaker" or self.topology_manager is None:
            return attack_info
        if self.episode_counts["breaker"] >= self.max_breaker_attacks:
            attack_info.update({"success": False, "blocked_reason": "breaker_limit_reached"})
            return attack_info
        if self.breaker_transient and len(self.active_breakers) >= self.max_active_breakers:
            attack_info.update({"success": False, "blocked_reason": "active_breaker_limit_reached"})
            return attack_info
        if self.exclude_islanding_breakers:
            active = self.topology_manager.get_non_islanding_active_lines()
        else:
            active = [idx for idx, line in enumerate(self.topology_manager.lines) if line.active]
        if not active:
            attack_info.update({"success": False, "blocked_reason": "no_safe_breaker_candidate"})
            return attack_info

        line_id = int(self.rng.choice(active))
        self.topology_manager.remove_edge(line_id)
        attack_info["line_id"] = line_id
        attack_info["targets"] = [
            self.topology_manager.lines[line_id].src,
            self.topology_manager.lines[line_id].dst,
        ]
        if self.breaker_transient:
            duration = int(self.rng.integers(self.breaker_duration_range[0], self.breaker_duration_range[1] + 1))
            self.active_breakers[line_id] = duration
            self.breaker_durations.append(duration)
            self.max_concurrent_breakers = max(self.max_concurrent_breakers, len(self.active_breakers))
            attack_info["duration"] = duration
            attack_info["remaining_steps"] = duration
        self.episode_counts["breaker"] += 1
        self.attack_success_count += 1
        return attack_info

    def apply_observation_attack(
        self, obs: dict[str, torch.Tensor], attack_info: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        modified = {key: value.clone() for key, value in obs.items()}
        attack_type = attack_info["type"]
        targets = attack_info.get("targets", [])
        if attack_type in {"fdi", "dos"}:
            successful_targets = targets if attack_type == "fdi" else [
                target for target in targets if self.rng.random() < self.dos_probability
            ]
            if successful_targets:
                duration = int(self.rng.integers(
                    self.observation_attack_duration_range[0],
                    self.observation_attack_duration_range[1] + 1,
                ))
                self.active_observation_attacks[attack_type] = {
                    "targets": successful_targets,
                    "remaining_steps": duration,
                }
                self.episode_counts[attack_type] += 1
                self.attack_success_count += 1
                attack_info.update({"success": True, "targets": successful_targets, "duration": duration})
            else:
                attack_info.update({"success": False, "targets": [], "dropped": False})

        for active_type in list(self.active_observation_attacks):
            state = self.active_observation_attacks[active_type]
            self.observation_attack_active_steps[active_type] += 1
            for target in state["targets"]:
                key = f"agent_{target}"
                if key not in modified:
                    continue
                if active_type == "fdi":
                    voltage_indices = [0, 4, 8]
                    for index in voltage_indices:
                        if index < modified[key].numel():
                            direction = -1.0 if float(modified[key][index]) > 1.0 else 1.0
                            modified[key][index] += direction * self.fdi_scale
                elif key in self.last_clean_observations:
                    modified[key] = self.last_clean_observations[key].clone()
                else:
                    modified[key].zero_()
            state["remaining_steps"] -= 1
            if state["remaining_steps"] <= 0:
                del self.active_observation_attacks[active_type]

        attack_info["active_observation_attacks"] = {
            key: dict(value) for key, value in self.active_observation_attacks.items()
        }
        self.last_clean_observations = {key: value.clone() for key, value in obs.items()}
        return modified, attack_info

    def step(self, attack_type: str, obs: Any) -> tuple[Any, Any]:
        attack_info = {"type": attack_type, "success": attack_type != "none", "targets": [0]}
        attack_info = self.apply_physical_attack(attack_info)
        if isinstance(obs, dict):
            return self.apply_observation_attack(obs, attack_info)
        if torch.is_tensor(obs):
            wrapped = {"agent_0": obs}
            modified, attack_info = self.apply_observation_attack(wrapped, attack_info)
            return modified["agent_0"], attack_info
        return obs, attack_info


if __name__ == "__main__":
    attack = AttackSimulator(breaker_duration_range=(3, 3))
    attack.seed(42)
    observation = {"agent_0": torch.ones(4)}
    changed, info = attack.apply_observation_attack(
        observation, {"type": "fdi", "success": True, "targets": [0]}
    )
    assert changed["agent_0"].shape == observation["agent_0"].shape
    print("attack_simulator ok", info, attack.get_episode_counts())
