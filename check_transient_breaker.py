from __future__ import annotations

import numpy as np

from env.attack_simulator import AttackSimulator
from env.cpgrid_env import CPGridEnv
from env.topology_manager import TopologyManager


def test_exact_duration() -> None:
    topology = TopologyManager(14)
    simulator = AttackSimulator(
        topology,
        max_breaker_attacks=1,
        breaker_transient=True,
        breaker_duration_range=(3, 3),
        max_active_breakers=1,
    )
    simulator.seed(42)
    info = simulator.apply_physical_attack({"type": "breaker", "success": True, "targets": []})
    line_id = info["line_id"]
    assert not topology.lines[line_id].active
    simulator.record_breaker_exposure()
    assert simulator.advance_breakers() == []
    simulator.record_breaker_exposure()
    assert simulator.advance_breakers() == []
    simulator.record_breaker_exposure()
    restored = simulator.advance_breakers()
    assert restored == [line_id]
    assert topology.lines[line_id].active
    assert simulator.get_breaker_state()["steps_under_breaker"] == 3


def test_concurrency_and_reset() -> None:
    env = CPGridEnv(
        max_steps=100,
        attack_probability=1.0,
        attack_weights={"fdi": 0.0, "dos": 0.0, "breaker": 1.0},
        max_breaker_attacks=6,
        breaker_transient=True,
        breaker_duration_range=(10, 10),
        max_active_breakers=2,
    )
    env.reset(seed=42)
    maximum_active = 0
    last_info = {}
    for _ in range(30):
        _, _, terminated, truncated, last_info = env.step(np.zeros(14, dtype=int))
        state = last_info["breaker_state"]
        maximum_active = max(maximum_active, state["active_breaker_count"])
        assert state["active_breaker_count"] <= 2
        assert last_info["attack_counts"]["breaker"] <= 6
        if terminated or truncated:
            break
    assert maximum_active == 2
    env.reset(seed=43)
    assert env.attack_simulator.get_breaker_state()["active_breaker_count"] == 0
    assert env.attack_simulator.get_episode_counts()["breaker"] == 0
    assert len(env.topology_manager.get_disconnected_lines()) == 0


def test_permanent_mode() -> None:
    topology = TopologyManager(14)
    simulator = AttackSimulator(
        topology,
        max_breaker_attacks=1,
        breaker_transient=False,
    )
    simulator.seed(42)
    info = simulator.apply_physical_attack({"type": "breaker", "success": True, "targets": []})
    line_id = info["line_id"]
    for _ in range(30):
        assert simulator.advance_breakers() == []
    assert not topology.lines[line_id].active
    assert simulator.get_breaker_state()["active_breaker_count"] == 0


if __name__ == "__main__":
    test_exact_duration()
    test_concurrency_and_reset()
    test_permanent_mode()
    print("Transient and permanent Breaker lifecycle tests passed.")
