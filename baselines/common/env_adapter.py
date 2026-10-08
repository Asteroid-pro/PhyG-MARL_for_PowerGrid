from __future__ import annotations

from typing import Any

import numpy as np
import torch

from config import NUM_AGENTS
from env.cpgrid_env import CPGridEnv


def make_env(steps: int) -> CPGridEnv:
    return CPGridEnv(
        max_steps=steps,
        attack_probability=0.35,
        attack_weights={"fdi": 0.45, "dos": 0.45, "breaker": 0.10},
        max_breaker_attacks=6,
        exclude_islanding_breakers=True,
        breaker_transient=True,
        breaker_duration_range=(10, 20),
        max_active_breakers=2,
        load_control_enabled=True,
        generator_control_enabled=True,
        enforce_q_lims=True,
    )


def global_state(env: CPGridEnv, node_feats: torch.Tensor, steps: int) -> torch.Tensor:
    disconnected = torch.zeros(NUM_AGENTS)
    for line_id in env._disconnected_lines():
        line = env.topology_manager.lines[line_id]
        disconnected[line.src] = 1.0
        disconnected[line.dst] = 1.0
    breaker = env.attack_simulator.get_breaker_state()
    return torch.cat((
        node_feats[:, :4].reshape(-1),
        disconnected,
        torch.tensor([breaker.get("active_breaker_count", 0) / 2.0, env.step_count / max(steps, 1)]),
    )).float()


def observation_tensor(observations: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.stack([
        torch.as_tensor(observations[f"agent_{index}"], dtype=torch.float32, device=device)
        for index in range(NUM_AGENTS)
    ])
