from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Step:
    observations: dict[str, Any]
    global_state: torch.Tensor
    action: torch.Tensor
    reward: float
    done: bool
    value: torch.Tensor
    log_prob: torch.Tensor
    hidden_state: torch.Tensor | None = None


class RolloutBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.steps: list[Step] = []

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)

    def ready(self) -> bool:
        return len(self.steps) >= self.capacity

    def returns_and_advantages(self, gamma: float, gae_lambda: float, bootstrap: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.tensor([step.reward for step in self.steps], dtype=torch.float32)
        dones = torch.tensor([step.done for step in self.steps], dtype=torch.float32)
        values = torch.stack([step.value.detach().cpu() for step in self.steps]).reshape(-1)
        next_values = torch.cat((values[1:], torch.tensor([bootstrap])))
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0)
        for index in range(len(self.steps) - 1, -1, -1):
            delta = rewards[index] + gamma * next_values[index] * (1.0 - dones[index]) - values[index]
            gae = delta + gamma * gae_lambda * (1.0 - dones[index]) * gae
            advantages[index] = gae
        return advantages + values, advantages
