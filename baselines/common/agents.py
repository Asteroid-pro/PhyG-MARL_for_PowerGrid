from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from baseline_config import BaselineConfig
from baselines.common.env_adapter import observation_tensor


class LocalPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, recurrent: bool = False):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        self.recurrent = recurrent
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True) if recurrent else None
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        encoded = torch.tanh(self.encoder(obs))
        if self.recurrent:
            output, hidden = self.gru(encoded.unsqueeze(1), hidden)
            encoded = output[:, -1]
        return self.head(encoded), hidden


class CentralPolicy(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, agents: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.head = nn.Linear(hidden_dim, agents * 3)
        self.agents = agents

    def forward(self, state: torch.Tensor):
        return self.head(self.encoder(state)).reshape(-1, self.agents, 3)


@dataclass
class Experience:
    observation: dict[str, Any]
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    done: bool
    old_log_prob: torch.Tensor
    value: torch.Tensor


def discounted_returns(experiences: list[Experience], config: BaselineConfig) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = torch.tensor([item.reward * 0.1 for item in experiences], dtype=torch.float32)
    dones = torch.tensor([item.done for item in experiences], dtype=torch.float32)
    values = torch.stack([item.value.detach().cpu() for item in experiences]).reshape(-1)
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0
    for index in reversed(range(len(experiences))):
        delta = rewards[index] + config.gamma * next_value * (1.0 - dones[index]) - values[index]
        gae = delta + config.gamma * config.gae_lambda * (1.0 - dones[index]) * gae
        advantages[index] = gae
        returns[index] = gae + values[index]
        next_value = values[index]
    return returns.to(values.device), advantages.to(values.device)


def ppo_loss(new_log_prob, old_log_prob, advantage, entropy, clip_ratio, entropy_coef):
    ratio = torch.exp(new_log_prob - old_log_prob.detach())
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    policy = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    return policy - entropy_coef * entropy.mean()
