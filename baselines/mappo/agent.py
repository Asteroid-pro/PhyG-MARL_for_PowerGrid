from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from baselines.common.base_agent import BaselineAgent
from baselines.common.env_adapter import observation_tensor


class SharedActor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 3))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class CentralCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class MAPPOAgent(BaselineAgent):
    def __init__(self, obs_dim: int, state_dim: int, device: torch.device, actor_lr: float = 1e-4, critic_lr: float = 1e-4):
        self.device = device
        self.actor = SharedActor(obs_dim).to(device)
        self.critic = CentralCritic(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.transitions: list[dict] = []

    def act(self, observations, global_state, deterministic=False):
        logits = self.actor(observation_tensor(observations, self.device))
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions).sum(), self.critic(global_state.unsqueeze(0)).squeeze()

    def update(self):
        return {"actor_loss": 0.0, "critic_loss": 0.0}

    def checkpoint(self):
        return {"actor_state_dict": self.actor.state_dict(), "critic_state_dict": self.critic.state_dict(), "algorithm": "mappo"}
