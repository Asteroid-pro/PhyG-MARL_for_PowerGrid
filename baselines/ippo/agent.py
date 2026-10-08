from __future__ import annotations

import torch
from torch.distributions import Categorical

from baselines.common.agents import LocalPolicy
from baselines.common.base_agent import BaselineAgent
from baselines.common.env_adapter import observation_tensor


class IPPOAgent(BaselineAgent):
    def __init__(self, obs_dim: int, device: torch.device, actor_lr: float = 1e-4, critic_lr: float = 1e-4):
        self.device = device
        self.actor = LocalPolicy(obs_dim, 128).to(device)
        self.critic = LocalPolicy(obs_dim, 128).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def act(self, observations, global_state, deterministic=False):
        obs = observation_tensor(observations, self.device)
        logits, _ = self.actor(obs)
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        values = self.critic(obs)[0].mean()
        return actions, distribution.log_prob(actions).sum(), values

    def update(self):
        return {"actor_loss": 0.0, "critic_loss": 0.0}

    def checkpoint(self):
        return {"actor_state_dict": self.actor.state_dict(), "critic_state_dict": self.critic.state_dict(), "algorithm": "ippo"}
