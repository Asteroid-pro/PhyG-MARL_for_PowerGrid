from __future__ import annotations

import torch
from torch.distributions import Categorical

from baselines.common.base_agent import BaselineAgent
from baselines.common.agents import CentralPolicy


class PPOAgent(BaselineAgent):
    def __init__(self, state_dim: int, device: torch.device, agents: int = 14):
        self.device = device
        self.actor = CentralPolicy(state_dim, 128, agents).to(device)
        self.critic = torch.nn.Sequential(torch.nn.Linear(state_dim, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1)).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

    def act(self, observations, global_state, deterministic=False):
        logits = self.actor(global_state.unsqueeze(0)).squeeze(0)
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions).sum(), self.critic(global_state.unsqueeze(0)).squeeze()

    def update(self):
        return {"actor_loss": 0.0, "critic_loss": 0.0}

    def checkpoint(self):
        return {"actor_state_dict": self.actor.state_dict(), "critic_state_dict": self.critic.state_dict(), "algorithm": "ppo"}
