from __future__ import annotations

import torch
from torch.distributions import Categorical

from baselines.common.base_agent import BaselineAgent
from baselines.common.env_adapter import observation_tensor
from .gat_encoder import GATEncoder


class GATMAPPOAgent(BaselineAgent):
    def __init__(self, obs_dim: int, state_dim: int, device: torch.device, actor_lr: float = 1e-4, critic_lr: float = 1e-4):
        self.device = device
        self.encoder = GATEncoder(obs_dim).to(device)
        self.actor_head = torch.nn.Linear(128 * 2, 3).to(device)
        self.critic = torch.nn.Sequential(torch.nn.Linear(state_dim, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1)).to(device)
        actor_parameters = list(self.encoder.parameters()) + list(self.actor_head.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_parameters, lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def act(self, observations, global_state, deterministic=False):
        observations_tensor = observation_tensor(observations, self.device)
        encoded = self.encoder(observations_tensor)
        pooled = encoded.mean(dim=0, keepdim=True).expand(encoded.shape[0], -1)
        logits = self.actor_head(torch.cat((encoded, pooled), dim=-1))
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions).sum(), self.critic(global_state.unsqueeze(0)).squeeze()

    def update(self):
        return {"actor_loss": 0.0, "critic_loss": 0.0}

    def checkpoint(self):
        return {"encoder_state_dict": self.encoder.state_dict(), "actor_head_state_dict": self.actor_head.state_dict(), "critic_state_dict": self.critic.state_dict(), "algorithm": "gat_mappo"}
