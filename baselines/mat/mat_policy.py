"""
MAT 独立 baseline：使用 Transformer 编码多智能体局部观测，并按母线顺序自回归生成离散动作。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import NUM_AGENTS


class MATPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128, heads: int = 4, layers: int = 2):
        super().__init__()
        self.observation_projection = nn.Linear(obs_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, batch_first=True, dropout=0.0, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.action_embedding = nn.Embedding(3, hidden_dim)
        self.decoder = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 3)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        tokens = torch.tanh(self.observation_projection(observations)).unsqueeze(0)
        return self.encoder(tokens).squeeze(0)

    def autoregressive(self, encoded: torch.Tensor, actions: torch.Tensor | None = None, deterministic: bool = False):
        hidden = encoded.mean(dim=0)
        logits_list, actions_list, log_probs, entropies = [], [], [], []
        previous = torch.zeros((), dtype=torch.long, device=encoded.device)
        for index in range(NUM_AGENTS):
            hidden = self.decoder(self.action_embedding(previous), hidden)
            logits = self.action_head(hidden + encoded[index])
            distribution = Categorical(logits=logits)
            action = actions[index] if actions is not None else (torch.argmax(logits) if deterministic else distribution.sample())
            logits_list.append(logits)
            actions_list.append(action)
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            previous = action
        return torch.stack(logits_list), torch.stack(actions_list), torch.stack(log_probs).sum(), torch.stack(entropies).mean()

    def forward(self, observations: torch.Tensor, actions: torch.Tensor | None = None, deterministic: bool = False):
        encoded = self.encode(observations)
        logits, selected_actions, log_prob, entropy = self.autoregressive(encoded, actions, deterministic)
        value = self.value_head(encoded.mean(dim=0)).squeeze(-1)
        return logits, selected_actions, log_prob, entropy, value
