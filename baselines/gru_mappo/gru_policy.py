from __future__ import annotations

import torch
import torch.nn as nn


class GRUPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, observations: torch.Tensor, hidden_state: torch.Tensor | None = None):
        encoded = torch.tanh(self.encoder(observations)).unsqueeze(1)
        output, hidden_state = self.gru(encoded, hidden_state)
        return self.head(output[:, -1]), hidden_state
