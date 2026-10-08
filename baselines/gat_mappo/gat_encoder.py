from __future__ import annotations

import torch
import torch.nn as nn


class GATEncoder(nn.Module):
    """Generic attention aggregator over bus tokens; it does not use PhyG modules."""

    def __init__(self, node_dim: int = 4, hidden_dim: int = 128, heads: int = 4):
        super().__init__()
        self.projection = nn.Linear(node_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        del edge_features, injection
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        projected = torch.tanh(self.projection(node_features))
        messages = torch.zeros_like(projected)
        degree = torch.zeros(projected.shape[0], device=projected.device)
        sources, targets = edge_index.long()
        messages.index_add_(0, targets, projected[sources])
        degree.index_add_(0, targets, torch.ones_like(targets, dtype=degree.dtype))
        neighborhood = messages / degree.clamp_min(1.0).unsqueeze(-1)
        tokens = torch.stack((projected, neighborhood), dim=0).mean(dim=0).unsqueeze(0)
        attended, _ = self.attention(tokens, tokens, tokens)
        return attended.squeeze(0)
