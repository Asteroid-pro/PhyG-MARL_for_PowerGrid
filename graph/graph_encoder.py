"""
电网图编码器：将节点、线路和节点注入信息编码为动态物理图表示。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import GRAPH_NUM_LAYERS, HIDDEN_DIM
from .physics_gnn import PhysicsGNNLayer


class GraphEncoder(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int = HIDDEN_DIM, num_layers: int = GRAPH_NUM_LAYERS):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([PhysicsGNNLayer(hidden_dim, hidden_dim, edge_dim=hidden_dim) for _ in range(num_layers)])
        self.pool_q = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, node_feats: torch.Tensor, edge_feats: torch.Tensor, edge_index: torch.Tensor, node_injection: torch.Tensor):
        # node_feats: [N, node_in_dim], edge_feats: [E, edge_in_dim], edge_index: [2, E], node_injection: [N]
        x = self.node_proj(node_feats)  # [N, hidden_dim]
        e = self.edge_proj(edge_feats)  # [E, hidden_dim]
        for layer in self.layers:
            x = layer(x, edge_index, e, node_injection)  # [N, hidden_dim]
        attn_logits = (x * self.pool_q).sum(dim=-1)  # [N]
        attn = torch.softmax(attn_logits, dim=0).unsqueeze(-1)  # [N, 1]
        graph_embedding = torch.sum(attn * x, dim=0)  # [hidden_dim]
        return x, graph_embedding


if __name__ == "__main__":
    N, E = 6, 8
    enc = GraphEncoder(node_in_dim=4, edge_in_dim=3, hidden_dim=16, num_layers=2)
    node_feats = torch.randn(N, 4)
    edge_feats = torch.randn(E, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 5], [1, 2, 3, 4, 5, 0, 1, 2]], dtype=torch.long)
    node_injection = torch.randn(N)
    node_emb, graph_emb = enc(node_feats, edge_feats, edge_index, node_injection)
    assert node_emb.shape == (N, 16)
    assert graph_emb.shape == (16,)
    print("graph_encoder ok", node_emb.shape, graph_emb.shape)
