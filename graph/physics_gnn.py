"""
物理感知 GNN 层：使用电网边消息、边注意力和 KCL 残差更新节点表示。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from config import PHYSICS_EDGE_DIM, PHYSICS_LAMBDA_KCL


class PhysicsGNNLayer(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = PHYSICS_EDGE_DIM, lambda_kcl: float = PHYSICS_LAMBDA_KCL):
        super().__init__(aggr="add")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.lambda_kcl = lambda_kcl

        self.W_s = nn.Parameter(torch.empty(out_dim, in_dim))
        self.W_n = nn.Parameter(torch.empty(out_dim, in_dim))
        self.W_kcl = nn.Parameter(torch.empty(out_dim, 1))
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * out_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_s)
        nn.init.xavier_uniform_(self.W_n)
        nn.init.xavier_uniform_(self.W_kcl)
        for m in self.attn_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, node_injection: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim], edge_index: [2, E], edge_attr: [E, edge_dim], node_injection: [N]
        x = torch.nan_to_num(x, nan=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
        node_injection = torch.nan_to_num(node_injection, nan=0.0)
        x_self = F.linear(x, self.W_s)  # [N, out_dim]
        kcl = self._compute_kcl(edge_index, edge_attr, node_injection)  # [N, 1]
        kcl = torch.clamp(kcl, -100, 100)
        msg = self.propagate(edge_index, x=x, edge_attr=edge_attr)  # [N, out_dim]
        out = x_self + msg - self.lambda_kcl * F.linear(kcl, self.W_kcl)  # [N, out_dim]
        out = torch.nan_to_num(out, nan=0.0)
        return torch.tanh(out)

    def _compute_kcl(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, node_injection: torch.Tensor) -> torch.Tensor:
        # node_injection: [N], edge_attr[:,0]: [E]
        num_nodes = int(node_injection.size(0))
        flow = edge_attr[:, 0].view(-1) if edge_attr.numel() else torch.zeros(0, device=node_injection.device)
        src = edge_index[0]
        residual = node_injection.clone().float()  # [N]
        if flow.numel() > 0:
            residual = residual.index_add(0, src, -flow)
        return residual.view(num_nodes, 1)  # [N, 1]

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x_i/x_j: [E, in_dim], edge_attr: [E, edge_dim]
        h_i = F.linear(x_i, self.W_n)  # [E, out_dim]
        h_j = F.linear(x_j, self.W_n)  # [E, out_dim]
        attn_in = torch.cat([h_i, h_j, edge_attr], dim=-1)  # [E, 2*out_dim+edge_dim]
        alpha = torch.sigmoid(self.attn_mlp(attn_in))  # [E, 1]
        return alpha * h_j  # [E, out_dim]


if __name__ == "__main__":
    N, E, in_dim, out_dim = 6, 8, 5, 7
    x = torch.randn(N, in_dim)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 5], [1, 2, 3, 4, 5, 0, 1, 2]], dtype=torch.long)
    edge_attr = torch.randn(E, 1)
    node_injection = torch.randn(N)
    layer = PhysicsGNNLayer(in_dim, out_dim)
    y = layer(x, edge_index, edge_attr, node_injection)
    assert y.shape == (N, out_dim)
    print("physics_gnn ok", y.shape)
