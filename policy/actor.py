"""
PhyG 共享 Actor：融合局部观测与图—时序表示，为各母线输出三分类防御动作。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import ACTOR_HIDDEN_DIM


class GraphActor(nn.Module):
    def __init__(self, obs_dim: int, mem_dim: int, action_dim: int, hidden_dim: int = ACTOR_HIDDEN_DIM, use_memory_gate: bool = False):
        super().__init__()
        self.use_memory_gate = bool(use_memory_gate)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        if self.use_memory_gate:
            self.memory_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + mem_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        # obs: [obs_dim] or [B, obs_dim], mem: [mem_dim] or [B, mem_dim]
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [1, obs_dim]
        obs_emb = self.obs_encoder(obs)  # [B, hidden_dim]
        if mem.dim() == 1:
            mem = mem.unsqueeze(0).expand(obs_emb.size(0), -1)  # [B, mem_dim]
        elif mem.dim() == 2 and mem.size(0) == 1:
            mem = mem.expand(obs_emb.size(0), -1)  # [B, mem_dim]
        elif mem.dim() == 2 and mem.size(0) != obs_emb.size(0):
            mem = mem[:1].expand(obs_emb.size(0), -1)  # [B, mem_dim]
        if self.use_memory_gate:
            mem = torch.sigmoid(self.memory_gate_logit) * mem
        feats = torch.cat([obs_emb, mem], dim=-1)  # [B, hidden_dim + mem_dim]
        feats = torch.nan_to_num(feats, nan=0.0)
        logits = self.head(feats)  # [B, action_dim]
        logits = torch.clamp(logits, -10, 10)
        return logits


if __name__ == "__main__":
    actor = GraphActor(obs_dim=12, mem_dim=16, action_dim=3)
    obs = torch.randn(4, 12)
    mem = torch.randn(16)
    logits = actor(obs, mem)
    assert logits.shape == (4, 3)
    print("actor ok", logits.shape)
