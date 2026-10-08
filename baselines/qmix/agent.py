"""
QMIX 独立 baseline：使用局部 Agent-Q、全局 Mixing Network、Episode Replay 和目标网络完成离散 MARL 训练。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn

from config import DEVICE, NUM_AGENTS


class AgentQNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(obs_dim + NUM_AGENTS, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, observations, agent_ids, hidden=None):
        x = torch.cat((observations, agent_ids), dim=-1).unsqueeze(1)
        out, hidden = self.gru(x, hidden)
        return self.head(out[:, -1]), hidden


class MixingNetwork(nn.Module):
    def __init__(self, state_dim: int, agents: int = NUM_AGENTS, hidden_dim: int = 128):
        super().__init__()
        self.agents, self.hidden_dim = agents, hidden_dim
        self.hyper_w1 = nn.Linear(state_dim, agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, agent_values, state):
        batch = agent_values.shape[0]
        weights1 = torch.abs(self.hyper_w1(state)).view(batch, self.agents, self.hidden_dim)
        bias1 = self.hyper_b1(state).view(batch, 1, self.hidden_dim)
        hidden = torch.relu(torch.bmm(agent_values.unsqueeze(1), weights1) + bias1)
        weights2 = torch.abs(self.hyper_w2(state)).view(batch, self.hidden_dim, 1)
        bias2 = self.hyper_b2(state).view(batch, 1, 1)
        return (torch.bmm(hidden, weights2) + bias2).view(batch)


@dataclass
class Episode:
    observations: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class QMIXRuntime:
    def __init__(self, obs_dim: int, state_dim: int, hidden_dim: int = 128, capacity: int = 5000):
        self.agent_q = AgentQNetwork(obs_dim, hidden_dim).to(DEVICE)
        self.mixer = MixingNetwork(state_dim, NUM_AGENTS, hidden_dim).to(DEVICE)
        self.target_agent_q = AgentQNetwork(obs_dim, hidden_dim).to(DEVICE)
        self.target_mixer = MixingNetwork(state_dim, NUM_AGENTS, hidden_dim).to(DEVICE)
        self.target_agent_q.load_state_dict(self.agent_q.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimizer = torch.optim.Adam(list(self.agent_q.parameters()) + list(self.mixer.parameters()), lr=5e-4)
        self.replay = deque(maxlen=capacity)
        self.hidden = None
        self.update_count = 0

    def reset_episode(self):
        self.hidden = None

    def _agent_ids(self, batch: int):
        return torch.eye(NUM_AGENTS, device=DEVICE).unsqueeze(0).expand(batch, -1, -1)

    @torch.no_grad()
    def act(self, observations: torch.Tensor, state: torch.Tensor, epsilon: float):
        ids = torch.eye(NUM_AGENTS, device=DEVICE)
        q, self.hidden = self.agent_q(observations, ids, self.hidden)
        actions = q.argmax(-1)
        if epsilon > 0:
            mask = torch.rand(NUM_AGENTS, device=DEVICE) < epsilon
            actions[mask] = torch.randint(0, 3, (int(mask.sum()),), device=DEVICE)
        return actions

    def add_episode(self, episode: Episode):
        self.replay.append(episode)

    def update(self, batch_size: int = 32, gamma: float = 0.99):
        if len(self.replay) < batch_size:
            return {"loss": 0.0, "updates": self.update_count}
        batch = random.sample(self.replay, batch_size)
        obs = torch.stack([e.observations for e in batch]).to(DEVICE)
        states = torch.stack([e.states for e in batch]).to(DEVICE)
        actions = torch.stack([e.actions for e in batch]).to(DEVICE)
        rewards = torch.stack([e.rewards for e in batch]).to(DEVICE)
        dones = torch.stack([e.dones for e in batch]).to(DEVICE)
        batch_count, time = obs.shape[:2]
        ids = self._agent_ids(batch_count * time).reshape(batch_count * time, NUM_AGENTS, NUM_AGENTS)
        q, _ = self.agent_q(obs.reshape(batch_count * time, NUM_AGENTS, -1).reshape(-1, obs.shape[-1]), ids.reshape(-1, NUM_AGENTS), None)
        q = q.view(batch_count, time, NUM_AGENTS, 3).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        total_q = self.mixer(q.reshape(-1, NUM_AGENTS), states.reshape(-1, states.shape[-1])).view(batch_count, time)
        with torch.no_grad():
            target_q, _ = self.target_agent_q(obs.reshape(batch_count * time, NUM_AGENTS, -1).reshape(-1, obs.shape[-1]), ids.reshape(-1, NUM_AGENTS), None)
            target_q = target_q.view(batch_count, time, NUM_AGENTS, 3).max(-1).values
            next_total = self.target_mixer(target_q.reshape(-1, NUM_AGENTS), states.reshape(-1, states.shape[-1])).view(batch_count, time)
            target = rewards + gamma * (1.0 - dones) * next_total
        loss = nn.functional.smooth_l1_loss(total_q, target)
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(self.agent_q.parameters()) + list(self.mixer.parameters()), 10.0); self.optimizer.step()
        self.update_count += 1
        if self.update_count % 200 == 0:
            self.target_agent_q.load_state_dict(self.agent_q.state_dict()); self.target_mixer.load_state_dict(self.mixer.state_dict())
        return {"loss": float(loss.item()), "updates": self.update_count}

    def checkpoint(self, method, seed, episode, config):
        return {"method": method, "seed": seed, "episode": episode, "agent_q_state_dict": self.agent_q.state_dict(), "mixer_state_dict": self.mixer.state_dict(), "config": config.__dict__}
