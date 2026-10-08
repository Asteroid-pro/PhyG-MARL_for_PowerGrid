"""
Baseline PPO 更新引擎：统一处理 rollout、GAE、PPO clipped objective、Actor/Critic 更新和 IPPO 局部价值更新。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.distributions import Categorical

from baseline_config import BaselineConfig


@dataclass
class Transition:
    obs: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    done: bool
    log_prob: torch.Tensor
    value: torch.Tensor
    graph_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    hidden_state: torch.Tensor | None = None
    episode_start: bool = False


class OnPolicyPPO:
    def __init__(self, actor: nn.Module, critic: nn.Module, actor_forward: Callable,
                 critic_forward: Callable, config: BaselineConfig, device: torch.device):
        self.actor, self.critic = actor, critic
        self.actor_forward, self.critic_forward = actor_forward, critic_forward
        self.config, self.device = config, device
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
        self.buffer: list[Transition] = []
        self.update_count = 0

    def collect(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def update(self) -> dict[str, float]:
        if not self.buffer:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "explained_variance": 0.0}
        rewards = torch.tensor([t.reward * 0.1 for t in self.buffer], device=self.device)
        dones = torch.tensor([t.done for t in self.buffer], dtype=torch.float32, device=self.device)
        values = torch.stack([t.value.detach() for t in self.buffer]).reshape(-1)
        returns = torch.zeros_like(rewards); advantages = torch.zeros_like(rewards)
        gae = torch.zeros((), device=self.device); next_value = torch.zeros((), device=self.device)
        for i in reversed(range(len(self.buffer))):
            delta = rewards[i] + self.config.gamma * next_value * (1.0 - dones[i]) - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1.0 - dones[i]) * gae
            advantages[i] = gae; returns[i] = gae + values[i]; next_value = values[i]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_loss_value = critic_loss_value = entropy_value = 0.0
        for _ in range(self.config.ppo_epochs):
            for index in torch.randperm(len(self.buffer)).tolist():
                transition = self.buffer[index]
                logits = self.actor_forward(transition.obs, transition.state, transition.graph_data, transition.hidden_state)
                distribution = Categorical(logits=logits)
                new_log_prob = distribution.log_prob(transition.action).sum()
                ratio = torch.exp(new_log_prob - transition.log_prob.detach())
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                actor_loss = -torch.minimum(ratio * advantages[index], clipped * advantages[index]).mean() - self.config.entropy_coef * distribution.entropy().mean()
                self.actor_optimizer.zero_grad(); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5); self.actor_optimizer.step()
                predicted = self.critic_forward(transition.obs, transition.state)
                critic_loss = nn.functional.smooth_l1_loss(predicted, returns[index])
                self.critic_optimizer.zero_grad(); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5); self.critic_optimizer.step()
                actor_loss_value += float(actor_loss.item()); critic_loss_value += float(critic_loss.item()); entropy_value += float(distribution.entropy().mean().item())
        count = max(len(self.buffer) * self.config.ppo_epochs, 1)
        explained = 1.0 - torch.var(returns - values) / (torch.var(returns) + 1e-8)
        self.buffer.clear(); self.update_count += 1
        return {"actor_loss": actor_loss_value / count, "critic_loss": critic_loss_value / count, "entropy": entropy_value / count, "explained_variance": float(explained.item()), "update_count": self.update_count}


class IPPOEngine(OnPolicyPPO):
    def update(self) -> dict[str, float]:
        if not self.buffer:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "explained_variance": 0.0}
        values = torch.stack([transition.value for transition in self.buffer])
        rewards = torch.tensor([transition.reward * 0.1 for transition in self.buffer], device=self.device)
        dones = torch.tensor([transition.done for transition in self.buffer], dtype=torch.float32, device=self.device)
        returns = torch.zeros_like(values)
        advantages = torch.zeros_like(values)
        next_values = torch.zeros(values.shape[1], device=self.device)
        gae = torch.zeros(values.shape[1], device=self.device)
        for index in reversed(range(len(self.buffer))):
            delta = rewards[index] + self.config.gamma * next_values * (1.0 - dones[index]) - values[index]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1.0 - dones[index]) * gae
            advantages[index] = gae; returns[index] = gae + values[index]; next_values = values[index].detach()
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(self.config.ppo_epochs):
            for index in range(len(self.buffer)):
                transition = self.buffer[index]
                logits = self.actor_forward(transition.obs, transition.state, transition.graph_data, transition.hidden_state)
                distribution = Categorical(logits=logits); log_probs = distribution.log_prob(transition.action)
                ratios = torch.exp(log_probs - transition.log_prob.detach())
                clipped = torch.clamp(ratios, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                actor_loss = -torch.minimum(ratios * normalized_advantages[index], clipped * normalized_advantages[index]).mean() - self.config.entropy_coef * distribution.entropy().mean()
                self.actor_optimizer.zero_grad(); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5); self.actor_optimizer.step()
                local_obs = transition.obs
                predicted = self.critic_forward(local_obs, transition.state)
                critic_loss = nn.functional.smooth_l1_loss(predicted, returns[index])
                self.critic_optimizer.zero_grad(); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5); self.critic_optimizer.step()
        self.buffer.clear(); self.update_count += 1
        return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item()), "entropy": float(distribution.entropy().mean().item()), "explained_variance": 0.0, "update_count": self.update_count}
