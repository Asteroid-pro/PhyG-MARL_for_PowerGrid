from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def clipped_ppo_update(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    logits_and_values: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    clip_ratio: float,
    entropy_coef: float,
) -> dict[str, float]:
    actor_loss = torch.zeros((), device=next(actor.parameters()).device)
    critic_loss = torch.zeros_like(actor_loss)
    entropy_total = torch.zeros_like(actor_loss)
    count = 0
    for logits, action, old_log_prob in logits_and_values:
        distribution = Categorical(logits=logits)
        log_prob = distribution.log_prob(action)
        ratio = torch.exp(log_prob - old_log_prob.detach())
        advantage = torch.ones_like(ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        actor_loss = actor_loss - torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
        entropy_total = entropy_total + distribution.entropy().mean()
        count += 1
    if count == 0:
        return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
    actor_loss = actor_loss / count - entropy_coef * entropy_total / count
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
    actor_optimizer.step()
    return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item()), "entropy": float((entropy_total / count).item())}


def value_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(predicted, target)
