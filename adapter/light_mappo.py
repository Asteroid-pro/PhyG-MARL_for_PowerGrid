"""
轻量化 MAPPO 优化器：执行 GAE、PPO clipped objective、Actor/Critic 更新与训练统计。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    MAPPO_ACTOR_LR_END,
    MAPPO_BATCH_SIZE,
    MAPPO_CLIP_RATIO,
    MAPPO_CRITIC_LR_END,
    MAPPO_CRITIC_LR_START,
    MAPPO_EPOCHS,
    MAPPO_ENTROPY_COEF,
    MAPPO_GAMMA,
    MAPPO_LAM,
    MAPPO_LR,
    MAPPO_LR_TOTAL_UPDATES,
    MAPPO_USE_LR_SCHEDULE,
)


@dataclass
class Transition:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    value: torch.Tensor
    log_prob: torch.Tensor
    actor_memory: torch.Tensor
    global_state: torch.Tensor
    node_feats: torch.Tensor | None = None
    edge_feats: torch.Tensor | None = None
    edge_index: torch.Tensor | None = None
    node_injection: torch.Tensor | None = None
    episode_start: bool = False


class LightMAPPO:
    def __init__(self, actor, critic, lr: float = 1e-4, gamma: float = MAPPO_GAMMA, lam: float = MAPPO_LAM, clip_ratio: float = MAPPO_CLIP_RATIO, entropy_coef: float = MAPPO_ENTROPY_COEF, epochs: int = 5, batch_size: int = MAPPO_BATCH_SIZE, target_kl: float = 0.02, graph_encoder=None, memory=None, representation_mode: str = "frozen", representation_lr: float = 1e-5, sequence_length: int = 32, critic_lr: float | None = None, value_clip: float = 0.0, critic_epochs: int = 10, normalize_returns: bool = True):
        if representation_mode not in {"frozen", "joint"}:
            raise ValueError("representation_mode must be 'frozen' or 'joint'")
        self.actor = actor
        self.critic = critic
        self.graph_encoder = graph_encoder
        self.memory = memory
        self.representation_mode = representation_mode
        self.sequence_length = int(sequence_length)
        if self.representation_mode == "joint" and (graph_encoder is None or memory is None):
            raise ValueError("joint mode requires graph_encoder and memory")
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.value_clip = float(value_clip)
        self.critic_epochs = max(int(critic_epochs), 1)
        self.normalize_returns = bool(normalize_returns)
        self.return_mean = 0.0
        self.return_var = 1.0
        self.return_count = 1e-4
        self.action_dim = None
        self.buffer: List[Transition] = []
        self.use_lr_schedule = bool(MAPPO_USE_LR_SCHEDULE)
        self.lr_total_updates = max(int(MAPPO_LR_TOTAL_UPDATES), 1)
        self.lr_update_count = 0
        self.actor_lr_start = float(lr)
        self.actor_lr_end = float(MAPPO_ACTOR_LR_END)
        self.critic_lr_start = float(critic_lr if critic_lr is not None else lr)
        self.critic_lr_end = min(float(MAPPO_CRITIC_LR_END), self.critic_lr_start)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr_start)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr_start)
        self.representation_optim = None
        self.graph_trainable = self.representation_mode == "joint"
        self.memory_trainable = self.representation_mode == "joint"
        if self.representation_mode == "joint":
            self.representation_optim = torch.optim.Adam(
                list(self.graph_encoder.parameters()) + list(self.memory.parameters()),
                lr=float(representation_lr),
            )

    def collect_experience(self, obs_list, actions_list, rewards_list, dones_list, values_list, log_probs_list, actor_memory, global_state, graph_data=None, episode_start=False):
        graph_data = graph_data or (None, None, None, None)
        node_feats, edge_feats, edge_index, node_injection = graph_data
        def cpu_copy(value):
            return value.detach().cpu() if torch.is_tensor(value) else value
        self.buffer.append(Transition(
            obs=torch.stack([o.detach().cpu() if torch.is_tensor(o) else torch.as_tensor(o) for o in obs_list]),
            action=torch.as_tensor(actions_list, dtype=torch.long),
            reward=torch.as_tensor(rewards_list, dtype=torch.float32),
            done=torch.as_tensor(dones_list, dtype=torch.float32),
            value=torch.as_tensor(values_list, dtype=torch.float32),
            log_prob=torch.as_tensor(log_probs_list, dtype=torch.float32),
            actor_memory=actor_memory.detach().cpu() if torch.is_tensor(actor_memory) else torch.as_tensor(actor_memory, dtype=torch.float32),
            global_state=global_state.detach().cpu() if torch.is_tensor(global_state) else torch.as_tensor(global_state, dtype=torch.float32),
            node_feats=cpu_copy(node_feats),
            edge_feats=cpu_copy(edge_feats),
            edge_index=cpu_copy(edge_index),
            node_injection=cpu_copy(node_injection),
            episode_start=bool(episode_start),
        ))

    def _linear_lr(self, start_lr: float, end_lr: float, update_idx: int) -> float:
        progress = min(max(update_idx, 0) / float(self.lr_total_updates), 1.0)
        return start_lr + (end_lr - start_lr) * progress

    def _update_learning_rates(self):
        if not self.use_lr_schedule:
            return
        actor_lr = self._linear_lr(self.actor_lr_start, self.actor_lr_end, self.lr_update_count)
        critic_lr = self._linear_lr(self.critic_lr_start, self.critic_lr_end, self.lr_update_count)
        for group in self.actor_optim.param_groups:
            group["lr"] = actor_lr
        for group in self.critic_optim.param_groups:
            group["lr"] = critic_lr

    def get_current_lrs(self):
        values = {
            "actor_lr": float(self.actor_optim.param_groups[0]["lr"]),
            "critic_lr": float(self.critic_optim.param_groups[0]["lr"]),
        }
        if self.representation_optim is not None:
            values["representation_lr"] = float(self.representation_optim.param_groups[0]["lr"])
        return values

    def denormalize_value(self, value: torch.Tensor) -> torch.Tensor:
        if not self.normalize_returns:
            return value
        return value * float(np.sqrt(self.return_var) + 1e-8) + self.return_mean

    def _update_return_statistics(self, returns: torch.Tensor) -> None:
        values = returns.detach().reshape(-1).double()
        batch_count = values.numel()
        if batch_count == 0:
            return
        batch_mean = float(values.mean().item())
        batch_var = float(values.var(unbiased=False).item())
        delta = batch_mean - self.return_mean
        total_count = self.return_count + batch_count
        new_mean = self.return_mean + delta * batch_count / total_count
        m_a = self.return_var * self.return_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.return_count * batch_count / total_count
        self.return_mean = new_mean
        self.return_var = max(m2 / total_count, 1e-8)
        self.return_count = total_count

    def set_representation_trainable(self, graph_trainable: bool, memory_trainable: bool) -> None:
        if self.graph_encoder is not None:
            for parameter in self.graph_encoder.parameters():
                parameter.requires_grad_(graph_trainable)
        if self.memory is not None:
            for parameter in self.memory.parameters():
                parameter.requires_grad_(memory_trainable)
        self.graph_trainable = bool(graph_trainable)
        self.memory_trainable = bool(memory_trainable)

    def buffer_add(self, obs_list, actions_list, rewards_dict, dones_dict, values_list, log_probs_list, global_state):
        keys = sorted(rewards_dict.keys())
        rewards_list = [rewards_dict[k] for k in keys]
        if any((r is None) or (not np.isfinite(float(r))) for r in rewards_list):
            return
        dones_list = [dones_dict[k] for k in keys]
        self.collect_experience(obs_list, actions_list, rewards_list, dones_list, values_list, log_probs_list, global_state)

    @staticmethod
    def compute_gae(rewards, dones, values, gamma, lam, bootstrap_value=0.0):
        adv = torch.zeros_like(rewards)
        gae = 0.0
        next_value = torch.as_tensor(bootstrap_value, dtype=values.dtype, device=values.device)
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae
            next_value = values[t]
        return adv, adv + values

    def get_actions(self, obs_list, memory_vec, deterministic=False):
        obs = torch.stack([o if torch.is_tensor(o) else torch.as_tensor(o, dtype=torch.float32) for o in obs_list])
        obs = torch.nan_to_num(obs, nan=0.0).to(next(self.actor.parameters()).device)
        memory_vec = torch.nan_to_num(memory_vec, nan=0.0).to(obs.device)
        logits = self.actor(obs, memory_vec)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        if self.action_dim is None:
            self.action_dim = int(logits.shape[-1])
        assert logits.shape[-1] == self.action_dim, f"logits dim {logits.shape[-1]} != action_dim {self.action_dim}"
        dist = torch.distributions.Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        max_action = int(actions.max().item()) if actions.numel() else -1
        if max_action >= self.action_dim:
            raise ValueError(f"动作最大值 {max_action} 超出动作空间维度 {self.action_dim}")
        log_probs = dist.log_prob(actions)
        return actions.to(torch.long).tolist(), log_probs.detach(), None

    def _update_joint(self, bootstrap_value: float = 0.0):
        device = next(self.actor.parameters()).device
        transitions = self.buffer
        if any(t.node_feats is None for t in transitions):
            raise ValueError("joint mode requires graph data in every transition")
        obs_time = torch.stack([t.obs for t in transitions]).to(device)
        actions_time = torch.stack([t.action for t in transitions]).long().to(device)
        rewards_time = torch.stack([t.reward for t in transitions]).to(device)
        dones_time = torch.stack([t.done for t in transitions]).to(device)
        values_time = torch.stack([t.value for t in transitions]).to(device)
        old_log_probs_time = torch.stack([t.log_prob for t in transitions]).to(device)
        advantages_time = torch.zeros_like(rewards_time)
        returns_time = torch.zeros_like(rewards_time)
        for agent_idx in range(rewards_time.shape[1]):
            advantages_time[:, agent_idx], returns_time[:, agent_idx] = self.compute_gae(
                rewards_time[:, agent_idx], dones_time[:, agent_idx], values_time[:, agent_idx],
                self.gamma, self.lam, bootstrap_value,
            )
        adv_mean = advantages_time.mean()
        adv_std = advantages_time.std()
        normalized_advantages = (advantages_time - adv_mean) / (adv_std + 1e-8)
        if self.normalize_returns:
            self._update_return_statistics(returns_time)
            return_mean = torch.as_tensor(self.return_mean, dtype=returns_time.dtype, device=device)
            return_std = torch.as_tensor(np.sqrt(self.return_var), dtype=returns_time.dtype, device=device)
            critic_targets = (returns_time - return_mean) / (return_std + 1e-8)
        else:
            return_mean = torch.zeros((), dtype=returns_time.dtype, device=device)
            return_std = torch.ones((), dtype=returns_time.dtype, device=device)
            critic_targets = returns_time

        episode_ranges = []
        episode_start = 0
        for index, transition in enumerate(transitions):
            if index > episode_start and transition.episode_start:
                episode_ranges.append((episode_start, index))
                episode_start = index
            if bool(transition.done.any()) or index == len(transitions) - 1:
                episode_ranges.append((episode_start, index + 1))
                episode_start = index + 1
        if episode_start < len(transitions):
            episode_ranges.append((episode_start, len(transitions)))
        blocks = []
        for ep_start, ep_end in episode_ranges:
            for start in range(ep_start, ep_end, self.sequence_length):
                blocks.append((start, min(start + self.sequence_length, ep_end), ep_start))

        actor_loss_val = critic_loss_val = entropy_val = 0.0
        actor_grad_norm = critic_grad_norm = 0.0
        graph_grad_norm = memory_grad_norm = 0.0
        graph_parameter_delta = memory_parameter_delta = 0.0
        graph_parameter_relative_delta = memory_parameter_relative_delta = 0.0
        value_pred_mean = value_pred_std = value_target_mean = value_target_std = 0.0
        value_error_mean = value_error_abs_mean = value_clip_fraction = 0.0
        approx_kl_val = clip_fraction_val = explained_variance_val = 0.0
        actual_epochs = 0
        early_stop = False
        num_agents = obs_time.shape[1]
        history_length = self.memory.history_length
        for _ in range(self.epochs):
            actual_epochs += 1
            np.random.shuffle(blocks)
            epoch_kls = []
            for start, end, ep_start in blocks:
                context_start = max(ep_start, start - history_length + 1)
                graph_embeddings = []
                for index in range(context_start, end):
                    transition = transitions[index]
                    _, embedding = self.graph_encoder(
                        transition.node_feats.to(device), transition.edge_feats.to(device),
                        transition.edge_index.to(device), transition.node_injection.to(device),
                    )
                    if index < start:
                        embedding = embedding.detach()
                    graph_embeddings.append(embedding)
                zero = torch.zeros_like(graph_embeddings[0])
                windows = []
                for index in range(start, end):
                    relative = index - context_start
                    window = graph_embeddings[max(0, relative - history_length + 1):relative + 1]
                    windows.append(torch.stack([zero] * (history_length - len(window)) + window))
                memory_outputs = self.memory.forward_sequence(torch.stack(windows))[:, -1]
                valid_steps = end - start
                b_obs = obs_time[start:end].reshape(valid_steps * num_agents, -1)
                b_actions = actions_time[start:end].reshape(-1)
                b_old_logp = old_log_probs_time[start:end].reshape(-1)
                b_adv = normalized_advantages[start:end].reshape(-1)
                b_returns = critic_targets[start:end].reshape(-1)
                actor_memory = memory_outputs.repeat_interleave(num_agents, dim=0)
                logits = torch.nan_to_num(self.actor(b_obs, actor_memory), nan=0.0, posinf=1e6, neginf=-1e6)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_actions)
                ratio = torch.exp(logp - b_old_logp)
                clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * b_adv
                actor_loss = -torch.min(ratio * b_adv, clipped).mean() - self.entropy_coef * dist.entropy().mean()
                approx_kl = (b_old_logp - logp).mean()
                clip_fraction = (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean()

                graph_before = [p.detach().clone() for p in self.graph_encoder.parameters()] if self.graph_trainable else []
                memory_before = [p.detach().clone() for p in self.memory.parameters()] if self.memory_trainable else []
                self.actor_optim.zero_grad()
                self.representation_optim.zero_grad()
                actor_loss.backward()
                actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5).item())
                if self.graph_trainable:
                    graph_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.graph_encoder.parameters(), 0.5).item())
                if self.memory_trainable:
                    memory_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.memory.parameters(), 0.5).item())
                self.actor_optim.step()
                if self.graph_trainable or self.memory_trainable:
                    self.representation_optim.step()
                if graph_before:
                    graph_after = list(self.graph_encoder.parameters())
                    delta_sq = sum(torch.sum((after.detach() - before) ** 2) for before, after in zip(graph_before, graph_after))
                    base_sq = sum(torch.sum(before ** 2) for before in graph_before)
                    graph_parameter_delta = float(torch.sqrt(delta_sq).item())
                    graph_parameter_relative_delta = float((torch.sqrt(delta_sq) / (torch.sqrt(base_sq) + 1e-12)).item())
                if memory_before:
                    memory_after = list(self.memory.parameters())
                    delta_sq = sum(torch.sum((after.detach() - before) ** 2) for before, after in zip(memory_before, memory_after))
                    base_sq = sum(torch.sum(before ** 2) for before in memory_before)
                    memory_parameter_delta = float(torch.sqrt(delta_sq).item())
                    memory_parameter_relative_delta = float((torch.sqrt(delta_sq) / (torch.sqrt(base_sq) + 1e-12)).item())

                global_states = torch.stack([t.global_state for t in transitions[start:end]]).to(device)
                critic_input = global_states.repeat_interleave(num_agents, dim=0)
                critic_values = self.critic(critic_input).squeeze(-1)
                critic_loss = F.smooth_l1_loss(critic_values, b_returns)
                value_error = critic_values.detach() - b_returns
                value_pred_mean = float(critic_values.detach().mean().item())
                value_pred_std = float(critic_values.detach().std().item())
                value_target_mean = float(b_returns.mean().item())
                value_target_std = float(b_returns.std().item())
                value_error_mean = float(value_error.mean().item())
                value_error_abs_mean = float(value_error.abs().mean().item())
                value_clip_fraction = 0.0
                self.critic_optim.zero_grad()
                critic_loss.backward()
                critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5).item())
                self.critic_optim.step()

                returns_var = torch.var(b_returns)
                if returns_var.item() > 1e-8:
                    explained_variance_val = float((1.0 - torch.var(b_returns - critic_values.detach()) / returns_var).item())
                actor_loss_val = float(actor_loss.item())
                critic_loss_val = float(critic_loss.item())
                entropy_val = float(dist.entropy().mean().item())
                approx_kl_val = float(approx_kl.item())
                clip_fraction_val = float(clip_fraction.item())
                epoch_kls.append(approx_kl_val)
            if epoch_kls and float(np.mean(epoch_kls)) > self.target_kl:
                early_stop = True
                break

        for _ in range(max(self.critic_epochs - actual_epochs, 0)):
            critic_blocks = list(blocks)
            np.random.shuffle(critic_blocks)
            for start, end, _ in critic_blocks:
                num_agents = obs_time.shape[1]
                global_states = torch.stack([t.global_state for t in transitions[start:end]]).to(device)
                critic_input = global_states.repeat_interleave(num_agents, dim=0)
                b_returns = critic_targets[start:end].reshape(-1)
                critic_values = self.critic(critic_input).squeeze(-1)
                critic_loss = F.smooth_l1_loss(critic_values, b_returns)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5).item())
                self.critic_optim.step()
                value_error = critic_values.detach() - b_returns
                critic_loss_val = float(critic_loss.item())
                value_pred_mean = float(critic_values.detach().mean().item())
                value_pred_std = float(critic_values.detach().std().item())
                value_target_mean = float(b_returns.mean().item())
                value_target_std = float(b_returns.std().item())
                value_error_mean = float(value_error.mean().item())
                value_error_abs_mean = float(value_error.abs().mean().item())
                returns_var = torch.var(b_returns)
                if returns_var.item() > 1e-8:
                    explained_variance_val = float((1.0 - torch.var(value_error) / returns_var).item())

        self.lr_update_count += 1
        self._update_learning_rates()
        self.buffer.clear()
        return {
            "actor_loss": actor_loss_val, "critic_loss": critic_loss_val, "entropy": entropy_val,
            "approx_kl": approx_kl_val, "clip_fraction": clip_fraction_val,
            "explained_variance": explained_variance_val,
            "actor_grad_norm": actor_grad_norm, "critic_grad_norm": critic_grad_norm,
            "graph_grad_norm": graph_grad_norm, "memory_grad_norm": memory_grad_norm,
            "graph_parameter_delta": graph_parameter_delta, "memory_parameter_delta": memory_parameter_delta,
            "graph_parameter_relative_delta": graph_parameter_relative_delta, "memory_parameter_relative_delta": memory_parameter_relative_delta,
            "value_pred_mean": value_pred_mean, "value_pred_std": value_pred_std,
            "value_target_mean": value_target_mean, "value_target_std": value_target_std,
            "value_error_mean": value_error_mean, "value_error_abs_mean": value_error_abs_mean,
            "value_clip_fraction": value_clip_fraction,
            "memory_gate": float(torch.sigmoid(self.actor.memory_gate_logit).item()) if getattr(self.actor, "use_memory_gate", False) else 1.0,
            "memory_gate_logit": float(self.actor.memory_gate_logit.item()) if getattr(self.actor, "use_memory_gate", False) else 0.0,
            "graph_trainable": int(self.graph_trainable), "memory_trainable": int(self.memory_trainable),
            "adv_mean": float(adv_mean.item()), "adv_std": float(adv_std.item()),
            "policy_entropy": entropy_val, "actual_update_epochs": actual_epochs,
            "early_stop_by_kl": float(early_stop), "update_count": self.lr_update_count,
        }

    def update(self, bootstrap_value: float = 0.0):
        if self.representation_mode == "joint":
            if not self.buffer:
                return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "graph_grad_norm": 0.0, "memory_grad_norm": 0.0, "update_count": self.lr_update_count}
            return self._update_joint(bootstrap_value)
        if not self.buffer:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "explained_variance": 0.0,
                "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0,
                "adv_mean": 0.0,
                "adv_std": 0.0,
                "policy_entropy": 0.0,
            }
        if any(not np.isfinite(float(t.reward.mean().item())) for t in self.buffer):
            self.buffer.clear()
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "explained_variance": 0.0,
                "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0,
                "adv_mean": 0.0,
                "adv_std": 0.0,
                "policy_entropy": 0.0,
            }
        device = next(self.actor.parameters()).device
        obs_time = torch.stack([t.obs for t in self.buffer], dim=0).to(device)
        actions_time = torch.stack([t.action for t in self.buffer], dim=0).long().to(device)
        rewards_time = torch.stack([t.reward for t in self.buffer], dim=0).to(device)
        dones_time = torch.stack([t.done for t in self.buffer], dim=0).to(device)
        values_time = torch.stack([t.value for t in self.buffer], dim=0).to(device)
        log_probs_time = torch.stack([t.log_prob for t in self.buffer], dim=0).to(device)
        memory_time = torch.stack([t.actor_memory for t in self.buffer], dim=0).to(device)
        state_time = torch.stack([t.global_state for t in self.buffer], dim=0).to(device)
        num_steps, num_agents = rewards_time.shape

        advantages_time = torch.zeros_like(rewards_time)
        returns_time = torch.zeros_like(rewards_time)
        for agent_idx in range(num_agents):
            agent_adv, agent_returns = self.compute_gae(
                rewards_time[:, agent_idx],
                dones_time[:, agent_idx],
                values_time[:, agent_idx],
                self.gamma,
                self.lam,
                bootstrap_value,
            )
            advantages_time[:, agent_idx] = agent_adv
            returns_time[:, agent_idx] = agent_returns

        obs = obs_time.reshape(num_steps * num_agents, -1)
        actions = actions_time.reshape(-1)
        old_log_probs = log_probs_time.reshape(-1)
        returns = returns_time.reshape(-1)
        advantages_raw = advantages_time.reshape(-1)
        actor_memories = memory_time.repeat_interleave(num_agents, dim=0)
        global_states = state_time.repeat_interleave(num_agents, dim=0)
        if actions.numel() > 0:
            max_action = int(actions.max().item())
            if self.action_dim is None:
                self.action_dim = int(max_action + 1)
            if max_action >= self.action_dim:
                raise ValueError(f"动作最大值 {max_action} 超出动作空间维度 {self.action_dim}")
        adv_mean = float(advantages_raw.mean().item())
        adv_std = float(advantages_raw.std().item())
        advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)
        total = obs.size(0)
        idx = torch.randperm(total, device=device)
        actor_loss_val = critic_loss_val = entropy_val = 0.0
        approx_kl_val = 0.0
        clip_fraction_val = 0.0
        actor_grad_norm_val = 0.0
        critic_grad_norm_val = 0.0
        policy_entropy_val = 0.0
        policy_entropy_count = 0
        explained_variance_val = 0.0
        actual_update_epochs = 0
        early_stop_by_kl = False

        returns_var = torch.var(returns)
        for _ in range(self.epochs):
            actual_update_epochs += 1
            idx = torch.randperm(total, device=device)
            for start in range(0, total, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                b_obs = obs[batch_idx]
                b_actions = actions[batch_idx]
                b_adv = advantages[batch_idx]
                b_returns = returns[batch_idx]
                b_old_logp = old_log_probs[batch_idx]
                b_actor_memory = actor_memories[batch_idx]
                b_global = global_states[batch_idx]
                logits = self.actor(b_obs, b_actor_memory)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                assert logits.shape[-1] == self.action_dim, f"logits dim {logits.shape[-1]} != action_dim {self.action_dim}"
                max_action = int(b_actions.max().item()) if b_actions.numel() else -1
                if max_action >= self.action_dim:
                    raise ValueError(f"动作最大值 {max_action} 超出动作空间维度 {self.action_dim}")
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_actions.long())
                ratio = torch.exp(logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist.entropy().mean()
                values_pred = self.critic(b_global)
                critic_loss = F.smooth_l1_loss(values_pred.squeeze(-1), b_returns, beta=1.0)

                approx_kl_val = float((b_old_logp - logp).mean().item())
                clip_fraction_val = float((torch.abs(ratio - 1.0) > self.clip_ratio).float().mean().item())
                if returns_var.item() > 1e-8:
                    explained_variance_val = (
                        1.0 - torch.var(b_returns - values_pred.squeeze(-1)) / returns_var
                    ).detach().item()
                else:
                    explained_variance_val = 0.0
                policy_entropy_val = float(dist.entropy().mean().item())
                policy_entropy_count += 1

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                (actor_loss + critic_loss).backward()
                actor_grad_norm_val = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5).item())
                critic_grad_norm_val = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5).item())
                self.actor_optim.step()
                self.critic_optim.step()

                actor_loss_val = float(actor_loss.item())
                critic_loss_val = float(critic_loss.item())
                entropy_val = float(dist.entropy().mean().item())
            if approx_kl_val > self.target_kl:
                early_stop_by_kl = True
                break

        self.lr_update_count += 1
        self._update_learning_rates()
        self.buffer.clear()
        return {
            "actor_loss": actor_loss_val,
            "critic_loss": critic_loss_val,
            "entropy": entropy_val,
            "approx_kl": approx_kl_val,
            "clip_fraction": clip_fraction_val,
            "explained_variance": explained_variance_val,
            "actor_grad_norm": actor_grad_norm_val,
            "critic_grad_norm": critic_grad_norm_val,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "policy_entropy": policy_entropy_val if policy_entropy_count > 0 else entropy_val,
            "actual_update_epochs": actual_update_epochs,
            "early_stop_by_kl": float(early_stop_by_kl),
            "update_count": self.lr_update_count,
        }
