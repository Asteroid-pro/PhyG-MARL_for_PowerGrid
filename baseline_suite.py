from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import DEVICE, HIDDEN_DIM, NUM_AGENTS
from env.cpgrid_env import CPGridEnv
from graph.graph_encoder import GraphEncoder
from utils.seed import set_seed

METHODS = ("mappo", "ippo", "ppo", "gru_mappo", "gat_mappo")
SEEDS = (42, 43, 44)


def env_factory(steps: int) -> CPGridEnv:
    return CPGridEnv(
        max_steps=steps,
        attack_probability=0.35,
        attack_weights={"fdi": 0.45, "dos": 0.45, "breaker": 0.10},
        max_breaker_attacks=6,
        exclude_islanding_breakers=True,
        breaker_transient=True,
        breaker_duration_range=(10, 20),
        max_active_breakers=2,
        load_control_enabled=True,
        generator_control_enabled=True,
        enforce_q_lims=True,
    )


def global_state(env: CPGridEnv, node: torch.Tensor, steps: int) -> torch.Tensor:
    disconnected = torch.zeros(NUM_AGENTS)
    for line_id in env._disconnected_lines():
        line = env.topology_manager.lines[line_id]
        disconnected[line.src] = 1.0
        disconnected[line.dst] = 1.0
    breaker = env.attack_simulator.get_breaker_state()
    return torch.cat((
        node[:, :4].reshape(-1),
        disconnected,
        torch.tensor([breaker.get("active_breaker_count", 0) / 2.0, env.step_count / max(steps, 1)]),
    )).float()


class LocalActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))

    def forward(self, obs: torch.Tensor, **_) -> torch.Tensor:
        return self.net(obs)


class CentralActor(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, NUM_AGENTS * 3)

    def forward(self, state: torch.Tensor, **_) -> torch.Tensor:
        return self.head(self.encoder(state)).reshape(-1, NUM_AGENTS, 3)


class RecurrentActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3)

    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor | None = None, **_):
        encoded = torch.tanh(self.encoder(obs)).unsqueeze(1)
        output, hidden_state = self.gru(encoded, hidden_state)
        return self.head(output[:, -1]), hidden_state


class GATActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.node = nn.Linear(obs_dim, hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 3))

    def forward(self, obs: torch.Tensor, **_):
        node = torch.tanh(self.node(obs)).unsqueeze(0)
        attended, _ = self.attention(node, node, node)
        return self.head(torch.cat((node.squeeze(0), attended.squeeze(0)), dim=-1))


def build_method(method: str, obs_dim: int, state_dim: int):
    if method == "ppo":
        actor = CentralActor(state_dim).to(DEVICE)
        critic = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(), nn.Linear(128, 1)).to(DEVICE)
    elif method == "gru_mappo":
        actor = RecurrentActor(obs_dim).to(DEVICE)
        critic = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(), nn.Linear(128, 1)).to(DEVICE)
    elif method == "gat_mappo":
        actor = GATActor(obs_dim).to(DEVICE)
        critic = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(), nn.Linear(128, 1)).to(DEVICE)
    else:
        actor = LocalActor(obs_dim).to(DEVICE)
        critic_dim = obs_dim if method == "ippo" else state_dim
        critic = nn.Sequential(nn.Linear(critic_dim, 128), nn.Tanh(), nn.Linear(128, 1)).to(DEVICE)
    return actor, critic


def action_and_logprob(method, actor, observations, state, recurrent_state=None):
    obs = torch.stack([torch.as_tensor(observations[f"agent_{i}"], dtype=torch.float32, device=DEVICE) for i in range(NUM_AGENTS)])
    if method == "ppo":
        logits = actor(state.unsqueeze(0)).squeeze(0)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        return actions.cpu().numpy(), dist.log_prob(actions).sum(), recurrent_state
    if method == "gru_mappo":
        logits, recurrent_state = actor(obs, recurrent_state)
    else:
        logits = actor(obs)
    dist = Categorical(logits=logits)
    actions = dist.sample()
    return actions.cpu().numpy(), dist.log_prob(actions).sum(), recurrent_state


def critic_value(method, critic, observations, state):
    if method == "ippo":
        obs = torch.stack([torch.as_tensor(observations[f"agent_{i}"], dtype=torch.float32, device=DEVICE) for i in range(NUM_AGENTS)])
        return critic(obs).squeeze(-1).mean()
    return critic(state.unsqueeze(0)).squeeze()


def update(method, actor, critic, actor_opt, critic_opt, transitions, gamma=0.99):
    rewards = torch.tensor([x[2] for x in transitions], dtype=torch.float32, device=DEVICE) * 0.1
    dones = torch.tensor([x[3] for x in transitions], dtype=torch.float32, device=DEVICE)
    values = torch.stack([x[4] for x in transitions]).detach()
    returns = torch.zeros_like(rewards)
    running = torch.zeros((), device=DEVICE)
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running * (1.0 - dones[i])
        returns[i] = running
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    actor_loss = torch.zeros((), device=DEVICE)
    critic_loss = torch.zeros((), device=DEVICE)
    for transition, target, advantage in zip(transitions, returns, advantages):
        obs, state, action, old_logp = transition[0], transition[1], transition[5], transition[6]
        if method == "ppo":
            logits = actor(state.unsqueeze(0)).squeeze(0)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(action).sum()
        elif method == "gru_mappo":
            obs_tensor = torch.stack([torch.as_tensor(obs[f"agent_{i}"], dtype=torch.float32, device=DEVICE) for i in range(NUM_AGENTS)])
            logits, _ = actor(obs_tensor, None)
            logp = Categorical(logits=logits).log_prob(action).sum()
        else:
            obs_tensor = torch.stack([torch.as_tensor(obs[f"agent_{i}"], dtype=torch.float32, device=DEVICE) for i in range(NUM_AGENTS)])
            logits = actor(obs_tensor)
            logp = Categorical(logits=logits).log_prob(action).sum()
        ratio = torch.exp(logp - old_logp.detach())
        clipped = torch.clamp(ratio, 0.85, 1.15) * advantage
        actor_loss = actor_loss - torch.minimum(ratio * advantage, clipped) - 0.003 * Categorical(logits=logits).entropy().mean()
        current_value = critic_value(method, critic, obs, state)
        critic_loss = critic_loss + torch.nn.functional.smooth_l1_loss(current_value, target)
    actor_opt.zero_grad(); (actor_loss / len(transitions)).backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5); actor_opt.step()
    critic_opt.zero_grad(); (critic_loss / len(transitions)).backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5); critic_opt.step()


def train_one(method: str, seed: int, episodes: int, steps: int, output: Path):
    set_seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    env = env_factory(steps)
    obs_dim = env.observation_space["agent_0"].shape[0]
    state_dim = NUM_AGENTS * 4 + NUM_AGENTS + 2
    actor, critic = build_method(method, obs_dim, state_dim)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-4)
    rows = []
    for episode in range(episodes):
        observations, _ = env.reset(seed=seed + episode)
        recurrent_state = None
        transitions = []
        total_reward = 0.0
        loads, highs = [], []
        survived = 0
        for _ in range(steps):
            node, _, _, _ = env.get_graph_data()
            state = global_state(env, node, steps).to(DEVICE)
            value = critic_value(method, critic, observations, state)
            actions, logp, recurrent_state = action_and_logprob(method, actor, observations, state, recurrent_state)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            transitions.append((observations, state, float(reward), bool(terminated or truncated), value, torch.as_tensor(actions, dtype=torch.long, device=DEVICE), logp))
            observations = next_obs; total_reward += float(reward)
            metrics = info.get("powerflow_metrics")
            if metrics:
                loads.append(float(metrics["load_ratio"])); highs.append(float(metrics["high_voltage_rate"]))
            if terminated or truncated:
                survived = int(truncated and info.get("termination_reason") == "time_limit")
                break
        if transitions:
            update(method, actor, critic, actor_opt, critic_opt, transitions)
        rows.append({"seed": seed, "episode": episode, "raw_reward": total_reward, "survived_to_limit": survived, "mean_load_ratio": np.mean(loads) if loads else np.nan, "mean_high_voltage_rate": np.mean(highs) if highs else np.nan})
        if (episode + 1) % 25 == 0:
            print(f"{method} seed {seed}: episode {episode + 1}/{episodes}")
    pd.DataFrame(rows).to_csv(output / "training_metrics_advanced.csv", index=False)
    torch.save({"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(), "method": method, "seed": seed, "episode": episodes - 1}, output / "final.pt")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train five independent baseline families without modifying PhyG.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output_dir", type=Path, default=Path("baseline_comparison"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        for method in args.methods:
            for seed in args.seeds:
                train_one(method, seed, args.episodes, args.steps, args.output_dir / method / f"seed_{seed}")
    print(f"Baseline suite completed: {args.output_dir}")


if __name__ == "__main__":
    main()
