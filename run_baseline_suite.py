"""
独立 baseline 训练入口：在统一 CPGridEnv 协议下训练 PPO、IPPO、MAPPO、GRU-MAPPO、GAT-MAPPO、QMIX 和 MAT。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical

from baseline_config import BaselineConfig, SEEDS, TRAINABLE_METHODS
from baselines.common.checkpoint import save_checkpoint
from baselines.common.env_adapter import global_state, make_env, observation_tensor
from baselines.common.ppo_engine import IPPOEngine, OnPolicyPPO, Transition
from baselines.gat_mappo.gat_encoder import GATEncoder
from baselines.gru_mappo.gru_policy import GRUPolicy
from baselines.mappo.agent import CentralCritic, SharedActor
from baselines.mat.mat_policy import MATPolicy
from baselines.qmix.agent import Episode, QMIXRuntime
from baselines.common.agents import CentralPolicy
from config import DEVICE, NODE_IN_DIM, NUM_AGENTS
from utils.seed import set_seed


class MethodRuntime:
    def __init__(self, method: str, obs_dim: int, state_dim: int, config: BaselineConfig):
        self.method, self.config = method, config
        self.hidden = None
        self.qmix_episode = []
        if method == "qmix":
            self.qmix = QMIXRuntime(obs_dim, state_dim, config.hidden_dim)
            self.actor = self.qmix.agent_q
            self.critic = self.qmix.mixer
            self.engine = None
            return
        if method == "mat":
            self.actor = MATPolicy(obs_dim, config.hidden_dim).to(DEVICE)
            self.critic = self.actor.value_head
            self.engine = OnPolicyPPO(self.actor, self.critic, self.actor_forward, self.critic_forward, config, DEVICE)
            return
        if method == "ppo": 
            self.actor = CentralPolicy(state_dim, config.hidden_dim, NUM_AGENTS).to(DEVICE)
        elif method == "gru_mappo":
            self.actor = GRUPolicy(obs_dim, config.hidden_dim).to(DEVICE)
        elif method == "gat_mappo":
            self.encoder = GATEncoder(NODE_IN_DIM, config.hidden_dim).to(DEVICE)
            self.head = nn.Linear(config.hidden_dim * 2, 3).to(DEVICE)
            self.actor = nn.ModuleList((self.encoder, self.head))
        else:
            self.actor = SharedActor(obs_dim, config.hidden_dim).to(DEVICE)
        critic_dim = obs_dim if method == "ippo" else state_dim
        self.critic = CentralCritic(critic_dim, config.hidden_dim).to(DEVICE)
        engine_class = IPPOEngine if method == "ippo" else OnPolicyPPO
        self.engine = engine_class(self.actor, self.critic, self.actor_forward, self.critic_forward, config, DEVICE)

    def reset_episode(self):
        self.hidden = None
        if self.method == "qmix":
            self.qmix.reset_episode()
            self.qmix_episode = []

    def actor_forward(self, obs, state, graph_data=None, hidden_state=None):
        if self.method == "mat":
            logits, _, _, _, _ = self.actor(obs)
            return logits
        if self.method == "ppo":
            return self.actor(state.unsqueeze(0)).squeeze(0)
        if self.method == "gru_mappo":
            logits, _ = self.actor(obs, hidden_state)
            return logits
        if self.method == "gat_mappo":
            if graph_data is None:
                raise ValueError("GAT-MAPPO requires real graph data")
            node_feats, edge_feats, edge_index, injection = graph_data
            encoded = self.encoder(node_feats.to(DEVICE), edge_feats.to(DEVICE), edge_index.to(DEVICE), injection.to(DEVICE))
            pooled = encoded.mean(0, keepdim=True).expand(NUM_AGENTS, -1)
            return self.head(torch.cat((encoded, pooled), -1))
        return self.actor(obs)

    def critic_forward(self, obs, state, graph_data=None, hidden_state=None):
        if self.method == "mat":
            return self.actor(obs)[-1]
        if self.method == "ippo": 
            return self.critic(obs)
        return self.critic(state.unsqueeze(0)).squeeze()

    @torch.no_grad()
    def act(self, obs, state, graph_data=None, deterministic=False):
        if self.method == "qmix":
            epsilon = 0.0 if deterministic else 0.05
            actions = self.qmix.act(obs, state, epsilon)
            return actions, torch.zeros(NUM_AGENTS, device=DEVICE), torch.zeros((), device=DEVICE)
        if self.method == "mat":
            _, actions, log_prob, _, value = self.actor(obs, deterministic=deterministic)
            return actions, log_prob, value
        logits = self.actor_forward(obs, state, graph_data, self.hidden)
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits, -1) if deterministic else distribution.sample()
        if self.method == "ippo":
            value = self.critic_forward(obs, state, graph_data, self.hidden)
        else:
            value = self.critic_forward(obs, state, graph_data, self.hidden)
        return action, distribution.log_prob(action), value

    def checkpoint(self, method, seed, episode, config):
        payload = {"method": method, "seed": seed, "episode": episode, "actor_state_dict": self.actor.state_dict(),
                   "critic_state_dict": self.critic.state_dict(), "config": config.__dict__}
        return payload


def evaluate(runtime: MethodRuntime, episodes: int, steps: int, seed_base: int):
    env = make_env(steps); rows = []
    with torch.no_grad():
        for episode in range(episodes):
            observations, _ = env.reset(seed=seed_base + episode); runtime.reset_episode()
            total, loads, highs = 0.0, [], []
            for _ in range(steps):
                node, edge, edge_index, injection = env.get_graph_data(); state = global_state(env, node, steps).to(DEVICE)
                obs = observation_tensor(observations, DEVICE)
                graph_data = (node, edge, edge_index, injection)
                action, _, _ = runtime.act(obs, state, graph_data, True)
                observations, reward, terminated, truncated, info = env.step(action.cpu().numpy()); total += reward
                metrics = info.get("powerflow_metrics")
                if metrics: loads.append(metrics["load_ratio"]); highs.append(metrics["high_voltage_rate"])
                if terminated or truncated: break
            rows.append((int(truncated and info.get("termination_reason") == "time_limit"), total, np.mean(loads), np.mean(highs)))
    env.close(); values = np.asarray(rows)
    return {"eval_survival_rate": values[:, 0].mean(), "eval_mean_reward": values[:, 1].mean(),
            "eval_mean_load_ratio": values[:, 2].mean(), "eval_high_voltage_rate": values[:, 3].mean()}


def train(method: str, seed: int, config: BaselineConfig, output: Path):
    set_seed(seed); output.mkdir(parents=True, exist_ok=True); env = make_env(config.steps)
    obs_dim = env.observation_space["agent_0"].shape[0]; state_dim = NUM_AGENTS * 4 + NUM_AGENTS + 2
    runtime = MethodRuntime(method, obs_dim, state_dim, config)
    rows, eval_rows = [], []; best_balanced = best_reward = -float("inf"); best_survival = -float("inf")
    for episode in range(config.episodes):
        observations, _ = env.reset(seed=seed + episode); runtime.reset_episode(); total = 0.0; loads, highs = [], []
        terminated = truncated = False
        for _ in range(config.steps):
            node, edge, edge_index, injection = env.get_graph_data(); state = global_state(env, node, config.steps).to(DEVICE)
            obs = observation_tensor(observations, DEVICE); graph_data = (node, edge, edge_index, injection)
            action, log_prob, value = runtime.act(obs, state, graph_data)
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            if method != "qmix":
                runtime.engine.collect(Transition(obs.detach(), state.detach(), action.detach(), reward, terminated or truncated, log_prob.detach(), value.detach(), graph_data=tuple(item.detach() if torch.is_tensor(item) else item for item in graph_data)))
            else:
                runtime.qmix_episode.append((obs.detach(), state.detach(), action.detach(), float(reward) * 0.1, float(terminated or truncated)))
            observations = next_obs; total += reward
            metrics = info.get("powerflow_metrics")
            if metrics: loads.append(metrics["load_ratio"]); highs.append(metrics["high_voltage_rate"])
            if method != "qmix" and len(runtime.engine.buffer) >= config.rollout_steps: runtime.engine.update()
            if terminated or truncated: break
        survived = int(truncated and info.get("termination_reason") == "time_limit")
        rows.append({"seed": seed, "episode": episode, "raw_reward": total, "survived_to_limit": survived,
                     "mean_load_ratio": np.mean(loads), "mean_high_voltage_rate": np.mean(highs)})
        if (episode + 1) % config.evaluation_interval == 0:
            metrics = evaluate(runtime, 20, config.steps, seed + 50000)
            balanced = metrics["eval_survival_rate"] + .5 * metrics["eval_mean_load_ratio"] - .5 * metrics["eval_high_voltage_rate"]
            record = {"episode": episode, **metrics, "balanced_score": balanced}; eval_rows.append(record)
            payload = runtime.checkpoint(method, seed, episode, config); save_checkpoint(output / "latest.pt", payload)
            if balanced > best_balanced: best_balanced = balanced; save_checkpoint(output / "best_balanced.pt", payload)
            if metrics["eval_mean_reward"] > best_reward: best_reward = metrics["eval_mean_reward"]; save_checkpoint(output / "best_reward.pt", payload)
            if metrics["eval_survival_rate"] > best_survival: best_survival = metrics["eval_survival_rate"]; save_checkpoint(output / "best_survival.pt", payload)
            print(f"{method} seed={seed} episode={episode + 1}: reward={metrics['eval_mean_reward']:.2f}")
        if method == "qmix" and runtime.qmix_episode:
            episode_data = list(zip(*runtime.qmix_episode))
            runtime.qmix.add_episode(Episode(
                observations=torch.stack(episode_data[0]),
                states=torch.stack(episode_data[1]),
                actions=torch.stack(episode_data[2]),
                rewards=torch.tensor(episode_data[3], dtype=torch.float32),
                dones=torch.tensor(episode_data[4], dtype=torch.float32),
            ))
            runtime.qmix.update(batch_size=1, gamma=config.gamma)
            runtime.qmix_episode = []
    if method != "qmix" and runtime.engine.buffer: runtime.engine.update()
    save_checkpoint(output / "final.pt", runtime.checkpoint(method, seed, config.episodes - 1, config))
    pd.DataFrame(rows).to_csv(output / "training_metrics_advanced.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(output / "evaluation_metrics.csv", index=False)
    with (output / "experiment_config.json").open("w") as file: json.dump({"method": method, "seed": seed, **config.__dict__}, file, indent=2)
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Run five paper-grade baseline methods using a common protocol.")
    parser.add_argument("--methods", nargs="+", choices=TRAINABLE_METHODS, default=list(TRAINABLE_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS)); parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200); parser.add_argument("--output_dir", type=Path, default=Path("baseline_comparison"))
    args = parser.parse_args(); config = BaselineConfig(episodes=args.episodes, steps=args.steps)
    for method in args.methods:
        for seed in args.seeds: train(method, seed, config, args.output_dir / method / f"seed_{seed}")


if __name__ == "__main__": main()
