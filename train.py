"""
PhyG 主训练入口：在 IEEE14 电网攻防环境中训练物理感知 MAPPO 策略。
包括环境交互、GraphEncoder/TemporalMemory 表示、Actor-Critic 更新、评估和 checkpoint 保存。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

from adapter.light_mappo import LightMAPPO
from config import (
    ACTION_DIM,
    ACTOR_HIDDEN_DIM,
    DEVICE,
    EDGE_IN_DIM,
    GRAPH_NUM_LAYERS,
    HIDDEN_DIM,
    HISTORY_LEN,
    MAPPO_BATCH_SIZE,
    MAPPO_CLIP_RATIO,
    MAPPO_ENTROPY_COEF,
    MAPPO_EPOCHS,
    MAPPO_GAMMA,
    MAPPO_LAM,
    MAPPO_LR,
    MAX_EPISODES,
    MEMORY_HISTORY_LEN,
    NODE_IN_DIM,
    NUM_AGENTS,
    OBS_DIM,
    OUTPUT_DIR,
    SEED,
    STEPS_PER_EPISODE,
)
from env.cpgrid_env import CPGridEnv
from graph.graph_encoder import GraphEncoder
from memory.transformer_memory import TemporalMemory
from policy.actor import GraphActor
from utils.metrics_logger import EpisodeMetricsLogger
from utils.seed import set_seed


def flatten_obs(obs):
    if torch.is_tensor(obs):
        return obs.flatten().float()
    return torch.as_tensor(obs, dtype=torch.float32).flatten()


def build_critic_state(env, node_feats, steps, device):
    disconnected_mask = torch.zeros(NUM_AGENTS, dtype=torch.float32, device=device)
    for line_id in env._disconnected_lines():
        line = env.topology_manager.lines[line_id]
        disconnected_mask[line.src] = 1.0
        disconnected_mask[line.dst] = 1.0
    breaker_state = env.attack_simulator.get_breaker_state()
    return torch.cat([
        node_feats[:, :4].to(device).reshape(-1),
        disconnected_mask,
        torch.tensor([
            float(breaker_state.get("active_breaker_count", 0)) / 2.0,
            float(env.step_count) / max(float(steps), 1.0),
        ], dtype=torch.float32, device=device),
    ])


def save_checkpoint(path, actor, critic, graph_encoder, memory, mappo, episode, config):
    payload = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "graph_encoder_state_dict": graph_encoder.state_dict(),
        "memory_state_dict": memory.state_dict(),
        "actor_optimizer_state_dict": mappo.actor_optim.state_dict(),
        "critic_optimizer_state_dict": mappo.critic_optim.state_dict(),
        "update_count": mappo.lr_update_count,
        "episode": episode,
        "memory_gate": float(torch.sigmoid(actor.memory_gate_logit).item()) if getattr(actor, "use_memory_gate", False) else 1.0,
        "graph_trainable": int(getattr(mappo, "graph_trainable", False)),
        "memory_trainable": int(getattr(mappo, "memory_trainable", False)),
        "return_normalization_state": {
            "mean": float(getattr(mappo, "return_mean", 0.0)),
            "var": float(getattr(mappo, "return_var", 1.0)),
            "count": float(getattr(mappo, "return_count", 0.0)),
        },
        "config": config,
    }
    if mappo.representation_optim is not None:
        payload["representation_optimizer_state_dict"] = mappo.representation_optim.state_dict()
    torch.save(payload, path)


def evaluate_deterministic(actor, graph_encoder, memory, eval_episodes, steps, seed_base):
    eval_env = CPGridEnv(max_steps=steps, max_breaker_attacks=6, breaker_transient=True, breaker_duration_range=(10, 20), max_active_breakers=2, load_control_enabled=True, generator_control_enabled=True, enforce_q_lims=True)
    eval_memory = TemporalMemory(hidden_dim=HIDDEN_DIM, history_length=MEMORY_HISTORY_LEN).to(DEVICE)
    eval_memory.load_state_dict(memory.state_dict())
    actor_was_training = actor.training
    graph_was_training = graph_encoder.training
    memory_was_training = memory.training
    actor.eval()
    graph_encoder.eval()
    eval_memory.eval()
    records = []
    with torch.no_grad():
        for index in range(eval_episodes):
            observations, _ = eval_env.reset(seed=seed_base + index)
            eval_memory.clear()
            total_reward = 0.0
            load_ratios, high_rates = [], []
            terminated = truncated = False
            episode_steps = 0
            while not (terminated or truncated) and episode_steps < steps:
                node, edge, edge_index, injection = eval_env.get_graph_data()
                _, embedding = graph_encoder(node.to(DEVICE), edge.to(DEVICE), edge_index.to(DEVICE), torch.as_tensor(injection, dtype=torch.float32, device=DEVICE))
                eval_memory.update(embedding)
                memory_vector = eval_memory.get_memory().to(DEVICE)
                obs_list = [torch.as_tensor(observations[f"agent_{i}"], dtype=torch.float32, device=DEVICE) for i in range(NUM_AGENTS)]
                logits = actor(torch.stack(obs_list), memory_vector)
                actions = torch.argmax(logits, dim=-1).cpu().numpy()
                observations, reward, terminated, truncated, info = eval_env.step(actions)
                total_reward += float(reward)
                episode_steps += 1
                metrics = info.get("powerflow_metrics")
                if metrics is not None:
                    load_ratios.append(float(metrics["load_ratio"]))
                    high_rates.append(float(metrics["high_voltage_rate"]))
            records.append((int(truncated and info.get("termination_reason") == "time_limit"), episode_steps, total_reward, np.mean(load_ratios) if load_ratios else np.nan, np.mean(high_rates) if high_rates else np.nan))
    eval_env.close()
    actor.train(actor_was_training)
    graph_encoder.train(graph_was_training)
    memory.train(memory_was_training)
    values = np.asarray(records, dtype=float)
    return {"eval_survival_rate": float(np.mean(values[:, 0])), "eval_mean_steps": float(np.mean(values[:, 1])), "eval_mean_reward": float(np.mean(values[:, 2])), "eval_mean_load_ratio": float(np.nanmean(values[:, 3])), "eval_high_voltage_rate": float(np.nanmean(values[:, 4]))}


def _save_experiment_config(output_dir: Path, seed_value, episodes: int, steps: int, success_threshold: float):
    payload = {
        "seed": seed_value,
        "episodes": episodes,
        "steps_per_episode": steps,
        "success_threshold": success_threshold,
        "note": "All robustness runs use identical settings except for the random seed.",
    }
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main(
    episodes=MAX_EPISODES,
    steps=STEPS_PER_EPISODE,
    success_threshold: float = 0.0,
    seed=SEED,
    output_dir=OUTPUT_DIR,
    representation_mode: str = "frozen",
    representation_lr: float = 5e-6,
    sequence_length: int = 32,
    actor_lr: float = 5e-5,
    critic_lr: float = 5e-5,
    memory_unfreeze_episode: int = 100,
    graph_unfreeze_episode: int = 200,
    entropy_coef: float = 0.005,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        set_seed(seed)
        print("=================================")
        print(f"Random Seed : {seed}")
        print("=================================")
    else:
        print("=================================")
        print("Random Seed : Random")
        print("=================================")
    _save_experiment_config(output_dir, seed, episodes, steps, success_threshold)
    env = CPGridEnv(
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
    graph_encoder = GraphEncoder(node_in_dim=NODE_IN_DIM, edge_in_dim=EDGE_IN_DIM, hidden_dim=HIDDEN_DIM, num_layers=GRAPH_NUM_LAYERS).to(DEVICE)
    memory = TemporalMemory(hidden_dim=HIDDEN_DIM, history_length=MEMORY_HISTORY_LEN).to(DEVICE)
    sample_key = list(env.observation_space.keys())[0]
    obs_dim = env.observation_space[sample_key].shape[0]
    action_dim = int(env.action_space.nvec[0])
    assert action_dim == ACTION_DIM
    actor = GraphActor(
        obs_dim,
        HIDDEN_DIM,
        action_dim,
        hidden_dim=ACTOR_HIDDEN_DIM,
        use_memory_gate=representation_mode == "joint",
    ).to(DEVICE)
    critic_state_dim = NUM_AGENTS * 4 + NUM_AGENTS + 2
    critic = torch.nn.Sequential(torch.nn.Linear(critic_state_dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(DEVICE)
    mappo = LightMAPPO(
        actor,
        critic,
        lr=actor_lr,
        gamma=MAPPO_GAMMA,
        lam=MAPPO_LAM,
        clip_ratio=MAPPO_CLIP_RATIO,
        entropy_coef=entropy_coef,
        epochs=5,
        batch_size=MAPPO_BATCH_SIZE,
        target_kl=0.02,
        graph_encoder=graph_encoder,
        memory=memory,
        representation_mode=representation_mode,
        representation_lr=representation_lr,
        sequence_length=sequence_length,
        critic_lr=critic_lr,
        value_clip=0.2,
        critic_epochs=5,
        normalize_returns=False,
    )
    if representation_mode == "joint":
        mappo.set_representation_trainable(False, False)

    metrics_logger = EpisodeMetricsLogger(success_threshold=success_threshold)
    csv_path = output_dir / "training_metrics_advanced.csv"
    ppo_csv_path = output_dir / "ppo_update_metrics.csv"
    eval_csv_path = output_dir / "evaluation_metrics.csv"
    eval_fields = ["episode", "completed_episodes", "update_count", "environment_steps", "eval_survival_rate", "eval_mean_steps", "eval_mean_reward", "eval_mean_load_ratio", "eval_high_voltage_rate", "balanced_score", "is_best_balanced", "is_best_survival", "is_best_reward"]
    ppo_fields = ["update_count", "episode", "environment_steps", "actor_loss", "critic_loss", "value_pred_mean", "value_pred_std", "value_target_mean", "value_target_std", "value_error_mean", "value_error_abs_mean", "value_clip_fraction", "explained_variance", "approx_kl", "clip_fraction", "actor_grad_norm", "critic_grad_norm", "graph_grad_norm", "memory_grad_norm", "graph_parameter_delta", "memory_parameter_delta", "graph_parameter_relative_delta", "memory_parameter_relative_delta", "memory_gate", "memory_gate_logit", "graph_trainable", "memory_trainable", "adv_mean", "adv_std", "policy_entropy", "actual_update_epochs", "early_stop_by_kl", "actor_lr", "critic_lr", "representation_lr"]
    csv_fields = [
        "seed",
        "episode",
        "raw_reward",
        "avg_50",
        "avg_100",
        "std_50",
        "success_rate_50",
        "total_reward",
        "steps",
        "steps_avg_50",
        "steps_std_50",
        "survived_to_limit",
        "survival_rate_50",
        "termination_reason",
        "fdi_count",
        "dos_count",
        "breaker_count",
        "disconnected_line_count",
        "restored_line_count",
        "max_concurrent_breakers",
        "breaker_active_steps",
        "steps_under_breaker",
        "mean_breaker_duration",
        "active_breaker_count_at_end",
        "mean_load_ratio",
        "mean_low_voltage_rate",
        "mean_high_voltage_rate",
        "mean_voltage_violation_rate",
        "max_line_loading_percent",
        "line_overload_rate",
        "line_overload_excess_mean",
        "action_0_ratio",
        "action_1_ratio",
        "action_2_ratio",
        "breaker_action_0_ratio",
        "breaker_action_1_ratio",
        "breaker_action_2_ratio",
        "normal_action_0_ratio",
        "normal_action_1_ratio",
        "normal_action_2_ratio",
        "load_bus_action_0_ratio",
        "load_bus_action_1_ratio",
        "load_bus_action_2_ratio",
        "generator_bus_action_0_ratio",
        "generator_bus_action_1_ratio",
        "generator_bus_action_2_ratio",
        "alive_reward_sum",
        "load_reward_sum",
        "voltage_penalty_sum",
        "load_shedding_cost_sum",
        "generator_support_cost_sum",
        "line_overload_penalty_sum",
        "terminal_penalty_sum",
        "did_update",
        "actor_loss",
        "critic_loss",
        "explained_variance",
        "approx_kl",
        "clip_fraction",
        "actor_grad_norm",
        "critic_grad_norm",
        "graph_grad_norm",
        "memory_grad_norm",
        "graph_parameter_delta",
        "memory_parameter_delta",
        "graph_parameter_relative_delta",
        "memory_parameter_relative_delta",
        "memory_gate",
        "memory_gate_logit",
        "graph_trainable",
        "memory_trainable",
        "value_pred_mean",
        "value_target_mean",
        "value_error_abs_mean",
        "value_clip_fraction",
        "adv_mean",
        "adv_std",
        "policy_entropy",
        "actual_update_epochs",
        "early_stop_by_kl",
        "update_count",
        "actor_lr",
        "critic_lr",
    ]

    history = []
    advanced_history = []

    total_environment_steps = 0
    best_balanced_score = float("-inf")
    best_reward = float("-inf")
    best_survival_key = (float("-inf"), float("-inf"), float("-inf"))
    run_config = {
        "version": "V3.1", "seed": seed, "episodes": episodes, "steps_per_episode": steps,
        "rollout_steps": 1024, "reward_scale": 0.1, "actor_lr": actor_lr, "critic_lr": critic_lr,
        "ppo_epochs": 5, "target_kl": 0.02, "max_grad_norm": 0.5, "entropy_coef": entropy_coef,
        "attack_probability": 0.35, "attack_weights": {"fdi": 0.45, "dos": 0.45, "breaker": 0.10},
        "breaker_transient": True, "breaker_duration_range": [10, 20], "max_active_breakers": 2,
        "max_breaker_attacks": 6, "exclude_islanding_breakers": True, "enforce_q_lims": True,
        "load_control_enabled": True, "generator_control_enabled": True,
        "terminal_penalties": {"powerflow_not_converged": -10.0, "invalid_voltage": -10.0, "voltage_collapse": -15.0},
        "line_overload_weight": 0.5, "observation_dim": obs_dim, "action_dim": action_dim,
        "critic_state_dim": critic_state_dim,
        "representation_mode": representation_mode,
        "representation_lr": representation_lr,
        "sequence_length": sequence_length,
        "transformer_dropout": 0.0,
        "memory_gate_initial": float(torch.sigmoid(actor.memory_gate_logit).item()) if actor.use_memory_gate else 1.0,
        "memory_unfreeze_episode": memory_unfreeze_episode,
        "graph_unfreeze_episode": graph_unfreeze_episode,
        "value_clip": 0.2,
        "normalize_returns": False,
        "critic_epochs": 5,
        "evaluation_interval_episodes": 25,
        "entropy_coef": entropy_coef,
        "checkpoint_selection": {
            "best_balanced": "survival + 0.5 * load_ratio - 0.5 * high_voltage_rate",
            "best_survival": "lexicographic: survival, load_ratio, negative high_voltage_rate",
            "best_reward": "maximum eval_mean_reward",
        },
        "evaluation_interval_episodes": 25,
    }
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as config_f:
        json.dump(run_config, config_f, ensure_ascii=False, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as f, ppo_csv_path.open("w", newline="", encoding="utf-8") as ppo_f, eval_csv_path.open("w", newline="", encoding="utf-8") as eval_f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        ppo_writer = csv.DictWriter(ppo_f, fieldnames=ppo_fields)
        eval_writer = csv.DictWriter(eval_f, fieldnames=eval_fields)
        writer.writeheader()
        ppo_writer.writeheader()
        eval_writer.writeheader()

        for episode in range(episodes):
            if representation_mode == "joint":
                mappo.set_representation_trainable(
                    graph_trainable=episode >= graph_unfreeze_episode,
                    memory_trainable=episode >= memory_unfreeze_episode,
                )
            obs_dict, info = env.reset(seed=seed if episode == 0 else None)
            memory.clear()
            total_reward = 0.0
            done = False
            truncated = False
            episode_step = 0
            physical_values = {name: [] for name in ("load_ratio", "low_voltage_rate", "high_voltage_rate", "voltage_violation_rate", "max_line_loading_percent", "line_overload_rate", "line_overload_excess_mean")}
            reward_sums = {name: 0.0 for name in ("alive_reward", "load_reward", "voltage_penalty", "load_shedding_cost", "generator_support_cost", "line_overload_penalty", "terminal_penalty")}
            did_update = 0
            action_counts = np.zeros(3, dtype=np.int64)
            breaker_action_counts = np.zeros(3, dtype=np.int64)
            normal_action_counts = np.zeros(3, dtype=np.int64)
            load_bus_action_counts = np.zeros(3, dtype=np.int64)
            generator_bus_action_counts = np.zeros(3, dtype=np.int64)
            load_bus_indices = np.asarray(sorted(env.powerflow_solver.bus_load_indices), dtype=int)
            generator_bus_indices = np.asarray(sorted(env.powerflow_solver.generator_buses | env.powerflow_solver.slack_buses), dtype=int)
            last_graph_embed = torch.zeros(HIDDEN_DIM, device=DEVICE)
            last_stats = {
                "actor_loss": np.nan,
                "critic_loss": np.nan,
                "entropy": np.nan,
                "approx_kl": np.nan,
                "clip_fraction": np.nan,
                "explained_variance": np.nan,
                "actor_grad_norm": np.nan,
                "critic_grad_norm": np.nan,
                "graph_grad_norm": np.nan,
                "memory_grad_norm": np.nan,
                "graph_parameter_delta": np.nan,
                "memory_parameter_delta": np.nan,
                "graph_parameter_relative_delta": np.nan,
                "memory_parameter_relative_delta": np.nan,
                "value_pred_mean": np.nan,
                "value_target_mean": np.nan,
                "value_error_abs_mean": np.nan,
                "value_clip_fraction": np.nan,
                "adv_mean": np.nan,
                "adv_std": np.nan,
                "policy_entropy": np.nan,
                "actual_update_epochs": np.nan,
                "early_stop_by_kl": np.nan,
            }
            current_lrs = mappo.get_current_lrs()

            while not (done or truncated) and episode_step < steps:
                node_feats, edge_feats, edge_index, node_injection = env.get_graph_data()
                if node_feats is None:
                    break
                node_feats = node_feats.to(DEVICE)
                edge_feats = edge_feats.to(DEVICE)
                edge_index = edge_index.to(DEVICE)
                node_injection = torch.as_tensor(node_injection, dtype=torch.float32, device=DEVICE)
                _, graph_embed = graph_encoder(node_feats, edge_feats, edge_index, node_injection)
                memory.update(graph_embed)
                mem_vec = memory.get_memory().to(DEVICE)
                last_graph_embed = graph_embed.detach()

                obs_list = []
                for i in range(NUM_AGENTS):
                    obs = obs_dict[f"agent_{i}"]
                    if not torch.is_tensor(obs):
                        obs = torch.tensor(obs, dtype=torch.float32)
                    obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).to(device=DEVICE, dtype=torch.float32)
                    assert torch.isfinite(obs).all(), "obs contains NaN/Inf"
                    assert obs.shape[0] == obs_dim, f"obs dim {obs.shape[0]} != obs_dim {obs_dim}"
                    obs_list.append(obs)

                actions, log_probs, _ = mappo.get_actions(obs_list, mem_vec)
                next_obs, reward, terminated, truncated, info = env.step(actions)
                action_array = np.asarray(actions, dtype=int)
                action_counts += np.bincount(action_array, minlength=3)
                breaker_state_for_actions = info.get("breaker_state", {}) if isinstance(info, dict) else {}
                if int(breaker_state_for_actions.get("active_breaker_count", 0)) > 0:
                    breaker_action_counts += np.bincount(action_array, minlength=3)
                else:
                    normal_action_counts += np.bincount(action_array, minlength=3)
                load_bus_action_counts += np.bincount(action_array[load_bus_indices], minlength=3)
                generator_bus_action_counts += np.bincount(action_array[generator_bus_indices], minlength=3)
                pf_metrics = info.get("powerflow_metrics") if isinstance(info, dict) else None
                if pf_metrics is not None:
                    for name in physical_values:
                        physical_values[name].append(float(pf_metrics[name]))
                reward_components = info.get("reward_components", {}) if isinstance(info, dict) else {}
                for name in reward_sums:
                    reward_sums[name] += float(reward_components.get(name, 0.0))
                if not np.isfinite(reward):
                    reward = -5.0
                    terminated = True
                global_state = build_critic_state(env, node_feats, steps, DEVICE)
                normalized_value = critic(global_state.unsqueeze(0))
                value = float(mappo.denormalize_value(normalized_value).item())
                values = [value for _ in range(NUM_AGENTS)]
                mappo.collect_experience(
                    obs_list,
                    torch.tensor(actions, dtype=torch.long),
                    [reward * 0.1] * NUM_AGENTS,
                    [terminated or truncated] * NUM_AGENTS,
                    values,
                    log_probs.tolist(),
                    mem_vec.detach(),
                    global_state.detach(),
                    graph_data=(node_feats, edge_feats, edge_index, node_injection) if representation_mode == "joint" else None,
                    episode_start=episode_step == 0,
                )
                total_reward += reward
                episode_step += 1
                total_environment_steps += 1
                obs_dict = next_obs
                done = terminated

                if len(mappo.buffer) >= 1024:
                    bootstrap_value = 0.0
                    if not (terminated or truncated):
                        next_node_feats, _, _, _ = env.get_graph_data()
                        next_state = build_critic_state(env, next_node_feats, steps, DEVICE)
                        bootstrap_value = float(mappo.denormalize_value(critic(next_state.unsqueeze(0))).item())
                    last_stats = mappo.update(bootstrap_value=bootstrap_value)
                    did_update = 1
                    current_lrs = mappo.get_current_lrs()
                    ppo_row = {
                        "update_count": last_stats["update_count"], "episode": episode,
                        "environment_steps": total_environment_steps,
                        **{name: last_stats[name] for name in ppo_fields if name in last_stats},
                        **current_lrs,
                    }
                    ppo_writer.writerow({name: ppo_row.get(name, "") for name in ppo_fields})
                    ppo_f.flush()
                    print(f"Episode {episode} rollout update: {last_stats}, LRs: {current_lrs}")

            current_lrs = mappo.get_current_lrs()

            termination_reason = info.get("termination_reason") if isinstance(info, dict) else None
            attack_counts = info.get("attack_counts", {}) if isinstance(info, dict) else {}
            fdi_count = int(attack_counts.get("fdi", 0))
            dos_count = int(attack_counts.get("dos", 0))
            breaker_count = int(attack_counts.get("breaker", 0))
            disconnected_line_count = int(info.get("disconnected_line_count", 0)) if isinstance(info, dict) else 0
            breaker_state = info.get("breaker_state", {}) if isinstance(info, dict) else {}
            restored_line_count = int(breaker_state.get("restored_line_count", 0))
            max_concurrent_breakers = int(breaker_state.get("max_concurrent_breakers", 0))
            breaker_active_steps = int(breaker_state.get("breaker_active_steps", 0))
            steps_under_breaker = int(breaker_state.get("steps_under_breaker", 0))
            mean_breaker_duration = float(breaker_state.get("mean_breaker_duration", 0.0))
            active_breaker_count_at_end = int(breaker_state.get("active_breaker_count", 0))
            physical_means = {name: float(np.mean(values)) if values else float("nan") for name, values in physical_values.items()}
            action_total = max(int(action_counts.sum()), 1)
            action_ratios = action_counts / action_total
            def ratios_or_nan(counts):
                total = int(counts.sum())
                return counts / total if total else np.full(3, np.nan)
            breaker_action_ratios = ratios_or_nan(breaker_action_counts)
            normal_action_ratios = ratios_or_nan(normal_action_counts)
            load_bus_action_ratios = ratios_or_nan(load_bus_action_counts)
            generator_bus_action_ratios = ratios_or_nan(generator_bus_action_counts)
            survived_to_limit = bool(truncated and termination_reason == "time_limit")
            metrics_logger.update(total_reward, episode_step, survived_to_limit)
            (
                raw_reward,
                avg_50,
                avg_100,
                std_50,
                success_rate_50,
                steps_avg_50,
                steps_std_50,
                survival_rate_50,
            ) = metrics_logger.summary()
            row = {
                "seed": seed,
                "episode": episode,
                "raw_reward": float(raw_reward),
                "avg_50": float(avg_50),
                "avg_100": float(avg_100),
                "std_50": float(std_50),
                "success_rate_50": float(success_rate_50),
                "total_reward": float(total_reward),
                "steps": int(episode_step),
                "steps_avg_50": float(steps_avg_50),
                "steps_std_50": float(steps_std_50),
                "survived_to_limit": int(survived_to_limit),
                "survival_rate_50": float(survival_rate_50),
                "termination_reason": termination_reason or "unknown",
                "fdi_count": fdi_count,
                "dos_count": dos_count,
                "breaker_count": breaker_count,
                "disconnected_line_count": disconnected_line_count,
                "restored_line_count": restored_line_count,
                "max_concurrent_breakers": max_concurrent_breakers,
                "breaker_active_steps": breaker_active_steps,
                "steps_under_breaker": steps_under_breaker,
                "mean_breaker_duration": mean_breaker_duration,
                "active_breaker_count_at_end": active_breaker_count_at_end,
                "mean_load_ratio": physical_means["load_ratio"],
                "mean_low_voltage_rate": physical_means["low_voltage_rate"],
                "mean_high_voltage_rate": physical_means["high_voltage_rate"],
                "mean_voltage_violation_rate": physical_means["voltage_violation_rate"],
                "max_line_loading_percent": physical_means["max_line_loading_percent"],
                "line_overload_rate": physical_means["line_overload_rate"],
                "line_overload_excess_mean": physical_means["line_overload_excess_mean"],
                "action_0_ratio": float(action_ratios[0]),
                "action_1_ratio": float(action_ratios[1]),
                "action_2_ratio": float(action_ratios[2]),
                "breaker_action_0_ratio": float(breaker_action_ratios[0]),
                "breaker_action_1_ratio": float(breaker_action_ratios[1]),
                "breaker_action_2_ratio": float(breaker_action_ratios[2]),
                "normal_action_0_ratio": float(normal_action_ratios[0]),
                "normal_action_1_ratio": float(normal_action_ratios[1]),
                "normal_action_2_ratio": float(normal_action_ratios[2]),
                "load_bus_action_0_ratio": float(load_bus_action_ratios[0]),
                "load_bus_action_1_ratio": float(load_bus_action_ratios[1]),
                "load_bus_action_2_ratio": float(load_bus_action_ratios[2]),
                "generator_bus_action_0_ratio": float(generator_bus_action_ratios[0]),
                "generator_bus_action_1_ratio": float(generator_bus_action_ratios[1]),
                "generator_bus_action_2_ratio": float(generator_bus_action_ratios[2]),
                "alive_reward_sum": reward_sums["alive_reward"],
                "load_reward_sum": reward_sums["load_reward"],
                "voltage_penalty_sum": reward_sums["voltage_penalty"],
                "load_shedding_cost_sum": reward_sums["load_shedding_cost"],
                "generator_support_cost_sum": reward_sums["generator_support_cost"],
                "line_overload_penalty_sum": reward_sums["line_overload_penalty"],
                "terminal_penalty_sum": reward_sums["terminal_penalty"],
                "did_update": did_update,
                "actor_loss": float(last_stats["actor_loss"]),
                "critic_loss": float(last_stats["critic_loss"]),
                "explained_variance": float(last_stats.get("explained_variance", 0.0)),
                "approx_kl": float(last_stats.get("approx_kl", 0.0)),
                "clip_fraction": float(last_stats.get("clip_fraction", 0.0)),
                "actor_grad_norm": float(last_stats.get("actor_grad_norm", 0.0)),
                "critic_grad_norm": float(last_stats.get("critic_grad_norm", 0.0)),
                "graph_grad_norm": float(last_stats.get("graph_grad_norm", 0.0)),
                "memory_grad_norm": float(last_stats.get("memory_grad_norm", 0.0)),
                "graph_parameter_delta": float(last_stats.get("graph_parameter_delta", 0.0)),
                "memory_parameter_delta": float(last_stats.get("memory_parameter_delta", 0.0)),
                "graph_parameter_relative_delta": float(last_stats.get("graph_parameter_relative_delta", 0.0)),
                "memory_parameter_relative_delta": float(last_stats.get("memory_parameter_relative_delta", 0.0)),
                "memory_gate": float(torch.sigmoid(actor.memory_gate_logit).item()) if actor.use_memory_gate else 1.0,
                "memory_gate_logit": float(actor.memory_gate_logit.item()) if actor.use_memory_gate else 0.0,
                "graph_trainable": int(mappo.graph_trainable),
                "memory_trainable": int(mappo.memory_trainable),
                "value_pred_mean": float(last_stats.get("value_pred_mean", 0.0)),
                "value_target_mean": float(last_stats.get("value_target_mean", 0.0)),
                "value_error_abs_mean": float(last_stats.get("value_error_abs_mean", 0.0)),
                "value_clip_fraction": float(last_stats.get("value_clip_fraction", 0.0)),
                "adv_mean": float(last_stats.get("adv_mean", 0.0)),
                "adv_std": float(last_stats.get("adv_std", 0.0)),
                "policy_entropy": float(last_stats.get("policy_entropy", 0.0)),
                "actual_update_epochs": last_stats.get("actual_update_epochs", np.nan),
                "early_stop_by_kl": float(last_stats.get("early_stop_by_kl", np.nan)),
                "update_count": int(last_stats.get("update_count", mappo.lr_update_count)),
                "actor_lr": float(current_lrs["actor_lr"]),
                "critic_lr": float(current_lrs["critic_lr"]),
            }
            writer.writerow(row)
            f.flush()
            if (episode + 1) % 25 == 0:
                eval_metrics = evaluate_deterministic(actor, graph_encoder, memory, 20, steps, (seed or 0) + 50000)
                balanced_score = eval_metrics["eval_survival_rate"] + 0.5 * eval_metrics["eval_mean_load_ratio"] - 0.5 * eval_metrics["eval_high_voltage_rate"]
                survival_key = (eval_metrics["eval_survival_rate"], eval_metrics["eval_mean_load_ratio"], -eval_metrics["eval_high_voltage_rate"])
                is_best_balanced = int(balanced_score > best_balanced_score)
                is_best_survival = int(survival_key > best_survival_key)
                is_best_reward = int(eval_metrics["eval_mean_reward"] > best_reward)
                eval_writer.writerow({"episode": episode, "completed_episodes": episode + 1, "update_count": mappo.lr_update_count, "environment_steps": total_environment_steps, **eval_metrics, "balanced_score": balanced_score, "is_best_balanced": is_best_balanced, "is_best_survival": is_best_survival, "is_best_reward": is_best_reward})
                eval_f.flush()
                save_checkpoint(output_dir / "latest.pt", actor, critic, graph_encoder, memory, mappo, episode, run_config)
                if is_best_balanced:
                    best_balanced_score = balanced_score
                    save_checkpoint(output_dir / "best_balanced.pt", actor, critic, graph_encoder, memory, mappo, episode, run_config)
                if is_best_survival:
                    best_survival_key = survival_key
                    save_checkpoint(output_dir / "best_survival.pt", actor, critic, graph_encoder, memory, mappo, episode, run_config)
                if is_best_reward:
                    best_reward = eval_metrics["eval_mean_reward"]
                    save_checkpoint(output_dir / "best_reward.pt", actor, critic, graph_encoder, memory, mappo, episode, run_config)

            advanced_history.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "raw_reward": float(raw_reward),
                    "total_reward": float(total_reward),
                    "steps": int(episode_step),
                    "steps_avg_50": float(steps_avg_50),
                    "steps_std_50": float(steps_std_50),
                    "survived_to_limit": int(survived_to_limit),
                    "survival_rate_50": float(survival_rate_50),
                    "termination_reason": termination_reason or "unknown",
                    "avg_50": float(avg_50),
                    "avg_100": float(avg_100),
                    "std_50": float(std_50),
                    "success_rate_50": float(success_rate_50),
                    "actor_loss": float(last_stats["actor_loss"]),
                    "critic_loss": float(last_stats["critic_loss"]),
                    "entropy": float(last_stats["entropy"]),
                    "approx_kl": float(last_stats["approx_kl"]),
                    "clip_fraction": float(last_stats["clip_fraction"]),
                    "explained_variance": float(last_stats["explained_variance"]),
                    "actor_grad_norm": float(last_stats["actor_grad_norm"]),
                    "critic_grad_norm": float(last_stats["critic_grad_norm"]),
                    "graph_grad_norm": float(last_stats.get("graph_grad_norm", np.nan)),
                    "memory_grad_norm": float(last_stats.get("memory_grad_norm", np.nan)),
                    "graph_parameter_delta": float(last_stats.get("graph_parameter_delta", np.nan)),
                    "memory_parameter_delta": float(last_stats.get("memory_parameter_delta", np.nan)),
                    "graph_parameter_relative_delta": float(last_stats.get("graph_parameter_relative_delta", np.nan)),
                    "memory_parameter_relative_delta": float(last_stats.get("memory_parameter_relative_delta", np.nan)),
                    "memory_gate": float(torch.sigmoid(actor.memory_gate_logit).item()) if actor.use_memory_gate else 1.0,
                    "memory_gate_logit": float(actor.memory_gate_logit.item()) if actor.use_memory_gate else 0.0,
                    "graph_trainable": int(mappo.graph_trainable),
                    "memory_trainable": int(mappo.memory_trainable),
                    "value_pred_mean": float(last_stats.get("value_pred_mean", np.nan)),
                    "value_target_mean": float(last_stats.get("value_target_mean", np.nan)),
                    "value_error_abs_mean": float(last_stats.get("value_error_abs_mean", np.nan)),
                    "value_clip_fraction": float(last_stats.get("value_clip_fraction", np.nan)),
                    "adv_mean": float(last_stats["adv_mean"]),
                    "adv_std": float(last_stats["adv_std"]),
                    "policy_entropy": float(last_stats["policy_entropy"]),
                    "actor_lr": float(current_lrs["actor_lr"]),
                    "critic_lr": float(current_lrs["critic_lr"]),
                }
            )

            history.append(
                {
                    "episode": episode,
                    "total_reward": float(total_reward),
                    "steps": int(episode_step),
                    "actor_loss": float(last_stats["actor_loss"]),
                    "critic_loss": float(last_stats["critic_loss"]),
                    "entropy": float(last_stats["entropy"]),
                }
            )
            print(
                f"Episode {episode}, raw_reward={raw_reward:.2f}, avg_50={avg_50:.2f}, avg_100={avg_100:.2f}, std_50={std_50:.2f}, success_rate_50={success_rate_50:.2f}"
            )
            print(
                f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {episode_step}, "
                f"steps_avg_50={steps_avg_50:.2f}, survival_rate_50={survival_rate_50:.2f}, "
                f"termination_reason={termination_reason or 'unknown'}, attacks="
                f"(FDI={fdi_count}, DoS={dos_count}, Breaker={breaker_count}), "
                f"restored={restored_line_count}, max_concurrent={max_concurrent_breakers}, "
                f"active_at_end={active_breaker_count_at_end}, Stats: {last_stats}"
            )

    if mappo.buffer:
        final_stats = mappo.update(bootstrap_value=0.0)
        current_lrs = mappo.get_current_lrs()
        with ppo_csv_path.open("a", newline="", encoding="utf-8") as ppo_f:
            ppo_writer = csv.DictWriter(ppo_f, fieldnames=ppo_fields)
            final_row = {"update_count": final_stats["update_count"], "episode": episodes - 1, "environment_steps": total_environment_steps, **{name: final_stats[name] for name in ppo_fields if name in final_stats}, **current_lrs}
            ppo_writer.writerow({name: final_row.get(name, "") for name in ppo_fields})
    save_checkpoint(output_dir / "final.pt", actor, critic, graph_encoder, memory, mappo, episodes - 1, run_config)
    df = pd.DataFrame(history)
    advanced_df = pd.DataFrame(advanced_history)
    excel_path = output_dir / "training_metrics.xlsx"
    df.to_excel(excel_path, index=False)
    advanced_metrics_path = output_dir / "training_metrics_advanced_2.xlsx"
    advanced_df.to_excel(advanced_metrics_path, index=False)

    if MATPLOTLIB_AVAILABLE:
        advanced_df = pd.read_csv(csv_path)
        episodes_arr = advanced_df["episode"].to_numpy()
        raw_reward = advanced_df["raw_reward"].to_numpy()
        avg_50 = advanced_df["avg_50"].to_numpy()
        avg_100 = advanced_df["avg_100"].to_numpy()
        std_50 = advanced_df["std_50"].to_numpy()
        success_rate_50 = advanced_df["success_rate_50"].to_numpy()

        fig_adv, axes_adv = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

        axes_adv[0].plot(episodes_arr, raw_reward, color="lightgray", linewidth=1.0, label="raw_reward")
        axes_adv[0].plot(episodes_arr, avg_50, color="tab:blue", linewidth=2.2, label="avg_50")
        axes_adv[0].plot(episodes_arr, avg_100, color="tab:orange", linewidth=2.2, linestyle="--", label="avg_100")
        axes_adv[0].set_ylabel("Reward")
        axes_adv[0].grid(True, alpha=0.3)
        axes_adv[0].legend()

        axes_adv[1].plot(episodes_arr, avg_50, color="tab:blue", linewidth=2.0, label="avg_50")
        axes_adv[1].fill_between(episodes_arr, avg_50 - std_50, avg_50 + std_50, color="tab:blue", alpha=0.2, label="avg_50 ± std_50")
        axes_adv[1].set_ylabel("Stability")
        axes_adv[1].grid(True, alpha=0.3)
        axes_adv[1].legend()

        axes_adv[2].plot(episodes_arr, success_rate_50, color="tab:green", linewidth=2.0, label="success_rate_50")
        axes_adv[2].set_ylabel("Success Rate")
        axes_adv[2].set_xlabel("Episode")
        axes_adv[2].set_ylim(0.0, 1.0)
        axes_adv[2].grid(True, alpha=0.3)
        axes_adv[2].legend()

        fig_adv.tight_layout()
        advanced_plot_path = output_dir / "training_curves_1000eps.png"
        fig_adv.savefig(advanced_plot_path, dpi=150)
        plt.close(fig_adv)
        print(f"Saved advanced plots to: {advanced_plot_path}")

        steps_arr = advanced_df["steps"].to_numpy()
        steps_avg_50 = advanced_df["steps_avg_50"].to_numpy()
        steps_std_50 = advanced_df["steps_std_50"].to_numpy()
        survival_rate_50 = advanced_df["survival_rate_50"].to_numpy()
        fig_survival, axes_survival = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        axes_survival[0].plot(episodes_arr, steps_arr, color="lightgray", linewidth=1.0, label="steps")
        axes_survival[0].plot(episodes_arr, steps_avg_50, color="tab:blue", linewidth=2.2, label="steps_avg_50")
        axes_survival[0].set_ylabel("Episode Steps")
        axes_survival[0].grid(True, alpha=0.3)
        axes_survival[0].legend()

        axes_survival[1].plot(episodes_arr, steps_avg_50, color="tab:orange", linewidth=2.0, label="steps_avg_50")
        axes_survival[1].fill_between(
            episodes_arr,
            np.maximum(0.0, steps_avg_50 - steps_std_50),
            np.minimum(float(steps), steps_avg_50 + steps_std_50),
            color="tab:orange",
            alpha=0.2,
            label="steps_avg_50 ± steps_std_50",
        )
        axes_survival[1].set_ylabel("Step Stability")
        axes_survival[1].set_ylim(0.0, float(steps) * 1.05)
        axes_survival[1].grid(True, alpha=0.3)
        axes_survival[1].legend()

        axes_survival[2].plot(episodes_arr, survival_rate_50, color="tab:green", linewidth=2.2, label="survival_rate_50")
        axes_survival[2].set_ylabel("Survival Rate")
        axes_survival[2].set_xlabel("Episode")
        axes_survival[2].set_ylim(0.0, 1.0)
        axes_survival[2].grid(True, alpha=0.3)
        axes_survival[2].legend()

        fig_survival.tight_layout()
        survival_plot_path = output_dir / "survival_curves.png"
        fig_survival.savefig(survival_plot_path, dpi=150)
        plt.close(fig_survival)
        print(f"Saved survival plots to: {survival_plot_path}")

        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        axes[0].plot(df["episode"], df["total_reward"], color="tab:blue", linewidth=2.0, label="Total Reward")
        axes[0].set_ylabel("Total Reward")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(df["episode"], df["steps"], color="tab:orange", linewidth=2.0, label="Steps")
        axes[1].set_ylabel("Steps")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(df["episode"], df["actor_loss"], label="actor_loss", color="tab:green", linewidth=2.0)
        axes[2].plot(df["episode"], df["critic_loss"], label="critic_loss", color="tab:red", linewidth=2.0)
        axes[2].plot(df["episode"], df["entropy"], label="entropy", color="tab:purple", linewidth=2.0)
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Loss / Entropy")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        fig.tight_layout()
        plot_path = output_dir / "training_curves.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved plots to: {plot_path}")
    else:
        print("matplotlib is not available; skipped plotting and only saved CSV logs.")

    print(f"Saved advanced metrics to: {csv_path}")
    print(f"Saved advanced metrics xlsx to: {advanced_metrics_path}")
    print(f"Saved metrics to: {excel_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--steps", type=int, default=STEPS_PER_EPISODE)
    parser.add_argument("--success_threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--representation_mode", choices=("frozen", "joint"), default="frozen")
    parser.add_argument("--representation_lr", type=float, default=5e-6)
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--actor_lr", type=float, default=5e-5)
    parser.add_argument("--critic_lr", type=float, default=5e-5)
    parser.add_argument("--memory_unfreeze_episode", type=int, default=100)
    parser.add_argument("--graph_unfreeze_episode", type=int, default=200)
    parser.add_argument("--entropy_coef", type=float, default=0.005)
    args = parser.parse_args()
    main(
        episodes=args.episodes,
        steps=args.steps,
        success_threshold=args.success_threshold,
        seed=args.seed,
        output_dir=args.output_dir,
        representation_mode=args.representation_mode,
        representation_lr=args.representation_lr,
        sequence_length=args.sequence_length,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        memory_unfreeze_episode=args.memory_unfreeze_episode,
        graph_unfreeze_episode=args.graph_unfreeze_episode,
        entropy_coef=args.entropy_coef,
    )
