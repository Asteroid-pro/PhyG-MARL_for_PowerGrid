"""
PhyG checkpoint 评估工具：重建 GraphEncoder、TemporalMemory 和 Actor-Critic，并执行复现、重建检查及统一离线评估。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    ACTION_DIM,
    ACTOR_HIDDEN_DIM,
    DEVICE,
    EDGE_IN_DIM,
    GRAPH_NUM_LAYERS,
    HIDDEN_DIM,
    MEMORY_HISTORY_LEN,
    NODE_IN_DIM,
    NUM_AGENTS,
)
from env.cpgrid_env import CPGridEnv
from graph.graph_encoder import GraphEncoder
from memory.transformer_memory import TemporalMemory
from policy.actor import GraphActor
from utils.seed import set_seed

CHECKPOINT_TYPES = ("best_balanced", "best_survival", "final")


def seed_directory(root: Path, training_seed: int) -> Path:
    candidates = (
        root / f"results_joint_v3_seed{training_seed}",
        root / f"seed_{training_seed}",
    )
    for directory in candidates:
        if directory.is_dir():
            return directory
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing result directory for seed {training_seed}: {expected}")


def make_env(steps: int) -> CPGridEnv:
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


def reconstruct_modules(training_seed: int, checkpoint_path: Path, steps: int):
    set_seed(training_seed)
    env = make_env(steps)
    graph_encoder = GraphEncoder(
        node_in_dim=NODE_IN_DIM,
        edge_in_dim=EDGE_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=GRAPH_NUM_LAYERS,
    ).to(DEVICE)
    memory = TemporalMemory(hidden_dim=HIDDEN_DIM, history_length=MEMORY_HISTORY_LEN).to(DEVICE)
    obs_dim = env.observation_space["agent_0"].shape[0]
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    checkpoint_config = checkpoint.get("config", {})
    actor = GraphActor(
        obs_dim,
        HIDDEN_DIM,
        ACTION_DIM,
        hidden_dim=ACTOR_HIDDEN_DIM,
        use_memory_gate=checkpoint_config.get("version") in {"V3.1", "V3.2", "V3.3"},
    ).to(DEVICE)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    if "graph_encoder_state_dict" in checkpoint:
        graph_encoder.load_state_dict(checkpoint["graph_encoder_state_dict"])
    if "memory_state_dict" in checkpoint:
        memory.load_state_dict(checkpoint["memory_state_dict"])
    actor.eval()
    graph_encoder.eval()
    memory.eval()
    return env, graph_encoder, memory, actor, checkpoint


def evaluate_checkpoint(
    training_seed: int,
    checkpoint_type: str,
    checkpoint_path: Path,
    test_seeds: list[int],
    steps: int,
) -> tuple[pd.DataFrame, dict]:
    env, graph_encoder, memory, actor, checkpoint = reconstruct_modules(
        training_seed, checkpoint_path, steps
    )
    records = []
    with torch.no_grad():
        for test_seed in test_seeds:
            observations, info = env.reset(seed=test_seed)
            memory.clear()
            terminated = truncated = False
            episode_steps = 0
            total_reward = 0.0
            loads, low_rates, high_rates = [], [], []
            action_counts = np.zeros(3, dtype=np.int64)
            while not (terminated or truncated) and episode_steps < steps:
                node, edge, edge_index, injection = env.get_graph_data()
                if node is None:
                    break
                _, embedding = graph_encoder(
                    node.to(DEVICE),
                    edge.to(DEVICE),
                    edge_index.to(DEVICE),
                    torch.as_tensor(injection, dtype=torch.float32, device=DEVICE),
                )
                memory.update(embedding)
                memory_vector = memory.get_memory().to(DEVICE)
                obs_batch = torch.stack([
                    torch.as_tensor(observations[f"agent_{i}"], dtype=torch.float32, device=DEVICE)
                    for i in range(NUM_AGENTS)
                ])
                actions = torch.argmax(actor(obs_batch, memory_vector), dim=-1).cpu().numpy()
                action_counts += np.bincount(actions, minlength=3)
                observations, reward, terminated, truncated, info = env.step(actions)
                total_reward += float(reward)
                episode_steps += 1
                metrics = info.get("powerflow_metrics") if isinstance(info, dict) else None
                if metrics is not None:
                    loads.append(float(metrics["load_ratio"]))
                    low_rates.append(float(metrics["low_voltage_rate"]))
                    high_rates.append(float(metrics["high_voltage_rate"]))
            action_total = max(int(action_counts.sum()), 1)
            records.append({
                "training_seed": training_seed,
                "checkpoint_type": checkpoint_type,
                "checkpoint_episode": int(checkpoint.get("episode", -1)),
                "test_seed": test_seed,
                "survived_to_limit": int(truncated and info.get("termination_reason") == "time_limit"),
                "steps": episode_steps,
                "total_reward": total_reward,
                "mean_load_ratio": float(np.mean(loads)) if loads else np.nan,
                "mean_low_voltage_rate": float(np.mean(low_rates)) if low_rates else np.nan,
                "mean_high_voltage_rate": float(np.mean(high_rates)) if high_rates else np.nan,
                "termination_reason": info.get("termination_reason", "unknown"),
                "action_0_ratio": float(action_counts[0] / action_total),
                "action_1_ratio": float(action_counts[1] / action_total),
                "action_2_ratio": float(action_counts[2] / action_total),
            })
    env.close()
    frame = pd.DataFrame(records)
    summary = summarize(frame)
    summary.update({
        "training_seed": training_seed,
        "checkpoint_type": checkpoint_type,
        "checkpoint_episode": int(checkpoint.get("episode", -1)),
        "checkpoint_path": str(checkpoint_path),
        "graph_encoder_saved": int("graph_encoder_state_dict" in checkpoint),
        "memory_saved": int("memory_state_dict" in checkpoint),
    })
    return frame, summary


def summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    survival = frame["survived_to_limit"].astype(float)
    n = len(frame)
    p = float(survival.mean())
    margin = 1.96 * np.sqrt(p * (1.0 - p) / n) if n else np.nan
    return {
        "test_episodes": n,
        "survival_rate": p,
        "survival_ci95_low": max(0.0, p - margin),
        "survival_ci95_high": min(1.0, p + margin),
        "mean_steps": float(frame["steps"].mean()),
        "mean_reward": float(frame["total_reward"].mean()),
        "mean_load_ratio": float(frame["mean_load_ratio"].mean()),
        "mean_low_voltage_rate": float(frame["mean_low_voltage_rate"].mean()),
        "mean_high_voltage_rate": float(frame["mean_high_voltage_rate"].mean()),
        "powerflow_not_converged_rate": float((frame["termination_reason"] == "powerflow_not_converged").mean()),
        "invalid_voltage_rate": float((frame["termination_reason"] == "invalid_voltage").mean()),
    }


def original_evaluation_row(root: Path, training_seed: int, checkpoint_episode: int) -> pd.Series:
    path = seed_directory(root, training_seed) / "evaluation_metrics.csv"
    frame = pd.read_csv(path)
    rows = frame[frame["episode"] == checkpoint_episode]
    if rows.empty:
        raise ValueError(f"No original evaluation row for seed {training_seed}, episode {checkpoint_episode}")
    return rows.iloc[-1]


def run_reconstruction_check(root: Path, output_dir: Path, seeds: list[int], steps: int, smoke_only: bool) -> None:
    rows = []
    episode_frames = []
    for training_seed in seeds:
        checkpoint_path = seed_directory(root, training_seed) / "best_balanced.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_episode = int(checkpoint.get("episode", -1))
        eval_seeds = list(range(training_seed + 50000, training_seed + 50005)) if smoke_only else list(range(training_seed + 50000, training_seed + 50020))
        episodes, summary = evaluate_checkpoint(
            training_seed, "best_balanced", checkpoint_path, eval_seeds, steps
        )
        original = original_evaluation_row(root, training_seed, checkpoint_episode)
        row = {
            **summary,
            "original_survival_rate": float(original["eval_survival_rate"]),
            "original_mean_steps": float(original["eval_mean_steps"]),
            "original_mean_reward": float(original["eval_mean_reward"]),
            "original_mean_load_ratio": float(original["eval_mean_load_ratio"]),
            "original_high_voltage_rate": float(original["eval_high_voltage_rate"]),
        }
        row["survival_error"] = row["survival_rate"] - row["original_survival_rate"]
        row["reward_error"] = row["mean_reward"] - row["original_mean_reward"]
        row["load_error"] = row["mean_load_ratio"] - row["original_mean_load_ratio"]
        row["high_voltage_error"] = row["mean_high_voltage_rate"] - row["original_high_voltage_rate"]
        row["reconstruction_passed"] = int(
            not smoke_only
            and abs(row["survival_error"]) < 1e-12
            and abs(row["reward_error"]) < 1e-3
            and abs(row["load_error"]) < 1e-5
            and abs(row["high_voltage_error"]) < 1e-5
        )
        rows.append(row)
        episode_frames.append(episodes)
    prefix = "smoke5" if smoke_only else "reconstruction20"
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_report.csv", index=False)
    pd.concat(episode_frames, ignore_index=True).to_csv(output_dir / f"{prefix}_episodes.csv", index=False)


def run_full_evaluation(root: Path, output_dir: Path, seeds: list[int], steps: int) -> None:
    reconstruction_path = output_dir / "reconstruction20_report.csv"
    if not reconstruction_path.exists() or not pd.read_csv(reconstruction_path)["reconstruction_passed"].all():
        raise RuntimeError("20-seed reconstruction has not passed for every training seed; full evaluation is blocked.")
    episode_frames, summaries = [], []
    test_seeds = list(range(60000, 60100))
    for training_seed in seeds:
        for checkpoint_type in CHECKPOINT_TYPES:
            path = seed_directory(root, training_seed) / f"{checkpoint_type}.pt"
            episodes, summary = evaluate_checkpoint(training_seed, checkpoint_type, path, test_seeds, steps)
            episode_frames.append(episodes)
            summaries.append(summary)
    episodes = pd.concat(episode_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    episodes.to_csv(output_dir / "offline_episode_results.csv", index=False)
    summary.to_csv(output_dir / "offline_checkpoint_summary.csv", index=False)
    grouped = summary.groupby("checkpoint_type").agg({
        "survival_rate": ["mean", "std"],
        "mean_steps": ["mean", "std"],
        "mean_reward": ["mean", "std"],
        "mean_load_ratio": ["mean", "std"],
        "mean_high_voltage_rate": ["mean", "std"],
    })
    grouped.to_csv(output_dir / "offline_model_type_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct graph/memory modules and evaluate frozen or joint checkpoints.")
    parser.add_argument("--root", type=Path, default=Path("results_actor_critic_v2_2_1_multiseed_1000eps"))
    parser.add_argument("--output_dir", type=Path, default=Path("offline_evaluation"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--mode", choices=["smoke5", "reconstruct20", "full100"], required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "evaluation_plan.json").open("w", encoding="utf-8") as file:
        json.dump({"root": str(args.root), "training_seeds": args.seeds, "steps": args.steps, "mode": args.mode}, file, indent=2)
    if args.mode == "smoke5":
        run_reconstruction_check(args.root, args.output_dir, args.seeds, args.steps, smoke_only=True)
    elif args.mode == "reconstruct20":
        run_reconstruction_check(args.root, args.output_dir, args.seeds, args.steps, smoke_only=False)
    else:
        run_full_evaluation(args.root, args.output_dir, args.seeds, args.steps)
    print(f"Offline evaluation mode '{args.mode}' completed: {args.output_dir}")


if __name__ == "__main__":
    main()
