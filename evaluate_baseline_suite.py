"""
Baseline 统一离线评估入口：加载各方法 checkpoint，在固定测试 seed 和确定性策略下生成逐 episode 结果及方法汇总。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline_config import METHODS, SEEDS, BaselineConfig
from baselines.common.env_adapter import global_state, make_env, observation_tensor
from config import DEVICE, NUM_AGENTS
from run_baseline_suite import MethodRuntime


def evaluate_saved_baseline(method: str, seed: int, root: Path, config: BaselineConfig) -> dict[str, float]:
    """Evaluate a saved baseline checkpoint on the fixed test seed set.

    The method-specific agent loader is intentionally kept behind this function so
    the evaluation protocol remains identical for every baseline.
    """
    checkpoint_path = root / method / f"seed_{seed}" / "best_balanced.pt"
    if not checkpoint_path.exists():
        checkpoint_path = root / method / f"seed_{seed}" / "final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint for {method}, seed {seed}: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    env = make_env(config.steps)
    obs_dim = env.observation_space["agent_0"].shape[0]
    state_dim = NUM_AGENTS * 4 + NUM_AGENTS + 2
    runtime = MethodRuntime(method, obs_dim, state_dim, config)
    if method == "qmix":
        agent_q_state = checkpoint.get("agent_q_state_dict", checkpoint.get("actor_state_dict"))
        mixer_state = checkpoint.get("mixer_state_dict", checkpoint.get("critic_state_dict"))
        if agent_q_state is None or mixer_state is None:
            raise KeyError(
                "QMIX checkpoint must contain either "
                "agent_q_state_dict/mixer_state_dict or actor_state_dict/critic_state_dict"
            )
        runtime.qmix.agent_q.load_state_dict(agent_q_state)
        runtime.qmix.mixer.load_state_dict(mixer_state)
        runtime.qmix.target_agent_q.load_state_dict(agent_q_state)
        runtime.qmix.target_mixer.load_state_dict(mixer_state)
    else:
        runtime.actor.load_state_dict(checkpoint["actor_state_dict"])
        runtime.critic.load_state_dict(checkpoint["critic_state_dict"])
    records = []
    with torch.no_grad():
        for test_seed in config.test_seeds:
            observations, _ = env.reset(seed=test_seed)
            total_reward, loads, lows, highs = 0.0, [], [], []
            runtime.reset_episode()
            for episode_step in range(config.steps):
                node, edge, edge_index, injection = env.get_graph_data()
                state = global_state(env, node, config.steps).to(DEVICE)
                obs = observation_tensor(observations, DEVICE)
                graph_data = (node, edge, edge_index, injection)
                actions, _, _ = runtime.act(obs, state, graph_data, deterministic=True)
                observations, reward, terminated, truncated, info = env.step(actions.cpu().numpy())
                total_reward += float(reward)
                metrics = info.get("powerflow_metrics")
                if metrics:
                    loads.append(metrics["load_ratio"]); lows.append(metrics["low_voltage_rate"]); highs.append(metrics["high_voltage_rate"])
                if terminated or truncated:
                    break
            records.append({"method": method, "training_seed": seed, "test_seed": test_seed,
                            "survived_to_limit": int(truncated and info.get("termination_reason") == "time_limit"),
                            "steps": episode_step + 1, "total_reward": total_reward,
                            "mean_load_ratio": np.mean(loads), "mean_low_voltage_rate": np.mean(lows),
                            "mean_high_voltage_rate": np.mean(highs), "termination_reason": info.get("termination_reason", "unknown")})
    env.close()
    frame = pd.DataFrame(records)
    frame.to_csv(root / method / f"seed_{seed}" / "offline_episode_results.csv", index=False)
    return {"method": method, "training_seed": seed, "survival_rate": frame.survived_to_limit.mean(),
            "mean_steps": frame.steps.mean(), "mean_reward": frame.total_reward.mean(),
            "mean_load_ratio": frame.mean_load_ratio.mean(), "mean_low_voltage_rate": frame.mean_low_voltage_rate.mean(),
            "mean_high_voltage_rate": frame.mean_high_voltage_rate.mean(),
            "powerflow_not_converged_rate": (frame.termination_reason == "powerflow_not_converged").mean()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline checkpoints on common test seeds.")
    parser.add_argument("--root", type=Path, default=Path("baseline_comparison"))
    parser.add_argument("--output_dir", type=Path, default=Path("baseline_comparison/evaluation"))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in args.methods:
        if method == "phyg_v2_2_2":
            continue
        for seed in args.seeds:
            rows.append(evaluate_saved_baseline(method, seed, args.root, BaselineConfig()))
    summary = pd.DataFrame(rows)
    summary["balanced_score"] = summary["survival_rate"] + 0.5 * summary["mean_load_ratio"] - 0.5 * summary["mean_high_voltage_rate"]
    summary.to_csv(args.output_dir / "baseline_checkpoint_summary.csv", index=False)
    method_summary = summary.groupby("method", as_index=False).agg(
        survival_rate_mean=("survival_rate", "mean"), survival_rate_std=("survival_rate", "std"),
        mean_reward_mean=("mean_reward", "mean"), mean_reward_std=("mean_reward", "std"),
        mean_load_ratio_mean=("mean_load_ratio", "mean"), mean_load_ratio_std=("mean_load_ratio", "std"),
        mean_high_voltage_rate_mean=("mean_high_voltage_rate", "mean"), mean_high_voltage_rate_std=("mean_high_voltage_rate", "std"),
        balanced_score_mean=("balanced_score", "mean"), balanced_score_std=("balanced_score", "std"),
    )
    method_summary.to_csv(args.output_dir / "method_summary.csv", index=False)


if __name__ == "__main__":
    main()
