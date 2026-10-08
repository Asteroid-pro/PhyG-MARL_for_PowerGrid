from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from env.cpgrid_env import CPGridEnv


POLICY_NAMES = ("all_0", "all_1", "all_2", "random", "voltage_rule")


def select_actions(
    policy: str,
    observations: dict[str, torch.Tensor],
    rng: np.random.Generator,
) -> np.ndarray:
    if policy == "all_0":
        return np.zeros(14, dtype=int)
    if policy == "all_1":
        return np.ones(14, dtype=int)
    if policy == "all_2":
        return np.full(14, 2, dtype=int)
    if policy == "random":
        return rng.integers(0, 3, size=14, dtype=int)
    if policy == "voltage_rule":
        actions = np.zeros(14, dtype=int)
        for agent in range(14):
            voltage = float(observations[f"agent_{agent}"][0].item())
            actions[agent] = 2 if voltage < 0.90 else 1 if voltage < 0.95 else 0
        return actions
    raise ValueError(f"Unknown policy: {policy}")


def evaluate_policy(
    policy: str,
    episodes: int,
    max_steps: int,
    base_seed: int,
    max_breakers: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for episode in range(episodes):
        episode_seed = base_seed + episode
        env = CPGridEnv(max_steps=max_steps, max_breaker_attacks=max_breakers)
        observations, _ = env.reset(seed=episode_seed)
        action_rng = np.random.default_rng(episode_seed + 1_000_000)
        total_reward = 0.0
        load_ratios: list[float] = []
        action_counts = np.zeros(3, dtype=int)
        terminated = truncated = False
        info: dict = {}
        step_count = 0

        while not (terminated or truncated) and step_count < max_steps:
            actions = select_actions(policy, observations, action_rng)
            action_counts += np.bincount(actions, minlength=3)
            observations, reward, terminated, truncated, info = env.step(actions)
            total_reward += float(reward)
            step_count += 1
            if "load_ratio" in info:
                load_ratios.append(float(info["load_ratio"]))

        reason = info.get("termination_reason", "unknown")
        attack_counts = info.get("attack_counts", {})
        action_total = max(int(action_counts.sum()), 1)
        rows.append(
            {
                "policy": policy,
                "max_breakers": max_breakers,
                "episode": episode,
                "seed": episode_seed,
                "steps": step_count,
                "survived_to_limit": int(truncated and reason == "time_limit"),
                "termination_reason": reason,
                "total_reward": total_reward,
                "mean_step_reward": total_reward / max(step_count, 1),
                "mean_load_ratio": float(np.mean(load_ratios)) if load_ratios else 0.0,
                "fdi_count": int(attack_counts.get("fdi", 0)),
                "dos_count": int(attack_counts.get("dos", 0)),
                "breaker_count": int(attack_counts.get("breaker", 0)),
                "disconnected_line_count": int(info.get("disconnected_line_count", 0)),
                "action_0_ratio": action_counts[0] / action_total,
                "action_1_ratio": action_counts[1] / action_total,
                "action_2_ratio": action_counts[2] / action_total,
            }
        )
        env.close()
    return pd.DataFrame(rows)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (max_breakers, policy), frame in details.groupby(["max_breakers", "policy"], sort=True):
        records.append(
            {
                "max_breakers": max_breakers,
                "policy": policy,
                "episodes": len(frame),
                "mean_steps": frame["steps"].mean(),
                "std_steps": frame["steps"].std(ddof=1),
                "survival_rate": frame["survived_to_limit"].mean(),
                "mean_reward": frame["total_reward"].mean(),
                "mean_step_reward": frame["mean_step_reward"].mean(),
                "mean_load_ratio": frame["mean_load_ratio"].mean(),
                "invalid_voltage_rate": (frame["termination_reason"] == "invalid_voltage").mean(),
                "powerflow_failure_rate": (frame["termination_reason"] == "powerflow_not_converged").mean(),
                "mean_breaker_count": frame["breaker_count"].mean(),
                "action_0_ratio": frame["action_0_ratio"].mean(),
                "action_1_ratio": frame["action_1_ratio"].mean(),
                "action_2_ratio": frame["action_2_ratio"].mean(),
            }
        )
    return pd.DataFrame(records)


def difficulty_recommendation(summary: pd.DataFrame) -> dict:
    candidates = []
    for max_breakers, frame in summary.groupby("max_breakers"):
        rates = frame["survival_rate"].to_numpy(dtype=float)
        mean_rate = float(rates.mean())
        spread = float(rates.max() - rates.min())
        in_target = 0.30 <= mean_rate <= 0.70
        candidates.append(
            {
                "max_breakers": int(max_breakers),
                "mean_baseline_survival": mean_rate,
                "policy_survival_spread": spread,
                "in_30_70_target": in_target,
                "score": abs(mean_rate - 0.50) - 0.25 * spread,
            }
        )
    recommended = min(candidates, key=lambda item: item["score"])
    return {
        "criteria": "Prefer mean baseline survival in [0.30, 0.70] and visible separation between policies.",
        "evaluated_difficulties": candidates,
        "recommended_max_breakers": recommended["max_breakers"],
        "target_reached": recommended["in_30_70_target"],
        "next_step": (
            "Proceed to reward and Critic tuning."
            if recommended["in_30_70_target"]
            else "Neither tested setting fully reaches the target; inspect results before increasing difficulty further."
        ),
    }


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    difficulties = sorted(summary["max_breakers"].unique())
    x = np.arange(len(POLICY_NAMES))
    width = 0.8 / len(difficulties)
    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
    for index, max_breakers in enumerate(difficulties):
        frame = summary[summary["max_breakers"] == max_breakers].set_index("policy").reindex(POLICY_NAMES)
        offset = (index - (len(difficulties) - 1) / 2) * width
        axes[0].bar(x + offset, frame["survival_rate"], width, label=f"max Breaker={max_breakers}")
        axes[1].bar(x + offset, frame["mean_steps"], width, label=f"max Breaker={max_breakers}")
        axes[2].bar(x + offset, frame["mean_load_ratio"], width, label=f"max Breaker={max_breakers}")
    axes[0].axhspan(0.3, 0.7, color="tab:green", alpha=0.1, label="target range")
    axes[0].set_ylabel("Survival Rate")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("Mean Steps")
    axes[2].set_ylabel("Mean Load Ratio")
    axes[2].set_xlabel("Fixed Policy")
    axes[2].set_xticks(x, POLICY_NAMES)
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    path = output_dir / "fixed_policy_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed defense policies and select environment difficulty.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--base_seed", type=int, default=10000)
    parser.add_argument("--output_dir", type=Path, default=Path("results_fixed_policy_baselines"))
    parser.add_argument("--always_compare_4", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        evaluate_policy(policy, args.episodes, args.steps, args.base_seed, max_breakers=3)
        for policy in POLICY_NAMES
    ]
    first_summary = summarize(pd.concat(frames, ignore_index=True))
    all_near_ninety = bool((first_summary["survival_rate"] >= 0.85).all())

    if all_near_ninety or args.always_compare_4:
        frames.extend(
            evaluate_policy(policy, args.episodes, args.steps, args.base_seed, max_breakers=4)
            for policy in POLICY_NAMES
        )

    details = pd.concat(frames, ignore_index=True)
    summary = summarize(details)
    recommendation = difficulty_recommendation(summary)
    details.to_csv(args.output_dir / "fixed_policy_details.csv", index=False)
    summary.to_csv(args.output_dir / "fixed_policy_summary.csv", index=False)
    with (args.output_dir / "difficulty_recommendation.json").open("w", encoding="utf-8") as file:
        json.dump(recommendation, file, ensure_ascii=False, indent=2)
    plot_path = plot_summary(summary, args.output_dir)

    print(summary.to_string(index=False))
    print(f"Saved details: {args.output_dir / 'fixed_policy_details.csv'}")
    print(f"Saved summary: {args.output_dir / 'fixed_policy_summary.csv'}")
    print(f"Saved recommendation: {args.output_dir / 'difficulty_recommendation.json'}")
    print(f"Saved comparison plot: {plot_path}")
    print(f"Recommended max_breakers: {recommendation['recommended_max_breakers']}")
    print(recommendation["next_step"])


if __name__ == "__main__":
    main()
