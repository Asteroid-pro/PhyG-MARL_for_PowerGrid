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


SCALING_CONFIGS = {
    "current": {0: 1.00, 1: 0.95, 2: 0.90},
    "medium": {0: 1.00, 1: 0.85, 2: 0.70},
    "strong": {0: 1.00, 1: 0.70, 2: 0.40},
}
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


def evaluate_configuration(
    scaling_name: str,
    scaling: dict[int, float],
    policy: str,
    episodes: int,
    max_steps: int,
    base_seed: int,
    max_breakers: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for episode in range(episodes):
        episode_seed = base_seed + episode
        env = CPGridEnv(
            max_steps=max_steps,
            max_breaker_attacks=max_breakers,
            breaker_transient=True,
            breaker_duration_range=(10, 20),
            max_active_breakers=2,
            load_scaling_by_action=scaling,
        )
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
                "scaling_config": scaling_name,
                "action_0_scaling": scaling[0],
                "action_1_scaling": scaling[1],
                "action_2_scaling": scaling[2],
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
    group_columns = ["scaling_config", "policy"]
    for (scaling_name, policy), frame in details.groupby(group_columns, sort=False):
        records.append(
            {
                "scaling_config": scaling_name,
                "action_0_scaling": frame["action_0_scaling"].iloc[0],
                "action_1_scaling": frame["action_1_scaling"].iloc[0],
                "action_2_scaling": frame["action_2_scaling"].iloc[0],
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


def build_recommendation(summary: pd.DataFrame) -> dict:
    indexed = summary.set_index(["scaling_config", "policy"])
    current = indexed.loc[("current", "all_2")]
    medium = indexed.loc[("medium", "all_2")]
    strong = indexed.loc[("strong", "all_2")]
    survival_gain_medium = float(medium["survival_rate"] - current["survival_rate"])
    survival_gain_strong = float(strong["survival_rate"] - current["survival_rate"])
    step_gain_strong = float(strong["mean_steps"] - current["mean_steps"])
    load_loss_medium = float(current["mean_load_ratio"] - medium["mean_load_ratio"])
    load_loss_strong = float(current["mean_load_ratio"] - strong["mean_load_ratio"])

    shedding_affects_survival = survival_gain_strong >= 0.10
    shedding_insufficient = survival_gain_strong < 0.05 and step_gain_strong < 10.0
    maximum_gain = max(survival_gain_strong, 0.0)
    medium_captures_most_gain = maximum_gain > 0 and survival_gain_medium >= 0.8 * maximum_gain
    survival_load_tradeoff = survival_gain_strong > 0 and load_loss_strong > 0

    if shedding_insufficient:
        conclusion = (
            "Load shedding is insufficient for the dominant failure mode. Do not keep increasing shedding; "
            "consider transient Breaker attacks, line restoration, or generator voltage/reactive support."
        )
        recommended_config = None
    elif medium_captures_most_gain:
        conclusion = "Medium shedding captures at least 80% of the strong-shedding survival gain and is preferred."
        recommended_config = "medium"
    elif shedding_affects_survival:
        conclusion = (
            "Load shedding affects survival, but strong shedding is required. Refine amplitudes between medium "
            "and strong before training."
        )
        recommended_config = "refine_between_medium_and_strong"
    else:
        conclusion = "The effect is measurable but inconclusive; increase episode count before changing the action design."
        recommended_config = None

    return {
        "diagnostic_basis": "Compare all_2 under identical attack seeds across current, medium, and strong scaling.",
        "survival_gain_medium_vs_current": survival_gain_medium,
        "survival_gain_strong_vs_current": survival_gain_strong,
        "mean_steps_gain_strong_vs_current": step_gain_strong,
        "load_ratio_loss_medium_vs_current": load_loss_medium,
        "load_ratio_loss_strong_vs_current": load_loss_strong,
        "load_shedding_affects_survival": shedding_affects_survival,
        "load_shedding_effect_insufficient": shedding_insufficient,
        "medium_captures_at_least_80_percent_of_strong_gain": medium_captures_most_gain,
        "survival_load_tradeoff": survival_load_tradeoff,
        "recommended_config": recommended_config,
        "conclusion": conclusion,
    }


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    x = np.arange(len(POLICY_NAMES))
    width = 0.8 / len(SCALING_CONFIGS)
    fig, axes = plt.subplots(4, 1, figsize=(14, 17), sharex=True)
    metrics = (
        ("survival_rate", "Survival Rate"),
        ("mean_steps", "Mean Steps"),
        ("mean_load_ratio", "Mean Load Ratio"),
        ("mean_step_reward", "Mean Step Reward"),
    )
    for config_index, config_name in enumerate(SCALING_CONFIGS):
        frame = summary[summary["scaling_config"] == config_name].set_index("policy").reindex(POLICY_NAMES)
        offset = (config_index - (len(SCALING_CONFIGS) - 1) / 2) * width
        for axis, (metric, _) in zip(axes, metrics):
            axis.bar(x + offset, frame[metric], width, label=config_name)
    for axis, (_, label) in zip(axes, metrics):
        axis.set_ylabel(label)
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
    axes[0].set_ylim(0.0, 1.0)
    axes[2].set_ylim(0.0, 1.05)
    axes[-1].set_xlabel("Fixed Policy")
    axes[-1].set_xticks(x, POLICY_NAMES)
    fig.tight_layout()
    path = output_dir / "load_shedding_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether stronger load shedding changes grid survival.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--max_breakers", type=int, default=4)
    parser.add_argument("--base_seed", type=int, default=10000)
    parser.add_argument("--output_dir", type=Path, default=Path("results_load_shedding_diagnostic"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for scaling_name, scaling in SCALING_CONFIGS.items():
        for policy in POLICY_NAMES:
            print(f"Evaluating scaling={scaling_name}, policy={policy}")
            frames.append(
                evaluate_configuration(
                    scaling_name,
                    scaling,
                    policy,
                    args.episodes,
                    args.steps,
                    args.base_seed,
                    args.max_breakers,
                )
            )

    details = pd.concat(frames, ignore_index=True)
    summary = summarize(details)
    recommendation = build_recommendation(summary)
    details_path = args.output_dir / "load_shedding_details.csv"
    summary_path = args.output_dir / "load_shedding_summary.csv"
    recommendation_path = args.output_dir / "load_shedding_recommendation.json"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    with recommendation_path.open("w", encoding="utf-8") as file:
        json.dump(recommendation, file, ensure_ascii=False, indent=2)
    plot_path = plot_summary(summary, args.output_dir)

    print(summary.to_string(index=False))
    print(f"Saved details: {details_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved recommendation: {recommendation_path}")
    print(f"Saved comparison plot: {plot_path}")
    print(f"Conclusion: {recommendation['conclusion']}")


if __name__ == "__main__":
    main()
