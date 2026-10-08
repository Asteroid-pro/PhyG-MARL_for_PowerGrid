from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def run_training(seed: int, episodes: int, steps: int, output_dir: Path) -> None:
    command = [
        sys.executable,
        "train.py",
        "--seed",
        str(seed),
        "--episodes",
        str(episodes),
        "--steps",
        str(steps),
        "--success_threshold",
        "0.0",
        "--output_dir",
        str(output_dir),
    ]
    print(f"\n{'=' * 72}\nStarting seed {seed}: {output_dir}\n{'=' * 72}")
    subprocess.run(command, check=True)


def load_results(parent_dir: Path, seeds: list[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        csv_path = parent_dir / f"seed_{seed}" / "training_metrics_advanced.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing training metrics for seed {seed}: {csv_path}")
        frame = pd.read_csv(csv_path)
        frame["experiment_seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_core_metrics(data: pd.DataFrame, seeds: list[int], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    colors = plt.get_cmap("tab10")

    for index, seed in enumerate(seeds):
        frame = data[data["experiment_seed"] == seed].sort_values("episode")
        color = colors(index)
        episode = frame["episode"]

        axes[0, 0].plot(episode, frame["avg_50"], color=color, linewidth=1.8, label=f"Seed {seed} avg_50")
        axes[0, 0].plot(episode, frame["avg_100"], color=color, linewidth=1.2, linestyle="--", alpha=0.85, label=f"Seed {seed} avg_100")
        axes[0, 1].plot(episode, frame["survival_rate_50"], color=color, linewidth=1.8, label=f"Seed {seed}")
        axes[1, 0].plot(episode, frame["mean_load_ratio"].rolling(50, min_periods=1).mean(), color=color, linewidth=1.8, label=f"Seed {seed}")
        axes[1, 1].plot(episode, frame["mean_high_voltage_rate"].rolling(50, min_periods=1).mean(), color=color, linewidth=1.8, label=f"Seed {seed}")

    axes[0, 0].set_title("Reward Moving Averages")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 1].set_title("50-Episode Survival Rate")
    axes[0, 1].set_ylabel("Survival Rate")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[1, 0].set_title("50-Episode Mean Load Ratio")
    axes[1, 0].set_ylabel("Load Ratio")
    axes[1, 1].set_title("50-Episode Mean High-Voltage Rate")
    axes[1, 1].set_ylabel("High-Voltage Rate")

    for axis in axes.flat:
        axis.set_xlabel("Episode")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    fig.suptitle("Multi-Seed Actor/Critic V2.2.1 Core Metrics", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and compare multi-seed Actor/Critic experiments.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results_actor_critic_v2_2_1_multiseed_1000eps"),
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip training and regenerate comparison outputs from existing seed folders.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        for seed in args.seeds:
            run_training(seed, args.episodes, args.steps, args.output_dir / f"seed_{seed}")

    combined = load_results(args.output_dir, args.seeds)
    combined_path = args.output_dir / "multiseed_training_metrics.csv"
    figure_path = args.output_dir / "multiseed_core_metrics.png"
    combined.to_csv(combined_path, index=False)
    plot_core_metrics(combined, args.seeds, figure_path)
    print(f"\nCombined metrics saved to: {combined_path}")
    print(f"Comparison figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
