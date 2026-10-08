from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train import main as train_one_seed


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
REQUIRED_COLUMNS = {
    "episode",
    "avg_50",
    "std_50",
    "success_rate_50",
    "steps",
    "steps_avg_50",
    "steps_std_50",
    "survival_rate_50",
}


def _load_seed_metrics(results_dir: Path, seed: int, episodes: int) -> pd.DataFrame:
    csv_path = results_dir / f"seed_{seed}" / "training_metrics_advanced.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics for seed {seed}: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Seed {seed} metrics are missing columns: {sorted(missing)}")
    if len(frame) != episodes:
        raise ValueError(f"Seed {seed} has {len(frame)} episodes; expected {episodes}")
    if frame["episode"].duplicated().any():
        raise ValueError(f"Seed {seed} contains duplicate episode values")

    expected = np.arange(episodes)
    actual = frame["episode"].to_numpy()
    if not np.array_equal(actual, expected):
        raise ValueError(f"Seed {seed} episode values are not aligned from 0 to {episodes - 1}")

    return frame


def _is_completed(output_dir: Path, episodes: int) -> bool:
    csv_path = output_dir / "training_metrics_advanced.csv"
    if not csv_path.exists():
        return False
    try:
        frame = pd.read_csv(csv_path, usecols=["episode"])
    except (ValueError, pd.errors.EmptyDataError):
        return False
    return len(frame) == episodes and np.array_equal(frame["episode"].to_numpy(), np.arange(episodes))


def aggregate_results(results_dir: Path, seeds: tuple[int, ...], episodes: int) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    frames = {seed: _load_seed_metrics(results_dir, seed, episodes) for seed in seeds}
    episodes_arr = np.arange(episodes)

    avg_50 = np.stack([frames[seed]["avg_50"].to_numpy(dtype=float) for seed in seeds])
    std_50 = np.stack([frames[seed]["std_50"].to_numpy(dtype=float) for seed in seeds])
    success = np.stack([frames[seed]["success_rate_50"].to_numpy(dtype=float) for seed in seeds])
    steps_avg = np.stack([frames[seed]["steps_avg_50"].to_numpy(dtype=float) for seed in seeds])
    steps_std = np.stack([frames[seed]["steps_std_50"].to_numpy(dtype=float) for seed in seeds])
    survival = np.stack([frames[seed]["survival_rate_50"].to_numpy(dtype=float) for seed in seeds])

    summary = pd.DataFrame(
        {
            "episode": episodes_arr,
            "avg_50_mean": avg_50.mean(axis=0),
            "avg_50_std": avg_50.std(axis=0, ddof=1),
            "std_50_mean": std_50.mean(axis=0),
            "std_50_std": std_50.std(axis=0, ddof=1),
            "success_rate_50_mean": success.mean(axis=0),
            "success_rate_50_std": success.std(axis=0, ddof=1),
            "steps_avg_50_mean": steps_avg.mean(axis=0),
            "steps_avg_50_std": steps_avg.std(axis=0, ddof=1),
            "steps_std_50_mean": steps_std.mean(axis=0),
            "survival_rate_50_mean": survival.mean(axis=0),
            "survival_rate_50_std": survival.std(axis=0, ddof=1),
            "n_seeds": len(seeds),
        }
    )
    summary.to_csv(results_dir / "seed_summary.csv", index=False)
    return frames, summary


def plot_seed_results(results_dir: Path, seeds: tuple[int, ...], frames: dict[int, pd.DataFrame], summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))

    for color, seed in zip(colors, seeds):
        frame = frames[seed]
        episode = frame["episode"].to_numpy()
        axes[0].plot(episode, frame["avg_50"], color=color, linewidth=1.4, alpha=0.8, label=f"seed {seed}")
        axes[1].plot(episode, frame["std_50"], color=color, linewidth=1.4, alpha=0.8, label=f"seed {seed}")
        axes[2].plot(episode, frame["success_rate_50"], color=color, linewidth=1.4, alpha=0.8, label=f"seed {seed}")

    episode = summary["episode"].to_numpy()
    axes[0].plot(episode, summary["avg_50_mean"], color="black", linewidth=2.5, label="5-seed mean")
    axes[1].plot(episode, summary["std_50_mean"], color="black", linewidth=2.5, label="5-seed mean")
    axes[2].plot(episode, summary["success_rate_50_mean"], color="black", linewidth=2.5, label="5-seed mean")

    axes[0].set_ylabel("avg_50")
    axes[0].set_title("Reward trend across random seeds")
    axes[1].set_ylabel("std_50")
    axes[1].set_title("Within-seed reward stability (50-episode window)")
    axes[2].set_ylabel("success_rate_50")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Success rate across random seeds")
    axes[2].set_ylim(0.0, 1.0)

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=3)

    fig.tight_layout()
    plot_path = results_dir / "seed_test.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_seed_survival_results(
    results_dir: Path,
    seeds: tuple[int, ...],
    frames: dict[int, pd.DataFrame],
    summary: pd.DataFrame,
    max_steps: int,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))

    for color, seed in zip(colors, seeds):
        frame = frames[seed]
        episode = frame["episode"].to_numpy()
        axes[0].plot(episode, frame["steps"], color=color, linewidth=0.8, alpha=0.35, label=f"seed {seed}")
        axes[1].plot(episode, frame["steps_avg_50"], color=color, linewidth=1.5, alpha=0.85, label=f"seed {seed}")
        axes[2].plot(episode, frame["survival_rate_50"], color=color, linewidth=1.5, alpha=0.85, label=f"seed {seed}")

    episode = summary["episode"].to_numpy()
    axes[1].plot(episode, summary["steps_avg_50_mean"], color="black", linewidth=2.5, label="5-seed mean")
    axes[2].plot(episode, summary["survival_rate_50_mean"], color="black", linewidth=2.5, label="5-seed mean")

    axes[0].set_ylabel("Episode Steps")
    axes[0].set_title("Raw survival steps across random seeds")
    axes[0].set_ylim(0.0, float(max_steps) * 1.05)
    axes[1].set_ylabel("steps_avg_50")
    axes[1].set_title("50-episode average survival steps")
    axes[1].set_ylim(0.0, float(max_steps) * 1.05)
    axes[2].set_ylabel("survival_rate_50")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Rate of episodes reaching the time limit")
    axes[2].set_ylim(0.0, 1.0)

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=3)

    fig.tight_layout()
    plot_path = results_dir / "seed_survival_test.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and summarize the five-seed robustness experiment.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--success_threshold", type=float, default=0.0)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true", help="Retrain seeds whose complete CSV already exists.")
    parser.add_argument("--summary_only", action="store_true", help="Skip training and rebuild summary outputs only.")
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        for seed in seeds:
            seed_dir = args.results_dir / f"seed_{seed}"
            if not args.force and _is_completed(seed_dir, args.episodes):
                print(f"Seed {seed} already has {args.episodes} complete episodes; skipping.")
                continue
            print(f"Starting seed {seed}: episodes={args.episodes}, steps={args.steps}")
            train_one_seed(
                episodes=args.episodes,
                steps=args.steps,
                success_threshold=args.success_threshold,
                seed=seed,
                output_dir=seed_dir,
            )

    frames, summary = aggregate_results(args.results_dir, seeds, args.episodes)
    plot_path = plot_seed_results(args.results_dir, seeds, frames, summary)
    survival_plot_path = plot_seed_survival_results(
        args.results_dir, seeds, frames, summary, args.steps
    )
    print(f"Saved seed summary to: {args.results_dir / 'seed_summary.csv'}")
    print(f"Saved robustness plot to: {plot_path}")
    print(f"Saved survival robustness plot to: {survival_plot_path}")


if __name__ == "__main__":
    main()
