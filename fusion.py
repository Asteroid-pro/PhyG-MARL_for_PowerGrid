from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLUMNS = {
    "episode",
    "avg_50",
    "avg_100",
    "survival_rate_50",
    "mean_load_ratio",
    "mean_high_voltage_rate",
}


def seed_from_directory(directory: Path, frame: pd.DataFrame) -> int:
    match = re.search(r"seed[_-]?(\d+)", directory.name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if "seed" in frame.columns and frame["seed"].notna().any():
        return int(frame["seed"].dropna().iloc[0])
    raise ValueError(f"Cannot determine seed from directory or CSV: {directory}")


def load_experiments(parent: Path, seeds: list[int]) -> tuple[pd.DataFrame, list[int]]:
    csv_paths = [
        parent / f"results_joint_v3_seed{seed}" / "training_metrics_advanced.csv"
        for seed in seeds
    ]
    missing_paths = [path for path in csv_paths if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing joint V3 training metrics:\n{missing_text}")

    frames: list[pd.DataFrame] = []
    seen_seeds: set[int] = set()
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        seed = seed_from_directory(csv_path.parent, frame)
        if seed in seen_seeds:
            raise ValueError(f"Duplicate training seed {seed}: {csv_path}")
        seen_seeds.add(seed)
        frame = frame.copy()
        frame["experiment_seed"] = seed
        frame["source_directory"] = csv_path.parent.name
        frames.append(frame)
        print(f"Loaded seed {seed}: {csv_path} ({len(frame)} episodes)")

    loaded_seeds = sorted(seen_seeds)
    return pd.concat(frames, ignore_index=True), loaded_seeds


def plot_comparison(data: pd.DataFrame, seeds: list[int], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    colors = plt.get_cmap("tab10")

    for index, seed in enumerate(seeds):
        frame = data[data["experiment_seed"] == seed].sort_values("episode")
        episode = frame["episode"]
        color = colors(index % 10)

        axes[0, 0].plot(
            episode, frame["avg_50"], color=color, linewidth=1.9,
            label=f"Seed {seed} avg_50",
        )
        axes[0, 0].plot(
            episode, frame["avg_100"], color=color, linewidth=1.2,
            linestyle="--", alpha=0.85, label=f"Seed {seed} avg_100",
        )
        axes[0, 1].plot(
            episode, frame["survival_rate_50"], color=color,
            linewidth=1.9, label=f"Seed {seed}",
        )
        axes[1, 0].plot(
            episode, frame["mean_load_ratio"].rolling(50, min_periods=1).mean(),
            color=color, linewidth=1.9, label=f"Seed {seed}",
        )
        axes[1, 1].plot(
            episode,
            frame["mean_high_voltage_rate"].rolling(50, min_periods=1).mean(),
            color=color, linewidth=1.9, label=f"Seed {seed}",
        )

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

    fig.suptitle("Joint GNN/Transformer Multi-Seed Training Comparison", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse independent joint-training results into a multi-seed comparison."
    )
    parser.add_argument(
        "--input_dir", type=Path, default=Path("result_joint_v3_multiple_seeds"),
        help="Directory containing results_joint_v3_seed42/43/44.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44],
        help="Training seeds whose results_joint_v3_seed<seed> directories are loaded.",
    )
    parser.add_argument(
        "--output_name", default="joint_v3_multiseed_core_metrics.png",
        help="Output comparison figure filename.",
    )
    args = parser.parse_args()

    data, seeds = load_experiments(args.input_dir, args.seeds)
    combined_path = args.input_dir / "joint_v3_multiseed_training_metrics.csv"
    figure_path = args.input_dir / args.output_name
    data.to_csv(combined_path, index=False)
    plot_comparison(data, seeds, figure_path)

    print(f"Seeds: {seeds}")
    print(f"Combined metrics saved to: {combined_path}")
    print(f"Comparison figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
