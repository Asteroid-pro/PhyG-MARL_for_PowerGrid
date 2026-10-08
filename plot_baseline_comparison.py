"""
Baseline 训练曲线绘图工具：读取多 seed episode 日志，生成各方法独立的 2×2 训练曲线图。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHOD_LABELS = {
    "mappo": "MAPPO",
    "ippo": "IPPO",
    "ppo": "PPO",
    "gru_mappo": "GRU-MAPPO",
    "gat_mappo": "GAT-MAPPO",
}


def load_training_results(root: Path, methods: list[str], seeds: list[int]) -> pd.DataFrame:
    frames = []
    for method in methods:
        for seed in seeds:
            path = root / method / f"seed_{seed}" / "training_metrics_advanced.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing baseline metrics: {path}")
            frame = pd.read_csv(path)
            frame["method"] = method
            frame["seed"] = seed
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_method_curve(method_data: pd.DataFrame, method: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    colors = plt.get_cmap("tab10")
    grouped = method_data.groupby(["seed", "episode"])[
        ["raw_reward", "survived_to_limit", "mean_load_ratio", "mean_high_voltage_rate"]
    ].mean().reset_index()
    episodes = sorted(grouped["episode"].unique())
    metrics = ["raw_reward", "survived_to_limit", "mean_load_ratio", "mean_high_voltage_rate"]
    axes_titles = ("Reward Moving Average", "50-Episode Survival Rate", "50-Episode Mean Load Ratio", "50-Episode Mean High-Voltage Rate")
    for seed_index, seed in enumerate(sorted(grouped["seed"].unique())):
        seed_data = grouped[grouped["seed"] == seed].set_index("episode").reindex(episodes)
        color = colors(seed_index % 10)
        label = f"Seed {seed}"
        axes[0, 0].plot(episodes, seed_data["raw_reward"].rolling(50, min_periods=1).mean(), color=color, linewidth=1.5, label=label)
        axes[0, 1].plot(episodes, seed_data["survived_to_limit"].rolling(50, min_periods=1).mean(), color=color, linewidth=1.5, label=label)
        axes[1, 0].plot(episodes, seed_data["mean_load_ratio"].rolling(50, min_periods=1).mean(), color=color, linewidth=1.5, label=label)
        axes[1, 1].plot(episodes, seed_data["mean_high_voltage_rate"].rolling(50, min_periods=1).mean(), color=color, linewidth=1.5, label=label)
    for axis, title in zip(axes.flat, axes_titles):
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[0, 1].set_ylim(-0.02, 1.02)
    fig.suptitle(f"{METHOD_LABELS.get(method, method)} Multi-Seed Training Curves", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(data: pd.DataFrame, output_dir: Path, methods: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        plot_method_curve(data[data["method"] == method], method, output_dir / f"{method}_training_curves.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline multi-seed training curves.")
    parser.add_argument("--input_dir", type=Path, default=Path("baseline_comparison"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=list(METHOD_LABELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    data = load_training_results(args.input_dir, args.methods, args.seeds)
    data.to_csv(args.input_dir / "baseline_multiseed_training_metrics.csv", index=False)
    output_dir = args.output_dir or args.input_dir / "training_curves"
    plot_training_curves(data, output_dir, args.methods)
    print(f"Saved {len(args.methods)} independent 2x2 figures to {output_dir}")


if __name__ == "__main__":
    main()
