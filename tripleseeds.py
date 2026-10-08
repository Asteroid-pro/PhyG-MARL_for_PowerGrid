from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fusion import plot_comparison


DEFAULT_SEEDS = (42, 43, 44)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run V3.3 joint training sequentially for seeds 42, 43, and 44."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("result_joint_v3_3_multiple_seeds"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in DEFAULT_SEEDS:
        seed_dir = args.output_dir / f"results_joint_v3_3_seed{seed}"
        command = [
            sys.executable,
            "train.py",
            "--episodes", str(args.episodes),
            "--steps", str(args.steps),
            "--seed", str(seed),
            "--representation_mode", "joint",
            "--output_dir", str(seed_dir),
        ]
        print(f"\n{'=' * 72}")
        print(f"Starting V3.3 joint training: seed={seed}, output={seed_dir}")
        print(f"{'=' * 72}")
        subprocess.run(command, check=True)
        print(f"Seed {seed} completed successfully.")

    frames = []
    for seed in DEFAULT_SEEDS:
        csv_path = args.output_dir / f"results_joint_v3_3_seed{seed}" / "training_metrics_advanced.csv"
        frame = pd.read_csv(csv_path)
        frame["experiment_seed"] = seed
        frame["source_directory"] = csv_path.parent.name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined_path = args.output_dir / "joint_multiseed_training_metrics.csv"
    figure_path = args.output_dir / "joint_v3_3_multiseed_core_metrics.png"
    combined.to_csv(combined_path, index=False)
    plot_comparison(combined, list(DEFAULT_SEEDS), figure_path)

    print(f"All three seeds completed. Results saved under: {args.output_dir}")
    print(f"Combined metrics saved to: {combined_path}")
    print(f"Comparison figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
