from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_multiseed_experiment import load_results, plot_core_metrics


def cpu_ranges(count: int, cores_per_process: int) -> list[str]:
    return [f"{index * cores_per_process}-{(index + 1) * cores_per_process - 1}" for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent training seeds concurrently on Linux.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--cores_per_process", type=int, default=10)
    parser.add_argument("--no_affinity", action="store_true", help="Do not bind processes with taskset.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results_actor_critic_v2_2_2_multiseed_parallel_1000eps"),
    )
    args = parser.parse_args()
    if args.cores_per_process < 1:
        parser.error("--cores_per_process must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranges = cpu_ranges(len(args.seeds), args.cores_per_process)
    processes: list[tuple[int, subprocess.Popen, object]] = []

    for index, seed in enumerate(args.seeds):
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "train.py",
            "--seed", str(seed),
            "--episodes", str(args.episodes),
            "--steps", str(args.steps),
            "--success_threshold", "0.0",
            "--output_dir", str(seed_dir),
        ]
        if not args.no_affinity:
            command = ["taskset", "-c", ranges[index], *command]
        environment = os.environ.copy()
        thread_count = str(args.cores_per_process)
        environment.update({
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
        })
        log_file = (seed_dir / "train.log").open("w", encoding="utf-8", buffering=1)
        print(f"Starting seed {seed} on CPU {ranges[index] if not args.no_affinity else 'unbound'}; log: {seed_dir / 'train.log'}")
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
        )
        processes.append((seed, process, log_file))

    failures = []
    try:
        for seed, process, log_file in processes:
            return_code = process.wait()
            log_file.close()
            if return_code != 0:
                failures.append((seed, return_code))
            print(f"Seed {seed} finished with exit code {return_code}")
    except KeyboardInterrupt:
        print("Stopping all training processes...")
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, log_file in processes:
            process.wait()
            log_file.close()
        raise

    if failures:
        details = ", ".join(f"seed {seed}: exit {code}" for seed, code in failures)
        raise RuntimeError(f"Parallel experiment failed: {details}")

    combined = load_results(args.output_dir, args.seeds)
    combined_path = args.output_dir / "multiseed_training_metrics.csv"
    figure_path = args.output_dir / "multiseed_core_metrics.png"
    combined.to_csv(combined_path, index=False)
    plot_core_metrics(combined, args.seeds, figure_path)
    print(f"Combined metrics saved to: {combined_path}")
    print(f"Comparison figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
