"""
Baseline 并行启动器：将方法与训练 seed 拆分为独立进程，并限制并发任务数和 CPU 线程数。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from baseline_config import SEEDS, TRAINABLE_METHODS


def run_task(
    method: str,
    seed: int,
    episodes: int,
    steps: int,
    output_dir: Path,
    threads_per_task: int,
) -> tuple[str, int]:
    environment = os.environ.copy()
    thread_count = str(threads_per_task)
    environment.update({
        "OMP_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
        "NUMEXPR_NUM_THREADS": thread_count,
    })
    command = [
        sys.executable,
        "run_baseline_suite.py",
        "--methods", method,
        "--seeds", str(seed),
        "--episodes", str(episodes),
        "--steps", str(steps),
        "--output_dir", str(output_dir),
    ]
    print(f"Starting method={method}, seed={seed}", flush=True)
    subprocess.run(command, check=True, env=environment)
    print(f"Completed method={method}, seed={seed}", flush=True)
    return method, seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run independent baseline method/seed jobs with bounded parallelism."
    )
    parser.add_argument(
        "--methods", nargs="+", choices=TRAINABLE_METHODS, default=list(TRAINABLE_METHODS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output_dir", type=Path, default=Path("baseline_comparison"))
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--threads_per_task", type=int, default=6)
    args = parser.parse_args()

    if args.max_workers < 1 or args.threads_per_task < 1:
        parser.error("--max_workers and --threads_per_task must be positive")

    tasks = [(method, seed) for method in args.methods for seed in args.seeds]
    print(
        f"Launching {len(tasks)} tasks with max_workers={args.max_workers}, "
        f"threads_per_task={args.threads_per_task}",
        flush=True,
    )

    failures: list[tuple[str, int, Exception]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                run_task,
                method,
                seed,
                args.episodes,
                args.steps,
                args.output_dir,
                args.threads_per_task,
            ): (method, seed)
            for method, seed in tasks
        }
        for future in as_completed(futures):
            method, seed = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append((method, seed, error))
                print(f"Failed method={method}, seed={seed}: {error}", flush=True)

    if failures:
        details = ", ".join(f"{method}/seed_{seed}" for method, seed, _ in failures)
        raise SystemExit(f"{len(failures)} baseline task(s) failed: {details}")

    print(f"All baseline tasks completed successfully: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
