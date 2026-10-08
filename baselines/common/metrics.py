from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def write_episode_metrics(rows: Iterable[dict], path: Path) -> None:
    frame = pd.DataFrame(list(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def summarize_evaluations(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["method", "training_seed"], as_index=False)
    summary = grouped.agg(
        survival_rate=("survived_to_limit", "mean"),
        mean_steps=("steps", "mean"),
        mean_reward=("total_reward", "mean"),
        mean_load_ratio=("mean_load_ratio", "mean"),
        mean_low_voltage_rate=("mean_low_voltage_rate", "mean"),
        mean_high_voltage_rate=("mean_high_voltage_rate", "mean"),
    )
    summary["balanced_score"] = summary["survival_rate"] + 0.5 * summary["mean_load_ratio"] - 0.5 * summary["mean_high_voltage_rate"]
    return summary
