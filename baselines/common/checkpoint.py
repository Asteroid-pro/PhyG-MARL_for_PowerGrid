from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def choose_best(summary: list[dict[str, float]]) -> dict[str, float]:
    return max(summary, key=lambda row: float(row.get("balanced_score", float("-inf"))))
