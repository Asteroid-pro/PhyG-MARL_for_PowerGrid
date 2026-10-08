from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaselineAgent(ABC):
    """Common contract used by every independently implemented baseline."""

    @abstractmethod
    def act(self, observations: dict[str, Any], global_state: torch.Tensor, deterministic: bool = False) -> tuple[Any, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def update(self) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError
