from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

import numpy as np


@dataclass
class EpisodeMetricsLogger:
    success_threshold: float = 0.0
    reward_history: List[float] = field(default_factory=list)
    window_50: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    window_100: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    steps_window_50: Deque[int] = field(default_factory=lambda: deque(maxlen=50))
    survived_window_50: Deque[bool] = field(default_factory=lambda: deque(maxlen=50))

    def update(self, total_reward: float, steps: int, survived_to_limit: bool):
        self.reward_history.append(float(total_reward))
        self.window_50.append(float(total_reward))
        self.window_100.append(float(total_reward))
        self.steps_window_50.append(int(steps))
        self.survived_window_50.append(bool(survived_to_limit))

    def summary(self):
        raw_reward = self.reward_history[-1]
        avg_50 = float(np.mean(self.window_50)) if self.window_50 else raw_reward
        avg_100 = float(np.mean(self.window_100)) if self.window_100 else raw_reward
        std_50 = float(np.std(self.window_50)) if self.window_50 else 0.0
        success_rate_50 = float(np.mean(np.array(self.window_50) > self.success_threshold)) if self.window_50 else 0.0
        steps_avg_50 = float(np.mean(self.steps_window_50)) if self.steps_window_50 else 0.0
        steps_std_50 = float(np.std(self.steps_window_50)) if self.steps_window_50 else 0.0
        survival_rate_50 = float(np.mean(self.survived_window_50)) if self.survived_window_50 else 0.0
        return (
            raw_reward,
            avg_50,
            avg_100,
            std_50,
            success_rate_50,
            steps_avg_50,
            steps_std_50,
            survival_rate_50,
        )
