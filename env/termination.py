from __future__ import annotations

from typing import Any

import numpy as np


class Termination:
    def __init__(self, max_steps: int = 200, critical_voltage: float = 0.7, critical_bus_fraction: float = 0.3):
        self.max_steps = max_steps
        self.critical_voltage = critical_voltage
        self.critical_bus_fraction = critical_bus_fraction

    def check(self, pf_result: Any, step_count: int) -> tuple[bool, bool, str | None]:
        if pf_result is None:
            return True, False, "powerflow_not_converged"
        voltage = np.asarray(pf_result.get("voltage_pu", []), dtype=float)
        if voltage.size == 0 or not np.all(np.isfinite(voltage)):
            return True, False, "invalid_voltage"
        if float(np.mean(voltage < self.critical_voltage)) >= self.critical_bus_fraction:
            return True, False, "voltage_collapse"
        if step_count >= self.max_steps:
            return False, True, "time_limit"
        return False, False, None
