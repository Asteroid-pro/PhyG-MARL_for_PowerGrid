from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from env.cpgrid_env import CPGridEnv

POLICIES = ("all_hold", "load_only", "gen_only_mild", "gen_only_strong", "combined_mild", "combined_strong", "voltage_rule", "random")
METRICS = ("load_ratio", "min_voltage", "max_voltage", "low_voltage_rate", "high_voltage_rate", "voltage_violation_rate", "gen_q_total_mvar", "gen_q_abs_total_mvar", "ext_grid_q_total_mvar", "ext_grid_q_abs_total_mvar", "gen_q_limit_hit_count", "gen_q_limit_hit_rate")


def select_actions(policy: str, observations: dict[str, torch.Tensor], support_buses: set[int], rng: np.random.Generator) -> np.ndarray:
    actions = np.zeros(14, dtype=int)
    if policy == "all_hold": return actions
    if policy == "random": return rng.integers(0, 3, size=14, dtype=int)
    if policy == "load_only": actions[:] = 2
    elif policy == "gen_only_mild": actions[list(support_buses)] = 1
    elif policy == "gen_only_strong": actions[list(support_buses)] = 2
    elif policy == "combined_mild": actions[:] = 1
    elif policy == "combined_strong": actions[:] = 2
    elif policy == "voltage_rule":
        for bus in range(14):
            voltage = float(observations[f"agent_{bus}"][0].item())
            actions[bus] = 2 if voltage < 0.90 else 1 if voltage < 0.95 else 0
    else: raise ValueError(f"Unknown policy: {policy}")
    return actions


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def evaluate(policy: str, episodes: int, max_steps: int, base_seed: int, max_breakers: int) -> pd.DataFrame:
    rows = []
    load_enabled = policy not in {"gen_only_mild", "gen_only_strong"}
    generator_enabled = policy != "load_only"
    for episode in range(episodes):
        seed = base_seed + episode
        env = CPGridEnv(
            max_steps=max_steps,
            max_breaker_attacks=max_breakers,
            load_control_enabled=load_enabled,
            generator_control_enabled=generator_enabled,
        )
        observations, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1_000_000)
        values = {name: [] for name in METRICS}
        breaker_values = {name: [] for name in METRICS}
        normal_values = {name: [] for name in METRICS}
        action_counts = np.zeros(3, dtype=int)
        total_reward, steps = 0.0, 0
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated) and steps < max_steps:
            actions = select_actions(policy, observations, env.powerflow_solver.support_buses, rng)
            action_counts += np.bincount(actions, minlength=3)
            observations, reward, terminated, truncated, info = env.step(actions)
            total_reward += float(reward)
            steps += 1
            pf = info.get("powerflow_metrics")
            if pf is not None:
                period = breaker_values if info.get("breaker_active", False) else normal_values
                for name in METRICS:
                    values[name].append(float(pf[name]))
                    period[name].append(float(pf[name]))
        reason = info.get("termination_reason", "unknown")
        attack = info.get("attack_counts", {})
        breaker_state = info.get("breaker_state", {})
        action_total = max(int(action_counts.sum()), 1)
        row = {
            "policy": policy, "episode": episode, "seed": seed, "steps": steps,
            "valid_powerflow_steps": len(values["load_ratio"]),
            "breaker_active_valid_steps": len(breaker_values["load_ratio"]),
            "normal_valid_steps": len(normal_values["load_ratio"]),
            "survived_to_limit": int(truncated and reason == "time_limit"),
            "termination_reason": reason, "total_reward": total_reward,
            "mean_step_reward": total_reward / max(steps, 1),
            "mean_load_ratio": mean_or_nan(values["load_ratio"]),
            "mean_min_voltage": mean_or_nan(values["min_voltage"]),
            "worst_min_voltage": min(values["min_voltage"], default=np.nan),
            "mean_max_voltage": mean_or_nan(values["max_voltage"]),
            "worst_max_voltage": max(values["max_voltage"], default=np.nan),
            "mean_low_voltage_rate": mean_or_nan(values["low_voltage_rate"]),
            "mean_high_voltage_rate": mean_or_nan(values["high_voltage_rate"]),
            "mean_voltage_violation_rate": mean_or_nan(values["voltage_violation_rate"]),
            "mean_gen_q_total_mvar": mean_or_nan(values["gen_q_total_mvar"]),
            "mean_gen_q_abs_total_mvar": mean_or_nan(values["gen_q_abs_total_mvar"]),
            "mean_ext_grid_q_total_mvar": mean_or_nan(values["ext_grid_q_total_mvar"]),
            "mean_ext_grid_q_abs_total_mvar": mean_or_nan(values["ext_grid_q_abs_total_mvar"]),
            "mean_gen_q_limit_hit_count": mean_or_nan(values["gen_q_limit_hit_count"]),
            "mean_gen_q_limit_hit_rate": mean_or_nan(values["gen_q_limit_hit_rate"]),
            "gen_q_limit_hit_step_rate": float(np.mean(np.asarray(values["gen_q_limit_hit_count"]) > 0)) if values["gen_q_limit_hit_count"] else np.nan,
            "fdi_count": int(attack.get("fdi", 0)), "dos_count": int(attack.get("dos", 0)),
            "breaker_count": int(attack.get("breaker", 0)),
            "restored_line_count": int(breaker_state.get("restored_line_count", 0)),
            "max_concurrent_breakers": int(breaker_state.get("max_concurrent_breakers", 0)),
            "steps_under_breaker": int(breaker_state.get("steps_under_breaker", 0)),
            "action_0_ratio": action_counts[0] / action_total, "action_1_ratio": action_counts[1] / action_total, "action_2_ratio": action_counts[2] / action_total,
        }
        for prefix, period in (("breaker", breaker_values), ("normal", normal_values)):
            row[f"{prefix}_mean_load_ratio"] = mean_or_nan(period["load_ratio"])
            row[f"{prefix}_mean_min_voltage"] = mean_or_nan(period["min_voltage"])
            row[f"{prefix}_worst_min_voltage"] = min(period["min_voltage"], default=np.nan)
            row[f"{prefix}_mean_max_voltage"] = mean_or_nan(period["max_voltage"])
            row[f"{prefix}_mean_low_voltage_rate"] = mean_or_nan(period["low_voltage_rate"])
            row[f"{prefix}_mean_high_voltage_rate"] = mean_or_nan(period["high_voltage_rate"])
            row[f"{prefix}_mean_voltage_violation_rate"] = mean_or_nan(period["voltage_violation_rate"])
            row[f"{prefix}_mean_gen_q_abs_total_mvar"] = mean_or_nan(period["gen_q_abs_total_mvar"])
            row[f"{prefix}_mean_ext_grid_q_abs_total_mvar"] = mean_or_nan(period["ext_grid_q_abs_total_mvar"])
        rows.append(row)
        env.close()
    return pd.DataFrame(rows)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    records = []
    averaged = [column for column in details.columns if column.startswith(("mean_", "breaker_mean_", "normal_mean_"))]
    for policy, frame in details.groupby("policy", sort=False):
        reason = frame["termination_reason"]
        record = {"policy": policy, "episodes": len(frame), "mean_steps": frame["steps"].mean(), "std_steps": frame["steps"].std(ddof=1), "survival_rate": frame["survived_to_limit"].mean(), "time_limit_rate": (reason == "time_limit").mean(), "powerflow_failure_rate": (reason == "powerflow_not_converged").mean(), "invalid_voltage_rate": (reason == "invalid_voltage").mean(), "voltage_collapse_rate": (reason == "voltage_collapse").mean(), "unknown_termination_rate": (~reason.isin(["time_limit", "powerflow_not_converged", "invalid_voltage", "voltage_collapse"])).mean(), "worst_min_voltage": frame["worst_min_voltage"].min(), "worst_max_voltage": frame["worst_max_voltage"].max(), "mean_breaker_count": frame["breaker_count"].mean(), "mean_steps_under_breaker": frame["steps_under_breaker"].mean()}
        record.update({column: frame[column].mean() for column in averaged})
        records.append(record)
    return pd.DataFrame(records)


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(summary)); labels = summary["policy"]
    fig, axes = plt.subplots(4, 1, figsize=(15, 17), sharex=True)
    axes[0].bar(x - .2, summary["survival_rate"], .2, label="survival"); axes[0].bar(x, summary["powerflow_failure_rate"], .2, label="powerflow failure"); axes[0].bar(x + .2, summary["invalid_voltage_rate"], .2, label="invalid voltage")
    axes[1].bar(x - .15, summary["breaker_mean_low_voltage_rate"], .3, label="low"); axes[1].bar(x + .15, summary["breaker_mean_high_voltage_rate"], .3, label="high")
    axes[2].bar(x - .15, summary["normal_mean_low_voltage_rate"], .3, label="low"); axes[2].bar(x + .15, summary["normal_mean_high_voltage_rate"], .3, label="high")
    axes[3].bar(x, summary["mean_load_ratio"]); axes[3].set_ylabel("Mean Load Ratio")
    axes[0].set_ylabel("Termination Rates"); axes[1].set_ylabel("Breaker Voltage Rates"); axes[2].set_ylabel("Normal Voltage Rates")
    axes[3].set_xticks(x, labels, rotation=30, ha="right")
    for axis in axes: axis.grid(True, axis="y", alpha=.3); axis.legend() if axis is not axes[3] else None
    fig.tight_layout(); fig.savefig(output_dir / "generator_support_comparison.png", dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(14, 6)); axis.bar(x - .2, summary["mean_gen_q_abs_total_mvar"], .4, label="gen"); axis.bar(x + .2, summary["mean_ext_grid_q_abs_total_mvar"], .4, label="ext_grid"); axis.set_xticks(x, labels, rotation=30, ha="right"); axis.set_ylabel("Mean Absolute Q (Mvar)"); axis.grid(True, axis="y", alpha=.3); axis.legend(); fig.tight_layout(); fig.savefig(output_dir / "generator_support_reactive_power.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate heterogeneous generator voltage support policies.")
    parser.add_argument("--episodes", type=int, default=100); parser.add_argument("--steps", type=int, default=200); parser.add_argument("--max_breakers", type=int, default=6); parser.add_argument("--base_seed", type=int, default=10000); parser.add_argument("--output_dir", type=Path, default=Path("results_generator_support")); args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    details = pd.concat([evaluate(policy, args.episodes, args.steps, args.base_seed, args.max_breakers) for policy in POLICIES], ignore_index=True); summary = summarize(details)
    details.to_csv(args.output_dir / "generator_support_details.csv", index=False); summary.to_csv(args.output_dir / "generator_support_summary.csv", index=False); plot_results(summary, args.output_dir)
    print(summary.to_string(index=False)); print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__": main()
