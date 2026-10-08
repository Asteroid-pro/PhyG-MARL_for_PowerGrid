from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import ACTION_DIM, ACTOR_HIDDEN_DIM, DEVICE, EDGE_IN_DIM, GRAPH_NUM_LAYERS, HIDDEN_DIM, MEMORY_HISTORY_LEN, NODE_IN_DIM, NUM_AGENTS
from env.cpgrid_env import CPGridEnv
from graph.graph_encoder import GraphEncoder
from memory.transformer_memory import TemporalMemory
from policy.actor import GraphActor

FIXED_METHODS = ("all_hold", "load_only_mild", "load_only_strong", "gen_only_mild", "combined_mild", "random", "voltage_rule")


def make_env(steps: int, method: str) -> CPGridEnv:
    return CPGridEnv(
        max_steps=steps, attack_probability=0.35,
        attack_weights={"fdi": 0.45, "dos": 0.45, "breaker": 0.10},
        max_breaker_attacks=6, exclude_islanding_breakers=True,
        breaker_transient=True, breaker_duration_range=(10, 20), max_active_breakers=2,
        load_control_enabled=method not in {"gen_only_mild"},
        generator_control_enabled=method not in {"load_only_mild", "load_only_strong"},
        enforce_q_lims=True,
    )


def fixed_actions(method: str, observations: dict, rng: np.random.Generator) -> np.ndarray:
    if method == "all_hold": return np.zeros(NUM_AGENTS, dtype=int)
    if method in {"load_only_mild", "gen_only_mild", "combined_mild"}: return np.ones(NUM_AGENTS, dtype=int)
    if method == "load_only_strong": return np.full(NUM_AGENTS, 2, dtype=int)
    if method == "random": return rng.integers(0, 3, size=NUM_AGENTS)
    actions = np.zeros(NUM_AGENTS, dtype=int)
    for index in range(NUM_AGENTS):
        voltage = float(torch.as_tensor(observations[f"agent_{index}"])[0])
        actions[index] = 2 if voltage < 0.95 else (1 if voltage < 0.98 else 0)
    return actions


def load_mappo(checkpoint_path: Path, steps: int):
    env = make_env(steps, "mappo")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    graph = GraphEncoder(NODE_IN_DIM, EDGE_IN_DIM, HIDDEN_DIM, GRAPH_NUM_LAYERS).to(DEVICE)
    memory = TemporalMemory(HIDDEN_DIM, MEMORY_HISTORY_LEN).to(DEVICE)
    actor = GraphActor(env.observation_space["agent_0"].shape[0], HIDDEN_DIM, ACTION_DIM, ACTOR_HIDDEN_DIM).to(DEVICE)
    graph.load_state_dict(checkpoint["graph_encoder_state_dict"])
    memory.load_state_dict(checkpoint["memory_state_dict"])
    actor.load_state_dict(checkpoint["actor_state_dict"])
    graph.eval(); memory.eval(); actor.eval()
    return env, graph, memory, actor, checkpoint


def evaluate(method: str, test_seeds: list[int], steps: int, training_seed: int | None = None, checkpoint_path: Path | None = None) -> pd.DataFrame:
    if checkpoint_path:
        env, graph, memory, actor, checkpoint = load_mappo(checkpoint_path, steps)
        checkpoint_episode = int(checkpoint.get("episode", -1))
    else:
        env, graph, memory, actor, checkpoint_episode = make_env(steps, method), None, None, None, -1
    rows = []
    with torch.no_grad():
        for test_seed in test_seeds:
            obs, info = env.reset(seed=test_seed)
            if memory is not None: memory.clear()
            rng = np.random.default_rng(test_seed + 100000)
            done = truncated = False; count = 0; reward_sum = 0.0
            values = {k: [] for k in ("load_ratio", "low_voltage_rate", "high_voltage_rate")}
            action_counts = np.zeros(3, dtype=int)
            while not (done or truncated) and count < steps:
                if actor is None:
                    actions = fixed_actions(method, obs, rng)
                else:
                    node, edge, edge_index, injection = env.get_graph_data()
                    _, embedding = graph(node.to(DEVICE), edge.to(DEVICE), edge_index.to(DEVICE), torch.as_tensor(injection, dtype=torch.float32, device=DEVICE))
                    memory.update(embedding)
                    batch = torch.stack([torch.as_tensor(obs[f"agent_{i}"], dtype=torch.float32, device=DEVICE) for i in range(NUM_AGENTS)])
                    actions = torch.argmax(actor(batch, memory.get_memory().to(DEVICE)), dim=-1).cpu().numpy()
                action_counts += np.bincount(actions, minlength=3)
                obs, reward, done, truncated, info = env.step(actions)
                reward_sum += float(reward); count += 1
                metrics = info.get("powerflow_metrics")
                if metrics:
                    for key in values: values[key].append(float(metrics[key]))
            total_actions = max(action_counts.sum(), 1)
            rows.append({"method": method, "training_seed": training_seed, "checkpoint_episode": checkpoint_episode, "test_seed": test_seed, "survived_to_limit": int(truncated and info.get("termination_reason") == "time_limit"), "steps": count, "total_reward": reward_sum, "termination_reason": info.get("termination_reason", "unknown"), **{f"mean_{k}": np.mean(v) if v else np.nan for k, v in values.items()}, **{f"action_{i}_ratio": action_counts[i] / total_actions for i in range(3)}})
    env.close()
    return pd.DataFrame(rows)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "training_seed"]
    rows = []
    for group_key, frame in data.groupby(keys, dropna=False):
        p = frame.survived_to_limit.mean(); n = len(frame); margin = 1.96 * np.sqrt(p * (1 - p) / n)
        rows.append({"method": group_key[0], "training_seed": group_key[1], "episodes": n, "survival_rate": p, "survival_ci95_low": max(0, p-margin), "survival_ci95_high": min(1, p+margin), "mean_steps": frame.steps.mean(), "mean_reward": frame.total_reward.mean(), "mean_load_ratio": frame.mean_load_ratio.mean(), "mean_high_voltage_rate": frame.mean_high_voltage_rate.mean(), "powerflow_not_converged_rate": (frame.termination_reason == "powerflow_not_converged").mean(), "invalid_voltage_rate": (frame.termination_reason == "invalid_voltage").mean()})
    return pd.DataFrame(rows)


def paired(data: pd.DataFrame) -> pd.DataFrame:
    mappo = data[data.method == "mappo_best"].groupby("test_seed").agg({"survived_to_limit":"mean", "total_reward":"mean", "mean_load_ratio":"mean", "mean_high_voltage_rate":"mean"})
    rows=[]
    for method in FIXED_METHODS:
        base=data[data.method==method].set_index("test_seed")
        rows.append({"baseline":method,"mappo_success_baseline_fail":int(((mappo.survived_to_limit==1)&(base.survived_to_limit==0)).sum()),"baseline_success_mappo_fail":int(((mappo.survived_to_limit==0)&(base.survived_to_limit==1)).sum()),"reward_difference":(mappo.total_reward-base.total_reward).mean(),"load_ratio_difference":(mappo.mean_load_ratio-base.mean_load_ratio).mean(),"high_voltage_rate_difference":(mappo.mean_high_voltage_rate-base.mean_high_voltage_rate).mean()})
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, path: Path):
    fixed=summary[summary.method!="mappo_best"]; mappo=summary[summary.method=="mappo_best"]
    display=pd.concat([fixed, pd.DataFrame([{"method":"mappo_best_mean", **{c:mappo[c].mean() for c in ["survival_rate","mean_reward","mean_load_ratio","mean_high_voltage_rate"]}}])], ignore_index=True)
    fig, axes=plt.subplots(2,2,figsize=(15,10)); metrics=[("survival_rate","Survival Rate"),("mean_reward","Reward"),("mean_load_ratio","Load Ratio"),("mean_high_voltage_rate","High-Voltage Rate")]
    for ax,(column,title) in zip(axes.flat,metrics): ax.bar(display.method,display[column]); ax.set_title(title); ax.tick_params(axis="x",rotation=35); ax.grid(axis="y",alpha=.3)
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description="Evaluate fixed baselines and MAPPO checkpoints on common test seeds.")
    parser.add_argument("--checkpoint_root",type=Path,default=Path("results_actor_critic_v2_2_2_multiseed_parallel_1000eps")); parser.add_argument("--output_dir",type=Path,default=Path("baseline_evaluation")); parser.add_argument("--training_seeds",nargs="+",type=int,default=[42,43,44]); parser.add_argument("--test_seed_start",type=int,default=60000); parser.add_argument("--test_episodes",type=int,default=100); parser.add_argument("--steps",type=int,default=200)
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); test_seeds=list(range(args.test_seed_start,args.test_seed_start+args.test_episodes)); frames=[]
    for method in FIXED_METHODS: print("Evaluating",method); frames.append(evaluate(method,test_seeds,args.steps))
    for seed in args.training_seeds: print("Evaluating MAPPO seed",seed); frames.append(evaluate("mappo_best",test_seeds,args.steps,seed,args.checkpoint_root/f"seed_{seed}"/"best_balanced.pt"))
    data=pd.concat(frames,ignore_index=True); summary=summarize(data); data.to_csv(args.output_dir/"baseline_episode_results.csv",index=False); summary.to_csv(args.output_dir/"baseline_summary.csv",index=False); paired(data).to_csv(args.output_dir/"paired_comparison.csv",index=False); plot(summary,args.output_dir/"baseline_comparison.png"); print("Baseline evaluation completed:",args.output_dir)


if __name__=="__main__": main()
