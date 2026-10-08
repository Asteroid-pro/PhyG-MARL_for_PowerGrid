"""
模型开销测试工具：统计总参数量、可训练参数量、模型大小和策略推理延迟。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline_config import BaselineConfig, TRAINABLE_METHODS
from baselines.common.env_adapter import global_state, make_env, observation_tensor
from baselines.gat_mappo.gat_encoder import GATEncoder
from baselines.gru_mappo.gru_policy import GRUPolicy
from baselines.mat.mat_policy import MATPolicy
from baselines.mappo.agent import CentralCritic, SharedActor
from baselines.qmix.agent import QMIXRuntime
from config import ACTOR_HIDDEN_DIM, DEVICE, EDGE_IN_DIM, GRAPH_NUM_LAYERS, HIDDEN_DIM, MEMORY_HISTORY_LEN, NODE_IN_DIM, NUM_AGENTS
from graph.graph_encoder import GraphEncoder
from memory.transformer_memory import TemporalMemory
from policy.actor import GraphActor
from run_baseline_suite import MethodRuntime


def parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable


def checkpoint_path(root: Path, method: str, seed: int) -> Path:
    path = root / method / f"seed_{seed}" / "best_balanced.pt"
    if not path.exists():
        path = root / method / f"seed_{seed}" / "final.pt"
    return path


def load_baseline(method: str, root: Path, seed: int, env, config: BaselineConfig):
    obs_dim = env.observation_space["agent_0"].shape[0]
    state_dim = NUM_AGENTS * 4 + NUM_AGENTS + 2
    runtime = MethodRuntime(method, obs_dim, state_dim, config)
    checkpoint = torch.load(checkpoint_path(root, method, seed), map_location=DEVICE)
    if method == "qmix":
        agent_q = checkpoint.get("agent_q_state_dict", checkpoint.get("actor_state_dict"))
        mixer = checkpoint.get("mixer_state_dict", checkpoint.get("critic_state_dict"))
        runtime.qmix.agent_q.load_state_dict(agent_q)
        runtime.qmix.mixer.load_state_dict(mixer)
        runtime.qmix.target_agent_q.load_state_dict(agent_q)
        runtime.qmix.target_mixer.load_state_dict(mixer)
        modules = [runtime.qmix.agent_q, runtime.qmix.mixer]
    else:
        runtime.actor.load_state_dict(checkpoint["actor_state_dict"])
        runtime.critic.load_state_dict(checkpoint["critic_state_dict"])
        modules = list(runtime.actor.modules()) + list(runtime.critic.modules())
    return runtime, modules


def build_phyg(root: Path, seed: int, env):
    path = root / f"seed_{seed}" / "best_balanced.pt"
    checkpoint = torch.load(path, map_location=DEVICE)
    graph = GraphEncoder(NODE_IN_DIM, EDGE_IN_DIM, HIDDEN_DIM, GRAPH_NUM_LAYERS).to(DEVICE)
    memory = TemporalMemory(HIDDEN_DIM, MEMORY_HISTORY_LEN).to(DEVICE)
    obs_dim = env.observation_space["agent_0"].shape[0]
    actor = GraphActor(obs_dim, HIDDEN_DIM, 3, hidden_dim=ACTOR_HIDDEN_DIM, use_memory_gate=False).to(DEVICE)
    critic = torch.nn.Sequential(torch.nn.Linear(NUM_AGENTS * 4 + NUM_AGENTS + 2, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(DEVICE)
    graph.load_state_dict(checkpoint["graph_encoder_state_dict"])
    memory.load_state_dict(checkpoint["memory_state_dict"])
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    return graph, memory, actor, critic


def timed_call(function, warmup: int, iterations: int) -> list[float]:
    with torch.no_grad():
        for _ in range(warmup):
            function()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(iterations):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            function()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values), value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark parameter counts and deterministic policy inference latency.")
    parser.add_argument("--baseline_root", type=Path, default=Path("baseline_comparison"))
    parser.add_argument("--phyg_root", type=Path, default=Path("results_actor_critic_v2_2_2_multiseed_parallel_1000eps"))
    parser.add_argument("--output_dir", type=Path, default=Path("baseline_comparison/benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--methods", nargs="+", default=list(TRAINABLE_METHODS) + ["phyg_v2_2_2"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(200); observations, _ = env.reset(seed=60000)
    node, edge, edge_index, injection = env.get_graph_data()
    state = global_state(env, node, 200).to(DEVICE); obs = observation_tensor(observations, DEVICE)
    config = BaselineConfig()
    rows = []
    for method in args.methods:
        if method == "phyg_v2_2_2":
            graph, memory, actor, critic = build_phyg(args.phyg_root, args.seed, env)
            memory.clear(); graph.eval(); memory.eval(); actor.eval(); critic.eval()
            def call():
                _, embedding = graph(node.to(DEVICE), edge.to(DEVICE), edge_index.to(DEVICE), injection.to(DEVICE)); memory.update(embedding); vector = memory.get_memory().to(DEVICE); actor(obs, vector)
            modules = [graph, memory, actor, critic]
        else:
            runtime, modules = load_baseline(method, args.baseline_root, args.seed, env, config)
            runtime.reset_episode()
            graph_data = (node, edge, edge_index, injection)
            def call(runtime=runtime, method=method, graph_data=graph_data):
                runtime.act(obs, state, graph_data, deterministic=True)
        samples = timed_call(call, args.warmup, args.iterations)
        total = sum(parameters(module)[0] for module in modules); trainable = sum(parameters(module)[1] for module in modules)
        model_size = sum(parameter.numel() * parameter.element_size() for module in modules for parameter in module.parameters()) / (1024 ** 2)
        rows.append({"method": method, "seed": args.seed, "total_parameters": total, "trainable_parameters": trainable, "frozen_parameters": total - trainable, "model_size_mb": model_size, "mean_latency_ms": statistics.mean(samples), "p50_latency_ms": percentile(samples, 50), "p95_latency_ms": percentile(samples, 95), "decisions_per_second": 1000.0 / statistics.mean(samples)})
    pd.DataFrame(rows).to_csv(args.output_dir / "model_cost_summary.csv", index=False)
    env.close()
    print(f"Saved benchmark results to {args.output_dir / 'model_cost_summary.csv'}")


if __name__ == "__main__":
    main()
