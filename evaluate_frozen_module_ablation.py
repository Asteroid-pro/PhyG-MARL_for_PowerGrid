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

MODES = ("full", "no_gnn", "no_transformer", "local_only")


def make_env(steps: int) -> CPGridEnv:
    return CPGridEnv(max_steps=steps, attack_probability=0.35,
        attack_weights={"fdi": 0.45, "dos": 0.45, "breaker": 0.10},
        max_breaker_attacks=6, exclude_islanding_breakers=True,
        breaker_transient=True, breaker_duration_range=(10, 20), max_active_breakers=2,
        load_control_enabled=True, generator_control_enabled=True, enforce_q_lims=True)


def load_modules(root: Path, training_seed: int, steps: int):
    env = make_env(steps); checkpoint_path=root/f"seed_{training_seed}"/"best_balanced.pt"; checkpoint=torch.load(checkpoint_path,map_location=DEVICE)
    graph=GraphEncoder(NODE_IN_DIM,EDGE_IN_DIM,HIDDEN_DIM,GRAPH_NUM_LAYERS).to(DEVICE); memory=TemporalMemory(HIDDEN_DIM,MEMORY_HISTORY_LEN).to(DEVICE); actor=GraphActor(env.observation_space["agent_0"].shape[0],HIDDEN_DIM,ACTION_DIM,ACTOR_HIDDEN_DIM).to(DEVICE)
    graph.load_state_dict(checkpoint["graph_encoder_state_dict"]); memory.load_state_dict(checkpoint["memory_state_dict"]); actor.load_state_dict(checkpoint["actor_state_dict"]); graph.eval(); memory.eval(); actor.eval()
    return env,graph,memory,actor,int(checkpoint.get("episode",-1))


def memory_input(mode: str, env: CPGridEnv, graph: GraphEncoder, memory: TemporalMemory) -> torch.Tensor:
    if mode == "local_only": return torch.zeros(HIDDEN_DIM,device=DEVICE)
    if mode == "no_gnn":
        memory.update(torch.zeros(HIDDEN_DIM,device=DEVICE)); return memory.get_memory().to(DEVICE)
    node,edge,index,injection=env.get_graph_data(); _,embedding=graph(node.to(DEVICE),edge.to(DEVICE),index.to(DEVICE),torch.as_tensor(injection,dtype=torch.float32,device=DEVICE))
    if mode == "no_transformer": return embedding
    memory.update(embedding); return memory.get_memory().to(DEVICE)


def evaluate(root: Path, seed: int, mode: str, test_seeds: list[int], steps: int) -> pd.DataFrame:
    env,graph,memory,actor,checkpoint_episode=load_modules(root,seed,steps); rows=[]
    with torch.no_grad():
        for test_seed in test_seeds:
            obs,info=env.reset(seed=test_seed); memory.clear(); done=truncated=False; episode_steps=0; reward_sum=0.0; action_counts=np.zeros(3,dtype=int); values={k:[] for k in ("load_ratio","low_voltage_rate","high_voltage_rate")}
            while not (done or truncated) and episode_steps < steps:
                mem=memory_input(mode,env,graph,memory); batch=torch.stack([torch.as_tensor(obs[f"agent_{i}"],dtype=torch.float32,device=DEVICE) for i in range(NUM_AGENTS)]); actions=torch.argmax(actor(batch,mem),dim=-1).cpu().numpy(); action_counts += np.bincount(actions,minlength=3)
                obs,reward,done,truncated,info=env.step(actions); reward_sum += float(reward); episode_steps += 1; metrics=info.get("powerflow_metrics")
                if metrics:
                    for key in values: values[key].append(float(metrics[key]))
            total=max(action_counts.sum(),1); rows.append({"ablation_mode":mode,"training_seed":seed,"checkpoint_episode":checkpoint_episode,"test_seed":test_seed,"survived_to_limit":int(truncated and info.get("termination_reason")=="time_limit"),"steps":episode_steps,"total_reward":reward_sum,"termination_reason":info.get("termination_reason","unknown"),**{f"mean_{k}":np.mean(v) if v else np.nan for k,v in values.items()},**{f"action_{i}_ratio":action_counts[i]/total for i in range(3)}})
    env.close(); return pd.DataFrame(rows)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for key,x in data.groupby(["ablation_mode","training_seed"]):
        p=x.survived_to_limit.mean(); margin=1.96*np.sqrt(p*(1-p)/len(x)); rows.append({"ablation_mode":key[0],"training_seed":key[1],"episodes":len(x),"survival_rate":p,"survival_ci95_low":max(0,p-margin),"survival_ci95_high":min(1,p+margin),"mean_steps":x.steps.mean(),"mean_reward":x.total_reward.mean(),"mean_load_ratio":x.mean_load_ratio.mean(),"mean_high_voltage_rate":x.mean_high_voltage_rate.mean(),"powerflow_not_converged_rate":(x.termination_reason=="powerflow_not_converged").mean(),"invalid_voltage_rate":(x.termination_reason=="invalid_voltage").mean(),"action_0_ratio":x.action_0_ratio.mean(),"action_1_ratio":x.action_1_ratio.mean(),"action_2_ratio":x.action_2_ratio.mean()})
    return pd.DataFrame(rows)


def paired(data: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for seed,x in data.groupby("training_seed"):
        full=x[x.ablation_mode=="full"].set_index("test_seed")
        for mode in MODES[1:]:
            other=x[x.ablation_mode==mode].set_index("test_seed"); rows.append({"training_seed":seed,"ablation_mode":mode,"full_success_ablation_fail":int(((full.survived_to_limit==1)&(other.survived_to_limit==0)).sum()),"ablation_success_full_fail":int(((full.survived_to_limit==0)&(other.survived_to_limit==1)).sum()),"reward_change_vs_full":(other.total_reward-full.total_reward).mean(),"load_ratio_change_vs_full":(other.mean_load_ratio-full.mean_load_ratio).mean(),"high_voltage_change_vs_full":(other.mean_high_voltage_rate-full.mean_high_voltage_rate).mean(),"action_change_rate":float(np.mean(np.any(np.column_stack([other[f"action_{i}_ratio"] for i in range(3)])!=np.column_stack([full[f"action_{i}_ratio"] for i in range(3)]),axis=1)))})
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame,path: Path):
    mean=summary.groupby("ablation_mode")[["survival_rate","mean_reward","mean_load_ratio","mean_high_voltage_rate"]].agg(["mean","std"]).reindex(MODES); fig,axes=plt.subplots(2,2,figsize=(14,10)); metrics=[("survival_rate","Survival Rate"),("mean_reward","Reward"),("mean_load_ratio","Load Ratio"),("mean_high_voltage_rate","High-Voltage Rate")]
    for ax,(metric,title) in zip(axes.flat,metrics): ax.bar(MODES,mean[(metric,"mean")],yerr=mean[(metric,"std")],capsize=5); ax.set_title(title); ax.grid(axis="y",alpha=.3)
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description="Zero-shot inference ablation of frozen GNN and Transformer modules."); p.add_argument("--checkpoint_root",type=Path,default=Path("results_actor_critic_v2_2_2_multiseed_parallel_1000eps")); p.add_argument("--output_dir",type=Path,default=Path("frozen_module_ablation")); p.add_argument("--training_seeds",nargs="+",type=int,default=[42,43,44]); p.add_argument("--test_seed_start",type=int,default=60000); p.add_argument("--test_episodes",type=int,default=100); p.add_argument("--steps",type=int,default=200); args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); test_seeds=list(range(args.test_seed_start,args.test_seed_start+args.test_episodes)); frames=[]
    for seed in args.training_seeds:
        for mode in MODES: print("Evaluating seed",seed,mode); frames.append(evaluate(args.checkpoint_root,seed,mode,test_seeds,args.steps))
    data=pd.concat(frames,ignore_index=True); summary=summarize(data); data.to_csv(args.output_dir/"frozen_module_ablation_episodes.csv",index=False); summary.to_csv(args.output_dir/"frozen_module_ablation_summary.csv",index=False); paired(data).to_csv(args.output_dir/"frozen_module_ablation_paired.csv",index=False); plot(summary,args.output_dir/"frozen_module_ablation.png"); print("Frozen module ablation completed:",args.output_dir)


if __name__=="__main__": main()
