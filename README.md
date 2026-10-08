# PhyG-MARL-Lite

IEEE 14-bus 电力信息物理系统中的物理感知多智能体协同防御项目。

> 本 README 介绍当前代码、实验协议、运行命令和结果目录。论文写作依据、方法细节、实验结果解释及结论边界请参阅 `paper.md`。

## 1. 项目简介

本项目在 IEEE 14-bus 电网和 Pandapower AC 潮流环境中，研究 FDI、DoS 和暂态 Breaker 攻击下的多智能体主动防御。

14 个母线分别建模为 14 个防御智能体。每个智能体根据局部观测选择三分类防御动作；训练阶段使用集中式 Critic，执行阶段使用共享 Actor 分布式产生联合动作。

PhyG 的主要组成是：

- IEEE 14-bus Pandapower AC 潮流仿真；
- 14 个母线级防御智能体；
- FDI、DoS 和 Breaker 外生攻击；
- 动态线路拓扑和 Breaker 自动恢复；
- Physics-aware GraphEncoder；
- Transformer TemporalMemory；
- 参数共享 Actor；
- 集中式 Critic；
- MAPPO/PPO 训练；
- 冻结图—时序表示的 V2.2.2 主版本；
- 端到端联合训练的 V3.1 对照版本；
- PPO、IPPO、MAPPO、GRU-MAPPO、GAT-MAPPO、QMIX 和 MAT 独立 baseline。

## 2. 当前主实验协议

正式对比实验统一使用：

```text
电网：IEEE 14-bus
智能体数量：14
最大 episode 步数：200
训练 episodes：1000
训练 seeds：42、43、44
测试 seeds：60000～60099
攻击概率：0.35
攻击权重：FDI 0.45、DoS 0.45、Breaker 0.10
Breaker 持续时间：10～20 步
最大同时激活 Breaker：2
每回合最大 Breaker 攻击次数：6
```

所有 baseline 共用：

- `CPGridEnv`；
- 相同 Reward；
- 相同攻击配置；
- 相同动作语义；
- 相同训练 seed；
- 相同测试 seed；
- 相同离线评估指标。

PhyG 原训练代码不依赖 baseline 框架，baseline 通过独立目录和入口运行。

## 3. 环境和动作

### 3.1 局部观测

当前环境通过 `observation_space` 动态定义每个智能体的局部观测维度。当前正式环境中每个智能体为 21 维局部观测。

`config.py` 中的旧常量 `OBS_DIM=246` 不代表当前运行时的局部观测维度，论文和实验报告应以环境实际 `observation_space` 为准。

### 3.2 全局 Critic 状态

集中式 Critic 使用 72 维全局状态，主要包括：

- 14 个节点的物理特征；
- 受断线影响的节点 mask；
- 活跃 Breaker 数量；
- 当前时间进度。

IPPO 不使用该全局状态，其 Actor 和 Critic 只使用单个智能体的局部观测。

### 3.3 动作空间

联合动作空间为：

```text
MultiDiscrete([3, 3, ..., 3])
```

每个智能体输出动作 0、1、2。动作含义取决于母线角色：

- 负荷母线：对应负荷 scaling 1.00、0.95、0.90；
- 发电机或平衡节点：对应电压支撑增量 0、0.01、0.02 pu。

## 4. 攻击模型

### FDI

FDI 主要篡改目标智能体的局部观测，不直接改变真实潮流状态。

### DoS

DoS 主要造成目标智能体的观测冻结、丢失或通信阻断，不直接改变真实潮流状态。

### Breaker

Breaker 直接断开输电线路，改变真实拓扑、潮流分布和后续图数据。Breaker 持续时间结束后线路恢复。


## 5. Reward 和评价指标

Reward 由以下因素组成：

- 生存奖励；
- 负荷保持奖励；
- 电压偏差惩罚；
- 低电压和高电压惩罚；
- 线路过载惩罚；
- 负荷减载成本；
- 发电机电压支撑成本；
- 潮流失败和电压崩溃终止惩罚。

主要指标：

- `survival_rate`：运行到 200 步 time limit 的 episode 比例；
- `mean_reward`：每个测试 episode 的 Reward 总和平均值；
- `mean_load_ratio`：平均负荷保持率；
- `mean_high_voltage_rate`：高电压比例；
- `mean_low_voltage_rate`：低电压比例；
- `mean_steps`：平均存活步数；
- `powerflow_not_converged_rate`：潮流不收敛比例。

综合分数：

\[
BalancedScore=SurvivalRate+0.5\times LoadRatio-0.5\times HighVoltageRate.
\]

## 6. 目录说明

### PhyG 主代码

```text
env/                         电网环境、攻击、潮流和拓扑
├── cpgrid_env.py
├── attack_simulator.py
├── powerflow_solver.py
└── topology_manager.py

graph/                       Physics-aware 图编码器
memory/                      Transformer 时序记忆
policy/                      GraphActor 等策略网络
adapter/                     LightMAPPO 更新逻辑
train.py                     PhyG 单 seed 训练入口
offline_checkpoint_evaluation.py  PhyG checkpoint 评估
```

### 独立 baseline 代码

```text
baseline_config.py
run_baseline_suite.py
parallel_baseline_runner.py
evaluate_baseline_suite.py
plot_baseline_comparison.py
benchmark_model_cost.py

baselines/
├── common/
├── mappo/
├── ippo/
├── ppo/
├── gru_mappo/
├── gat_mappo/
├── qmix/
└── mat/
```

baseline 不修改 PhyG 的 `train.py`、环境、GNN、Transformer 或 V2.2.2 checkpoint。

## 7. 安装和环境检查

建议使用独立 Conda 环境：

```bash
python3 -m pip install -r requirements.txt
```

Windows 也可以使用：

```powershell
python -m pip install -r requirements.txt
```

检查核心依赖：

```bash
python3 -c "import torch, gymnasium, pandapower, numpy, pandas, matplotlib; print('dependencies ok')"
```

Windows：

```powershell
python -c "import torch, gymnasium, pandapower, numpy, pandas, matplotlib; print('dependencies ok')"
```

## 8. PhyG 训练

小规模 smoke test：

```bash
python3 train.py --seed 42 --episodes 10 --steps 200 --success_threshold 0.0 --output_dir results_smoke/seed_42
```

正式单 seed 训练：

```bash
python3 train.py --seed 42 --episodes 1000 --steps 200 --success_threshold 0.0 --output_dir results_phyg/seed_42
```

PhyG 多 seed 训练结果通常包含：

```text
experiment_config.json
training_metrics_advanced.csv
evaluation_metrics.csv
best_balanced.pt
best_survival.pt
best_reward.pt
final.pt
latest.pt
```

## 9. Baseline 正式训练

五种基础 baseline：

```bash
python3 run_baseline_suite.py --methods mappo ippo ppo gru_mappo gat_mappo --seeds 42 43 44 --episodes 1000 --steps 200 --output_dir baseline_comparison
```

加入 QMIX 和 MAT 后，运行全部七种 baseline：

```bash
python3 run_baseline_suite.py --methods mappo ippo ppo gru_mappo gat_mappo qmix mat --seeds 42 43 44 --episodes 1000 --steps 200 --output_dir baseline_comparison
```

建议正式实验前先运行：

```bash
python3 run_baseline_suite.py --methods qmix mat --seeds 42 --episodes 20 --steps 200 --output_dir qmix_mat_validation
```

每个方法和 seed 的结果目录：

```text
baseline_comparison/<method>/seed_<seed>/
├── training_metrics_advanced.csv
├── evaluation_metrics.csv
├── experiment_config.json
├── best_balanced.pt
├── best_reward.pt
├── best_survival.pt
├── final.pt
└── latest.pt
```

## 10. Baseline 并行训练

可以使用独立并行启动脚本，不修改训练代码：

```bash
python3 parallel_baseline_runner.py --methods mappo ippo ppo gru_mappo gat_mappo qmix mat --seeds 42 43 44 --episodes 1000 --steps 200 --output_dir baseline_comparison --max_workers 4 --threads_per_task 6
```

Windows：

```powershell
python parallel_baseline_runner.py --methods mappo ippo ppo gru_mappo gat_mappo qmix mat --seeds 42 43 44 --episodes 1000 --steps 200 --output_dir baseline_comparison --max_workers 4 --threads_per_task 6
```

每个“方法 + seed”是独立进程，结果写入独立目录。建议先使用 4 个并发任务确认内存和 CPU 占用，再根据机器资源提高并发数。

## 11. Baseline 统一离线评估

统一评估使用相同的 100 个测试 seed：

```bash
python3 evaluate_baseline_suite.py --root baseline_comparison --output_dir baseline_comparison/evaluation --methods mappo ippo ppo gru_mappo gat_mappo qmix mat --seeds 42 43 44
```

Windows：

```powershell
python evaluate_baseline_suite.py --root baseline_comparison --output_dir baseline_comparison/evaluation --methods mappo ippo ppo gru_mappo gat_mappo qmix mat --seeds 42 43 44
```

输出：

```text
baseline_comparison/evaluation/
├── baseline_checkpoint_summary.csv
├── method_summary.csv
├── significance_by_training_seed.csv
└── significance_summary.csv
```

每个方法和 seed 目录会生成：

```text
offline_episode_results.csv
```

它保存每个测试 seed 的 Reward、步数、生存结果、负荷率、电压率和终止原因。

## 12. PhyG V2.2.2 离线评估

PhyG checkpoint 使用专用脚本。建议按以下顺序执行：

```bash
python3 offline_checkpoint_evaluation.py --root results_actor_critic_v2_2_2_multiseed_parallel_1000eps --output_dir offline_evaluation_v2_2_2 --seeds 42 43 44 --steps 200 --mode smoke5
```

```bash
python3 offline_checkpoint_evaluation.py --root results_actor_critic_v2_2_2_multiseed_parallel_1000eps --output_dir offline_evaluation_v2_2_2 --seeds 42 43 44 --steps 200 --mode reconstruct20
```

```bash
python3 offline_checkpoint_evaluation.py --root results_actor_critic_v2_2_2_multiseed_parallel_1000eps --output_dir offline_evaluation_v2_2_2 --seeds 42 43 44 --steps 200 --mode full100
```

关键输出：

```text
offline_evaluation_v2_2_2/offline_episode_results.csv
offline_evaluation_v2_2_2/offline_checkpoint_summary.csv
offline_evaluation_v2_2_2/offline_model_type_summary.csv
```

## 13. 训练曲线

现在可以为七种 baseline 分别生成独立 2×2 图：

```bash
python3 plot_baseline_comparison.py --input_dir baseline_comparison --output_dir baseline_comparison/training_curves --methods mappo ippo ppo gru_mappo gat_mappo qmix mat --seeds 42 43 44
```

Windows：

```powershell
python plot_baseline_comparison.py --input_dir baseline_comparison --output_dir baseline_comparison/training_curves --methods mappo ippo ppo gru_mappo gat_mappo qmix mat --seeds 42 43 44
```

输出：

```text
baseline_comparison/training_curves/
├── mappo_training_curves.png
├── ippo_training_curves.png
├── ppo_training_curves.png
├── gru_mappo_training_curves.png
├── gat_mappo_training_curves.png
├── qmix_training_curves.png
└── mat_training_curves.png
```

每张图包含：

- Reward Moving Average；
- 50-Episode Survival Rate；
- 50-Episode Mean Load Ratio；
- 50-Episode Mean High-Voltage Rate。


## 14. 参数量和推理开销

不需要重新训练即可测试：

```bash
python3 benchmark_model_cost.py --methods mappo ippo ppo gru_mappo gat_mappo qmix mat phyg_v2_2_2 --seed 42 --warmup 200 --iterations 1000 --output_dir baseline_comparison/benchmark
```

Windows：

```powershell
python benchmark_model_cost.py --methods mappo ippo ppo gru_mappo gat_mappo qmix mat phyg_v2_2_2 --seed 42 --warmup 200 --iterations 1000 --output_dir baseline_comparison/benchmark
```

输出：

```text
baseline_comparison/benchmark/model_cost_summary.csv
```

统计内容：

- 总参数量；
- 可训练参数量；
- 冻结参数量；
- 模型大小；
- 平均推理延迟；
- P50/P95 延迟；
- 每秒决策次数。


## 15. 严格统计显著性

当以下文件完整存在时，可以进行严格配对分析：

```text
baseline_comparison/<method>/seed_<seed>/offline_episode_results.csv
offline_evaluation_v2_2_2/offline_episode_results.csv
```

配对键：

```text
training_seed + test_seed
```

指标和方法：

- Reward、步数、负荷率、高电压率：配对 Wilcoxon 符号秩检验；
- 生存结果：McNemar 或精确配对二项检验；
- 多重比较：Holm 校正；
- 同时报告均值差、95% 置信区间和实际胜/平/负数量。


## 16. 当前结果摘要

已有统一离线评估显示，PhyG V2.2.2 的 `best_balanced` checkpoint 约为：

```text
生存率：94.33% ± 2.52%
Reward：116.60 ± 3.90
负荷保持率：97.41% ± 2.22%
高电压率：65.13% ± 1.78%
```

七种 baseline 的统一离线结果表已保存到：

```text
baseline_comparison/evaluation/method_summary.csv
```

当前结果的合理解释是：

- PhyG 的生存率、Reward 和综合平衡表现最好；
- MAPPO 的负荷保持率和高电压率具有局部优势；
- PPO 是较强的综合 baseline；
- GRU-MAPPO 在部分训练 seed 上较稳定；
- QMIX 推理快但综合性能有限；
- GAT-MAPPO 没有稳定超过普通 MAPPO；
- MAT 当前参数量和推理延迟明显较高，且训练波动较大。


> 在当前 IEEE14 攻防环境、训练预算和统一测试协议下，PhyG 以可接受的推理成本取得了更好的安全—供电综合性能。



---

项目当前版本：2026/8/24
