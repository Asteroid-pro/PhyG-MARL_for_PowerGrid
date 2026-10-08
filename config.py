"""
项目配置模块：集中管理 PhyG 环境、图编码器、时序记忆、策略网络和训练参数。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

from pathlib import Path

import torch

# -------------------------
# Environment / data config
# -------------------------
NUM_AGENTS = 14  # env.cpgrid_env.CPGridEnv: 智能体数量，与 IEEE14 节点数一致
N_BUS = 14  # env.topology_manager.TopologyManager / env.cpgrid_env.CPGridEnv: 电网节点数
OBS_DIM = 246  # env.cpgrid_env.CPGridEnv: 每个智能体的观测向量维度
ACTION_DIM = 3  # env.cpgrid_env.CPGridEnv / policy.actor.GraphActor: 离散动作空间维度
NODE_IN_DIM = 4  # graph.graph_encoder.GraphEncoder: 节点输入特征维度 [V, angle, P_load, Q_load]
EDGE_IN_DIM = 3  # graph.graph_encoder.GraphEncoder: 边输入特征维度 [P_flow, Q_flow, status]

# -------------------------
# Graph / GNN config
# -------------------------
HIDDEN_DIM = 64  # graph.graph_encoder.GraphEncoder / memory.transformer_memory.TemporalMemory: 图嵌入维度
GRAPH_NUM_LAYERS = 2  # graph.graph_encoder.GraphEncoder: PhysicsGNNLayer 堆叠层数
PHYSICS_EDGE_DIM = 1  # graph.physics_gnn.PhysicsGNNLayer: 注意力 MLP 输入中的边特征维度
PHYSICS_LAMBDA_KCL = 0.1  # graph.physics_gnn.PhysicsGNNLayer: KCL 残差投影权重

# -------------------------
# Memory config
# -------------------------
MEMORY_HISTORY_LEN = 10  # memory.transformer_memory.TemporalMemory: 历史序列长度
HISTORY_LEN = MEMORY_HISTORY_LEN  # train.py / memory.transformer_memory.TemporalMemory: 历史长度别名
MEMORY_NUM_LAYERS = 2  # memory.transformer_memory.TemporalMemory: Transformer Encoder 层数
MEMORY_NHEAD = 4  # memory.transformer_memory.TemporalMemory: 多头注意力头数

# -------------------------
# Policy / actor config
# -------------------------
ACTOR_HIDDEN_DIM = 128  # policy.actor.GraphActor: 策略网络隐藏层宽度

# -------------------------
# Training config
# -------------------------
SEED = 42  # train.py / utils.seed.set_seed: 全局随机种子；None 表示保持随机训练
MAX_EPISODES = 50  # train.py: 默认训练 episode 数
STEPS_PER_EPISODE = 100  # train.py: 每个 episode 的最大步数
DEVICE = torch.device("cpu")  # train.py: 训练默认设备（当前强制 CPU）
OUTPUT_DIR = Path("results")  # train.py: 训练结果输出目录

# -------------------------
# MAPPO / PPO config
# -------------------------
#目前最好的一组是：
# MAPPO_LR = 1e-4  # adapter.light_mappo.LightMAPPO: actor/critic 学习率 1.修改前为3e-4 2.修改前为1e-4
# MAPPO_GAMMA = 0.99  # adapter.light_mappo.LightMAPPO: 折扣因子
# MAPPO_LAM = 0.95  # adapter.light_mappo.LightMAPPO: GAE λ
# MAPPO_CLIP_RATIO = 0.15  # adapter.light_mappo.LightMAPPO: PPO clip 比例 1.修改前为0.2 2.修改前为0.1，改为0.15后恢复为0.2
# MAPPO_ENTROPY_COEF = 0.01  # adapter.light_mappo.LightMAPPO: 熵正则系数
# MAPPO_EPOCHS = 10  # adapter.light_mappo.LightMAPPO: 每次更新的 epoch 数
# MAPPO_BATCH_SIZE = 64  # adapter.light_mappo.LightMAPPO: 更新时 mini-batch 大小

MAPPO_LR = 1e-4  # adapter.light_mappo.LightMAPPO: actor/critic 基础学习率
MAPPO_GAMMA = 0.99  # adapter.light_mappo.LightMAPPO: 折扣因子
MAPPO_LAM = 0.95  # adapter.light_mappo.LightMAPPO: GAE λ
MAPPO_CLIP_RATIO = 0.15  # adapter.light_mappo.LightMAPPO: PPO clip 比例
MAPPO_ENTROPY_COEF = 0.01  # adapter.light_mappo.LightMAPPO: 熵正则系数
MAPPO_EPOCHS = 10  # adapter.light_mappo.LightMAPPO: 每次更新的 epoch 数
MAPPO_BATCH_SIZE = 128  # adapter.light_mappo.LightMAPPO: 更新时 mini-batch 大小
MAPPO_USE_LR_SCHEDULE = True  # adapter.light_mappo.LightMAPPO: 是否启用线性学习率衰减
MAPPO_LR_TOTAL_UPDATES = 3000  # adapter.light_mappo.LightMAPPO: 线性衰减总 update 次数估计
MAPPO_ACTOR_LR_END = 1e-5  # adapter.light_mappo.LightMAPPO: actor 最终学习率
MAPPO_CRITIC_LR_START = 1e-4  # adapter.light_mappo.LightMAPPO: critic 初始学习率
MAPPO_CRITIC_LR_END = 2e-5  # adapter.light_mappo.LightMAPPO: critic 最终学习率

# -------------------------
# Reward config
# -------------------------
REWARD_VOLTAGE_WEIGHT = 0.3  # env.reward_calculator.RewardCalculator: 电压稳定项权重
REWARD_LOAD_WEIGHT = 0.7  # env.reward_calculator.RewardCalculator: 负荷保持项权重
REWARD_ATTACK_PENALTY = 0.5  # env.reward_calculator.RewardCalculator: 攻击惩罚项权重
REWARD_ACTION_COST = 0.02  # env.reward_calculator.RewardCalculator: 非零动作成本系数
REWARD_CRASH_PENALTY = -1.0  # env.reward_calculator.RewardCalculator: 潮流不收敛/系统崩溃时的惩罚
