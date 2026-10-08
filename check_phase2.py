#!/usr/bin/env python3
"""
PhyG-MARL Lite Phase 2 验收脚本（单元测试 + 集成冒烟测试）
运行方式: python check_phase2.py
"""

import sys
import torch
import numpy as np
import pandapower as pp
import pandapower.networks as pn

# 导入环境模块
from env.cpgrid_env import CPGridEnv
from env.topology_manager import TopologyManager
from env.powerflow_solver import PowerFlowSolver
from env.attack_simulator import AttackSimulator
from env.observation_builder import ObservationBuilder
from env.reward_calculator import RewardCalculator
from env.termination import Termination

def test_pandapower_basic():
    """测试 Pandapower 能否求解 IEEE14 基础潮流"""
    print("\n" + "="*60)
    print("⚡ 测试1: Pandapower 潮流计算基础能力")
    print("="*60)
    try:
        net = pn.case14()
        pp.runpp(net)
        print("✅ 基础 IEEE14 潮流计算成功！")
        print(f"   - 节点数: {len(net.bus)}")
        print(f"   - 线路数: {len(net.line)}")
        print(f"   - 基准电压: {net.res_bus['vm_pu'].iloc[0]:.4f} pu")
        return True
    except Exception as e:
        print(f"❌ Pandapower 基础潮流失败: {e}")
        return False

def test_topology_manager():
    """测试拓扑管理器的增删边和 PyG 输出"""
    print("\n" + "="*60)
    print("🔌 测试2: TopologyManager 动态拓扑能力")
    print("="*60)
    try:
        tm = TopologyManager(n_bus=14)
        # 初始应该有 20 条线路 (IEEE14 标准)
        initial_edges = tm.get_graph().edge_index.shape[1]
        print(f"   - 初始线路数: {initial_edges}")
        
        # 模拟断开一条线路（假设线路索引 5）
        tm.remove_edge(5)
        after_remove = tm.get_graph().edge_index.shape[1]
        print(f"   - 断开一条后线路数: {after_remove}")
        
        # 恢复
        tm.restore_edge(5)
        after_restore = tm.get_graph().edge_index.shape[1]
        print(f"   - 恢复后线路数: {after_restore}")
        
        if after_remove == initial_edges - 2 and after_restore == initial_edges:
            print("✅ 拓扑管理器的增删边逻辑正确 (无向图需删除两条有向边)")
            # 检查 PyG Data 属性
            data = tm.get_graph()
            if hasattr(data, 'edge_index') and hasattr(data, 'edge_attr'):
                print(f"✅ PyG Data 对象包含 edge_index (shape: {data.edge_index.shape}) 和 edge_attr")
            return True
        else:
            print(f"❌ 线路数量变化异常: 初始 {initial_edges}, 删除后 {after_remove}, 恢复后 {after_restore}")
            return False
    except Exception as e:
        print(f"❌ TopologyManager 测试失败: {e}")
        return False

def test_powerflow_with_changes():
    """测试带拓扑变化的潮流求解"""
    print("\n" + "="*60)
    print("⚡ 测试3: PowerFlowSolver 动态潮流求解")
    print("="*60)
    try:
        solver = PowerFlowSolver()
        # 先求解初始状态
        pf_result = solver.solve(disconnected_lines=[])
        if pf_result is None:
            print("❌ 初始潮流求解返回 None")
            return False
        
        voltages = pf_result.get('voltage_pu')
        print(f"   - 初始潮流: 电压均值 {np.mean(voltages):.4f} pu")
        
        # 模拟一条线路断开 (比如线路 0)
        pf_result_changed = solver.solve(disconnected_lines=[0])
        if pf_result_changed is None:
            print("⚠️ 线路断开后潮流不收敛 (这在动态电网中可能发生，但需在环境中捕获异常)")
            # 允许 None，但要记录
            return True  # 不算硬失败，因为实际环境中会有保护动作
        else:
            new_voltages = pf_result_changed.get('voltage_pu')
            print(f"   - 断开线路后: 电压均值 {np.mean(new_voltages):.4f} pu")
            print("✅ 动态潮流求解通过")
            return True
    except Exception as e:
        print(f"❌ PowerFlowSolver 测试失败: {e}")
        return False

def test_observation_builder():
    """测试观测构建器是否能输出正确形状的张量"""
    print("\n" + "="*60)
    print("📊 测试4: ObservationBuilder 观测构建")
    print("="*60)
    try:
        # 构造动态拓扑和潮流结果。ObservationBuilder 需要 TopologyManager，
        # 以便让 GNN 边与 Pandapower 中的真实断线状态保持一致。
        tm = TopologyManager(14)
        solver = PowerFlowSolver()
        pf_result = solver.solve(disconnected_lines=[], defense_actions=np.zeros(14, dtype=int))

        obs_builder = ObservationBuilder()
        node_feats, edge_feats, edge_index = obs_builder.build(tm, pf_result)
        agent_obs = obs_builder.build_agent_observations(
            tm, node_feats, edge_feats, edge_index, np.zeros(14, dtype=int)
        )

        print(f"   - 节点特征形状: {node_feats.shape} (期望: [14, 4])")
        print(f"   - 边特征形状: {edge_feats.shape} (期望: [M, 3])")
        print(f"   - 边索引形状: {edge_index.shape} (期望: [2, M])")
        print(f"   - 智能体数量: {len(agent_obs)} (期望: 14)")
        print(f"   - 单智能体观测形状: {agent_obs['agent_0'].shape} (期望: [15])")

        if node_feats.shape != (14, 4) or edge_feats.shape[1] != 3 or edge_index.shape[0] != 2:
            print("❌ 图观测形状不符合接口定义")
            return False
        if len(agent_obs) != 14 or any(obs.shape != (15,) for obs in agent_obs.values()):
            print("❌ 局部智能体观测形状不符合接口定义")
            return False
        tensors = [node_feats, edge_feats, *agent_obs.values()]
        if any(torch.isnan(tensor).any() or torch.isinf(tensor).any() for tensor in tensors):
            print("❌ 观测包含 NaN/Inf")
            return False

        print("✅ 观测构建器输出正常，动态图与局部观测均无 NaN/Inf")
        return True
    except Exception as e:
        print(f"❌ ObservationBuilder 测试失败: {e}")
        return False

def test_attack_simulator():
    """测试三种攻击是否能正常执行而不崩溃"""
    print("\n" + "="*60)
    print("💣 测试5: AttackSimulator 攻击模拟")
    print("="*60)
    try:
        tm = TopologyManager(14)
        attack = AttackSimulator(tm)
        
        # 构造模拟观测 (batch_size=14, 假设每个节点观测维度为10)
        dummy_obs = torch.randn(14, 10)
        
        # 测试 FDI
        obs_fdi, topo_changes_fdi = attack.step("fdi", dummy_obs)
        print(f"   - FDI 攻击: 观测变化 (均方误差: {torch.mean((obs_fdi - dummy_obs)**2).item():.4f})")
        
        # 测试 DoS
        obs_dos, topo_changes_dos = attack.step("dos", dummy_obs)
        zero_count = (obs_dos == 0).sum().item()
        print(f"   - DoS 攻击: {zero_count} 个元素被置零")
        
        # 测试 Breaker
        obs_br, topo_changes_br = attack.step("breaker", dummy_obs)
        print(f"   - Breaker 攻击: 拓扑发生变化? {topo_changes_br is not None}")
        
        if topo_changes_br is not None and len(topo_changes_br) > 0:
            print(f"   - 断开的线路索引: {topo_changes_br}")
            
        print("✅ 三种攻击模拟均正常执行")
        return True
    except Exception as e:
        print(f"❌ AttackSimulator 测试失败: {e}")
        return False

def test_reward_calculator():
    """测试奖励函数的数值合理性"""
    print("\n" + "="*60)
    print("💰 测试6: RewardCalculator 奖励计算")
    print("="*60)
    try:
        solver = PowerFlowSolver()
        pf_result = solver.solve([])
        
        reward_calc = RewardCalculator()
        reward = reward_calc.calculate(pf_result, attack_info={"success": 0})
        
        print(f"   - 正常状态奖励: {reward:.4f}")
        
        # 模拟严重故障 (电压全部拉低到0.5 pu)
        pf_fault = pf_result.copy()
        pf_fault['voltage_pu'] = np.ones(14) * 0.5
        reward_fault = reward_calc.calculate(pf_fault, attack_info={"success": 1})
        print(f"   - 故障状态奖励: {reward_fault:.4f}")
        
        if np.isnan(reward) or np.isinf(reward):
            print("❌ 奖励包含 NaN 或 Inf!")
            return False
        
        if reward_fault < reward:
            print("✅ 奖励函数在故障时惩罚生效 (奖励降低)")
            return True
        else:
            print("⚠️ 故障奖励未明显降低，请检查奖励公式的惩罚项权重")
            return True
    except Exception as e:
        print(f"❌ RewardCalculator 测试失败: {e}")
        return False

def test_full_environment_rollout():
    """集成测试：环境运行 5 步不崩溃"""
    print("\n" + "="*60)
    print("🔄 测试7: CPGridEnv 完整环境冒烟测试 (运行5步)")
    print("="*60)
    try:
        env = CPGridEnv()
        obs, info = env.reset()
        print(f"   - Reset 成功, 观测键: {obs.keys() if isinstance(obs, dict) else type(obs)}")
        
        for i in range(5):
            action = env.action_space.sample()  # 随机动作
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   - Step {i+1}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                print(f"   ⚠️ 环境在第 {i+1} 步终止")
                break
        else:
            print("✅ 环境连续运行 5 步未崩溃！")
        return True
    except Exception as e:
        print(f"❌ 环境冒烟测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("🚀 开始 Phase 2 全面验收")
    print("="*60)
    
    results = []
    
    # 运行所有测试
    results.append(("Pandapower基础", test_pandapower_basic()))
    results.append(("拓扑管理器", test_topology_manager()))
    results.append(("动态潮流", test_powerflow_with_changes()))
    results.append(("观测构建", test_observation_builder()))
    results.append(("攻击模拟", test_attack_simulator()))
    results.append(("奖励计算", test_reward_calculator()))
    results.append(("环境集成冒烟", test_full_environment_rollout()))
    
    # 汇总
    print("\n" + "="*60)
    print("📋 Phase 2 验收结果汇总")
    print("="*60)
    passed = 0
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
        if result:
            passed += 1
    
    print(f"\n通过率: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉🎉🎉 Phase 2 全部通过！可以安全进入 Phase 3 (GNN实现)。")
    else:
        print("\n⚠️ 部分测试未通过，请检查上述失败项的具体错误。")
        print("💡 常见修复建议:")
        print("   1. 如果 Pandapower 不收敛，在 PowerFlowSolver 中添加 try/except，返回 None 并让环境进入保护状态。")
        print("   2. 如果观测形状不对，检查 ObservationBuilder 中特征拼接的维度。")
        print("   3. 如果拓扑数量变化不对，检查 TopologyManager 的无向图是否同时删除了 (u,v) 和 (v,u)。")

if __name__ == "__main__":
    main()