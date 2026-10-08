#!/usr/bin/env python3
"""
PhyG-MARL Lite Phase 1 验收检查脚本
运行方式: python check_phase1.py
"""

import os
import sys
import importlib
import traceback

def check_file_exists(module_path, description):
    """检查文件是否存在"""
    if os.path.exists(module_path):
        print(f"✅ [文件] {description} 存在")
        return True
    else:
        print(f"❌ [文件] {description} 缺失！路径: {module_path}")
        return False

def check_import(module_name, class_name):
    """检查是否能成功导入类"""
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        print(f"✅ [导入] {class_name} 导入成功 (来自 {module_name})")
        return True, module
    except Exception as e:
        print(f"❌ [导入] {class_name} 导入失败！错误: {e}")
        traceback.print_exc()
        return False, None

def main():
    print("=" * 60)
    print("🚀 开始验收 Phase 1 骨架搭建")
    print("=" * 60)

    # 1. 检查核心目录和文件是否存在
    files_to_check = [
        ("env/topology_manager.py", "拓扑管理器"),
        ("env/powerflow_solver.py", "潮流计算器"),
        ("env/attack_simulator.py", "攻击模拟器"),
        ("env/observation_builder.py", "观测构建器"),
        ("env/reward_calculator.py", "奖励计算器"),
        ("env/termination.py", "终止判定"),
        ("env/cpgrid_env.py", "主环境类")
    ]
    
    all_files_ok = True
    for path, desc in files_to_check:
        if not check_file_exists(path, desc):
            all_files_ok = False

    if not all_files_ok:
        print("\n❌ 关键文件缺失，请检查 Cursor 生成是否完整。")
        return

    # 2. 检查 requirements.txt
    if check_file_exists("requirements.txt", "依赖清单"):
        print("📦 请确保已执行: pip install -r requirements.txt (如未安装依赖)")
    
    # 3. 检查各模块能否独立导入 (无语法错误)
    print("\n" + "=" * 60)
    print("🔍 检查模块语法与基础导入")
    print("=" * 60)
    
    imports_to_check = [
        ("env.topology_manager", "TopologyManager"),
        ("env.powerflow_solver", "PowerFlowSolver"),
        ("env.attack_simulator", "AttackSimulator"),
        ("env.observation_builder", "ObservationBuilder"),
        ("env.reward_calculator", "RewardCalculator"),
        ("env.termination", "Termination"),
        ("env.cpgrid_env", "CPGridEnv")
    ]
    
    imported_modules = {}
    for mod_name, cls_name in imports_to_check:
        success, mod = check_import(mod_name, cls_name)
        if success:
            imported_modules[cls_name] = mod
        else:
            print(f"⚠️ 请检查 {mod_name} 中是否有未闭合的括号、缺失的 import 等基础语法错误。")
            return

    # 4. 关键测试：尝试实例化 Gym 环境 (检查接口规范)
    print("\n" + "=" * 60)
    print("🎮 测试 Gymnasium 环境接口规范 (reset & step 框架)")
    print("=" * 60)
    try:
        CPGridEnv = imported_modules["CPGridEnv"].CPGridEnv
        env = CPGridEnv()
        
        # 检查是否定义了 action_space 和 observation_space
        if hasattr(env, 'action_space') and hasattr(env, 'observation_space'):
            print(f"✅ action_space 已定义: {env.action_space}")
            print(f"✅ observation_space 已定义: {env.observation_space}")
        else:
            print("❌ 环境缺少 action_space 或 observation_space 定义！")
            return

        # 测试 reset
        print("\n▶ 执行 env.reset()...")
        obs, info = env.reset()
        print(f"✅ reset 返回 obs 类型: {type(obs)}, info 类型: {type(info)}")
        
        # 测试 step (传入一个 dummy action, 比如 0)
        print("\n▶ 执行 env.step(0) ...")
        obs, reward, terminated, truncated, info = env.step(0)
        print(f"✅ step 返回成功！")
        print(f"   - obs: {type(obs)}")
        print(f"   - reward: {reward}")
        print(f"   - terminated: {terminated}, truncated: {truncated}")
        
        print("\n🎉 Phase 1 骨架验收通过！环境接口符合 Gymnasium 规范，且无导入错误。")
        print("👉 可以放心进入 Phase 2：填充具体逻辑。")
        
    except Exception as e:
        print(f"\n❌ 环境实例化或交互测试失败: {e}")
        traceback.print_exc()
        print("\n💡 常见原因：")
        print("   1. cpgrid_env.py 中的 reset 或 step 没有返回正确的元组数量 (应为 5 个)。")
        print("   2. action_space 或 observation_space 未使用 gym.spaces 定义。")
        print("   3. 模块之间存在循环导入（例如 env 内互相 import）。")

if __name__ == "__main__":
    main()