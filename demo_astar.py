#!/usr/bin/env python3
"""
军事对抗仿真环境演示 - A*路径规划版本

在这个版本中，智能体使用A*算法进行路径规划，可以避开障碍物

使用示例:
    python demo_A*.py --map attack_focused --render=True --max_turns=50
    python demo_A*.py --map defense_focused --render=False --max_turns=100
    python demo_A*.py --map balanced --render=True --cell_size=8
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from environment import MilitaryEnvironment
from visualization import MilitaryVisualizer
from utils import find_nearest_enemy, a_star_pathfinding, euclidean_distance


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='军事对抗仿真环境演示 - A*版本')

    parser.add_argument('--map', type=str, required=True,
                       choices=['attack_focused', 'defense_focused', 'balanced'],
                       help='选择地图配置')

    parser.add_argument('--render', type=str, default='True',
                       choices=['True', 'False'],
                       help='是否启用可视化渲染')

    parser.add_argument('--max_turns', type=int, default=50,
                       help='最大仿真回合数')

    parser.add_argument('--cell_size', type=int, default=6,
                       help='每个格子在屏幕上占用的像素大小')

    parser.add_argument('--animation_delay', type=int, default=200,
                       help='动画每帧延迟（毫秒）')

    return parser.parse_args()


def generate_a_star_actions(team: str, env: MilitaryEnvironment) -> Dict[str, Dict]:
    """
    生成使用A*算法的智能体动作

    Args:
        team: 队伍 ('ally' 或 'enemy')
        env: 环境对象

    Returns:
        动作字典 {agent_id: action}
    """
    actions = {}
    agents = env.map.ally_agents if team == 'ally' else env.map.enemy_agents
    enemy_positions = env.map.get_enemy_positions(team)

    # 获取静态障碍物位置
    obstacles = set(env.map.obstacle_positions)

    for agent in agents:
        if not agent.alive:
            continue

        action = {}

        # 寻找最近的敌人
        nearest_enemy = find_nearest_enemy(tuple(agent.position), enemy_positions)

        if nearest_enemy:
            # 计算到敌人的距离
            distance = euclidean_distance(tuple(agent.position), nearest_enemy)

            # 如果距离很近（小于等于3格），直接移动即可，无需A*
            if distance <= 3:
                dx = nearest_enemy[0] - agent.position[0]
                dy = nearest_enemy[1] - agent.position[1]
                action['move'] = (1 if dx > 0 else -1 if dx < 0 else 0,
                                1 if dy > 0 else -1 if dy < 0 else 0)
            else:
                # 距离较远，使用A*算法规划路径
                start_pos = tuple(agent.position)
                goal_pos = nearest_enemy

                # 动态添加其他智能体的位置作为障碍物（只考虑大的智能体）
                dynamic_obstacles = set(obstacles)  # 复制静态障碍物

                # 只考虑距离较近的其他智能体作为动态障碍物
                other_agents = env.map.enemy_agents if team == 'ally' else env.map.ally_agents
                for other_agent in other_agents:
                    if (other_agent.alive and
                        euclidean_distance(tuple(agent.position), tuple(other_agent.position)) <= 5):
                        # 只将距离5格内的其他智能体作为障碍物
                        dynamic_obstacles.update(other_agent.occupied_cells)

                path = a_star_pathfinding(start_pos, goal_pos, list(dynamic_obstacles), env.map.map_size)

                if path and len(path) > 1:
                    # 路径存在，取下一个位置
                    next_pos = path[1]  # path[0]是当前位置

                    # 计算移动方向
                    dx = next_pos[0] - agent.position[0]
                    dy = next_pos[1] - agent.position[1]

                    # 确保移动距离不超过智能体的速度
                    if abs(dx) <= agent.speed and abs(dy) <= agent.speed:
                        action['move'] = (dx, dy)
                    else:
                        # 如果距离太远，只朝目标方向移动一步
                        action['move'] = (1 if dx > 0 else -1 if dx < 0 else 0,
                                        1 if dy > 0 else -1 if dy < 0 else 0)
                else:
                    # 找不到路径或路径太短，直接移动
                    dx = nearest_enemy[0] - agent.position[0]
                    dy = nearest_enemy[1] - agent.position[1]
                    action['move'] = (1 if dx > 0 else -1 if dx < 0 else 0,
                                    1 if dy > 0 else -1 if dy < 0 else 0)
        else:
            # 没有敌人，不动
            action['move'] = (0, 0)

        # 攻击逻辑：如果在攻击范围内，攻击最近敌人
        if agent.can_attack and agent.reload_counter == 0 and nearest_enemy:
            distance = euclidean_distance(tuple(agent.position), nearest_enemy)
            if distance <= agent.attack_range:
                action['attack'] = True
            else:
                action['attack'] = False
        else:
            action['attack'] = False

        # 特殊能力（暂时为空）
        action['special'] = False

        actions[agent.id] = action

    return actions


def main():
    """主函数"""
    args = parse_arguments()

    # 转换字符串参数为布尔值
    render = args.render == 'True'

    print("=== 军事对抗仿真环境演示 - A*路径规划版本 ===")
    print(f"地图: {args.map}")
    print(f"可视化: {'启用' if render else '禁用'}")
    print(f"最大回合数: {args.max_turns}")
    print("智能体控制: A*路径规划算法")
    print("-" * 50)

    # 检查地图配置文件是否存在
    map_config_path = f"maps/{args.map}.json"
    if not os.path.exists(map_config_path):
        print(f"错误: 地图配置文件 '{map_config_path}' 不存在!")
        return 1

    try:
        # 初始化环境
        print("正在初始化仿真环境...")
        env = MilitaryEnvironment(map_config_path)

        # 初始化可视化器
        visualizer = None
        if render:
            print("正在初始化可视化器...")
            visualizer = MilitaryVisualizer(env, cell_size=args.cell_size)

        # 重置环境
        print("正在重置环境...")
        observation = env.reset()

        # 运行仿真
        print("开始仿真 (使用A*路径规划)...")
        start_time = time.time()

        # 初始化路径规划缓存（减少A*调用频率）
        path_cache = {'ally': {}, 'enemy': {}}
        cache_turn = {'ally': 0, 'enemy': 0}

        if render and visualizer:
            # 带可视化的仿真
            frame_count = 0
            while not env.game_over and frame_count < args.max_turns:
                # 使用缓存的A*算法生成动作（每3回合重新规划一次）
                current_turn = frame_count // 3
                if cache_turn['ally'] != current_turn:
                    ally_actions = generate_a_star_actions('ally', env)
                    path_cache['ally'] = ally_actions
                    cache_turn['ally'] = current_turn
                else:
                    ally_actions = path_cache['ally']

                if cache_turn['enemy'] != current_turn:
                    enemy_actions = generate_a_star_actions('enemy', env)
                    path_cache['enemy'] = enemy_actions
                    cache_turn['enemy'] = current_turn
                else:
                    enemy_actions = path_cache['enemy']

                # 执行一步
                observations, rewards, done, info = env.step(ally_actions, enemy_actions)

                # 渲染当前状态
                visualizer.render_frame(observations)

                frame_count += 1

                # 打印状态信息
                if frame_count % 10 == 0:
                    print(f"回合 {frame_count}: 我方存活 {info['ally_alive']}, "
                          f"敌方存活 {info['enemy_alive']}")

                # 控制帧率（已经在visualizer内部控制，这里可以适当延时）
                if args.animation_delay > 50:  # 如果延时大于50ms才使用额外延时
                    time.sleep((args.animation_delay - 50) / 1000.0)

            # 显示最终结果
            if env.game_over:
                print(f"\n游戏结束! 胜利者: {env.winner}")
            else:
                print(f"\n达到最大回合数 ({args.max_turns})，仿真结束")

        else:
            # 纯文本仿真
            turn_count = 0
            while not env.game_over and turn_count < args.max_turns:
                # 使用缓存的A*算法生成动作（每3回合重新规划一次）
                current_turn = turn_count // 3
                if cache_turn['ally'] != current_turn:
                    ally_actions = generate_a_star_actions('ally', env)
                    path_cache['ally'] = ally_actions
                    cache_turn['ally'] = current_turn
                else:
                    ally_actions = path_cache['ally']

                if cache_turn['enemy'] != current_turn:
                    enemy_actions = generate_a_star_actions('enemy', env)
                    path_cache['enemy'] = enemy_actions
                    cache_turn['enemy'] = current_turn
                else:
                    enemy_actions = path_cache['enemy']

                observations, rewards, done, info = env.step(ally_actions, enemy_actions)

                turn_count += 1

                # 每10回合打印一次状态
                if turn_count % 10 == 0:
                    print(f"回合 {turn_count}: 我方存活 {info['ally_alive']}, "
                          f"敌方存活 {info['enemy_alive']}, "
                          f"我方奖励 {rewards['ally']:.2f}, "
                          f"敌方奖励 {rewards['enemy']:.2f}")

                    # 控制台渲染
                    env.render(mode='console')

            # 显示最终结果
            if env.game_over:
                print(f"\n游戏结束! 胜利者: {env.winner}")
            else:
                print(f"\n达到最大回合数 ({args.max_turns})，仿真结束")

        # 计算运行时间
        end_time = time.time()
        duration = end_time - start_time
        print(f"运行时间: {duration:.2f}秒")
        # 输出统计信息
        if env.reward_history['ally']:
            avg_ally_reward = sum(env.reward_history['ally']) / len(env.reward_history['ally'])
            avg_enemy_reward = sum(env.reward_history['enemy']) / len(env.reward_history['enemy'])
            print(f"平均我方奖励: {avg_ally_reward:.2f}")
            print(f"平均敌方奖励: {avg_enemy_reward:.2f}")
        # 清理资源
        if visualizer:
            visualizer.close()

        print("A*路径规划仿真完成!")
        return 0

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
