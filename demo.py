#!/usr/bin/env python3
"""
军事对抗仿真环境演示脚本

使用示例:
    python demo.py --map attack_focused --render=True --max_turns=50
    python demo.py --map defense_focused --render=False --max_turns=100
    python demo.py --map balanced --render=True --save_animation=True
"""

import argparse
import sys
import os
import time
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from environment import MilitaryEnvironment
from visualization import MilitaryVisualizer


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='军事对抗仿真环境演示')

    parser.add_argument('--map', type=str, required=True,
                       choices=['attack_focused', 'defense_focused', 'balanced'],
                       help='选择地图配置')

    parser.add_argument('--render', type=str, default='True',
                       choices=['True', 'False'],
                       help='是否启用可视化渲染')

    parser.add_argument('--max_turns', type=int, default=50,
                       help='最大仿真回合数')

    parser.add_argument('--save_animation', type=str, default='False',
                       choices=['True', 'False'],
                       help='是否保存动画')

    parser.add_argument('--output_dir', type=str, default='output',
                       help='输出目录')

    parser.add_argument('--cell_size', type=int, default=6,
                       help='每个格子在屏幕上占用的像素大小（默认6）')

    parser.add_argument('--animation_delay', type=int, default=200,
                       help='动画每帧延迟（毫秒）')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    # 转换字符串参数为布尔值
    render = args.render == 'True'
    save_animation = args.save_animation == 'True'

    print("=== 军事对抗仿真环境演示 ===")
    print(f"地图: {args.map}")
    print(f"可视化: {'启用' if render else '禁用'}")
    print(f"最大回合数: {args.max_turns}")
    print(f"保存动画: {'是' if save_animation else '否'}")
    print("-" * 40)

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
        print("开始仿真...")
        start_time = time.time()

        if render and visualizer:
            # 带可视化的仿真
            frame_count = 0
            while not env.game_over and frame_count < args.max_turns:
                # 执行一步
                observations, rewards, done, info = env.step()

                # 渲染当前状态
                visualizer.render_frame(observations)
                visualizer.show()

                frame_count += 1

                # 打印状态信息
                if frame_count % 10 == 0:
                    print(f"回合 {frame_count}: 我方存活 {info['ally_alive']}, "
                          f"敌方存活 {info['enemy_alive']}")

                # 短暂暂停
                time.sleep(0.1)

            # 显示最终结果
            if env.game_over:
                print(f"\n游戏结束! 胜利者: {env.winner}")
            else:
                print(f"\n达到最大回合数 ({args.max_turns})，仿真结束")

            # 保存最终帧
            if save_animation:
                os.makedirs(args.output_dir, exist_ok=True)
                final_frame_path = os.path.join(args.output_dir, f"{args.map}_final.png")
                visualizer.save_frame(final_frame_path)
                print(f"最终帧已保存到: {final_frame_path}")

        else:
            # 纯文本仿真
            turn_count = 0
            while not env.game_over and turn_count < args.max_turns:
                observations, rewards, done, info = env.step()

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
        # 保存动画（如果启用）
        if save_animation and render and visualizer:
            print("正在运行动画...")
            os.makedirs(args.output_dir, exist_ok=True)
            # Pygame动画会实时显示，这里不需要额外保存
            visualizer.animate_simulation(max_turns=args.max_turns,
                                        delay=args.animation_delay)

        # 清理资源
        if visualizer:
            visualizer.close()

        print("仿真完成!")
        return 0

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
