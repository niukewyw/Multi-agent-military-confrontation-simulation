"""
仿真环境主类
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
try:
    from .map import Map
    from .agent import Agent
    from .utils import find_nearest_enemy, a_star_pathfinding, get_direction_vector, euclidean_distance
except ImportError:
    from map import Map
    from agent import Agent
    from utils import find_nearest_enemy, a_star_pathfinding, get_direction_vector, euclidean_distance


class MilitaryEnvironment:
    """军事仿真环境"""

    def __init__(self, map_config_path: str):
        """
        初始化环境

        Args:
            map_config_path: 地图配置文件路径
        """
        self.map_config_path = map_config_path  # 保存配置文件路径
        self.map = Map(map_config_path)
        self.turn_counter = 0
        self.game_over = False
        self.winner = None

        # MARL接口
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()

        # 奖励历史
        self.reward_history = {'ally': [], 'enemy': []}

    def _define_observation_space(self) -> Dict:
        """定义观察空间"""
        return {
            'local_map_shape': (31, 31),  # 视野范围内的局部地图 (假设最大视野15)
            'num_features': 4,  # 地图元素类型 + 距离信息
            'agent_features': 5  # 血量比例、位置、是否可以攻击等
        }

    def _define_action_space(self) -> Dict:
        """定义动作空间"""
        return {
            'move': {'num_actions': 5},  # 不动、上、下、左、右
            'attack': {'num_actions': 2},  # 不攻击、攻击最近敌人
            'special': {'num_actions': 1}  # 特殊能力（针对自爆无人机等）
        }

    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        # 重新初始化地图
        self.map = Map(self.map_config_path)
        self.turn_counter = 0
        self.game_over = False
        self.winner = None
        self.reward_history = {'ally': [], 'enemy': []}

        return self._get_global_observation()

    def step(self, ally_actions: Optional[Dict] = None, enemy_actions: Optional[Dict] = None) -> Tuple[Dict, Dict, bool, Dict]:
        """
        执行一步仿真

        Args:
            ally_actions: 我方动作字典 {agent_id: action}
            enemy_actions: 敌方动作字典 {agent_id: action}

        Returns:
            observations, rewards, done, info
        """
        if self.game_over:
            return self._get_global_observation(), {'ally': 0, 'enemy': 0}, True, {'winner': self.winner}

        # 如果没有提供动作，使用规则AI
        if ally_actions is None:
            ally_actions = self._generate_rule_actions('ally')
        if enemy_actions is None:
            enemy_actions = self._generate_rule_actions('enemy')

        # 执行动作
        ally_rewards = self._execute_actions('ally', ally_actions)
        enemy_rewards = self._execute_actions('enemy', enemy_actions)

        # 更新智能体状态
        self._update_agents()

        # 检查游戏结束条件
        self._check_game_end()

        # 获取观察
        observations = self._get_global_observation()

        # 计算奖励
        rewards = {
            'ally': sum(ally_rewards.values()) / len(ally_rewards) if ally_rewards else 0,
            'enemy': sum(enemy_rewards.values()) / len(enemy_rewards) if enemy_rewards else 0
        }

        # 记录奖励历史
        self.reward_history['ally'].append(rewards['ally'])
        self.reward_history['enemy'].append(rewards['enemy'])

        self.turn_counter += 1

        info = {
            'turn': self.turn_counter,
            'winner': self.winner,
            'ally_alive': sum(1 for agent in self.map.ally_agents if agent.alive),
            'enemy_alive': sum(1 for agent in self.map.enemy_agents if agent.alive)
        }

        return observations, rewards, self.game_over, info

    def _generate_rule_actions(self, team: str) -> Dict[str, Dict]:
        """生成规则AI动作"""
        actions = {}
        agents = self.map.ally_agents if team == 'ally' else self.map.enemy_agents
        enemy_positions = self.map.get_enemy_positions(team)

        for agent in agents:
            if not agent.alive:
                continue

            action = {}

            # 移动逻辑：向最近敌人移动
            nearest_enemy = find_nearest_enemy(tuple(agent.position), enemy_positions)
            if nearest_enemy:
                direction = get_direction_vector(tuple(agent.position), nearest_enemy)
                action['move'] = direction
            else:
                action['move'] = (0, 0)  # 不动

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

    def _execute_actions(self, team: str, actions: Dict[str, Dict]) -> Dict[str, float]:
        """执行动作"""
        rewards = {}
        agents = self.map.ally_agents if team == 'ally' else self.map.enemy_agents

        for agent in agents:
            if not agent.alive:
                rewards[agent.id] = 0
                continue

            agent_actions = actions.get(agent.id, {'move': (0, 0), 'attack': False, 'special': False})
            reward = 0

            # 执行移动
            move_action = agent_actions.get('move', (0, 0))
            if move_action != (0, 0):
                # 计算新位置
                new_x = agent.position[0] + move_action[0] * agent.speed
                new_y = agent.position[1] + move_action[1] * agent.speed

                # 检查是否可以移动
                if self.map.can_move_to(agent, (new_x, new_y)):
                    agent.move(move_action, self.map.map_size)
                    reward += 0.1  # 移动奖励

            # 执行攻击
            if agent_actions.get('attack', False):
                enemy_positions = self.map.get_enemy_positions(team)
                nearest_enemy = find_nearest_enemy(tuple(agent.position), enemy_positions)

                if nearest_enemy:
                    damage = agent.attack(nearest_enemy)
                    if damage:
                        # 找到被攻击的智能体并造成伤害
                        target_agents = self.map.enemy_agents if team == 'ally' else self.map.ally_agents
                        for target in target_agents:
                            if tuple(target.position) == nearest_enemy:
                                if not target.take_damage(damage):
                                    reward += 10  # 消灭敌人奖励
                                else:
                                    reward += 1  # 造成伤害奖励
                                break

            rewards[agent.id] = reward

        return rewards

    def _update_agents(self):
        """更新所有智能体状态"""
        for agent in self.map.get_all_agents():
            agent.update()

        # 更新地图网格
        self.map.update_grid()

    def _check_game_end(self):
        """检查游戏结束条件"""
        ally_alive = any(agent.alive for agent in self.map.ally_agents)
        enemy_alive = any(agent.alive for agent in self.map.enemy_agents)

        if not ally_alive and not enemy_alive:
            self.game_over = True
            self.winner = 'draw'
        elif not ally_alive:
            self.game_over = True
            self.winner = 'enemy'
        elif not enemy_alive:
            self.game_over = True
            self.winner = 'ally'

    def _get_global_observation(self) -> Dict[str, Any]:
        """获取全局观察"""
        # 为简化，返回地图状态和智能体信息
        return {
            'map_grid': self.map.grid.copy(),
            'ally_agents': [
                {
                    'position': tuple(agent.position),
                    'health': agent.health,
                    'type': agent.agent_type,
                    'alive': agent.alive
                }
                for agent in self.map.ally_agents
            ],
            'enemy_agents': [
                {
                    'position': tuple(agent.position),
                    'health': agent.health,
                    'type': agent.agent_type,
                    'alive': agent.alive
                }
                for agent in self.map.enemy_agents
            ],
            'turn': self.turn_counter
        }

    def get_agent_observation(self, agent: Agent) -> Dict[str, Any]:
        """获取单个智能体的观察（用于MARL）"""
        enemy_positions = self.map.get_enemy_positions(agent.team)
        return agent.get_observation(self.map.grid, enemy_positions)

    def render(self, mode: str = 'console') -> Optional[Any]:
        """渲染环境"""
        if mode == 'console':
            self._render_console()
        elif mode == 'visual':
            # 这里将实现可视化渲染
            pass

    def _render_console(self):
        """控制台渲染"""
        print(f"\n=== 回合 {self.turn_counter} ===")
        print(f"我方存活: {sum(1 for agent in self.map.ally_agents if agent.alive)}")
        print(f"敌方存活: {sum(1 for agent in self.map.enemy_agents if agent.alive)}")

        if self.game_over:
            print(f"游戏结束! 胜利者: {self.winner}")

        # 简化的地图显示（只显示智能体位置）
        print("\n地图概览:")
        ally_positions = {(agent.position[0], agent.position[1]): agent for agent in self.map.ally_agents if agent.alive}
        enemy_positions = {(agent.position[0], agent.position[1]): agent for agent in self.map.enemy_agents if agent.alive}

        # 显示前10x10区域作为示例
        for y in range(min(10, self.map.map_size[1])):
            row = ""
            for x in range(min(10, self.map.map_size[0])):
                if (x, y) in ally_positions:
                    row += "A"
                elif (x, y) in enemy_positions:
                    row += "E"
                elif (x, y) in self.map.obstacle_positions:
                    row += "#"
                else:
                    row += "."
            print(row)
