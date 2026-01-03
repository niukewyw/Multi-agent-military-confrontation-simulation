"""
地图类定义
"""
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    from .agent import Agent
except ImportError:
    from agent import Agent


class Map:
    """地图类"""

    # 地图元素常量
    EMPTY = 0
    OBSTACLE = 1
    ALLY_AGENT = 2
    ENEMY_AGENT = 3

    def __init__(self, map_config_path: str):
        """
        初始化地图

        Args:
            map_config_path: 地图配置文件路径
        """
        with open(map_config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.name = self.config['name']
        self.map_size = tuple(self.config['map_size'])

        # 初始化地图网格
        self.grid = np.zeros(self.map_size, dtype=int)

        # 智能体列表
        self.ally_agents: List[Agent] = []
        self.enemy_agents: List[Agent] = []

        # 障碍物位置
        self.obstacle_positions: List[Tuple[int, int]] = []

        # 初始化地图
        self._initialize_map()

    def _initialize_map(self):
        """初始化地图内容"""
        # 生成障碍物
        self._generate_obstacles()

        # 创建智能体
        self._create_agents()

    def _generate_obstacles(self):
        """生成障碍物"""
        obstacles_config = self.config.get('obstacles', {})

        # 随机生成障碍物
        random_config = obstacles_config.get('random_generation', {})
        density = random_config.get('density', 0.1)
        cluster_prob = random_config.get('cluster_probability', 0.3)
        min_cluster = random_config.get('min_cluster_size', 3)
        max_cluster = random_config.get('max_cluster_size', 15)

        # 简单随机障碍物生成
        total_cells = self.map_size[0] * self.map_size[1]
        num_obstacles = int(total_cells * density)

        # 预定义区域障碍物
        predefined_zones = obstacles_config.get('predefined_zones', [])
        for zone in predefined_zones:
            center = zone['center']
            radius = zone['radius']
            zone_density = zone['density']

            # 在圆形区域内生成障碍物
            for x in range(max(0, center[0] - radius), min(self.map_size[0], center[0] + radius + 1)):
                for y in range(max(0, center[1] - radius), min(self.map_size[1], center[1] + radius + 1)):
                    if np.linalg.norm([x - center[0], y - center[1]]) <= radius:
                        if np.random.random() < zone_density:
                            self.grid[x, y] = self.OBSTACLE
                            self.obstacle_positions.append((x, y))

        # 随机障碍物（避免与预定义区域重叠）
        attempts = 0
        placed = 0
        max_attempts = num_obstacles * 10

        while placed < num_obstacles and attempts < max_attempts:
            x = np.random.randint(0, self.map_size[0])
            y = np.random.randint(0, self.map_size[1])

            if self.grid[x, y] == self.EMPTY:
                # 检查是否在预定义区域内（简单检查，避免过于密集）
                in_predefined = False
                for zone in predefined_zones:
                    center = zone['center']
                    radius = zone['radius']
                    if np.linalg.norm([x - center[0], y - center[1]]) <= radius:
                        in_predefined = True
                        break

                if not in_predefined:
                    self.grid[x, y] = self.OBSTACLE
                    self.obstacle_positions.append((x, y))
                    placed += 1

            attempts += 1

    def _create_agents(self):
        """创建智能体"""
        # 创建我方智能体
        for agent_config in self.config['ally_agents']:
            agent_type = agent_config['type']
            count = agent_config['count']
            position_range = agent_config['position_range']

            for _ in range(count):
                # 获取智能体尺寸以便正确检查位置
                agent_size = self._get_agent_size(agent_type)
                position = self._get_random_position(position_range, agent_size[0], agent_size[1])
                if position:
                    agent = Agent(agent_type, 'ally', position)
                    self.ally_agents.append(agent)
                    self._place_agent_on_grid(agent, self.ALLY_AGENT)

        # 创建敌方智能体
        for agent_config in self.config['enemy_agents']:
            agent_type = agent_config['type']
            count = agent_config['count']
            position_range = agent_config['position_range']

            for _ in range(count):
                # 获取智能体尺寸以便正确检查位置
                agent_size = self._get_agent_size(agent_type)
                position = self._get_random_position(position_range, agent_size[0], agent_size[1])
                if position:
                    agent = Agent(agent_type, 'enemy', position)
                    self.enemy_agents.append(agent)
                    self._place_agent_on_grid(agent, self.ENEMY_AGENT)

    def _get_random_position(self, position_range: List[List[int]], length: int = 1, width: int = 1) -> Optional[Tuple[int, int]]:
        """
        在指定范围内获取随机位置

        Args:
            position_range: [[x_min, x_max], [y_min, y_max]]
            length: 智能体长度
            width: 智能体宽度

        Returns:
            随机位置或None（如果找不到合适位置）
        """
        x_min, x_max = position_range[0]
        y_min, y_max = position_range[1]

        # 调整范围以确保智能体完全在范围内
        x_max = max(x_min, x_max - length + 1)
        y_max = max(y_min, y_max - width + 1)

        max_attempts = 100
        for _ in range(max_attempts):
            x = np.random.randint(x_min, x_max + 1)
            y = np.random.randint(y_min, y_max + 1)

            # 检查位置是否可用（不与障碍物或现有智能体重叠）
            if self._is_position_available(x, y, length, width):
                return (x, y)

        return None

    def _get_agent_size(self, agent_type: str) -> Tuple[int, int]:
        """
        获取智能体的尺寸

        Args:
            agent_type: 智能体类型

        Returns:
            (length, width) 元组
        """
        try:
            import json
            import os
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'agent_types', f'{agent_type}.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return (config.get('length', 1), config.get('width', 1))
        except:
            pass
        return (1, 1)  # 默认尺寸

    def _is_position_available(self, x: int, y: int, length: int, width: int) -> bool:
        """检查位置是否可用"""
        for dx in range(length):
            for dy in range(width):
                check_x, check_y = x + dx, y + dy
                if (check_x >= self.map_size[0] or check_y >= self.map_size[1] or
                    self.grid[check_x, check_y] != self.EMPTY):
                    return False
        return True

    def _place_agent_on_grid(self, agent: Agent, agent_type: int):
        """在网格上放置智能体"""
        for cell_x, cell_y in agent.occupied_cells:
            if 0 <= cell_x < self.map_size[0] and 0 <= cell_y < self.map_size[1]:
                self.grid[cell_x, cell_y] = agent_type

    def update_grid(self):
        """更新网格状态"""
        # 重置网格
        self.grid = np.zeros(self.map_size, dtype=int)

        # 重新放置障碍物
        for obs_x, obs_y in self.obstacle_positions:
            self.grid[obs_x, obs_y] = self.OBSTACLE

        # 重新放置智能体
        for agent in self.ally_agents:
            if agent.alive:
                self._place_agent_on_grid(agent, self.ALLY_AGENT)

        for agent in self.enemy_agents:
            if agent.alive:
                self._place_agent_on_grid(agent, self.ENEMY_AGENT)

    def get_all_agents(self) -> List[Agent]:
        """获取所有智能体"""
        return self.ally_agents + self.enemy_agents

    def get_enemy_positions(self, team: str) -> List[Tuple[int, int]]:
        """获取敌方智能体位置"""
        if team == 'ally':
            return [tuple(agent.position) for agent in self.enemy_agents if agent.alive]
        else:
            return [tuple(agent.position) for agent in self.ally_agents if agent.alive]

    def is_valid_position(self, position: Tuple[int, int], length: int = 1, width: int = 1) -> bool:
        """检查位置是否有效"""
        x, y = position
        return (0 <= x < self.map_size[0] - length + 1 and
                0 <= y < self.map_size[1] - width + 1)

    def can_move_to(self, agent: Agent, new_position: Tuple[int, int]) -> bool:
        """检查智能体是否可以移动到指定位置"""
        x, y = new_position

        # 检查边界
        if not self.is_valid_position((x, y), agent.length, agent.width):
            return False

        # 检查是否与障碍物碰撞
        for cell_x, cell_y in agent.occupied_cells:
            dx, dy = cell_x - agent.position[0], cell_y - agent.position[1]
            check_x, check_y = x + dx, y + dy

            if self.grid[check_x, check_y] == self.OBSTACLE:
                return False

        return True

    def __repr__(self):
        return f"Map({self.name}, size={self.map_size})"
