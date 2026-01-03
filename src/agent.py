"""
智能体类定义
"""
import json
import os
import numpy as np
from typing import Tuple, List, Dict, Optional

# 工具函数导入
try:
    from .utils import manhattan_distance, euclidean_distance
except ImportError:
    from utils import manhattan_distance, euclidean_distance


class Agent:
    """智能体基类"""

    def __init__(self, agent_type: str, team: str, position: Tuple[int, int], config_path: str = "configs/agent_types/"):
        """
        初始化智能体

        Args:
            agent_type: 智能体类型 ('tank', 'suicide_drone', etc.)
            team: 队伍 ('ally' 或 'enemy')
            position: 初始位置 (x, y)
            config_path: 配置文件路径
        """
        self.agent_type = agent_type
        self.team = team
        self.position = np.array(position, dtype=int)
        self.id = f"{team}_{agent_type}_{id(self)}"

        # 加载配置文件
        config_file = os.path.join(config_path, f"{agent_type}.json")
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 初始化属性
        self.health = self.config['health']
        self.max_health = self.config['health']
        self.speed = self.config['speed']
        self.can_attack = self.config['can_attack']
        self.attack_range = self.config['attack_range']
        self.attack_damage = self.config['attack_damage']
        self.vision_range = self.config['vision_range']
        self.armor = self.config['armor']
        self.reload_time = self.config['reload_time']

        # 状态变量
        self.alive = True
        self.reload_counter = 0
        self.last_action = None

        # 尺寸
        self.length = self.config['length']
        self.width = self.config['width']

        # 计算占据的格子范围
        self.occupied_cells = self._calculate_occupied_cells()

    def _calculate_occupied_cells(self) -> List[Tuple[int, int]]:
        """计算智能体占据的格子"""
        cells = []
        x, y = self.position

        # 简单的矩形占据计算（可以根据需要优化为更复杂的形状）
        for dx in range(self.length):
            for dy in range(self.width):
                cells.append((x + dx, y + dy))

        return cells

    def move(self, direction: Tuple[int, int], map_size: Tuple[int, int]) -> bool:
        """
        移动智能体

        Args:
            direction: 移动方向 (dx, dy)
            map_size: 地图尺寸

        Returns:
            是否成功移动
        """
        if not self.alive:
            return False

        dx, dy = direction
        new_x = self.position[0] + dx * self.speed
        new_y = self.position[1] + dy * self.speed

        # 检查边界
        if (0 <= new_x < map_size[0] - self.length + 1 and
            0 <= new_y < map_size[1] - self.width + 1):
            self.position[0] = new_x
            self.position[1] = new_y
            self.occupied_cells = self._calculate_occupied_cells()
            return True

        return False

    def take_damage(self, damage: int) -> bool:
        """
        受到伤害

        Args:
            damage: 伤害值

        Returns:
            是否存活
        """
        actual_damage = max(1, damage - self.armor)
        self.health -= actual_damage

        if self.health <= 0:
            self.alive = False
            self.health = 0

        return self.alive

    def attack(self, target_position: Tuple[int, int]) -> Optional[int]:
        """
        攻击目标

        Args:
            target_position: 目标位置

        Returns:
            造成的伤害值，如果无法攻击则返回None
        """
        if not self.alive or not self.can_attack or self.reload_counter > 0:
            return None

        # 计算距离
        distance = np.linalg.norm(np.array(target_position) - self.position)

        if distance <= self.attack_range:
            self.reload_counter = self.reload_time
            return self.attack_damage

        return None

    def update(self):
        """更新智能体状态（每回合调用）"""
        if self.reload_counter > 0:
            self.reload_counter -= 1

    def get_observation(self, full_map: np.ndarray, enemy_positions: List[Tuple[int, int]]) -> Dict:
        """
        获取智能体的观察

        Args:
            full_map: 完整地图
            enemy_positions: 敌方智能体位置列表

        Returns:
            观察字典
        """
        vision_radius = self.vision_range
        x, y = self.position

        # 获取视野范围内的地图
        x_start = max(0, x - vision_radius)
        x_end = min(full_map.shape[0], x + vision_radius + 1)
        y_start = max(0, y - vision_radius)
        y_end = min(full_map.shape[1], y + vision_radius + 1)

        local_map = full_map[x_start:x_end, y_start:y_end]

        # 获取视野内的敌人
        visible_enemies = []
        for enemy_pos in enemy_positions:
            enemy_x, enemy_y = enemy_pos
            if (x_start <= enemy_x < x_end and y_start <= enemy_y < y_end):
                visible_enemies.append((enemy_x - x_start, enemy_y - y_start))

        return {
            'local_map': local_map,
            'visible_enemies': visible_enemies,
            'health_ratio': self.health / self.max_health,
            'can_attack': self.can_attack and self.reload_counter == 0,
            'position': (x, y)
        }

    def __repr__(self):
        return f"{self.team}_{self.agent_type}({self.position}, health={self.health})"
