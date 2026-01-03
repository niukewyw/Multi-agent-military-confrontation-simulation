"""
工具函数
"""
import numpy as np
from typing import List, Tuple, Optional
import heapq


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """计算曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """计算欧几里得距离"""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def find_nearest_enemy(agent_position: Tuple[int, int], enemy_positions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """找到最近的敌人"""
    if not enemy_positions:
        return None

    min_distance = float('inf')
    nearest_enemy = None

    for enemy_pos in enemy_positions:
        distance = euclidean_distance(agent_position, enemy_pos)
        if distance < min_distance:
            min_distance = distance
            nearest_enemy = enemy_pos

    return nearest_enemy


def a_star_pathfinding(start: Tuple[int, int], goal: Tuple[int, int],
                      obstacles: List[Tuple[int, int]], map_size: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    A*路径规划算法

    Args:
        start: 起始位置
        goal: 目标位置
        obstacles: 障碍物位置列表
        map_size: 地图尺寸

    Returns:
        路径列表（从起点到终点）或None（如果找不到路径）
    """
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 上下左右
            nx, ny = x + dx, y + dy
            if 0 <= nx < map_size[0] and 0 <= ny < map_size[1] and (nx, ny) not in obstacles:
                neighbors.append((nx, ny))
        return neighbors

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current = heapq.heappop(frontier)[1]

        if current == goal:
            break

        for neighbor in get_neighbors(current):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    # 重建路径
    if goal not in came_from:
        return None

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def get_direction_vector(current_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> Tuple[int, int]:
    """获取从当前位置到目标位置的方向向量（标准化）"""
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]

    # 标准化方向
    if dx != 0:
        dx = 1 if dx > 0 else -1
    if dy != 0:
        dy = 1 if dy > 0 else -1

    return (dx, dy)


def is_in_range(position: Tuple[int, int], center: Tuple[int, int], radius: float) -> bool:
    """检查位置是否在圆形范围内"""
    return euclidean_distance(position, center) <= radius


def calculate_damage_reduction(damage: int, armor: int) -> int:
    """计算护甲减伤后的实际伤害"""
    return max(1, damage - armor)


def normalize_position(position: Tuple[int, int], map_size: Tuple[int, int]) -> Tuple[int, int]:
    """确保位置在地图边界内"""
    x = max(0, min(position[0], map_size[0] - 1))
    y = max(0, min(position[1], map_size[1] - 1))
    return (x, y)
