"""
可视化模块 - 基于Pygame
"""
import pygame
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    from .environment import MilitaryEnvironment
except ImportError:
    from environment import MilitaryEnvironment


class MilitaryVisualizer:
    """军事仿真可视化器 - 基于Pygame"""

    # 颜色定义 (RGB格式)
    COLORS = {
        'background': (0, 0, 0),           # 黑色背景
        'obstacle': (105, 105, 105),     # 深灰色障碍物
        'grid_line': (30, 30, 30),       # 网格线（非常浅）
        'text': (255, 255, 255),         # 白色文字
        'ally_tank': (65, 105, 225),     # 皇家蓝坦克
        'ally_artillery': (220, 20, 60), # 深红色火炮
        'ally_drone': (255, 215, 0),     # 金黄色无人机
        'ally_vehicle': (50, 205, 50),   # 绿色战车
        'ally_scout': (255, 99, 71),     # 橙红色侦察车
        'enemy_tank': (139, 0, 0),       # 深红色坦克
        'enemy_artillery': (255, 69, 0), # 橙红色火炮
        'enemy_drone': (218, 165, 32),   # 浅金色无人机
        'enemy_vehicle': (34, 139, 34),  # 森林绿战车
        'enemy_scout': (205, 92, 92),    # 印度红侦察车
        'health_bar_bg': (255, 0, 0),    # 血量条背景（红色）
        'health_bar_fg': (0, 255, 0),    # 血量条前景（绿色）
        'ui_panel': (0, 0, 0, 180),      # UI面板（半透明黑色）
    }

    # 智能体形状大小
    AGENT_SIZE = 8
    CELL_SIZE = 4  # 每个格子在屏幕上占用的像素

    def __init__(self, env: MilitaryEnvironment, cell_size: int = 6, max_screen_size: Tuple[int, int] = (3840, 2160)):
        """
        初始化可视化器 - 完整显示地图

        Args:
            env: 仿真环境
            cell_size: 每个格子占用的像素大小 (默认6)
            max_screen_size: 最大屏幕尺寸 (width, height)
        """
        self.env = env
        self.cell_size = cell_size
        self.max_screen_size = max_screen_size

        # 计算完整的地图尺寸
        map_width, map_height = env.map.map_size

        # 计算实际窗口大小 (UI放在左边)
        self.ui_width = 250  # UI面板宽度
        self.map_width_pixels = map_width * self.cell_size
        self.map_height_pixels = map_height * self.cell_size
        self.screen_width = self.map_width_pixels + self.ui_width
        self.screen_height = max(self.map_height_pixels, 400)  # 最小高度

        # 如果超过最大屏幕尺寸，缩小cell_size
        if self.screen_width > max_screen_size[0] or self.screen_height > max_screen_size[1]:
            # 计算可用空间（减去UI宽度）
            available_width = max_screen_size[0] - self.ui_width
            available_height = max_screen_size[1]

            scale_x = available_width / self.map_width_pixels
            scale_y = available_height / self.map_height_pixels
            scale = min(scale_x, scale_y, 1.0)  # 不放大

            self.cell_size = max(1, int(self.cell_size * scale))
            self.map_width_pixels = map_width * self.cell_size
            self.map_height_pixels = map_height * self.cell_size
            self.screen_width = self.map_width_pixels + self.ui_width
            self.screen_height = max(self.map_height_pixels, available_height)

        # 显示完整的地图
        self.display_width = map_width
        self.display_height = map_height

        # 视图中心点（地图中心）
        self.view_center = (map_width // 2, map_height // 2)

        # Pygame初始化
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"军事对抗仿真 - {self.env.map.name}")

        # 字体 - 使用英文字体
        try:
            self.font = pygame.font.SysFont('arial', 16)
        except:
            self.font = pygame.font.Font(None, 16)

        # 小字体
        self.small_font = pygame.font.SysFont('arial', 12)

        # 时钟控制帧率
        self.clock = pygame.time.Clock()
        self.fps = 10  # 仿真帧率

        print(f"Pygame可视化器初始化完成 - 窗口大小: {self.screen_width}x{self.screen_height}")

    def render_frame(self, frame_data: Optional[Dict] = None) -> None:
        """
        渲染单帧

        Args:
            frame_data: 帧数据，如果为None则使用当前环境状态
        """
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 清空屏幕
        self.screen.fill(self.COLORS['background'])

        # 获取当前观察
        if frame_data is None:
            observation = self.env._get_global_observation()
        else:
            observation = frame_data

        # 显示完整地图
        x_start, y_start = 0, 0
        x_end, y_end = self.env.map.map_size

        # 绘制地图
        self._draw_map(x_start, y_start, x_end, y_end)

        # 绘制智能体
        self._draw_agents(observation['ally_agents'], 'ally', x_start, y_start)
        self._draw_agents(observation['enemy_agents'], 'enemy', x_start, y_start)

        # 绘制UI
        self._draw_ui(observation)

        # 更新显示
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_map(self, x_start: int, y_start: int, x_end: int, y_end: int):
        """绘制地图"""
        # 绘制障碍物
        for obs_x, obs_y in self.env.map.obstacle_positions:
            if x_start <= obs_x < x_end and y_start <= obs_y < y_end:
                screen_x = (obs_x - x_start) * self.cell_size
                screen_y = (obs_y - y_start) * self.cell_size

                pygame.draw.rect(self.screen, self.COLORS['obstacle'],
                               (screen_x, screen_y, self.cell_size, self.cell_size))

        # 绘制网格线（只在cell_size >= 4时绘制，避免太密集）
        if self.cell_size >= 4:
            for x in range(x_start, x_end + 1):
                screen_x = (x - x_start) * self.cell_size
                pygame.draw.line(self.screen, self.COLORS['grid_line'],
                               (screen_x, 0), (screen_x, (y_end - y_start) * self.cell_size), 1)

            for y in range(y_start, y_end + 1):
                screen_y = (y - y_start) * self.cell_size
                pygame.draw.line(self.screen, self.COLORS['grid_line'],
                               (0, screen_y), ((x_end - x_start) * self.cell_size, screen_y), 1)

    def _draw_agents(self, agents: List[Dict], team: str, x_start: int, y_start: int):
        """绘制智能体"""
        for agent_data in agents:
            if not agent_data['alive']:
                continue

            pos_x, pos_y = agent_data['position']
            agent_type = agent_data['type']

            # 获取智能体配置信息
            agent_config = self._get_agent_config(agent_type)
            if not agent_config:
                continue

            length = agent_config.get('length', 1)
            width = agent_config.get('width', 1)
            max_health = agent_config.get('health', 100)

            # 检查是否在显示区域内（需要考虑智能体实际尺寸）
            if not (x_start <= pos_x < x_start + self.display_width - length + 1 and
                    y_start <= pos_y < y_start + self.display_height - width + 1):
                continue

            # 获取颜色
            color_key = f"{team}_{agent_type}"
            color = self.COLORS.get(color_key, self.COLORS[f'{team}_tank'])

            # 计算屏幕坐标
            screen_x = (pos_x - x_start) * self.cell_size
            screen_y = (pos_y - y_start) * self.cell_size

            # 绘制智能体主体（使用实际尺寸）
            agent_width = length * self.cell_size
            agent_height = width * self.cell_size
            agent_rect = pygame.Rect(screen_x, screen_y, agent_width, agent_height)
            pygame.draw.rect(self.screen, color, agent_rect)

            # 绘制边框（更细更淡）
            pygame.draw.rect(self.screen, (200, 200, 200), agent_rect, 1)

            # 移除智能体内显示的类型标签（保持简洁）

            # 添加队伍标识（在智能体下方）
            team_label = team.upper()
            try:
                team_surface = self.small_font.render(team_label, True, (255, 255, 255))
                team_rect = team_surface.get_rect(center=(screen_x + agent_width//2,
                                                        screen_y + agent_height + 12))
                self.screen.blit(team_surface, team_rect)
            except:
                pass

            # 添加血量标注（在智能体右上角）
            current_health = agent_data['health']
            health_text = f"{current_health}/{max_health}"
            try:
                health_surface = self.small_font.render(health_text, True, (255, 255, 0))  # 黄色文字
                health_rect = health_surface.get_rect(topleft=(screen_x + agent_width + 2,
                                                              screen_y - 2))
                self.screen.blit(health_surface, health_rect)
            except:
                pass

    def _get_agent_config(self, agent_type: str) -> Dict:
        """获取智能体配置信息"""
        try:
            import json
            import os
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'agent_types', f'{agent_type}.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return {}

    def _draw_ui(self, observation: Dict):
        """绘制UI界面（右侧）"""
        ui_x = self.map_width_pixels + 10  # UI在地图右侧
        ui_y = 10

        # 计算需要的面板高度（根据文本行数）
        text_lines = 15  # 减少行数，UI更简洁
        panel_height = text_lines * 20 + 20  # 每行20像素，加20像素边距
        panel_width = self.ui_width - 20

        # 绘制UI背景面板
        panel_rect = pygame.Rect(ui_x, ui_y, panel_width, panel_height)
        panel_surface = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        panel_surface.fill(self.COLORS['ui_panel'])
        self.screen.blit(panel_surface, panel_rect)

        # 绘制边框
        pygame.draw.rect(self.screen, (255, 255, 255), panel_rect, 2)

        # 绘制文本信息 (英文)
        turn_text = f"Turn: {observation['turn']}"
        ally_text = f"Ally Alive: {sum(1 for agent in observation['ally_agents'] if agent['alive'])}"
        enemy_text = f"Enemy Alive: {sum(1 for agent in observation['enemy_agents'] if agent['alive'])}"

        status = "In Progress"
        if self.env.game_over:
            winner = self.env.winner if self.env.winner else "Draw"
            status = f"Game Over - Winner: {winner}"

        # 智能体类型和颜色对应说明（统一显示）
        agent_info = [
            ("Tank(TA)", 'ally_tank'),
            ("Artillery(AR)", 'ally_artillery'),
            ("Drone(DR)", 'ally_drone'),
            ("Vehicle(VE)", 'ally_vehicle'),
            ("Scout Car(SC)", 'ally_scout')
        ]

        # 渲染所有文本
        all_texts = [
            turn_text, ally_text, enemy_text, status,
            "", "Agent Types:", "",  # 空行
        ]

        # 添加智能体类型信息
        for agent_name, color_key in agent_info:
            if agent_name == "":
                all_texts.append("")
            elif color_key:
                all_texts.append(f"  {agent_name}")
            else:
                all_texts.append(agent_name)

        y_offset = 10
        for i, text in enumerate(all_texts):
            if text.strip() == "" and i > 4:  # 智能体类型之间的分隔
                continue

            text_surface = self.font.render(text, True, self.COLORS['text'])
            text_x = ui_x + 10

            # 为智能体类型添加颜色块
            # 智能体类型行 (索引7-11)
            if i >= 7 and i <= 11:
                agent_idx = i - 7
                if agent_idx < len(agent_info) and agent_info[agent_idx][1]:
                    color_key = agent_info[agent_idx][1]
                    color = self.COLORS.get(color_key, (255, 255, 255))
                    # 绘制颜色块
                    pygame.draw.rect(self.screen, color,
                                   (text_x, ui_y + y_offset - 2, 12, 12))
                    pygame.draw.rect(self.screen, (255, 255, 255),
                                   (text_x, ui_y + y_offset - 2, 12, 12), 1)
                    text_x += 18  # 颜色块宽度 + 间距

            self.screen.blit(text_surface, (text_x, ui_y + y_offset))
            y_offset += 18

    def show(self):
        """显示当前帧并处理事件"""
        self.render_frame()
        pygame.display.flip()

    def save_frame(self, filename: str):
        """保存当前帧"""
        pygame.image.save(self.screen, filename)

    def animate_simulation(self, max_turns: int = 100, delay: int = 100):
        """
        运行仿真动画

        Args:
            max_turns: 最大回合数
            delay: 每帧延迟（毫秒）
        """
        print("开始仿真动画，按ESC键退出...")

        running = True
        turn = 0

        while running and turn < max_turns and not self.env.game_over:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            # 执行仿真步骤
            self.env.step()
            turn += 1

            # 渲染
            self.render_frame()

            # 延迟
            pygame.time.delay(delay)

        print(f"动画结束 - 总回合数: {turn}")

    def close(self):
        """关闭可视化器"""
        pygame.quit()