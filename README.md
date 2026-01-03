# 军事对战栅格化仿真环境

这是一个基于Python的多智能体强化学习（MARL）军事对抗仿真环境，支持300x300格子的地图和多种智能体类型的对战模拟。

## 功能特性

- **大规模地图**: 300x300格子的仿真环境
- **多种智能体**: 坦克、火炮、自爆无人机、战车、侦察车、侦察型无人机
- **智能体配置**: 基于JSON的灵活配置系统
- **多种地图**: 攻击侧重、防御侧重、平衡地图
- **规则AI**: 内置简单AI行为（追击、攻击等）
- **MARL接口**: 为多智能体强化学习算法预留接口
- **实时可视化**: 基于Pygame的可视化渲染
- **命令行工具**: 便捷的demo脚本

## 项目结构

```
├── configs/agent_types/     # 智能体配置文件
│   ├── tank.json           # 坦克配置
│   ├── artillery.json      # 火炮配置
│   ├── suicide_drone.json  # 自爆无人机配置
│   ├── armored_vehicle.json # 战车配置
│   ├── scout_vehicle.json  # 侦察车配置
│   └── scout_drone.json    # 侦察无人机配置
├── maps/                   # 地图配置文件
│   ├── attack_focused.json # 攻击侧重地图
│   ├── defense_focused.json # 防御侧重地图
│   └── balanced.json       # 平衡地图
├── src/                    # 源代码
│   ├── __init__.py
│   ├── agent.py           # 智能体类
│   ├── map.py             # 地图类
│   ├── environment.py     # 仿真环境
│   ├── visualization.py   # 可视化模块
│   └── utils.py           # 工具函数
├── demo.py                # 主演示脚本
├── requirements.txt       # 依赖列表
└── README.md             # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 运行演示

```bash
# 激活环境
conda activate pymarl

# 攻击侧重地图，可视化模式（完整显示300x300地图）
python demo.py --map attack_focused --render=True --max_turns=50

# 高清可视化模式（每个格子15像素，适合大屏幕）
python demo.py --map attack_focused --render=True --cell_size=15 --max_turns=50

# 防御侧重地图，纯文本模式
python demo.py --map defense_focused --render=False --max_turns=100

# 平衡地图，可视化模式
python demo.py --map balanced --render=True --max_turns=30
```

### 命令行参数

- `--map`: 选择地图配置 (attack_focused/defense_focused/balanced)
- `--render`: 是否启用可视化 (True/False)
- `--max_turns`: 最大仿真回合数 (默认50)
- `--save_animation`: 是否保存动画 (True/False)
- `--output_dir`: 输出目录 (默认output)
- `--cell_size`: 每个格子在屏幕上占用的像素大小 (默认6，建议8-15以获得更好可视化效果)
- `--animation_delay`: 动画每帧延迟(毫秒) (默认200)

## 智能体类型

| 类型 | 特点 | 攻击范围 | 移动速度 | 特殊能力 |
|------|------|----------|----------|----------|
| 坦克 | 重装甲，高血量 | 8格 | 2格/回合 | 重装甲 |
| 火炮 | 远程打击，高伤害 | 25格 | 1格/回合 | 远程打击 |
| 自爆无人机 | 高速，自爆攻击 | 0格(接触) | 4格/回合 | 自爆 |
| 战车 | 中型装甲，均衡 | 6格 | 3格/回合 | 中型装甲 |
| 侦察车 | 高速，广视野 | 0格 | 4格/回合 | 侦察 |
| 侦察无人机 | 高视野，低血量 | 0格 | 5格/回合 | 远程侦察 |

## 地图配置

### 攻击侧重地图
- 大量坦克、火炮和自爆无人机
- 强调进攻能力和火力输出
- 适合测试攻击型策略

### 防御侧重地图
- 大量侦察单位和防护型智能体
- 强调情报收集和生存能力
- 适合测试防御和侦察策略

### 平衡地图
- 均衡的各类智能体配置
- 综合测试各种能力的平衡性

## 技术架构

### 核心组件

1. **Agent类**: 智能体对象，包含属性、行为和状态管理
2. **Map类**: 地图管理，包含障碍物生成和智能体放置
3. **MilitaryEnvironment类**: 仿真环境主类，实现回合制仿真
4. **MilitaryVisualizer类**: 可视化渲染，支持实时显示和动画

### MARL接口

环境预留了MARL算法接口：

- **观察空间**: 局部地图视图、智能体状态、敌方位置
- **动作空间**: 移动方向、攻击目标、特殊能力
- **奖励函数**: 基于伤害、生存、消灭敌人的奖励机制

### 扩展性

项目设计考虑了未来扩展：

- 易于添加新的智能体类型
- 支持自定义地图配置
- 为MAPPO、QMIX、VDN等算法预留集成接口
- 可扩展的奖励函数和终止条件

## 开发计划

- [x] 智能体配置系统
- [x] 地图生成和配置
- [x] 核心仿真引擎
- [x] 规则AI行为
- [x] 可视化模块
- [x] 命令行工具
- [ ] MARL算法集成 (MAPPO/QMIX/VDN)
- [ ] 性能优化
- [ ] 更多地图类型
- [ ] 高级AI策略

## 使用示例

### Python API

```python
from src.environment import MilitaryEnvironment
from src.visualization import MilitaryVisualizer

# 创建环境
env = MilitaryEnvironment("maps/attack_focused.json")

# 创建可视化器
visualizer = MilitaryVisualizer(env)

# 重置环境
observation = env.reset()

# 运行仿真
for _ in range(100):
    observations, rewards, done, info = env.step()
    visualizer.render_frame(observations)

    if done:
        break

# 清理资源
visualizer.close()
```

## 许可证

本项目仅用于学术和研究目的。
