"""
@File    : enhanced_config.py
@Author  : Bin MEI
@Date    : 2025-10-18
@Desc    : 2D path planning algorithm


支持：
1. 公里级冰区尺度
2. 多种预设船型
3. 真实船舶动力学参数
4. 多种冰况场景
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import numpy as np


class SimulationScale(Enum):
    """仿真尺度"""
    SMALL = "小尺度 (100m×150m)"      # 原始尺度，用于快速测试
    MEDIUM = "中尺度 (500m×800m)"     # 中等尺度
    LARGE = "大尺度 (2km×3km)"        # 公里级，接近真实冰区
    REALISTIC = "真实尺度 (5km×8km)"  # 完整冰区
    SAM_REAL = "SAM真实冰场 (7.3km×3.5km)"  # SAM分割结果原始尺度


class IceCondition(Enum):
    """冰况等级 - 基于WMO海冰分类"""
    OPEN_WATER = "开阔水域 (0-10%)"          # 几乎无冰
    VERY_OPEN = "很稀疏冰 (10-30%)"          # 偶有浮冰
    OPEN_DRIFT = "稀疏浮冰 (30-50%)"         # 可航行
    CLOSE_PACK = "密集浮冰 (50-70%)"         # 需要破冰能力
    VERY_CLOSE = "很密集冰 (70-90%)"         # 严重阻碍
    COMPACT = "密实冰 (90-100%)"             # 极难通行
    CONSOLIDATED = "固定冰"                  # 几乎不可通行


@dataclass
class ScalePreset:
    """尺度预设参数"""
    world_width: float           # 世界宽度 (m)
    world_height: float          # 世界高度 (m)
    window_width: int            # 窗口宽度 (像素)
    window_height: int           # 窗口高度 (像素)
    physics_substeps: int        # 物理子步数
    path_layers: int             # 路径规划层数
    ice_min_size: float          # 最小冰块尺寸 (m)
    max_ice_count: int           # 最大冰块数量


@dataclass  
class IcePreset:
    """冰况预设参数"""
    coverage: float              # 覆盖率 (0-1)
    thickness_mean: float        # 平均冰厚 (m)
    thickness_std: float         # 冰厚标准差
    concentration: float         # 密集度
    size_scale: float            # 冰块尺寸缩放
    distribution: Dict[str, float]  # 类型分布


# ========== 预设库 ==========

# 窗口尺寸已优化，适配常见屏幕分辨率
SCALE_PRESETS: Dict[SimulationScale, ScalePreset] = {
    SimulationScale.SMALL: ScalePreset(
        world_width=100.0, world_height=150.0,
        window_width=700, window_height=750,
        physics_substeps=4, path_layers=20,      # 增加子步数防止穿透
        ice_min_size=0.5, max_ice_count=200
    ),
    SimulationScale.MEDIUM: ScalePreset(
        world_width=500.0, world_height=800.0,
        window_width=750, window_height=800,
        physics_substeps=6, path_layers=40,      # 增加子步数防止穿透
        ice_min_size=1.0, max_ice_count=500
    ),
    SimulationScale.LARGE: ScalePreset(
        world_width=2000.0, world_height=3000.0,
        window_width=800, window_height=850,
        physics_substeps=8, path_layers=60,      # 增加子步数防止穿透
        ice_min_size=2.0, max_ice_count=1000
    ),
    SimulationScale.REALISTIC: ScalePreset(
        world_width=5000.0, world_height=8000.0,
        window_width=850, window_height=900,
        physics_substeps=10, path_layers=100,    # 增加子步数防止穿透
        ice_min_size=3.0, max_ice_count=2000
    ),
    # SAM真实冰场尺度 (2025-12-17新增)
    # 横向布局: 7.3km(宽) × 3.5km(高)，船从左→右航行
    SimulationScale.SAM_REAL: ScalePreset(
        world_width=7300.0, world_height=3500.0,
        window_width=1200, window_height=600,
        physics_substeps=10, path_layers=80,     # 增加子步数防止穿透
        ice_min_size=5.0, max_ice_count=3000
    ),
}

ICE_PRESETS: Dict[IceCondition, IcePreset] = {
    IceCondition.OPEN_WATER: IcePreset(
        coverage=0.05, thickness_mean=0.2, thickness_std=0.1,
        concentration=0.05, size_scale=0.5,
        distribution={'Fragment': 0.8, 'Small Floe': 0.2}
    ),
    IceCondition.VERY_OPEN: IcePreset(
        coverage=0.20, thickness_mean=0.3, thickness_std=0.15,
        concentration=0.2, size_scale=0.8,
        distribution={'Fragment': 0.5, 'Small Floe': 0.35, 'Medium Floe': 0.15}
    ),
    IceCondition.OPEN_DRIFT: IcePreset(
        coverage=0.40, thickness_mean=0.5, thickness_std=0.2,
        concentration=0.4, size_scale=1.0,
        distribution={'Fragment': 0.3, 'Small Floe': 0.35, 'Medium Floe': 0.25, 'Large Floe': 0.1}
    ),
    IceCondition.CLOSE_PACK: IcePreset(
        coverage=0.60, thickness_mean=0.8, thickness_std=0.3,
        concentration=0.6, size_scale=1.5,  # 增大尺寸
        distribution={'Fragment': 0.15, 'Small Floe': 0.25, 'Medium Floe': 0.25, 'Large Floe': 0.20, 'Ice Bank': 0.15}  # 更多大冰块
    ),
    IceCondition.VERY_CLOSE: IcePreset(
        coverage=0.80, thickness_mean=1.2, thickness_std=0.4,
        concentration=0.8, size_scale=2.0,  # 增大尺寸
        distribution={'Fragment': 0.10, 'Small Floe': 0.20, 'Medium Floe': 0.25, 'Large Floe': 0.25, 'Ice Bank': 0.20}  # 更多大冰块
    ),
    IceCondition.COMPACT: IcePreset(
        coverage=0.95, thickness_mean=1.8, thickness_std=0.5,
        concentration=0.95, size_scale=2.5,  # 增大尺寸
        distribution={'Fragment': 0.05, 'Small Floe': 0.15, 'Medium Floe': 0.25, 'Large Floe': 0.30, 'Ice Bank': 0.25}  # 更多大冰块
    ),
}


class EnhancedConfig:
    """
    增强版配置类
    
    支持场景预设、船型选择、动态调参
    """
    
    def __init__(self, 
                 scale: SimulationScale = SimulationScale.MEDIUM,
                 ice_condition: IceCondition = IceCondition.CLOSE_PACK,
                 ship_type: str = "RESEARCH_VESSEL"):
        """
        初始化配置
        
        Args:
            scale: 仿真尺度
            ice_condition: 冰况等级
            ship_type: 船型名称
        """
        self._scale = scale
        self._ice_condition = ice_condition
        self._ship_type = ship_type
        
        self._load_presets()
    
    def _load_presets(self):
        """加载预设参数"""
        scale_preset = SCALE_PRESETS[self._scale]
        ice_preset = ICE_PRESETS[self._ice_condition]
        
        # ==================== 显示参数 ====================
        self.WINDOW_WIDTH = scale_preset.window_width
        self.WINDOW_HEIGHT = scale_preset.window_height
        self.FPS = 60
        
        self.GAME_AREA_WIDTH = self.WINDOW_WIDTH - 200
        self.CONTROL_AREA_WIDTH = 200
        
        # ==================== 仿真区域 ====================
        self.WORLD_WIDTH = scale_preset.world_width
        self.WORLD_HEIGHT = scale_preset.world_height
        
        # ==================== 船舶动力学 ====================
        # 使用ship_dynamics模块的真实船舶模型
        self.USE_REALISTIC_DYNAMICS = True
        self.SHIP_TYPE = self._ship_type
        
        # 向后兼容的简化参数
        ship_lengths = {
            "ICEBREAKER_SMALL": 50.0,
            "ICEBREAKER_MEDIUM": 122.5,
            "ICEBREAKER_LARGE": 173.0,
            "CARGO_SHIP": 150.0,
            "TANKER": 250.0,
            "RESEARCH_VESSEL": 80.0,
        }
        ship_widths = {
            "ICEBREAKER_SMALL": 12.0,
            "ICEBREAKER_MEDIUM": 22.3,
            "ICEBREAKER_LARGE": 34.0,
            "CARGO_SHIP": 25.0,
            "TANKER": 44.0,
            "RESEARCH_VESSEL": 16.0,
        }
        
        self.SHIP_LENGTH = ship_lengths.get(self._ship_type, 80.0)
        self.SHIP_WIDTH = ship_widths.get(self._ship_type, 16.0)
        self.SHIP_MASS = self._estimate_mass()
        self.MIN_TURNING_RADIUS = self.SHIP_LENGTH * 3.5
        
        # 动力学参数
        self.SHIP_MAX_FORCE = self.SHIP_MASS * 8.0  # 推力/质量比
        self.SHIP_MAX_SPEED = 15.0 * 0.5144  # 15节
        self.SHIP_FRICTION = 0.5
        self.SHIP_ELASTICITY = 0.05
        
        # 船只初始位置（归一化）
        # SAM_REAL尺度使用横向航行（左→右）
        if self._scale == SimulationScale.SAM_REAL:
            self.SHIP_START_X = 0.05   # 左侧出发
            self.SHIP_START_Y = 0.5    # 中间位置
        else:
            self.SHIP_START_X = 0.5    # 中心
            self.SHIP_START_Y = 0.1    # 底部

        # 船首初始朝向（弧度）
        # 约定：heading=0朝右，heading=90°(pi/2)朝上
        if self._scale == SimulationScale.SAM_REAL:
            self.SHIP_START_HEADING = 0.0
        else:
            self.SHIP_START_HEADING = np.pi / 2
        
        # ==================== 冰块参数 ====================
        self.ICE_COVERAGE = ice_preset.coverage
        self.ICE_THICKNESS = ice_preset.thickness_mean
        self.ICE_CONCENTRATION = ice_preset.concentration
        
        # 冰块类型分布
        self.ICE_DISTRIBUTION = ice_preset.distribution
        
        # 冰块尺寸范围（根据尺度调整）
        size_scale = ice_preset.size_scale
        
        # 根据世界尺度额外放大冰块
        world_scale_factor = max(1.0, (self.WORLD_WIDTH / 100.0) ** 0.3)  # 尺度越大，冰块越大
        
        base_sizes = {
            'Ice Bank': (20, 40),       # 增大：20-40m -> 放大后可达60-100m
            'Large Floe': (12, 25),     # 增大：12-25m -> 放大后可达30-60m
            'Medium Floe': (5, 12),     # 5-12m
            'Small Floe': (2, 5),       # 2-5m
            'Fragment': (0.5, 2)        # 0.5-2m
        }
        self.ICE_SIZE_RANGES = {
            k: (v[0] * size_scale * world_scale_factor, v[1] * size_scale * world_scale_factor) 
            for k, v in base_sizes.items()
        }
        
        print(f"   Ice Bank size: {self.ICE_SIZE_RANGES['Ice Bank'][0]:.0f}-{self.ICE_SIZE_RANGES['Ice Bank'][1]:.0f}m")
        
        # 冰块密度
        self.ICE_DENSITY_PER_AREA = {
            'Ice Bank': 800.0,
            'Large Floe': 500.0,
            'Medium Floe': 350.0,
            'Small Floe': 200.0,
            'Fragment': 120.0
        }
        
        # 冰块摩擦与弹性 (增加摩擦使冰块更难被推动)
        self.ICE_FRICTION = 0.6           # 增加摩擦力（原0.3）
        self.ICE_ELASTICITY = 0.05        # 降低弹性（原0.1）减少弹跳
        self.SHIP_ICE_FRICTION = 0.5      # 船-冰摩擦
        self.SHIP_ICE_ELASTICITY = 0.02   # 船-冰弹性（极低，避免反弹）
        
        # ==================== 路径参数 ====================
        # SAM_REAL尺度使用横向航行（左→右）
        if self._scale == SimulationScale.SAM_REAL:
            self.GOAL_X = 0.95   # 右侧目标
            self.GOAL_Y = 0.5    # 中间位置
        else:
            self.GOAL_X = 0.5    # 中心
            self.GOAL_Y = 0.9    # 顶部
        self.PATH_TYPE = "A*"

        self.ICE_THETA_ABLATION = "A0"#消融设计
        self.ICE_THETA_USE_SOFT_COST = True
        self.ICE_THETA_ENABLE_ANY_ANGLE = True
        self.ICE_THETA_LOS_STEP_CELLS = 0.5
        self.ICE_THETA_COST_SAMPLE_STEP_CELLS = 0.5
        
        self.ENABLE_MID_ROUTE_ICE_BARRIER = False
        self.MID_ROUTE_BARRIER_COLLISION_FAIL_COUNT = 8
        self.MID_ROUTE_BARRIER_WIDTH_RATIO = 0.65
        self.MID_ROUTE_BARRIER_THICKNESS_M = self.SHIP_LENGTH * 4.0

        self.RANDOM_FIELD_INJECT_HUGE_ICE = True
        self.RANDOM_FIELD_HUGE_ICE_COUNT = 2
        self.RANDOM_FIELD_HUGE_ICE_MIN_SIZE_M = self.SHIP_LENGTH * 1.6
        self.RANDOM_FIELD_HUGE_ICE_MAX_SIZE_M = self.SHIP_LENGTH * 3.0
        self.RANDOM_FIELD_HUGE_ICE_TYPE = 'Large Floe'
        #减速逻辑
        self.SHIP_ICE_COLLISION_COOLDOWN_S = 0.2
        self.SHIP_ICE_COLLISION_SPEED_REDUCTION_SCALE = 0.35#原来0.6
        self.SHIP_ICE_COLLISION_SPEED_REDUCTION_MAX = 0.35#原来0.55
        self.SHIP_ICE_COLLISION_DIRECTIONAL = True
        self.SHIP_ICE_COLLISION_SIDE_FACTOR = 0.25#原来0.35
        self.SHIP_ICE_COLLISION_ALIGN_POWER = 2.0
        self.SHIP_ICE_COLLISION_RELVEL_REF_MPS = 3.5#原2.5

        self.RANDOM_FIELD_ENABLE_CURVED_ICE_BANK = False
        self.RANDOM_FIELD_CURVED_ICE_BANK_BLOCKS = 8
        self.RANDOM_FIELD_CURVED_ICE_BANK_MIN_SIZE_M = self.SHIP_LENGTH * 2.0
        self.RANDOM_FIELD_CURVED_ICE_BANK_MAX_SIZE_M = self.SHIP_LENGTH * 3.5
        self.RANDOM_FIELD_CURVED_ICE_BANK_TYPE = 'Ice Bank'
        self.RANDOM_FIELD_CURVED_ICE_BANK_AMPLITUDE_M = self.SHIP_LENGTH * 5.0
        self.RANDOM_FIELD_CURVED_ICE_BANK_WAVELENGTH_M = self.SHIP_LENGTH * 18.0
        self.RANDOM_FIELD_CURVED_ICE_BANK_WIDTH_RATIO = 0.65
        self.RANDOM_FIELD_CURVED_ICE_BANK_GAP_RATIO = 0.35

        self.RANDOM_FIELD_ENABLE_MID_BAND_OBSTACLES = True
        self.RANDOM_FIELD_MID_BAND_T_MIN = 0.35
        self.RANDOM_FIELD_MID_BAND_T_MAX = 0.70
        self.RANDOM_FIELD_MID_BAND_COUNT = 14
        self.RANDOM_FIELD_MID_BAND_MIN_SIZE_M = self.SHIP_LENGTH * 1.2
        self.RANDOM_FIELD_MID_BAND_MAX_SIZE_M = self.SHIP_LENGTH * 2.2
        self.RANDOM_FIELD_MID_BAND_TYPE = 'Large Floe'
        self.RANDOM_FIELD_MID_BAND_LATERAL_SPREAD_M = self.SHIP_LENGTH * 7.0

        self.RANDOM_FIELD_CLEAR_START_GOAL_RADIUS_M = self.SHIP_LENGTH * 6.0
        self.RANDOM_FIELD_BACKGROUND_LARGE_FLOE_WEIGHT_SCALE = 0.6
        self.RANDOM_FIELD_BACKGROUND_ICE_BANK_WEIGHT_SCALE = 0.4

        self.ASTAR_MAX_ITERATIONS_CAP = 600000
        self.ASTAR_MAX_ITERATIONS_FACTOR = 4.0

        self.RANDOM_FIELD_MID_BAND_STRIP_CHAIN = True
        self.RANDOM_FIELD_MID_BAND_STRIP_SEGMENTS = 3       #原来5
        self.RANDOM_FIELD_MID_BAND_STRIP_MIN_LENGTH_M = self.SHIP_LENGTH * 2.8
        self.RANDOM_FIELD_MID_BAND_STRIP_MAX_LENGTH_M = self.SHIP_LENGTH * 4.5
        self.RANDOM_FIELD_MID_BAND_STRIP_WIDTH_RATIO = 0.22
        self.RANDOM_FIELD_MID_BAND_STRIP_GAP_M = self.SHIP_LENGTH * 0.8#原来0.6
        self.RANDOM_FIELD_MID_BAND_DENSITY_SCALE = 0.5#原来0.75

        # 中央大型多边形障碍物
        self.RANDOM_FIELD_ENABLE_CENTER_OBSTACLE = True
        self.RANDOM_FIELD_CENTER_OBSTACLE_SIZE_M = self.SHIP_LENGTH * 2.0  # 大面积障碍物
        self.RANDOM_FIELD_CENTER_OBSTACLE_VERTICES = 6  # 多边形顶点数
        self.RANDOM_FIELD_CENTER_OBSTACLE_T = 0.5  # 沿直线的位置 (0.5=正中央)
        self.RANDOM_FIELD_CENTER_OBSTACLE_OFFSET_M = 0.0  # 横向偏移
        # 边缘区域细碎冰+稀薄冰（让A*绕行到边缘）
        self.RANDOM_FIELD_ENABLE_EDGE_SPARSE_ZONE = False             # 默认关闭，可手动开启
        self.RANDOM_FIELD_EDGE_ZONE_WIDTH_RATIO = 0.15  # 边缘区域宽度占地图宽度比例
        self.RANDOM_FIELD_EDGE_DENSITY_SCALE = 0.5  # 边缘区域冰密度缩放(更稀薄)0.7
        self.RANDOM_FIELD_EDGE_SIZE_SCALE = 0.5  # 边缘区域冰尺寸缩放(更细碎)0.7

        self.ICE_THETA_WEIGHT_DISTANCE = 1.0
        self.ICE_THETA_WEIGHT_ICE_RESISTANCE = 0.8
        self.ICE_THETA_WEIGHT_SAFETY = 0.6
        self.ICE_THETA_WEIGHT_TURN = 0.35
        self.ICE_THETA_WEIGHT_LINE_COST = 0.9

        self.ICE_THETA_POST_SMOOTH = True
        self.ASTAR_POST_SMOOTH = True

        self.ICE_THETA_POST_SMOOTH_METHOD = 'chaikin'
        self.ICE_THETA_POST_SMOOTH_ITERS = 2

        self.ICE_THETA_HARD_OBSTACLE_TYPES = ('Medium Floe', 'Large Floe', 'Vast Floe', 'Ice Bank')
        self.ICE_THETA_HARD_OBSTACLE_MIN_SIZE_M = self.SHIP_LENGTH * 1.2
        self.ICE_THETA_HARD_OBSTACLE_BUFFER_M = self.SHIP_WIDTH * 0.75  #离大冰距离原来0.75
        self.ICE_THETA_SOFT_CORE_COST_MULT = 2.0

        self.ASTAR_WEIGHT_DISTANCE = 1.0
        self.ASTAR_WEIGHT_ICE_RESISTANCE = 2.0
        self.ASTAR_WEIGHT_SAFETY = 1.5
        self.ASTAR_WEIGHT_TURN = 0.5
        
        # 动态规划
        self.ENABLE_DYNAMIC_REPLANNING = True
        self.REPLANNING_INTERVAL = max(10.0, self.WORLD_HEIGHT / 100)
        
        # 路径跟随（根据船长调整）
        self.PATH_FOLLOW_DISTANCE = self.SHIP_LENGTH * 0.3
        self.PATH_LOOKAHEAD = self.SHIP_LENGTH * 1.5

        self.HYBRID_ASTAR_STEP = max(10.0, self.SHIP_LENGTH * 0.35)
        self.HYBRID_ASTAR_HEADINGS = 16
        
        # PID参数（根据船舶质量自适应）
        self.PID_KP = self.SHIP_MASS * 5.0
        self.PID_KD = self.SHIP_MASS * 0.8
        
        # ==================== 物理参数 ====================
        self.PHYSICS_DT = 1/60.0
        self.PHYSICS_SUBSTEPS = scale_preset.physics_substeps
        self.GRAVITY = (0, 0)
        self.DAMPING = 0.95       #越小阻尼越大
        
        # ==================== 冰阻力参数 ====================
        self.ICE_DENSITY = 917.0
        self.WATER_DENSITY = 1025.0
        self.CHANNEL_WIDTH_RATIO = 1.2

        self.PLANNING_SPEED_MPS = min(self.SHIP_MAX_SPEED, 4.0)
        self.ICE_RESISTANCE_COST_SCALE = 1e-5
        
        # ==================== 流场参数 ====================
        self.ENABLE_CURRENT = False
        self.CURRENT_STRENGTH = 0.1
        self.CURRENT_DIRECTION = 90
        
        # ==================== 控制参数 ====================
        self.MANUAL_FORCE = self.SHIP_MASS * 3.0
        self.MANUAL_TORQUE = self.SHIP_MASS * 4.0
        
        # ==================== 其他参数 ====================
        self.AUTO_SWITCH_ALGORITHM = True
        self.GOAL_REACH_DISTANCE = self.SHIP_LENGTH * 1.0

        # ==================== 冰块质量与惯性参数 (极地真实性调整) ====================
        # 大冰块阈值：超过此尺寸的冰块质量大幅增加，难以推动
        self.IMMOVABLE_ICE_THRESHOLD_SIZE = self.SHIP_LENGTH * 5.0   # 5倍船长 → 完全不可移动
        self.HEAVY_ICE_THRESHOLD_SIZE = self.SHIP_LENGTH * 2.5       # 2.5倍船长 → 重型冰块
        self.LARGE_ICE_THRESHOLD_SIZE = self.SHIP_LENGTH * 1.5       # 1.5倍船长 → 大型冰块
        self.INJECTED_OBSTACLE_USE_MASS_MULTIPLIERS = True  #启用质量倍增器
        self.INJECTED_OBSTACLE_MASS_MULTIPLIER_SCALE = 0.0
        # 质量倍增器：使冰块更难被推动（极地冰山/冰岸模拟）
        self.HEAVY_ICE_MASS_MULTIPLIER = 100.0   # 重型冰块质量×50（原100）
        self.LARGE_ICE_MASS_MULTIPLIER = 30.0    # 大型冰块质量×30（原8）
        # 额外：中型冰块也增加质量
        self.MEDIUM_ICE_MASS_MULTIPLIER = 8.0    # 中型冰块质量×2原来10
        
        # 颜色
        self.COLOR_BACKGROUND = (255, 255, 255)
        self.COLOR_WATER = (240, 248, 255)
        self.COLOR_SHIP = (180, 50, 50)
        self.COLOR_SHIP_OUTLINE = (100, 0, 0)
        self.COLOR_ICE = {
            'Ice Bank': (80, 130, 160),
            'Large Floe': (120, 160, 185),
            'Medium Floe': (150, 185, 205),
            'Small Floe': (180, 205, 220),
            'Fragment': (200, 220, 235)
        }
        self.COLOR_PATH = (255, 0, 0)
        self.COLOR_PATH_VISITED = (255, 150, 150)
        
        self.ICE_OUTLINE_WIDTH = 1
        self.ICE_OUTLINE_COLOR = (60, 60, 60)
        self.SHIP_OUTLINE_WIDTH = 3
        self.PATH_WIDTH = max(2, 6 - int(np.log10(self.WORLD_WIDTH)))
        
        # 输出
        self.OUTPUT_DIR = Path("output_enhanced_simulation")
        self.RECORD_VIDEO = True
        self.VIDEO_FPS = 30
        self.RECORD_DATA = True
        self.DATA_INTERVAL = 0.1

        self.EXPORT_ICE_RESISTANCE_HEATMAP = True

        # ==================== 碰撞与性能开关 ====================
        # 大规模冰场下冰-冰碰撞会导致碰撞对数爆炸，默认关闭，只保留船-冰碰撞
        self.ENABLE_ICE_ICE_COLLISION = False
        self.ENABLE_LOCAL_ICE_ICE_COLLISION = True
        self.LOCAL_ICE_ICE_RADIUS = 750.0  # 扩大冰-冰碰撞范围
        self.LOCAL_ICE_ICE_UPDATE_INTERVAL_FRAMES = 10
        self.ENABLE_DISTANT_ICE_SLEEP = True
        self.DISTANT_ICE_SLEEP_RADIUS = 700.0
        self.ICE_SLEEP_SPEED_THRESHOLD = 0.08
        self.ICE_SLEEP_ANGVEL_THRESHOLD = 0.15
# ... (其他代码保持不变)
        self.ICE_SLEEP_SCAN_BATCH = 400
        self.SPACE_SLEEP_TIME_THRESHOLD = 0.7
        self.SPACE_IDLE_SPEED_THRESHOLD = 0.12

        self.FILTER_SAM_LARGE_REGIONS = False
        self.SAM_MAX_VALID_ICE_AREA = 500000
        # pymunk ShapeFilter 类别
        self.SHIP_COLLISION_CATEGORY = 0b1
        self.ICE_COLLISION_CATEGORY = 0b10
        
        self.SHOW_INFO_PANEL = True
        self.SHOW_FORCES = False
        self.SHOW_VELOCITIES = False
        
        # ⭐ 性能优化选项
        self.FAST_ICE_GENERATION = True  # 快速冰场生成（空间网格加速）
        self.MAX_ICE_BLOCKS = self._calculate_max_ice_blocks()  # 最大冰块数
    
    def _calculate_max_ice_blocks(self) -> int:
        """根据尺度计算合理的冰块数量上限"""
        area = self.WORLD_WIDTH * self.WORLD_HEIGHT
        # 每1000m²约10-30个冰块
        base_count = int(area / 1000 * 20)
        # 大尺度时限制总数
        if area > 1e6:  # >1km²
            return min(base_count, 800)
        elif area > 1e5:  # >0.1km²
            return min(base_count, 500)
        else:
            return min(base_count, 300)
    
    def _estimate_mass(self) -> float:
        """根据船舶尺寸估算质量"""
        # 简化估算: mass ≈ Cb * L * B * d * rho_steel_equivalent
        L = self.SHIP_LENGTH
        B = self.SHIP_WIDTH
        d = L * 0.07  # 吃水约为船长的7%
        Cb = 0.6  # 方形系数
        rho_eff = 250  # 等效密度 kg/m³
        return Cb * L * B * d * rho_eff
    
    @property
    def scale_name(self) -> str:
        return self._scale.value
    
    @property
    def ice_condition_name(self) -> str:
        return self._ice_condition.value
    
    def get_summary(self) -> str:
        """获取配置摘要"""
        return f"""
========== 仿真配置摘要 ==========
尺度: {self.scale_name}
  - 区域: {self.WORLD_WIDTH}m × {self.WORLD_HEIGHT}m
  - 窗口: {self.WINDOW_WIDTH} × {self.WINDOW_HEIGHT}

冰况: {self.ice_condition_name}
  - 覆盖率: {self.ICE_COVERAGE*100:.0f}%
  - 冰厚: {self.ICE_THICKNESS:.2f}m
  - 密集度: {self.ICE_CONCENTRATION*100:.0f}%

船舶: {self._ship_type}
  - 尺寸: {self.SHIP_LENGTH}m × {self.SHIP_WIDTH}m
  - 质量: {self.SHIP_MASS/1000:.0f}t
  - 推力: {self.SHIP_MAX_FORCE/1000:.0f}kN
====================================
"""


# ========== 快速配置函数 ==========

def quick_config(
    scale: str = "medium",
    ice: str = "close_pack", 
    ship: str = "research"
) -> EnhancedConfig:
    """
    快速创建配置
    
    Args:
        scale: small/medium/large/realistic
        ice: open/sparse/close/very_close/compact
        ship: icebreaker_s/icebreaker_m/icebreaker_l/cargo/tanker/research
    
    Returns:
        EnhancedConfig实例
    """
    scale_map = {
        "small": SimulationScale.SMALL,
        "medium": SimulationScale.MEDIUM,
        "large": SimulationScale.LARGE,
        "realistic": SimulationScale.REALISTIC,
    }
    
    ice_map = {
        "open": IceCondition.OPEN_WATER,
        "sparse": IceCondition.OPEN_DRIFT,
        "close": IceCondition.CLOSE_PACK,
        "very_close": IceCondition.VERY_CLOSE,
        "compact": IceCondition.COMPACT,
    }
    
    ship_map = {
        "icebreaker_s": "ICEBREAKER_SMALL",
        "icebreaker_m": "ICEBREAKER_MEDIUM",
        "icebreaker_l": "ICEBREAKER_LARGE",
        "cargo": "CARGO_SHIP",
        "tanker": "TANKER",
        "research": "RESEARCH_VESSEL",
    }
    
    return EnhancedConfig(
        scale=scale_map.get(scale.lower(), SimulationScale.MEDIUM),
        ice_condition=ice_map.get(ice.lower(), IceCondition.CLOSE_PACK),
        ship_type=ship_map.get(ship.lower(), "RESEARCH_VESSEL")
    )


def create_from_scenario(scenario_name: str, scale: str = "medium", 
                         ship: str = "research") -> 'EnhancedConfig':
    """
    从真实世界场景创建配置
    
    Args:
        scenario_name: 场景名称 (见 ice_scenarios.py)
        scale: 尺度 (small/medium/large/realistic)
        ship: 船型
    
    Example:
        config = create_from_scenario("nsr_summer", scale="large", ship="icebreaker_m")
    """
    try:
        from ice_scenarios import get_scenario, IceFeatureType
    except ImportError:
        print("Warning: ice_scenarios.py not found, using default config")
        return quick_config(scale, "close", ship)
    
    scenario = get_scenario(scenario_name)
    if not scenario:
        print(f"Warning: Scenario '{scenario_name}' not found, using default")
        return quick_config(scale, "close", ship)
    
    # 尺度映射
    scale_map = {
        "small": SimulationScale.SMALL,
        "medium": SimulationScale.MEDIUM,
        "large": SimulationScale.LARGE,
        "realistic": SimulationScale.REALISTIC,
    }
    
    # 船型映射
    ship_map = {
        "icebreaker_s": "ICEBREAKER_SMALL",
        "icebreaker_m": "ICEBREAKER_MEDIUM",
        "icebreaker_l": "ICEBREAKER_LARGE",
        "cargo": "CARGO_SHIP",
        "tanker": "TANKER",
        "research": "RESEARCH_VESSEL",
    }
    
    # 将场景的冰特征分布转换为我们的冰块类型
    # 映射: IceFeatureType -> 我们的类型名
    feature_to_type = {
        IceFeatureType.BRASH_ICE: 'Fragment',
        IceFeatureType.CAKE_ICE: 'Small Floe',
        IceFeatureType.SMALL_FLOE: 'Medium Floe',
        IceFeatureType.MEDIUM_FLOE: 'Large Floe',
        IceFeatureType.BIG_FLOE: 'Ice Bank',
        IceFeatureType.VAST_FLOE: 'Ice Bank',
        IceFeatureType.GROWLER: 'Medium Floe',  # 小冰山当中型浮冰
        IceFeatureType.BERGY_BIT: 'Large Floe',
        IceFeatureType.SMALL_ICEBERG: 'Ice Bank',
        IceFeatureType.MEDIUM_ICEBERG: 'Ice Bank',
        IceFeatureType.PRESSURE_RIDGE: 'Ice Bank',
        IceFeatureType.HUMMOCK: 'Large Floe',
        IceFeatureType.RUBBLE: 'Fragment',
        IceFeatureType.PANCAKE_ICE: 'Fragment',
    }
    
    # 转换分布
    distribution = {}
    for feature_type, prob in scenario.feature_distribution.items():
        our_type = feature_to_type.get(feature_type, 'Medium Floe')
        distribution[our_type] = distribution.get(our_type, 0) + prob
    
    # 确保分布有效
    if not distribution:
        distribution = {'Medium Floe': 0.4, 'Small Floe': 0.3, 'Fragment': 0.3}
    
    # 创建自定义IcePreset
    ice_preset = IcePreset(
        coverage=scenario.ice_coverage,
        thickness_mean=scenario.avg_thickness,
        thickness_std=scenario.thickness_std,
        concentration=scenario.ice_concentration,
        size_scale=1.5,  # 可调整
        distribution=distribution
    )
    
    # 根据难度调整尺寸缩放
    if scenario.navigation_difficulty >= 8:
        ice_preset.size_scale = 2.0
    elif scenario.navigation_difficulty >= 5:
        ice_preset.size_scale = 1.5
    else:
        ice_preset.size_scale = 1.2
    
    # 创建配置
    config = EnhancedConfig(
        scale=scale_map.get(scale.lower(), SimulationScale.MEDIUM),
        ice_condition=IceCondition.CLOSE_PACK,  # 占位，实际用ice_preset
        ship_type=ship_map.get(ship.lower(), "RESEARCH_VESSEL")
    )
    
    # 覆盖冰况参数
    config.ICE_COVERAGE = ice_preset.coverage
    config.ICE_THICKNESS = ice_preset.thickness_mean
    config.ICE_CONCENTRATION = ice_preset.concentration
    config.ICE_DISTRIBUTION = ice_preset.distribution
    
    # 根据场景特性添加额外冰特征
    config.HAS_ICEBERGS = scenario.has_icebergs
    config.HAS_FAST_ICE = scenario.has_fast_ice
    config.HAS_PRESSURE_RIDGES = scenario.has_pressure_ridges
    config.SCENARIO_NAME = scenario.name
    config.SCENARIO_DIFFICULTY = scenario.navigation_difficulty
    
    print(f"\n  Loaded scenario: {scenario.name}")
    print(f"  Location: {scenario.location}")
    print(f"  Difficulty: {scenario.navigation_difficulty}/10")
    
    return config


def list_available_scenarios():
    """列出所有可用场景"""
    try:
        from ice_scenarios import print_all_scenarios
        print_all_scenarios()
    except ImportError:
        print("ice_scenarios.py not found")


if __name__ == "__main__":
    # 测试配置
    print("=" * 50)
    print("Enhanced Config System Test")
    print("=" * 50)
    
    # 测试不同配置组合
    configs = [
        ("Small + Open", quick_config("small", "open", "research")),
        ("Medium + Close", quick_config("medium", "close", "icebreaker_m")),
        ("Large + Very Close", quick_config("large", "very_close", "icebreaker_l")),
    ]
    
    for name, cfg in configs:
        print(f"\n[{name}]")
        print(cfg.get_summary())
    
    # 测试真实场景
    print("\n" + "=" * 50)
    print("Real-World Scenario Test")
    print("=" * 50)
    
    list_available_scenarios()
    
    print("\nLoading Northern Sea Route Summer...")
    nsr_config = create_from_scenario("nsr_summer", scale="large", ship="icebreaker_m")
    print(nsr_config.get_summary())
    
    print("\n Config system test complete")
