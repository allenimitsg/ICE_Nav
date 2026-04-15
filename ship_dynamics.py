"""
@File    : ship_dynamics.py
@Author  : Bin MEI
@Date    : 2025-11-01
@Desc    : 2D path planning algorithm.


船舶操纵动力学模型 - 3-DOF MMG Model
======================================

1. 基于MMG (Maneuvering Modeling Group) 标准的船舶操纵模型
2. 考虑舵力、螺旋桨推力、船体水动力
3. 多种真实船型预设（破冰船、货船、科考船）
4. 公里级冰区尺度支持

参考文献：
- Yasukawa, H., & Yoshimura, Y. (2015). Introduction of MMG standard method for ship maneuvering predictions.
- ITTC Recommended Procedures (7.5-02-06-01)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from enum import Enum


class ShipType(Enum):
    """船型枚举"""
    ICEBREAKER_SMALL = "小型破冰船"      # 40-60m
    ICEBREAKER_MEDIUM = "中型破冰船"     # 80-120m  
    ICEBREAKER_LARGE = "大型破冰船"      # 150m+
    CARGO_SHIP = "冰区货船"              # 100-200m
    TANKER = "冰区油轮"                  # 150-300m
    RESEARCH_VESSEL = "科考船"           # 50-100m
    # ========== 大连海事大学校船 ==========
    DMU_YUKUN = "育鲲号"                 # 116m 专用教学实习船
    DMU_YUPENG = "育鹏号"                # 199.8m 多用途重吊船
    DMU_XINHONGZHUAN = "新红专号"        # 69.83m 智能实训船
    CUSTOM = "自定义"


@dataclass
class ShipParameters:
    """
    船舶参数数据类
    
    基于真实船舶的主尺度和水动力系数
    """
    # ========== 主尺度 ==========
    name: str = "Default Ship"
    ship_type: ShipType = ShipType.ICEBREAKER_SMALL
    
    Lpp: float = 80.0           # 垂线间长 (m)
    B: float = 18.0             # 型宽 (m)
    d: float = 6.0              # 吃水 (m)
    displacement: float = 5000.0  # 排水量 (t)
    Cb: float = 0.65            # 方形系数
    
    # ========== 螺旋桨参数（破冰船专用）==========
    Dp: float = 4.0             # 螺旋桨直径 (m)
    pitch_ratio: float = 0.8    # 螺距比 P/D
    n_propellers: int = 2       # 螺旋桨数量（破冰船通常2-3个）
    max_rpm: float = 150.0      # 最大转速 (rpm)
    min_rpm: float = -80.0      # 最小转速 (倒车)
    propulsion_power: float = 5000.0  # 推进功率 (kW)
    
    # ========== 侧推器参数（破冰船关键设备）==========
    has_bow_thruster: bool = True       # 是否有艏侧推
    has_stern_thruster: bool = True     # 是否有艉侧推
    bow_thruster_force: float = 150.0   # 艏侧推推力 (kN)
    stern_thruster_force: float = 100.0 # 艉侧推推力 (kN)
    
    # ========== 舵参数 ==========
    rudder_area: float = 8.0    # 舵面积 (m²)
    rudder_aspect_ratio: float = 1.5  # 舵展弦比
    max_rudder_angle: float = 35.0    # 最大舵角 (度)
    rudder_rate: float = 30.0   # 舵速 (度/秒) - 超快速舵机
    
    # ========== 附加质量系数 ==========
    # 无量纲化附加质量 (m' = m_add / (0.5 * rho * L^2 * d))
    mx_prime: float = 0.022     # 纵向附加质量系数
    my_prime: float = 0.223     # 横向附加质量系数
    Jz_prime: float = 0.011     # 转动惯量附加系数
    
    # ========== 船体水动力导数 (无量纲) ==========
    # 阻力系数
    X_vv_prime: float = -0.040  # 横移阻力
    X_vr_prime: float = 0.002   # 交叉耦合
    X_rr_prime: float = 0.011   # 回转阻力
    
    # 横向力系数
    Y_v_prime: float = -0.315   # 横移线性阻尼
    Y_r_prime: float = 0.083    # 回转线性阻尼
    Y_vvv_prime: float = -1.607 # 横移非线性
    Y_vvr_prime: float = 0.379  # 交叉耦合
    Y_vrr_prime: float = -0.391
    Y_rrr_prime: float = 0.008
    
    # 转艏力矩系数
    N_v_prime: float = -0.137   # 横移产生的转艏力矩
    N_r_prime: float = -0.015   # 回转阻尼(减小)
    N_vvv_prime: float = -0.030
    N_vvr_prime: float = -0.294
    N_vrr_prime: float = 0.055
    N_rrr_prime: float = -0.013
    
    # ========== 冰区性能参数 ==========
    ice_class: str = "PC5"      # 极地船级 (PC1-PC7, 1最强)
    ice_belt_thickness: float = 0.025  # 冰带厚度 (m)
    bow_angle: float = 25.0     # 艏柱倾角 (度)
    stem_angle: float = 30.0    # 水线艏角 (度)
    
    # ========== 性能极限 ==========
    max_speed: float = 15.0     # 最大航速 (节)
    min_turning_radius: float = 0.0  # 最小转弯半径 (m)，0表示自动计算
    
    def __post_init__(self):
        """后处理计算派生参数"""
        # 自动计算最小转弯半径（约3-5倍船长）
        if self.min_turning_radius == 0:
            self.min_turning_radius = self.Lpp * 3.5
        
        # 计算质量
        self.mass = self.displacement * 1000  # kg
        
        # 计算惯性矩（大幅减小以增加灵活性）
        self.Izz = self.mass * (self.Lpp ** 2) * 0.08 ** 2  # 非常小的回转半径


# ========== 预设船型库 ==========
SHIP_PRESETS: Dict[ShipType, ShipParameters] = {
    
    ShipType.ICEBREAKER_SMALL: ShipParameters(
        name="雪龙型科考破冰船",
        ship_type=ShipType.ICEBREAKER_SMALL,
        Lpp=50.0, B=12.0, d=4.5,
        displacement=2500.0, Cb=0.60,
        Dp=2.5, max_rpm=180, n_propellers=2,
        propulsion_power=5000.0,  # 5MW
        rudder_area=4.0, max_rudder_angle=35.0,
        ice_class="PC6", bow_angle=22.0,
        max_speed=14.0,
    ),
    
    ShipType.ICEBREAKER_MEDIUM: ShipParameters(
        name="雪龙2号型",
        ship_type=ShipType.ICEBREAKER_MEDIUM,
        Lpp=122.5, B=22.3, d=7.85,
        displacement=13990.0, Cb=0.58,
        Dp=4.5, max_rpm=150, n_propellers=2,
        propulsion_power=15000.0,  # 雪龙2号约15MW推进功率
        rudder_area=12.0, max_rudder_angle=35.0,
        ice_class="PC3", bow_angle=20.0, stem_angle=25.0,
        max_speed=15.0,
    ),
    
    ShipType.ICEBREAKER_LARGE: ShipParameters(
        name="核动力破冰船 (Arktika级)",
        ship_type=ShipType.ICEBREAKER_LARGE,
        Lpp=173.3, B=34.0, d=11.0,
        displacement=33540.0, Cb=0.62,
        Dp=6.2, max_rpm=120, n_propellers=3,
        propulsion_power=60000.0,  # 核动力约60MW
        rudder_area=25.0, max_rudder_angle=40.0,
        ice_class="PC1", bow_angle=18.0,
        max_speed=22.0,
    ),
    
    ShipType.CARGO_SHIP: ShipParameters(
        name="冰区加强货船",
        ship_type=ShipType.CARGO_SHIP,
        Lpp=150.0, B=25.0, d=9.0,
        displacement=20000.0, Cb=0.78,
        Dp=5.5, max_rpm=100, n_propellers=1,
        propulsion_power=8000.0,  # 8MW
        rudder_area=18.0, max_rudder_angle=35.0,
        ice_class="IA Super", bow_angle=30.0,
        max_speed=14.0,
    ),
    
    ShipType.TANKER: ShipParameters(
        name="冰区穿梭油轮",
        ship_type=ShipType.TANKER,
        Lpp=250.0, B=44.0, d=14.5,
        displacement=100000.0, Cb=0.82,
        Dp=7.0, max_rpm=80, n_propellers=1,
        propulsion_power=15000.0,  # 15MW
        rudder_area=40.0, max_rudder_angle=35.0,
        ice_class="IA", bow_angle=35.0,
        max_speed=12.0,
    ),
    
    ShipType.RESEARCH_VESSEL: ShipParameters(
        name="极地科考船",
        ship_type=ShipType.RESEARCH_VESSEL,
        Lpp=80.0, B=16.0, d=5.5,
        displacement=4500.0, Cb=0.58,
        Dp=3.2, max_rpm=160, n_propellers=2,
        propulsion_power=6000.0,  # 6MW
        rudder_area=6.0, max_rudder_angle=35.0,
        ice_class="PC5", bow_angle=23.0,
        max_speed=13.0,
    ),
    
    # ==================== 大连海事大学校船 ====================
    # 数据来源: 中国航海学会、国防科工局、信德海事网等公开资料
    
    ShipType.DMU_YUKUN: ShipParameters(
        # 育鲲号 - 大连海事大学首艘专用远洋教学实习船
        # 2008年投入使用，上海船舶设计研究院设计，武昌造船厂建造
        # 我国第一艘现代化的专用远洋教学实习船
        name="育鲲号 (DMU Yu Kun)",
        ship_type=ShipType.DMU_YUKUN,
        Lpp=116.0, B=18.0, d=5.5,          # 总长116m，型宽18m，吃水约5.5m
        displacement=5500.0, Cb=0.60,       # 排水量约5500吨（估算）
        Dp=3.8, max_rpm=150, n_propellers=1,
        propulsion_power=4000.0,            # 推进功率约4MW（估算）
        has_bow_thruster=True,              # 配备首侧推器
        has_stern_thruster=False,
        bow_thruster_force=120.0,           # 首侧推约120kN（估算）
        rudder_area=8.0, max_rudder_angle=35.0,
        ice_class="B1",                     # 一般冰区加强
        bow_angle=28.0,
        max_speed=15.0,                     # 最大航速约15节
        # 特色：可调螺距螺旋桨(CPP)、轴带发电机、减摇鳍
        # 配员236人（40名船员教师+196名实习学生）
    ),
    
    ShipType.DMU_YUPENG: ShipParameters(
        # 育鹏号 - 中国第二代第一艘现代化多用途载货教学实习船
        # 2016年建造，2017年首航
        # 远洋单桨柴油机驱动的多用途重吊船
        name="育鹏号 (DMU Yu Peng)",
        ship_type=ShipType.DMU_YUPENG,
        Lpp=189.00, B=27.8, d=11.0,          # 总长199.8m，型宽27.8m，设计吃水10.3m
        displacement=38000.0, Cb=0.75,      # 满载排水量约38000吨，载重吨29774DWT
        Dp=6.0, max_rpm=100, n_propellers=1,# 单桨柴油机
        propulsion_power=12000.0,           # 推进功率约12MW（估算）
        has_bow_thruster=True,
        has_stern_thruster=True,
        bow_thruster_force=200.0,           # 艏侧推约200kN（估算）
        stern_thruster_force=150.0,         # 艉侧推约150kN（估算）
        rudder_area=35.0, max_rudder_angle=35.0,
        ice_class="IA",                     # 冰区加强
        bow_angle=30.0,
        max_speed=18.3,                     # 服务航速17.5节
        # 特色：续航力15000海里，装箱量1769TEU
        # 可装运散货、杂货、工程件和集装箱
    ),
    
    ShipType.DMU_XINHONGZHUAN: ShipParameters(
        # 新红专号 - 全球首艘集远程遥控、自主航行与教学实训于一身的智能船
        # 2024年交付，上海船舶设计研究院设计
        # 大连海事大学科技创新"十四五"规划重大建设项目
        name="新红专号 (DMU Xin Hong Zhuan)",
        ship_type=ShipType.DMU_XINHONGZHUAN,
        Lpp=69.83, B=10.9, d=3.5,           # 船长69.83m，型宽10.9m，型深5m
        displacement=1430.0, Cb=0.55,       # 排水量1430吨
        Dp=2.5, max_rpm=0, n_propellers=2,  # 双全电力吊舱推进（无传统螺旋桨转速概念）
        propulsion_power=3000.0,            # 2×1500kW = 3MW吊舱推进
        has_bow_thruster=True,
        has_stern_thruster=False,
        bow_thruster_force=80.0,            # 艏侧推约80kN（估算）
        rudder_area=0.0,                    # 吊舱推进，无传统舵
        max_rudder_angle=180.0,             # 吊舱可360°旋转
        rudder_rate=15.0,                   # 吊舱转向较慢
        ice_class="B1",
        bow_angle=25.0,
        max_speed=17.5,                     # 服务航速17.5节，设计航速18节
        # 特色：远程遥控、自主航行、自主避碰
        # 智能化水平世界领先，可容纳35名科研人员
        # 发电：3×1520kW柴电机组
    ),
}


class ShipDynamicsModel:
    """
    船舶3-DOF MMG操纵运动模型
    
    状态变量: [x, y, psi, u, v, r]
    - x, y: 船舶位置 (m)
    - psi: 艏向角 (rad), 0=东, pi/2=北
    - u: 纵向速度 surge (m/s)
    - v: 横向速度 sway (m/s)  
    - r: 艏摇角速度 yaw rate (rad/s)
    
    控制输入: [n, delta]
    - n: 螺旋桨转速 (rpm)
    - delta: 舵角 (rad)
    """
    
    # 物理常数
    RHO_WATER = 1025.0      # 海水密度 kg/m³
    RHO_ICE = 917.0         # 海冰密度 kg/m³
    G = 9.81                # 重力加速度
    
    def __init__(self, ship_params: ShipParameters):
        self.params = ship_params
        self._compute_dimensional_coefficients()
        
        # 状态 [x, y, psi, u, v, r]
        self.state = np.zeros(6)
        
        # 控制输入
        self.n = 0.0        # 螺旋桨转速 (rpm)
        self.delta = 0.0    # 舵角 (rad)
        self.delta_cmd = 0.0  # 舵角指令
        
        # 侧推器控制（破冰船专用）
        self.bow_thruster_cmd = 0.0    # 艏侧推指令 (-1到+1)
        self.stern_thruster_cmd = 0.0  # 艉侧推指令 (-1到+1)
        
        # 外力（冰阻力等）
        self.external_force = np.zeros(3)  # [Fx, Fy, Mz]
    
    def _compute_dimensional_coefficients(self):
        """将无量纲系数转换为有量纲系数"""
        p = self.params
        L = p.Lpp
        d = p.d
        rho = self.RHO_WATER
        
        # 附加质量（有量纲）
        m = p.mass
        self.mx = p.mx_prime * 0.5 * rho * L**2 * d
        self.my = p.my_prime * 0.5 * rho * L**2 * d
        self.Jz = p.Izz + p.Jz_prime * 0.5 * rho * L**4 * d
        
        # 有效质量矩阵
        self.m11 = m + self.mx
        self.m22 = m + self.my
        self.m33 = self.Jz
        
        # 计算推力系数 (简化的KT曲线)
        self.KT0 = 0.4  # 系柱推力系数
        self.KT1 = -0.5  # 进速系数
        
    def set_state(self, x: float, y: float, psi: float, 
                  u: float = 0, v: float = 0, r: float = 0):
        """设置船舶状态"""
        self.state = np.array([x, y, psi, u, v, r])
    
    def set_control(self, n_cmd: float, delta_cmd: float):
        """
        设置控制指令
        
        Args:
            n_cmd: 螺旋桨转速指令 (rpm)
            delta_cmd: 舵角指令 (度)
        """
        # 限幅
        self.n = np.clip(n_cmd, self.params.min_rpm, self.params.max_rpm)
        self.delta_cmd = np.clip(np.radians(delta_cmd), 
                                  -np.radians(self.params.max_rudder_angle),
                                  np.radians(self.params.max_rudder_angle))
    
    def apply_external_force(self, Fx: float, Fy: float, Mz: float):
        """
        施加外部力（冰阻力、碰撞力等）
        
        Args:
            Fx: 纵向力 (N)
            Fy: 横向力 (N)
            Mz: 转艏力矩 (N·m)
        """
        self.external_force = np.array([Fx, Fy, Mz])
    
    def _compute_propeller_thrust(self, u: float) -> float:
        """
        计算螺旋桨推力（简化但可靠的模型）
        
        直接使用系泊推力公式，确保破冰船有足够推力
        """
        if abs(self.n) < 1.0:
            return 0.0
        
        # 转速比例 (相对最大转速)
        n_ratio = abs(self.n) / self.params.max_rpm
        
        # 系泊推力公式: T = 13 * P^0.8 (kW -> kN)
        # 这是经验公式，适用于破冰船
        power_kw = self.params.propulsion_power
        T_bollard_kn = 13.0 * (power_kw ** 0.8)  # kN
        T_bollard = T_bollard_kn * 1000 * n_ratio  # 转换为N
        
        # 高速时推力下降（推进效率曲线）
        u_design = self.params.max_speed * 0.5144  # 设计航速 m/s
        if abs(u) > 1.0:
            # 速度越高，推力越低（但不低于60%系泊推力）
            speed_factor = max(0.6, 1.0 - 0.4 * abs(u) / u_design)
            T = T_bollard * speed_factor
        else:
            # 低速/系泊状态使用全推力
            T = T_bollard
        
        # 推力方向
        T = T * np.sign(self.n)
        
        # 推力减额（破冰船较小）
        thrust_deduction = 0.1
        T_effective = T * (1 - thrust_deduction)
        
        # 调试输出（每5秒打印一次）
        if hasattr(self, '_last_thrust_print'):
            import time
            if time.time() - self._last_thrust_print > 5.0:
                print(f"  [DEBUG] Thrust: {T_effective/1e6:.2f} MN, RPM: {self.n:.0f}, n_ratio: {n_ratio:.2f}")
                self._last_thrust_print = time.time()
        else:
            import time
            self._last_thrust_print = time.time()
        
        return T_effective
    
    def _compute_rudder_force(self, u: float, v: float, r: float) -> Tuple[float, float, float]:
        """
        计算舵力
        
        Returns:
            (X_R, Y_R, N_R): 舵力和力矩
        """
        L = self.params.Lpp
        AR = self.params.rudder_area
        
        # 舵处流速（考虑螺旋桨加速）
        n_rps = self.n / 60.0
        Dp = self.params.Dp
        
        # 螺旋桨尾流加速
        if abs(n_rps) > 0.01:
            CT = 8 * self.KT0 / (np.pi * (1 + 0.8)**2)  # 推力载荷系数
            k_prop = 0.5 * (1 + np.sqrt(1 + CT))
        else:
            k_prop = 1.0
        
        # 舵处有效流速
        u_R = u * k_prop * 0.9  # 考虑尾流
        v_R = v - r * L * 0.5  # 舵处横向速度
        
        U_R = np.sqrt(u_R**2 + v_R**2 + 1e-6)
        
        # 有效攻角
        alpha_R = self.delta - np.arctan2(v_R, u_R + 1e-6)
        
        # 舵法向力 (升力公式)
        # Cl ≈ 2π sin(α) 小攻角近似
        # 考虑展弦比修正
        AR_geo = self.params.rudder_aspect_ratio
        Cl = 2 * np.pi * AR_geo / (AR_geo + 2) * np.sin(alpha_R)
        
        F_N = 0.5 * self.RHO_WATER * AR * U_R**2 * Cl
        
        # ===== 增强舵效：10倍舵力 =====
        F_N *= 10.0  # 让船更灵活
        
        # 转换到船体坐标系
        # 右舵(delta>0)时：水流被偏向右舷，舵受到向左（port）的反作用力
        # 这个力作用在船尾，产生使船向右转的力矩
        X_R = -F_N * np.sin(self.delta)  # 阻力分量（总是向后）
        Y_R = F_N * np.cos(self.delta)   # 横向力：右舵→正（向port）
        
        # 舵力矩（舵在船尾，x_R < 0）
        # 右舵：Y_R > 0，x_R < 0 → N_R = x_R * Y_R < 0 → r减小 → 向右转 ✓
        x_R = -L * 0.5  # 舵位置（船尾）
        N_R = x_R * Y_R  # 力矩 = 力臂 × 力
        
        return X_R, Y_R, N_R
    
    def _compute_hull_forces(self, u: float, v: float, r: float) -> Tuple[float, float, float]:
        """
        计算船体水动力
        
        使用MMG模型的多项式形式
        """
        p = self.params
        L = p.Lpp
        d = p.d
        rho = self.RHO_WATER
        
        # 特征速度
        U = np.sqrt(u**2 + v**2 + 1e-6)
        
        # 无量纲化速度
        v_prime = v / (U + 1e-6)
        r_prime = r * L / (U + 1e-6)
        
        # 动压
        q = 0.5 * rho * U**2
        
        # 纵向阻力 (主要来自摩擦阻力和形状阻力)
        Cf = 0.003  # 摩擦阻力系数
        S = L * d * 2.5  # 湿表面积估算
        R_0 = -Cf * q * S  # 直航阻力
        
        # MMG船体力
        X_H = R_0 + q * L * d * (
            p.X_vv_prime * v_prime**2 +
            p.X_vr_prime * v_prime * r_prime +
            p.X_rr_prime * r_prime**2
        )
        
        Y_H = q * L * d * (
            p.Y_v_prime * v_prime +
            p.Y_r_prime * r_prime +
            p.Y_vvv_prime * v_prime**3 +
            p.Y_vvr_prime * v_prime**2 * r_prime +
            p.Y_vrr_prime * v_prime * r_prime**2 +
            p.Y_rrr_prime * r_prime**3
        )
        
        N_H = q * L**2 * d * (
            p.N_v_prime * v_prime +
            p.N_r_prime * r_prime +
            p.N_vvv_prime * v_prime**3 +
            p.N_vvr_prime * v_prime**2 * r_prime +
            p.N_vrr_prime * v_prime * r_prime**2 +
            p.N_rrr_prime * r_prime**3
        )
        
        return X_H, Y_H, N_H
    
    def _compute_thruster_force(self) -> Tuple[float, float]:
        """
        计算侧推器力（艏侧推 + 艉侧推）
        
        破冰船专用侧推系统，用于低速时辅助转向
        
        Returns:
            (Y_T, N_T): 横向力和转艏力矩
        """
        p = self.params
        
        # 检查是否有侧推器
        if not getattr(p, 'has_bow_thruster', False):
            return 0.0, 0.0
        
        # 获取侧推指令（-1到+1）
        bow_cmd = getattr(self, 'bow_thruster_cmd', 0.0)
        stern_cmd = getattr(self, 'stern_thruster_cmd', 0.0)
        
        # 侧推力（kN → N）
        F_bow = bow_cmd * p.bow_thruster_force * 1000
        F_stern = stern_cmd * getattr(p, 'stern_thruster_force', 0.0) * 1000
        
        # 总横向力
        Y_T = F_bow + F_stern
        
        # 转艏力矩（艏推在船头产生正力矩，艉推在船尾产生负力矩）
        # 艏推位置约在船头0.4L处，艉推在船尾-0.4L处
        x_bow = 0.4 * p.Lpp
        x_stern = -0.4 * p.Lpp
        
        N_T = F_bow * x_bow + F_stern * x_stern
        
        return Y_T, N_T
    
    def compute_ice_resistance(self, ice_concentration: float, 
                                ice_thickness: float,
                                ship_speed: float,
                                ice_floe_diameter: float = None,
                                channel_width_factor: float = 3.0) -> float:
        """
        混合冰阻力模型：Jeong经验公式 + Lindqvist物理公式
        
        - 碎冰区 (C_ice < 0.7): 使用Jeong公式（基于Araon破冰船模型试验）
        - 密集区 (C_ice >= 0.7): 使用Lindqvist公式（连续破冰）
        
        Jeong公式 (1.9):
        R_ice = 10^(2.651) × Fn_h^(-1.665) × (D_f/B)^(1.019) × C_ice^5.196 
                × (W_ch/B)^(-1.211) × 0.5 × ρ_ice × B × h_ice × V^2
        
        Args:
            ice_concentration: 冰密集度 (0-1)
            ice_thickness: 冰厚 (m)
            ship_speed: 船速 (m/s)
            ice_floe_diameter: 冰块直径 (m)，None时自动估算
            channel_width_factor: 航道宽度 = 船宽 × 此系数
        
        Returns:
            冰阻力 (N)
        """
        if ice_concentration < 0.1 or ice_thickness < 0.05:
            return 0.0
        
        p = self.params
        B = p.B
        
        # 根据冰密集度选择公式
        if ice_concentration < 0.7:
            # ===== Jeong经验公式（碎冰区）=====
            return self._jeong_ice_resistance(
                ice_concentration, ice_thickness, ship_speed,
                ice_floe_diameter, channel_width_factor
            )
        else:
            # ===== Lindqvist物理公式（密集区）=====
            return self._lindqvist_ice_resistance(
                ice_concentration, ice_thickness, ship_speed
            )
    
    def _jeong_ice_resistance(self, C_ice: float, h_ice: float, V_ship: float,
                               D_f: float = None, W_ch_factor: float = 3.0) -> float:
        """
        Jeong经验公式 (Colbourne无量纲分析法)
        
        基于"Araon"破冰船模型试验，适用于碎冰区
        
        R_ice = 10^(2.651) × Fn_h^(-1.665) × (D_f/B)^(1.019) × C_ice^5.196 
                × (W_ch/B)^(-1.211) × 0.5 × ρ_ice × B × h_ice × V^2
        """
        p = self.params
        g = 9.81
        rho_ice = 917.0  # 冰密度 kg/m³
        B = p.B
        
        # 确保速度有效
        V_ship = max(V_ship, 0.5)  # 最小0.5 m/s避免除零
        
        # Froude数（基于冰厚）
        Fn_h = V_ship / np.sqrt(g * h_ice)
        Fn_h = max(Fn_h, 0.1)  # 避免过小
        
        # 冰块等效直径（如果未提供，根据冰厚估算）
        if D_f is None:
            D_f = h_ice * 10  # 经验：直径约为厚度的10倍
        D_f = max(D_f, 1.0)  # 最小1米
        
        # 航道宽度
        W_ch = B * W_ch_factor
        
        # Jeong公式系数
        K = 10**(2.651)  # 系数
        
        # 无量纲参数
        term1 = Fn_h ** (-1.665)
        term2 = (D_f / B) ** (1.019)
        term3 = C_ice ** 5.196
        term4 = (W_ch / B) ** (-1.211)
        
        # 基础阻力项
        R_base = 0.5 * rho_ice * B * h_ice * V_ship**2
        
        # 总阻力
        R_ice = K * term1 * term2 * term3 * term4 * R_base
        
        # 限制合理范围（避免极端值）
        R_ice = min(R_ice, 10e6)  # 最大10 MN
        
        return R_ice
    
    def _lindqvist_ice_resistance(self, ice_concentration: float,
                                   ice_thickness: float,
                                   ship_speed: float) -> float:
        """
        Lindqvist物理公式（密集冰区/连续破冰）
        
        R_ice = R_b (破碎) + R_c (压碎) + R_f (摩擦)
        """
        p = self.params
        L = p.Lpp
        B = p.B
        
        # 冰力学参数
        sigma_b = 500e3   # 冰弯曲强度 Pa
        sigma_c = 2000e3  # 冰压缩强度 Pa
        mu = 0.15         # 冰-船摩擦系数
        
        # 艏角
        phi = np.radians(p.bow_angle)
        psi = np.radians(p.stem_angle)
        
        # 破碎力分量
        R_b = 0.003 * sigma_b * B * ice_thickness**1.5 / np.sqrt(np.cos(phi) + 1e-6)
        
        # 压碎力分量  
        R_c = 0.5 * sigma_c * ice_thickness * B * np.tan(phi + psi/2)
        
        # 摩擦力分量
        R_f = mu * (R_b + R_c) * (1 + 0.5 * ice_concentration)
        
        # 速度修正（低速时阻力增大）
        if ship_speed < 1.0:
            speed_factor = 1.5
        else:
            speed_factor = 1.0 + 0.5 / ship_speed
        
        # 密集度修正
        conc_factor = ice_concentration ** 1.5
        
        R_ice = (R_b + R_c + R_f) * speed_factor * conc_factor
        
        return R_ice
    
    def step(self, dt: float) -> np.ndarray:
        """
        单步积分
        
        Args:
            dt: 时间步长 (s)
        
        Returns:
            新状态 [x, y, psi, u, v, r]
        """
        # 舵机动态（一阶延迟 + 速度限制）
        delta_error = self.delta_cmd - self.delta
        max_delta_rate = np.radians(self.params.rudder_rate) * dt
        delta_change = np.clip(delta_error, -max_delta_rate, max_delta_rate)
        self.delta += delta_change
        
        # 当前状态
        x, y, psi, u, v, r = self.state
        
        # 计算各分力
        T = self._compute_propeller_thrust(u)
        X_R, Y_R, N_R = self._compute_rudder_force(u, v, r)
        X_H, Y_H, N_H = self._compute_hull_forces(u, v, r)
        
        # ========== 侧推器力 ==========
        Y_T, N_T = self._compute_thruster_force()
        
        # 总力（包含侧推）
        X = T + X_R + X_H + self.external_force[0]
        Y = Y_R + Y_H + Y_T + self.external_force[1]
        N = N_R + N_H + N_T + self.external_force[2]
        
        # 运动方程 (考虑附加质量和耦合)
        # m11 * du/dt = X + (m22) * v * r
        # m22 * dv/dt = Y - (m11) * u * r
        # m33 * dr/dt = N + (m11 - m22) * u * v
        
        du = (X + self.m22 * v * r) / self.m11
        dv = (Y - self.m11 * u * r) / self.m22
        dr = (N + (self.m11 - self.m22) * u * v) / self.m33
        
        # 速度更新（欧拉积分）
        u_new = u + du * dt
        v_new = v + dv * dt
        r_new = r + dr * dt
        
        # 速度限幅
        max_speed = self.params.max_speed * 0.5144  # 节转m/s
        speed = np.sqrt(u_new**2 + v_new**2)
        if speed > max_speed:
            scale = max_speed / speed
            u_new *= scale
            v_new *= scale
        
        # 位置更新（船体坐标系 -> 大地坐标系）
        dx = u_new * np.cos(psi) - v_new * np.sin(psi)
        dy = u_new * np.sin(psi) + v_new * np.cos(psi)
        dpsi = r_new
        
        x_new = x + dx * dt
        y_new = y + dy * dt
        psi_new = psi + dpsi * dt
        
        # 角度归一化
        psi_new = np.arctan2(np.sin(psi_new), np.cos(psi_new))
        
        # 更新状态
        self.state = np.array([x_new, y_new, psi_new, u_new, v_new, r_new])
        
        # 清除外力（每步需重新设置）
        self.external_force = np.zeros(3)
        
        return self.state
    
    @property
    def position(self) -> Tuple[float, float]:
        return self.state[0], self.state[1]
    
    @property
    def heading(self) -> float:
        return self.state[2]
    
    @property
    def velocity(self) -> Tuple[float, float, float]:
        return self.state[3], self.state[4], self.state[5]
    
    @property
    def speed(self) -> float:
        return np.sqrt(self.state[3]**2 + self.state[4]**2)
    
    def get_turning_radius(self) -> float:
        """计算当前转弯半径"""
        u, v, r = self.state[3:6]
        if abs(r) < 1e-6:
            return float('inf')
        return np.sqrt(u**2 + v**2) / abs(r)


class AutoPilot:
    """
    船舶自动舵 - 高级轨迹跟踪控制器
    
    核心改进（基于导师建议）：
    1. 45°规则 - 引导点角度约束，确保平滑跟踪
    2. Cross-Track Error (XTE) 控制 - 最小化航迹偏差
    3. 侧推器辅助 - 低速时使用艏艉侧推
    4. 更远的前视距离 - 提前规划转向
    """
    
    def __init__(self, ship_model: ShipDynamicsModel):
        self.ship = ship_model
        
        # ========== 45°规则参数 ==========
        self.max_guidance_angle = 45.0   # 最大引导角（度）- 导师要求!
        self.lookahead_multiplier = 12.0 # 前视距离 = 船长 × 此系数（增大！）
        self.min_lookahead = 80.0        # 最小前视距离(米)
        self.max_lookahead = 250.0       # 最大前视距离(米)
        
        # ========== XTE控制参数 ==========
        self.Kp_xte = 0.02       # 横向偏差比例增益
        self.max_xte_correction = 15.0  # XTE修正的最大舵角贡献（度）
        
        # ========== 航向PD控制 ==========
        self.Kp_heading = 2.0    # 航向比例系数
        self.Kd_heading = 4.0    # 航向微分系数（增大阻尼）
        
        # ========== 侧推器控制 ==========
        self.use_thruster = True         # 是否使用侧推
        self.thruster_speed_threshold = 3.0  # 低于此速度使用侧推（m/s）
        self.Kp_thruster = 0.5           # 侧推比例增益
        
        # 舵角限制
        self.max_rudder_soft = 20.0
        self.max_rudder_hard = 35.0
        
        # 速度控制
        self.target_speed = 6.0
        
        # 路径点
        self.waypoints = []
        self.current_wp_index = 0
        
        # 状态记录
        self.last_xte = 0.0      # 上一步的横向偏差
        self.bow_thruster_cmd = 0.0  # 艏侧推指令（-1到+1）
        self.need_replan = False     # 是否需要重规划（大偏差时设置）
        
    def set_waypoints(self, waypoints: list):
        """设置航路点列表 [(x1,y1), (x2,y2), ...]"""
        self.waypoints = waypoints
        self.current_wp_index = 0
    
    def set_target_speed(self, speed_knots: float):
        """设置目标航速（节）"""
        self.target_speed = speed_knots * 0.5144  # 转m/s
    
    def compute_control(self, ice_positions: list = None) -> Tuple[float, float]:
        """
        高级自动舵 - 45°规则 + XTE控制 + 侧推辅助
        
        核心改进：
        1. 引导点必须在主航线方向45°以内
        2. 计算并修正横向轨迹偏差(XTE)
        3. 低速时使用侧推器辅助转向
        """
        if not self.waypoints or len(self.waypoints) < 2:
            return 100.0, 0.0
        
        x, y = self.ship.position
        psi = self.ship.heading
        u = self.ship.state[3]  # 纵向速度
        v = self.ship.state[4]  # 横向速度
        r = self.ship.state[5]  # 转艏角速度
        L = self.ship.params.Lpp
        r_deg = np.degrees(r)
        speed = np.sqrt(u*u + v*v)
        
        # ===== 1. 计算前视距离（基于速度自适应）=====
        lookahead = max(
            self.min_lookahead,
            min(self.max_lookahead, L * self.lookahead_multiplier)
        )
        # 高速时增加前视距离
        if speed > 5.0:
            lookahead *= (1 + (speed - 5.0) * 0.1)
        
        # ===== 2. 找满足45°规则的引导点 =====
        # 
        # 45°规则（导师要求）：
        # 三角形：A(船舶正横点) - S(船舶中心) - G(引导点)
        # 顶角 ∠G < 45°，即 arctan(XTE / dist_SG) < 45°
        # 等价于：引导点距离 > 横向偏差 (XTE)
        #
        # ★ 关键限制：最多跳过50个航点，防止跳到路径末尾 ★
        #
        target_idx = self.current_wp_index
        max_skip = 50  # 最多跳过50个点
        skip_count = 0
        
        for i in range(self.current_wp_index, len(self.waypoints)):
            wp = self.waypoints[i]
            dx = wp[0] - x
            dy = wp[1] - y
            dist_to_wp = np.sqrt(dx*dx + dy*dy)
            
            # 跳过太近的点（但也计入跳过次数）
            if dist_to_wp < lookahead * 0.3:
                target_idx = i + 1
                skip_count += 1
                if skip_count >= max_skip:
                    target_idx = i
                    break
                continue
            
            # 计算Cross-Track Error (XTE) = 船到航线段的垂直距离
            if i > 0:
                wp_prev = self.waypoints[i - 1]
            else:
                wp_prev = self.waypoints[0]
            
            # 航线向量
            track_vec = np.array([wp[0] - wp_prev[0], wp[1] - wp_prev[1]])
            track_len = np.linalg.norm(track_vec)
            
            if track_len > 1e-6:
                track_unit = track_vec / track_len
                ship_vec = np.array([x - wp_prev[0], y - wp_prev[1]])
                # XTE = 叉积 = 船到航线的垂直距离
                xte_local = abs(track_unit[0] * ship_vec[1] - track_unit[1] * ship_vec[0])
            else:
                xte_local = 0.0
            
            # ★ 45°规则检查（有跳过限制）★
            if xte_local > 0.1 and skip_count < max_skip:
                apex_angle = np.degrees(np.arctan2(xte_local, dist_to_wp))
                
                if apex_angle > self.max_guidance_angle:
                    # 顶角太大，尝试找更远的引导点
                    if i < len(self.waypoints) - 1:
                        target_idx = i + 1
                        skip_count += 1
                        continue
            
            # 检查引导点是否在船前方
            angle_to_wp = np.arctan2(dy, dx) - psi
            angle_to_wp = np.arctan2(np.sin(angle_to_wp), np.cos(angle_to_wp))
            if abs(angle_to_wp) > np.pi * 0.6 and skip_count < max_skip:  # 放宽到108°
                target_idx = i + 1
                skip_count += 1
                continue
            
            target_idx = i
            break
        
        target_idx = min(target_idx, len(self.waypoints) - 1)
        self.current_wp_index = target_idx
        
        # ★ 大偏差警告（用于触发重规划）★
        self.need_replan = False
        if skip_count >= max_skip:
            self.need_replan = True  # 告诉上层需要重规划
        
        # ===== 3. 计算Cross-Track Error (XTE) =====
        target = self.waypoints[target_idx]
        
        # 找航线段（从前一点到目标点）
        if target_idx > 0:
            wp_prev = self.waypoints[target_idx - 1]
        else:
            wp_prev = self.waypoints[0]
        
        # 航线向量
        track_vec = np.array([target[0] - wp_prev[0], target[1] - wp_prev[1]])
        track_len = np.linalg.norm(track_vec)
        
        if track_len > 1e-6:
            track_unit = track_vec / track_len
            # 船相对于前一航点的位置
            ship_vec = np.array([x - wp_prev[0], y - wp_prev[1]])
            # XTE = 叉积 = 船到航线的垂直距离（正=在航线右侧）
            xte = track_unit[0] * ship_vec[1] - track_unit[1] * ship_vec[0]
        else:
            xte = 0.0
        
        self.last_xte = xte
        
        # ===== 4. 计算目标航向 =====
        dx = target[0] - x
        dy = target[1] - y
        psi_target = np.arctan2(dy, dx)
        
        # 航向误差
        err = psi_target - psi
        err = np.arctan2(np.sin(err), np.cos(err))
        err_deg = np.degrees(err)
        
        # ===== 5. PD控制 + XTE修正 =====
        # 主控制：航向误差
        delta_heading = -self.Kp_heading * err_deg + self.Kd_heading * r_deg
        
        # XTE修正：偏离航线时增加舵角修正
        xte_correction = -self.Kp_xte * xte * 100  # 缩放到合适范围
        xte_correction = np.clip(xte_correction, -self.max_xte_correction, self.max_xte_correction)
        
        delta_deg = delta_heading + xte_correction
        
        # ===== 6. 侧推器控制（低速时辅助转向）=====
        self.bow_thruster_cmd = 0.0
        if self.use_thruster and self.ship.params.has_bow_thruster:
            if speed < self.thruster_speed_threshold:
                # 低速时使用侧推辅助
                # 正的err_deg表示需要左转 → 正的侧推（向左推）
                self.bow_thruster_cmd = np.clip(
                    self.Kp_thruster * err_deg / 45.0,  # 归一化
                    -1.0, 1.0
                )
        
        # ===== 7. 车钟控制 =====
        n_cmd = 100
        
        # 小航向误差时可以稍微加速
        if abs(err_deg) < 10:
            n_cmd = 110
        # 大转弯时适度减速（让舵效更好）
        elif abs(err_deg) > 30:
            n_cmd = 90
        
        # ===== 8. 限幅 =====
        delta_cmd = np.clip(delta_deg, -35.0, 35.0)
        n_cmd = np.clip(n_cmd, 50, self.ship.params.max_rpm)
        
        return n_cmd, delta_cmd
    
    def _find_lookahead_point(self, x, y, distance):
        """找到指定前视距离的航点"""
        for i in range(self.current_wp_index, len(self.waypoints)):
            wp = self.waypoints[i]
            dist = np.sqrt((wp[0] - x)**2 + (wp[1] - y)**2)
            if dist >= distance:
                return wp
        return self.waypoints[-1]
    
    @property
    def reached_goal(self) -> bool:
        return self.current_wp_index >= len(self.waypoints)


# ========== 工具函数 ==========

def create_ship(ship_type: ShipType = ShipType.RESEARCH_VESSEL) -> ShipDynamicsModel:
    """
    快速创建船舶模型
    
    Args:
        ship_type: 船型枚举
    
    Returns:
        ShipDynamicsModel 实例
    """
    params = SHIP_PRESETS.get(ship_type, SHIP_PRESETS[ShipType.RESEARCH_VESSEL])
    return ShipDynamicsModel(params)


def create_custom_ship(
    length: float = 80.0,
    beam: float = 18.0,
    draft: float = 6.0,
    displacement: float = 5000.0,
    **kwargs
) -> ShipDynamicsModel:
    """
    创建自定义船舶
    
    Args:
        length: 船长 (m)
        beam: 船宽 (m)
        draft: 吃水 (m)
        displacement: 排水量 (t)
        **kwargs: 其他参数
    
    Returns:
        ShipDynamicsModel 实例
    """
    params = ShipParameters(
        name="Custom Ship",
        ship_type=ShipType.CUSTOM,
        Lpp=length,
        B=beam,
        d=draft,
        displacement=displacement,
        **kwargs
    )
    return ShipDynamicsModel(params)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("船舶动力学模型测试")
    print("=" * 60)
    
    # 创建80米科考船
    ship = create_custom_ship(length=80.0, beam=16.0, draft=5.5, displacement=4500.0)
    
    print(f"\n船舶参数:")
    print(f"  船长: {ship.params.Lpp} m")
    print(f"  船宽: {ship.params.B} m")
    print(f"  排水量: {ship.params.displacement} t")
    print(f"  最小转弯半径: {ship.params.min_turning_radius:.1f} m")
    
    # 设置初始状态
    ship.set_state(x=0, y=0, psi=np.pi/2, u=5.0)
    
    # 设置控制
    ship.set_control(n_cmd=100, delta_cmd=15)  # 15度右满舵
    
    # 仿真10秒
    print(f"\n模拟转向 (舵角=15°):")
    for t in range(100):
        state = ship.step(0.1)
        if t % 20 == 0:
            print(f"  t={t*0.1:.1f}s: 位置=({state[0]:.1f}, {state[1]:.1f}), "
                  f"航向={np.degrees(state[2]):.1f}°, 转弯半径={ship.get_turning_radius():.1f}m")
    
    print("\n✓ 测试完成")
