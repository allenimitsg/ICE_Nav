"""
@File    : ice_physics_simulator_enhanced.py
@Author  : Bin MEI
@Date    : 2025-12-22
@Desc    : Draft version for upcoming paper.
"""



"""
增强版冰区物理仿真系统
========================
集成MMG船舶操纵动力学模型

主要改进：
1. 真实船舶3-DOF操纵模型（舵、桨、船体水动力）
2. 多船型预设支持
3. 公里级尺度优化
4. 实时动力学状态显示
"""

import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
import cv2
import json
import os
import time
import csv
import math
from queue import Queue
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# 导入船舶动力学模型
from ship_dynamics import (
    ShipDynamicsModel, ShipParameters, SHIP_PRESETS, ShipType,
    AutoPilot, create_ship, create_custom_ship
)
from enhanced_config import EnhancedConfig, SimulationScale, IceCondition
from ice_generator import IceGenerator
from visualization_comparison import ComparisonVisualizer, AlgorithmResult


class EnhancedShip:
    """
    增强版船舶类
    
    结合PyMunk碰撞检测和MMG动力学模型
    """
    
    def __init__(self, space: pymunk.Space, config: EnhancedConfig):
        self.config = config
        self.space = space
        
        # ========== 1. 创建MMG动力学模型 ==========
        if hasattr(config, 'USE_REALISTIC_DYNAMICS') and config.USE_REALISTIC_DYNAMICS:
            self._init_realistic_dynamics(config)
        else:
            self._init_simple_dynamics(config)
        
        # ========== 2. 创建PyMunk碰撞形状 ==========
        self._init_pymunk_body(config)
        
        # ========== 3. 自动舵控制器 ==========
        if self.use_mmg:
            self.autopilot = AutoPilot(self.dynamics)
            self.autopilot.Kp_heading = 2.5
            self.autopilot.Kd_heading = 1.8
        
        # ========== 4. 统计数据 ==========
        self.path_index = 0
        self.total_distance = 0.0
        self.collision_count = 0
        self.collision_forces = []
        self.collision_details = []
        self.trajectory = []
        self.ice_resistance_history = []
        self.current_ice_resistance = 0.0
        
        # 碰撞能量统计 (新增)
        self.collision_energy_history = []  # 每次碰撞的能量 (J)
        self.total_collision_energy = 0.0   # 累计碰撞能量 (J)
        
        # 重规划标志
        self.force_replan = False  # 当偏差太大时由autopilot设置
        
        # 动力学记录
        self.rudder_angle_history = []
        self.propeller_rpm_history = []
        self.speed_history = []
        self.heading_history = []
        
        # MMG速度分量记录 (u, v, r)
        self.time_history = []          # 时间戳
        self.surge_velocity_history = []  # u - 纵荡速度 (m/s)
        self.sway_velocity_history = []   # v - 横荡速度 (m/s)
        self.yaw_rate_history = []        # r - 艏摇角速度 (rad/s)
        self.position_x_history = []      # x 位置
        self.position_y_history = []      # y 位置
    
    def _init_realistic_dynamics(self, config):
        """初始化真实MMG动力学"""
        self.use_mmg = True
        
        ship_type_str = getattr(config, 'SHIP_TYPE', 'RESEARCH_VESSEL')
        
        # 检查是否自定义船舶
        if ship_type_str == "CUSTOM" and hasattr(config, '_sim_config'):
            custom = config._sim_config.custom_ship
            self.dynamics = create_custom_ship(
                length=custom['length'],
                beam=custom['beam'],
                draft=custom['draft'],
                displacement=custom['displacement']
            )
        else:
            # 使用预设船型
            try:
                ship_type = ShipType[ship_type_str]
                self.dynamics = create_ship(ship_type)
            except KeyError:
                print(f"⚠️ 未知船型 {ship_type_str}，使用默认科考船")
                self.dynamics = create_ship(ShipType.RESEARCH_VESSEL)
        
        # 初始化状态
        start_x = config.SHIP_START_X * config.WORLD_WIDTH
        start_y = config.SHIP_START_Y * config.WORLD_HEIGHT
        self.dynamics.set_state(
            x=start_x,
            y=start_y,
            psi=float(getattr(config, 'SHIP_START_HEADING', np.pi / 2)),
            u=0.0
        )
        
        print(f"  ✓ MMG动力学模型已初始化")
        print(f"    船长: {self.dynamics.params.Lpp}m")
        print(f"    船宽: {self.dynamics.params.B}m")
        print(f"    最大舵角: {self.dynamics.params.max_rudder_angle}°")
        print(f"    最小转弯半径: {self.dynamics.params.min_turning_radius:.0f}m")
    
    def _init_simple_dynamics(self, config):
        """初始化简化动力学（后备模式）"""
        self.use_mmg = False
        self.dynamics = None
        print("  ⚠️ 使用简化动力学模型")
    
    def _init_pymunk_body(self, config):
        """初始化PyMunk刚体（用于碰撞检测）"""
        width = config.SHIP_WIDTH
        length = config.SHIP_LENGTH
        
        # 船形顶点（本地坐标系：+X为艏向，与heading=0一致）
        # heading=0时船头朝右，heading=90°时船头朝上
        vertices = [
            (length/2, 0),              # 艏部（船头）
            (length/4, width/3),
            (0, width/2),
            (-length/3, width/2.2),
            (-length/2, width/4),
            (-length/2, 0),             # 艉部（船尾）
            (-length/2, -width/4),
            (-length/3, -width/2.2),
            (0, -width/2),
            (length/4, -width/3),
        ]
        
        # 创建刚体（运动学模式 - 位置由MMG模型控制）
        if self.use_mmg:
            # 运动学刚体：位置由外部控制，但仍参与碰撞
            self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        else:
            # 动力学刚体：由pymunk物理引擎控制
            mass = config.SHIP_MASS
            moment = pymunk.moment_for_poly(mass, vertices)
            self.body = pymunk.Body(mass, moment)
        
        # 初始位置
        start_x = config.SHIP_START_X * config.WORLD_WIDTH
        start_y = config.SHIP_START_Y * config.WORLD_HEIGHT
        self.body.position = (start_x, start_y)
        self.body.angle = float(getattr(config, 'SHIP_START_HEADING', np.pi / 2))
        
        # 碰撞形状 - 使用船-冰专用摩擦/弹性参数
        self.shape = pymunk.Poly(self.body, vertices)
        self.shape.friction = getattr(config, 'SHIP_ICE_FRICTION', config.SHIP_FRICTION)
        self.shape.elasticity = getattr(config, 'SHIP_ICE_ELASTICITY', config.SHIP_ELASTICITY)
        self.shape.collision_type = 1
        ship_cat = getattr(config, 'SHIP_COLLISION_CATEGORY', 0b1)
        ice_cat = getattr(config, 'ICE_COLLISION_CATEGORY', 0b10)
        self.shape.filter = pymunk.ShapeFilter(categories=ship_cat, mask=(ship_cat | ice_cat))
        
        self.space.add(self.body, self.shape)
    
    def set_path(self, path: List[Tuple[float, float]]):
        """设置航路点"""
        self.path_index = 0
        if self.use_mmg:
            self.autopilot.set_waypoints(path)
    
    def update(self, dt: float, path: List[Tuple[float, float]], config):
        """
        更新船舶状态
        
        Args:
            dt: 时间步长
            path: 航路点列表
            config: 配置对象
        """
        if self.use_mmg:
            self._update_mmg(dt, path, config)
        else:
            self._update_simple(dt, path, config)
        
        # 记录轨迹（保留完整数据用于导出）
        x, y = self.body.position
        angle = self.body.angle
        self.trajectory.append((x, y, angle, time.time()))
    
    def _update_mmg(self, dt: float, path: List[Tuple[float, float]], config):
        """使用MMG模型更新"""
        # 更新航路点
        if path and len(path) > 0:
            self.autopilot.waypoints = path
        
        # 计算冰阻力
        ice_resistance = self._calculate_ice_resistance(config)
        self.current_ice_resistance = ice_resistance
        self.ice_resistance_history.append(ice_resistance)
        
        # 获取当前纵向速度
        u = self.dynamics.state[3]  # surge velocity
        
        # 冰阻力应该反对运动方向，而不是始终向后
        # 如果船在前进(u>0)，冰阻力向后(-);如果船在后退(u<0)，冰阻力向前(+)
        if abs(u) > 0.1:
            ice_force = -ice_resistance * np.sign(u)
        else:
            # 静止时，冰阻力为零（需要推力才能启动）
            ice_force = 0.0
        
        self.dynamics.apply_external_force(ice_force, 0, 0)
        
        # DWA局部规划器：传入冰块位置
        ice_positions = getattr(self, '_nearby_ice', [])
        n_cmd, delta_cmd = self.autopilot.compute_control(ice_positions)
        self.dynamics.set_control(n_cmd, delta_cmd)
        
        # 传递侧推器指令（破冰船专用）
        if hasattr(self.autopilot, 'bow_thruster_cmd'):
            self.dynamics.bow_thruster_cmd = self.autopilot.bow_thruster_cmd
        
        # 调试输出已禁用以提高性能
        
        # 积分动力学
        prev_pos = pymunk.Vec2d(self.body.position.x, self.body.position.y)
        prev_angle = float(self.body.angle)
        state = self.dynamics.step(dt)

        new_pos = pymunk.Vec2d(float(state[0]), float(state[1]))
        new_angle = float(state[2])
        dv = (new_pos - prev_pos) * (1.0 / max(1e-6, dt))
        dpsi = new_angle - prev_angle
        dpsi = float(np.arctan2(np.sin(dpsi), np.cos(dpsi)))
        omega = dpsi / max(1e-6, dt)

        self.body.velocity = (dv.x, dv.y)
        self.body.angular_velocity = omega
        
        # 更新路径索引
        if self.autopilot.current_wp_index != self.path_index:
            self.path_index = self.autopilot.current_wp_index
        
        # 检查是否需要重规划（大偏差时自动触发）
        if hasattr(self.autopilot, 'need_replan') and self.autopilot.need_replan:
            self.force_replan = True  # 设置标志供主循环检查
        
        # 记录动力学状态（保留完整数据用于导出）
        self.rudder_angle_history.append(np.degrees(self.dynamics.delta))
        self.propeller_rpm_history.append(self.dynamics.n)
        self.speed_history.append(self.dynamics.speed)
        self.heading_history.append(np.degrees(state[2]))
        
        # 记录MMG速度分量 (u, v, r) 和位置
        self.surge_velocity_history.append(float(state[3]))   # u - 纵荡速度
        self.sway_velocity_history.append(float(state[4]))    # v - 横荡速度
        self.yaw_rate_history.append(float(state[5]))         # r - 艏摇角速度
        self.position_x_history.append(float(state[0]))       # x 位置
        self.position_y_history.append(float(state[1]))       # y 位置
    
    def _update_simple(self, dt: float, path: List[Tuple[float, float]], config):
        """使用简化模型更新（兼容原版）"""
        # 这里保持原有的PID控制逻辑
        if not path or self.path_index >= len(path):
            return
        
        current_pos = self.body.position
        target = path[self.path_index]
        target_pos = pymunk.Vec2d(target[0], target[1])
        
        distance = current_pos.get_distance(target_pos)
        if distance < config.PATH_FOLLOW_DISTANCE:
            self.path_index += 1
            if self.path_index >= len(path):
                return
            target = path[self.path_index]
            target_pos = pymunk.Vec2d(target[0], target[1])
        
        # 简化PID控制
        error = target_pos - current_pos
        force = error * config.PID_KP - self.body.velocity * config.PID_KD
        
        max_force = config.SHIP_MAX_FORCE
        if force.length > max_force:
            force = force.normalized() * max_force
        
        self.body.apply_force_at_world_point(force, self.body.position)
        
        # 航向控制
        if error.length > 0.1:
            desired_angle = np.arctan2(error.y, error.x)
            angle_error = desired_angle - self.body.angle
            while angle_error > np.pi: angle_error -= 2*np.pi
            while angle_error < -np.pi: angle_error += 2*np.pi
            self.body.torque = angle_error * config.SHIP_MAX_FORCE * 5.0
        
        # 速度限制
        if self.body.velocity.length > config.SHIP_MAX_SPEED:
            self.body.velocity = self.body.velocity.normalized() * config.SHIP_MAX_SPEED
        
        # 冰阻力
        ice_resistance = self._calculate_ice_resistance(config)
        self.current_ice_resistance = ice_resistance
        self.ice_resistance_history.append(ice_resistance)
    
    def _calculate_ice_resistance(self, config) -> float:
        """计算冰阻力（确保有值）"""
        resistance = 0.0
        
        if self.use_mmg:
            # 使用MMG模型的冰阻力计算
            speed = self.dynamics.speed
            resistance = self.dynamics.compute_ice_resistance(
                ice_concentration=config.ICE_CONCENTRATION,
                ice_thickness=config.ICE_THICKNESS,
                ship_speed=speed
            )
        
        # 如果MMG返回0或未使用MMG，使用简化公式
        if resistance < 1.0:
            if self.use_mmg:
                speed = self.dynamics.speed
            else:
                speed = self.body.velocity.length
            
            if speed > 0.1:
                L = config.SHIP_LENGTH
                B = config.SHIP_WIDTH
                h_i = getattr(config, 'ICE_THICKNESS', 0.8)
                C_i = getattr(config, 'ICE_CONCENTRATION', 0.6)
                
                rho_i = getattr(config, 'ICE_DENSITY', 917.0)
                sigma_c = 2.0e6  # 冰压碎强度
                
                # Jeong简化公式
                R_c = 0.5 * sigma_c * h_i * B
                R_i = R_c * C_i * (1 + 0.1 * speed)
                
                # 碰撞冰块时额外阻力
                if self.collision_count > 0:
                    R_i *= (1 + 0.05 * min(self.collision_count, 20))
                
                resistance = R_i
        
        return resistance
    
    def apply_manual_control(self, keys, config, virtual_keys=None):
        """手动控制"""
        if self.use_mmg:
            # MMG模式：控制舵角和转速
            delta_cmd = 0
            n_cmd = 0
            
            # 前进/后退 -> 转速
            if keys[pygame.K_w] or keys[pygame.K_UP] or (virtual_keys and virtual_keys.get('w')):
                n_cmd = self.dynamics.params.max_rpm * 0.8
            if keys[pygame.K_s] or keys[pygame.K_DOWN] or (virtual_keys and virtual_keys.get('s')):
                n_cmd = self.dynamics.params.min_rpm
            
            # 左转/右转 -> 舵角
            if keys[pygame.K_a] or keys[pygame.K_LEFT] or (virtual_keys and virtual_keys.get('a')):
                delta_cmd = -self.dynamics.params.max_rudder_angle
            if keys[pygame.K_d] or keys[pygame.K_RIGHT] or (virtual_keys and virtual_keys.get('d')):
                delta_cmd = self.dynamics.params.max_rudder_angle
            
            self.dynamics.set_control(n_cmd, delta_cmd)
        else:
            # 简化模式：直接施力
            force_magnitude = config.MANUAL_FORCE
            torque = config.MANUAL_TORQUE
            
            if keys[pygame.K_w] or keys[pygame.K_UP] or (virtual_keys and virtual_keys.get('w')):
                angle = self.body.angle
                force = (force_magnitude * np.cos(angle), force_magnitude * np.sin(angle))
                self.body.apply_force_at_world_point(force, self.body.position)
            
            if keys[pygame.K_s] or keys[pygame.K_DOWN] or (virtual_keys and virtual_keys.get('s')):
                angle = self.body.angle
                force = (-force_magnitude * np.cos(angle), -force_magnitude * np.sin(angle))
                self.body.apply_force_at_world_point(force, self.body.position)
            
            if keys[pygame.K_a] or keys[pygame.K_LEFT] or (virtual_keys and virtual_keys.get('a')):
                self.body.apply_force_at_local_point((-torque, 0), (0, 5))
            
            if keys[pygame.K_d] or keys[pygame.K_RIGHT] or (virtual_keys and virtual_keys.get('d')):
                self.body.apply_force_at_local_point((torque, 0), (0, 5))
    
    def reset(self, config):
        """重置船舶状态"""
        start_x = config.SHIP_START_X * config.WORLD_WIDTH
        start_y = config.SHIP_START_Y * config.WORLD_HEIGHT
        
        self.body.position = (start_x, start_y)
        self.body.angle = float(getattr(config, 'SHIP_START_HEADING', np.pi / 2))
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0
        
        if self.use_mmg:
            self.dynamics.set_state(x=start_x, y=start_y, psi=float(getattr(config, 'SHIP_START_HEADING', np.pi/2)), u=0, v=0, r=0)
            self.autopilot.current_wp_index = 0
        
        self.path_index = 0
        self.collision_count = 0
        self.collision_forces = []
        self.collision_details = []
        self.trajectory = []
        self.ice_resistance_history = []
        self.collision_energy_history = []  # 重置碰撞能量
        self.total_collision_energy = 0.0
        self.rudder_angle_history = []
        self.propeller_rpm_history = []
        self.speed_history = []
        self.heading_history = []
        
        # 重置MMG速度分量记录
        self.time_history = []
        self.surge_velocity_history = []
        self.sway_velocity_history = []
        self.yaw_rate_history = []
        self.position_x_history = []
        self.position_y_history = []
    
    @property
    def dynamics_info(self) -> Dict:
        """获取动力学状态信息"""
        if self.use_mmg:
            return {
                'speed': self.dynamics.speed,
                'heading': np.degrees(self.dynamics.heading),
                'rudder_angle': np.degrees(self.dynamics.delta),
                'propeller_rpm': self.dynamics.n,
                'turning_radius': self.dynamics.get_turning_radius(),
                'surge': self.dynamics.state[3],
                'sway': self.dynamics.state[4],
                'yaw_rate': np.degrees(self.dynamics.state[5]),
            }
        else:
            return {
                'speed': self.body.velocity.length,
                'heading': np.degrees(self.body.angle),
                'rudder_angle': 0,
                'propeller_rpm': 0,
                'turning_radius': float('inf'),
            }


class IceBlock:
    """冰块实体（与原版相同）"""
    
    def __init__(self, space: pymunk.Space, ice_data: Dict, config):
        self.space = space
        self.ice_type = ice_data['type']
        self.size = ice_data['size']

        self.source = ice_data.get('source', None)

        area = float(ice_data.get('area', self.size ** 2 * 0.8))
        density = float(config.ICE_DENSITY_PER_AREA.get(self.ice_type, 200.0))
        mass = float(ice_data.get('mass', area * density))

        # 冰块尺寸阈值
        large_th = float(getattr(config, 'LARGE_ICE_THRESHOLD_SIZE', config.SHIP_LENGTH * 1.5))
        heavy_th = float(getattr(config, 'HEAVY_ICE_THRESHOLD_SIZE', config.SHIP_LENGTH * 2.5))
        immovable_th = float(getattr(config, 'IMMOVABLE_ICE_THRESHOLD_SIZE', config.SHIP_LENGTH * 5.0))
        # 质量倍增器
        medium_mul = float(getattr(config, 'MEDIUM_ICE_MASS_MULTIPLIER', 5.0))
        large_mul = float(getattr(config, 'LARGE_ICE_MASS_MULTIPLIER', 30.0))
        heavy_mul = float(getattr(config, 'HEAVY_ICE_MASS_MULTIPLIER', 100.0))

        if self.source == 'InjectedCenter':
            # 中央障碍物需要“可推动但很费力”，避免变成不可通行的墙
            large_mul = min(float(large_mul), 8.0)
            heavy_mul = min(float(heavy_mul), 20.0)

        if self.source in ('InjectedMidBand', 'InjectedHuge', 'InjectedCurvedBank', 'InjectedCenter'):
            use_full = bool(getattr(config, 'INJECTED_OBSTACLE_USE_MASS_MULTIPLIERS', False))
            if not use_full:
                try:
                    s = float(getattr(config, 'INJECTED_OBSTACLE_MASS_MULTIPLIER_SCALE', 0.0) or 0.0)
                except Exception:
                    s = 0.0
                s = max(0.0, min(1.0, s))

                def _blend(m):
                    m = float(m)
                    return 1.0 + s * (m - 1.0)

                medium_mul = _blend(medium_mul)
                large_mul = _blend(large_mul)
                heavy_mul = _blend(heavy_mul)

        # 按尺寸分级增加质量
        if self.size >= heavy_th:
            mass *= heavy_mul
        elif self.size >= large_th:
            mass *= large_mul
        elif self.size >= config.SHIP_LENGTH * 0.8:  # 中型冰块
            mass *= medium_mul

        immovable = (self.ice_type == 'Ice Bank') or (self.size >= immovable_th)

        vertices = ice_data.get('vertices', self._generate_vertices())
        
        # ★ 简化处理：尝试凸包，失败则用圆形 ★
        try:
            from pymunk import Vec2d
            if len(vertices) >= 3:
                vec_verts = [Vec2d(v[0], v[1]) for v in vertices]
                hull = pymunk.util.convex_hull(vec_verts)
                if len(hull) >= 3:
                    vertices = [(v.x, v.y) for v in hull]
        except:
            pass
        
        if len(vertices) < 3:
            vertices = self._generate_vertices()

        if immovable:
            self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        else:
            moment = pymunk.moment_for_poly(mass, vertices)
            self.body = pymunk.Body(mass, moment)
        self.body.position = ice_data['center']
        
        # 创建多边形形状
        self.shape = pymunk.Poly(self.body, vertices)
        if immovable:
            self.shape.friction = min(1.0, config.ICE_FRICTION * 2.0)
            self.shape.elasticity = max(0.01, config.ICE_ELASTICITY * 0.5)
        else:
            self.shape.friction = config.ICE_FRICTION
            self.shape.elasticity = config.ICE_ELASTICITY
        self.shape.collision_type = 2
        try:
            if self.source == 'InjectedCenter':
                setattr(self.shape, 'is_center_obstacle', True)
        except Exception:
            pass
        ship_cat = getattr(config, 'SHIP_COLLISION_CATEGORY', 0b1)
        ice_cat = getattr(config, 'ICE_COLLISION_CATEGORY', 0b10)
        enable_ice_ice = getattr(config, 'ENABLE_ICE_ICE_COLLISION', False) or getattr(config, 'ENABLE_LOCAL_ICE_ICE_COLLISION', False)
        mask = (ship_cat | ice_cat) if enable_ice_ice else ship_cat
        self.shape.filter = pymunk.ShapeFilter(categories=ice_cat, mask=mask)
        
        space.add(self.body, self.shape)
    
    def _generate_vertices(self):
        """生成随机多边形顶点"""
        n_vertices = np.random.randint(5, 9)
        angles = np.sort(np.random.uniform(0, 2*np.pi, n_vertices))
        radii = self.size/2 * np.random.uniform(0.7, 1.0, n_vertices)
        vertices = [(r*np.cos(a), r*np.sin(a)) for r, a in zip(radii, angles)]
        return vertices


class EnhancedPhysicsSimulator:
    """
    增强版物理仿真器
    
    集成MMG船舶动力学，支持多船型、多尺度
    """
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        
        print("\n" + "="*60)
        print("🚢 增强版冰区物理仿真系统")
        print("="*60)
        print(f"  尺度: {config.WORLD_WIDTH}m × {config.WORLD_HEIGHT}m")
        print(f"  船型: {getattr(config, 'SHIP_TYPE', 'RESEARCH_VESSEL')}")
        print(f"  动力学: {'MMG真实模型' if getattr(config, 'USE_REALISTIC_DYNAMICS', True) else '简化模型'}")
        
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        pygame.display.set_caption("碎冰区船舶航行动态仿真系统 - V1.0")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_cn = None
        try:
            self.font_cn = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 16)
        except:
            self.font_cn = self.font
        
        # 物理空间
        self.space = pymunk.Space()
        self.space.gravity = config.GRAVITY
        self.space.damping = config.DAMPING
        if getattr(config, 'ENABLE_DISTANT_ICE_SLEEP', False):
            self.space.sleep_time_threshold = getattr(config, 'SPACE_SLEEP_TIME_THRESHOLD', 0.7)
            self.space.idle_speed_threshold = getattr(config, 'SPACE_IDLE_SPEED_THRESHOLD', 0.12)
        
        # 绘制选项
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        # 生成冰场（使用快速模式）或加载SAM真实冰场
        self.generator = IceGenerator(config)
        
        # 检查是否使用SAM真实冰场 (2025-12-17新增)
        use_sam = getattr(config, 'USE_SAM_ICE_FIELD', False)
        sam_path = getattr(config, 'SAM_JSON_PATH', None)
        
        if use_sam and sam_path:
            print("\n🧊 正在加载SAM真实冰场...")
            max_blocks = getattr(config, 'SAM_MAX_ICE_BLOCKS', None)
            load_mode = getattr(config, 'SAM_LOAD_MODE', 'tiles')
            folder_path = getattr(config, 'SAM_FOLDER_PATH', '')
            
            if load_mode == 'tiles' and folder_path:
                # 直接从tile masks加载（推荐，避免合并脚本的冰块连接问题）
                self.ice_blocks_data = self.generator.load_from_tile_masks(
                    folder_path=folder_path,
                    meters_per_pixel=0.89,
                    max_ice_blocks=max_blocks
                )
            else:
                # 从合并后的JSON加载
                self.ice_blocks_data = self.generator.load_from_sam_json(
                    json_path=sam_path,
                    scale_to_world=False,
                    max_ice_blocks=max_blocks
                )
        else:
            print("\n🧊 正在生成随机冰场...")
            fast_mode = getattr(config, 'FAST_ICE_GENERATION', True)
            self.ice_blocks_data = self.generator.generate_ice_field(seed=42, fast_mode=fast_mode)

        self._ice_block_fast = []
        try:
            for ice in self.ice_blocks_data:
                cx, cy = ice.get('center', (None, None))
                if cx is None or cy is None:
                    continue
                r = ice.get('radius', None)
                if r is None:
                    verts = ice.get('vertices', None)
                    if verts:
                        try:
                            r = max(float(np.hypot(vx, vy)) for vx, vy in verts)
                        except Exception:
                            r = None
                if r is None:
                    continue
                rr = float(r) * 0.85
                self._ice_block_fast.append((float(cx), float(cy), rr * rr))
        except Exception:
            self._ice_block_fast = []

        self._adjust_start_goal_if_blocked()
        
        # 生成路径
        print("🗺️  正在生成路径...")
        raw_path = self._generate_path(config.PATH_TYPE)
        self.path = self._post_process_path(raw_path, smooth=True)
        print(f"   路径点数: {len(self.path)}")
        
        # 创建船舶（使用增强版）
        print("🚢 创建船舶...")
        self.ship = EnhancedShip(self.space, config)
        self.ship.set_path(self.path)
        
        # 创建冰块
        print("❄️  创建冰块...")
        self.ice_blocks = []
        for ice_data in self.ice_blocks_data:
            ice_block = IceBlock(self.space, ice_data, config)
            self.ice_blocks.append(ice_block)

        self._center_obstacle_shapes = []
        try:
            for ib in self.ice_blocks:
                try:
                    shp = getattr(ib, 'shape', None)
                    if shp is not None and bool(getattr(shp, 'is_center_obstacle', False)):
                        self._center_obstacle_shapes.append(shp)
                except Exception:
                    continue
        except Exception:
            self._center_obstacle_shapes = []

        self._mid_route_barrier = None
        self._mid_route_barrier_data = None
        self._mid_route_barrier_collision_count = 0
        self._task_failed = False
        self._failure_reason = None
        self.barrier_button_rect = None
        if bool(getattr(self.config, 'ENABLE_MID_ROUTE_ICE_BARRIER', False)):
            try:
                self._enable_mid_route_barrier(True)
            except Exception:
                pass
        
        # 注：不再分离重叠冰块，保持SAM真实冰场的原始位置
        # self._separate_overlapping_ice()
        
        # 保存冰块初始状态
        self.ice_blocks_initial_state = []
        for ice_block in self.ice_blocks:
            self.ice_blocks_initial_state.append({
                'position': tuple(ice_block.body.position),
                'velocity': tuple(ice_block.body.velocity),
                'angle': ice_block.body.angle,
                'angular_velocity': ice_block.body.angular_velocity
            })
        
        # 设置碰撞处理
        self._setup_collision_handlers()

        self._pending_ship_ice_contacts = set()
        self._ship_ice_last_processed_time = {}
        
        # 算法对比 - 默认仅对比新算法 vs A* vs 直线
        self.algorithm_results = {}
        self._planner_status_by_algorithm = {}
        self.algorithm_sequence = ["Ice-Theta*", "A*", "Straight"]
        try:
            self.current_algorithm_index = self.algorithm_sequence.index(self.config.PATH_TYPE)
        except ValueError:
            self.current_algorithm_index = 0
        self.goal_reached = False
        self.journey_start_time = 0.0
        
        # 控制模式
        self.control_mode = "auto"
        
        # 虚拟按键状态（用于点击控制）
        self.virtual_keys = {'w': False, 's': False, 'a': False, 'd': False}
        self.button_rects = {}       # 方向键按钮
        self.path_button_rects = {}  # 算法切换按钮
        self.mode_button_rect = None # 模式切换按钮
        
        # 自动运行模式（默认开启，依次运行3个算法）
        self.auto_run_algorithms = True
        
        # ========== 摄像机系统（缩放和平移）==========
        self.camera_zoom = 1.0           # 缩放倍数 (1.0=全局视图)
        self.camera_center = None        # 摄像机中心 (None=自动跟随船)
        self.camera_follow_ship = False  # 是否跟随船舶
        self.zoom_levels = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # 预设缩放级别
        self.current_zoom_index = 2      # 当前缩放级别索引 (1.0)
        self.zoom_button_rects = {}      # 缩放按钮
        self.locate_button_rect = None   # 定位按钮
        
        # 仿真状态
        self.running = True
        self.paused = False
        self.simulation_time = 0.0
        self.last_replan_time = 0.0
        self._replan_count = 0
        
        # 动态重规划配置
        self.enable_dynamic_replan = True
        self.replan_interval = 30.0        # 每30秒（大幅减少频率避免卡顿）
        
        # ========== 性能优化配置 ==========
        ice_count = len(self.ice_blocks)
        self.performance_mode = ice_count > 200  # 200块以上开启
        self.ultra_performance = ice_count > 500  # 500块以上超级优化
        self.frame_skip = 0
        self.max_frame_skip = 3 if self.ultra_performance else (2 if self.performance_mode else 0)
        self.spatial_grid = {}
        self.grid_cell_size = 100
        self._build_spatial_grid()
        
        # 动态调整FPS
        if self.ultra_performance:
            self.config.FPS = 30  # 降到30帧
            self.max_frame_skip = 3
            print(f"⚡⚡ 超级性能模式 (冰块: {ice_count}, FPS: 30, 远处冰块冻结)")
        elif self.performance_mode:
            self.config.FPS = 45  # 降到45帧
            self.max_frame_skip = 2
            print(f"⚡ 性能模式已开启 (冰块: {ice_count}, FPS: 45, 远处冰块冻结)")
        
        # ===== 后台规划器（不阻塞主循环）=====
        self._path_queue = Queue(maxsize=1)
        self._planning_thread = None
        self._planning_in_progress = False
        self._pending_path = None          # 待应用的新路径
        
        # 仿真速度控制
        self.speed_multiplier = 1          # 倍速 (1, 2, 4, 8)
        self.speed_options = [1, 2, 4, 8]
        self.speed_button_rects = {}       # 倍速按钮
        
        # 输出目录
        self.output_dir = config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 视频录制
        self.recording = getattr(config, 'RECORD_VIDEO', True)
        self._video_writer = None
        self._video_out_path = None
        self._video_record_every_n = 1
        self._video_frame_counter = 0
        try:
            target_fps = int(getattr(self.config, 'VIDEO_FPS', 30))
        except Exception:
            target_fps = 30
        target_fps = max(1, target_fps)
        self._video_record_every_n = max(1, int(round(float(self.config.FPS) / float(target_fps))))

        self._local_ice_set = set()
        self._local_mask_update_counter = 0
        self._ice_sleep_scan_index = 0
        
        print("\n✓ 初始化完成！")
        print("  按 M 键切换手动/自动模式")
        print("  按 1/2/3 键切换算法")
        print("  按 +/- 或滚轮 缩放视图")
        print("  按 F 定位到船舶")
        print("  按 ESC 退出\n")
    
    def _generate_path(self, path_type: str) -> List[Tuple[float, float]]:
        """生成路径"""
        if path_type == "Straight":
            path = self.generator.generate_straight_path()
        elif path_type == "A*":
            path = self.generator.generate_a_star_path(self.ice_blocks_data)
        elif path_type == "Ice-Theta*":
            path = self.generator.generate_ice_theta_star_path(self.ice_blocks_data)
        elif path_type == "Hybrid A*":
            path = self.generator.generate_hybrid_a_star_path(self.ice_blocks_data)
        elif path_type == "RRT*":
            return self.generator.generate_rrt_star_path(self.ice_blocks_data)
        elif path_type == "DWA":
            return self.generator.generate_dwa_path(self.ice_blocks_data)
        elif path_type == "APF":
            return self.generator.generate_apf_path(self.ice_blocks_data)
        else:
            path = self.generator.generate_a_star_path(self.ice_blocks_data)

        try:
            status = getattr(self.generator, 'last_planner_status', None)
            if status is None:
                status = getattr(self.config, 'LAST_PLANNER_STATUS', None)
            self._planner_status_by_algorithm[path_type] = status
        except Exception:
            pass

        return path

    def _path_step_size(self) -> float:
        L = float(getattr(self.config, 'SHIP_LENGTH', 80.0))
        step_size = max(8.0, L * 0.25)
        try:
            step_cfg = float(getattr(self.config, 'PATH_RESAMPLE_STEP_M', step_size))
            if step_cfg > 0:
                step_size = step_cfg
        except Exception:
            pass
        return float(step_size)

    def _post_process_path(self, raw_path: List[Tuple[float, float]], smooth: bool = True) -> List[Tuple[float, float]]:
        if not raw_path or len(raw_path) < 2:
            return raw_path or []
        step_size = self._path_step_size()
        try:
            algo = str(getattr(self.config, 'PATH_TYPE', '') or '')
            if algo == 'Ice-Theta*':
                smooth = bool(getattr(self.config, 'ICE_THETA_POST_SMOOTH', False))
            elif algo == 'A*':
                smooth = bool(getattr(self.config, 'ASTAR_POST_SMOOTH', True)) and bool(smooth)
            return self.generator.smooth_path(raw_path, step_size=step_size, smooth=bool(smooth))
        except Exception:
            return raw_path
    
    def _build_spatial_grid(self):
        """构建空间哈希网格（加速附近冰块查找）"""
        self.spatial_grid.clear()
        for ice_block in self.ice_blocks:
            pos = ice_block.body.position
            # 跳过无效位置（NaN或超出边界）
            if np.isnan(pos.x) or np.isnan(pos.y):
                continue
            if pos.x < 0 or pos.y < 0 or pos.x > 10000 or pos.y > 10000:
                continue
            cell_x = int(pos.x // self.grid_cell_size)
            cell_y = int(pos.y // self.grid_cell_size)
            key = (cell_x, cell_y)
            if key not in self.spatial_grid:
                self.spatial_grid[key] = []
            self.spatial_grid[key].append(ice_block)
    
    def _update_spatial_grid(self):
        """更新空间网格（冰块移动后调用）"""
        # 每隔一定帧数更新一次，避免每帧都重建
        if not hasattr(self, '_grid_update_counter'):
            self._grid_update_counter = 0
        self._grid_update_counter += 1
        if self._grid_update_counter % 120 == 0:  # 每120帧更新一次（约2秒）
            self._build_spatial_grid()
    
    def _get_nearby_ice_fast(self, x, y, radius):
        """快速获取附近冰块（使用空间哈希）"""
        nearby = []
        cell_radius = int(radius / self.grid_cell_size) + 1
        center_cell_x = int(x // self.grid_cell_size)
        center_cell_y = int(y // self.grid_cell_size)
        
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                key = (center_cell_x + dx, center_cell_y + dy)
                if key in self.spatial_grid:
                    for ice in self.spatial_grid[key]:
                        pos = ice.body.position
                        dist = np.sqrt((pos.x - x)**2 + (pos.y - y)**2)
                        if dist < radius:
                            nearby.append(ice)
        return nearby
    
    def _freeze_distant_ice(self):
        """冻结远处冰块（减少物理计算）- 使用sleep代替body_type切换"""
        # 每30帧更新一次
        if not hasattr(self, '_freeze_counter'):
            self._freeze_counter = 0
        self._freeze_counter += 1
        if self._freeze_counter % 30 != 0:
            return
        
        ship_pos = self.ship.body.position
        if np.isnan(ship_pos.x) or np.isnan(ship_pos.y):
            return
        
        # 根据场景大小调整冻结距离
        world_diag = np.sqrt(self.config.WORLD_WIDTH**2 + self.config.WORLD_HEIGHT**2)
        freeze_dist = max(400, world_diag * 0.4)  # 冻结距离
        wake_dist = freeze_dist * 0.6  # 激活距离
        
        for ice_block in self.ice_blocks:
            try:
                pos = ice_block.body.position
                if np.isnan(pos.x) or np.isnan(pos.y):
                    continue
                
                dist = np.sqrt((pos.x - ship_pos.x)**2 + (pos.y - ship_pos.y)**2)
                
                if dist > freeze_dist:
                    # 远处：让冰块静止（设置速度为0）
                    ice_block.body.velocity = (0, 0)
                    ice_block.body.angular_velocity = 0
            except:
                pass
    
    def _separate_overlapping_ice(self):
        """
        分离重叠的冰块，防止初始化时冰块互相穿透
        使用简单的推开算法：检测重叠并沿连线方向分离
        """
        n = len(self.ice_blocks)
        if n < 2:
            return
        
        print("   检测并分离重叠冰块...")
        separation_count = 0
        max_iterations = 5  # 最多迭代5次
        
        for iteration in range(max_iterations):
            moved = False
            for i in range(n):
                ice_a = self.ice_blocks[i]
                pos_a = ice_a.body.position
                size_a = getattr(ice_a, 'size', 20)
                
                for j in range(i + 1, n):
                    ice_b = self.ice_blocks[j]
                    pos_b = ice_b.body.position
                    size_b = getattr(ice_b, 'size', 20)
                    
                    # 计算距离
                    dx = pos_b.x - pos_a.x
                    dy = pos_b.y - pos_a.y
                    dist = np.sqrt(dx * dx + dy * dy)
                    
                    # 最小安全距离（两冰块半径之和 + 间隙）
                    min_dist = (size_a + size_b) * 0.5 + 2.0
                    
                    if dist < min_dist and dist > 0.01:
                        # 需要分离
                        overlap = min_dist - dist
                        # 归一化方向向量
                        nx = dx / dist
                        ny = dy / dist
                        # 各推开一半距离
                        push = overlap * 0.55
                        
                        # 只移动非静态冰块
                        if ice_a.body.body_type != pymunk.Body.STATIC:
                            ice_a.body.position = (pos_a.x - nx * push, pos_a.y - ny * push)
                            moved = True
                        if ice_b.body.body_type != pymunk.Body.STATIC:
                            ice_b.body.position = (pos_b.x + nx * push, pos_b.y + ny * push)
                            moved = True
                        
                        separation_count += 1
            
            if not moved:
                break
        
        if separation_count > 0:
            print(f"   ✓ 分离了 {separation_count} 对重叠冰块")
    
    def _setup_collision_handlers(self):
        """设置碰撞处理"""
        handler = self.space.add_collision_handler(1, 2)
        handler.begin = self._on_ship_ice_begin
        handler.post_solve = self._on_ship_ice_collision

        if getattr(self.config, 'ENABLE_ICE_ICE_COLLISION', False) or getattr(self.config, 'ENABLE_LOCAL_ICE_ICE_COLLISION', False):
            handler2 = self.space.add_collision_handler(2, 2)
            handler2.begin = self._on_ice_ice_collision

    def _on_ship_ice_begin(self, arbiter, space, data):
        shapes = arbiter.shapes
        ice_shape = shapes[1] if shapes[0].collision_type == 1 else shapes[0]
        self._pending_ship_ice_contacts.add(id(ice_shape))
        return True
    
    def _on_ship_ice_collision(self, arbiter, space, data):
        """船-冰碰撞回调"""
        shapes = arbiter.shapes
        ice_shape = shapes[1] if shapes[0].collision_type == 1 else shapes[0]

        now_t = float(getattr(self, 'simulation_time', 0.0))
        cooldown_s = float(getattr(self.config, 'SHIP_ICE_COLLISION_COOLDOWN_S', 0.2) or 0.2)
        if cooldown_s < 0:
            cooldown_s = 0.0
        ice_id = id(ice_shape)
        last_t = float(getattr(self, '_ship_ice_last_processed_time', {}).get(ice_id, -1e9))
        if (now_t - last_t) < cooldown_s:
            return True
        self._ship_ice_last_processed_time[ice_id] = now_t
        self._pending_ship_ice_contacts.discard(ice_id)

        self.ship.collision_count += 1

        if bool(getattr(ice_shape, 'is_mid_route_barrier', False)):
            self._mid_route_barrier_collision_count += 1
            limit_n = int(getattr(self.config, 'MID_ROUTE_BARRIER_COLLISION_FAIL_COUNT', 0) or 0)
            if limit_n > 0 and self._mid_route_barrier_collision_count >= limit_n:
                self._task_failed = True
                self._failure_reason = 'mid_route_barrier_collision_limit'
                self.running = False
        
        impulse_vec = arbiter.total_impulse
        impulse = impulse_vec.length
        self.ship.collision_forces.append(impulse)
        
        # ========== 计算碰撞能量 (AUTO-IceNav方法) ==========
        # 获取冰块质量
        ice_mass = ice_shape.body.mass if ice_shape.body.mass > 0 else 1000.0  # 默认1吨
        
        # 船舶质量
        ship_mass = self.config.SHIP_MASS
        
        # 相对速度 (碰撞前)
        ship_vel = self.ship.body.velocity
        ice_vel = ice_shape.body.velocity
        rel_vel = np.sqrt((ship_vel.x - ice_vel.x)**2 + (ship_vel.y - ice_vel.y)**2)
        
        # 碰撞能量公式 (基于AUTO-IceNav):
        # E = (V² × m_ice × m_ship × (m_ice + 2×m_ship)) / (2 × (m_ship + m_ice)²)
        if rel_vel > 0.1:
            collision_energy = (rel_vel**2 * ice_mass * ship_mass * (ice_mass + 2*ship_mass)) / \
                              (2 * (ship_mass + ice_mass)**2)
        else:
            # 备用方法：从冲量计算 E = J²/(2m_reduced)
            m_reduced = (ship_mass * ice_mass) / (ship_mass + ice_mass)
            collision_energy = (impulse**2) / (2 * m_reduced) if m_reduced > 0 else 0
        
        self.ship.collision_energy_history.append(collision_energy)
        self.ship.total_collision_energy += collision_energy
        
        # 记录碰撞详情
        for contact in arbiter.contact_point_set.points:
            world_point = contact.point_a
            
            # 转换到船体坐标系
            ship_pos = self.ship.body.position
            ship_angle = self.ship.body.angle
            
            dx = world_point.x - ship_pos.x
            dy = world_point.y - ship_pos.y
            
            local_x = dx * np.cos(-ship_angle) - dy * np.sin(-ship_angle)
            local_y = dx * np.sin(-ship_angle) + dy * np.cos(-ship_angle)
            
            # 冲量在船体坐标系的分量
            impulse_local_x = impulse_vec.x * np.cos(-ship_angle) - impulse_vec.y * np.sin(-ship_angle)
            impulse_local_y = impulse_vec.x * np.sin(-ship_angle) + impulse_vec.y * np.cos(-ship_angle)
            
            self.ship.collision_details.append({
                'position': tuple(ship_pos),
                'contact_point': (world_point.x, world_point.y),
                'local_x': local_x,
                'local_y': local_y,
                'force': impulse,
                'impulse_local_x': impulse_local_x,
                'impulse_local_y': impulse_local_y,
                'collision_energy': collision_energy,  # 新增
                'ice_mass': ice_mass,                  # 新增
                'relative_velocity': rel_vel,          # 新增
                'time': self.simulation_time
            })
        
        # 将碰撞力传递给MMG模型
        if self.ship.use_mmg:
            # 转换到船体坐标系
            angle = self.ship.body.angle
            Fx = -impulse_vec.x * np.cos(angle) - impulse_vec.y * np.sin(angle)
            Fy = impulse_vec.x * np.sin(angle) - impulse_vec.y * np.cos(angle)
            self.ship.dynamics.apply_external_force(Fx * 0.1, Fy * 0.1, 0)
        
        # ========== 碰撞减速效果 (增强版) ==========
        ship_mass = self.config.SHIP_MASS
        mass_ratio = ice_mass / ship_mass
        
        # 根据质量比设置减速系数
        if mass_ratio < 0.01:
            speed_reduction = 0.08   # 碎冰: 8%减速
        elif mass_ratio < 0.05:
            speed_reduction = 0.20   # 小冰: 20%减速
        elif mass_ratio < 0.2:
            speed_reduction = 0.40   # 中冰: 40%减速
        elif mass_ratio < 1.0:
            speed_reduction = 0.55   # 大冰: 55%减速
        else:
            speed_reduction = 0.70   # 巨冰/冰山: 70%减速

        try:
            scale = float(getattr(self.config, 'SHIP_ICE_COLLISION_SPEED_REDUCTION_SCALE', 1.0) or 1.0)
        except Exception:
            scale = 1.0
        try:
            max_red = float(getattr(self.config, 'SHIP_ICE_COLLISION_SPEED_REDUCTION_MAX', 0.9) or 0.9)
        except Exception:
            max_red = 0.9
        scale = max(0.0, scale)
        max_red = max(0.0, min(0.99, max_red))
        speed_reduction = float(speed_reduction) * scale

        try:
            rel_ref = float(getattr(self.config, 'SHIP_ICE_COLLISION_RELVEL_REF_MPS', 2.5) or 2.5)
        except Exception:
            rel_ref = 2.5
        rel_ref = max(1e-3, rel_ref)
        try:
            rel_factor = float(min(1.0, max(0.0, rel_vel / rel_ref)))
        except Exception:
            rel_factor = 1.0
        speed_reduction *= rel_factor

        if bool(getattr(self.config, 'SHIP_ICE_COLLISION_DIRECTIONAL', True)):
            try:
                mag = float(impulse_vec.length)
                if mag > 1e-6:
                    ux = float(impulse_vec.x) / mag
                    uy = float(impulse_vec.y) / mag
                    ang = float(self.ship.body.angle)
                    fx = float(np.cos(ang))
                    fy = float(np.sin(ang))
                    cos_align = abs(ux * fx + uy * fy)
                    try:
                        p = float(getattr(self.config, 'SHIP_ICE_COLLISION_ALIGN_POWER', 2.0) or 2.0)
                    except Exception:
                        p = 2.0
                    p = max(0.1, p)
                    try:
                        side = float(getattr(self.config, 'SHIP_ICE_COLLISION_SIDE_FACTOR', 0.35) or 0.35)
                    except Exception:
                        side = 0.35
                    side = max(0.0, min(1.0, side))
                    direction_factor = side + (1.0 - side) * (cos_align ** p)
                    speed_reduction *= float(direction_factor)
            except Exception:
                pass

        speed_reduction = max(0.0, min(max_red, float(speed_reduction)))

        # 中央障碍物：只做温和减速，避免"撞上直接停死"
        if bool(getattr(ice_shape, 'is_center_obstacle', False)):
            try:
                c_scale = float(getattr(self.config, 'CENTER_OBSTACLE_SPEED_REDUCTION_SCALE', 0.25) or 0.25)
            except Exception:
                c_scale = 0.25
            try:
                c_max = float(getattr(self.config, 'CENTER_OBSTACLE_SPEED_REDUCTION_MAX', 0.12) or 0.12)
            except Exception:
                c_max = 0.12
            c_scale = max(0.0, c_scale)
            c_max = max(0.0, min(0.5, c_max))
            speed_reduction = min(c_max, float(speed_reduction) * c_scale)
        
        # 应用减速到船舶动力学
        if self.ship.use_mmg and hasattr(self.ship, 'dynamics'):
            if hasattr(self.ship.dynamics, 'state') and len(self.ship.dynamics.state) >= 4:
                try:
                    current_u = float(self.ship.dynamics.state[3])
                    self.ship.dynamics.state[3] = current_u * (1.0 - speed_reduction)
                except Exception:
                    pass
                if len(self.ship.dynamics.state) >= 5:
                    try:
                        current_v = float(self.ship.dynamics.state[4])
                        self.ship.dynamics.state[4] = current_v * (1.0 - speed_reduction)
                    except Exception:
                        pass

        try:
            self.ship.body.velocity = self.ship.body.velocity * (1.0 - speed_reduction)
        except Exception:
            pass
        
        # ========== 限制冰块被撞后的速度 (水阻力效果) ==========
        ice_body = ice_shape.body
        if ice_body.body_type == pymunk.Body.DYNAMIC:
            ship_speed = self.ship.body.velocity.length
            
            if mass_ratio > 1.0:
                max_ice_speed = min(0.2, ship_speed * 0.05)
            elif mass_ratio > 0.5:
                max_ice_speed = min(0.3, ship_speed * 0.1)
            elif mass_ratio > 0.2:
                max_ice_speed = min(0.8, ship_speed * 0.2)
            elif mass_ratio > 0.1:
                max_ice_speed = min(1.5, ship_speed * 0.4)
            elif mass_ratio > 0.01:
                max_ice_speed = min(3.0, ship_speed * 0.7)
            else:
                max_ice_speed = min(5.0, ship_speed * 1.0)
            
            current_speed = ice_body.velocity.length
            if current_speed > max_ice_speed:
                scale = max_ice_speed / current_speed
                ice_body.velocity = ice_body.velocity * scale
            
            if mass_ratio > 0.1:
                max_angular = 0.05 / max(0.1, mass_ratio)
                if abs(ice_body.angular_velocity) > max_angular:
                    ice_body.angular_velocity = np.sign(ice_body.angular_velocity) * max_angular
        
        return True
    
    def _on_ice_ice_collision(self, arbiter, space, data):
        """冰-冰碰撞回调（简化版，避免性能问题）"""
        if getattr(self.config, 'ENABLE_ICE_ICE_COLLISION', False):
            return True

        if not getattr(self.config, 'ENABLE_LOCAL_ICE_ICE_COLLISION', False):
            return False

        try:
            ship_pos = self.ship.body.position
            r = float(getattr(self.config, 'LOCAL_ICE_ICE_RADIUS', 220.0))
            r2 = r * r
            p = arbiter.contact_point_set.points
            if p:
                c = p[0].point_a
                dx = float(c.x) - float(ship_pos.x)
                dy = float(c.y) - float(ship_pos.y)
            else:
                a = arbiter.shapes[0].body.position
                b = arbiter.shapes[1].body.position
                mx = 0.5 * (float(a.x) + float(b.x))
                my = 0.5 * (float(a.y) + float(b.y))
                dx = mx - float(ship_pos.x)
                dy = my - float(ship_pos.y)
            return (dx * dx + dy * dy) <= r2
        except Exception:
            return True
    
    def run(self):
        """主循环（带性能优化）"""
        self.journey_start_time = self.simulation_time
        self._frame_times = []  # FPS计算

        try:
            while self.running:
                frame_start = time.time()

                self._handle_events()

                if not self.paused:
                    # 根据倍速执行多次更新
                    for _ in range(self.speed_multiplier):
                        self._update()
                        # 更新空间索引（低频）
                        self._update_spatial_grid()
                        if self.goal_reached:
                            break

                # ========== 智能跳帧：卡顿时跳过渲染 ==========
                if self.performance_mode and self.frame_skip > 0:
                    self.frame_skip -= 1
                else:
                    self._draw()

                    if self.recording:
                        self._record_frame()

                # 帧率控制
                self.clock.tick(self.config.FPS)

                # 检测是否需要跳帧
                frame_time = time.time() - frame_start
                if self.performance_mode and frame_time > 0.05:  # >50ms = 卡顿
                    self.frame_skip = min(self.frame_skip + 1, self.max_frame_skip)
        except Exception as e:
            print(f"\n❌ 仿真异常退出: {e}")
            raise
        finally:
            self._cleanup()
    
    def _handle_events(self):
        """处理事件（键盘+鼠标+虚拟按钮）"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # ========== 键盘事件 ==========
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("暂停" if self.paused else "继续")
                elif event.key == pygame.K_m:
                    self.control_mode = "manual" if self.control_mode == "auto" else "auto"
                    print(f"切换到 {self.control_mode.upper()} 模式")
                elif event.key == pygame.K_b:
                    self._enable_mid_route_barrier(not bool(self._mid_route_barrier))
                elif event.key == pygame.K_1:
                    self._switch_algorithm_manual("A*")
                elif event.key == pygame.K_2:
                    self._switch_algorithm_manual("Ice-Theta*")
                elif event.key == pygame.K_3:
                    self._switch_algorithm_manual("Straight")
                # 缩放控制
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self._zoom_in()
                elif event.key == pygame.K_MINUS:
                    self._zoom_out()
                elif event.key == pygame.K_0:
                    self._zoom_reset()
                # 定位到船舶
                elif event.key == pygame.K_f:
                    self._locate_ship()
            
            # ========== 鼠标滚轮缩放 ==========
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self._zoom_in()
                elif event.y < 0:
                    self._zoom_out()
            
            # ========== 鼠标点击事件 ==========
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                
                # 检查缩放按钮
                for btn_name, rect in self.zoom_button_rects.items():
                    if rect.collidepoint(mouse_pos):
                        if btn_name == "zoom_in":
                            self._zoom_in()
                        elif btn_name == "zoom_out":
                            self._zoom_out()
                        elif btn_name == "zoom_reset":
                            self._zoom_reset()
                
                # 检查定位按钮
                if self.locate_button_rect and self.locate_button_rect.collidepoint(mouse_pos):
                    self._locate_ship()

                if self.barrier_button_rect and self.barrier_button_rect.collidepoint(mouse_pos):
                    self._enable_mid_route_barrier(not bool(self._mid_route_barrier))
                
                # 检查模式切换按钮
                if self.mode_button_rect and self.mode_button_rect.collidepoint(mouse_pos):
                    self.control_mode = "manual" if self.control_mode == "auto" else "auto"
                    print(f"点击切换到 {self.control_mode.upper()} 模式")
                
                # 检查算法切换按钮
                for algo_type, rect in self.path_button_rects.items():
                    if rect.collidepoint(mouse_pos):
                        self._switch_algorithm_manual(algo_type)
                
                # 检查倍速按钮
                for speed, rect in self.speed_button_rects.items():
                    if rect.collidepoint(mouse_pos):
                        self.speed_multiplier = speed
                        print(f"仿真速度: {speed}x")
                
                # 检查方向键按钮（手动模式）
                if self.control_mode == "manual":
                    for key, rect in self.button_rects.items():
                        if rect.collidepoint(mouse_pos):
                            self.virtual_keys[key] = True
            
            # ========== 鼠标释放事件 ==========
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                for key in self.virtual_keys:
                    self.virtual_keys[key] = False
    
    def _switch_algorithm_manual(self, algo_name: str):
        """手动切换算法（禁用自动切换）"""
        self.auto_run_algorithms = False  # 手动切换时禁用自动运行
        print(f"\n手动切换到 {algo_name} (自动运行已禁用)")
        self._reset_for_next_algorithm()
        self._change_algorithm(algo_name)

    def _adjust_start_goal_if_blocked(self):
        if not getattr(self, '_ice_block_fast', None):
            return

        def is_blocked(x: float, y: float) -> bool:
            for cx, cy, r2 in self._ice_block_fast:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy <= r2:
                    return True
            return False

        def find_free_near(x0: float, y0: float, max_r: float, step: float):
            if not is_blocked(x0, y0):
                return x0, y0
            wx = float(getattr(self.config, 'WORLD_WIDTH', 0.0))
            wy = float(getattr(self.config, 'WORLD_HEIGHT', 0.0))
            margin = max(5.0, step)
            r = step
            while r <= max_r:
                for ang_deg in range(0, 360, 15):
                    ang = np.radians(float(ang_deg))
                    x = x0 + r * float(np.cos(ang))
                    y = y0 + r * float(np.sin(ang))
                    x = max(margin, min(wx - margin, x))
                    y = max(margin, min(wy - margin, y))
                    if not is_blocked(x, y):
                        return x, y
                r += step
            return x0, y0

        wx = float(getattr(self.config, 'WORLD_WIDTH', 0.0))
        wy = float(getattr(self.config, 'WORLD_HEIGHT', 0.0))
        if wx <= 0 or wy <= 0:
            return

        step = max(8.0, float(getattr(self.config, 'SHIP_LENGTH', 80.0)) * 0.15)
        max_r = max(60.0, float(getattr(self.config, 'SHIP_LENGTH', 80.0)) * 3.0)

        sx = float(getattr(self.config, 'SHIP_START_X', 0.05)) * wx
        sy = float(getattr(self.config, 'SHIP_START_Y', 0.5)) * wy
        gx = float(getattr(self.config, 'GOAL_X', 0.95)) * wx
        gy = float(getattr(self.config, 'GOAL_Y', 0.5)) * wy

        new_sx, new_sy = find_free_near(sx, sy, max_r=max_r, step=step)
        new_gx, new_gy = find_free_near(gx, gy, max_r=max_r, step=step)

        moved = False
        if (abs(new_sx - sx) > 1e-6) or (abs(new_sy - sy) > 1e-6):
            moved = True
        if (abs(new_gx - gx) > 1e-6) or (abs(new_gy - gy) > 1e-6):
            moved = True

        if moved:
            try:
                self.config.SHIP_START_X = float(new_sx) / wx
                self.config.SHIP_START_Y = float(new_sy) / wy
                self.config.GOAL_X = float(new_gx) / wx
                self.config.GOAL_Y = float(new_gy) / wy
                print(f"   ✓ Start/Goal adjusted: start=({new_sx:.1f},{new_sy:.1f}), goal=({new_gx:.1f},{new_gy:.1f})")
            except Exception:
                pass

    def _enable_mid_route_barrier(self, enabled: bool):
        if enabled:
            if self._mid_route_barrier is not None:
                return
            wx = float(getattr(self.config, 'WORLD_WIDTH', 0.0))
            wy = float(getattr(self.config, 'WORLD_HEIGHT', 0.0))
            if wx <= 0 or wy <= 0:
                return
            cx = wx * 0.5
            cy = wy * 0.5
            span_y = float(getattr(self.config, 'MID_ROUTE_BARRIER_WIDTH_RATIO', 0.65))
            span_y = max(0.1, min(0.98, span_y))
            width = wy * span_y
            thickness = float(getattr(self.config, 'MID_ROUTE_BARRIER_THICKNESS_M', 200.0))
            thickness = max(10.0, min(thickness, wx * 0.8))

            vertices = [
                (-thickness * 0.5, -width * 0.5),
                (-thickness * 0.5,  width * 0.5),
                ( thickness * 0.5,  width * 0.5),
                ( thickness * 0.5, -width * 0.5),
            ]

            ice_data = {
                'type': 'Ice Bank',
                'size': max(width, thickness),
                'center': (cx, cy),
                'vertices': vertices,
                'area': float(width * thickness),
                'mass': float(getattr(self.config, 'SHIP_MASS', 1.0)) * 1e6,
            }
            ice_block = IceBlock(self.space, ice_data, self.config)
            try:
                setattr(ice_block.shape, 'is_mid_route_barrier', True)
            except Exception:
                pass
            self._mid_route_barrier = ice_block
            self._mid_route_barrier_data = ice_data
            self.ice_blocks.append(ice_block)
            try:
                self.ice_blocks_data.append(ice_data)
            except Exception:
                pass
            self._mid_route_barrier_collision_count = 0
            if getattr(self.ship, 'use_mmg', False):
                try:
                    self.ship.force_replan = True
                except Exception:
                    pass
        else:
            if self._mid_route_barrier is None:
                return
            try:
                self.space.remove(self._mid_route_barrier.shape, self._mid_route_barrier.body)
            except Exception:
                pass
            try:
                if self._mid_route_barrier in self.ice_blocks:
                    self.ice_blocks.remove(self._mid_route_barrier)
            except Exception:
                pass
            try:
                if self._mid_route_barrier_data in self.ice_blocks_data:
                    self.ice_blocks_data.remove(self._mid_route_barrier_data)
            except Exception:
                pass
            self._mid_route_barrier = None
            self._mid_route_barrier_data = None
            self._mid_route_barrier_collision_count = 0
            if getattr(self.ship, 'use_mmg', False):
                try:
                    self.ship.force_replan = True
                except Exception:
                    pass
    
    # ========== 摄像机控制方法 ==========
    
    def _zoom_in(self):
        """放大视图"""
        if self.current_zoom_index < len(self.zoom_levels) - 1:
            self.current_zoom_index += 1
            self.camera_zoom = self.zoom_levels[self.current_zoom_index]
            # 放大时自动定位到船舶
            if self.camera_zoom > 1.0:
                self.camera_follow_ship = True
            print(f"缩放: {self.camera_zoom:.1f}x")
    
    def _zoom_out(self):
        """缩小视图"""
        if self.current_zoom_index > 0:
            self.current_zoom_index -= 1
            self.camera_zoom = self.zoom_levels[self.current_zoom_index]
            # 缩小到全局时取消跟随
            if self.camera_zoom <= 1.0:
                self.camera_follow_ship = False
                self.camera_center = None
            print(f"缩放: {self.camera_zoom:.1f}x")
    
    def _zoom_reset(self):
        """重置缩放到全局视图"""
        self.current_zoom_index = 2  # 1.0x
        self.camera_zoom = 1.0
        self.camera_follow_ship = False
        self.camera_center = None
        print("视图已重置")
    
    def _locate_ship(self):
        """定位到船舶位置"""
        self.camera_follow_ship = True
        # 如果当前是全局视图，放大一点
        if self.camera_zoom <= 1.0:
            self.current_zoom_index = 3  # 2.0x
            self.camera_zoom = 2.0
        print(f"已定位到船舶 (缩放: {self.camera_zoom:.1f}x)")
    
    def _world_to_screen(self, world_x: float, world_y: float, base_scale: float) -> Tuple[int, int]:
        """世界坐标转屏幕坐标（带摄像机变换）"""
        if self.camera_follow_ship and self.camera_zoom > 1.0:
            center_x, center_y = self.ship.body.position
        else:
            if self.camera_center:
                center_x, center_y = self.camera_center
            else:
                center_x, center_y = (self.config.WORLD_WIDTH / 2), (self.config.WORLD_HEIGHT / 2)

        rel_x = (world_x - center_x) * base_scale * self.camera_zoom
        rel_y = (world_y - center_y) * base_scale * self.camera_zoom

        screen_cx = self.config.GAME_AREA_WIDTH // 2
        screen_cy = self.config.WINDOW_HEIGHT // 2

        screen_x = int(screen_cx + rel_x)
        screen_y = int(screen_cy - rel_y)

        return screen_x, screen_y
    
    def _update(self):
        """更新仿真（带性能优化）- 照抄old版本"""
        dt = self.config.PHYSICS_DT
        self.simulation_time += dt

        prev_ship_pos = pymunk.Vec2d(self.ship.body.position.x, self.ship.body.position.y)
        
        # ========== 动态路径重规划（降低频率）==========
        if self.control_mode == "auto" and self.enable_dynamic_replan:
            # 性能模式下每5帧检查一次，否则每2帧
            check_interval = 5 if self.performance_mode else 2
            if not hasattr(self, '_replan_check_counter'):
                self._replan_check_counter = 0
            self._replan_check_counter += 1
            if self._replan_check_counter % check_interval == 0:
                self._check_and_replan()
        
        # 控制船舶
        if self.control_mode == "auto":
            self.ship._reactive_delta = getattr(self, '_reactive_delta', 0.0)
            self.ship.update(dt, self.path, self.config)
        else:
            keys = pygame.key.get_pressed()
            self.ship.apply_manual_control(keys, self.config, virtual_keys=self.virtual_keys)
            if self.ship.use_mmg:
                self.ship.dynamics.step(dt)
                state = self.ship.dynamics.state
                self.ship.body.position = (state[0], state[1])
                self.ship.body.angle = state[2]
        
        # 记录时间戳（用于MMG数据导出）
        if self.ship.use_mmg:
            self.ship.time_history.append(self.simulation_time)

        if getattr(self.config, 'ENABLE_DISTANT_ICE_SLEEP', False) or self.performance_mode:
            self._wake_nearby_ice_for_ship()
        
        # 物理步进（性能模式下减少迭代）
        self.space.step(dt)

        try:
            ship_pos = self.ship.body.position
            for shp in getattr(self, '_center_obstacle_shapes', []) or []:
                try:
                    info = shp.point_query(ship_pos)
                    if getattr(info, 'distance', 0.0) < 0.0:
                        dist = float(getattr(info, 'distance', 0.0))
                        grad = getattr(info, 'gradient', None)
                        gx = float(getattr(grad, 'x', 0.0)) if grad is not None else 0.0
                        gy = float(getattr(grad, 'y', 0.0)) if grad is not None else 0.0
                        gl = float(np.hypot(gx, gy))
                        if gl < 1e-6:
                            try:
                                oc = shp.body.position
                                gx = float(ship_pos.x - oc.x)
                                gy = float(ship_pos.y - oc.y)
                                gl = float(np.hypot(gx, gy))
                            except Exception:
                                gl = 0.0
                        if gl < 1e-6:
                            gx, gy, gl = 1.0, 0.0, 1.0

                        gx /= gl
                        gy /= gl

                        # 将船“推出”多边形边界外一点点，而不是回退/停死
                        push = (-dist) + 1.0
                        new_x = float(ship_pos.x) + gx * push
                        new_y = float(ship_pos.y) + gy * push
                        self.ship.body.position = (new_x, new_y)

                        try:
                            self.ship.body.velocity = self.ship.body.velocity * 0.4
                        except Exception:
                            self.ship.body.velocity = (0, 0)
                        self.ship.body.angular_velocity = 0

                        if getattr(self.ship, 'use_mmg', False) and hasattr(self.ship, 'dynamics'):
                            try:
                                self.ship.dynamics.state[0] = float(new_x)
                                self.ship.dynamics.state[1] = float(new_y)
                                if len(self.ship.dynamics.state) >= 4:
                                    self.ship.dynamics.state[3] = float(self.ship.dynamics.state[3]) * 0.4
                                if len(self.ship.dynamics.state) >= 5:
                                    self.ship.dynamics.state[4] = float(self.ship.dynamics.state[4]) * 0.4
                            except Exception:
                                pass
                        break
                except Exception:
                    continue
        except Exception:
            pass
        
        # ========== 边界约束：防止船跑出画面 ==========
        margin = 20  # 边距
        ship_pos = self.ship.body.position
        clamped_x = max(margin, min(ship_pos.x, self.config.WORLD_WIDTH - margin))
        clamped_y = max(margin, min(ship_pos.y, self.config.WORLD_HEIGHT - margin))
        
        if ship_pos.x != clamped_x or ship_pos.y != clamped_y:
            self.ship.body.position = (clamped_x, clamped_y)
            # 同步MMG状态
            if self.ship.use_mmg:
                self.ship.dynamics.state[0] = clamped_x
                self.ship.dynamics.state[1] = clamped_y
        
        # 检查到达目标
        self._check_goal_reached()

    def _wake_nearby_ice_for_ship(self):
        if not hasattr(self, '_wake_counter'):
            self._wake_counter = 0
        self._wake_counter += 1
        interval = 8 if self.ultra_performance else (4 if self.performance_mode else 2)
        if self._wake_counter % interval != 0:
            return

        ship_pos = self.ship.body.position
        r = float(getattr(self.config, 'SHIP_ICE_WAKE_RADIUS', 260.0))
        nearby = self._get_nearby_ice_fast(float(ship_pos.x), float(ship_pos.y), r)
        for ice_block in nearby:
            try:
                body = ice_block.body
                if getattr(body, 'is_sleeping', False):
                    body.activate()
            except Exception:
                pass

    def _update_local_collision_and_sleep(self):
        self._local_mask_update_counter += 1
        update_interval = int(getattr(self.config, 'LOCAL_ICE_ICE_UPDATE_INTERVAL_FRAMES', 10))
        update_interval = max(1, update_interval)
        if self._local_mask_update_counter % update_interval != 0:
            return

        ship_pos = self.ship.body.position
        ship_cat = getattr(self.config, 'SHIP_COLLISION_CATEGORY', 0b1)
        ice_cat = getattr(self.config, 'ICE_COLLISION_CATEGORY', 0b10)

        if getattr(self.config, 'ENABLE_LOCAL_ICE_ICE_COLLISION', False):
            local_r = float(getattr(self.config, 'LOCAL_ICE_ICE_RADIUS', 220.0))
            nearby = self._get_nearby_ice_fast(ship_pos.x, ship_pos.y, local_r)
            self._local_ice_set = set(nearby)
            for ice_block in self._local_ice_set:
                try:
                    if getattr(ice_block.body, 'is_sleeping', False):
                        ice_block.body.activate()
                except Exception:
                    pass

        if getattr(self.config, 'ENABLE_DISTANT_ICE_SLEEP', False):
            ice_blocks = self.ice_blocks
            n = len(ice_blocks)
            if n == 0:
                return

            sleep_r = float(getattr(self.config, 'DISTANT_ICE_SLEEP_RADIUS', 700.0))
            sleep_r2 = sleep_r * sleep_r
            v_th = float(getattr(self.config, 'ICE_SLEEP_SPEED_THRESHOLD', 0.08))
            w_th = float(getattr(self.config, 'ICE_SLEEP_ANGVEL_THRESHOLD', 0.15))
            batch = int(getattr(self.config, 'ICE_SLEEP_SCAN_BATCH', 400))
            batch = max(50, batch)

            start = self._ice_sleep_scan_index % n
            end = min(start + batch, n)
            scan_slice = ice_blocks[start:end]
            self._ice_sleep_scan_index = 0 if end >= n else end

            sx = float(ship_pos.x)
            sy = float(ship_pos.y)
            for ice_block in scan_slice:
                body = ice_block.body
                if body.body_type != pymunk.Body.DYNAMIC:
                    continue
                dx = float(body.position.x) - sx
                dy = float(body.position.y) - sy
                d2 = dx * dx + dy * dy
                if d2 > sleep_r2:
                    continue
                else:
                    if getattr(body, 'is_sleeping', False):
                        try:
                            body.activate()
                        except Exception:
                            pass
    
    def _check_and_replan(self):
        """
        智能导航：收集冰块 + 检测是否需要重规划 - 照抄old版本
        """
        ship_pos = self.ship.body.position
        ship_heading = self.ship.body.angle
        
        # ========== 1. 收集附近冰块（使用空间索引加速）==========
        scan_range = 150.0
        nearby_blocks = self._get_nearby_ice_fast(ship_pos.x, ship_pos.y, scan_range)
        
        nearby_ice = []
        for ice_block in nearby_blocks:
            ice_pos = ice_block.body.position
            size = getattr(ice_block, 'size', 10)
            nearby_ice.append((ice_pos.x, ice_pos.y, size))
        
        self.ship._nearby_ice = nearby_ice
        
        # ========== 2. 多条件检测是否需要重规划 ==========
        goal_x = self.config.GOAL_X * self.config.WORLD_WIDTH
        goal_y = self.config.GOAL_Y * self.config.WORLD_HEIGHT
        
        dx_goal = goal_x - ship_pos.x
        dy_goal = goal_y - ship_pos.y
        heading_error = abs(np.arctan2(np.sin(np.arctan2(dy_goal, dx_goal) - ship_heading), 
                                        np.cos(np.arctan2(dy_goal, dx_goal) - ship_heading)))
        heading_error_deg = np.degrees(heading_error)
        
        dist_to_goal = np.sqrt(dx_goal**2 + dy_goal**2)
        
        # 初始化状态
        if not hasattr(self, '_last_dist_to_goal'):
            self._last_dist_to_goal = dist_to_goal
            self._wrong_way_count = 0
            self._stuck_count = 0
            self._last_positions = []
        
        # 条件1: 船头偏离目标超过90度
        going_wrong_way = heading_error_deg > 90
        
        # 条件2: 离目标越来越远
        if dist_to_goal > self._last_dist_to_goal + 3:
            self._wrong_way_count += 1
        else:
            self._wrong_way_count = max(0, self._wrong_way_count - 2)
        self._last_dist_to_goal = dist_to_goal
        
        # 条件3: 卡住检测（位置几乎不变）
        self._last_positions.append((ship_pos.x, ship_pos.y))
        if len(self._last_positions) > 60:
            self._last_positions.pop(0)
        
        is_stuck = False
        if len(self._last_positions) >= 60:
            start_pos = self._last_positions[0]
            end_pos = self._last_positions[-1]
            movement = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
            if movement < 2.0:
                is_stuck = True
                self._stuck_count += 1
            else:
                self._stuck_count = max(0, self._stuck_count - 1)
        
        # ========== 3. 智能重规划（带指数退避）==========
        if not hasattr(self, '_replan_backoff'):
            self._replan_backoff = 5.0
            self._consecutive_stuck_replans = 0
        
        autopilot_need_replan = getattr(self.ship, 'force_replan', False)
        if autopilot_need_replan:
            self.ship.force_replan = False
        
        truly_stuck = is_stuck and self._stuck_count > 60
        time_since_replan = self.simulation_time - self.last_replan_time
        
        if truly_stuck:
            if time_since_replan > self._replan_backoff:
                self._consecutive_stuck_replans += 1
                if self._consecutive_stuck_replans <= 2:
                    self._local_path_repair(ship_pos, ship_heading)
                elif self._consecutive_stuck_replans <= 4:
                    if hasattr(self.ship, 'dynamics'):
                        self.ship.dynamics.n_cmd = min(1.2, self.ship.dynamics.n_cmd + 0.1)
                self._replan_backoff = min(30.0, self._replan_backoff * 1.5)
                self.last_replan_time = self.simulation_time
                self._stuck_count = 0
                self._last_positions.clear()
        
        elif (going_wrong_way or self._wrong_way_count > 20) and time_since_replan > 15.0:
            self._local_path_repair(ship_pos, ship_heading)
            self._wrong_way_count = 0
            self.last_replan_time = self.simulation_time
            self._replan_backoff = 5.0
            self._consecutive_stuck_replans = 0
        
        if not is_stuck and self._consecutive_stuck_replans > 0:
            self._consecutive_stuck_replans = 0
            self._replan_backoff = 5.0
        
        # ========== 4. 检查并应用后台规划结果 ==========
        self._apply_pending_path()
        
        # ========== 5. 智能航点跳转 ==========
        self._smart_waypoint_jump(ship_pos)
        
    
    def _local_path_repair(self, ship_pos, ship_heading):
        """
        局部路径修复 - 朝前方（目标方向）插入绕行点
        """
        if not self.path or len(self.path) < 2:
            return
        
        current_idx = self.ship.path_index
        if current_idx >= len(self.path) - 1:
            return
        
        # 计算朝向目标的方向（不是船头方向！）
        goal_x = self.config.GOAL_X * self.config.WORLD_WIDTH
        goal_y = self.config.GOAL_Y * self.config.WORLD_HEIGHT
        dx_goal = goal_x - ship_pos.x
        dy_goal = goal_y - ship_pos.y
        goal_vec = pymunk.Vec2d(dx_goal, dy_goal)
        if goal_vec.length < 1e-6:
            return
        goal_dir = goal_vec.normalized()
        goal_angle = np.arctan2(goal_dir.y, goal_dir.x)
        
        # 寻找最佳绕行方向（基于目标方向，不是船头方向）
        best_angle = None
        best_score = -float('inf')
        
        # 只在目标方向的左右60度范围内搜索
        for angle_offset in [-45, -30, -15, 0, 15, 30, 45]:
            test_angle = goal_angle + np.radians(angle_offset)
            test_dist = 50  # 前方50米
            test_x = ship_pos.x + test_dist * np.cos(test_angle)
            test_y = ship_pos.y + test_dist * np.sin(test_angle)
            
            # 确保是朝目标方向
            test_vec = pymunk.Vec2d(test_x - ship_pos.x, test_y - ship_pos.y)
            if test_vec.dot(goal_dir) <= 0:
                continue
            
            # 检查该方向是否有冰块
            nearby = self._get_nearby_ice_fast(test_x, test_y, 30)
            ice_penalty = len(nearby) * 15
            
            progress = test_vec.dot(goal_dir)
            score = progress - ice_penalty
            
            if score > best_score:
                best_score = score
                best_angle = test_angle
        
        if best_angle is not None:
            # 插入绕行点
            detour_dist = 40
            detour_x = ship_pos.x + detour_dist * np.cos(best_angle)
            detour_y = ship_pos.y + detour_dist * np.sin(best_angle)
            detour_vec = pymunk.Vec2d(detour_x - ship_pos.x, detour_y - ship_pos.y)
            if detour_vec.dot(goal_dir) > 0:
                detour_proj = detour_vec.dot(goal_dir)
                insert_idx = current_idx + 1
                for i in range(current_idx + 1, len(self.path)):
                    wp_vec = pymunk.Vec2d(self.path[i][0] - ship_pos.x, self.path[i][1] - ship_pos.y)
                    wp_proj = wp_vec.dot(goal_dir)
                    if wp_proj > detour_proj:
                        insert_idx = i
                        break
                
                new_path = list(self.path[:insert_idx])
                new_path.append((detour_x, detour_y))
                new_path.extend(self.path[insert_idx:])

                self.path = self._post_process_path(new_path, smooth=False)
                self.ship.set_path(self.path)
                self.ship.path_index = max(0, insert_idx - 1)
                print(f"    → 局部修复：前方插入绕行点 ({detour_x:.0f}, {detour_y:.0f})")
    
    def _reactive_avoidance(self, ship_pos, ship_heading):
        """
        反应式避障 - 势场法
        实时计算避障力，调整舵角
        """
        if not hasattr(self, '_reactive_delta'):
            self._reactive_delta = 0.0
        
        # 扫描前方冰块（使用空间索引）
        avoid_force_x = 0.0
        avoid_force_y = 0.0
        
        scan_range = 80.0
        nearby_blocks = self._get_nearby_ice_fast(ship_pos.x, ship_pos.y, scan_range)
        
        for ice_block in nearby_blocks:
            ice_pos = ice_block.body.position
            dx = ice_pos.x - ship_pos.x
            dy = ice_pos.y - ship_pos.y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 1.0:
                size = getattr(ice_block, 'size', 10)
                # 斥力（反方向）
                repel_strength = size * 500 / (dist * dist)
                avoid_force_x -= dx / dist * repel_strength
                avoid_force_y -= dy / dist * repel_strength
        
        # 目标吸引力
        goal_x = self.config.GOAL_X * self.config.WORLD_WIDTH
        goal_y = self.config.GOAL_Y * self.config.WORLD_HEIGHT
        dx_goal = goal_x - ship_pos.x
        dy_goal = goal_y - ship_pos.y
        dist_goal = np.sqrt(dx_goal*dx_goal + dy_goal*dy_goal)
        
        if dist_goal > 1.0:
            attract_strength = 100.0
            avoid_force_x += dx_goal / dist_goal * attract_strength
            avoid_force_y += dy_goal / dist_goal * attract_strength
        
        # 计算合力方向
        if abs(avoid_force_x) > 0.1 or abs(avoid_force_y) > 0.1:
            desired_heading = np.arctan2(avoid_force_y, avoid_force_x)
            heading_error = desired_heading - ship_heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            # 转换为舵角修正
            self._reactive_delta = np.clip(np.degrees(heading_error) * 0.5, -15, 15)
        else:
            self._reactive_delta = 0.0
    
    def _smart_waypoint_jump(self, ship_pos):
        """
        智能航点跳转 - 简化稳定版 (参考旧版本)
        只在必要时更新，避免频繁切换导致卡顿
        """
        if len(self.path) < 2:
            return
        if not hasattr(self.ship, 'path_index'):
            return
        
        current_idx = int(self.ship.path_index)
        
        # 计算目标方向
        goal_x = self.config.GOAL_X * self.config.WORLD_WIDTH
        goal_y = self.config.GOAL_Y * self.config.WORLD_HEIGHT
        goal_dir = pymunk.Vec2d(goal_x - ship_pos.x, goal_y - ship_pos.y)
        if goal_dir.length < 1:
            return
        goal_dir = goal_dir.normalized()
        
        # 检查当前航点是否在船后方
        if current_idx < len(self.path):
            cur_wp = self.path[current_idx]
            to_cur = pymunk.Vec2d(cur_wp[0] - ship_pos.x, cur_wp[1] - ship_pos.y)
            
            # 如果当前航点在前方且距离合理，不需要跳转
            if to_cur.dot(goal_dir) > 0 and to_cur.length < 300:
                return
        
        # 需要重新找航点：找船前方最近的航点
        best_idx = None
        best_dist = float('inf')
        
        for i in range(len(self.path)):
            wp = self.path[i]
            to_wp = pymunk.Vec2d(wp[0] - ship_pos.x, wp[1] - ship_pos.y)
            
            # 只考虑在目标方向前方的航点
            if to_wp.dot(goal_dir) <= 0:
                continue
            
            dist = to_wp.length
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        # 更新航点（只在真正需要时）
        if best_idx is not None and best_idx != current_idx:
            self.ship.path_index = best_idx
            if hasattr(self.ship, 'autopilot'):
                self.ship.autopilot.current_wp_index = best_idx
        elif best_idx is None:
            # 没有前方航点，接近终点
            self.ship.path_index = len(self.path) - 1
    
    def _normalize_angle(self, angle):
        """归一化角度到 [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def _apply_pending_path(self):
        """
        应用后台规划完成的路径（非阻塞检查）
        """
        try:
            new_path = self._path_queue.get_nowait()
            if new_path and len(new_path) > 1:
                self.path = self._post_process_path(new_path, smooth=True)
                
                # 找到船前方最近的航点作为起始索引
                ship_pos = self.ship.body.position
                goal_x = self.config.GOAL_X * self.config.WORLD_WIDTH
                goal_y = self.config.GOAL_Y * self.config.WORLD_HEIGHT
                goal_vec = pymunk.Vec2d(goal_x - ship_pos.x, goal_y - ship_pos.y)
                start_idx = 0
                if goal_vec.length > 1e-6:
                    goal_dir = goal_vec.normalized()
                    best_dist = float('inf')
                    best_idx = None
                    for i, wp in enumerate(new_path):
                        wp_vec = pymunk.Vec2d(wp[0] - ship_pos.x, wp[1] - ship_pos.y)
                        if wp_vec.dot(goal_dir) <= 0:
                            continue
                        d = wp_vec.length
                        if d < best_dist:
                            best_dist = d
                            best_idx = i
                    if best_idx is not None:
                        start_idx = best_idx

                self.ship.path_index = start_idx
                if hasattr(self.ship, 'autopilot'):
                    self.ship.autopilot.waypoints = self.path
                    self.ship.autopilot.current_wp_index = start_idx
                self._replan_count += 1
                print(f"    → 新路径已应用 {len(self.path)} 个点，起始索引: {start_idx}")
        except:
            pass  # 队列为空，没有新路径
    
    def _force_replan_from_current(self):
        """强制从当前位置重新规划路径"""
        if not self._planning_in_progress:
            self._start_background_replan()
            self.last_replan_time = self.simulation_time
    
    def _start_background_replan(self):
        """启动后台规划线程"""
        if self._planning_in_progress:
            return
        
        self._planning_in_progress = True
        
        # 获取当前状态快照
        ship_pos = self.ship.body.position
        ship_x, ship_y = ship_pos.x, ship_pos.y
        path_type = self.config.PATH_TYPE
        goal_x = self.config.GOAL_X * self.config.WORLD_WIDTH
        goal_y = self.config.GOAL_Y * self.config.WORLD_HEIGHT
        
        # 获取冰块数据快照
        ice_data = []
        for ice_block in self.ice_blocks:
            ice_data.append({
                'center': tuple(ice_block.body.position),
                'size': getattr(ice_block, 'size', 20),
                'type': ice_block.ice_type
            })
        
        # 在后台线程中执行规划
        def plan_in_background():
            try:
                new_path = self._compute_path(ship_x, ship_y, goal_x, goal_y, path_type, ice_data)
                if new_path and len(new_path) > 1:
                    # 简单插值：确保路径有足够多的点
                    interpolated = self._interpolate_path(new_path, step=10.0)
                    try:
                        self._path_queue.get_nowait()
                    except Empty:
                        pass
                    self._path_queue.put(interpolated)
            except Exception:
                pass
            finally:
                self._planning_in_progress = False
        
        self._planning_thread = threading.Thread(target=plan_in_background, daemon=True)
        self._planning_thread.start()
    
    def _interpolate_path(self, path, step=10.0):
        """对路径进行插值，确保相邻点间距不超过step"""
        if len(path) < 2:
            return path
        
        result = [path[0]]
        for i in range(1, len(path)):
            p1 = path[i-1]
            p2 = path[i]
            dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            
            if dist > step:
                # 需要插值
                n_points = int(dist / step)
                for j in range(1, n_points + 1):
                    t = j / (n_points + 1)
                    px = p1[0] + (p2[0] - p1[0]) * t
                    py = p1[1] + (p2[1] - p1[1]) * t
                    result.append((px, py))
            
            result.append(p2)
        
        return result
    
    def _compute_path(self, start_x, start_y, goal_x, goal_y, path_type, ice_data):
        """
        计算路径（在后台线程中调用）
        使用快速算法版本，避免阻塞
        """
        if path_type == "Straight":
            # 直线算法
            return [(start_x, start_y), (goal_x, goal_y)]
        
        elif path_type == "A*":
            return self._fast_dstar_path(start_x, start_y, goal_x, goal_y, ice_data)

        elif path_type == "Ice-Theta*":
            return self._fast_dstar_path(start_x, start_y, goal_x, goal_y, ice_data)

        else:
            return self._fast_dstar_path(start_x, start_y, goal_x, goal_y, ice_data)
    
    def _fast_skeleton_path(self, start_x, start_y, goal_x, goal_y, ice_data):
        """快速Skeleton路径 - 分层避障"""
        path = [(start_x, start_y)]
        
        # 从起点Y到终点Y分层
        y_dist = goal_y - start_y
        if y_dist <= 0:
            return [(start_x, start_y), (goal_x, goal_y)]
        
        num_layers = min(20, max(5, int(y_dist / 20)))  # 限制层数
        current_x = start_x
        
        for i in range(1, num_layers):
            t = i / num_layers
            y = start_y + y_dist * t
            target_x = start_x + (goal_x - start_x) * t
            
            # 在目标X附近搜索最低代价点
            best_x = target_x
            min_cost = float('inf')
            
            for test_x in np.linspace(max(20, target_x - 60), min(self.config.WORLD_WIDTH - 20, target_x + 60), 15):
                cost = 0
                for ice in ice_data:
                    ix, iy = ice['center']
                    dist = np.sqrt((ix - test_x)**2 + (iy - y)**2)
                    size = ice.get('size', 10)
                    if dist < size * 1.5:
                        cost += size * 10 / (dist + 1)
                
                # 加入偏离惩罚
                cost += abs(test_x - target_x) * 0.5
                
                if cost < min_cost:
                    min_cost = cost
                    best_x = test_x
            
            path.append((best_x, y))
            current_x = best_x
        
        path.append((goal_x, goal_y))
        return path
    
    def _fast_dstar_path(self, start_x, start_y, goal_x, goal_y, ice_data):
        """快速D*Lite路径 - 简化A*"""
        grid_size = 15.0  # 粗网格，快速
        
        # 限制搜索区域
        margin = 100
        min_x = max(0, min(start_x, goal_x) - margin)
        max_x = min(self.config.WORLD_WIDTH, max(start_x, goal_x) + margin)
        min_y = max(0, min(start_y, goal_y) - margin)
        max_y = min(self.config.WORLD_HEIGHT, max(start_y, goal_y) + margin)
        
        grid_w = int((max_x - min_x) / grid_size) + 1
        grid_h = int((max_y - min_y) / grid_size) + 1
        
        # 限制网格大小
        if grid_w > 80 or grid_h > 80:
            grid_size = max((max_x - min_x) / 80, (max_y - min_y) / 80)
            grid_w = int((max_x - min_x) / grid_size) + 1
            grid_h = int((max_y - min_y) / grid_size) + 1
        
        # 快速构建代价图
        cost_map = np.ones((grid_h, grid_w))
        
        for ice in ice_data:
            cx, cy = ice['center']
            if cx < min_x or cx > max_x or cy < min_y or cy > max_y:
                continue
            size = ice.get('size', 10)
            gx = int((cx - min_x) / grid_size)
            gy = int((cy - min_y) / grid_size)
            r = int(size / grid_size) + 1
            
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < grid_w and 0 <= ny < grid_h:
                        d = np.sqrt(dx*dx + dy*dy) * grid_size
                        if d < size:
                            cost_map[ny, nx] = 999
                        elif d < size * 1.5:
                            cost_map[ny, nx] += 10
        
        # 简单A*
        start_g = (int((start_x - min_x) / grid_size), int((start_y - min_y) / grid_size))
        goal_g = (int((goal_x - min_x) / grid_size), int((goal_y - min_y) / grid_size))
        start_g = (max(0, min(start_g[0], grid_w-1)), max(0, min(start_g[1], grid_h-1)))
        goal_g = (max(0, min(goal_g[0], grid_w-1)), max(0, min(goal_g[1], grid_h-1)))
        
        open_set = [(0, start_g)]
        came_from = {}
        g_score = {start_g: 0}
        dirs = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
        
        for _ in range(1500):  # 限制迭代
            if not open_set:
                break
            _, cur = heapq.heappop(open_set)
            
            if cur == goal_g:
                path = []
                while cur in came_from:
                    path.append((min_x + cur[0] * grid_size, min_y + cur[1] * grid_size))
                    cur = came_from[cur]
                path.append((start_x, start_y))
                path.reverse()
                path.append((goal_x, goal_y))
                return path
            
            for dx, dy in dirs:
                nb = (cur[0] + dx, cur[1] + dy)
                if 0 <= nb[0] < grid_w and 0 <= nb[1] < grid_h and cost_map[nb[1], nb[0]] < 999:
                    ng = g_score[cur] + (1.4 if dx and dy else 1.0) * cost_map[nb[1], nb[0]]
                    if nb not in g_score or ng < g_score[nb]:
                        came_from[nb] = cur
                        g_score[nb] = ng
                        h = np.sqrt((nb[0] - goal_g[0])**2 + (nb[1] - goal_g[1])**2)
                        heapq.heappush(open_set, (ng + h, nb))
        
        return [(start_x, start_y), (goal_x, goal_y)]
    
    def _dstar_lite_search(self, start_x, start_y, goal_x, goal_y, ice_data):
        """
        D* Lite 路径搜索算法
        特点：考虑冰块代价的全局最优路径搜索
        优化：限制搜索区域以提高效率
        """
        # 网格参数（根据世界大小自适应）
        world_size = max(self.config.WORLD_WIDTH, self.config.WORLD_HEIGHT)
        grid_size = max(3.0, world_size / 150)  # 自适应网格大小
        
        # 计算搜索区域（扩大的矩形区域）
        margin = world_size * 0.3  # 30%边距
        min_x = max(0, min(start_x, goal_x) - margin)
        max_x = min(self.config.WORLD_WIDTH, max(start_x, goal_x) + margin)
        min_y = max(0, min(start_y, goal_y) - margin)
        max_y = min(self.config.WORLD_HEIGHT, max(start_y, goal_y) + margin)
        
        grid_width = int((max_x - min_x) / grid_size) + 1
        grid_height = int((max_y - min_y) / grid_size) + 1
        
        # 限制网格大小防止内存爆炸
        if grid_width > 200 or grid_height > 200:
            grid_size = max((max_x - min_x) / 200, (max_y - min_y) / 200)
            grid_width = int((max_x - min_x) / grid_size) + 1
            grid_height = int((max_y - min_y) / grid_size) + 1
        
        # 构建代价地图
        cost_map = np.ones((grid_height, grid_width))
        
        # 边界惩罚
        bm = 5
        if grid_width > bm * 2 and grid_height > bm * 2:
            cost_map[:, :bm] += 100.0
            cost_map[:, -bm:] += 100.0
            cost_map[:bm, :] += 100.0
            cost_map[-bm:, :] += 100.0
        
        # 冰块障碍
        for ice_info in ice_data:
            cx, cy = ice_info['center']
            size = ice_info.get('size', 20)
            ice_type = ice_info.get('type', 'Fragment')
            
            # 根据冰块类型设置权重
            if ice_type == 'Ice Bank':
                weight = 50.0
                expand = 2.0
            elif ice_type == 'Large Floe':
                weight = 30.0
                expand = 1.8
            elif ice_type == 'Medium Floe':
                weight = 15.0
                expand = 1.5
            elif ice_type == 'Small Floe':
                weight = 8.0
                expand = 1.3
            else:  # Fragment
                weight = 3.0
                expand = 1.2
            
            gx = int((cx - min_x) / grid_size)
            gy = int((cy - min_y) / grid_size)
            radius = int(size * expand / grid_size) + 2
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < grid_width and 0 <= ny < grid_height:
                        dist = np.sqrt(dx*dx + dy*dy) * grid_size
                        if dist < size:
                            cost_map[ny, nx] = 9999  # 不可通行
                        elif dist < size * expand:
                            cost_map[ny, nx] += weight * (1.0 - dist / (size * expand))
        
        # A*搜索（使用局部坐标）
        start_grid = (int((start_x - min_x) / grid_size), int((start_y - min_y) / grid_size))
        goal_grid = (int((goal_x - min_x) / grid_size), int((goal_y - min_y) / grid_size))
        
        start_grid = (max(0, min(start_grid[0], grid_width-1)),
                      max(0, min(start_grid[1], grid_height-1)))
        goal_grid = (max(0, min(goal_grid[0], grid_width-1)),
                     max(0, min(goal_grid[1], grid_height-1)))
        
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
        
        iterations = 0
        max_iterations = 5000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                # 重建路径（转换回世界坐标）
                path = []
                while current in came_from:
                    wx = min_x + current[0] * grid_size + grid_size / 2
                    wy = min_y + current[1] * grid_size + grid_size / 2
                    path.append((wx, wy))
                    current = came_from[current]
                path.append((start_x, start_y))
                path.reverse()
                path.append((goal_x, goal_y))
                return path
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height:
                    if cost_map[neighbor[1], neighbor[0]] >= 9999:
                        continue
                    
                    move_cost = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                    tentative_g = g_score[current] + move_cost * cost_map[neighbor[1], neighbor[0]]
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        h = np.sqrt((neighbor[0] - goal_grid[0])**2 + (neighbor[1] - goal_grid[1])**2)
                        heapq.heappush(open_set, (tentative_g + h * 1.5, neighbor))  # 加权启发式
        
        # 搜索失败，返回直线
        return [(start_x, start_y), (goal_x, goal_y)]
    
    def _generate_local_astar_path_static(self, start_x, start_y, goal_x, goal_y, ice_data, 
                                          grid_size=8.0, max_iter=2000):
        """静态版本的局部A*（线程安全，可配置参数）"""
        
        # 扩大搜索范围以找到更好的绕行路径
        margin = 150  # 增大边距
        min_x = min(start_x, goal_x) - margin
        max_x = max(start_x, goal_x) + margin
        min_y = min(start_y, goal_y) - margin
        max_y = max(start_y, goal_y) + margin
        
        min_x = max(0, min_x)
        max_x = min(self.config.WORLD_WIDTH, max_x)
        min_y = max(0, min_y)
        max_y = min(self.config.WORLD_HEIGHT, max_y)
        
        grid_width = int((max_x - min_x) / grid_size) + 1
        grid_height = int((max_y - min_y) / grid_size) + 1
        
        cost_map = np.ones((grid_height, grid_width))
        
        # 边界惩罚（向量化，快速）
        bm = 3
        if bm > 0 and grid_width > bm * 2 and grid_height > bm * 2:
            cost_map[:, :bm] += 50.0
            cost_map[:, -bm:] += 50.0
            cost_map[:bm, :] += 50.0
            cost_map[-bm:, :] += 50.0
        
        # 冰块障碍 - 增大影响范围确保避障
        for ice_info in ice_data:
            cx, cy = ice_info['center']
            size = ice_info.get('size', 20) * 1.5  # 放大避障范围
            
            # 只处理在搜索区域内的冰块
            if cx < min_x - size or cx > max_x + size:
                continue
            if cy < min_y - size or cy > max_y + size:
                continue
                
            gx = int((cx - min_x) / grid_size)
            gy = int((cy - min_y) / grid_size)
            radius = int(size / grid_size) + 3  # 更大的影响半径
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < grid_width and 0 <= ny < grid_height:
                        dist = np.sqrt(dx*dx + dy*dy) * grid_size
                        if dist < size * 0.8:
                            cost_map[ny, nx] = 999  # 完全阻塞
                        elif dist < size * 1.2:
                            cost_map[ny, nx] += 20.0  # 高代价
                        elif dist < size * 1.8:
                            cost_map[ny, nx] += 5.0   # 中等代价
        
        start_grid = (max(0, min(int((start_x - min_x) / grid_size), grid_width-1)),
                      max(0, min(int((start_y - min_y) / grid_size), grid_height-1)))
        goal_grid = (max(0, min(int((goal_x - min_x) / grid_size), grid_width-1)),
                     max(0, min(int((goal_y - min_y) / grid_size), grid_height-1)))
        
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
        
        iterations = 0
        
        while open_set and iterations < max_iter:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                path = []
                while current in came_from:
                    wx = min_x + current[0] * grid_size + grid_size / 2
                    wy = min_y + current[1] * grid_size + grid_size / 2
                    path.append((wx, wy))
                    current = came_from[current]
                path.append((start_x, start_y))
                path.reverse()
                path.append((goal_x, goal_y))
                return path
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height:
                    if cost_map[neighbor[1], neighbor[0]] >= 999:
                        continue
                    move_cost = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                    tentative_g = g_score[current] + move_cost * cost_map[neighbor[1], neighbor[0]]
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        h = np.sqrt((neighbor[0] - goal_grid[0])**2 + (neighbor[1] - goal_grid[1])**2)
                        heapq.heappush(open_set, (tentative_g + h, neighbor))
        
        return [(start_x, start_y), (goal_x, goal_y)]
    
    def _check_goal_reached(self):
        """检查是否到达目标"""
        goal_x = self.config.GOAL_X * self.config.WORLD_WIDTH
        goal_y = self.config.GOAL_Y * self.config.WORLD_HEIGHT
        
        ship_pos = self.ship.body.position
        distance = np.sqrt((ship_pos.x - goal_x)**2 + (ship_pos.y - goal_y)**2)
        
        if distance < self.config.GOAL_REACH_DISTANCE and not self.goal_reached:
            self.goal_reached = True
            journey_time = self.simulation_time - self.journey_start_time
            
            print(f"\n🎉 到达目标！")
            print(f"   算法: {self.config.PATH_TYPE}")
            print(f"   航行时间: {journey_time:.1f}s")
            print(f"   碰撞次数: {self.ship.collision_count}")
            
            # 保存结果
            self._save_algorithm_result(journey_time)
            
            # 自动切换到下一个算法（如果启用了自动运行）
            if self.auto_run_algorithms:
                self.current_algorithm_index += 1
                if self.current_algorithm_index < len(self.algorithm_sequence):
                    next_algo = self.algorithm_sequence[self.current_algorithm_index]
                    print(f"\n🔄 自动切换到算法: {next_algo}")
                    self._reset_for_next_algorithm()
                    self._change_algorithm(next_algo)
                else:
                    print("\n✅ 所有算法测试完成！")
                    self._generate_report()
                    if bool(getattr(self.config, 'AUTO_EXIT_ON_COMPLETE', False)):
                        self.running = False
                    else:
                        self.paused = True
            else:
                print("   (手动模式，不自动切换)")
                self.paused = True
    
    def _change_algorithm(self, algorithm: str):
        """切换算法"""
        self.config.PATH_TYPE = algorithm
        
        raw_path = self._generate_path(algorithm)
        self.path = self._post_process_path(raw_path, smooth=True)
        
        self.ship.path_index = 0
        if self.ship.use_mmg:
            self.ship.autopilot.set_waypoints(self.path)
            self.ship.autopilot.current_wp_index = 0
        
        self.goal_reached = False
        self.journey_start_time = self.simulation_time
        
        print(f"  路径点数: {len(self.path)}")
    
    def _reset_for_next_algorithm(self):
        """为下一个算法重置"""
        # 重置船舶
        self.ship.reset(self.config)
        
        # 重置冰块
        for i, ice_block in enumerate(self.ice_blocks):
            if i < len(self.ice_blocks_initial_state):
                initial = self.ice_blocks_initial_state[i]
                ice_block.body.position = initial['position']
                ice_block.body.velocity = initial['velocity']
                ice_block.body.angle = initial['angle']
                ice_block.body.angular_velocity = initial['angular_velocity']
            else:
                if ice_block is getattr(self, '_mid_route_barrier', None):
                    try:
                        pos = getattr(self, '_mid_route_barrier_data', {}) or {}
                        center = pos.get('center', None)
                        if center is not None:
                            ice_block.body.position = center
                        ice_block.body.velocity = (0, 0)
                        ice_block.body.angular_velocity = 0
                    except Exception:
                        pass

        if bool(getattr(self.config, 'ENABLE_MID_ROUTE_ICE_BARRIER', False)) and self._mid_route_barrier is None:
            try:
                self._enable_mid_route_barrier(True)
            except Exception:
                pass
    
    def _save_algorithm_result(self, journey_time: float):
        """保存算法结果"""
        algo_name = self.config.PATH_TYPE

        traj = self.ship.trajectory.copy()
        traj_distance = self._calculate_trajectory_length(traj)
        rudder_change_sum = self._calculate_rudder_change_sum()
        rudder_rate_mean_abs, rudder_rate_rms, rudder_rate_max_abs, rudder_rate_series = self._calculate_rudder_rate_metrics()
        curvature_sum, max_curvature = self._calculate_curvature_metrics(traj)
        safety_min, safety_avg = self._calculate_safety_metrics(traj)
        ice_work_mj = self._calculate_ice_work_mj()
        propulsive_energy_mj = self._calculate_propulsive_energy_mj()
        cte_metrics = self._calculate_path_following_metrics(traj, self.path)
        risk_metrics = self._calculate_collision_risk_metrics(self.ship.collision_details)
        
        self.algorithm_results[algo_name] = {
            'algorithm': algo_name,
            'journey_time': journey_time,
            'collision_count': self.ship.collision_count,
            'task_failed': bool(getattr(self, '_task_failed', False)),
            'failure_reason': getattr(self, '_failure_reason', None),
            'mid_route_barrier_enabled': bool(getattr(self, '_mid_route_barrier', None)),
            'mid_route_barrier_collision_count': int(getattr(self, '_mid_route_barrier_collision_count', 0) or 0),
            'mid_route_barrier_collision_fail_count': int(getattr(self.config, 'MID_ROUTE_BARRIER_COLLISION_FAIL_COUNT', 0) or 0),
            'ice_theta_ablation': getattr(self.config, 'ICE_THETA_ABLATION', None),
            'ice_theta_use_soft_cost': bool(getattr(self.config, 'ICE_THETA_USE_SOFT_COST', True)),
            'ice_theta_enable_any_angle': bool(getattr(self.config, 'ICE_THETA_ENABLE_ANY_ANGLE', True)),
            'ice_theta_los_step_cells': float(getattr(self.config, 'ICE_THETA_LOS_STEP_CELLS', 0.5)),
            'ice_theta_cost_sample_step_cells': float(getattr(self.config, 'ICE_THETA_COST_SAMPLE_STEP_CELLS', 0.5)),
            'planner_status': getattr(self, '_planner_status_by_algorithm', {}).get(algo_name, None),
            'ice_resistance_history': self.ship.ice_resistance_history.copy(),
            'avg_resistance': np.mean(self.ship.ice_resistance_history) if self.ship.ice_resistance_history else 0,
            'max_resistance': np.max(self.ship.ice_resistance_history) if self.ship.ice_resistance_history else 0,
            'total_resistance': np.sum(self.ship.ice_resistance_history) if self.ship.ice_resistance_history else 0,
            'trajectory': traj,
            'collision_details': self.ship.collision_details.copy(),
            'path_length': len(self.path),
            'trajectory_distance': traj_distance,
            'rudder_change_sum_deg': rudder_change_sum,
            'rudder_rate_mean_abs_deg_s': rudder_rate_mean_abs,
            'rudder_rate_rms_deg_s': rudder_rate_rms,
            'rudder_rate_max_abs_deg_s': rudder_rate_max_abs,
            'rudder_rate_series_deg_s': rudder_rate_series,
            'curvature_sum': curvature_sum,
            'max_curvature': max_curvature,
            'safety_min_clearance_m': safety_min,
            'safety_avg_clearance_m': safety_avg,
            'ice_work_mj': ice_work_mj,
            'propulsive_energy_mj': propulsive_energy_mj,
            'cte_center_mean_m': cte_metrics.get('cte_center_mean_m', 0.0),
            'cte_center_rms_m': cte_metrics.get('cte_center_rms_m', 0.0),
            'cte_center_p95_m': cte_metrics.get('cte_center_p95_m', 0.0),
            'cte_center_max_m': cte_metrics.get('cte_center_max_m', 0.0),
            'cte_tail_mean_m': cte_metrics.get('cte_tail_mean_m', 0.0),
            'cte_tail_rms_m': cte_metrics.get('cte_tail_rms_m', 0.0),
            'cte_tail_p95_m': cte_metrics.get('cte_tail_p95_m', 0.0),
            'cte_tail_max_m': cte_metrics.get('cte_tail_max_m', 0.0),
            'cte_time_s': cte_metrics.get('cte_time_s', []),
            'cte_center_series_m': cte_metrics.get('cte_center_series_m', []),
            'cte_tail_series_m': cte_metrics.get('cte_tail_series_m', []),
            'risk_metrics': risk_metrics,
            # 碰撞能量统计 (新增)
            'collision_energy_history': self.ship.collision_energy_history.copy(),
            'total_collision_energy': self.ship.total_collision_energy,
            'avg_collision_energy': np.mean(self.ship.collision_energy_history) if self.ship.collision_energy_history else 0,
            'max_collision_energy': np.max(self.ship.collision_energy_history) if self.ship.collision_energy_history else 0,
            # MMG特有数据
            'rudder_angle_history': self.ship.rudder_angle_history.copy() if self.ship.use_mmg else [],
            'propeller_rpm_history': self.ship.propeller_rpm_history.copy() if self.ship.use_mmg else [],
            'speed_history': self.ship.speed_history.copy() if self.ship.use_mmg else [],
            # MMG速度分量 (u, v, r)
            'time_history': self.ship.time_history.copy() if self.ship.use_mmg else [],
            'surge_velocity_history': self.ship.surge_velocity_history.copy() if self.ship.use_mmg else [],
            'sway_velocity_history': self.ship.sway_velocity_history.copy() if self.ship.use_mmg else [],
            'yaw_rate_history': self.ship.yaw_rate_history.copy() if self.ship.use_mmg else [],
            'position_x_history': self.ship.position_x_history.copy() if self.ship.use_mmg else [],
            'position_y_history': self.ship.position_y_history.copy() if self.ship.use_mmg else [],
            'heading_history': self.ship.heading_history.copy() if self.ship.use_mmg else [],
        }
        
        # 导出MMG速度数据到CSV
        if self.ship.use_mmg:
            self._export_mmg_velocity_csv(algo_name)
    
    def _generate_report(self):
        """生成完整对比报告（JSON + 高质量可视化图表）"""
        import matplotlib
        matplotlib.use('Agg')  # 非交互模式
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ========== 1. 保存JSON结果 ==========
        report_path = self.output_dir / "algorithm_comparison.json"
        
        serializable_results = {}
        for algo, data in self.algorithm_results.items():
            serializable_results[algo] = {
                k: (v if not isinstance(v, np.ndarray) else v.tolist())
                for k, v in data.items()
                if k not in ['trajectory', 'collision_details', 'ice_resistance_history',
                            'rudder_angle_history', 'propeller_rpm_history', 'speed_history',
                            'cte_time_s', 'cte_center_series_m', 'cte_tail_series_m',
                            'rudder_rate_series_deg_s']
            }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[Report] JSON saved: {report_path}")

        try:
            self._export_analysis_csvs()
        except Exception as e:
            print(f"   [Warning] CSV export failed: {e}")

        try:
            self._export_publication_summary_csv()
        except Exception as e:
            print(f"   [Warning] Publication summary export failed: {e}")
        
        if len(self.algorithm_results) < 1:
            print("   [Warning] No algorithm results, skip visualization")
            return
        
        # ========== 2. 使用新的可视化模块生成高质量图表 ==========
        try:
            viz = ComparisonVisualizer(str(self.output_dir))

            try:
                viz.set_ship_geometry(float(getattr(self.config, 'SHIP_LENGTH', 80.0)), float(getattr(self.config, 'SHIP_WIDTH', 16.0)))
            except Exception:
                pass

            try:
                outline = [(float(v.x), float(v.y)) for v in self.ship.shape.get_vertices()]
                if outline and len(outline) >= 3:
                    viz.set_ship_outline(outline)
            except Exception:
                pass
            
            # 设置冰场数据
            ice_data = []
            try:
                for ice_block, initial in zip(self.ice_blocks, getattr(self, 'ice_blocks_initial_state', []) or []):
                    pos = initial.get('position', None)
                    if pos is None:
                        continue
                    size = getattr(ice_block, 'size', 10)
                    ice_data.append((float(pos[0]), float(pos[1]), float(size)))
            except Exception:
                ice_data = []

            if not ice_data:
                for ice_block in self.ice_blocks:
                    pos = ice_block.body.position
                    size = getattr(ice_block, 'size', 10)
                    ice_data.append((pos.x, pos.y, size))
            
            # Get start/goal positions (handle different config attribute names)
            start_x = getattr(self.config, 'SHIP_START_X', getattr(self.config, 'START_X', 0.5))
            start_y = getattr(self.config, 'SHIP_START_Y', getattr(self.config, 'START_Y', 0.1))
            goal_x = getattr(self.config, 'GOAL_X', 0.5)
            goal_y = getattr(self.config, 'GOAL_Y', 0.9)
            
            start_pos = (start_x * self.config.WORLD_WIDTH, start_y * self.config.WORLD_HEIGHT)
            goal_pos = (goal_x * self.config.WORLD_WIDTH, goal_y * self.config.WORLD_HEIGHT)
            
            viz.set_ice_field(
                ice_data,
                (self.config.WORLD_WIDTH, self.config.WORLD_HEIGHT),
                start_pos, goal_pos
            )
            
            # 转换算法结果
            for algo_name, data in self.algorithm_results.items():
                result = AlgorithmResult(name=algo_name)
                
                # 轨迹
                result.trajectory = data.get('trajectory', [])
                
                # 碰撞点
                collision_details = data.get('collision_details', [])
                result.collision_points = [
                    (cd.get('contact_point', (0,0))[0], cd.get('contact_point', (0,0))[1])
                    for cd in collision_details if 'contact_point' in cd
                ]
                result.collision_details = collision_details
                
                # 时间序列数据
                ice_hist = data.get('ice_resistance_history', [])
                speed_hist = data.get('speed_history', [])
                journey_time = data.get('journey_time', 0)
                
                if ice_hist:
                    result.ice_resistance_history = ice_hist
                    result.time_history = list(np.linspace(0, journey_time, len(ice_hist)))
                
                if speed_hist:
                    result.speed_history = speed_hist

                rudder_hist = data.get('rudder_angle_history', [])
                if rudder_hist:
                    result.rudder_angle_history = rudder_hist

                rpm_hist = data.get('propeller_rpm_history', [])
                if rpm_hist:
                    result.propeller_rpm_history = rpm_hist
                
                # 统计指标
                result.total_time = journey_time
                result.total_distance = data.get('trajectory_distance', self._calculate_trajectory_length(result.trajectory))
                result.collision_count = data.get('collision_count', 0)
                result.avg_ice_resistance = data.get('avg_resistance', 0)
                result.max_ice_resistance = data.get('max_resistance', 0)
                result.total_ice_resistance = data.get('total_resistance', 0)
                result.avg_speed = np.mean(speed_hist) if speed_hist else 5.0
                
                # 碰撞能量统计 (新增)
                result.total_collision_energy = data.get('total_collision_energy', 0)
                result.avg_collision_energy = data.get('avg_collision_energy', 0)
                result.max_collision_energy = data.get('max_collision_energy', 0)
                
                ice_work_mj = data.get('ice_work_mj', 0.0)
                result.energy_consumption = float(ice_work_mj)

                result.propulsive_energy_mj = float(data.get('propulsive_energy_mj', 0.0))

                result.rudder_rate_mean_abs_deg_s = float(data.get('rudder_rate_mean_abs_deg_s', 0.0))
                result.rudder_rate_rms_deg_s = float(data.get('rudder_rate_rms_deg_s', 0.0))
                result.rudder_rate_max_abs_deg_s = float(data.get('rudder_rate_max_abs_deg_s', 0.0))
                result.cte_time_s = data.get('cte_time_s', [])
                result.cte_center_series_m = data.get('cte_center_series_m', [])
                result.cte_tail_series_m = data.get('cte_tail_series_m', [])
                result.cte_center_rms_m = float(data.get('cte_center_rms_m', 0.0))
                result.cte_tail_rms_m = float(data.get('cte_tail_rms_m', 0.0))
                result.risk_metrics = data.get('risk_metrics', {})

                safety_min = float(data.get('safety_min_clearance_m', 0.0))
                if safety_min > 0:
                    result.safety_score = float(np.tanh(safety_min / max(1.0, float(getattr(self.config, 'SHIP_WIDTH', 30.0)))))
                else:
                    result.safety_score = 1.0 / (1.0 + result.collision_count * 0.1)

                rudder_change_sum = float(data.get('rudder_change_sum_deg', 0.0))
                if rudder_change_sum > 0:
                    result.smoothness_score = 1.0 / (1.0 + rudder_change_sum / 360.0)
                else:
                    curvature_sum = float(data.get('curvature_sum', 0.0))
                    result.smoothness_score = 1.0 / (1.0 + curvature_sum)
                
                viz.add_result(algo_name, result)
            
            # 生成所有图表
            viz.generate_all()
            
        except Exception as e:
            print(f"   [Error] Visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        # ========== 3. 打印对比摘要 ==========
        algos = list(self.algorithm_results.keys())
        times = [self.algorithm_results[a].get('journey_time', 0) for a in algos]
        collisions = [self.algorithm_results[a].get('collision_count', 0) for a in algos]
        
        print("\n" + "="*100)
        print("Algorithm Comparison Summary")
        print("="*100)
        print(f"{'Algorithm':<15} {'Time(s)':<10} {'Collisions':<10} {'Collision E(MJ)':<16} {'Prop E(MJ)':<12} {'Ice Work(MJ)':<12} {'Total E(MJ)':<12}")
        print("-"*100)
        for algo in algos:
            data = self.algorithm_results[algo]
            t = data.get('journey_time', 0)
            c = data.get('collision_count', 0)
            coll_e = data.get('total_collision_energy', 0) / 1e6  # MJ
            r_total = data.get('total_resistance', 0) / 1e9  # GN
            ice_work = float(data.get('ice_work_mj', 0.0))
            prop_e = float(data.get('propulsive_energy_mj', 0.0))
            total_e = prop_e + ice_work
            print(f"{algo:<15} {t:<10.1f} {c:<10} {coll_e:<16.2f} {prop_e:<12.2f} {ice_work:<12.2f} {total_e:<12.2f}")
        print("="*100)
        
        # 找出最优算法 (考虑时间、碰撞、能量)
        if len(algos) > 1:
            max_time = max(times) if max(times) > 0 else 1
            max_coll = max(collisions) if max(collisions) > 0 else 1
            coll_energies = [self.algorithm_results[a].get('total_collision_energy', 0) for a in algos]
            max_coll_e = max(coll_energies) if max(coll_energies) > 0 else 1
            
            best_algo = min(algos, key=lambda a: (
                0.3 * self.algorithm_results[a].get('journey_time', 999) / max_time +
                0.3 * self.algorithm_results[a].get('collision_count', 999) / max_coll +
                0.4 * self.algorithm_results[a].get('total_collision_energy', 1e9) / max_coll_e
            ))
            print(f"\n[Best Algorithm]: {best_algo} (考虑时间30%+碰撞次数30%+碰撞能量40%)")
        
        print(f"\n[Output Directory]: {self.output_dir.absolute()}")

    def _export_mmg_velocity_csv(self, algo_name: str):
        """导出MMG速度分量数据到CSV并生成可视化图片"""
        import csv
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取数据
        time_data = self.ship.time_history
        u_data = self.ship.surge_velocity_history      # 纵荡速度
        v_data = self.ship.sway_velocity_history       # 横荡速度
        r_data = self.ship.yaw_rate_history            # 艏摇角速度
        x_data = self.ship.position_x_history          # x位置
        y_data = self.ship.position_y_history          # y位置
        heading_data = self.ship.heading_history       # 航向角
        speed_data = self.ship.speed_history           # 合速度
        rudder_data = self.ship.rudder_angle_history   # 舵角
        rpm_data = self.ship.propeller_rpm_history     # 螺旋桨转速
        
        if not time_data:
            print(f"   [Warning] No MMG data to export for {algo_name}")
            return
        
        # 确保所有数据长度一致（取最小长度）
        min_len = min(len(time_data), len(u_data), len(v_data), len(r_data))
        if min_len == 0:
            return
        
        # ========== 1. 导出CSV ==========
        csv_path = self.output_dir / f"mmg_velocity_{algo_name.replace('*', '_star')}.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow([
                'time_s',           # 时间(s)
                'x_m',              # x位置(m)
                'y_m',              # y位置(m)
                'u_m_s',            # 纵荡速度(m/s)
                'v_m_s',            # 横荡速度(m/s)
                'r_rad_s',          # 艏摇角速度(rad/s)
                'r_deg_s',          # 艏摇角速度(deg/s)
                'speed_m_s',        # 合速度(m/s)
                'speed_kn',         # 合速度(kn)
                'heading_deg',      # 航向角(deg)
                'rudder_deg',       # 舵角(deg)
                'rpm',              # 螺旋桨转速
            ])
            
            # 写入数据
            for i in range(min_len):
                t = time_data[i] if i < len(time_data) else 0
                x = x_data[i] if i < len(x_data) else 0
                y = y_data[i] if i < len(y_data) else 0
                u = u_data[i] if i < len(u_data) else 0
                v = v_data[i] if i < len(v_data) else 0
                r = r_data[i] if i < len(r_data) else 0
                spd = speed_data[i] if i < len(speed_data) else 0
                hdg = heading_data[i] if i < len(heading_data) else 0
                rud = rudder_data[i] if i < len(rudder_data) else 0
                rpm = rpm_data[i] if i < len(rpm_data) else 0
                
                writer.writerow([
                    f"{t:.3f}",
                    f"{x:.2f}",
                    f"{y:.2f}",
                    f"{u:.4f}",
                    f"{v:.4f}",
                    f"{r:.6f}",
                    f"{np.degrees(r):.4f}",
                    f"{spd:.4f}",
                    f"{spd * 1.944:.4f}",  # m/s to knots
                    f"{hdg:.2f}",
                    f"{rud:.2f}",
                    f"{rpm:.2f}",
                ])
        
        print(f"   [CSV] MMG velocity data exported: {csv_path}")
        
        # ========== 2. 生成可视化图片 ==========
        try:
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
            fig.suptitle(f'MMG Dynamics Data - {algo_name}', fontsize=14, fontweight='bold')
            
            time_arr = np.array(time_data[:min_len])
            
            # (0,0) 纵荡速度 u
            ax = axes[0, 0]
            ax.plot(time_arr, u_data[:min_len], 'b-', linewidth=1.2, label='u (surge)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Surge Velocity u (m/s)')
            ax.set_title('Surge Velocity')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # (0,1) 横荡速度 v
            ax = axes[0, 1]
            ax.plot(time_arr, v_data[:min_len], 'r-', linewidth=1.2, label='v (sway)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Sway Velocity v (m/s)')
            ax.set_title('Sway Velocity')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # (1,0) 艏摇角速度 r
            ax = axes[1, 0]
            r_deg = np.degrees(np.array(r_data[:min_len]))
            ax.plot(time_arr, r_deg, 'g-', linewidth=1.2, label='r (yaw rate)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Yaw Rate r (deg/s)')
            ax.set_title('Yaw Rate')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # (1,1) 合速度
            ax = axes[1, 1]
            speed_kn = np.array(speed_data[:min_len]) * 1.944
            ax.plot(time_arr, speed_kn, 'm-', linewidth=1.2, label='Speed')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Speed (knots)')
            ax.set_title('Ship Speed')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # (2,0) 舵角
            ax = axes[2, 0]
            ax.plot(time_arr, rudder_data[:min_len], 'c-', linewidth=1.2, label='Rudder')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Rudder Angle (deg)')
            ax.set_title('Rudder Angle')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # (2,1) 航向角
            ax = axes[2, 1]
            ax.plot(time_arr, heading_data[:min_len], 'orange', linewidth=1.2, label='Heading')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Heading (deg)')
            ax.set_title('Heading Angle')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            img_path = self.output_dir / f"mmg_velocity_{algo_name.replace('*', '_star')}.png"
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"   [IMG] MMG velocity plot saved: {img_path}")
            
            # ========== 3. 生成速度分量对比图 (u, v, r 在同一张图) ==========
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.set_title(f'MMG Velocity Components - {algo_name}', fontsize=12, fontweight='bold')
            
            ax2.plot(time_arr, u_data[:min_len], 'b-', linewidth=1.5, label='u - Surge (m/s)')
            ax2.plot(time_arr, v_data[:min_len], 'r-', linewidth=1.5, label='v - Sway (m/s)')
            
            # 为r创建第二个y轴（因为单位不同）
            ax2_r = ax2.twinx()
            ax2_r.plot(time_arr, r_deg, 'g--', linewidth=1.5, label='r - Yaw rate (deg/s)')
            ax2_r.set_ylabel('Yaw Rate r (deg/s)', color='g')
            ax2_r.tick_params(axis='y', labelcolor='g')
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.grid(True, alpha=0.3)
            
            # 合并图例
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_r.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.tight_layout()
            
            img_path2 = self.output_dir / f"mmg_uvr_combined_{algo_name.replace('*', '_star')}.png"
            plt.savefig(img_path2, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            
            print(f"   [IMG] MMG u-v-r combined plot saved: {img_path2}")
            
        except Exception as e:
            print(f"   [Warning] Failed to generate MMG plots: {e}")

    def _export_publication_summary_csv(self):
        out_path = self.output_dir / "publication_metrics_summary.csv"
        algos = list(self.algorithm_results.keys())
        if not algos:
            return

        def _risk_get(risk: dict, key: str, subkey: str, default=0.0):
            try:
                return float((risk.get(key, {}) or {}).get(subkey, default))
            except Exception:
                return float(default)

        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                'algorithm',
                'journey_time_s',
                'trajectory_distance_m',
                'ice_work_mj',
                'propulsive_energy_mj',
                'operational_energy_mj',
                'collision_energy_mj',
                'collision_count',
                'rudder_rate_rms_deg_s',
                'cte_center_rms_m',
                'cte_tail_rms_m',
                'risk_bow_count', 'risk_bow_energy_j', 'risk_bow_impulse',
                'risk_stern_count', 'risk_stern_energy_j', 'risk_stern_impulse',
                'risk_prop_count', 'risk_prop_energy_j', 'risk_prop_impulse',
                'risk_rudder_count', 'risk_rudder_energy_j', 'risk_rudder_impulse',
            ])

            for algo in algos:
                d = self.algorithm_results.get(algo, {}) or {}
                risk = d.get('risk_metrics', {}) or {}

                t = float(d.get('journey_time', 0.0))
                dist = float(d.get('trajectory_distance', 0.0))
                ice_work = float(d.get('ice_work_mj', 0.0))
                prop_e_mj = float(d.get('propulsive_energy_mj', 0.0))
                operational_e_mj = ice_work + prop_e_mj
                coll_e_mj = float(d.get('total_collision_energy', 0.0)) / 1e6
                coll_n = int(d.get('collision_count', 0))

                w.writerow([
                    algo,
                    t,
                    dist,
                    ice_work,
                    prop_e_mj,
                    operational_e_mj,
                    coll_e_mj,
                    coll_n,
                    float(d.get('rudder_rate_rms_deg_s', 0.0)),
                    float(d.get('cte_center_rms_m', 0.0)),
                    float(d.get('cte_tail_rms_m', 0.0)),
                    int(_risk_get(risk, 'bow', 'count', 0)), _risk_get(risk, 'bow', 'energy_sum', 0.0), _risk_get(risk, 'bow', 'impulse_sum', 0.0),
                    int(_risk_get(risk, 'stern', 'count', 0)), _risk_get(risk, 'stern', 'energy_sum', 0.0), _risk_get(risk, 'stern', 'impulse_sum', 0.0),
                    int(_risk_get(risk, 'propeller', 'count', 0)), _risk_get(risk, 'propeller', 'energy_sum', 0.0), _risk_get(risk, 'propeller', 'impulse_sum', 0.0),
                    int(_risk_get(risk, 'rudder', 'count', 0)), _risk_get(risk, 'rudder', 'energy_sum', 0.0), _risk_get(risk, 'rudder', 'impulse_sum', 0.0),
                ])
    
    def _calculate_trajectory_length(self, trajectory: list) -> float:
        """计算轨迹总长度"""
        if not trajectory or len(trajectory) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(trajectory)):
            try:
                x1 = float(trajectory[i][0])
                y1 = float(trajectory[i][1])
                x0 = float(trajectory[i-1][0])
                y0 = float(trajectory[i-1][1])
            except Exception:
                continue
            dx = x1 - x0
            dy = y1 - y0
            total += np.sqrt(dx*dx + dy*dy)
        return total

    def _calculate_rudder_change_sum(self) -> float:
        if not getattr(self.ship, 'use_mmg', False):
            return 0.0
        hist = getattr(self.ship, 'rudder_angle_history', None) or []
        if len(hist) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(hist)):
            total += abs(float(hist[i]) - float(hist[i-1]))
        return float(total)

    def _calculate_rudder_rate_metrics(self):
        if not getattr(self.ship, 'use_mmg', False):
            return 0.0, 0.0, 0.0, []
        hist = getattr(self.ship, 'rudder_angle_history', None) or []
        if len(hist) < 2:
            return 0.0, 0.0, 0.0, []
        dt = float(getattr(self.config, 'PHYSICS_DT', 1/60.0))
        if dt <= 0:
            dt = 1/60.0
        rates = []
        for i in range(1, len(hist)):
            rates.append((float(hist[i]) - float(hist[i-1])) / dt)
        if not rates:
            return 0.0, 0.0, 0.0, []
        abs_rates = [abs(r) for r in rates]
        mean_abs = float(np.mean(abs_rates))
        rms = float(np.sqrt(np.mean(np.square(abs_rates))))
        max_abs = float(np.max(abs_rates))
        return mean_abs, rms, max_abs, rates

    def _point_to_segment_distance(self, px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-12:
            return float(math.hypot(apx, apy))
        t = (apx * abx + apy * aby) / denom
        if t <= 0.0:
            cx, cy = ax, ay
        elif t >= 1.0:
            cx, cy = bx, by
        else:
            cx = ax + t * abx
            cy = ay + t * aby
        return float(math.hypot(px - cx, py - cy))

    def _distance_to_polyline(self, px: float, py: float, polyline: List[Tuple[float, float]]) -> float:
        if not polyline or len(polyline) < 2:
            return 0.0
        best = float('inf')
        for i in range(1, len(polyline)):
            ax, ay = float(polyline[i-1][0]), float(polyline[i-1][1])
            bx, by = float(polyline[i][0]), float(polyline[i][1])
            d = self._point_to_segment_distance(px, py, ax, ay, bx, by)
            if d < best:
                best = d
        if best == float('inf'):
            return 0.0
        return float(best)

    def _calculate_path_following_metrics(self, trajectory: list, planned_path: List[Tuple[float, float]]):
        if not trajectory or len(trajectory) < 2 or not planned_path or len(planned_path) < 2:
            return {}
        L = float(getattr(self.config, 'SHIP_LENGTH', 80.0))
        tail_offset = 0.5 * L
        xs = []
        ys = []
        angs = []
        for p in trajectory:
            try:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
                if len(p) >= 3:
                    angs.append(float(p[2]))
                else:
                    angs.append(0.0)
            except Exception:
                continue
        n = min(len(xs), len(ys), len(angs))
        if n < 2:
            return {}
        cte_center = []
        cte_tail = []
        for i in range(n):
            x = xs[i]
            y = ys[i]
            a = angs[i]
            cte_center.append(self._distance_to_polyline(x, y, planned_path))
            tx = x - math.cos(a) * tail_offset
            ty = y - math.sin(a) * tail_offset
            cte_tail.append(self._distance_to_polyline(tx, ty, planned_path))
        c_arr = np.asarray(cte_center, dtype=np.float64)
        t_arr = np.asarray(cte_tail, dtype=np.float64)
        journey_time = float(getattr(self, 'simulation_time', 0.0) - getattr(self, 'journey_start_time', 0.0))
        journey_time = max(0.0, float(journey_time))
        time_s = list(np.linspace(0.0, journey_time, len(c_arr)))
        return {
            'cte_center_mean_m': float(np.mean(c_arr)),
            'cte_center_rms_m': float(np.sqrt(np.mean(c_arr ** 2))),
            'cte_center_p95_m': float(np.percentile(c_arr, 95)),
            'cte_center_max_m': float(np.max(c_arr)),
            'cte_tail_mean_m': float(np.mean(t_arr)),
            'cte_tail_rms_m': float(np.sqrt(np.mean(t_arr ** 2))),
            'cte_tail_p95_m': float(np.percentile(t_arr, 95)),
            'cte_tail_max_m': float(np.max(t_arr)),
            'cte_time_s': time_s,
            'cte_center_series_m': [float(v) for v in c_arr.tolist()],
            'cte_tail_series_m': [float(v) for v in t_arr.tolist()],
        }

    def _calculate_collision_risk_metrics(self, collision_details: list):
        L = float(getattr(self.config, 'SHIP_LENGTH', 80.0))
        B = float(getattr(self.config, 'SHIP_WIDTH', 16.0))
        if not collision_details:
            return {
                'bow': {'count': 0, 'impulse_sum': 0.0, 'energy_sum': 0.0, 'impulse_max': 0.0, 'energy_max': 0.0},
                'stern': {'count': 0, 'impulse_sum': 0.0, 'energy_sum': 0.0, 'impulse_max': 0.0, 'energy_max': 0.0},
                'prop': {'count': 0, 'impulse_sum': 0.0, 'energy_sum': 0.0, 'impulse_max': 0.0, 'energy_max': 0.0},
                'rudder': {'count': 0, 'impulse_sum': 0.0, 'energy_sum': 0.0, 'impulse_max': 0.0, 'energy_max': 0.0},
            }

        def init_bin():
            return {'count': 0, 'impulse_sum': 0.0, 'energy_sum': 0.0, 'impulse_max': 0.0, 'energy_max': 0.0}

        bins = {'bow': init_bin(), 'stern': init_bin(), 'prop': init_bin(), 'rudder': init_bin()}
        for cd in collision_details:
            try:
                lx = float(cd.get('local_x', 0.0))
                ly = float(cd.get('local_y', 0.0))
                imp = float(cd.get('force', 0.0))
                eng = float(cd.get('collision_energy', 0.0))
            except Exception:
                continue

            if lx > 0.3 * L:
                key = 'bow'
            elif lx < -0.3 * L:
                key = 'stern'
            else:
                key = None

            if key is not None:
                b = bins[key]
                b['count'] += 1
                b['impulse_sum'] += imp
                b['energy_sum'] += eng
                b['impulse_max'] = max(b['impulse_max'], imp)
                b['energy_max'] = max(b['energy_max'], eng)

            if lx < -0.4 * L:
                if abs(ly) < 0.2 * B:
                    b = bins['prop']
                    b['count'] += 1
                    b['impulse_sum'] += imp
                    b['energy_sum'] += eng
                    b['impulse_max'] = max(b['impulse_max'], imp)
                    b['energy_max'] = max(b['energy_max'], eng)
                elif abs(ly) < 0.6 * B:
                    b = bins['rudder']
                    b['count'] += 1
                    b['impulse_sum'] += imp
                    b['energy_sum'] += eng
                    b['impulse_max'] = max(b['impulse_max'], imp)
                    b['energy_max'] = max(b['energy_max'], eng)

        return bins

    def _export_analysis_csvs(self):
        if not self.algorithm_results:
            return

        for algo, data in self.algorithm_results.items():
            journey_time = float(data.get('journey_time', 0.0))
            dt = float(getattr(self.config, 'PHYSICS_DT', 1/60.0))
            if dt <= 0:
                dt = 1/60.0

            ice_hist = data.get('ice_resistance_history', []) or []
            speed_hist = data.get('speed_history', []) or []
            rudder_hist = data.get('rudder_angle_history', []) or []
            rpm_hist = data.get('propeller_rpm_history', []) or []
            rudder_rate = data.get('rudder_rate_series_deg_s', []) or []
            cte_time = data.get('cte_time_s', []) or []
            cte_center = data.get('cte_center_series_m', []) or []
            cte_tail = data.get('cte_tail_series_m', []) or []

            n = min(len(ice_hist), len(speed_hist), len(rudder_hist), len(rpm_hist))
            if n > 0:
                out_path = self.output_dir / f"timeseries_{algo}.csv"
                with open(out_path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['t_s', 'speed_mps', 'rudder_deg', 'rudder_rate_deg_s', 'prop_rpm', 'ice_resistance_N', 'cte_center_m', 'cte_tail_m'])
                    for i in range(n):
                        t = i * dt
                        rr = float(rudder_rate[i]) if i < len(rudder_rate) else ''
                        c0 = float(cte_center[i]) if i < len(cte_center) else ''
                        c1 = float(cte_tail[i]) if i < len(cte_tail) else ''
                        w.writerow([t, float(speed_hist[i]), float(rudder_hist[i]), rr, float(rpm_hist[i]), float(ice_hist[i]), c0, c1])

            collision_details = data.get('collision_details', []) or []
            if collision_details:
                out_path = self.output_dir / f"collisions_{algo}.csv"
                with open(out_path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['time_s', 'local_x_m', 'local_y_m', 'impulse', 'impulse_local_x', 'impulse_local_y', 'collision_energy_J', 'ice_mass_kg', 'relative_velocity_mps'])
                    for cd in collision_details:
                        w.writerow([
                            cd.get('time', ''),
                            cd.get('local_x', ''),
                            cd.get('local_y', ''),
                            cd.get('force', ''),
                            cd.get('impulse_local_x', ''),
                            cd.get('impulse_local_y', ''),
                            cd.get('collision_energy', ''),
                            cd.get('ice_mass', ''),
                            cd.get('relative_velocity', ''),
                        ])

    def _calculate_curvature_metrics(self, trajectory: list) -> tuple:
        if not trajectory or len(trajectory) < 3:
            return 0.0, 0.0

        xy = []
        for p in trajectory:
            try:
                xy.append((float(p[0]), float(p[1])))
            except Exception:
                continue
        if len(xy) < 3:
            return 0.0, 0.0

        pts = np.asarray(xy, dtype=np.float64)
        curv_sum = 0.0
        curv_max = 0.0
        for i in range(1, len(pts) - 1):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            v1x = x1 - x0
            v1y = y1 - y0
            v2x = x2 - x1
            v2y = y2 - y1
            l1 = np.hypot(v1x, v1y)
            l2 = np.hypot(v2x, v2y)
            if l1 < 1e-6 or l2 < 1e-6:
                continue
            a1 = np.arctan2(v1y, v1x)
            a2 = np.arctan2(v2y, v2x)
            da = np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1))
            ds = 0.5 * (l1 + l2)
            kappa = abs(da) / max(1e-6, ds)
            curv_sum += kappa * ds
            curv_max = max(curv_max, kappa)
        return float(curv_sum), float(curv_max)

    def _calculate_safety_metrics(self, trajectory: list) -> tuple:
        if not trajectory or len(trajectory) < 1:
            return 0.0, 0.0
        if not hasattr(self, 'spatial_grid') or not self.spatial_grid:
            return 0.0, 0.0
        sample_step = 10
        min_clear = float('inf')
        sum_clear = 0.0
        cnt = 0
        scan_r = float(getattr(self.config, 'SAFETY_SCAN_RADIUS', 300.0))
        for i in range(0, len(trajectory), sample_step):
            try:
                x = float(trajectory[i][0])
                y = float(trajectory[i][1])
            except Exception:
                continue
            nearby = self._get_nearby_ice_fast(float(x), float(y), scan_r)
            local_min = float('inf')
            for ice in nearby:
                pos = ice.body.position
                size = float(getattr(ice, 'size', 10.0))
                d = float(np.hypot(float(pos.x) - float(x), float(pos.y) - float(y)) - (size * 0.5))
                if d < local_min:
                    local_min = d
            if local_min != float('inf'):
                min_clear = min(min_clear, local_min)
                sum_clear += local_min
                cnt += 1
        if cnt == 0 or min_clear == float('inf'):
            return 0.0, 0.0
        return float(max(0.0, min_clear)), float(max(0.0, sum_clear / cnt))

    def _calculate_ice_work_mj(self) -> float:
        r_hist = getattr(self.ship, 'ice_resistance_history', None) or []
        if not r_hist:
            return 0.0
        dt = float(getattr(self.config, 'PHYSICS_DT', 1/60.0))
        if getattr(self.ship, 'use_mmg', False):
            v_hist = getattr(self.ship, 'speed_history', None) or []
            n = min(len(r_hist), len(v_hist))
            if n <= 1:
                return 0.0
            work = 0.0
            for i in range(n):
                work += float(r_hist[i]) * max(0.0, float(v_hist[i])) * dt
            return float(work / 1e6)
        else:
            traj = getattr(self.ship, 'trajectory', None) or []
            dist = self._calculate_trajectory_length(traj)
            avg_r = float(np.mean(r_hist)) if r_hist else 0.0
            return float((avg_r * dist) / 1e6) if avg_r > 0 else 0.0

    def _calculate_propulsive_energy_mj(self) -> float:
        if not getattr(self.ship, 'use_mmg', False):
            return 0.0
        try:
            rpm_hist = getattr(self.ship, 'propeller_rpm_history', None) or []
            if not rpm_hist:
                return 0.0
            dt = float(getattr(self.config, 'PHYSICS_DT', 1/60.0))
            params = getattr(getattr(self.ship, 'dynamics', None), 'params', None)
            if params is None:
                return 0.0
            p_max_kw = float(getattr(params, 'propulsion_power', 0.0))
            max_rpm = float(getattr(params, 'max_rpm', 0.0))
            if p_max_kw <= 0.0 or max_rpm <= 1e-9:
                return 0.0
            e_j = 0.0
            for rpm in rpm_hist:
                frac = min(1.0, abs(float(rpm)) / max_rpm)
                p_w = (p_max_kw * 1000.0) * (frac ** 3)
                e_j += max(0.0, p_w) * dt
            return float(e_j / 1e6)
        except Exception:
            return 0.0
    
    def _draw(self):
        """绘制（带摄像机系统 + 性能优化）"""
        self.screen.fill(self.config.COLOR_WATER)
        
        # 基础缩放比例
        scale_x = self.config.GAME_AREA_WIDTH / self.config.WORLD_WIDTH
        scale_y = self.config.WINDOW_HEIGHT / self.config.WORLD_HEIGHT
        if self.camera_zoom <= 1.0 and (not self.camera_follow_ship):
            base_scale = max(scale_x, scale_y)
        else:
            base_scale = min(scale_x, scale_y)
        
        # ========== 确定视图中心 ==========
        # 放大时跟随船舶，缩小时看全局
        if self.camera_zoom > 1.0 or self.camera_follow_ship:
            ship_pos = self.ship.body.position
            view_center_x, view_center_y = ship_pos.x, ship_pos.y
            # 同时更新camera_center供其他函数使用
            self.camera_center = (ship_pos.x, ship_pos.y)
        elif self.camera_center:
            view_center_x, view_center_y = self.camera_center
        else:
            view_center_x = self.config.WORLD_WIDTH / 2
            view_center_y = self.config.WORLD_HEIGHT / 2
            self.camera_center = (view_center_x, view_center_y)
        
        # ========== 计算可见区域（世界坐标）==========
        effective_scale = base_scale * self.camera_zoom
        # 可见范围 = 屏幕尺寸 / 缩放 + 超大边距（防止冰块消失）
        view_half_w = (self.config.GAME_AREA_WIDTH / effective_scale) / 2 + 500
        view_half_h = (self.config.WINDOW_HEIGHT / effective_scale) / 2 + 500
        
        # 绘制冰块（只绘制视野内的）
        drawn_count = 0
        for ice_block in self.ice_blocks:
            pos = ice_block.body.position
            # 跳过无效位置
            if np.isnan(pos.x) or np.isnan(pos.y):
                continue
            # 边界检查（只绘制视野内）
            if abs(pos.x - view_center_x) > view_half_w or abs(pos.y - view_center_y) > view_half_h:
                continue
            self._draw_ice_block(ice_block, base_scale)
            drawn_count += 1
        
        # 绘制路径
        self._draw_path(base_scale)
        
        # 绘制船舶
        self._draw_ship(base_scale)
        
        # 绘制目标
        self._draw_goal(base_scale)
        
        # ===== UI层（固定位置，不受摄像机影响）=====
        # 右侧控制面板（最先绘制，作为背景）
        self._draw_control_panel()
        
        # 左下角简洁信息
        self._draw_info_overlay()
        
        # 缩放指示器
        self._draw_zoom_indicator()
        
        pygame.display.flip()
    
    def _draw_ice_block(self, ice_block, base_scale):
        """绘制冰块（带摄像机变换 + 性能优化）"""
        vertices = ice_block.shape.get_vertices()
        world_vertices = [ice_block.body.local_to_world(v) for v in vertices]
        
        # 使用摄像机变换
        screen_vertices = [self._world_to_screen(v.x, v.y, base_scale) for v in world_vertices]
        
        # 快速边界检查
        xs = [v[0] for v in screen_vertices]
        ys = [v[1] for v in screen_vertices]
        if max(xs) < 0 or min(xs) > self.config.GAME_AREA_WIDTH:
            return
        if max(ys) < 0 or min(ys) > self.config.WINDOW_HEIGHT:
            return
        
        # 性能优化：小冰块简化渲染
        size = max(xs) - min(xs)
        color = self.config.COLOR_ICE.get(ice_block.ice_type, (180, 200, 220))
        
        if self.performance_mode and size < 8:
            # 太小的冰块用矩形近似
            cx = int((max(xs) + min(xs)) / 2)
            cy = int((max(ys) + min(ys)) / 2)
            pygame.draw.circle(self.screen, color, (cx, cy), max(2, int(size/2)))
        else:
            pygame.draw.polygon(self.screen, color, screen_vertices)
            # 性能模式下跳过边框
            if not self.performance_mode:
                pygame.draw.polygon(self.screen, self.config.ICE_OUTLINE_COLOR, screen_vertices, 1)
    
    def _draw_path(self, base_scale):
        """绘制路径（带摄像机变换）"""
        if len(self.path) < 2:
            return
        
        # 已走过的路径
        visited_points = [self._world_to_screen(p[0], p[1], base_scale) 
                         for p in self.path[:self.ship.path_index + 1]]
        # 过滤屏幕外的点
        visited_points = [(x, y) for x, y in visited_points 
                         if -100 < x < self.config.GAME_AREA_WIDTH + 100]
        if len(visited_points) >= 2:
            pygame.draw.lines(self.screen, self.config.COLOR_PATH_VISITED, False, visited_points, 3)
        
        # 未走的路径
        remaining_points = [self._world_to_screen(p[0], p[1], base_scale) 
                           for p in self.path[self.ship.path_index:]]
        remaining_points = [(x, y) for x, y in remaining_points 
                           if -100 < x < self.config.GAME_AREA_WIDTH + 100]
        if len(remaining_points) >= 2:
            pygame.draw.lines(self.screen, self.config.COLOR_PATH, False, remaining_points, 2)
    
    def _draw_ship(self, base_scale):
        """绘制船舶（带摄像机变换）"""
        world_x, world_y = self.ship.body.position
        ship_x, ship_y = self._world_to_screen(world_x, world_y, base_scale)
        
        # 根据缩放调整标记大小
        marker_scale = max(1.0, self.camera_zoom * 0.5)
        
        # 外层：绿色圈（小脉冲）
        pulse = int(2 * np.sin(self.simulation_time * 3) * marker_scale) + int(10 * marker_scale)
        pygame.draw.circle(self.screen, (0, 255, 0), (ship_x, ship_y), pulse, 2)
        # 中层：红色外圈
        pygame.draw.circle(self.screen, (255, 50, 50), (ship_x, ship_y), int(6 * marker_scale), 2)
        # 内层：黄色实心
        pygame.draw.circle(self.screen, (255, 255, 0), (ship_x, ship_y), int(4 * marker_scale))
        # 中心点：黑色
        pygame.draw.circle(self.screen, (0, 0, 0), (ship_x, ship_y), int(2 * marker_scale))
        
        # 绘制船舶多边形（使用摄像机变换）
        if 0 <= ship_x < self.config.GAME_AREA_WIDTH and 0 <= ship_y < self.config.WINDOW_HEIGHT:
            vertices = self.ship.shape.get_vertices()
            world_vertices = [self.ship.body.local_to_world(v) for v in vertices]
            screen_vertices = [self._world_to_screen(v.x, v.y, base_scale) for v in world_vertices]
            
            if len(screen_vertices) >= 3:
                pygame.draw.polygon(self.screen, self.config.COLOR_SHIP, screen_vertices)
                pygame.draw.polygon(self.screen, (255, 0, 0), screen_vertices, 2)
        
        # 绘制航向指示线
        angle = self.ship.body.angle
        arrow_len = int(20 * marker_scale)
        end_x = ship_x + int(arrow_len * np.cos(angle))
        end_y = ship_y - int(arrow_len * np.sin(angle))
        pygame.draw.line(self.screen, (255, 0, 0), (ship_x, ship_y), (end_x, end_y), 3)
    
    def _draw_goal(self, base_scale):
        """绘制目标（带摄像机变换）"""
        goal_world_x = self.config.GOAL_X * self.config.WORLD_WIDTH
        goal_world_y = self.config.GOAL_Y * self.config.WORLD_HEIGHT
        goal_x, goal_y = self._world_to_screen(goal_world_x, goal_world_y, base_scale)
        
        # 根据缩放调整大小
        size = int(8 * max(1, self.camera_zoom * 0.5))
        pygame.draw.circle(self.screen, (0, 200, 0), (goal_x, goal_y), size)
        pygame.draw.circle(self.screen, (0, 100, 0), (goal_x, goal_y), size, 2)
    
    def _draw_info_overlay(self):
        """绘制左上角简洁信息覆盖层"""
        # 半透明背景
        overlay = pygame.Surface((190, 125))
        overlay.set_alpha(200)
        overlay.fill((255, 255, 255))
        self.screen.blit(overlay, (5, 5))
        
        x, y = 10, 10
        line_height = 18
        
        mode_str = 'AUTO' if self.control_mode == 'auto' else 'MANUAL'
        speed_kn = self.ship.body.velocity.length * 1.944  # m/s to knots
        
        # 重规划状态
        if self.enable_dynamic_replan:
            next_replan = self.replan_interval - (self.simulation_time - self.last_replan_time)
            replan_str = f"Replan:{self._replan_count}({next_replan:.0f}s)"
        else:
            replan_str = "Replan:OFF"
        
        # 倍速显示
        speed_str = f"x{self.speed_multiplier}" if self.speed_multiplier > 1 else ""
        
        barrier_n = int(getattr(self, '_mid_route_barrier_collision_count', 0) or 0)
        barrier_limit = int(getattr(self.config, 'MID_ROUTE_BARRIER_COLLISION_FAIL_COUNT', 0) or 0)
        barrier_on = bool(getattr(self, '_mid_route_barrier', None))
        barrier_str = f"Barrier:{'ON' if barrier_on else 'OFF'} {barrier_n}/{barrier_limit}" if barrier_limit > 0 else f"Barrier:{'ON' if barrier_on else 'OFF'}"

        info_lines = [
            f"[{mode_str}] {self.config.PATH_TYPE} {speed_str}",
            f"Time: {self.simulation_time:.1f}s",
            f"Speed: {speed_kn:.1f} kn",
            f"Collisions: {self.ship.collision_count}",
            f"Path: {self.ship.path_index}/{len(self.path)}",
            replan_str,
            barrier_str,
        ]
        
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(text, (x, y + i * line_height))
    
    def _draw_zoom_indicator(self):
        """绘制缩放指示器（左下角）"""
        x, y = 10, self.config.WINDOW_HEIGHT - 35
        
        # 背景
        bg = pygame.Surface((160, 30))
        bg.set_alpha(180)
        bg.fill((50, 50, 50))
        self.screen.blit(bg, (x, y))
        
        # 缩放条
        bar_x = x + 10
        bar_y = y + 10
        bar_width = 100
        bar_height = 10
        
        pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
        
        # 当前位置指示
        progress = self.current_zoom_index / (len(self.zoom_levels) - 1)
        indicator_x = bar_x + int(progress * bar_width)
        pygame.draw.circle(self.screen, (0, 255, 0), (indicator_x, bar_y + bar_height//2), 6)
        
        # 文字
        zoom_text = self.font.render(f"{self.camera_zoom:.1f}x", True, (255, 255, 255))
        self.screen.blit(zoom_text, (bar_x + bar_width + 10, bar_y - 3))
    
    def _draw_info_panel(self):
        """绘制信息面板（英文避免乱码）"""
        x, y = 10, 10
        line_height = 22
        
        mode_str = 'AUTO' if self.control_mode == 'auto' else 'MANUAL'
        info_lines = [
            f"[{mode_str}] Press M to switch",
            f"Algorithm: {self.config.PATH_TYPE}",
            f"Time: {self.simulation_time:.1f}s",
            f"Speed: {self.ship.body.velocity.length:.1f} m/s",
            f"Ice Resist: {self.ship.current_ice_resistance:.0f} N",
            f"Collisions: {self.ship.collision_count}",
            f"Path: {self.ship.path_index}/{len(self.path)}",
        ]
        
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(text, (x, y + i * line_height))
    
    def _draw_dynamics_panel(self):
        """绘制动力学信息面板（MMG模式，英文）"""
        x = self.config.GAME_AREA_WIDTH + 10
        y = 10
        line_height = 20
        
        info = self.ship.dynamics_info
        
        lines = [
            "=== Ship Dynamics ===",
            f"Speed: {info['speed']*1.944:.1f} kn",
            f"Heading: {info['heading']:.1f} deg",
            f"Rudder: {info['rudder_angle']:.1f} deg",
            f"RPM: {info['propeller_rpm']:.0f}",
            f"Turn R: {info['turning_radius']:.0f} m",
            "",
            f"Surge u: {info.get('surge', 0):.2f} m/s",
            f"Sway v: {info.get('sway', 0):.2f} m/s",
            f"Yaw r: {info.get('yaw_rate', 0):.2f} deg/s",
        ]
        
        for i, line in enumerate(lines):
            text = self.font.render(line, True, (0, 0, 100))
            self.screen.blit(text, (x, y + i * line_height))
    
    def _draw_control_panel(self):
        """绘制右侧控制面板（优化布局）"""
        control_x = self.config.GAME_AREA_WIDTH
        panel_width = self.config.WINDOW_WIDTH - self.config.GAME_AREA_WIDTH
        mode_x = control_x + 10
        btn_width = panel_width - 20
        btn_height = 28
        btn_spacing = 6
        
        # 面板背景
        panel_rect = pygame.Rect(control_x, 0, panel_width, self.config.WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, (245, 245, 250), panel_rect)
        pygame.draw.line(self.screen, (180, 180, 180), (control_x, 0), 
                        (control_x, self.config.WINDOW_HEIGHT), 2)
        
        current_y = 10
        
        # ========== 1. 模式切换 ==========
        self.mode_button_rect = pygame.Rect(mode_x, current_y, btn_width, btn_height)
        mode_color = (100, 200, 100) if self.control_mode == "auto" else (100, 100, 200)
        pygame.draw.rect(self.screen, mode_color, self.mode_button_rect)
        pygame.draw.rect(self.screen, (50, 50, 50), self.mode_button_rect, 2)
        mode_text = f"{self.control_mode.upper()} [M]"
        self._draw_button_text(mode_text, self.mode_button_rect, (255, 255, 255))
        current_y += btn_height + 15
        
        # ========== 2. 算法选择 ==========
        self._draw_section_title("ALGORITHM", control_x, panel_width, current_y)
        current_y += 20
        
        algorithms = [("1.A*", "A*"), ("2.Ice-Theta*", "Ice-Theta*"), ("3.Straight", "Straight")]
        for label, algo_type in algorithms:
            btn_rect = pygame.Rect(mode_x, current_y, btn_width, btn_height)
            is_selected = self.config.PATH_TYPE == algo_type
            color = (100, 200, 100) if is_selected else (220, 220, 220)
            pygame.draw.rect(self.screen, color, btn_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), btn_rect, 1)
            self.path_button_rects[algo_type] = btn_rect
            self._draw_button_text(label, btn_rect, (0, 0, 0))
            current_y += btn_height + btn_spacing
        
        # Auto-Run状态
        current_y += 5
        status = "AutoRun:ON" if self.auto_run_algorithms else "AutoRun:OFF"
        color = (0, 150, 0) if self.auto_run_algorithms else (150, 0, 0)
        status_surface = self.font.render(status, True, color)
        self.screen.blit(status_surface, (mode_x, current_y))
        current_y += 25
        
        # ========== 3. 视图控制 ==========
        self._draw_section_title("VIEW", control_x, panel_width, current_y)
        current_y += 20
        
        # 定位到船舶按钮
        self.locate_button_rect = pygame.Rect(mode_x, current_y, btn_width, btn_height)
        pygame.draw.rect(self.screen, (255, 200, 100), self.locate_button_rect)
        pygame.draw.rect(self.screen, (200, 150, 50), self.locate_button_rect, 2)
        self._draw_button_text("Locate Ship [F]", self.locate_button_rect, (0, 0, 0))
        current_y += btn_height + btn_spacing
        
        # 缩放按钮（一行三个）
        small_btn_width = (btn_width - 10) // 3
        zoom_btns = [("-", "zoom_out"), ("1:1", "zoom_reset"), ("+", "zoom_in")]
        for i, (label, name) in enumerate(zoom_btns):
            bx = mode_x + i * (small_btn_width + 5)
            btn_rect = pygame.Rect(bx, current_y, small_btn_width, btn_height)
            pygame.draw.rect(self.screen, (200, 200, 200), btn_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), btn_rect, 1)
            self.zoom_button_rects[name] = btn_rect
            self._draw_button_text(label, btn_rect, (0, 0, 0))
        current_y += btn_height + btn_spacing
        
        # 缩放值显示
        zoom_str = f"Zoom: {self.camera_zoom:.1f}x"
        zoom_surface = self.font.render(zoom_str, True, (50, 50, 150))
        self.screen.blit(zoom_surface, (mode_x, current_y))
        current_y += 20
        
        # ========== 4. 仿真速度控制 ==========
        self._draw_section_title("SPEED", control_x, panel_width, current_y)
        current_y += 18
        
        # 倍速按钮（一行四个）
        speed_btn_width = (btn_width - 15) // 4
        for i, speed in enumerate(self.speed_options):
            bx = mode_x + i * (speed_btn_width + 5)
            btn_rect = pygame.Rect(bx, current_y, speed_btn_width, btn_height)
            
            # 当前选中的高亮
            is_selected = self.speed_multiplier == speed
            color = (100, 200, 100) if is_selected else (200, 200, 200)
            pygame.draw.rect(self.screen, color, btn_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), btn_rect, 1)
            
            self.speed_button_rects[speed] = btn_rect
            self._draw_button_text(f"{speed}x", btn_rect, (0, 0, 0))
        current_y += btn_height + btn_spacing
        
        # ========== 5. 中途冰山/冰岸开关 ===========
        current_y += 8
        self._draw_section_title("BARRIER", control_x, panel_width, current_y)
        current_y += 18

        self.barrier_button_rect = pygame.Rect(mode_x, current_y, btn_width, btn_height)
        barrier_on = bool(getattr(self, '_mid_route_barrier', None))
        bcolor = (200, 120, 60) if barrier_on else (220, 220, 220)
        pygame.draw.rect(self.screen, bcolor, self.barrier_button_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), self.barrier_button_rect, 1)
        self._draw_button_text(f"Mid Barrier [{'ON' if barrier_on else 'OFF'}] [B]", self.barrier_button_rect, (0, 0, 0))
        current_y += btn_height + 2

        barrier_n = int(getattr(self, '_mid_route_barrier_collision_count', 0) or 0)
        barrier_limit = int(getattr(self.config, 'MID_ROUTE_BARRIER_COLLISION_FAIL_COUNT', 0) or 0)
        if barrier_limit > 0:
            txt = self.font.render(f"Barrier Coll: {barrier_n}/{barrier_limit}", True, (120, 60, 0))
        else:
            txt = self.font.render(f"Barrier Coll: {barrier_n}", True, (120, 60, 0))
        self.screen.blit(txt, (mode_x, current_y))
        current_y += 18

        # ========== 6. MMG动力学信息 ==========
        if self.ship.use_mmg:
            self._draw_section_title("DYNAMICS", control_x, panel_width, current_y)
            current_y += 18
            
            info = self.ship.dynamics_info
            dynamics_lines = [
                f"Spd:{info['speed']*1.944:.1f}kn",
                f"Hdg:{info['heading']:.0f}°",
                f"Rud:{info['rudder_angle']:.0f}°",
                f"RPM:{info['propeller_rpm']:.0f}",
            ]
            for line in dynamics_lines:
                text = self.font.render(line, True, (0, 0, 100))
                self.screen.blit(text, (mode_x, current_y))
                current_y += 16
        
        # ========== 6. 手动控制提示 ==========
        if self.control_mode == "manual":
            current_y += 15
            self._draw_section_title("MANUAL", control_x, panel_width, current_y)
            current_y += 18
            hint_lines = ["W:Forward S:Back", "A:Left D:Right"]
            for line in hint_lines:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (mode_x, current_y))
                current_y += 16
    
    def _draw_button_text(self, text: str, rect: pygame.Rect, color):
        """在按钮中心绘制文字"""
        surface = self.font.render(text, True, color)
        text_rect = surface.get_rect(center=rect.center)
        self.screen.blit(surface, text_rect)
    
    def _draw_section_title(self, title: str, control_x: int, panel_width: int, y: int):
        """绘制分节标题"""
        title_surface = self.font.render(title, True, (80, 80, 80))
        title_rect = title_surface.get_rect(center=(control_x + panel_width // 2, y))
        self.screen.blit(title_surface, title_rect)
    
    def _record_frame(self):
        """录制帧"""
        self._video_frame_counter += 1
        if self._video_frame_counter % self._video_record_every_n != 0:
            return

        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self._video_writer is None:
            video_path = self.output_dir / "simulation_video.mp4"
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(getattr(self.config, 'VIDEO_FPS', 30))
            fps = max(1, fps)
            self._video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            self._video_out_path = video_path

        if self._video_writer is not None:
            self._video_writer.write(frame)
    
    def _cleanup(self):
        """清理并生成报告"""
        # 保存当前算法结果（如果还没保存）
        if self.config.PATH_TYPE not in self.algorithm_results:
            journey_time = self.simulation_time - self.journey_start_time
            self._save_algorithm_result(journey_time)
        
        # 生成对比报告
        if self.algorithm_results:
            print("\n📊 正在生成对比报告...")
            self._generate_report()
        
        if self._video_writer is not None:
            self._video_writer.release()
            if self._video_out_path is not None:
                print(f"📹 视频已保存: {self._video_out_path}")

        pygame.quit()


if __name__ == "__main__":
    # 快速测试
    from enhanced_config import quick_config
    
    config = quick_config(scale="medium", ice="close", ship="research")
    print(config.get_summary())
    
    sim = EnhancedPhysicsSimulator(config)
    sim.run()
