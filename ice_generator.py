"""
@File    : ice_generator.py
@Author  : Bin MEI
@Date    : 2025-10-19
@Desc    : 2D path planning algorithm .

随机冰场生成器
"""
import numpy as np
import random
import heapq
from typing import List, Dict, Tuple
from scipy.interpolate import CubicSpline
from pathlib import Path

class IceGenerator:
    """生成随机冰块分布"""
    
    def __init__(self, config):
        self.config = config
        self.world_width = config.WORLD_WIDTH
        self.world_height = config.WORLD_HEIGHT
        self._heatmap_exported = False
        
        # 空间网格（加速重叠检测）
        self.grid_size = 20.0  # 网格单元大小
        self.spatial_grid = {}  # {(gx, gy): [ice_indices]}
        
    def generate_ice_field(self, seed=None, fast_mode=True) -> List[Dict]:
        """
        生成随机冰场（优化版）
        
        Args:
            seed: 随机种子
            fast_mode: True=快速模式（使用空间网格加速），False=精确模式
        
        Returns:
            List[Dict]: 冰块列表
        """
        import time
        start_time = time.time()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        ice_blocks = []
        self.spatial_grid = {}  # 重置空间网格
        
        # 计算需要生成的总面积
        total_area = self.world_width * self.world_height
        ice_area = total_area * self.config.ICE_COVERAGE
        
        # 根据区域大小动态调整参数
        scale_factor = max(1, (self.world_width * self.world_height) / 15000)  # 相对100x150的倍数
        
        # 大尺度时减少冰块数量、增大冰块尺寸
        if scale_factor > 5:
            ice_area *= 0.6  # 大尺度减少覆盖目标
            self.grid_size = 50.0
        elif scale_factor > 2:
            ice_area *= 0.8
            self.grid_size = 30.0
        else:
            self.grid_size = 20.0
        
        current_area = 0
        attempts = 0
        max_attempts = min(10000, int(5000 * np.sqrt(scale_factor)))  # 限制尝试次数
        failed_attempts = 0
        max_failed = 300
        
        print(f"   目标覆盖: {ice_area:.0f} m² ({self.config.ICE_COVERAGE*100:.0f}%)")
        print(f"   尺度因子: {scale_factor:.1f}x (网格={self.grid_size:.0f}m)")

        try:
            current_area += float(self._inject_special_obstacles(ice_blocks, fast_mode=fast_mode) or 0.0)
        except Exception:
            pass

        start_x_bg = float(self.config.SHIP_START_X) * float(self.world_width)
        start_y_bg = float(self.config.SHIP_START_Y) * float(self.world_height)
        goal_x_bg = float(self.config.GOAL_X) * float(self.world_width)
        goal_y_bg = float(self.config.GOAL_Y) * float(self.world_height)

        try:
            clear_r = float(getattr(self.config, 'RANDOM_FIELD_CLEAR_START_GOAL_RADIUS_M', 0.0) or 0.0)
        except Exception:
            clear_r = 0.0
        clear_r = max(0.0, clear_r)

        enable_mid_band = bool(getattr(self.config, 'RANDOM_FIELD_ENABLE_MID_BAND_OBSTACLES', False))
        mid_n = int(getattr(self.config, 'RANDOM_FIELD_MID_BAND_COUNT', 0) or 0)

        try:
            base_dist = dict(getattr(self.config, 'ICE_DISTRIBUTION', {}) or {})
        except Exception:
            base_dist = {}
        if enable_mid_band and mid_n > 0 and base_dist:
            try:
                lf_scale = float(getattr(self.config, 'RANDOM_FIELD_BACKGROUND_LARGE_FLOE_WEIGHT_SCALE', 1.0) or 1.0)
            except Exception:
                lf_scale = 1.0
            try:
                ib_scale = float(getattr(self.config, 'RANDOM_FIELD_BACKGROUND_ICE_BANK_WEIGHT_SCALE', 1.0) or 1.0)
            except Exception:
                ib_scale = 1.0
            lf_scale = max(0.0, lf_scale)
            ib_scale = max(0.0, ib_scale)
            if 'Large Floe' in base_dist:
                base_dist['Large Floe'] = base_dist.get('Large Floe', 0.0) * lf_scale
            if 'Ice Bank' in base_dist:
                base_dist['Ice Bank'] = base_dist.get('Ice Bank', 0.0) * ib_scale
            ssum = float(sum(base_dist.values()))
            if ssum > 0:
                base_dist = {k: float(v) / ssum for k, v in base_dist.items()}
        self._bg_ice_distribution = base_dist if base_dist else None

        # 边缘稀薄区域参数
        enable_edge_sparse = bool(getattr(self.config, 'RANDOM_FIELD_ENABLE_EDGE_SPARSE_ZONE', False))
        try:
            edge_width_ratio = float(getattr(self.config, 'RANDOM_FIELD_EDGE_ZONE_WIDTH_RATIO', 0.15) or 0.15)
        except Exception:
            edge_width_ratio = 0.15
        edge_width_ratio = max(0.05, min(0.3, edge_width_ratio))
        try:
            edge_density_scale = float(getattr(self.config, 'RANDOM_FIELD_EDGE_DENSITY_SCALE', 0.4) or 0.4)
        except Exception:
            edge_density_scale = 0.4
        edge_density_scale = max(0.1, min(1.0, edge_density_scale))
        try:
            edge_size_scale = float(getattr(self.config, 'RANDOM_FIELD_EDGE_SIZE_SCALE', 0.35) or 0.35)
        except Exception:
            edge_size_scale = 0.35
        edge_size_scale = max(0.1, min(1.0, edge_size_scale))
        edge_zone_width = self.world_width * edge_width_ratio
        
        while current_area < ice_area and attempts < max_attempts:
            attempts += 1
            
            # 随机选择冰块类型
            ice_type = self._random_ice_type()
            
            # 随机生成位置
            margin = self.grid_size * 0.5
            x = random.uniform(margin, self.world_width - margin)
            y = random.uniform(margin * 2, self.world_height - margin * 2)

            if clear_r > 0.0:
                if (np.hypot(x - start_x_bg, y - start_y_bg) < clear_r) or (np.hypot(x - goal_x_bg, y - goal_y_bg) < clear_r):
                    continue

            # 检测是否在边缘稀薄区域
            in_edge_zone = False
            if enable_edge_sparse:
                if x < edge_zone_width or x > (self.world_width - edge_zone_width):
                    in_edge_zone = True
                    # 仅减少大冰（不让边缘“全空”）
                    if ice_type in ('Ice Bank', 'Large Floe'):
                        if random.random() > edge_density_scale:
                            try:
                                dist = getattr(self, '_bg_ice_distribution', None) or self.config.ICE_DISTRIBUTION
                                candidates = [t for t in list(dist.keys()) if t not in ('Ice Bank', 'Large Floe')]
                                if candidates:
                                    weights = [float(dist.get(t, 0.0)) for t in candidates]
                                    ice_type = random.choices(candidates, weights=weights)[0]
                                else:
                                    continue
                            except Exception:
                                continue
            
            # 生成冰块尺寸（大尺度时放大冰块）
            size_range = self.config.ICE_SIZE_RANGES[ice_type]
            size_scale = 1.0 + 0.2 * min(3, np.log10(scale_factor + 1))  # 最多放大1.6倍
            # 边缘区域冰块更小
            if in_edge_zone:
                size_scale *= edge_size_scale
            size = random.uniform(size_range[0], size_range[1]) * size_scale
            
            # 生成冰块形状
            vertices = self._generate_ice_shape(size, ice_type)
            radius = max([np.sqrt(vx**2 + vy**2) for vx, vy in vertices])
            
            # 快速重叠检测（使用空间网格）
            if fast_mode:
                if self._check_overlap_fast(x, y, radius, ice_blocks):
                    failed_attempts += 1
                    if failed_attempts > max_failed:
                        break
                    continue
            else:
                min_dist = 0.5 if len(ice_blocks) < 200 else 0.3
                if self._check_overlap(x, y, vertices, ice_blocks, min_dist):
                    failed_attempts += 1
                    if failed_attempts > max_failed:
                        break
                    continue
            
            failed_attempts = 0
            
            # 计算质量
            area = self._calculate_polygon_area(vertices)
            density = self.config.ICE_DENSITY_PER_AREA[ice_type]
            mass = area * density
            
            # 创建冰块
            ice_block = {
                'type': ice_type,
                'center': (x, y),
                'vertices': vertices,
                'size': size,
                'mass': mass,
                'area': area,
                'radius': radius  # 缓存半径
            }
            
            ice_blocks.append(ice_block)
            current_area += area
            
            # 添加到空间网格
            self._add_to_grid(len(ice_blocks) - 1, x, y, radius)
        
        elapsed = time.time() - start_time
        print(f"   ✓ 生成 {len(ice_blocks)} 个冰块 ({elapsed:.2f}s)")
        print(f"   ✓ 覆盖率: {current_area / total_area * 100:.1f}%")
        
        # 统计各类型数量
        type_counts = {}
        for ice in ice_blocks:
            t = ice['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        for ice_type, count in sorted(type_counts.items()):
            print(f"   {ice_type}: {count}")
        
        # ========== 添加中央冰脊（Ice Ridge）：可选功能 ==========
        enable_ridge = getattr(self.config, 'ENABLE_ICE_RIDGE', False)
        
        if enable_ridge:
            print("\n   Generating ice ridge (enabled)...")
            
            # 计算起点和终点
            start_x = self.config.SHIP_START_X * self.world_width
            start_y = self.config.SHIP_START_Y * self.world_height
            goal_x = self.config.GOAL_X * self.world_width
            goal_y = self.config.GOAL_Y * self.world_height
            
            # 在路径中段（Y轴40%-60%位置）创建冰脊
            ridge_y_start = 0.4 * self.world_height
            ridge_y_end = 0.6 * self.world_height
            
            # 根据世界大小调整冰脊冰块数量
            scale_factor = max(1, self.world_width / 100)
            num_ridge_blocks = max(5, int(8 * np.sqrt(scale_factor)))
            offset_range = min(50, 8 * np.sqrt(scale_factor))
            
            ridge_positions = []
            
            for i in range(num_ridge_blocks):
                y = ridge_y_start + (ridge_y_end - ridge_y_start) * (i / max(1, num_ridge_blocks - 1))
                progress = (y - start_y) / (goal_y - start_y) if (goal_y - start_y) > 0 else 0.5
                x_center = start_x + (goal_x - start_x) * progress
                x = x_center + random.uniform(-offset_range, offset_range)
                margin = self.world_width * 0.05
                x = max(margin, min(self.world_width - margin, x))
                ridge_positions.append((x, y))
            
            # 生成冰脊冰块
            ridge_count = 0
            for x, y in ridge_positions:
                ice_type = 'Ice Bank' if ridge_count % 2 == 0 else 'Large Floe'
                size_range = self.config.ICE_SIZE_RANGES[ice_type]
                size = random.uniform(size_range[0] * 1.2, size_range[1])
                vertices = self._generate_ice_shape(size, ice_type)
                area = self._calculate_polygon_area(vertices)
                density = self.config.ICE_DENSITY_PER_AREA[ice_type]
                mass = area * density
                
                ice_block = {
                    'type': ice_type,
                    'center': (x, y),
                    'vertices': vertices,
                    'size': size,
                    'mass': mass
                }
                ice_blocks.append(ice_block)
                ridge_count += 1
            
            print(f"   + Ice ridge: {ridge_count} large blocks")
            print(f"   + Position: Y={ridge_y_start:.0f}-{ridge_y_end:.0f}m")
        else:
            print("\n   Ice ridge: DISABLED (分散浮冰模式)")
        
        print(f"   + Total: {len(ice_blocks)} ice blocks\n")
        
        return ice_blocks

    def _inject_special_obstacles(self, ice_blocks: List[Dict], fast_mode: bool = True) -> float:
        total_added_area = 0.0

        start_x = float(self.config.SHIP_START_X) * float(self.world_width)
        start_y = float(self.config.SHIP_START_Y) * float(self.world_height)
        goal_x = float(self.config.GOAL_X) * float(self.world_width)
        goal_y = float(self.config.GOAL_Y) * float(self.world_height)

        # ========== 1) Huge ice (optional) ==========
        inject_huge = bool(getattr(self.config, 'RANDOM_FIELD_INJECT_HUGE_ICE', False))
        huge_count = int(getattr(self.config, 'RANDOM_FIELD_HUGE_ICE_COUNT', 0) or 0)
        injected_huge = 0
        if inject_huge and huge_count > 0:
            huge_type = str(getattr(self.config, 'RANDOM_FIELD_HUGE_ICE_TYPE', 'Ice Bank') or 'Ice Bank')
            huge_type = huge_type if huge_type in getattr(self.config, 'ICE_SIZE_RANGES', {}) else 'Ice Bank'
            try:
                min_size = float(getattr(self.config, 'RANDOM_FIELD_HUGE_ICE_MIN_SIZE_M', self.config.SHIP_LENGTH * 2.0))
            except Exception:
                min_size = float(self.config.SHIP_LENGTH * 2.0)
            try:
                max_size = float(getattr(self.config, 'RANDOM_FIELD_HUGE_ICE_MAX_SIZE_M', self.config.SHIP_LENGTH * 5.0))
            except Exception:
                max_size = float(self.config.SHIP_LENGTH * 5.0)
            if max_size < min_size:
                max_size, min_size = min_size, max_size
            min_size = max(5.0, min_size)
            max_size = max(min_size, max_size)

            avoid_r = float(getattr(self.config, 'SHIP_LENGTH', 80.0)) * 4.0
            tries = 0
            max_tries = max(60, huge_count * 60)
            while injected_huge < huge_count and tries < max_tries:
                tries += 1
                size = random.uniform(min_size, max_size)
                vertices = self._generate_ice_shape(size, huge_type)
                radius = max([np.sqrt(vx**2 + vy**2) for vx, vy in vertices])
                margin = max(self.grid_size, radius + self.grid_size)
                if margin * 2 >= self.world_width or margin * 2 >= self.world_height:
                    break

                x = random.uniform(margin, self.world_width - margin)
                y = random.uniform(margin, self.world_height - margin)

                if (np.hypot(x - start_x, y - start_y) < avoid_r) or (np.hypot(x - goal_x, y - goal_y) < avoid_r):
                    continue

                if fast_mode:
                    if self._check_overlap_fast(x, y, radius, ice_blocks):
                        continue
                else:
                    if self._check_overlap(x, y, vertices, ice_blocks, 1.0):
                        continue

                area = self._calculate_polygon_area(vertices)
                density = float(self.config.ICE_DENSITY_PER_AREA.get(huge_type, self.config.ICE_DENSITY_PER_AREA.get('Ice Bank', 800.0)))
                mass = area * density

                ice_block = {
                    'type': huge_type,
                    'center': (x, y),
                    'vertices': vertices,
                    'size': float(size),
                    'mass': float(mass),
                    'area': float(area),
                    'radius': float(radius),
                    'source': 'InjectedHuge'
                }
                ice_blocks.append(ice_block)
                self._add_to_grid(len(ice_blocks) - 1, x, y, radius)
                injected_huge += 1
                total_added_area += float(area)

        if injected_huge > 0:
            print(f"   + Injected huge ice: {injected_huge} blocks")

        # ========== 2) Mid-band obstacles (optional) ==========
        enable_mid_band = bool(getattr(self.config, 'RANDOM_FIELD_ENABLE_MID_BAND_OBSTACLES', False))
        mid_n = int(getattr(self.config, 'RANDOM_FIELD_MID_BAND_COUNT', 0) or 0)
        injected_mid_groups = 0
        injected_mid_segments = 0
        if enable_mid_band and mid_n > 0:
            mid_type = str(getattr(self.config, 'RANDOM_FIELD_MID_BAND_TYPE', 'Large Floe') or 'Large Floe')
            mid_type = mid_type if mid_type in getattr(self.config, 'ICE_SIZE_RANGES', {}) else 'Large Floe'

            t_min = float(getattr(self.config, 'RANDOM_FIELD_MID_BAND_T_MIN', 0.35) or 0.35)
            t_max = float(getattr(self.config, 'RANDOM_FIELD_MID_BAND_T_MAX', 0.70) or 0.70)
            t_min = float(max(0.0, min(0.95, t_min)))
            t_max = float(max(0.05, min(1.0, t_max)))
            if t_max < t_min:
                t_min, t_max = t_max, t_min
            lateral = float(getattr(self.config, 'RANDOM_FIELD_MID_BAND_LATERAL_SPREAD_M', self.config.SHIP_LENGTH * 7.0) or (self.config.SHIP_LENGTH * 7.0))
            lateral = max(float(self.config.SHIP_LENGTH), lateral)

            avoid_r = float(getattr(self.config, 'SHIP_LENGTH', 80.0)) * 4.0
            vx = goal_x - start_x
            vy = goal_y - start_y
            vlen = float(np.hypot(vx, vy))
            if vlen < 1e-6:
                vlen = 1.0
            nx = -vy / vlen
            ny = vx / vlen

            strip_chain = bool(getattr(self.config, 'RANDOM_FIELD_MID_BAND_STRIP_CHAIN', False))
            seg_n = int(getattr(self.config, 'RANDOM_FIELD_MID_BAND_STRIP_SEGMENTS', 3) or 3)
            seg_n = max(1, min(12, seg_n))
            try:
                seg_len_min = float(getattr(self.config, 'RANDOM_FIELD_MID_BAND_STRIP_MIN_LENGTH_M', self.config.SHIP_LENGTH * 2.8))
            except Exception:
                seg_len_min = float(self.config.SHIP_LENGTH * 2.8)
            try:
                seg_len_max = float(getattr(self.config, 'RANDOM_FIELD_MID_BAND_STRIP_MAX_LENGTH_M', self.config.SHIP_LENGTH * 4.5))
            except Exception:
                seg_len_max = float(self.config.SHIP_LENGTH * 4.5)
            if seg_len_max < seg_len_min:
                seg_len_min, seg_len_max = seg_len_max, seg_len_min
            seg_len_min = max(float(self.config.SHIP_LENGTH) * 1.2, seg_len_min)
            seg_len_max = max(seg_len_min, seg_len_max)
            width_ratio = float(getattr(self.config, 'RANDOM_FIELD_MID_BAND_STRIP_WIDTH_RATIO', 0.22) or 0.22)
            width_ratio = float(max(0.10, min(0.60, width_ratio)))
            gap_m = float(getattr(self.config, 'RANDOM_FIELD_MID_BAND_STRIP_GAP_M', self.config.SHIP_LENGTH * 0.6) or (self.config.SHIP_LENGTH * 0.6))
            gap_m = max(0.0, gap_m)
            density_scale = float(getattr(self.config, 'RANDOM_FIELD_MID_BAND_DENSITY_SCALE', 0.25) or 0.25)
            density_scale = float(max(0.0, min(1.0, density_scale)))

            tries = 0
            max_tries = max(120, mid_n * 80)
            while injected_mid_groups < mid_n and tries < max_tries:
                tries += 1
                t = random.uniform(t_min, t_max)
                base_x = start_x + vx * t
                base_y = start_y + vy * t
                offset = random.uniform(-lateral, lateral)
                cx = base_x + nx * offset
                cy = base_y + ny * offset

                if (np.hypot(cx - start_x, cy - start_y) < avoid_r) or (np.hypot(cx - goal_x, cy - goal_y) < avoid_r):
                    continue

                if strip_chain:
                    rot_base = float(np.arctan2(ny, nx))
                    seg_added = 0
                    seg_tries = 0
                    max_seg_tries = 80 * seg_n
                    while seg_added < seg_n and seg_tries < max_seg_tries:
                        seg_tries += 1
                        seg_u = (seg_added - (seg_n - 1) * 0.5)
                        seg_x = cx + nx * seg_u * (gap_m + 0.6 * seg_len_min) + (vx / vlen) * random.uniform(-0.15, 0.15) * seg_len_min
                        seg_y = cy + ny * seg_u * (gap_m + 0.6 * seg_len_min) + (vy / vlen) * random.uniform(-0.15, 0.15) * seg_len_min

                        seg_len = random.uniform(seg_len_min, seg_len_max)
                        seg_w = seg_len * width_ratio * random.uniform(0.85, 1.20)
                        seg_rot = rot_base + random.uniform(-0.35, 0.35)
                        vertices = self._generate_strip_shape(seg_len, seg_w, seg_rot)
                        radius = float(0.55 * np.hypot(seg_len, seg_w))

                        margin = max(self.grid_size, radius + self.grid_size)
                        if margin * 2 >= self.world_width or margin * 2 >= self.world_height:
                            break
                        seg_x = max(margin, min(self.world_width - margin, seg_x))
                        seg_y = max(margin, min(self.world_height - margin, seg_y))

                        if (np.hypot(seg_x - start_x, seg_y - start_y) < avoid_r) or (np.hypot(seg_x - goal_x, seg_y - goal_y) < avoid_r):
                            continue

                        if fast_mode:
                            if self._check_overlap_fast(seg_x, seg_y, radius, ice_blocks):
                                continue
                        else:
                            if self._check_overlap(seg_x, seg_y, vertices, ice_blocks, 1.0):
                                continue

                        area = self._calculate_polygon_area(vertices)
                        density = float(self.config.ICE_DENSITY_PER_AREA.get(mid_type, self.config.ICE_DENSITY_PER_AREA.get('Large Floe', 400.0)))
                        mass = area * density * density_scale

                        ice_block = {
                            'type': mid_type,
                            'center': (seg_x, seg_y),
                            'vertices': vertices,
                            'size': float(seg_len),
                            'mass': float(mass),
                            'area': float(area),
                            'radius': float(radius),
                            'source': 'InjectedMidBand'
                        }
                        ice_blocks.append(ice_block)
                        self._add_to_grid(len(ice_blocks) - 1, seg_x, seg_y, radius)
                        seg_added += 1
                        injected_mid_segments += 1
                        total_added_area += float(area)

                    injected_mid_groups += 1
                    continue

                # fallback: blob obstacle
                try:
                    min_size = float(getattr(self.config, 'RANDOM_FIELD_MID_BAND_MIN_SIZE_M', self.config.SHIP_LENGTH * 1.2))
                except Exception:
                    min_size = float(self.config.SHIP_LENGTH * 1.2)
                try:
                    max_size = float(getattr(self.config, 'RANDOM_FIELD_MID_BAND_MAX_SIZE_M', self.config.SHIP_LENGTH * 2.2))
                except Exception:
                    max_size = float(self.config.SHIP_LENGTH * 2.2)
                if max_size < min_size:
                    max_size, min_size = min_size, max_size
                size = random.uniform(max(5.0, min_size), max(5.0, max_size))
                vertices = self._generate_ice_shape(size, mid_type)
                radius = max([np.sqrt(vx2**2 + vy2**2) for vx2, vy2 in vertices])
                margin = max(self.grid_size, radius + self.grid_size)
                if margin * 2 >= self.world_width or margin * 2 >= self.world_height:
                    break
                cx = max(margin, min(self.world_width - margin, cx))
                cy = max(margin, min(self.world_height - margin, cy))

                if fast_mode:
                    if self._check_overlap_fast(cx, cy, radius, ice_blocks):
                        continue
                else:
                    if self._check_overlap(cx, cy, vertices, ice_blocks, 1.0):
                        continue

                area = self._calculate_polygon_area(vertices)
                density = float(self.config.ICE_DENSITY_PER_AREA.get(mid_type, self.config.ICE_DENSITY_PER_AREA.get('Large Floe', 400.0)))
                mass = area * density
                ice_block = {
                    'type': mid_type,
                    'center': (cx, cy),
                    'vertices': vertices,
                    'size': float(size),
                    'mass': float(mass),
                    'area': float(area),
                    'radius': float(radius),
                    'source': 'InjectedMidBand'
                }
                ice_blocks.append(ice_block)
                self._add_to_grid(len(ice_blocks) - 1, cx, cy, radius)
                injected_mid_groups += 1
                injected_mid_segments += 1
                total_added_area += float(area)

        if injected_mid_groups > 0:
            print(f"   + Injected mid-band: groups={injected_mid_groups}, segments={injected_mid_segments}")

        # ========== 3) Curved bank (optional) ==========
        enable_curved_bank = bool(getattr(self.config, 'RANDOM_FIELD_ENABLE_CURVED_ICE_BANK', False))
        curved_n = int(getattr(self.config, 'RANDOM_FIELD_CURVED_ICE_BANK_BLOCKS', 0) or 0)
        injected_bank = 0
        if enable_curved_bank and curved_n > 0:
            bank_type = str(getattr(self.config, 'RANDOM_FIELD_CURVED_ICE_BANK_TYPE', 'Ice Bank') or 'Ice Bank')
            bank_type = bank_type if bank_type in getattr(self.config, 'ICE_SIZE_RANGES', {}) else 'Ice Bank'
            try:
                min_size = float(getattr(self.config, 'RANDOM_FIELD_CURVED_ICE_BANK_MIN_SIZE_M', self.config.SHIP_LENGTH * 2.5))
            except Exception:
                min_size = float(self.config.SHIP_LENGTH * 2.5)
            try:
                max_size = float(getattr(self.config, 'RANDOM_FIELD_CURVED_ICE_BANK_MAX_SIZE_M', self.config.SHIP_LENGTH * 4.5))
            except Exception:
                max_size = float(self.config.SHIP_LENGTH * 4.5)
            if max_size < min_size:
                max_size, min_size = min_size, max_size
            min_size = max(5.0, min_size)
            max_size = max(min_size, max_size)

            amp = float(getattr(self.config, 'RANDOM_FIELD_CURVED_ICE_BANK_AMPLITUDE_M', self.config.SHIP_LENGTH * 6.0) or 0.0)
            wl = float(getattr(self.config, 'RANDOM_FIELD_CURVED_ICE_BANK_WAVELENGTH_M', self.config.SHIP_LENGTH * 20.0) or 1.0)
            width_ratio = float(getattr(self.config, 'RANDOM_FIELD_CURVED_ICE_BANK_WIDTH_RATIO', 0.75) or 0.75)
            gap_ratio = float(getattr(self.config, 'RANDOM_FIELD_CURVED_ICE_BANK_GAP_RATIO', 0.18) or 0.18)
            width_ratio = float(max(0.2, min(0.95, width_ratio)))
            gap_ratio = float(max(0.0, min(0.6, gap_ratio)))
            if wl <= 1e-6:
                wl = 1.0

            avoid_r = float(getattr(self.config, 'SHIP_LENGTH', 80.0)) * 5.0
            span = float(self.world_height) * float(width_ratio)
            y0 = float(self.world_height) * 0.5 - span * 0.5
            y1 = float(self.world_height) * 0.5 + span * 0.5
            y0 = max(self.grid_size, min(self.world_height - self.grid_size, y0))
            y1 = max(self.grid_size, min(self.world_height - self.grid_size, y1))
            if y1 <= y0:
                y0, y1 = self.grid_size, self.world_height - self.grid_size

            base_x = float(self.world_width) * random.uniform(0.35, 0.65)
            phase = random.uniform(0.0, 2 * np.pi)

            tries = 0
            max_tries = max(120, curved_n * 80)
            while injected_bank < curved_n and tries < max_tries:
                tries += 1
                t = (injected_bank / max(1, curved_n - 1))
                y = y0 + (y1 - y0) * t
                x = base_x + amp * float(np.sin(2 * np.pi * (y / wl) + phase))
                size = random.uniform(min_size, max_size)
                vertices = self._generate_ice_shape(size, bank_type)
                radius = max([np.sqrt(vx**2 + vy**2) for vx, vy in vertices])
                margin = max(self.grid_size, radius + self.grid_size)
                if margin * 2 >= self.world_width or margin * 2 >= self.world_height:
                    break
                x = max(margin, min(self.world_width - margin, x))
                y = max(margin, min(self.world_height - margin, y))

                if (np.hypot(x - start_x, y - start_y) < avoid_r) or (np.hypot(x - goal_x, y - goal_y) < avoid_r):
                    injected_bank += 1
                    continue

                if fast_mode:
                    if self._check_overlap_fast(x, y, radius, ice_blocks):
                        injected_bank += 1
                        continue
                else:
                    if self._check_overlap(x, y, vertices, ice_blocks, 1.0):
                        injected_bank += 1
                        continue

                area = self._calculate_polygon_area(vertices)
                density = float(self.config.ICE_DENSITY_PER_AREA.get(bank_type, self.config.ICE_DENSITY_PER_AREA.get('Ice Bank', 800.0)))
                mass = area * density

                ice_block = {
                    'type': bank_type,
                    'center': (x, y),
                    'vertices': vertices,
                    'size': float(size),
                    'mass': float(mass),
                    'area': float(area),
                    'radius': float(radius),
                    'source': 'InjectedCurvedBank'
                }
                ice_blocks.append(ice_block)
                self._add_to_grid(len(ice_blocks) - 1, x, y, radius)
                total_added_area += float(area)

                if gap_ratio > 0 and random.random() < gap_ratio:
                    injected_bank += 2
                else:
                    injected_bank += 1

        if injected_bank > 0:
            print(f"   + Injected curved ice bank segments: {min(curved_n, injected_bank)}")

        # ========== 4) Center large obstacle (optional) ==========
        enable_center = bool(getattr(self.config, 'RANDOM_FIELD_ENABLE_CENTER_OBSTACLE', False))
        if enable_center:
            try:
                center_size = float(getattr(self.config, 'RANDOM_FIELD_CENTER_OBSTACLE_SIZE_M', self.config.SHIP_LENGTH * 6.0))
            except Exception:
                center_size = float(self.config.SHIP_LENGTH * 6.0)
            center_size = max(float(self.config.SHIP_LENGTH) * 2.0, center_size)
            try:
                n_verts = int(getattr(self.config, 'RANDOM_FIELD_CENTER_OBSTACLE_VERTICES', 8) or 8)
            except Exception:
                n_verts = 8
            n_verts = max(5, min(16, n_verts))
            try:
                center_t = float(getattr(self.config, 'RANDOM_FIELD_CENTER_OBSTACLE_T', 0.5) or 0.5)
            except Exception:
                center_t = 0.5
            center_t = max(0.2, min(0.8, center_t))
            try:
                center_offset = float(getattr(self.config, 'RANDOM_FIELD_CENTER_OBSTACLE_OFFSET_M', 0.0) or 0.0)
            except Exception:
                center_offset = 0.0

            vx = goal_x - start_x
            vy = goal_y - start_y
            vlen = float(np.hypot(vx, vy))
            if vlen < 1e-6:
                vlen = 1.0
            nx = -vy / vlen
            ny = vx / vlen

            cx = start_x + vx * center_t + nx * center_offset
            cy = start_y + vy * center_t + ny * center_offset

            # 生成不规则多边形顶点（确保凸多边形，角度严格递增）
            vertices = []
            angles = sorted([2.0 * np.pi * i / n_verts + random.uniform(-0.15, 0.15) for i in range(n_verts)])
            for angle in angles:
                r = center_size * 0.5 * random.uniform(0.75, 1.0)
                vertices.append((r * np.cos(angle), r * np.sin(angle)))

            radius = max([np.sqrt(vx2**2 + vy2**2) for vx2, vy2 in vertices])
            area = self._calculate_polygon_area(vertices)
            center_type = 'Large Floe'
            if center_type not in getattr(self.config, 'ICE_SIZE_RANGES', {}):
                center_type = 'Medium Floe'
            density = float(self.config.ICE_DENSITY_PER_AREA.get(center_type, 500.0))
            mass = area * density * 2.0

            ice_block = {
                'type': center_type,
                'center': (cx, cy),
                'vertices': vertices,
                'size': float(center_size),
                'mass': float(mass),
                'area': float(area),
                'radius': float(radius),
                'source': 'InjectedCenter'
            }
            ice_blocks.append(ice_block)
            self._add_to_grid(len(ice_blocks) - 1, cx, cy, radius)
            total_added_area += float(area)
            print(f"   + Injected center obstacle: size={center_size:.0f}m, vertices={n_verts}")

        return float(total_added_area)

    def _ice_resistance_dubrovin(self, ice_thickness: float, ice_size: float, 
                                    dist_to_center: float, ice_type: str) -> float:
        """
        基于DuBrovin经验公式的冰阻力计算 (参考大连海事大学论文)
        
        公式: R_ice = p1*φ + p2*Φ*Fr^n
        其中:
        - φ = (1/4)*B_ship^2 * sqrt(b_f*h_i*ρ_i) * (1 + 2*L_ship/B_ship * μ*α_H)
        - Φ = h_ice * D_f * ρ_ice * B_ship * [μ + tan(α_0)*(α_H + L_ship/B_ship * tan(α_0))]
        
        简化版本用于路径规划代价计算
        """
        # 船舶参数
        ship_L = float(getattr(self.config, 'SHIP_LENGTH', 100.0))
        ship_B = float(getattr(self.config, 'SHIP_WIDTH', 16.0))
        v = float(getattr(self.config, 'PLANNING_SPEED_MPS', 4.0))
        
        # 冰参数
        rho_ice = 917.0  # 冰密度 kg/m³
        mu = 0.15  # 冰-船摩擦系数
        alpha_H = 0.5  # 水平角度系数
        alpha_0 = 0.4  # 艏角 (约23°)
        b_f = 1.0  # 弯曲系数
        
        # 冰块半径
        ice_radius = ice_size / 2.0
        
        # 判断是否在冰块内部
        if dist_to_center < ice_radius:
            # 在冰块内部：最大阻力
            inside_factor = 1.0
        else:
            # 在冰块外部：阻力随距离衰减
            decay_dist = dist_to_center - ice_radius
            decay_range = ship_B * 2.0  # 衰减范围
            inside_factor = max(0.0, 1.0 - decay_dist / decay_range)
        
        if inside_factor <= 0:
            return 0.0
        
        # DuBrovin公式φ项 (破冰阻力)
        h_i = ice_thickness
        phi = 0.25 * (ship_B ** 2) * np.sqrt(b_f * h_i * rho_ice) * (1 + 2 * ship_L / ship_B * mu * alpha_H)
        
        # DuBrovin公式Φ项 (浸没阻力)  
        D_f = 1.0  # 阻力系数
        Phi = h_i * D_f * rho_ice * ship_B * (mu + np.tan(alpha_0) * (alpha_H + ship_L / ship_B * np.tan(alpha_0)))
        
        # Froude数影响
        g = 9.81
        Fr = v / np.sqrt(g * ship_L)
        n_ship = 1.5  # 指数
        
        # 冰块类型修正系数
        type_factor = {
            'Ice Bank': 5.0,      # 冰堤最难穿越
            'Large Floe': 3.0,    # 大浮冰
            'Vast Floe': 2.5,     # 巨型浮冰
            'Medium Floe': 1.5,   # 中浮冰
            'Small Floe': 0.8,    # 小浮冰
            'Brash Ice': 0.3,     # 碎冰
            'Fragment': 0.2,      # 碎片
        }.get(ice_type, 1.0)
        
        # 总阻力 (归一化后用于代价计算)
        p1, p2 = 1.0, 0.5
        R_ice = (p1 * phi + p2 * Phi * (Fr ** n_ship)) * type_factor * inside_factor
        
        # 归一化到合理范围
        scale = float(getattr(self.config, 'ICE_RESISTANCE_COST_SCALE', 1.0))
        return R_ice * scale / 1e6  # 归一化

    def export_ice_resistance_heatmap(self, resistance_map: np.ndarray, grid_size: float,
                                       ice_blocks: list = None, path: list = None):
        """
        导出高质量冰阻力热力图（论文级可视化）
        
        特点：
        1. 使用专业配色方案 (viridis/plasma)
        2. 叠加冰块轮廓
        3. 叠加规划路径
        4. 起点/终点标记
        5. 专业坐标轴和图例
        """
        if getattr(self.config, 'EXPORT_ICE_RESISTANCE_HEATMAP', True) is False:
            return
        if self._heatmap_exported:
            return
        self._heatmap_exported = True

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, Polygon
            from matplotlib.collections import PatchCollection
            import matplotlib.colors as mcolors
        except Exception:
            return

        output_dir = getattr(self.config, 'OUTPUT_DIR', Path('output_enhanced_simulation'))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / 'ice_resistance_heatmap.png'

        # 数据预处理
        scale = float(getattr(self.config, 'ICE_RESISTANCE_COST_SCALE', 1.0))
        if scale <= 0:
            scale = 1.0
        data = (np.asarray(resistance_map, dtype=np.float32) / scale).T
        
        # 对数变换使热力图更清晰
        data_log = np.log1p(data)
        vmin = float(np.nanpercentile(data_log, 5)) if np.isfinite(data_log).any() else 0.0
        vmax = float(np.nanpercentile(data_log, 98)) if np.isfinite(data_log).any() else 1.0

        # 创建高质量图像
        fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
        
        # 绘制热力图背景
        extent = [0.0, float(self.world_width), 0.0, float(self.world_height)]
        im = ax.imshow(data_log, origin='lower', extent=extent, 
                       cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='auto', alpha=0.85)
        
        # 注：不再叠加冰块轮廓，热力图本身已经显示了冰阻力分布
        # 冰块轮廓会遮挡热力图效果，且颜色与热力图冲突
        
        # 添加路径（如果有）
        if path and len(path) > 1:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'w-', linewidth=2.5, label='Planned Path', zorder=10)
            ax.plot(path_x, path_y, 'g--', linewidth=1.5, zorder=11)
        
        # 起点和终点
        start_x = self.config.SHIP_START_X * self.world_width
        start_y = self.config.SHIP_START_Y * self.world_height
        goal_x = self.config.GOAL_X * self.world_width
        goal_y = self.config.GOAL_Y * self.world_height
        
        ax.scatter(start_x, start_y, c='lime', s=200, marker='o', 
                  edgecolors='white', linewidths=2, zorder=15, label='Start')
        ax.scatter(goal_x, goal_y, c='red', s=250, marker='*', 
                  edgecolors='white', linewidths=2, zorder=15, label='Goal')
        
        # 专业配色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Ice Resistance Cost (log scale)', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        # 标题和坐标轴
        ax.set_title('Ice Field Resistance Heatmap', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('X Position (m)', fontsize=11)
        ax.set_ylabel('Y Position (m)', fontsize=11)
        
        # 网格和图例
        ax.grid(True, alpha=0.2, linestyle='--', color='white')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
        
        # 添加尺度信息
        info_text = f'Area: {self.world_width:.0f}m × {self.world_height:.0f}m'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"   📊 热力图已保存: {out_path}")

    def _point_line_distance(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _rdp_simplify(self, points: np.ndarray, epsilon: float) -> np.ndarray:
        if points.shape[0] < 3:
            return points
        a = points[0]
        b = points[-1]
        max_d = -1.0
        idx = -1
        for i in range(1, points.shape[0] - 1):
            d = self._point_line_distance(points[i], a, b)
            if d > max_d:
                max_d = d
                idx = i
        if max_d > epsilon and idx >= 0:
            left = self._rdp_simplify(points[: idx + 1], epsilon)
            right = self._rdp_simplify(points[idx:], epsilon)
            if left.shape[0] > 0 and right.shape[0] > 0:
                return np.vstack([left[:-1], right])
            return np.vstack([left, right])
        return np.vstack([a, b])
    
    def _random_ice_type(self) -> str:
        """根据分布随机选择冰块类型"""
        dist = getattr(self, '_bg_ice_distribution', None) or self.config.ICE_DISTRIBUTION
        types = list(dist.keys())
        weights = list(dist.values())
        return random.choices(types, weights=weights)[0]
    
    def _generate_ice_shape(self, size: float, ice_type: str) -> List[Tuple[float, float]]:
        """
        生成冰块形状（多边形或圆形/椭圆形顶点）
        
        Args:
            size: 冰块尺寸
            ice_type: 冰块类型
        
        Returns:
            顶点列表（相对中心的坐标）
        """
        # 随机选择形状类型
        shape_type = random.choice(['smooth', 'polygon'])
        
        if shape_type == 'smooth':
            # 生成圆滑形状（圆形或椭圆形）
            return self._generate_smooth_shape(size, ice_type)
        else:
            # 生成多边形
            return self._generate_polygon_shape(size, ice_type)
    
    def _generate_smooth_shape(self, size: float, ice_type: str) -> List[Tuple[float, float]]:
        """生成圆滑的冰块（圆形/椭圆形）"""
        num_vertices = 20  # 用多边形近似圆形
        
        # 随机选择圆形或椭圆形
        is_ellipse = random.random() > 0.3
        
        if is_ellipse:
            # 椭圆：长轴和短轴比例随机
            ratio = random.uniform(0.6, 0.9)
            a = size / 2  # 长半轴
            b = size / 2 * ratio  # 短半轴
            # 随机旋转角度
            rotation = random.uniform(0, 2 * np.pi)
        else:
            # 圆形
            a = b = size / 2
            rotation = 0
        
        vertices = []
        angle_step = 2 * np.pi / num_vertices
        
        for i in range(num_vertices):
            angle = i * angle_step
            # 椭圆参数方程
            x = a * np.cos(angle)
            y = b * np.sin(angle)
            
            # 应用旋转
            if rotation != 0:
                x_rot = x * np.cos(rotation) - y * np.sin(rotation)
                y_rot = x * np.sin(rotation) + y * np.cos(rotation)
                x, y = x_rot, y_rot
            
            # 添加微小随机扰动（让边缘更自然）
            noise = random.uniform(0.95, 1.05)
            vertices.append((x * noise, y * noise))
        
        return vertices

    def _generate_strip_shape(self, length: float, width: float, rotation: float) -> List[Tuple[float, float]]:
        length = max(1.0, float(length))
        width = max(0.5, float(width))
        half_l = 0.5 * length
        half_w = 0.5 * width

        pts = [
            (-half_l, -half_w),
            (-half_l * 0.35, -half_w * 1.05),
            (half_l * 0.35, -half_w * 0.95),
            (half_l, -half_w),
            (half_l * 1.02, 0.0),
            (half_l, half_w),
            (half_l * 0.35, half_w * 1.05),
            (-half_l * 0.35, half_w * 0.95),
            (-half_l, half_w),
            (-half_l * 1.02, 0.0),
        ]

        jitter = 0.10
        pts = [(x * random.uniform(1 - jitter, 1 + jitter), y * random.uniform(1 - jitter, 1 + jitter)) for x, y in pts]

        c = float(np.cos(rotation))
        s = float(np.sin(rotation))
        return [(x * c - y * s, x * s + y * c) for x, y in pts]
    
    def _generate_polygon_shape(self, size: float, ice_type: str) -> List[Tuple[float, float]]:
        """生成多边形冰块"""
        # 根据类型决定形状复杂度
        if ice_type == 'Ice Bank':
            num_vertices = random.randint(8, 12)  # 复杂形状
        elif ice_type == 'Large Floe':
            num_vertices = random.randint(6, 10)
        elif ice_type == 'Medium Floe':
            num_vertices = random.randint(5, 8)
        elif ice_type == 'Small Floe':
            num_vertices = random.randint(4, 6)
        else:  # Fragment
            num_vertices = random.randint(3, 5)
        
        # 生成不规则多边形
        vertices = []
        angle_step = 2 * np.pi / num_vertices
        
        for i in range(num_vertices):
            angle = i * angle_step + random.uniform(-0.3, 0.3)
            # 添加径向随机性
            radius = size / 2 * random.uniform(0.7, 1.0)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append((x, y))
        
        return vertices
    
    def _calculate_polygon_area(self, vertices: List[Tuple[float, float]]) -> float:
        """计算多边形面积（鞋带公式）"""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0
    
    def _get_grid_cells(self, x: float, y: float, radius: float) -> List[Tuple[int, int]]:
        """获取冰块覆盖的所有网格单元"""
        cells = []
        gx_min = int((x - radius) / self.grid_size)
        gx_max = int((x + radius) / self.grid_size)
        gy_min = int((y - radius) / self.grid_size)
        gy_max = int((y + radius) / self.grid_size)
        
        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                cells.append((gx, gy))
        return cells
    
    def _add_to_grid(self, ice_index: int, x: float, y: float, radius: float):
        """将冰块添加到空间网格"""
        cells = self._get_grid_cells(x, y, radius)
        for cell in cells:
            if cell not in self.spatial_grid:
                self.spatial_grid[cell] = []
            self.spatial_grid[cell].append(ice_index)
    
    def _check_overlap_fast(self, x: float, y: float, radius: float, 
                            ice_blocks: List[Dict], min_dist: float = 0.3) -> bool:
        """
        快速重叠检测（使用空间网格）
        
        只检查相邻网格内的冰块，O(1)平均复杂度
        对于注入障碍物，允许背景冰靠近放置（只检测实际重叠）
        """
        # 获取需要检查的网格单元
        cells = self._get_grid_cells(x, y, radius + min_dist)
        
        # 只检查相关网格内的冰块
        checked = set()
        for cell in cells:
            if cell in self.spatial_grid:
                for idx in self.spatial_grid[cell]:
                    if idx in checked:
                        continue
                    checked.add(idx)
                    
                    ice = ice_blocks[idx]
                    ex, ey = ice['center']
                    existing_radius = ice.get('radius', ice['size'] * 0.5)
                    
                    # 对于注入障碍物，使用更小的边界距离（允许背景冰靠近）
                    src = ice.get('source', '')
                    if src.startswith('Injected'):
                        effective_min_dist = -existing_radius * 0.3  # 允许轻微重叠/靠近
                    else:
                        effective_min_dist = min_dist
                    
                    distance = np.sqrt((x - ex)**2 + (y - ey)**2)
                    if distance < (radius + existing_radius + effective_min_dist):
                        return True
        
        return False
    
    def _check_overlap(self, x: float, y: float, 
                      vertices: List[Tuple[float, float]], 
                      existing_blocks: List[Dict],
                      min_distance: float = 0.5) -> bool:
        """
        检查冰块是否与已有冰块重叠
        
        Args:
            x, y: 新冰块中心
            vertices: 新冰块顶点
            existing_blocks: 已有冰块列表
            min_distance: 最小间距
        
        Returns:
            True if 重叠, False otherwise
        """
        # 简化检查：使用边界圆
        new_radius = max([np.sqrt(vx**2 + vy**2) for vx, vy in vertices])
        
        for ice in existing_blocks:
            ex, ey = ice['center']
            existing_radius = max([np.sqrt(vx**2 + vy**2) for vx, vy in ice['vertices']])
            
            distance = np.sqrt((x - ex)**2 + (y - ey)**2)
            if distance < (new_radius + existing_radius + min_distance):
                return True
        
        return False
    
    def generate_straight_path(self) -> List[Tuple[float, float]]:
        """生成直线路径"""
        start_x = self.config.SHIP_START_X * self.world_width
        start_y = self.config.SHIP_START_Y * self.world_height
        goal_x = self.config.GOAL_X * self.world_width
        goal_y = self.config.GOAL_Y * self.world_height
        
        # 生成直线上的点
        num_points = 20
        path = []
        for i in range(num_points + 1):
            t = i / num_points
            x = start_x + (goal_x - start_x) * t
            y = start_y + (goal_y - start_y) * t
            path.append((x, y))
        
        return path
    
    def generate_skeleton_path(self, ice_blocks: List[Dict]) -> List[Tuple[float, float]]:
        """
        生成骨架路径（改进版：全局搜索 + 向目标偏移）
        
        策略：自动检测航行方向（纵向/横向），沿主方向前进，
        在垂直方向搜索冰块密度最低的位置
        """
        start_x = self.config.SHIP_START_X * self.world_width
        start_y = self.config.SHIP_START_Y * self.world_height
        goal_x = self.config.GOAL_X * self.world_width
        goal_y = self.config.GOAL_Y * self.world_height
        
        path = [(start_x, start_y)]
        
        # 检测航行方向：X方向变化大=横向，Y方向变化大=纵向
        dx = abs(goal_x - start_x)
        dy = abs(goal_y - start_y)
        horizontal_mode = dx > dy  # 横向航行模式
        
        if horizontal_mode:
            print(f"   [Skeleton] 横向航行模式 (X: {start_x:.0f} → {goal_x:.0f})")
        else:
            print(f"   [Skeleton] 纵向航行模式 (Y: {start_y:.0f} → {goal_y:.0f})")
        
        num_layers = 30  # 增加层数提高精度
        
        # 根据航行方向设置主轴和副轴
        if horizontal_mode:
            # 横向航行：沿X轴前进，在Y方向搜索
            primary_start, primary_end = start_x, goal_x
            secondary_start, secondary_end = start_y, goal_y
            world_primary = self.world_width
            world_secondary = self.world_height
            current_secondary = start_y
        else:
            # 纵向航行：沿Y轴前进，在X方向搜索
            primary_start, primary_end = start_y, goal_y
            secondary_start, secondary_end = start_x, goal_x
            world_primary = self.world_height
            world_secondary = self.world_width
            current_secondary = start_x
        
        for i in range(1, num_layers):
            t = i / num_layers
            primary_pos = primary_start + (primary_end - primary_start) * t
            target_secondary = secondary_start + (secondary_end - secondary_start) * t
            
            # 搜索范围
            search_center = (current_secondary + target_secondary) / 2
            search_range = max(100, world_secondary * 0.4)
            
            best_secondary = search_center
            min_cost = float('inf')
            
            num_samples = max(40, int(search_range / 5))
            
            for test_secondary in np.linspace(
                max(10, search_center - search_range), 
                min(world_secondary - 10, search_center + search_range), 
                num_samples
            ):
                total_cost = 0
                search_radius = max(50.0, world_primary * 0.05)
                
                # 转换为(x, y)坐标
                if horizontal_mode:
                    test_x, test_y = primary_pos, test_secondary
                else:
                    test_x, test_y = test_secondary, primary_pos
                
                for ice in ice_blocks:
                    ice_x, ice_y = ice['center']
                    dist = np.sqrt((ice_x - test_x)**2 + (ice_y - test_y)**2)
                    
                    ice_size = ice.get('size', 5.0)
                    ice_type = ice.get('type', 'Fragment')
                    
                    # 根据冰块类型设置权重（大冰块强烈惩罚）
                    if ice_type == 'Ice Bank':
                        weight = 10.0  # 超大冰块：强烈避开
                        penalty_radius = ice_size * 1.5  # 扩大惩罚半径
                    elif ice_type == 'Large Floe':
                        weight = 5.0   # 大浮冰：强烈避开
                        penalty_radius = ice_size * 1.3
                    elif ice_type == 'Medium Floe':
                        weight = 2.5   # 中等浮冰：尽量避开
                        penalty_radius = ice_size * 1.2
                    elif ice_type == 'Small Floe':
                        weight = 1.0   # 小浮冰：可以穿过
                        penalty_radius = ice_size
                    else:  # Fragment
                        weight = 0.3   # 碎片：基本忽略
                        penalty_radius = ice_size * 0.8
                    
                    # 如果在惩罚半径内，计算代价
                    if dist < penalty_radius:
                        # 距离越近代价越高（使用指数衰减）
                        cost = weight * ice_size * (penalty_radius - dist) / (dist + 0.5)
                        total_cost += cost
                    elif dist < search_radius:
                        # 较远的冰块，轻微影响
                        cost = weight * ice_size * 0.1 / (dist + 1.0)
                        total_cost += cost
                
                # 加入"向目标移动"的奖励
                deviation_from_goal = abs(test_secondary - target_secondary)
                goal_attraction = deviation_from_goal * 0.5
                total_cost += goal_attraction
                
                # 加入"路径平滑"奖励
                turn_penalty = abs(test_secondary - current_secondary) * 0.3
                total_cost += turn_penalty
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_secondary = test_secondary
            
            # 添加路径点
            if horizontal_mode:
                path.append((primary_pos, best_secondary))
            else:
                path.append((best_secondary, primary_pos))
            current_secondary = best_secondary
        
        path.append((goal_x, goal_y))
        
        return path
    
    def generate_skeleton_path_from_position(self, ice_blocks: List[Dict], start_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        从指定位置生成骨架路径（用于动态重规划）
        改进版：全局搜索 + 向目标偏移
        """
        start_x, start_y = start_pos
        goal_x = self.config.GOAL_X * self.world_width
        goal_y = self.config.GOAL_Y * self.world_height
        
        path = [(start_x, start_y)]
        
        # 从当前Y位置到目标Y位置，动态分层
        y_remaining = goal_y - start_y
        if y_remaining <= 0:
            # 已经超过目标，直接连接
            return [(start_x, start_y), (goal_x, goal_y)]
        
        num_layers = max(5, int(y_remaining / 8.0))#分层8米
        current_x = start_x
        
        for i in range(1, num_layers):
            t = i / num_layers
            y = start_y + y_remaining * t
            
            # 动态搜索中心（向目标移动）
            target_x_at_layer = start_x + (goal_x - start_x) * t
            search_center = (current_x + target_x_at_layer) / 2
            search_range = 50
            
            best_x = search_center
            min_cost = float('inf')
            
            for test_x in np.linspace(
                max(5, search_center - search_range),
                min(self.world_width - 5, search_center + search_range),
                30
            ):
                total_cost = 0
                search_radius = 30.0  # 增大搜索半径到30米
                
                for ice in ice_blocks:
                    ice_x, ice_y = ice['center']
                    dist = np.sqrt((ice_x - test_x)**2 + (ice_y - y)**2)
                    
                    ice_size = ice.get('size', 5.0)
                    ice_type = ice.get('type', 'Fragment')
                    
                    # 根据冰块类型设置权重
                    if ice_type == 'Ice Bank':
                        weight = 10.0
                        penalty_radius = ice_size * 1.5
                    elif ice_type == 'Large Floe':
                        weight = 5.0
                        penalty_radius = ice_size * 1.3
                    elif ice_type == 'Medium Floe':
                        weight = 2.5
                        penalty_radius = ice_size * 1.2
                    elif ice_type == 'Small Floe':
                        weight = 1.0
                        penalty_radius = ice_size
                    else:  # Fragment
                        weight = 0.3
                        penalty_radius = ice_size * 0.8
                    
                    # 计算代价
                    if dist < penalty_radius:
                        cost = weight * ice_size * (penalty_radius - dist) / (dist + 0.5)
                        total_cost += cost
                    elif dist < search_radius:
                        cost = weight * ice_size * 0.1 / (dist + 1.0)
                        total_cost += cost
                
                # 向目标移动奖励
                deviation_from_goal = abs(test_x - target_x_at_layer)
                total_cost += deviation_from_goal * 0.5
                
                # 路径平滑奖励
                total_cost += abs(test_x - current_x) * 0.3
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_x = test_x
            
            path.append((best_x, y))
            current_x = best_x
        
        path.append((goal_x, goal_y))
        return path
    
    def generate_a_star_path(self, ice_blocks: List[Dict]) -> List[Tuple[float, float]]:
        """
        多目标代价A*算法 - 综合考虑距离、冰阻力、安全性
        
        代价函数: C = w_dist * 距离 + w_ice * 冰阻力 + w_safe * 危险度 + w_turn * 转向
        
        权重可调（用于实验对比）：
        - 经济模式: w_ice高, 避开厚冰
        - 安全模式: w_safe高, 远离大冰块
        - 快速模式: w_dist高, 路径短
        """
        print("   🔍 正在运行多目标A*算法...")
        
        # ========== 1. 权重配置（可调参数）==========
        weights = {
            'distance': float(getattr(self.config, 'ASTAR_WEIGHT_DISTANCE', 1.0)),
            'ice_resistance': float(getattr(self.config, 'ASTAR_WEIGHT_ICE_RESISTANCE', 2.0)),
            'safety': float(getattr(self.config, 'ASTAR_WEIGHT_SAFETY', 1.5)),
            'turn': float(getattr(self.config, 'ASTAR_WEIGHT_TURN', 0.5)),
        }
        
        # ========== 2. 创建代价地图（根据场景自动调整分辨率）==========
        # 大场景用大栅格，避免计算过慢
        world_area = self.world_width * self.world_height
        if world_area > 100000:  # 大于100x100x10的场景
            grid_size = 8.0  # 大栅格
        elif world_area > 30000:
            grid_size = 5.0  # 中栅格
        else:
            grid_size = 3.0  # 小栅格
        
        cols = int(self.world_width / grid_size)
        rows = int(self.world_height / grid_size)
        print(f"   📐 栅格: {cols}x{rows} (分辨率: {grid_size}m)")
        
        # 代价地图: 每个格子的通行代价
        cost_map = np.ones((cols, rows), dtype=np.float32)  # 基础代价=1
        obstacle_map = np.zeros((cols, rows), dtype=np.int8)  # 0=可通行, 1=障碍
        resistance_map = np.zeros((cols, rows), dtype=np.float32)
        
        # 船只安全半径
        ship_radius = self.config.SHIP_WIDTH / 2.0 + 1.0

        huge_ice_threshold = 7.0 * float(getattr(self.config, 'SHIP_LENGTH', 100.0))
        huge_ice_inflation = 6.0 * ship_radius
        
        # 边缘代价暂时禁用 - 可能导致路径规划失败
        # edge_band = max(30.0, 1.5 * float(getattr(self.config, 'SHIP_LENGTH', 100.0)))
        # edge_weight = 10.0
        # if edge_band > 0:
        #     for c in range(cols):
        #         ...
        
        # 计算每个格子的代价
        for ice in ice_blocks:
            cx, cy = ice['center']
            size = ice.get('size', 5.0)
            ice_type = ice.get('type', 'Medium Floe')
            
            # 根据冰块类型确定冰厚
            thickness_map = {
                'Brash Ice': 0.3,
                'Small Floe': 0.5,
                'Medium Floe': 0.8,
                'Large Floe': 1.2,
                'Vast Floe': 1.5,
            }
            ice_thickness = thickness_map.get(ice_type, 0.8)
            
            # 冰块影响范围（大幅减小以便找到路径）
            influence_radius = size / 2.0 + ship_radius * 1.5  # 影响范围
            collision_radius = size / 2.0  # 只标记冰块本身为障碍

            if size >= huge_ice_threshold:
                collision_radius = size / 2.0 + huge_ice_inflation
                influence_radius = collision_radius
            
            # 计算受影响的栅格
            min_c = max(0, int((cx - influence_radius) / grid_size))
            max_c = min(cols, int((cx + influence_radius) / grid_size) + 1)
            min_r = max(0, int((cy - influence_radius) / grid_size))
            max_r = min(rows, int((cy + influence_radius) / grid_size) + 1)
            
            for c in range(min_c, max_c):
                for r in range(min_r, max_r):
                    cell_x = c * grid_size + grid_size/2
                    cell_y = r * grid_size + grid_size/2
                    dist_to_ice = np.sqrt((cell_x - cx)**2 + (cell_y - cy)**2)
                    
                    # 碰撞区域 = 障碍物（冰块核心区域）
                    if dist_to_ice < collision_radius * 0.6:
                        obstacle_map[c, r] = 1
                        # 冰块内部也计算阻力用于热力图
                        r_term = self._ice_resistance_dubrovin(ice_thickness, size, dist_to_ice, ice_type)
                        resistance_map[c, r] = max(resistance_map[c, r], r_term)
                        continue
                    
                    # 影响区域 = 增加代价 (使用DuBrovin公式)
                    if dist_to_ice < influence_radius:
                        # 基于DuBrovin公式计算冰阻力
                        r_term = self._ice_resistance_dubrovin(ice_thickness, size, dist_to_ice, ice_type)
                        ice_cost = r_term * weights['ice_resistance']
                        
                        # 安全代价 (大冰块更危险)
                        ice_radius = size / 2.0
                        if dist_to_ice < ice_radius:
                            safety_factor = 1.0  # 在冰内部
                        else:
                            safety_factor = max(0, 1.0 - (dist_to_ice - ice_radius) / (influence_radius - ice_radius))
                        safety_cost = (size / 10.0) * safety_factor * weights['safety']
                        
                        # 累加代价
                        cost_map[c, r] += ice_cost + safety_cost
                        resistance_map[c, r] = max(resistance_map[c, r], r_term)

        self.export_ice_resistance_heatmap(resistance_map, grid_size, ice_blocks)
        
        # ========== 3. A* 搜索 ==========
        start_x = self.config.SHIP_START_X * self.world_width
        start_y = self.config.SHIP_START_Y * self.world_height
        goal_x = self.config.GOAL_X * self.world_width
        goal_y = self.config.GOAL_Y * self.world_height
        
        start_node = (int(start_x / grid_size), int(start_y / grid_size))
        goal_node = (int(goal_x / grid_size), int(goal_y / grid_size))
        
        # 确保起点终点在地图范围内
        start_node = (max(0, min(cols-1, start_node[0])), max(0, min(rows-1, start_node[1])))
        goal_node = (max(0, min(cols-1, goal_node[0])), max(0, min(rows-1, goal_node[1])))
        
        # 智能寻找有效起点和终点 - 如果被阻挡，寻找附近的可达点
        start_node = self._find_reachable_start(obstacle_map, start_node, goal_node, cols, rows)
        goal_node = self._find_reachable_goal(obstacle_map, goal_node, start_node, cols, rows)

        # 将实际起终点回写到config，保证仿真出生点/目标点/直线基线一致
        start_x = start_node[0] * grid_size + grid_size / 2.0
        start_y = start_node[1] * grid_size + grid_size / 2.0
        goal_x = goal_node[0] * grid_size + grid_size / 2.0
        goal_y = goal_node[1] * grid_size + grid_size / 2.0
        try:
            self.config.SHIP_START_X = float(start_x) / float(self.world_width)
            self.config.SHIP_START_Y = float(start_y) / float(self.world_height)
            self.config.GOAL_X = float(goal_x) / float(self.world_width)
            self.config.GOAL_Y = float(goal_y) / float(self.world_height)
        except Exception:
            pass
        
        # 清除起点和终点周围5x5区域的障碍
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                sx, sy = start_node[0] + dx, start_node[1] + dy
                if 0 <= sx < cols and 0 <= sy < rows:
                    obstacle_map[sx, sy] = 0
                gx, gy = goal_node[0] + dx, goal_node[1] + dy
                if 0 <= gx < cols and 0 <= gy < rows:
                    obstacle_map[gx, gy] = 0
        
        # 优先队列: (f_score, g_score, x, y, parent_dir)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_node[0], start_node[1], None))
        
        came_from = {}
        g_score = {start_node: 0}
        
        # 8方向移动 (dx, dy)
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        found_path = False
        iterations = 0
        try:
            cap = int(getattr(self.config, 'ASTAR_MAX_ITERATIONS_CAP', 500000) or 500000)
        except Exception:
            cap = 500000
        try:
            factor = float(getattr(self.config, 'ASTAR_MAX_ITERATIONS_FACTOR', 4.0) or 4.0)
        except Exception:
            factor = 4.0
        cap = max(10000, cap)
        factor = max(0.5, factor)
        max_iterations = min(cap, int(cols * rows * factor))
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)
            _, current_g, cx, cy, parent_dir = current
            current_pos = (cx, cy)
            
            if current_pos == goal_node:
                found_path = True
                break
            
            # 跳过已处理的（有更优路径）
            if current_pos in g_score and current_g > g_score[current_pos]:
                continue
                
            for i, (dx, dy) in enumerate(neighbors):
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)
                
                # 越界检查
                if not (0 <= nx < cols and 0 <= ny < rows):
                    continue
                    
                # 障碍物检查
                if obstacle_map[nx, ny] == 1:
                    continue
                
                # ===== 计算移动代价 =====
                # 1. 距离代价
                base_dist = 1.414 if (dx != 0 and dy != 0) else 1.0
                dist_cost = base_dist * weights['distance']
                
                # 2. 冰阻力/安全代价（从代价地图读取）
                cell_cost = cost_map[nx, ny]
                
                # 3. 转向惩罚
                turn_cost = 0.0
                if parent_dir is not None and parent_dir != i:
                    # 方向改变越大，惩罚越高
                    old_dx, old_dy = neighbors[parent_dir]
                    dot = dx * old_dx + dy * old_dy
                    # dot: 1=同向, 0=90度, -1=反向
                    turn_cost = (1 - dot) * weights['turn']
                
                # 总代价
                move_cost = dist_cost + cell_cost + turn_cost
                tentative_g = g_score[current_pos] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current_pos, i)
                    g_score[neighbor] = tentative_g
                    
                    # 启发函数: 欧几里得距离
                    h = np.sqrt((nx - goal_node[0])**2 + (ny - goal_node[1])**2)
                    f = tentative_g + h * weights['distance']
                    
                    heapq.heappush(open_set, (f, tentative_g, nx, ny, i))
        
        try:
            self.last_planner_status = {
                'planner': 'A*',
                'success': False,
                'fallback_used': False,
                'fallback_planner': None,
            }
            setattr(self.config, 'LAST_PLANNER_STATUS', self.last_planner_status)
        except Exception:
            pass

        # ========== 4. 重建路径 ==========
        path = []
        if found_path:
            current = goal_node
            while current in came_from:
                wx = current[0] * grid_size + grid_size/2
                wy = current[1] * grid_size + grid_size/2
                path.append((wx, wy))
                current, _ = came_from[current]
            path.append((start_x, start_y))
            path.reverse()
            
            # 路径简化 (Douglas-Peucker思想: 移除共线点)
            if len(path) > 3:
                simplified = [path[0]]
                for i in range(1, len(path) - 1):
                    # 检查是否共线
                    p0, p1, p2 = path[i-1], path[i], path[i+1]
                    v1 = (p1[0] - p0[0], p1[1] - p0[1])
                    v2 = (p2[0] - p1[0], p2[1] - p1[1])
                    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if cross > 0.5:  # 非共线则保留
                        simplified.append(p1)
                simplified.append(path[-1])
                path = simplified
            
            try:
                self.last_planner_status = {
                    'planner': 'A*',
                    'success': True,
                    'fallback_used': False,
                    'fallback_planner': None,
                    'iterations': int(iterations),
                }
                setattr(self.config, 'LAST_PLANNER_STATUS', self.last_planner_status)
            except Exception:
                pass

            print(f"   ✓ 多目标A*路径找到! 节点数: {len(path)}, 迭代: {iterations}")
            return path
        else:
            # 诊断信息
            obstacle_ratio = np.sum(obstacle_map) / (cols * rows) * 100
            print(f"   ⚠️ A*未找到路径(迭代{iterations}次)")
            print(f"      障碍物比例: {obstacle_ratio:.1f}%")
            print(f"      起点: {start_node}, 终点: {goal_node}")
            
            # 检查是否有从起点可达的邻居
            reachable = 0
            for dx, dy in neighbors:
                nx, ny = start_node[0] + dx, start_node[1] + dy
                if 0 <= nx < cols and 0 <= ny < rows and obstacle_map[nx, ny] == 0:
                    reachable += 1
            print(f"      起点可达邻居: {reachable}/8")
            
            # A*失败时直接回退到Skeleton（Skeleton更稳定可靠）
            print(f"   [Fallback] A* failed, using Skeleton algorithm...")
            try:
                self.last_planner_status = {
                    'planner': 'A*',
                    'success': False,
                    'fallback_used': True,
                    'fallback_planner': 'Skeleton',
                    'iterations': int(iterations),
                    'obstacle_ratio_pct': float(obstacle_ratio),
                    'start_node': tuple(start_node),
                    'goal_node': tuple(goal_node),
                }
                setattr(self.config, 'LAST_PLANNER_STATUS', self.last_planner_status)
            except Exception:
                pass
            return self.generate_skeleton_path(ice_blocks)

    def generate_ice_theta_star_path(self, ice_blocks: List[Dict]) -> List[Tuple[float, float]]:
        print("   ✨ 正在运行 Ice-Theta* 任意角度规划算法...")

        try:
            self.last_planner_status = {
                'planner': 'Ice-Theta*',
                'success': False,
                'fallback_used': False,
                'fallback_planner': None,
            }
            setattr(self.config, 'LAST_PLANNER_STATUS', self.last_planner_status)
        except Exception:
            pass

        use_soft_cost = bool(getattr(self.config, 'ICE_THETA_USE_SOFT_COST', True))
        enable_any_angle = bool(getattr(self.config, 'ICE_THETA_ENABLE_ANY_ANGLE', True))
        try:
            los_step_cells = float(getattr(self.config, 'ICE_THETA_LOS_STEP_CELLS', 0.5))
        except Exception:
            los_step_cells = 0.5
        try:
            cost_step_cells = float(getattr(self.config, 'ICE_THETA_COST_SAMPLE_STEP_CELLS', 0.5))
        except Exception:
            cost_step_cells = 0.5
        if los_step_cells <= 0:
            los_step_cells = 0.5
        if cost_step_cells <= 0:
            cost_step_cells = 0.5

        weights = {
            'distance': float(getattr(self.config, 'ICE_THETA_WEIGHT_DISTANCE', 1.0)),
            'ice_resistance': float(getattr(self.config, 'ICE_THETA_WEIGHT_ICE_RESISTANCE', 0.6)),
            'safety': float(getattr(self.config, 'ICE_THETA_WEIGHT_SAFETY', 0.4)),
            'turn': float(getattr(self.config, 'ICE_THETA_WEIGHT_TURN', 0.3)),
            'line_cost': float(getattr(self.config, 'ICE_THETA_WEIGHT_LINE_COST', 0.8)),
        }

        world_area = self.world_width * self.world_height
        min_stride_cells = 1.0 if world_area > 1e6 else 0.5
        los_step_cells = max(min_stride_cells, float(los_step_cells))
        cost_step_cells = max(min_stride_cells, float(cost_step_cells))
        if world_area > 100000:
            grid_size = 8.0
        else:
            grid_size = 3.0

        cols = int(self.world_width / grid_size)
        rows = int(self.world_height / grid_size)
        print(f"   📐 栅格: {cols}x{rows} (分辨率: {grid_size}m)")

        cost_map = np.ones((cols, rows), dtype=np.float32)
        obstacle_map = np.zeros((cols, rows), dtype=np.int8)
        resistance_map = np.zeros((cols, rows), dtype=np.float32)

        ship_radius = self.config.SHIP_WIDTH / 2.0 + 1.0
        ship_length = float(getattr(self.config, 'SHIP_LENGTH', 80.0))
        inflation_factors = {
            'Fragment': 0.3,
            'Small Floe': 0.6,
            'Medium Floe': 1.2,
            'Large Floe': 2.0,
            'Ice Bank': 3.0,
        }
        hard_core_ratio = {
            'Fragment': 0.55,
            'Small Floe': 0.65,
            'Medium Floe': 0.85,
            'Large Floe': 0.88,
            'Ice Bank': 0.92,
        }

        thickness_map = {
            'Brash Ice': 0.3,
            'Small Floe': 0.5,
            'Medium Floe': 0.8,
            'Large Floe': 1.2,
            'Vast Floe': 1.5,
            'Ice Bank': 2.5,
        }

        for ice in ice_blocks:
            cx, cy = ice['center']
            size = float(ice.get('size', 5.0))
            ice_type = ice.get('type', 'Medium Floe')
            ice_thickness = thickness_map.get(ice_type, 0.8)

            try:
                hard_types = set(getattr(self.config, 'ICE_THETA_HARD_OBSTACLE_TYPES', ()))
            except Exception:
                hard_types = set()
            try:
                hard_min_size = float(getattr(self.config, 'ICE_THETA_HARD_OBSTACLE_MIN_SIZE_M', 0.0) or 0.0)
            except Exception:
                hard_min_size = 0.0
            is_hard_obstacle = (ice_type in hard_types) or (size >= hard_min_size)

            try:
                hard_buffer_m = float(getattr(self.config, 'ICE_THETA_HARD_OBSTACLE_BUFFER_M', 0.0) or 0.0)
            except Exception:
                hard_buffer_m = 0.0
            try:
                soft_core_mult = float(getattr(self.config, 'ICE_THETA_SOFT_CORE_COST_MULT', 2.0) or 2.0)
            except Exception:
                soft_core_mult = 2.0
            soft_core_mult = max(1.0, soft_core_mult)

            r0 = size / 2.0
            inflate = float(inflation_factors.get(ice_type, 1.2)) * ship_radius
            if ice_type not in inflation_factors and size >= 1.5 * ship_length:
                inflate = max(inflate, 1.6 * ship_radius)
            r_hard = r0 * float(hard_core_ratio.get(ice_type, 0.75))
            r_soft = r0 + inflate
            r_obstacle = r0 + hard_buffer_m + (grid_size * 0.75)
            influence_radius = max(r_soft, r_obstacle)

            min_c = max(0, int((cx - influence_radius) / grid_size))
            max_c = min(cols, int((cx + influence_radius) / grid_size) + 1)
            min_r = max(0, int((cy - influence_radius) / grid_size))
            max_r = min(rows, int((cy + influence_radius) / grid_size) + 1)

            for c in range(min_c, max_c):
                for r in range(min_r, max_r):
                    cell_x = c * grid_size + grid_size / 2
                    cell_y = r * grid_size + grid_size / 2
                    dist_to_ice = np.sqrt((cell_x - cx) ** 2 + (cell_y - cy) ** 2)
                    if is_hard_obstacle and dist_to_ice < r_obstacle:
                        obstacle_map[c, r] = 1
                        r_term = self._ice_resistance_dubrovin(ice_thickness, size, dist_to_ice, ice_type)
                        resistance_map[c, r] = max(resistance_map[c, r], r_term)
                        continue

                    if dist_to_ice < influence_radius:
                        r_term = self._ice_resistance_dubrovin(ice_thickness, size, dist_to_ice, ice_type)
                        resistance_map[c, r] = max(resistance_map[c, r], r_term)

                        if not use_soft_cost:
                            continue
                        if dist_to_ice <= r_hard:
                            decay = 1.0
                        else:
                            t = (dist_to_ice - r_hard) / max(1e-6, (r_soft - r_hard))
                            t = float(max(0.0, min(1.0, t)))
                            decay = (1.0 - t) ** 2
                        ice_cost = r_term * weights['ice_resistance'] * (0.25 + 0.75 * decay)
                        safety_cost = (size / 10.0) * decay * weights['safety']
                        cell_cost = ice_cost + safety_cost
                        if (not is_hard_obstacle) and (dist_to_ice < r0):
                            cell_cost *= soft_core_mult
                        cost_map[c, r] = max(cost_map[c, r], 1.0 + cell_cost)

        self.export_ice_resistance_heatmap(resistance_map, grid_size, ice_blocks)

        start_x = self.config.SHIP_START_X * self.world_width
        start_y = self.config.SHIP_START_Y * self.world_height
        goal_x = self.config.GOAL_X * self.world_width
        goal_y = self.config.GOAL_Y * self.world_height

        start_node = (int(start_x / grid_size), int(start_y / grid_size))
        goal_node = (int(goal_x / grid_size), int(goal_y / grid_size))

        start_node = (max(0, min(cols - 1, start_node[0])), max(0, min(rows - 1, start_node[1])))
        goal_node = (max(0, min(cols - 1, goal_node[0])), max(0, min(rows - 1, goal_node[1])))

        start_node = self._find_reachable_start(obstacle_map, start_node, goal_node, cols, rows)
        goal_node = self._find_reachable_goal(obstacle_map, goal_node, start_node, cols, rows)

        # 将实际起终点回写到config，保证仿真出生点/目标点/直线基线一致
        start_x = float(start_node[0] * grid_size + grid_size / 2.0)
        start_y = float(start_node[1] * grid_size + grid_size / 2.0)
        goal_x = float(goal_node[0] * grid_size + grid_size / 2.0)
        goal_y = float(goal_node[1] * grid_size + grid_size / 2.0)
        try:
            self.config.SHIP_START_X = float(start_x) / float(self.world_width)
            self.config.SHIP_START_Y = float(start_y) / float(self.world_height)
            self.config.GOAL_X = float(goal_x) / float(self.world_width)
            self.config.GOAL_Y = float(goal_y) / float(self.world_height)
        except Exception:
            pass

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                sx, sy = start_node[0] + dx, start_node[1] + dy
                if 0 <= sx < cols and 0 <= sy < rows:
                    obstacle_map[sx, sy] = 0
                gx, gy = goal_node[0] + dx, goal_node[1] + dy
                if 0 <= gx < cols and 0 <= gy < rows:
                    obstacle_map[gx, gy] = 0

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        def in_bounds(n):
            return 0 <= n[0] < cols and 0 <= n[1] < rows

        def is_free(n):
            return obstacle_map[n[0], n[1]] == 0

        def to_world(n):
            return (n[0] * grid_size + grid_size / 2.0, n[1] * grid_size + grid_size / 2.0)

        def heading(a, b):
            return float(np.arctan2(b[1] - a[1], b[0] - a[0]))

        def heuristic(n):
            dx = (n[0] - goal_node[0])
            dy = (n[1] - goal_node[1])
            return float(np.hypot(dx, dy))

        def _iter_line_cells(a, b, step_cells=1.0):
            x0, y0 = int(a[0]), int(a[1])
            x1, y1 = int(b[0]), int(b[1])
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            stride = int(np.ceil(max(1.0, float(step_cells))))
            k = 0
            while True:
                if (k % stride) == 0 or (x0 == x1 and y0 == y1):
                    yield (x0, y0)
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
                k += 1

        def line_of_sight(a, b):
            for p in _iter_line_cells(a, b, step_cells=los_step_cells):
                if not in_bounds(p) or obstacle_map[p[0], p[1]] == 1:
                    return False
            return True

        def segment_cost(a, b):
            ax, ay = a
            bx, by = b
            dx = bx - ax
            dy = by - ay
            dist_cells = float(np.hypot(dx, dy))
            if dist_cells < 1e-6:
                return 0.0
            csum = 0.0
            cnt = 0
            for p in _iter_line_cells(a, b, step_cells=cost_step_cells):
                if in_bounds(p):
                    csum += float(cost_map[p[0], p[1]])
                    cnt += 1
            avg_cell = (csum / max(1, cnt))
            dist_m = dist_cells * grid_size
            avg_excess = max(0.0, float(avg_cell) - 1.0)
            return float(weights['distance'] * dist_m + weights['line_cost'] * avg_excess * dist_m)

        def turn_penalty(prev, cur, nxt):
            if prev is None:
                return 0.0
            a1 = heading(prev, cur)
            a2 = heading(cur, nxt)
            da = float(np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1)))
            return abs(da) * float(getattr(self.config, 'SHIP_LENGTH', 80.0)) * 0.02 * weights['turn']

        open_set = []
        heapq.heappush(open_set, (heuristic(start_node), 0.0, start_node))
        parent = {start_node: None}
        g = {start_node: 0.0}

        found = False
        it = 0
        max_iter = min(600000, cols * rows * 2)

        while open_set and it < max_iter:
            it += 1
            _, cur_g, cur = heapq.heappop(open_set)
            if cur_g > g.get(cur, float('inf')):
                continue
            if cur == goal_node:
                found = True
                break

            cur_parent = parent.get(cur)
            for i, (dx, dy) in enumerate(neighbors):
                nb = (cur[0] + dx, cur[1] + dy)
                if not in_bounds(nb) or not is_free(nb):
                    continue

                best_from = cur
                best_cost = g[cur] + segment_cost(cur, nb) + turn_penalty(parent.get(cur), cur, nb)

                if enable_any_angle and cur_parent is not None and line_of_sight(cur_parent, nb):
                    alt = g[cur_parent] + segment_cost(cur_parent, nb) + turn_penalty(parent.get(cur_parent), cur_parent, nb)
                    if alt < best_cost:
                        best_cost = alt
                        best_from = cur_parent

                if best_cost < g.get(nb, float('inf')):
                    parent[nb] = best_from
                    g[nb] = best_cost
                    f = best_cost + heuristic(nb) * grid_size * 0.8
                    heapq.heappush(open_set, (f, best_cost, nb))

        if not found:
            obstacle_ratio = np.sum(obstacle_map) / (cols * rows) * 100
            print(f"   ⚠️ Ice-Theta*未找到路径(迭代{it}次), 障碍物比例: {obstacle_ratio:.1f}%")
            print(f"   [Fallback] Ice-Theta* failed, using A*...")
            try:
                self.last_planner_status = {
                    'planner': 'Ice-Theta*',
                    'success': False,
                    'fallback_used': True,
                    'fallback_planner': 'A*',
                    'iterations': int(it),
                    'obstacle_ratio_pct': float(obstacle_ratio),
                }
                setattr(self.config, 'LAST_PLANNER_STATUS', self.last_planner_status)
            except Exception:
                pass
            return self.generate_a_star_path(ice_blocks)

        path = []
        cur = goal_node
        while cur is not None:
            wx, wy = to_world(cur)
            path.append((float(wx), float(wy)))
            cur = parent.get(cur)
        path.reverse()

        if path:
            path[0] = (float(start_x), float(start_y))
            path[-1] = (float(goal_x), float(goal_y))

        try:
            self.last_planner_status = {
                'planner': 'Ice-Theta*',
                'success': True,
                'fallback_used': False,
                'fallback_planner': None,
                'iterations': int(it),
            }
            setattr(self.config, 'LAST_PLANNER_STATUS', self.last_planner_status)
        except Exception:
            pass

        print(f"   ✓ Ice-Theta*路径找到! 节点数: {len(path)}, 迭代: {it}")
        return path
    
    def _find_valid_position(self, obstacle_map: np.ndarray, target_node: tuple, 
                               reference_node: tuple, cols: int, rows: int,
                               search_towards_reference: bool = False,
                               position_name: str = "position") -> tuple:
        """
        智能寻找有效位置（起点或终点）
        
        如果原位置被冰块阻挡，在附近螺旋搜索可达的替代点
        
        Args:
            obstacle_map: 障碍物地图
            target_node: 原始目标节点（起点或终点）
            reference_node: 参考节点（用于确定搜索方向）
            cols, rows: 地图尺寸
            search_towards_reference: True=向参考点方向搜索，False=远离参考点
            position_name: 位置名称（用于日志）
            
        Returns:
            有效的节点（可能是原目标或备选点）
        """
        tx, ty = target_node
        
        # 检查原位置是否可达
        if 0 <= tx < cols and 0 <= ty < rows:
            # 检查5x5区域内是否有足够空间
            clear_count = 0
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = tx + dx, ty + dy
                    if 0 <= nx < cols and 0 <= ny < rows:
                        if obstacle_map[nx, ny] == 0:
                            clear_count += 1
            
            if clear_count >= 12:  # 至少48%区域无障碍（降低阈值）
                return target_node
        
        print(f"   ⚠️ [{position_name}] 原位置被阻挡，搜索备选位置...")
        
        # 计算搜索方向
        dir_x = reference_node[0] - tx
        dir_y = reference_node[1] - ty
        if not search_towards_reference:
            dir_x, dir_y = -dir_x, -dir_y
        dir_len = max(1, np.sqrt(dir_x**2 + dir_y**2))
        dir_x, dir_y = dir_x / dir_len, dir_y / dir_len
        
        best_pos = target_node
        best_score = -float('inf')
        
        # 在原位置周围搜索（半径逐渐增大）
        max_search_radius = min(cols, rows) // 3
        
        for radius in range(3, max_search_radius, 2):
            found_better = False
            
            for angle_deg in range(0, 360, 10):
                angle = np.radians(angle_deg)
                nx = int(tx + radius * np.cos(angle))
                ny = int(ty + radius * np.sin(angle))
                
                if not (0 <= nx < cols and 0 <= ny < rows):
                    continue
                
                # 检查该点周围是否足够空旷
                clear_count = 0
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        cx, cy = nx + dx, ny + dy
                        if 0 <= cx < cols and 0 <= cy < rows:
                            if obstacle_map[cx, cy] == 0:
                                clear_count += 1
                
                if clear_count < 10:  # 降低空旷度要求
                    continue
                
                # 计算评分
                dist_to_original = np.sqrt((nx - tx)**2 + (ny - ty)**2)
                
                # 是否在期望方向上
                to_new_x = nx - tx
                to_new_y = ny - ty
                to_new_len = max(1, np.sqrt(to_new_x**2 + to_new_y**2))
                direction_score = (to_new_x * dir_x + to_new_y * dir_y) / to_new_len
                
                # 综合评分：优先接近原位置，其次考虑方向
                score = -dist_to_original * 1.0 + direction_score * 30
                
                if score > best_score:
                    best_score = score
                    best_pos = (nx, ny)
                    found_better = True
            
            if found_better:
                break
        
        if best_pos != target_node:
            print(f"   ✓ [{position_name}] 找到备选位置: {target_node} -> {best_pos}")
        else:
            print(f"   ⚠️ [{position_name}] 未找到更好位置，使用原位置")
        
        return best_pos

    def _find_reachable_goal(self, obstacle_map: np.ndarray, goal_node: tuple, 
                              start_node: tuple, cols: int, rows: int) -> tuple:
        """智能寻找可达目标点（向后兼容包装）"""
        return self._find_valid_position(obstacle_map, goal_node, start_node, 
                                         cols, rows, search_towards_reference=False,
                                         position_name="Goal")
    
    def _find_reachable_start(self, obstacle_map: np.ndarray, start_node: tuple, 
                               goal_node: tuple, cols: int, rows: int) -> tuple:
        """智能寻找可达起点"""
        return self._find_valid_position(obstacle_map, start_node, goal_node, 
                                         cols, rows, search_towards_reference=True,
                                         position_name="Start")

    def _heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def generate_optimal_path(self, ice_blocks: List[Dict]) -> List[Tuple[float, float]]:
        """
        生成优化路径
        """
        # 使用新的A*算法
        return self.generate_a_star_path(ice_blocks)

    def generate_hybrid_a_star_path(self, ice_blocks: List[Dict]) -> List[Tuple[float, float]]:
        """Hybrid A*（简化版）：状态=(x,y,heading)，带最小转弯半径约束。"""
        print("   🧭 正在运行Hybrid A*算法...")

        # 1) 复用A*的代价/障碍构建策略（含大冰硬避让+边界软代价带）
        world_area = self.world_width * self.world_height
        if world_area > 100000:
            grid_size = 8.0
        elif world_area > 30000:
            grid_size = 5.0
        else:
            grid_size = 3.0

        cols = int(self.world_width / grid_size)
        rows = int(self.world_height / grid_size)
        print(f"   📐 栅格: {cols}x{rows} (分辨率: {grid_size}m)")

        cost_map = np.ones((cols, rows), dtype=np.float32)
        obstacle_map = np.zeros((cols, rows), dtype=np.int8)
        resistance_map = np.zeros((cols, rows), dtype=np.float32)

        ship_radius = self.config.SHIP_WIDTH / 2.0 + 1.0
        huge_ice_threshold = 7.0 * float(getattr(self.config, 'SHIP_LENGTH', 100.0))
        huge_ice_inflation = 6.0 * ship_radius
        
        # 边缘代价暂时禁用 - 可能导致路径规划失败
        # edge_band = max(30.0, 1.5 * float(getattr(self.config, 'SHIP_LENGTH', 100.0)))
        # edge_weight = 10.0
        # if edge_band > 0:
        #     ...

        weights = {
            'ice_resistance': 2.0,
            'safety': 1.5,
        }

        for ice in ice_blocks:
            cx, cy = ice['center']
            size = float(ice.get('size', 5.0))
            ice_type = ice.get('type', 'Medium Floe')
            thickness_map = {
                'Brash Ice': 0.3,
                'Small Floe': 0.5,
                'Medium Floe': 0.8,
                'Large Floe': 1.2,
                'Vast Floe': 1.5,
                'Ice Bank': 2.5,
            }
            ice_thickness = thickness_map.get(ice_type, 0.8)

            influence_radius = size / 2.0 + ship_radius * 1.5
            collision_radius = size / 2.0
            if size >= huge_ice_threshold or ice_type == 'Ice Bank':
                collision_radius = size / 2.0 + huge_ice_inflation
                influence_radius = collision_radius

            min_c = max(0, int((cx - influence_radius) / grid_size))
            max_c = min(cols, int((cx + influence_radius) / grid_size) + 1)
            min_r = max(0, int((cy - influence_radius) / grid_size))
            max_r = min(rows, int((cy + influence_radius) / grid_size) + 1)

            for c in range(min_c, max_c):
                for r in range(min_r, max_r):
                    cell_x = c * grid_size + grid_size / 2
                    cell_y = r * grid_size + grid_size / 2
                    dist_to_ice = np.sqrt((cell_x - cx) ** 2 + (cell_y - cy) ** 2)
                    if dist_to_ice < collision_radius * 0.6:
                        obstacle_map[c, r] = 1
                        # 冰块内部也计算阻力用于热力图
                        r_term = self._ice_resistance_dubrovin(ice_thickness, size, dist_to_ice, ice_type)
                        resistance_map[c, r] = max(resistance_map[c, r], r_term)
                        continue
                    if dist_to_ice < influence_radius:
                        # 使用DuBrovin公式
                        r_term = self._ice_resistance_dubrovin(ice_thickness, size, dist_to_ice, ice_type)
                        ice_cost = r_term * weights['ice_resistance']
                        ice_radius = size / 2.0
                        if dist_to_ice < ice_radius:
                            safety_factor = 1.0
                        else:
                            safety_factor = max(0, 1.0 - (dist_to_ice - ice_radius) / (influence_radius - ice_radius))
                        safety_cost = (size / 10.0) * safety_factor * weights['safety']
                        cost_map[c, r] += ice_cost + safety_cost
                        resistance_map[c, r] = max(resistance_map[c, r], r_term)

        self.export_ice_resistance_heatmap(resistance_map, grid_size, ice_blocks)

        # 2) Hybrid A* 搜索
        start_x = self.config.SHIP_START_X * self.world_width
        start_y = self.config.SHIP_START_Y * self.world_height
        goal_x = self.config.GOAL_X * self.world_width
        goal_y = self.config.GOAL_Y * self.world_height

        step = float(getattr(self.config, 'HYBRID_ASTAR_STEP', max(10.0, self.config.SHIP_LENGTH * 0.35)))
        headings_n = int(getattr(self.config, 'HYBRID_ASTAR_HEADINGS', 16))
        headings_n = max(8, min(72, headings_n))
        heading_bin = 2.0 * np.pi / headings_n
        min_turn_r = float(getattr(self.config, 'MIN_TURNING_RADIUS', self.config.SHIP_LENGTH * 3.5))

        def heading_to_bin(theta: float) -> int:
            t = (theta + np.pi) % (2.0 * np.pi) - np.pi
            return int(((t + np.pi) / (2.0 * np.pi)) * headings_n) % headings_n

        def cell_of(x: float, y: float):
            return int(x / grid_size), int(y / grid_size)

        def in_bounds(x: float, y: float) -> bool:
            return 0 <= x < self.world_width and 0 <= y < self.world_height

        def is_free(x: float, y: float) -> bool:
            cx, cy = cell_of(x, y)
            if not (0 <= cx < cols and 0 <= cy < rows):
                return False
            return obstacle_map[cx, cy] == 0

        def heuristic(x: float, y: float) -> float:
            return np.hypot(goal_x - x, goal_y - y)

        # 初始航向：朝向目标
        init_heading = np.arctan2(goal_y - start_y, goal_x - start_x)
        start_state = (float(start_x), float(start_y), heading_to_bin(init_heading))

        open_set = []
        heapq.heappush(open_set, (heuristic(start_x, start_y), 0.0, start_state))
        came_from = {}
        g_score = {start_state: 0.0}

        max_iter = min(600000, cols * rows * headings_n // 2)
        it = 0

        # 三种动作：直行、左转、右转（用最小转弯半径生成离散运动元）
        delta_theta = step / max(1e-6, min_turn_r)
        actions = [0.0, delta_theta, -delta_theta]

        while open_set and it < max_iter:
            it += 1
            _, cur_g, cur = heapq.heappop(open_set)
            x, y, hb = cur
            if cur_g > g_score.get(cur, float('inf')):
                continue

            if np.hypot(goal_x - x, goal_y - y) < max(step * 2.0, self.config.SHIP_LENGTH * 0.5):
                # 回溯
                path = [(goal_x, goal_y)]
                node = cur
                while node in came_from:
                    px, py, _ = node
                    path.append((px, py))
                    node = came_from[node]
                path.append((start_x, start_y))
                path.reverse()
                print(f"   ✓ Hybrid A*路径找到! 节点数: {len(path)}, 迭代: {it}")
                return path

            theta = (hb * heading_bin) - np.pi

            for dth in actions:
                nt = theta + dth
                nx = x + step * np.cos(nt)
                ny = y + step * np.sin(nt)
                if not in_bounds(nx, ny):
                    continue
                if not is_free(nx, ny):
                    continue

                cc, rr = cell_of(nx, ny)
                cell_cost = float(cost_map[cc, rr])

                turn_pen = abs(dth) * self.config.SHIP_LENGTH * 0.05
                move_cost = (step / max(1.0, self.config.SHIP_LENGTH)) + cell_cost + turn_pen

                ns = (float(nx), float(ny), heading_to_bin(nt))
                ng = cur_g + move_cost
                if ng < g_score.get(ns, float('inf')):
                    came_from[ns] = cur
                    g_score[ns] = ng
                    f = ng + heuristic(nx, ny) / max(1.0, self.config.SHIP_LENGTH)
                    heapq.heappush(open_set, (f, ng, ns))

        print(f"   ⚠️ Hybrid A*未找到路径(迭代{it}次)，回退到A*...")
        return self.generate_a_star_path(ice_blocks)

    def smooth_path(self, path: List[Tuple[float, float]], step_size: float = 1.0, 
                    smooth: bool = True) -> List[Tuple[float, float]]:
        """
        使用三次样条插值平滑路径（参考AUTO-IceNav实现）
        
        Args:
            path: 原始路径点列表 [(x1,y1), (x2,y2), ...]
            step_size: 重采样步长（米），越小曲线越平滑
            smooth: True使用CubicSpline（平滑曲线），False使用线性插值（折线）
        
        Returns:
            平滑后的路径点列表
        """
        if len(path) < 3:
            return path  # 点太少无法平滑

        dedup = [path[0]]
        for p in path[1:]:
            if (abs(p[0] - dedup[-1][0]) > 1e-6) or (abs(p[1] - dedup[-1][1]) > 1e-6):
                dedup.append(p)
        if len(dedup) < 3:
            return dedup

        pts = np.array(dedup, dtype=np.float64)
        eps = max(step_size * 0.8, float(getattr(self.config, 'SHIP_LENGTH', 80.0)) * 0.05)
        simplified = self._rdp_simplify(pts, eps)
        if simplified.shape[0] >= 3:
            path = [(float(x), float(y)) for x, y in simplified]

        # 转换为numpy数组
        path_array = np.array(path)
        
        # 计算累积弧长（沿路径的距离）
        diffs = np.diff(path_array, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        
        total_length = arc_length[-1]
        if total_length < step_size:
            return path  # 路径太短

        method = 'cubic'
        try:
            algo = str(getattr(self.config, 'PATH_TYPE', '') or '')
            if algo == 'Ice-Theta*':
                method = str(getattr(self.config, 'ICE_THETA_POST_SMOOTH_METHOD', method) or method)
            method = str(getattr(self.config, 'PATH_SMOOTH_METHOD', method) or method)
        except Exception:
            method = 'cubic'
        method = str(method or 'cubic').lower()

        if smooth and method == 'chaikin':
            iters = 2
            try:
                iters = int(getattr(self.config, 'ICE_THETA_POST_SMOOTH_ITERS', iters))
            except Exception:
                iters = 2
            iters = max(0, min(6, iters))

            pts0 = path_array.astype(np.float64)
            for _ in range(iters):
                if pts0.shape[0] < 3:
                    break
                new_pts = [pts0[0]]
                for i in range(pts0.shape[0] - 1):
                    p = pts0[i]
                    q = pts0[i + 1]
                    a = 0.75 * p + 0.25 * q
                    b = 0.25 * p + 0.75 * q
                    new_pts.append(a)
                    new_pts.append(b)
                new_pts.append(pts0[-1])
                pts0 = np.vstack(new_pts)

            diffs2 = np.diff(pts0, axis=0)
            seg2 = np.sqrt(np.sum(diffs2 ** 2, axis=1))
            arc2 = np.concatenate([[0.0], np.cumsum(seg2)])
            total2 = float(arc2[-1])
            if total2 <= 1e-6:
                return path

            new_arc_lengths = np.arange(0.0, total2, float(step_size))
            if new_arc_lengths.size == 0:
                new_arc_lengths = np.array([0.0, total2], dtype=np.float64)
            if new_arc_lengths[-1] < total2:
                new_arc_lengths = np.append(new_arc_lengths, total2)

            smooth_x = np.interp(new_arc_lengths, arc2, pts0[:, 0])
            smooth_y = np.interp(new_arc_lengths, arc2, pts0[:, 1])
            smooth_path = [(float(x), float(y)) for x, y in zip(smooth_x, smooth_y)]

        else:
            # 使用样条插值创建平滑函数
            if smooth:
                # CubicSpline 三次样条 - 产生平滑曲线
                try:
                    cs_x = CubicSpline(arc_length, path_array[:, 0])
                    cs_y = CubicSpline(arc_length, path_array[:, 1])
                except ValueError:
                    # 如果有重复点导致插值失败，回退到线性
                    from scipy.interpolate import interp1d
                    cs_x = interp1d(arc_length, path_array[:, 0], kind='linear')
                    cs_y = interp1d(arc_length, path_array[:, 1], kind='linear')
            else:
                # 线性插值 - 保持折线
                from scipy.interpolate import interp1d
                cs_x = interp1d(arc_length, path_array[:, 0], kind='linear')
                cs_y = interp1d(arc_length, path_array[:, 1], kind='linear')

            # 按固定步长重采样
            new_arc_lengths = np.arange(0, total_length, step_size)
            # 确保终点被包含
            if new_arc_lengths[-1] < total_length:
                new_arc_lengths = np.append(new_arc_lengths, total_length)
            
            # 生成平滑路径点
            smooth_x = cs_x(new_arc_lengths)
            smooth_y = cs_y(new_arc_lengths)

            smooth_path = [(float(x), float(y)) for x, y in zip(smooth_x, smooth_y)]

        try:
            w = float(getattr(self, 'world_width', 0.0) or 0.0)
            h = float(getattr(self, 'world_height', 0.0) or 0.0)
            if w > 0 and h > 0:
                smooth_path = [
                    (min(max(0.0, x), w), min(max(0.0, y), h))
                    for (x, y) in smooth_path
                ]
        except Exception:
            pass
        
        print(f"   🔄 路径平滑: {len(path)}点 → {len(smooth_path)}点 (步长={step_size}m)")
        
        return smooth_path
    
    def generate_apf_path(self, ice_blocks: List[Dict]) -> List[Tuple[float, float]]:
        """
        加权A*路径规划（无障碍版）
        
        核心思想：
        - 不设置不可通行区域
        - 用代价权重表示冰块影响
        - 大冰块=高代价（自然绕行）
        - 小冰块=低代价（可穿越）
        - 保证100%找到路径
        
        Returns:
            路径点列表
        """
        print(f"   🎯 正在运行加权A*算法（保证找到路径）...")
        
        # 起点和终点
        start_x = self.config.SHIP_START_X * self.world_width
        start_y = self.config.SHIP_START_Y * self.world_height
        goal_x = self.config.GOAL_X * self.world_width
        goal_y = self.config.GOAL_Y * self.world_height
        
        # ========== 1. 创建代价地图 ==========
        # 使用较大栅格以加快速度
        grid_size = max(10.0, min(self.world_width, self.world_height) / 80)
        cols = int(self.world_width / grid_size)
        rows = int(self.world_height / grid_size)
        
        print(f"      栅格: {cols}x{rows}, 分辨率: {grid_size:.1f}m")
        
        # 代价地图：基础代价=1，冰块区域代价更高
        cost_map = np.ones((cols, rows), dtype=np.float32)
        
        # 冰块类型的代价权重
        type_weights = {
            'Ice Bank': 50.0,
            'Vast Floe': 30.0,
            'Large Floe': 15.0,
            'Medium Floe': 5.0,
            'Small Floe': 2.0,
            'Fragment': 1.0,
            'Brash Ice': 0.5,
        }
        
        # 填充代价地图
        for ice in ice_blocks:
            cx, cy = ice['center']
            size = ice.get('size', 5.0)
            ice_type = ice.get('type', 'Medium Floe')
            
            weight = type_weights.get(ice_type, 5.0)
            
            # 影响半径
            influence_radius = size * 1.5
            grid_radius = int(influence_radius / grid_size) + 1
            
            gx = int(cx / grid_size)
            gy = int(cy / grid_size)
            
            # 在影响范围内增加代价
            for dx in range(-grid_radius, grid_radius + 1):
                for dy in range(-grid_radius, grid_radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < cols and 0 <= ny < rows:
                        dist = np.sqrt(dx**2 + dy**2) * grid_size
                        if dist < influence_radius:
                            # 代价随距离衰减
                            added_cost = weight * (1 - dist / influence_radius)
                            cost_map[nx, ny] += added_cost
        
        # ========== 2. A*搜索（无障碍，只有代价差异）==========
        start_node = (int(start_x / grid_size), int(start_y / grid_size))
        goal_node = (int(goal_x / grid_size), int(goal_y / grid_size))
        
        start_node = (max(0, min(cols-1, start_node[0])), max(0, min(rows-1, start_node[1])))
        goal_node = (max(0, min(cols-1, goal_node[0])), max(0, min(rows-1, goal_node[1])))
        
        # 优先队列
        open_set = []
        heapq.heappush(open_set, (0, 0, start_node[0], start_node[1]))
        
        came_from = {}
        g_score = {start_node: 0}
        
        # 8方向移动
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        iterations = 0
        max_iterations = cols * rows * 2
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current_g, cx, cy = heapq.heappop(open_set)
            current_pos = (cx, cy)
            
            if current_pos == goal_node:
                break
            
            if g_score.get(current_pos, float('inf')) < current_g:
                continue
            
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                
                if not (0 <= nx < cols and 0 <= ny < rows):
                    continue
                
                # 移动代价 = 基础代价 × 目标格子的代价
                move_cost = (1.414 if (dx != 0 and dy != 0) else 1.0) * cost_map[nx, ny]
                tentative_g = current_g + move_cost
                
                neighbor_pos = (nx, ny)
                if tentative_g < g_score.get(neighbor_pos, float('inf')):
                    came_from[neighbor_pos] = current_pos
                    g_score[neighbor_pos] = tentative_g
                    
                    # 启发式：欧几里得距离
                    h = np.sqrt((nx - goal_node[0])**2 + (ny - goal_node[1])**2)
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, tentative_g, nx, ny))
        
        # ========== 3. 回溯路径 ==========
        path = []
        current = goal_node
        while current in came_from:
            wx = current[0] * grid_size + grid_size / 2
            wy = current[1] * grid_size + grid_size / 2
            path.append((wx, wy))
            current = came_from[current]
        path.append((start_x, start_y))
        path.reverse()
        
        # 确保终点
        path.append((goal_x, goal_y))
        
        print(f"   ✓ 加权A*完成! 路径点: {len(path)}, 迭代: {iterations}")
        return path
    
    def generate_rrt_path(self, ice_blocks: List[Dict], max_iterations: int = 8000) -> List[Tuple[float, float]]:
        """
        RRT* (Rapidly-exploring Random Tree Star) 路径规划 [备用]
        
        优势：
        - 在密集障碍物环境中更可靠
        - 渐进式优化，路径质量高
        - 不依赖栅格分辨率
        
        Args:
            ice_blocks: 冰块列表
            max_iterations: 最大迭代次数
            
        Returns:
            路径点列表
        """
        print(f"   🌳 正在运行RRT*算法 (最大迭代: {max_iterations})...")
        
        # 起点和终点
        start_x = self.config.SHIP_START_X * self.world_width
        start_y = self.config.SHIP_START_Y * self.world_height
        goal_x = self.config.GOAL_X * self.world_width
        goal_y = self.config.GOAL_Y * self.world_height
        
        start = (start_x, start_y)
        goal = (goal_x, goal_y)
        
        # 构建障碍物列表 (x, y, radius) - 只考虑大冰块
        obstacles = []
        ship_half_width = self.config.SHIP_WIDTH / 2.0
        
        for ice in ice_blocks:
            cx, cy = ice['center']
            size = ice.get('size', 5.0)
            ice_type = ice.get('type', 'Medium Floe')
            
            # 只标记大型冰块为障碍物，小冰块可以撞开
            if ice_type in ['Ice Bank', 'Vast Floe']:
                # 大型冰块：必须绕行
                radius = size * 0.5 + ship_half_width
                obstacles.append((cx, cy, radius))
            elif ice_type == 'Large Floe':
                # 较大冰块：尽量绕行
                radius = size * 0.4 + ship_half_width * 0.8
                obstacles.append((cx, cy, radius))
            elif ice_type == 'Medium Floe' and size > 8:
                # 中等冰块：只有较大的才标记
                radius = size * 0.3 + ship_half_width * 0.5
                obstacles.append((cx, cy, radius))
            # 小冰块和碎冰：忽略（船可以撞开）
        
        print(f"      障碍物数量: {len(obstacles)} (大型冰块)")
        
        # 如果障碍物很少，尝试直接连接
        if len(obstacles) < 10:
            print(f"   ✓ 障碍物少，使用直线路径")
            return [(start_x, start_y), (goal_x, goal_y)]
        
        # RRT* 参数 - 优化为更激进的探索
        step_size = min(self.world_width, self.world_height) / 30.0  # 更大步长
        goal_sample_rate = 0.15  # 15%概率直接采样目标
        goal_threshold = step_size * 3  # 更大的到达阈值
        neighbor_radius = step_size * 2  # 邻域半径
        
        # 节点类: (x, y, parent_index, cost)
        nodes = [(start[0], start[1], -1, 0.0)]
        
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def is_collision_free(p1, p2):
            """检查两点之间是否无碰撞（优化版）"""
            dist = distance(p1, p2)
            if dist < 0.1:
                return True
            
            # 边界检查（起点和终点）
            if p2[0] < 5 or p2[0] > self.world_width - 5:
                return False
            if p2[1] < 5 or p2[1] > self.world_height - 5:
                return False
            
            # 计算线段中点
            mx = (p1[0] + p2[0]) / 2
            my = (p1[1] + p2[1]) / 2
            
            # 只检查线段附近的障碍物（加速）
            check_radius = dist / 2 + step_size
            
            for ox, oy, r in obstacles:
                # 快速排除远处障碍物
                if abs(ox - mx) > check_radius + r or abs(oy - my) > check_radius + r:
                    continue
                
                # 点到线段的最短距离检查
                # 使用简化的检查：只检查起点、中点、终点
                for px, py in [(p1[0], p1[1]), (mx, my), (p2[0], p2[1])]:
                    if (px - ox)**2 + (py - oy)**2 < r**2:
                        return False
            
            return True
        
        def steer(from_node, to_point):
            """从节点向目标点延伸一步"""
            fx, fy = from_node[0], from_node[1]
            tx, ty = to_point[0], to_point[1]
            
            dist = distance((fx, fy), (tx, ty))
            if dist <= step_size:
                return (tx, ty)
            
            theta = np.arctan2(ty - fy, tx - fx)
            new_x = fx + step_size * np.cos(theta)
            new_y = fy + step_size * np.sin(theta)
            return (new_x, new_y)
        
        goal_node_index = -1
        
        # 计算起点到终点的方向（用于偏向采样）
        dir_x = goal[0] - start[0]
        dir_y = goal[1] - start[1]
        path_length = np.sqrt(dir_x**2 + dir_y**2)
        
        for iteration in range(max_iterations):
            # 智能采样策略
            rand = np.random.random()
            if rand < goal_sample_rate:
                # 直接采样目标
                sample = goal
            elif rand < 0.5:
                # 在起点-终点走廊内采样（更高效）
                t = np.random.random()  # 沿路径的位置
                corridor_width = min(self.world_width, self.world_height) * 0.3
                
                # 走廊中心线上的点
                center_x = start[0] + t * dir_x
                center_y = start[1] + t * dir_y
                
                # 垂直于路径方向的偏移
                perp_x = -dir_y / path_length
                perp_y = dir_x / path_length
                offset = np.random.uniform(-corridor_width/2, corridor_width/2)
                
                sample = (
                    np.clip(center_x + offset * perp_x, 5, self.world_width - 5),
                    np.clip(center_y + offset * perp_y, 5, self.world_height - 5)
                )
            else:
                # 全局随机采样
                sample = (
                    np.random.uniform(5, self.world_width - 5),
                    np.random.uniform(5, self.world_height - 5)
                )
            
            # 找最近节点
            min_dist = float('inf')
            nearest_idx = 0
            for i, node in enumerate(nodes):
                d = distance((node[0], node[1]), sample)
                if d < min_dist:
                    min_dist = d
                    nearest_idx = i
            
            nearest_node = nodes[nearest_idx]
            
            # 向采样点延伸
            new_point = steer(nearest_node, sample)
            
            # 碰撞检测
            if not is_collision_free((nearest_node[0], nearest_node[1]), new_point):
                continue
            
            # RRT* 优化：在邻域内找最优父节点
            new_cost = nearest_node[3] + distance((nearest_node[0], nearest_node[1]), new_point)
            best_parent_idx = nearest_idx
            best_cost = new_cost
            
            # 搜索邻域
            neighbor_indices = []
            for i, node in enumerate(nodes):
                if distance((node[0], node[1]), new_point) < neighbor_radius:
                    neighbor_indices.append(i)
            
            for idx in neighbor_indices:
                node = nodes[idx]
                potential_cost = node[3] + distance((node[0], node[1]), new_point)
                if potential_cost < best_cost:
                    if is_collision_free((node[0], node[1]), new_point):
                        best_parent_idx = idx
                        best_cost = potential_cost
            
            # 添加新节点
            new_node = (new_point[0], new_point[1], best_parent_idx, best_cost)
            new_node_idx = len(nodes)
            nodes.append(new_node)
            
            # 检查是否到达目标
            if distance(new_point, goal) < goal_threshold:
                if is_collision_free(new_point, goal):
                    # 添加目标节点
                    goal_cost = best_cost + distance(new_point, goal)
                    nodes.append((goal[0], goal[1], new_node_idx, goal_cost))
                    goal_node_index = len(nodes) - 1
                    print(f"   ✓ RRT*找到路径! 迭代: {iteration}, 节点数: {len(nodes)}")
                    break
            
            # 进度报告
            if iteration > 0 and iteration % 1000 == 0:
                print(f"      RRT*进度: {iteration}/{max_iterations}, 节点: {len(nodes)}")
        
        # 回溯路径
        if goal_node_index >= 0:
            path = []
            idx = goal_node_index
            while idx >= 0:
                node = nodes[idx]
                path.append((node[0], node[1]))
                idx = node[2]
            path.reverse()
            return path
        else:
            print(f"   ⚠️ RRT*未找到路径，回退到Skeleton")
            return self.generate_skeleton_path(ice_blocks)
    
    # ========== SAM真实冰场加载 (2025-12-17新增) ==========
    
    def load_from_sam_json(self, json_path: str, 
                           scale_to_world: bool = True,
                           max_ice_blocks: int = None) -> List[Dict]:
        """
        从SAM分割结果JSON加载真实冰场
        
        Args:
            json_path: final_ice_field.json 的路径
            scale_to_world: 是否缩放到仿真器世界尺寸
            max_ice_blocks: 最大冰块数量限制 (None=不限制)
        
        Returns:
            List[Dict]: 仿真器格式的冰块列表
        """
        import json
        import time
        
        print(f"\n🧊 [SAM Loader] 从真实冰场加载...")
        print(f"   文件: {json_path}")
        start_time = time.time()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            sam_data = json.load(f)
        
        # 元数据
        img_h, img_w = sam_data['image_size']  # 像素
        meters_per_pixel = sam_data['meters_per_pixel']
        physical_width = img_w * meters_per_pixel   # 真实宽度(米)
        physical_height = img_h * meters_per_pixel  # 真实高度(米)
        
        print(f"   原始尺寸: {img_w}×{img_h} px")
        print(f"   物理尺寸: {physical_width:.0f}×{physical_height:.0f} m")
        print(f"   冰块总数: {sam_data['total_ice_blocks']}")
        
        # 计算缩放比例 (如果需要适配仿真器世界)
        if scale_to_world:
            scale_x = self.world_width / physical_width
            scale_y = self.world_height / physical_height
            # 使用较小的缩放比例保持比例一致
            scale = min(scale_x, scale_y)
            print(f"   缩放比例: {scale:.4f} (适配 {self.world_width}×{self.world_height} m)")
        else:
            scale = 1.0
            print(f"   使用原始尺度: 1:1 (无缩放)")

        enable_large_region_filter = bool(getattr(self.config, 'FILTER_SAM_LARGE_REGIONS', False))
        max_valid_ice_area = getattr(self.config, 'SAM_MAX_VALID_ICE_AREA', None)
        if enable_large_region_filter and max_valid_ice_area:
            original_count = len(sam_data['ice_blocks'])
            sam_data['ice_blocks'] = [b for b in sam_data['ice_blocks'] if b['area_m2'] < max_valid_ice_area]
            filtered_count = original_count - len(sam_data['ice_blocks'])
            if filtered_count > 0:
                print(f"   ⚠️ 过滤 {filtered_count} 个异常大区域 (可能是水域误识别)")
        
        # 冰块类型推断阈值 (基于面积 m²) - 适配真实尺度 (船100-200m)
        # 面积阈值基于等效边长: Fragment<22m, Small<70m, Medium<220m, Large<700m
        TYPE_THRESHOLDS = {
            'Fragment': (0, 500),           # <22m边长, 可推开
            'Small Floe': (500, 5000),      # 22-70m, 阻碍航行
            'Medium Floe': (5000, 50000),   # 70-220m, 需绕行
            'Large Floe': (50000, 500000),  # 220-700m, 强烈避开
        }
        
        # 冰块物理属性 (厚度m, 密度kg/m³)
        ICE_PROPERTIES = {
            'Fragment':    {'thickness': 0.3, 'density': 800},
            'Small Floe':  {'thickness': 0.5, 'density': 850},
            'Medium Floe': {'thickness': 0.8, 'density': 900},
            'Large Floe':  {'thickness': 1.2, 'density': 900},
            'Ice Bank':    {'thickness': 2.0, 'density': 900},
        }
        
        ice_blocks = []
        sam_ice_list = sam_data['ice_blocks']
        
        # 限制冰块数量
        if max_ice_blocks and len(sam_ice_list) > max_ice_blocks:
            # 按面积排序，保留最大的冰块
            sam_ice_list = sorted(sam_ice_list, key=lambda x: x['area_m2'], reverse=True)
            sam_ice_list = sam_ice_list[:max_ice_blocks]
            print(f"   限制冰块数: {max_ice_blocks}")
        
        for ice_data in sam_ice_list:
            points = ice_data['points']      # [[x1,y1], [x2,y2], ...]
            area_m2 = ice_data['area_m2']
            center_px = ice_data['center']   # [cx, cy] 像素
            
            # 1. 推断冰块类型
            ice_type = 'Large Floe'  # 默认为最大类型
            for t, (lo, hi) in TYPE_THRESHOLDS.items():
                if lo <= area_m2 < hi:
                    ice_type = t
                    break
            
            # 2. 像素坐标 → 物理坐标 (米)
            # 仿真器渲染坐标系与图像一致（Y轴向下），不需要翻转
            cx_m = center_px[0] * meters_per_pixel * scale
            cy_m = (img_h - center_px[1]) * meters_per_pixel * scale
            
            # 3. 顶点: 绝对像素 → 相对于中心的偏移 (米)
            vertices = []
            for px, py in points:
                # 转为相对于中心的偏移
                dx = (px - center_px[0]) * meters_per_pixel * scale
                dy = (center_px[1] - py) * meters_per_pixel * scale  # 不翻转Y
                vertices.append((dx, dy))
            
            # 4. 计算等效半径和尺寸
            if vertices:
                radius = max([np.sqrt(vx**2 + vy**2) for vx, vy in vertices])
            else:
                radius = np.sqrt(area_m2 / np.pi)  # 假设圆形
            
            size = radius * 2  # 等效直径

            immovable_threshold = float(getattr(self.config, 'IMMOVABLE_ICE_THRESHOLD_SIZE', 0.0))
            if immovable_threshold > 0 and size >= immovable_threshold:
                ice_type = 'Ice Bank'
            
            # 5. 计算质量 (mass = area × thickness × density)
            props = ICE_PROPERTIES.get(ice_type, {'thickness': 0.5, 'density': 900})
            scaled_area = area_m2 * (scale ** 2)  # 缩放后的面积
            mass = scaled_area * props['thickness'] * props['density']
            
            # 6. 边界检查
            if cx_m < 0 or cx_m > self.world_width or cy_m < 0 or cy_m > self.world_height:
                continue
            
            # 7. 构建仿真器格式
            ice_block = {
                'type': ice_type,
                'center': (cx_m, cy_m),
                'vertices': vertices,
                'size': size,
                'mass': mass,
                'area': area_m2 * (scale ** 2),
                'radius': radius,
                'source': 'SAM'  # 标记来源
            }
            
            ice_blocks.append(ice_block)
        
        elapsed = time.time() - start_time
        print(f"   ✓ 加载 {len(ice_blocks)} 个冰块 ({elapsed:.2f}s)")
        
        # 统计各类型
        type_counts = {}
        for ice in ice_blocks:
            t = ice['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        for ice_type, count in sorted(type_counts.items()):
            print(f"      {ice_type}: {count}")
        
        return ice_blocks
    
    # ========== Tile Masks加载  ==========
    
    def load_from_tile_masks(self, folder_path: str, 
                              meters_per_pixel: float = 0.89,
                              max_ice_blocks: int = None,
                              min_area_pixels: int = 50) -> List[Dict]:
        """
        
        
        Args:
            folder_path: output_highres/xxx/ 文件夹路径
            meters_per_pixel: 每像素米数 (默认0.89)
            max_ice_blocks: 最大冰块数量限制
            min_area_pixels: 最小冰块面积(像素)
        """
        import cv2
        import time
        from pathlib import Path
        
        folder = Path(folder_path)
        masks_dir = folder / "masks"
        meta_file = folder / "tiles_meta.json"
        
        print(f"\n🧊 [Tile Loader] 直接从tile masks加载...")
        print(f"   文件夹: {folder_path}")
        start_time = time.time()
        
        # 读取元数据
        import json
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        img_w = meta['image_size'][1]  # width
        img_h = meta['image_size'][0]  # height
        tiles = meta['tiles']
        
        print(f"   图像尺寸: {img_w}×{img_h} px")
        print(f"   Tile数量: {len(tiles)}")
        
        # 冰块类型阈值
        TYPE_THRESHOLDS = {
            'Fragment': (0, 500),
            'Small Floe': (500, 5000),
            'Medium Floe': (5000, 50000),
            'Large Floe': (50000, 500000),
        }
        
        ICE_PROPERTIES = {
            'Fragment':    {'thickness': 0.3, 'density': 800},
            'Small Floe':  {'thickness': 0.5, 'density': 850},
            'Medium Floe': {'thickness': 0.8, 'density': 900},
            'Large Floe':  {'thickness': 1.2, 'density': 900},
        }
        
        all_ice_blocks = []
        
        for tile_info in tiles:
            tile_name = tile_info['name']
            x_off = tile_info['x_offset']
            y_off = tile_info['y_offset']
            
            mask_file = masks_dir / f"{tile_name}_mask.png"
            if not mask_file.exists():
                continue
            
            # 读取mask
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            # 连通域分析 
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for label_id in range(1, num_labels):
                area_px = stats[label_id, cv2.CC_STAT_AREA]
                
                if area_px < min_area_pixels:
                    continue
                
                # 局部中心 -> 全局中心
                local_cx, local_cy = centroids[label_id]
                global_cx = local_cx + x_off
                global_cy = local_cy + y_off
                
                # 提取轮廓
                component_mask = (labels == label_id).astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                cnt = max(contours, key=cv2.contourArea)
                approx = cv2.approxPolyDP(cnt, 1.5, True)
                
                # 转换为全局坐标的顶点 (相对于中心)
                vertices = []
                for pt in approx.reshape(-1, 2):
                    global_x = pt[0] + x_off
                    global_y = pt[1] + y_off
                    # 相对于中心的偏移 (米)
                    dx = (global_x - global_cx) * meters_per_pixel
                    dy = (global_y - global_cy) * meters_per_pixel
                    vertices.append((dx, dy))
                
                # 面积和类型
                area_m2 = area_px * (meters_per_pixel ** 2)
                
                ice_type = 'Large Floe'
                for t, (lo, hi) in TYPE_THRESHOLDS.items():
                    if lo <= area_m2 < hi:
                        ice_type = t
                        break
                
                # 物理属性
                props = ICE_PROPERTIES.get(ice_type, {'thickness': 0.5, 'density': 900})
                mass = area_m2 * props['thickness'] * props['density']
                
                # 中心坐标 (米)
                cx_m = global_cx * meters_per_pixel
                cy_m = global_cy * meters_per_pixel
                
                # 等效半径
                if vertices:
                    radius = max([np.sqrt(vx**2 + vy**2) for vx, vy in vertices])
                else:
                    radius = np.sqrt(area_m2 / np.pi)
                
                all_ice_blocks.append({
                    'type': ice_type,
                    'center': (cx_m, cy_m),
                    'vertices': vertices,
                    'mass': mass,
                    'size': radius * 2,
                    'radius': radius,
                    'area_m2': area_m2,
                })
        
        # 限制数量
        if max_ice_blocks and len(all_ice_blocks) > max_ice_blocks:
            all_ice_blocks = sorted(all_ice_blocks, key=lambda x: x['area_m2'], reverse=True)
            all_ice_blocks = all_ice_blocks[:max_ice_blocks]
            print(f"   限制冰块数: {max_ice_blocks}")
        
        elapsed = time.time() - start_time
        print(f"   ✓ 加载 {len(all_ice_blocks)} 个冰块 ({elapsed:.2f}s)")
        
        # 统计
        type_counts = {}
        for ice in all_ice_blocks:
            t = ice['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        for ice_type, count in sorted(type_counts.items()):
            print(f"      {ice_type}: {count}")
        
        return all_ice_blocks
    
    # ========== Ice-Aware Lattice Planner (2025-12-18新增) ==========
    
    def generate_lattice_path(self, ice_blocks: List[Dict]) -> List[Tuple[float, float]]:
        """
        Ice-Aware Theta* Planner - 冰况感知的任意角度路径规划
        
        核心创新 (基于Theta*算法 - Nash et al. 2007):
        1. 任意角度路径规划，不受栅格方向限制
        2. 视线检查(Line-of-Sight)实现路径平滑
        3. 代价函数融合: 冰阻力 + 碰撞风险 + 距离
        4. 后处理添加船舶运动学约束
        
        优势: 比传统Lattice更可靠，路径更平滑
        
        Returns:
            路径点列表 [(x, y), ...]
        """
        print(f"   🧊 正在运行 APF-A* 冰区路径规划 (人工势场+A*)...")
        
        # 起点和终点
        start_x = self.config.SHIP_START_X * self.world_width
        start_y = self.config.SHIP_START_Y * self.world_height
        goal_x = self.config.GOAL_X * self.world_width
        goal_y = self.config.GOAL_Y * self.world_height
        
        ship_length = float(getattr(self.config, 'SHIP_LENGTH', 80.0))
        
        # ========== 1. 构建人工势场 (APF) ==========
        # 参考: Khatib (1986) "Real-Time Obstacle Avoidance"
        # 冰区航行论文: arXiv:2409.11326 "Autonomous Navigation in Ice-Covered Waters"
        
        grid_size = max(15.0, min(self.world_width, self.world_height) / 80)  # 增大网格提高性能
        cols = int(self.world_width / grid_size) + 1
        rows = int(self.world_height / grid_size) + 1
        print(f"      栅格: {cols}x{rows} (分辨率: {grid_size:.1f}m)")
        
        # 势场地图: U_total = U_att (吸引势) + U_rep (排斥势)
        potential_map = np.zeros((cols, rows), dtype=np.float32)
        
        # ========== 1.1 吸引势场 U_att ==========
        # U_att = 0.5 * k_att * d^2 (目标点吸引)
        k_att = 1.0  # 吸引系数
        goal_gx, goal_gy = int(goal_x / grid_size), int(goal_y / grid_size)
        goal_gx = max(0, min(cols-1, goal_gx))
        goal_gy = max(0, min(rows-1, goal_gy))
        
        # 计算起点到目标的主方向（用于防止路径绕远）
        start_to_goal_dx = goal_x - start_x
        start_to_goal_dy = goal_y - start_y
        start_to_goal_len = np.sqrt(start_to_goal_dx**2 + start_to_goal_dy**2)
        if start_to_goal_len > 1:
            goal_dir_x = start_to_goal_dx / start_to_goal_len
            goal_dir_y = start_to_goal_dy / start_to_goal_len
        else:
            goal_dir_x, goal_dir_y = 1.0, 0.0
        
        for i in range(cols):
            for j in range(rows):
                dist_to_goal = np.sqrt((i - goal_gx)**2 + (j - goal_gy)**2)
                potential_map[i, j] = 0.5 * k_att * dist_to_goal  # 吸引势
        
        # ========== 1.2 排斥势场 U_rep (冰块) ==========
        # U_rep = 0.5 * k_rep * (1/d - 1/d0)^2  当 d < d0
        # 参考冰块大小调整排斥强度
        
        # 冰块排斥系数 (按类型) - 拉大差距，中大型冰必须避开
        ice_repulsion = {
            'Ice Bank': 50000.0,   # 冰岸/大冰山: 必须绕行!
            'Vast Floe': 30000.0,  # 巨冰: 必须绕行
            'Large Floe': 15000.0, # 大冰: 必须避开
            'Medium Floe': 8000.0, # 中冰: 强烈避开 (提高4倍)
            'Small Floe': 5.0,     # 小冰: 几乎无代价
            'Fragment': 0.5,       # 碎冰: 优先通过
        }
        
        # 边界排斥 - 硬边界，必须在冰块之前设置
        boundary_margin = 8  # 边界安全距离（格子数）
        boundary_map = np.zeros((cols, rows), dtype=bool)  # 硬边界标记
        for i in range(cols):
            for j in range(rows):
                dist_to_edge = min(i, j, cols - 1 - i, rows - 1 - j)
                if dist_to_edge < 2:
                    # 硬边界：不可通过
                    boundary_map[i, j] = True
                    potential_map[i, j] = 1e9
                elif dist_to_edge < boundary_margin:
                    # 软边界：高代价
                    potential_map[i, j] += 100000.0 * (1.0 - dist_to_edge / boundary_margin)
        
        # 冰块排斥势
        for ice in ice_blocks:
            cx, cy = ice['center']
            size = ice.get('size', ice.get('radius', 10) * 2)
            ice_type = ice.get('type', 'Medium Floe')
            k_rep = ice_repulsion.get(ice_type, 50.0)
            
            # 排斥影响半径 - 按冰块大小分层
            ice_radius = size / 2
            if ice_type == 'Ice Bank' or size > ship_length * 3:
                # 大冰山: 影响范围 = 冰块半径 + 4倍船长
                d0 = (ice_radius + ship_length * 4.0) / grid_size
            elif ice_type in ['Vast Floe', 'Large Floe']:
                # 大冰: 影响范围 = 冰块半径 + 3倍船长
                d0 = (ice_radius + ship_length * 3.0) / grid_size
            elif ice_type == 'Medium Floe':
                # 中冰: 影响范围 = 冰块半径 + 2.5倍船长
                d0 = (ice_radius + ship_length * 2.5) / grid_size
            else:
                # 小冰/碎冰: 影响范围 = 冰块半径 + 1倍船长 (允许靠近)
                d0 = (ice_radius + ship_length * 1.0) / grid_size
            
            gx, gy = int(cx / grid_size), int(cy / grid_size)
            r_cells = int(d0) + 2
            
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < cols and 0 <= ny < rows:
                        # 到冰块中心的距离(格子数)
                        d = np.sqrt(dx*dx + dy*dy)
                        # 减去冰块半径得到到冰块边缘的距离
                        d_edge = max(0.1, d - ice_radius/grid_size)
                        
                        if d_edge < d0:
                            # 排斥势 (指数衰减,更强的排斥)
                            rep = k_rep * np.exp(-d_edge / (d0 * 0.3))
                            potential_map[nx, ny] += rep
        
        print(f"      势场范围: {potential_map.min():.1f} - {potential_map.max():.1f}")
        
        start_grid = (int(start_x / grid_size), int(start_y / grid_size))
        goal_grid = (int(goal_x / grid_size), int(goal_y / grid_size))
        # 确保起点终点在有效范围内（远离硬边界）
        start_grid = (max(3, min(cols-3, start_grid[0])), max(3, min(rows-3, start_grid[1])))
        goal_grid = (max(3, min(cols-3, goal_grid[0])), max(3, min(rows-3, goal_grid[1])))
        
        # ========== 2. A* 搜索 (基于势场代价 + 转向惩罚) ==========
        def heuristic(node):
            # 欧几里得距离作为启发式
            return np.sqrt((node[0] - goal_grid[0])**2 + (node[1] - goal_grid[1])**2)
        
        def get_potential_cost(n1, n2):
            """简化的代价函数：距离 + 势场 + 方向"""
            dist = np.sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)
            pot = potential_map[n2[0], n2[1]]
            
            # 方向惩罚：鼓励朝目标方向移动
            dx = n2[0] - n1[0]
            dy = n2[1] - n1[1]
            move_len = np.sqrt(dx*dx + dy*dy)
            if move_len > 0.01:
                dot = (dx * goal_dir_x + dy * goal_dir_y) / move_len
                if dot < 0:
                    direction_penalty = 10.0  # 反方向惩罚
                else:
                    direction_penalty = 0
            else:
                direction_penalty = 0
            
            # 势场代价（简化）
            pot_cost = pot * 0.1
            
            return dist + pot_cost + direction_penalty
        
        def distance(n1, n2):
            return np.sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)
        
        # A* 主循环 (基于势场 + 转向惩罚)
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {start_grid: start_grid}
        g_score = {start_grid: 0}
        closed_set = set()
        
        # 8方向邻居
        neighbors_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        
        max_iterations = 50000  # 减少迭代上限提高性能
        iterations = 0
        found = False
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            closed_set.add(current)
            
            # 到达目标
            if current == goal_grid or distance(current, goal_grid) < 3:
                if current != goal_grid:
                    goal_grid = current
                found = True
                break
            
            for dn in neighbors_8:
                neighbor = (current[0] + dn[0], current[1] + dn[1])
                
                # 边界检查
                if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                    continue
                # 硬边界检查
                if boundary_map[neighbor[0], neighbor[1]]:
                    continue
                if neighbor in closed_set:
                    continue
                
                # 基于势场的代价计算（简化版）
                new_g = g_score[current] + get_potential_cost(current, neighbor)
                
                if new_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = new_g
                    f = new_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f, neighbor))
        
        # ========== 4. 回溯路径 ==========
        if found:
            path_grid = []
            current = goal_grid
            while current != start_grid:
                path_grid.append(current)
                current = came_from.get(current, start_grid)
            path_grid.append(start_grid)
            path_grid.reverse()
            
            # 转换为世界坐标
            path = [(g[0] * grid_size + grid_size/2, g[1] * grid_size + grid_size/2) for g in path_grid]
            
            # ========== 路径平滑处理 ==========
            # 第1步: 移除冗余点（共线点）
            def remove_collinear(pts, threshold=0.1):
                """移除近似共线的中间点"""
                if len(pts) <= 2:
                    return pts
                result = [pts[0]]
                for i in range(1, len(pts) - 1):
                    p0, p1, p2 = result[-1], pts[i], pts[i+1]
                    # 计算向量
                    v1 = (p1[0]-p0[0], p1[1]-p0[1])
                    v2 = (p2[0]-p1[0], p2[1]-p1[1])
                    len1 = np.sqrt(v1[0]**2 + v1[1]**2)
                    len2 = np.sqrt(v2[0]**2 + v2[1]**2)
                    if len1 > 0.1 and len2 > 0.1:
                        # 叉积判断共线
                        cross = abs(v1[0]*v2[1] - v1[1]*v2[0]) / (len1 * len2)
                        if cross > threshold:  # 不共线，保留
                            result.append(p1)
                    else:
                        result.append(p1)
                result.append(pts[-1])
                return result
            
            # 第2步: 贝塞尔曲线平滑
            def bezier_smooth(pts, num_points=3):
                """在转折点之间插入平滑过渡"""
                if len(pts) <= 2:
                    return pts
                result = [pts[0]]
                for i in range(1, len(pts) - 1):
                    p0, p1, p2 = pts[i-1], pts[i], pts[i+1]
                    # 计算转向角
                    v1 = (p1[0]-p0[0], p1[1]-p0[1])
                    v2 = (p2[0]-p1[0], p2[1]-p1[1])
                    len1 = np.sqrt(v1[0]**2 + v1[1]**2)
                    len2 = np.sqrt(v2[0]**2 + v2[1]**2)
                    if len1 > 1 and len2 > 1:
                        cos_a = (v1[0]*v2[0] + v1[1]*v2[1]) / (len1 * len2)
                        cos_a = np.clip(cos_a, -1, 1)
                        angle = np.arccos(cos_a)
                        # 转向大于30度时插入平滑点
                        if angle > np.radians(30):
                            # 二次贝塞尔曲线控制点
                            for t in np.linspace(0.2, 0.8, num_points):
                                bx = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
                                by = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
                                result.append((bx, by))
                        else:
                            result.append(p1)
                    else:
                        result.append(p1)
                result.append(pts[-1])
                return result
            
            # 第3步: 移除回头路和环路（确保始终向目标前进）
            def remove_loops_and_backtrack(pts, goal):
                """移除路径中的回头路和环路"""
                if len(pts) <= 2:
                    return pts
                
                goal_x, goal_y = goal
                result = [pts[0]]
                
                for i in range(1, len(pts)):
                    p_curr = pts[i]
                    p_last = result[-1]
                    
                    # 计算当前点和上一个保留点到目标的距离
                    dist_curr = np.sqrt((p_curr[0]-goal_x)**2 + (p_curr[1]-goal_y)**2)
                    dist_last = np.sqrt((p_last[0]-goal_x)**2 + (p_last[1]-goal_y)**2)
                    
                    # 只保留朝目标前进的点（允许小幅回退）
                    if dist_curr < dist_last + 50:  # 允许50米的容差
                        # 检查是否与之前的点形成环路
                        is_loop = False
                        for j in range(max(0, len(result)-10), len(result)-1):
                            p_prev = result[j]
                            d = np.sqrt((p_curr[0]-p_prev[0])**2 + (p_curr[1]-p_prev[1])**2)
                            if d < 30:  # 与30米内的旧点重合 = 环路
                                is_loop = True
                                # 直接跳到这个点，删除中间的环
                                result = result[:j+1]
                                break
                        
                        if not is_loop:
                            result.append(p_curr)
                
                # 确保终点在路径中
                if len(result) > 0 and pts[-1] != result[-1]:
                    result.append(pts[-1])
                
                return result
            
            # 简化平滑：只移除共线点，不做复杂处理
            simplified = remove_collinear(path, 0.1)
            
            print(f"   ✓ APF-A* 路径找到! 原始: {len(path_grid)}, 简化: {len(simplified)}, 迭代: {iterations}")
            return simplified
        else:
            print(f"   ⚠️ Theta* 未找到路径(迭代{iterations}次)，回退到A*...")
            return self.generate_a_star_path(ice_blocks)
