"""复刻 calculate_drag_coefficient_v2.m 的核心求解流程。

使用 SciPy 的 LU 分解反复求解同一个线性系统；矩阵和右端项仍与 MATLAB 相同。
LU 分解是把矩阵拆成便于求解的两个三角矩阵，不改变梁方程或边界条件。
"""

from __future__ import annotations

import math
import warnings
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.linalg import lu_factor, lu_solve

from . import defaults as cfg
from .numerics import calculate_shielding_coefficient, compute_angle_with_large_deformation, finite_difference_matrix
from .types import ColumnDeformation, ConvergenceWarning, DragResult, NumericalOptions, NumericalStabilityWarning


def _validate_inputs(
    D: float, H: float, E: float, L: float, t: float, h: float, H_soft: float,
    b: float, U: float, rho: float, Cd_soft: float, Cd_cyl: float,
    theta_deg: ArrayLike, max_iter: int, tol: float, options: NumericalOptions,
    shielding_func_type: str, min_shielding: float, max_shielding: float,
) -> np.ndarray:
    """验证外界输入，返回未经归一化的一维安装角度数组。

    参数采用 MATLAB 名称：长度单位 m，E 单位 Pa，U 单位 m/s，rho 单位 kg/m³。
    max_iter 为非负整数；tol 为位移残差容差；options 为数值网格及每列数量设置。
    拒绝非法参数，避免把 NaN、无穷值或不存在的叶片几何送入迭代器。
    """
    # 叶片、材料、密度和间距必须为正；圆柱尺寸、覆盖高度和流速可为零。
    positive = {"E": E, "L": L, "t": t, "h": h, "b": b, "rho": rho, "tol": tol}
    nonnegative = {"D": D, "H": H, "H_soft": H_soft, "U": U,
                   "Cd_soft": Cd_soft, "Cd_cyl": Cd_cyl}
    for name, value in positive.items():
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须是有限正数")
    for name, value in nonnegative.items():
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} 必须是有限非负数")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 0:
        raise ValueError("max_iter 必须是非负整数，0 表示刚性参考模式")

    # 标量角度代表一列；其他输入必须是一维序列，不能静默展平二维安装配置。
    angles = np.asarray(theta_deg, dtype=float)
    if angles.ndim == 0:
        angles = angles.reshape(1)
    if angles.ndim != 1 or not angles.size or not np.all(np.isfinite(angles)):
        raise ValueError("theta_deg 必须是非空、有限的一维角度序列或单个角度")
    if not isinstance(options, NumericalOptions):
        raise TypeError("options 必须是 NumericalOptions")
    for name, value, minimum in (("nodes", options.nodes, cfg.MINIMUM_NODES),
                                 ("blades_per_column", options.blades_per_column, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} 必须是至少 {minimum} 的整数")
    if not np.isfinite(options.initial_angle_difference_deg) or options.initial_angle_difference_deg <= 0:
        raise ValueError("initial_angle_difference_deg 必须是有限正数")
    if not isinstance(options.enable_shielding, bool):
        raise ValueError("enable_shielding 必须是 bool")
    if not isinstance(shielding_func_type, str):
        raise ValueError("shielding_func_type 必须是字符串")
    # 在单列或刚性模式也检查遮蔽参数，保证接口不会接受不合法的设置。
    calculate_shielding_coefficient(0.0, options.initial_angle_difference_deg,
                                   shielding_func_type, min_shielding, max_shielding)
    return angles


def _initial_relaxation(E: float) -> float:
    """按杨氏模量 E（Pa）返回每列开始时的松弛系数。"""
    if E < cfg.VERY_SOFT_E:
        return cfg.RELAXATION_VERY_SOFT
    if E < cfg.SOFT_E:
        return cfg.RELAXATION_SOFT
    if E < cfg.STIFF_E:
        return cfg.RELAXATION_MEDIUM
    return cfg.RELAXATION_STIFF


def _loading_strategy(E: float, U: float) -> tuple[int, float]:
    """根据 E（Pa）及完整流速 U（m/s），返回加载步骤数和角度平滑系数。"""
    if E < cfg.VERY_SOFT_E:
        steps = max(cfg.VERY_SOFT_MIN_STEPS,
                    min(cfg.VERY_SOFT_MAX_STEPS, math.ceil(U / cfg.VERY_SOFT_VELOCITY_STEP)))
        return steps, cfg.ANGLE_SMOOTH_VERY_SOFT
    if E < cfg.STIFF_E:
        steps = max(cfg.MEDIUM_MIN_STEPS,
                    min(cfg.MEDIUM_MAX_STEPS, math.ceil(U / cfg.MEDIUM_VELOCITY_STEP)))
        return steps, cfg.ANGLE_SMOOTH_MEDIUM
    return 1, cfg.ANGLE_SMOOTH_STIFF


def _solve_drag(
    D: float, H: float, E: float, L: float, t: float, h: float,
    H_soft: float, b: float, U: float, rho: float, Cd_soft: float, Cd_cyl: float,
    theta_deg: ArrayLike, max_iter: int, tol: float,
    shielding_func_type: str, min_shielding: float, max_shielding: float,
    options: NumericalOptions,
) -> DragResult:
    """内部核心；参数顺序与 MATLAB 一致，返回带诊断信息的 DragResult。

    D/H: 圆柱直径/高度；E: 杨氏模量；L/t/h: 叶片长度/厚度/高度。
    H_soft/b: 覆盖高度/间距，原算法固定每列5片后不再使用这两个参数。
    U/rho: 流速/密度；Cd_soft/Cd_cyl: 叶片/圆柱阻力系数；theta_deg: 每列角度。
    max_iter/tol: 迭代次数/容差；其余参数指定遮蔽规则及数值网格设置。
    """
    angles_deg = _validate_inputs(D, H, E, L, t, h, H_soft, b, U, rho, Cd_soft,
                                  Cd_cyl, theta_deg, max_iter, tol, options,
                                  shielding_func_type, min_shielding, max_shielding)
    theta = np.deg2rad(angles_deg)
    count = options.blades_per_column
    n_total = theta.size * count
    cylinder_force = 0.5 * rho * Cd_cyl * D * H * U**2
    reference_area = D * H + n_total * h * L
    denominator = 0.5 * rho * U**2 * reference_area
    messages: list[str] = []

    # 刚性分支有独立的 |sin(theta)| 公式，按原程序跳过形状求解及遮蔽。
    if max_iter == 0:
        blade_force = 0.0
        for angle in theta:
            single = 0.5 * rho * Cd_soft * h * L * abs(np.sin(angle)) * U**2
            blade_force += count * single
        total = float(cylinder_force + blade_force)
        coefficient = total / denominator if denominator else math.nan
        return DragResult(total, float(blade_force), float(cylinder_force), float(coefficient),
                          0.0, (), None, None, ())

    # 建立网格及抗弯刚度，低刚度时保留原 MATLAB 告警。
    n = options.nodes
    dx = L / (n - 1)
    x = np.linspace(0.0, L, n)
    inertia = h * t**3 / cfg.SECTION_INERTIA_DIVISOR
    stiffness = E * inertia
    if not np.isfinite(stiffness) or stiffness <= 0:
        raise ValueError("E*h*t³/12 必须是有限正数；请检查单位和数值范围")
    if stiffness < cfg.LOW_STIFFNESS_WARNING:
        message = f"抗弯刚度 EI={stiffness:.3e} N·m² 过低，可能出现数值不稳定"
        messages.append(message)
        warnings.warn(message, NumericalStabilityWarning, stacklevel=3)

    # 矩阵不随迭代改变，因此只做一次 LU 分解；所有边界右端载荷仍保留。
    matrix = finite_difference_matrix(n, dx)
    factors = lu_factor(matrix, check_finite=False)
    columns: list[ColumnDeformation] = []
    steps, smoothing = _loading_strategy(E, U)
    base_alpha = _initial_relaxation(E)

    # 每列只求一片代表性叶片，最终乘以同列数量；不同列分别求解。
    for column_index, initial_angle in enumerate(theta):
        previous_w = np.zeros(n)
        if abs(initial_angle - np.pi) < cfg.ANGLE_MATCH_TOLERANCE_RAD:
            normalized_x = x / L
            previous_w = cfg.INITIAL_PERTURBATION * L * normalized_x**2 * (1 - normalized_x)
        previous_angle = np.full(n, initial_angle)
        alpha = base_alpha
        if (0 <= initial_angle <= np.pi / 2) or (3 * np.pi / 2 <= initial_angle <= 2 * np.pi):
            alpha = min(alpha, cfg.CONSERVATIVE_ANGLE_RELAXATION)
        # 与原函数一样：非刚性软材料会提高次数下限并收紧容差。
        if E < cfg.SOFT_LIMIT_E:
            alpha = min(alpha, cfg.SOFT_MAX_RELAXATION)
            max_iter = max(max_iter, cfg.SOFT_MIN_ITERATIONS)
            tol = min(tol, cfg.SOFT_MAX_TOLERANCE)
        per_step = max(cfg.MIN_ITERATIONS_PER_STEP, math.ceil(max_iter / steps))
        previous_residual = math.inf
        residual = math.inf
        iterations = 0
        converged = False

        # 载荷从较小流速逐级增至完整流速，状态和松弛系数跨步骤延续。
        for step in range(1, steps + 1):
            effective_velocity = U * (step / steps)
            for _ in range(per_step):
                iterations += 1
                if iterations > max_iter:
                    break
                gradient = np.gradient(previous_w, dx, edge_order=1)
                _, total_angle = compute_angle_with_large_deformation(
                    gradient, initial_angle, L, E, previous_angle, smoothing)
                previous_angle = total_angle
                sine = np.sin(total_angle)

                # 精确保留原程序的小角度数值保护及 sign(0)=0 行为。
                if (abs(initial_angle - np.pi) < cfg.ANGLE_MATCH_TOLERANCE_RAD
                        and np.all(np.abs(sine) < cfg.PI_SINE_THRESHOLD)):
                    sine = np.sign(sine) * np.maximum(np.abs(sine), cfg.PI_SINE_FLOOR)
                elif (abs(initial_angle) < np.deg2rad(cfg.NEAR_ZERO_ANGLE_DEG)
                      or abs(initial_angle - 2 * np.pi) < np.deg2rad(cfg.NEAR_ZERO_ANGLE_DEG)):
                    small = np.abs(sine) < cfg.NEAR_ZERO_SINE_THRESHOLD
                    sine[small] = np.sign(sine[small]) * cfg.NEAR_ZERO_SINE_FLOOR
                normal_velocity = effective_velocity * sine
                load = 0.5 * rho * Cd_soft * h * np.abs(normal_velocity)**2 * np.sign(normal_velocity)

                # 完整右端求解后才将首两个位移置零，不能提前清零载荷边界项。
                new_w = lu_solve(factors, load / stiffness, check_finite=False)
                if not np.all(np.isfinite(new_w)):
                    raise FloatingPointError("梁方程得到非有限位移，请检查输入参数和单位")
                new_w[0] = 0.0
                new_w[1] = new_w[0]
                maximum_trial = float(np.max(np.abs(new_w)))
                if maximum_trial > cfg.MAX_DEFORMATION_LENGTHS * L:
                    new_w *= cfg.MAX_DEFORMATION_LENGTHS * L / maximum_trial

                # 使用松弛更新前的残差调节 alpha，不能改用更新后的位移差。
                delta = new_w - previous_w
                residual = float(np.max(np.abs(delta)))
                if (residual > previous_residual * cfg.RESIDUAL_GROWTH_RATIO
                        and iterations > cfg.GROWTH_MIN_ITERATIONS):
                    alpha = max(alpha * cfg.RELAXATION_SHRINK, cfg.MIN_RELAXATION)
                elif residual < previous_residual * cfg.RESIDUAL_SHRINK_RATIO:
                    alpha = min(alpha * cfg.RELAXATION_GROW, cfg.MAX_RELAXATION)
                if E < cfg.SOFT_LIMIT_E and np.max(np.abs(new_w)) > cfg.SOFT_ENERGY_LENGTHS * L:
                    alpha = max(alpha * cfg.SOFT_ENERGY_RELAXATION, cfg.SOFT_ENERGY_MIN_RELAXATION)
                previous_residual = residual
                previous_w = previous_w + alpha * delta

                # 最后加载步骤要求严格 tol；中间步骤容许十倍 tol 后继续加载。
                if step == steps:
                    if residual < tol:
                        converged = True
                        break
                elif residual < tol * cfg.INTERMEDIATE_TOLERANCE_FACTOR:
                    break
            if converged:
                break

        # 最终角度再调用一次含平滑的转换，受力使用完整 U，不再使用小角保护。
        final_gradient = np.gradient(previous_w, dx, edge_order=1)
        _, final_angle = compute_angle_with_large_deformation(
            final_gradient, initial_angle, L, E, previous_angle, smoothing)
        normal_final = U * np.sin(final_angle)
        final_load = 0.5 * rho * Cd_soft * h * np.abs(normal_final)**2 * np.sign(normal_final)
        column_force = float(trapezoid(final_load, x)) * count

        # 由局部角度积分得到尖端/中点位置；MATLAB round(n/2) 采用向上取半。
        x_position = cumulative_trapezoid(np.cos(final_angle), x, initial=0.0)
        y_position = cumulative_trapezoid(np.sin(final_angle), x, initial=0.0)
        mid_index = math.floor(n / 2 + 0.5) - 1
        column = ColumnDeformation(
            x.copy(), previous_w.copy(), np.rad2deg(final_angle), converged, iterations,
            residual, column_force, column_force, float(x_position[-1]), float(y_position[-1]),
            float(np.rad2deg(np.arctan2(y_position[mid_index], x_position[mid_index]))),
            float(np.rad2deg(np.arctan2(y_position[-1], x_position[-1]))),
            float(np.rad2deg(final_angle[mid_index])))
        columns.append(column)
        if not converged:
            message = f"第 {column_index + 1} 列未收敛（残差={residual:.3e} m，计数={iterations}）"
            messages.append(message)
            warnings.warn(message, ConvergenceWarning, stacklevel=3)

    # 遮蔽只改变第二列的已求解力；角度差不采用最短圆周差。
    shielding = None
    if len(columns) >= 2 and options.enable_shielding:
        difference = abs(columns[1].total_angle_deg[-1] - columns[0].total_angle_deg[-1])
        coefficient = calculate_shielding_coefficient(
            difference, options.initial_angle_difference_deg,
            shielding_func_type, min_shielding, max_shielding)
        columns[1].force_after_shielding_n *= coefficient
        shielding = {"enabled": True, "shielding_coefficient": coefficient,
                     "current_angle_diff": float(difference),
                     "initial_angle_diff": options.initial_angle_difference_deg}

    # 先逐列积分再取绝对值，不能对载荷逐节点取绝对值后积分。
    blade_force = float(sum(abs(column.force_after_shielding_n) for column in columns))
    total = float(cylinder_force + blade_force)
    coefficient = total / denominator if denominator else math.nan
    if not np.isfinite(total):
        raise FloatingPointError("总力非有限，请检查输入参数和单位")
    return DragResult(total, blade_force, float(cylinder_force), float(coefficient),
                      max(float(np.max(np.abs(column.displacement_m))) for column in columns),
                      tuple(columns), all(column.converged for column in columns), shielding, tuple(messages))


def calculate_drag_coefficient_v2(
    D: float, H: float, E: float, L: float, t: float, h: float, H_soft: float,
    b: float, U: float, rho: float, Cd_soft: float, Cd_cyl: float,
    theta_deg: ArrayLike, max_iter: int, tol: float,
    shielding_func_type: str = cfg.DEFAULT_SHIELDING_TYPE,
    min_shielding: float = cfg.DEFAULT_MIN_SHIELDING,
    max_shielding: float = cfg.DEFAULT_MAX_SHIELDING,
    *, options: NumericalOptions | None = None,
) -> tuple[float, float, float, list[dict]]:
    """保留 MATLAB 参数顺序，返回 (C_total, F_total, w_max, deformation_data)。

    D/H: 圆柱直径/高度（m）；E: 杨氏模量（Pa）；L/t/h: 叶片长度/厚度/高度（m）。
    H_soft/b: 覆盖高度/间距（m），在默认固定5片模式下不影响结果。
    U: 流速（m/s）；rho: 密度（kg/m³）；Cd_soft/Cd_cyl: 无量纲阻力系数。
    theta_deg: 每列安装角度（度）；max_iter: 次数（0为刚性）；tol: 位移容差（m）。
    最后三个位置参数为遮蔽类型及系数范围；options 可调整默认网格/每列数量。
    """
    result = _solve_drag(D, H, E, L, t, h, H_soft, b, U, rho, Cd_soft, Cd_cyl,
                         theta_deg, max_iter, tol, shielding_func_type,
                         min_shielding, max_shielding, options if options is not None else NumericalOptions())
    return result.drag_coefficient, result.total_force_n, result.max_deformation_m, result.matlab_deformation_data()
