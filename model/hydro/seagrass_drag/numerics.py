"""逐项复刻三个 MATLAB 辅助函数，不重新解释或修正原物理模型。"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import defaults as cfg


def finite_difference_matrix(n: int, dx: float) -> NDArray[np.float64]:
    """构造 n×n 差分矩阵；n 是节点数，dx 是相邻节点距离（m）。

    返回 NumPy 二维浮点数组，逐行对应 finite_difference_matrix.m。
    特别保留原文件对全部边界行统一除 dx⁴、第二行另除 dx 的做法。
    """
    # 过少节点会使边界模板互相覆盖；有限正步长保证矩阵可以正常构造。
    if isinstance(n, bool) or not isinstance(n, Integral) or n < cfg.MINIMUM_NODES:
        raise ValueError(f"n 必须是至少 {cfg.MINIMUM_NODES} 的整数")
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx 必须是有限正数")
    matrix = np.zeros((n, n), dtype=float)

    # 根部位移约束和一阶斜率约束，Python 从索引 0 开始。
    matrix[0, 0] = 1.0
    matrix[1, :2] = np.asarray(cfg.ROOT_SLOPE_STENCIL) / dx

    # 内部节点使用五点模板近似四阶导数。
    for index in range(2, n - 2):
        matrix[index, index - 2:index + 3] = cfg.INTERIOR_STENCIL

    # 尖端两行对应原程序的弯矩和剪力条件，最后统一缩放。
    matrix[-2, -3:] = cfg.TIP_MOMENT_STENCIL
    matrix[-1, -4:] = cfg.TIP_SHEAR_STENCIL
    return matrix / dx**4


def compute_angle_with_large_deformation(
    dw_dx: ArrayLike, theta_initial: float, L: float, E: float,
    prev_total_angle_rad: ArrayLike | None = None,
    smoothing_factor: float = cfg.DEFAULT_HELPER_SMOOTHING,
    max_angle_change_per_iter: float | None = None,
    tanh_scale_factor: float | None = None,
    use_tanh_threshold: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """将变形斜率转成角度，返回（变形角度，总角度），单位弧度。

    dw_dx: 一维节点斜率数组；theta_initial: 原始安装角度（弧度）。
    E: 杨氏模量（Pa）；prev_total_angle_rad: 上一轮总角度，None/空数组表示无历史。
    smoothing_factor: 新角度权重；变化大时自动减小，保留更多上一轮结果。
    L 和最后三个参数只为兼容 MATLAB 接口保留，原函数没有使用它们。
    """
    # 不将输入安装角度预先归一化，避免改变 180°及360°附近的分支行为。
    gradient = np.asarray(dw_dx, dtype=float).reshape(-1)
    if gradient.size == 0 or not np.all(np.isfinite(gradient)):
        raise ValueError("dw_dx 必须包含有限斜率数值")
    if not np.isfinite(theta_initial) or not np.isfinite(E) or E <= 0:
        raise ValueError("初始角度必须有限，E 必须是有限正数")
    if not np.isfinite(smoothing_factor) or not 0 <= smoothing_factor <= 1:
        raise ValueError("smoothing_factor 必须在 0 到 1 之间")

    # 分材料缩放后使用反双曲正弦；保留乘法顺序以减少数值对照误差。
    if E < cfg.SOFT_E:
        equivalent = np.arcsinh(gradient * cfg.SLOPE_COMMON_SCALE * cfg.SLOPE_SOFT_SCALE)
        equivalent *= cfg.SLOPE_SOFT_OUTPUT_SCALE
    else:
        equivalent = np.arcsinh(gradient * cfg.SLOPE_COMMON_SCALE * cfg.SLOPE_HARD_SCALE)
        equivalent *= cfg.SLOPE_HARD_OUTPUT_SCALE

    # 原程序缩小整个数组，而不是逐节点裁剪极端值。
    maximum = float(np.max(np.abs(equivalent)))
    if maximum > cfg.MAX_STABLE_SLOPE:
        equivalent *= cfg.MAX_STABLE_SLOPE / maximum * cfg.SLOPE_REDUCTION_FACTOR
    total = theta_initial + equivalent

    # 有上一轮结果时，先限制异常跳变，再逐节点决定新旧角度的混合比例。
    if prev_total_angle_rad is not None and np.size(prev_total_angle_rad):
        previous = np.asarray(prev_total_angle_rad, dtype=float).reshape(-1)
        if previous.shape != gradient.shape or not np.all(np.isfinite(previous)):
            raise ValueError("上一轮角度必须与 dw_dx 同形状且全部有限")
        change = total - previous
        max_change = np.deg2rad(cfg.MAX_ANGLE_JUMP_DEG)
        unstable = np.abs(change) > max_change
        change[unstable] = np.sign(change[unstable]) * max_change * cfg.ANGLE_JUMP_REDUCTION
        total = previous + change
        local_rate = np.abs(total - previous)
        local_factor = smoothing_factor * (local_rate <= np.deg2rad(cfg.SMOOTH_CHANGE_DEG))
        adaptive = smoothing_factor * cfg.SMOOTH_BASE_WEIGHT + local_factor * cfg.SMOOTH_LOCAL_WEIGHT
        total = adaptive * total + (1 - adaptive) * previous

    # 与 MATLAB mod 一样，得到 [0, 2π) 内总角度，最后减原始安装角得到变形角。
    total = np.mod(total, 2 * np.pi)
    return total - theta_initial, total


def calculate_shielding_coefficient(
    current_angle_diff_deg: float, initial_angle_diff_deg: float,
    func_type: str = cfg.DEFAULT_SHIELDING_TYPE,
    min_shielding: float = cfg.DEFAULT_MIN_SHIELDING,
    max_shielding: float = cfg.DEFAULT_MAX_SHIELDING,
) -> float:
    """返回第二列所保留的受力比例，不表示被遮蔽掉的比例。

    两个角度差参数均使用度；func_type 选择曲线；min/max 指定系数范围。
    未知 func_type 按原程序使用线性曲线；指数分支在零角度差时略高于最小值。
    """
    # 验证输入后，将角度差比例限制到 [0, 1]，保持原分母保护值。
    values = (current_angle_diff_deg, initial_angle_diff_deg, min_shielding, max_shielding)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("遮蔽参数必须是有限数值")
    if not 0 <= min_shielding <= max_shielding <= 1:
        raise ValueError("遮蔽系数须满足 0 <= min_shielding <= max_shielding <= 1")
    ratio = current_angle_diff_deg / max(initial_angle_diff_deg, cfg.SHIELDING_DENOMINATOR_FLOOR)
    ratio = min(max(ratio, 0.0), 1.0)

    # 分支逐项对应 MATLAB switch；保留 linear 原有的减法运算顺序。
    kind = func_type.lower()
    if kind == "linear":
        base = 1 - (1 - ratio)
    elif kind == "quadratic":
        base = 1 - (1 - ratio)**2
    elif kind == "cubic":
        base = 1 - (1 - ratio)**3
    elif kind == "exponential":
        base = float(np.exp(-cfg.SHIELDING_EXPONENT * (1 - ratio)))
    elif kind == "sqrt":
        base = float(np.sqrt(ratio))
    else:
        base = ratio
    coefficient = min_shielding + base * (max_shielding - min_shielding)
    return float(max(min_shielding, min(max_shielding, coefficient)))
