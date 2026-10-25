"""结果结构与可显式调整的数值设置；属性名称标明单位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import defaults as cfg


@dataclass(frozen=True)
class NumericalOptions:
    """默认忠实复刻原模型；修改任一属性都会改变基准模型。

    nodes: 每片叶片沿长度的离散节点数。
    blades_per_column: 每列相同叶片数量，原函数强制设为 5。
    initial_angle_difference_deg: 遮蔽计算的参考角度差，单位度。
    enable_shielding: 是否在变形求解后修正第二列的受力。
    """

    nodes: int = cfg.DEFAULT_NODES
    blades_per_column: int = cfg.DEFAULT_BLADES_PER_COLUMN
    initial_angle_difference_deg: float = cfg.DEFAULT_INITIAL_ANGLE_DIFF_DEG
    enable_shielding: bool = cfg.DEFAULT_ENABLE_SHIELDING


@dataclass
class ColumnDeformation:
    """一列代表性叶片的计算结果。

    position_m、displacement_m、total_angle_deg: 节点位置、变形与总角度数组。
    converged: 最后一个流速加载步骤是否达到原程序的残差要求。
    iterations: 原程序方式记录的迭代计数，可能包含触发上限后的一次检查。
    residual_m: 最后一轮松弛更新前的最大位移残差。
    force_before_shielding_n、force_after_shielding_n: 保留正负方向的整列积分力。
    tip_x_m、tip_y_m: 根据最终角度积分获得的尖端坐标。
    angle_mid_to_horizontal_deg、angle_tip_to_horizontal_deg: 根部到中点/尖端的连线角。
    mid_angle_deg: 叶片中间节点的局部角度。
    """

    position_m: NDArray[np.float64]
    displacement_m: NDArray[np.float64]
    total_angle_deg: NDArray[np.float64]
    converged: bool
    iterations: int
    residual_m: float
    force_before_shielding_n: float
    force_after_shielding_n: float
    tip_x_m: float
    tip_y_m: float
    angle_mid_to_horizontal_deg: float
    angle_tip_to_horizontal_deg: float
    mid_angle_deg: float


@dataclass
class DragResult:
    """一次完整预测，简单接口只返回其中的 total_force_n。

    总力为圆柱力加各列遮蔽后积分力绝对值之和，单位 N。
    drag_coefficient 在零流速时为 NaN，因为原公式的分母为零。
    columns 在刚性模式为空；converged 在刚性模式为 None，表示无需迭代。
    shielding 存储遮蔽开关、系数和角度差；messages 收集数值告警文本。
    """

    total_force_n: float
    blade_force_n: float
    cylinder_force_n: float
    drag_coefficient: float
    max_deformation_m: float
    columns: tuple[ColumnDeformation, ...]
    converged: bool | None
    shielding: dict[str, Any] | None
    messages: tuple[str, ...]

    def matlab_deformation_data(self) -> list[dict[str, Any]]:
        """返回接近 MATLAB struct 数组的字典列表，用于原接口和数值对照。

        返回值保留原代码的空 slope_deg 字段及仅第一列拥有 x 的实际行为。
        建议新调用者直接使用 columns，其中每列都有节点位置和实际积分力。
        """
        # 刚性分支在 MATLAB 中返回一条空结构，而不是计算叶片形状。
        empty = np.empty(0, dtype=float)
        if not self.columns:
            return [{"x": empty.copy(), "w": empty.copy(), "slope_deg": empty.copy(),
                     "total_angle_deg": empty.copy()}]

        # 把 Python 属性映射为原字段名，便于逐节点对照原 MATLAB 输出。
        records: list[dict[str, Any]] = []
        for index, column in enumerate(self.columns):
            records.append({
                "x": column.position_m.copy() if index == 0 else empty.copy(),
                "w": column.displacement_m.copy(), "slope_deg": empty.copy(),
                "total_angle_deg": column.total_angle_deg.copy(),
                "converged": column.converged, "iterations": column.iterations,
                "x_tip_final": column.tip_x_m, "y_tip_final": column.tip_y_m,
                "angle_mid_to_horizontal": column.angle_mid_to_horizontal_deg,
                "angle_tip_to_horizontal": column.angle_tip_to_horizontal_deg,
                "mid_angle_deg": column.mid_angle_deg,
            })
        # 原程序把第二列所用的遮蔽信息放在第一列的结构中。
        if self.shielding is not None:
            # MATLAB struct 数组字段统一，其他列也会出现一个空 shielding 字段。
            for record in records:
                record["shielding"] = []
            records[0]["shielding"] = dict(self.shielding)
        return records


class ConvergenceWarning(RuntimeWarning):
    """没有达到原程序收敛条件；返回值仍是当前迭代状态计算得到的力。"""


class NumericalStabilityWarning(RuntimeWarning):
    """材料抗弯刚度低于原程序的告警阈值。"""
