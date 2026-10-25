"""面向外部程序的接口，参数名明确表示几何量及其单位。"""

from __future__ import annotations

from numpy.typing import ArrayLike

from . import defaults as cfg
from .model import _solve_drag
from .types import DragResult, NumericalOptions


def predict_drag(
    *, cylinder_diameter_m: float, cylinder_height_m: float, young_modulus_pa: float,
    blade_length_m: float, blade_thickness_m: float, blade_height_m: float,
    flow_velocity_m_s: float, initial_angles_deg: ArrayLike,
    water_density_kg_m3: float = cfg.DEFAULT_DENSITY,
    blade_drag_coefficient: float = cfg.DEFAULT_BLADE_CD,
    cylinder_drag_coefficient: float = cfg.DEFAULT_CYLINDER_CD,
    max_iterations: int = cfg.DEFAULT_MAX_ITERATIONS,
    tolerance_m: float = cfg.DEFAULT_TOLERANCE,
    shielding_type: str = cfg.DEFAULT_SHIELDING_TYPE,
    min_shielding: float = cfg.DEFAULT_MIN_SHIELDING,
    max_shielding: float = cfg.DEFAULT_MAX_SHIELDING,
    numerical_options: NumericalOptions | None = None,
) -> DragResult:
    """根据几何、材料和水流输入预测阻力，返回总力及变形/收敛诊断。

    cylinder_diameter_m/cylinder_height_m: 圆柱支撑的直径/高度（m），允许零尺寸。
    young_modulus_pa: 叶片杨氏模量（Pa）；blade_length_m/thickness_m/height_m:
    叶片长度/厚度/高度（m），其中 height 是原 MATLAB 的 h，而非覆盖总高度。
    flow_velocity_m_s: 非负水流速度（m/s）；initial_angles_deg: 每列安装角度（度）。
    water_density_kg_m3: 水密度；两个 drag_coefficient 参数分别是叶片/圆柱阻力系数。
    max_iterations: 每列原始迭代次数，0为刚性；软材料可能自动提高到3000。
    tolerance_m: 原算法位移残差阈值；shielding_type/min/max: 遮蔽曲线及保留力范围。
    numerical_options: 节点数、每列叶片数量、遮蔽参考角差及遮蔽开关。
    返回 DragResult；total_force_n 为所需的总力（N），包括叶片与圆柱。
    """
    # H_soft 与 b 在原函数中被固定每列数量覆盖，新接口不要求无效几何输入。
    return _solve_drag(
        cylinder_diameter_m, cylinder_height_m, young_modulus_pa, blade_length_m,
        blade_thickness_m, blade_height_m, cfg.DEFAULT_COVERED_HEIGHT, cfg.DEFAULT_SPACING,
        flow_velocity_m_s, water_density_kg_m3, blade_drag_coefficient,
        cylinder_drag_coefficient, initial_angles_deg, max_iterations, tolerance_m,
            shielding_type, min_shielding, max_shielding,
            numerical_options if numerical_options is not None else NumericalOptions())


def calculate_force(**parameters) -> float:
    """返回总阻力浮点数，单位 N；所有关键字参数与 predict_drag 完全相同。

    必填参数是圆柱直径/高度、材料模量、叶片长度/厚度/高度、水流速度和安装角度。
    其他参数及默认值见 predict_drag 的说明；未收敛时保留原行为并发出告警。
    若需要变形数组、叶片力和圆柱力分量、收敛状态，请直接调用 predict_drag。
    """
    # 保持唯一计算路径，避免简单接口与详细接口出现不同的受力公式。
    return predict_drag(**parameters).total_force_n
