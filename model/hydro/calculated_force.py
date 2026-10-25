"""把转盘角和物理参数适配到原 seagrass_drag 求解器，返回整株默认力。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping


# 固定列顺序来自实验定义；原脚本的遮蔽与顺序有关，因此不能重新排序。
FULL_ROTATION_DEGREES = 360.0
COLUMN_ANGLE_OFFSETS_DEG = (60.0, 180.0, 300.0)
DEFAULT_FORCE_CACHE_SIZE = 4096  # 同参数、角度、流速只求解一次；缓存单位 N 的标量。
CALCULATED_CONFIG_VERSION = 3
LEGACY_CALCULATED_CONFIG_VERSION = 2  # 旧 checkpoint 仍使用全局 E，保留原求解含义。
SUPPORTED_CALCULATED_CONFIG_VERSIONS = (LEGACY_CALCULATED_CONFIG_VERSION, CALCULATED_CONFIG_VERSION)
STIFFNESS_UNIT = "N*m^2"  # JSONL 的 stiffness 是实测弯曲刚度 EI，而不是杨氏模量 E。
RECTANGULAR_SECOND_MOMENT_DIVISOR = 12.0  # 矩形截面二次矩 I=h*t³/12。
CALCULATION_BACKEND = "seagrass_drag_v0.1.0"  # 写入 checkpoint 的求解算法标识。
MINIMUM_SOLVER_NODES = 5  # 原差分矩阵至少需要五个节点。
SUPPORTED_SHIELDING_TYPES = frozenset(
    {"linear", "linear_bounded", "quadratic", "cubic", "exponential", "sqrt"}
)

# 分组仅声明数据契约；所有物理和求解参数均从 physical.yaml 显式读取。
PARAMETER_GROUPS = {
    "geometry": (
        "cylinder_diameter_m", "cylinder_height_m", "blade_length_m",
        "blade_thickness_m", "blade_height_m", "blades_per_column",
    ),
    "material": ("young_modulus_pa",),
    "fluid": ("water_density_kg_m3",),
    "drag": ("blade_drag_coefficient", "cylinder_drag_coefficient"),
    "solver": (
        "nodes", "max_iterations", "tolerance_m", "shielding_type",
        "min_shielding", "max_shielding", "enable_shielding",
        "initial_angle_difference_deg",
    ),
}
NONNEGATIVE_PARAMETERS = frozenset({
    "cylinder_diameter_m", "cylinder_height_m",
    "blade_drag_coefficient", "cylinder_drag_coefficient",
})
INTEGER_PARAMETER_MINIMUMS = {"nodes": MINIMUM_SOLVER_NODES, "blades_per_column": 1,
                            "max_iterations": 0}


def finite_number(value: Any, name: str) -> float:
    """读取有限实数，拒绝布尔值、字符串及非有限值；name 用于中文报错。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须为有限数值。")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} 必须为有限数值。")
    return parsed


def column_angles_for_turntable(angle: float) -> tuple[float, float, float]:
    """把转盘角 angle（度）转换为三列相对来流角，均归一化到 [0, 360)。

    支持任意有限整数或小数，返回顺序固定的三个角度；不使用两状态分组。
    """

    normalized = finite_number(angle, "转盘角度") % FULL_ROTATION_DEGREES
    return tuple((normalized + offset) % FULL_ROTATION_DEGREES
                 for offset in COLUMN_ANGLE_OFFSETS_DEG)


@lru_cache(maxsize=DEFAULT_FORCE_CACHE_SIZE)
def _cached_single_drag(
    parameters: tuple[tuple[str, Any], ...], angle: float, flow_speed: float,
    stiffness: float | None = None,
) -> float:
    """以完整求解参数（含 E）、角度、流速和原始 EI 为缓存键计算整株力。

    参数：parameters 是实际传给求解器的不可变键值对；angle 是归一化转盘角；
    flow_speed 是 m/s 流速；stiffness 仅标识原始 EI，None 表示历史全局 E 模式。
    返回值：整株阻力（N）。EI 不直接传入原求解器，避免改变其原始接口。
    """

    # 延迟导入：历史查表 checkpoint 仍可独立读取；新版求解器位于项目内部。
    from .seagrass_drag import NumericalOptions, predict_drag

    # 四项数值设置交给 NumericalOptions，其余参数直接传原公共接口。
    values = dict(parameters)
    options = NumericalOptions(
        nodes=values.pop("nodes"),
        blades_per_column=values.pop("blades_per_column"),
        initial_angle_difference_deg=values.pop("initial_angle_difference_deg"),
        enable_shielding=values.pop("enable_shielding"),
    )
    result = predict_drag(
        **values, flow_velocity_m_s=flow_speed,
        initial_angles_deg=column_angles_for_turntable(angle), numerical_options=options,
    )
    # 原程序已汇总三列全部叶片和一次圆柱力；不能额外乘三或乘每列片数。
    # 未收敛时保留原脚本的告警与返回行为，不把它静默改成刚性或查表计算。
    force = float(result.total_force_n)
    if not math.isfinite(force) or force < 0.0:
        raise FloatingPointError("物理求解器返回的整株默认力必须为有限非负数。")
    return force


@dataclass(frozen=True)
class CalculatedForceParameters:
    """可哈希、不可变的计算参数；没有流速或角度，它们由每条样本提供。

    属性 values：经校验、排序的参数键值对，既用于求解也用于 checkpoint。
    属性 version：3 从逐株 EI 恢复 E；2 保留历史全局 E。
    方法 single_drag_for_angle 返回 N；to_dict 返回可序列化的完整物理配置。
    """

    values: tuple[tuple[str, Any], ...]
    version: int = LEGACY_CALCULATED_CONFIG_VERSION

    def single_drag_for_angle(
        self, angle: float, flow_speed: float, stiffness: float | None = None,
    ) -> float:
        """按角度、流速和逐株 EI 计算力，相同实际求解参数复用缓存。

        参数：angle 为转盘角（度）；flow_speed 为非负流速（m/s）；
        stiffness 为有限正数 EI（N·m²）。version3 必须提供 EI；version2
        未提供时保留配置中的 E，提供时按同一截面公式覆盖 E。
        返回值：原求解器已汇总三列叶片与圆柱的整株阻力，单位 N。
        """

        normalized = finite_number(angle, "转盘角度") % FULL_ROTATION_DEGREES
        speed = finite_number(flow_speed, "flow_speed")
        if speed < 0.0:
            raise ValueError("flow_speed 必须为非负数，单位 m/s。")
        # 原求解器接收杨氏模量 E。适配层利用固定几何的矩形截面二次矩恢复 E，
        # 不修改原求解器内部任何公式，也不再次乘叶片数量或列数。
        values = dict(self.values)
        bending_stiffness = None
        if stiffness is None:
            if self.version == CALCULATED_CONFIG_VERSION:
                raise ValueError("物理配置 version3 必须提供逐株 stiffness（EI，N·m²）。")
        else:
            bending_stiffness = finite_number(stiffness, "stiffness（EI，N·m²）")
            if bending_stiffness <= 0.0:
                raise ValueError("stiffness（EI，N·m²）必须为有限正数。")
            # 极端但仍有限的几何输入也可能在三次方时溢出，转成明确的参数错误。
            try:
                second_moment = (
                    values["blade_height_m"] * values["blade_thickness_m"] ** 3
                    / RECTANGULAR_SECOND_MOMENT_DIVISOR
                )
            except OverflowError as error:
                raise ValueError("叶片几何产生的截面二次矩 I 必须为有限正数。") from error
            if not math.isfinite(second_moment) or second_moment <= 0.0:
                raise ValueError("叶片几何产生的截面二次矩 I 必须为有限正数。")
            young_modulus = bending_stiffness / second_moment
            if not math.isfinite(young_modulus) or young_modulus <= 0.0:
                raise ValueError("由 stiffness/I 换算的 young_modulus_pa 必须为有限正数。")
            values["young_modulus_pa"] = young_modulus

        # 同时保存原始 EI 和实际 E，避免极接近的 EI 在浮点换算后碰到同一缓存键。
        return _cached_single_drag(tuple(sorted(values.items())), normalized, speed, bending_stiffness)

    def to_dict(self) -> dict[str, Any]:
        """保存参数与算法标识；不保存缓存或由参数计算出的确切力。"""

        values = dict(self.values)
        grouped = {
            group: {key: values[key] for key in keys}
            for group, keys in PARAMETER_GROUPS.items() if group != "material"
        }
        # v3 只保存 EI 的单位。实际 EI 来自逐株数据，不能被全局 E 覆盖。
        grouped["material"] = (
            {"stiffness_unit": STIFFNESS_UNIT}
            if self.version == CALCULATED_CONFIG_VERSION
            else {"young_modulus_pa": values["young_modulus_pa"]}
        )
        return {
            "version": self.version, "drag_unit": "N",
            "backend": CALCULATION_BACKEND,
            **grouped,
        }


def calculated_parameters_from_mapping(payload: Mapping[str, Any]) -> CalculatedForceParameters:
    """校验新版 YAML/checkpoint；拒绝漏填、拼错、无效几何和无效求解参数。"""

    version = payload.get("version")
    if isinstance(version, bool) or version not in SUPPORTED_CALCULATED_CONFIG_VERSIONS:
        raise ValueError("计算配置 version 必须为 2 或 3。")
    expected = {"version", "drag_unit", "backend", *PARAMETER_GROUPS}
    if set(payload) != expected:
        raise ValueError(
            f"计算配置顶层字段不正确：缺少={sorted(expected - set(payload))}，"
            f"额外={sorted(set(payload) - expected)}。"
        )
    if payload["backend"] != CALCULATION_BACKEND:
        raise ValueError(f"backend 必须为 {CALCULATION_BACKEND!r}。")

    # 字段名标明单位；严格分组让 YAML 的拼写错误在数据加载前暴露。
    values: dict[str, Any] = {}
    for group, keys in PARAMETER_GROUPS.items():
        if group == "material" and version == CALCULATED_CONFIG_VERSION:
            keys = ("stiffness_unit",)
        section = payload[group]
        if not isinstance(section, Mapping) or set(section) != set(keys):
            raise ValueError(f"{group} 必须且只能包含参数 {list(keys)}。")
        if group == "material" and version == CALCULATED_CONFIG_VERSION:
            if section["stiffness_unit"] != STIFFNESS_UNIT:
                raise ValueError(f"material.stiffness_unit 必须为 {STIFFNESS_UNIT!r}。")
        else:
            values.update(section)

    # 计数和开关不能被浮点或字符串替代，也不能把 bool 当作整数。
    for key, minimum in INTEGER_PARAMETER_MINIMUMS.items():
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} 必须为至少 {minimum} 的整数。")
    if not isinstance(values["enable_shielding"], bool):
        raise ValueError("enable_shielding 必须为布尔值。")
    if (not isinstance(values["shielding_type"], str)
            or values["shielding_type"] not in SUPPORTED_SHIELDING_TYPES):
        raise ValueError(f"shielding_type 必须为 {sorted(SUPPORTED_SHIELDING_TYPES)} 之一。")

    # 几何和材料为正数；圆柱尺寸、阻力系数允许零；遮蔽比例单独检查上下界。
    special = {*INTEGER_PARAMETER_MINIMUMS, "enable_shielding", "shielding_type"}
    for key in set(values) - special:
        values[key] = finite_number(values[key], key)
        if key in {"min_shielding", "max_shielding"}:
            continue
        if key in NONNEGATIVE_PARAMETERS:
            if values[key] < 0.0:
                raise ValueError(f"{key} 必须为非负数。")
        elif values[key] <= 0.0:
            raise ValueError(f"{key} 必须为正数。")
    if not 0.0 <= values["min_shielding"] <= values["max_shielding"] <= 1.0:
        raise ValueError("遮蔽比例必须满足 0 <= min_shielding <= max_shielding <= 1。")
    return CalculatedForceParameters(tuple(sorted(values.items())), int(version))
