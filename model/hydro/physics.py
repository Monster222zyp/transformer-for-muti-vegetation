"""水草状态与物理单株基准阻力配置。

该模块把实验测得的单株阻力与神经网络超参数分离。训练和评估都先把 YAML
解析为不可变的 :class:`PhysicalConfig`，随后 Dataset 只通过该对象查询状态和阻力。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


# 当前实验固定包含六个角度和四档相对流速。把协议写成常量可以在训练开始前
# 发现漏填、重复或多填的物理测量，而不是等到某一条样本被读取时才失败。
EXPECTED_ANGLES_DEG = frozenset({0, 60, 120, 180, 240, 300})
EXPECTED_FLOW_SPEEDS = (0.1, 0.2, 0.3, 0.4)
EXPECTED_STATE_IDS = frozenset({1, 2})
DEFAULT_PHYSICAL_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "physical.yaml"
)


@dataclass(frozen=True)
class PhysicalConfig:
    """经过完整校验的双状态物理参数。

    属性：
        version: 物理配置格式版本，当前必须为 ``1``。
        drag_unit: 单株阻力单位，当前必须为 ``N``，可直接与 ``FX_0`` 相乘比较。
        angles_by_state: 状态编号到角度集合的映射。
        single_drag_by_state_and_speed: 状态和流速到单株基准阻力的二维查找表。
    """

    version: int
    drag_unit: str
    angles_by_state: dict[int, tuple[int, ...]]
    single_drag_by_state_and_speed: dict[int, dict[float, float]]

    def state_for_angle(self, angle: int) -> int:
        """返回角度对应的状态编号。

        参数：
            angle: 从构型反查得到的旋转角度，单位为 degree。

        返回值：
            ``1`` 或 ``2``。
        """

        normalized_angle = int(angle) % 360
        for state_id, angles in self.angles_by_state.items():
            if normalized_angle in angles:
                return state_id
        raise ValueError(f"角度 {angle}° 没有配置水草状态。")

    def single_drag_for(self, state_id: int, flow_speed: float) -> float:
        """查询指定状态和流速的单株基准阻力。

        参数：
            state_id: 水草状态编号，只能为 ``1`` 或 ``2``。
            flow_speed: 当前工况流速，单位为 ``m/s``。

        返回值：
            单根水草的物理基准阻力，单位为 ``N``。
        """

        if state_id not in self.single_drag_by_state_and_speed:
            raise ValueError(f"未知水草状态：{state_id}")
        for configured_speed, drag in self.single_drag_by_state_and_speed[state_id].items():
            if math.isclose(float(flow_speed), configured_speed, rel_tol=0.0, abs_tol=1.0e-9):
                return drag
        raise ValueError(
            f"状态 {state_id} 没有配置流速 {flow_speed:g} m/s 的单株阻力。"
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可安全写入 JSON/checkpoint 的普通字典。"""

        return {
            "version": self.version,
            "drag_unit": self.drag_unit,
            "states": {
                str(state_id): {
                    "angles_deg": list(self.angles_by_state[state_id]),
                    "single_drag_by_flow_speed": {
                        f"{speed:.1f}": drag
                        for speed, drag in sorted(
                            self.single_drag_by_state_and_speed[state_id].items()
                        )
                    },
                }
                for state_id in sorted(self.angles_by_state)
            },
        }


def _parse_state_id(raw_state_id: Any) -> int:
    """把 YAML/JSON 中的字符串或整数状态键统一转换为整数。"""

    try:
        state_id = int(raw_state_id)
    except (TypeError, ValueError) as error:
        raise ValueError(f"非法水草状态编号：{raw_state_id!r}") from error
    return state_id


def physical_config_from_mapping(payload: Mapping[str, Any]) -> PhysicalConfig:
    """解析并严格校验一个物理配置字典。

    参数：
        payload: 从 YAML、JSON 或 checkpoint 读取的顶层映射。

    返回值：
        完整校验后的 :class:`PhysicalConfig`。
    """

    version = int(payload.get("version", -1))
    if version != 1:
        raise ValueError(f"physical config version 必须为 1，实际为 {version}。")
    drag_unit = str(payload.get("drag_unit", "")).strip()
    if drag_unit != "N":
        raise ValueError(f"drag_unit 当前必须为 'N'，实际为 {drag_unit!r}。")

    raw_states = payload.get("states")
    if not isinstance(raw_states, Mapping):
        raise ValueError("physical config 的 states 必须是映射。")
    states = {_parse_state_id(key): value for key, value in raw_states.items()}
    if set(states) != EXPECTED_STATE_IDS:
        raise ValueError("physical config 必须且只能包含状态 1 和状态 2。")

    angles_by_state: dict[int, tuple[int, ...]] = {}
    drag_table: dict[int, dict[float, float]] = {}
    collected_angles: list[int] = []
    for state_id in sorted(states):
        state_payload = states[state_id]
        if not isinstance(state_payload, Mapping):
            raise ValueError(f"状态 {state_id} 的配置必须是映射。")

        raw_angles = state_payload.get("angles_deg")
        if not isinstance(raw_angles, list) or not raw_angles:
            raise ValueError(f"状态 {state_id} 的 angles_deg 必须是非空列表。")
        angles = tuple(int(angle) for angle in raw_angles)
        if len(set(angles)) != len(angles):
            raise ValueError(f"状态 {state_id} 的 angles_deg 包含重复角度。")
        angles_by_state[state_id] = angles
        collected_angles.extend(angles)

        raw_drag_table = state_payload.get("single_drag_by_flow_speed")
        if not isinstance(raw_drag_table, Mapping):
            raise ValueError(
                f"状态 {state_id} 的 single_drag_by_flow_speed 必须是映射。"
            )
        normalized_table: dict[float, float] = {}
        for raw_speed, raw_drag in raw_drag_table.items():
            try:
                speed = float(raw_speed)
                drag = float(raw_drag)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"状态 {state_id} 包含非数值流速或阻力：{raw_speed!r}: {raw_drag!r}"
                ) from error
            if not math.isfinite(drag) or drag <= 0.0:
                raise ValueError(
                    f"状态 {state_id}、流速 {speed:g} 的单株阻力必须是有限正数。"
                )
            if speed in normalized_table:
                raise ValueError(f"状态 {state_id} 重复配置流速 {speed:g}。")
            normalized_table[speed] = drag
        if set(normalized_table) != set(EXPECTED_FLOW_SPEEDS):
            raise ValueError(
                f"状态 {state_id} 必须完整配置流速 {list(EXPECTED_FLOW_SPEEDS)}。"
            )
        drag_table[state_id] = normalized_table

    if len(set(collected_angles)) != len(collected_angles):
        raise ValueError("两个状态的 angles_deg 不能包含重复角度。")
    if set(collected_angles) != EXPECTED_ANGLES_DEG:
        raise ValueError(
            f"两个状态必须完整覆盖角度 {sorted(EXPECTED_ANGLES_DEG)}。"
        )

    return PhysicalConfig(version, drag_unit, angles_by_state, drag_table)


def load_physical_config(
    source: str | Path | Mapping[str, Any] | PhysicalConfig | None = None,
) -> PhysicalConfig:
    """从文件、字典或已解析对象加载物理配置。

    参数：
        source: YAML 路径、checkpoint 中的字典、已有 ``PhysicalConfig``，或 ``None``。
            ``None`` 会读取项目默认的 ``model/configs/physical.yaml``。

    返回值：
        完整校验后的 :class:`PhysicalConfig`。
    """

    if isinstance(source, PhysicalConfig):
        return source
    if isinstance(source, Mapping):
        return physical_config_from_mapping(source)

    config_path = DEFAULT_PHYSICAL_CONFIG_PATH if source is None else Path(source)
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到物理参数文件：{config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("物理参数文件顶层必须是键值映射。")
    return physical_config_from_mapping(payload)
