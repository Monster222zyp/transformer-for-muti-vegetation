"""HydroTransformer 使用的 JSONL Dataset 和动态 padding 批处理函数。"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .geometry import POINT_COUNT, ROW_LENGTHS, build_hex_coordinates, parse_layout
from .physics import PhysicalConfig, load_physical_config


# 所有可选参数的默认值集中硬编码在代码开头，命令行或调用方仍可显式覆盖。
DEFAULT_TENSOR_DTYPE = torch.float32
DEFAULT_NEGATIVE_TARGET_POLICY = "clamp_to_zero"
DATASET_SCHEMA_VERSION = 1
FULL_ROTATION_DEGREES = 360
MIRROR_SAMPLE_SUFFIX = "__mirror_y"

# JSONL 顶层、逐株对象和测量对象采用严格字段集合。拒绝拼写错误或静默遗漏，
# 可以让上游数据格式变化在训练开始前立即暴露。
DATASET_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "model_id",
        "angle",
        "flow_speed",
        "vegetation_layout",
        "plants",
        "measurements",
    }
)
PLANT_FIELDS = frozenset({"plant_index", "grid_index", "x", "y", "angle"})
MEASUREMENT_FIELDS = frozenset({"TX", "TY", "TZ", "FX_0", "FY_0", "FZ"})

# 当前训练协议只允许把微小负阻力截断到零。数据文件始终保留原始 FX_0，
# 负值处理属于训练语义，因此不能提前固化进 dataset.jsonl。
SUPPORTED_NEGATIVE_TARGET_POLICIES = (DEFAULT_NEGATIVE_TARGET_POLICY,)


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    """确认一个 JSON 值是键值对象，并返回便于后续读取的 mapping。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} 必须是 JSON object。")
    return value


def _require_exact_fields(
    payload: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    """校验 JSON object 的字段集合与数据契约完全一致。"""

    actual = set(payload)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(f"{context} 字段不符合契约：缺少={missing}，额外={extra}。")


def _parse_integer(value: object, context: str) -> int:
    """读取严格整数；明确拒绝在 Python 中也属于 int 子类的 bool。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} 必须是整数。")
    return value


def _parse_finite_float(value: object, context: str) -> float:
    """读取有限浮点数，并拒绝 bool、NaN 和正负无穷。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} 必须是数值。")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{context} 必须是有限数值。")
    return parsed


class HydroDataset(Dataset[dict[str, Any]]):
    """读取 ``dataset.jsonl``，把每行转换为一张可变长度水草集合。

    参数：
        dataset_path: Experiment 汇总脚本生成的 ``dataset.jsonl`` 路径。
        dtype: 模型浮点张量类型，默认值为代码开头的 ``torch.float32``。
        negative_target_policy: 负标签处理策略；当前只支持 ``clamp_to_zero``。
        physical_config: 物理配置路径、配置字典、已解析对象或 ``None``；``None``
            会读取 ``model/configs/physical.yaml``。

    注意：
        每根水草的原始 ``angle`` 只交给物理配置查找单株默认阻力。Dataset 会保留
        ``plant_angles`` 供审计，但训练 forward 不读取这个张量，因此角度不会进入
        神经网络。
    """

    def __init__(
        self,
        dataset_path: str | Path,
        dtype: torch.dtype = DEFAULT_TENSOR_DTYPE,
        negative_target_policy: str = DEFAULT_NEGATIVE_TARGET_POLICY,
        physical_config: str | Path | dict[str, Any] | PhysicalConfig | None = None,
    ) -> None:
        # 先校验训练策略，使配置拼写错误不会被后续路径错误掩盖。
        if negative_target_policy not in SUPPORTED_NEGATIVE_TARGET_POLICIES:
            supported = ", ".join(SUPPORTED_NEGATIVE_TARGET_POLICIES)
            raise ValueError(
                f"不支持 negative_target_policy={negative_target_policy!r}；"
                f"当前允许值：{supported}。"
            )

        self.dataset_path = Path(dataset_path).resolve()
        self.dtype = dtype
        self.negative_target_policy = negative_target_policy
        self.physical_config = load_physical_config(physical_config)
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict[str, Any]]:
        """逐行解析 JSONL，并在错误信息中报告准确的物理行号。"""

        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"找不到模型数据集：{self.dataset_path}")

        samples: list[dict[str, Any]] = []
        seen_sample_ids: set[str] = set()
        canonical_coordinates = build_hex_coordinates()

        with self.dataset_path.open("r", encoding="utf-8-sig") as jsonl_file:
            for source_index, raw_line in enumerate(jsonl_file):
                jsonl_line_number = source_index + 1
                if not raw_line.strip():
                    raise ValueError(f"JSONL 第 {jsonl_line_number} 行为空。")
                try:
                    decoded = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSONL 第 {jsonl_line_number} 行不是合法 JSON：{error.msg}。"
                    ) from error

                record = _require_mapping(decoded, f"JSONL 第 {jsonl_line_number} 行")
                _require_exact_fields(
                    record,
                    DATASET_RECORD_FIELDS,
                    f"JSONL 第 {jsonl_line_number} 行",
                )
                if _parse_integer(
                    record["schema_version"],
                    f"JSONL 第 {jsonl_line_number} 行 schema_version",
                ) != DATASET_SCHEMA_VERSION:
                    raise ValueError(
                        f"JSONL 第 {jsonl_line_number} 行 schema_version 必须为 "
                        f"{DATASET_SCHEMA_VERSION}。"
                    )

                sample_id = record["sample_id"]
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise ValueError(
                        f"JSONL 第 {jsonl_line_number} 行 sample_id 必须是非空字符串。"
                    )
                if sample_id in seen_sample_ids:
                    raise ValueError(f"JSONL 包含重复 sample_id：{sample_id}")
                seen_sample_ids.add(sample_id)

                model_id = _parse_integer(
                    record["model_id"], f"sample {sample_id} 的 model_id"
                )
                if model_id < 1:
                    raise ValueError(f"sample {sample_id} 的 model_id 必须为正整数。")
                sample_angle = _parse_integer(
                    record["angle"], f"sample {sample_id} 的 angle"
                )
                flow_speed = _parse_finite_float(
                    record["flow_speed"], f"sample {sample_id} 的 flow_speed"
                )

                layout_value = record["vegetation_layout"]
                if not isinstance(layout_value, str):
                    raise ValueError(
                        f"sample {sample_id} 的 vegetation_layout 必须是 37 位字符串。"
                    )
                parsed_layout = parse_layout(layout_value)

                measurements = _require_mapping(
                    record["measurements"], f"sample {sample_id} 的 measurements"
                )
                _require_exact_fields(
                    measurements,
                    MEASUREMENT_FIELDS,
                    f"sample {sample_id} 的 measurements",
                )
                numeric_measurements = {
                    field: _parse_finite_float(
                        measurements[field], f"sample {sample_id} 的 measurements.{field}"
                    )
                    for field in MEASUREMENT_FIELDS
                }

                raw_plants = record["plants"]
                if not isinstance(raw_plants, list) or not raw_plants:
                    raise ValueError(f"sample {sample_id} 的 plants 必须是非空数组。")

                positions: list[tuple[float, float]] = []
                plant_angles: list[int] = []
                grid_indices: list[int] = []
                single_drag_values: list[float] = []
                for expected_plant_index, raw_plant in enumerate(raw_plants):
                    plant = _require_mapping(
                        raw_plant,
                        f"sample {sample_id} 的 plants[{expected_plant_index}]",
                    )
                    _require_exact_fields(
                        plant,
                        PLANT_FIELDS,
                        f"sample {sample_id} 的 plants[{expected_plant_index}]",
                    )
                    plant_index = _parse_integer(
                        plant["plant_index"], f"sample {sample_id} 的 plant_index"
                    )
                    if plant_index != expected_plant_index:
                        raise ValueError(
                            f"sample {sample_id} 的 plant_index 必须从 0 连续递增。"
                        )
                    grid_index = _parse_integer(
                        plant["grid_index"], f"sample {sample_id} 的 grid_index"
                    )
                    if not 0 <= grid_index < POINT_COUNT:
                        raise ValueError(
                            f"sample {sample_id} 的 grid_index 必须位于 0 到 "
                            f"{POINT_COUNT - 1}。"
                        )
                    if grid_indices and grid_index <= grid_indices[-1]:
                        raise ValueError(
                            f"sample {sample_id} 的 grid_index 必须严格递增且不重复。"
                        )

                    x_coordinate = _parse_finite_float(
                        plant["x"], f"sample {sample_id} 的 plant[{plant_index}].x"
                    )
                    y_coordinate = _parse_finite_float(
                        plant["y"], f"sample {sample_id} 的 plant[{plant_index}].y"
                    )
                    expected_x, expected_y = canonical_coordinates[grid_index]
                    if not (
                        math.isclose(x_coordinate, expected_x, rel_tol=0.0, abs_tol=1.0e-12)
                        and math.isclose(
                            y_coordinate, expected_y, rel_tol=0.0, abs_tol=1.0e-12
                        )
                    ):
                        raise ValueError(
                            f"sample {sample_id} 的 grid_index={grid_index} 与 x/y 不一致。"
                        )

                    plant_angle = _parse_integer(
                        plant["angle"],
                        f"sample {sample_id} 的 plant[{plant_index}].angle",
                    )
                    if plant_angle != sample_angle:
                        raise ValueError(
                            f"sample {sample_id} 当前要求每根水草 angle 都等于 sample angle。"
                        )

                    grid_indices.append(grid_index)
                    positions.append((x_coordinate, y_coordinate))
                    plant_angles.append(plant_angle)
                    # 角度只在这里进入物理查表，结果是单株默认阻力；后续模型 forward
                    # 只接收 single_drag，不接收 sample_angle 或 plant_angles。
                    single_drag_values.append(
                        self.physical_config.single_drag_for_angle(
                            plant_angle, flow_speed
                        )
                    )

                occupied_grid_indices = [
                    index for index, occupied in enumerate(parsed_layout) if occupied
                ]
                if grid_indices != occupied_grid_indices:
                    raise ValueError(
                        f"sample {sample_id} 的 plants.grid_index 与 vegetation_layout 不一致。"
                    )

                raw_target = numeric_measurements["FX_0"]
                if self.negative_target_policy == DEFAULT_NEGATIVE_TARGET_POLICY:
                    target = max(raw_target, 0.0)
                else:  # pragma: no cover - 构造函数已经拒绝所有未知策略。
                    raise AssertionError("负标签策略未经实现。")

                samples.append(
                    {
                        "positions": torch.tensor(positions, dtype=self.dtype),
                        "plant_angles": torch.tensor(plant_angles, dtype=torch.long),
                        "single_drag": torch.tensor(single_drag_values, dtype=self.dtype),
                        "plant_mask": torch.ones(len(positions), dtype=torch.bool),
                        "global_features": torch.tensor([flow_speed], dtype=self.dtype),
                        "target_drag": torch.tensor(target, dtype=self.dtype),
                        "raw_target_drag": torch.tensor(raw_target, dtype=self.dtype),
                        "sample_id": sample_id,
                        "model_id": model_id,
                        "angle": sample_angle,
                        "flow_speed": flow_speed,
                        "vegetation_layout": layout_value,
                        # 行号在同一份稳定排序的 JSONL 内继续用作 checkpoint 索引。
                        "source_index": source_index,
                    }
                )

        if not samples:
            raise ValueError(f"模型数据集没有 sample：{self.dataset_path}")
        return samples

    def __len__(self) -> int:
        """返回 JSONL 中的 sample 数量。"""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回指定 JSONL 行转换后的模型 sample。"""

        return self.samples[index]


class MirrorTrainingDataset(Dataset[dict[str, Any]]):
    """为已划分的训练子集按需提供原样本和关于竖直 x 轴的镜像样本。

    参数：
        dataset: 只包含训练索引的子集，不能传入尚未划分的完整评估数据。

    镜像将 y 取负，角度变为 (-angle) % 360；当前角度配置中镜像前后
    单株初始力相同，因此保留 single_drag 和总阻力标签。原始 Dataset 不变。
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        """每条训练数据提供原样和镜像两条记录。"""
        return len(self.dataset) * 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        """前半部分返回原样本，后半部分生成独立镜像，避免修改共享张量。"""
        sample_count = len(self.dataset)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < sample_count:
            return self.dataset[index]

        mirrored = deepcopy(self.dataset[index - sample_count])
        mirrored["positions"][..., 1] *= -1
        mirrored["angle"] = (-int(mirrored["angle"])) % FULL_ROTATION_DEGREES
        mirrored["plant_angles"] = (-mirrored["plant_angles"]) % FULL_ROTATION_DEGREES
        mirrored["sample_id"] = str(mirrored["sample_id"]) + MIRROR_SAMPLE_SUFFIX

        # 镜像后重新按上到下、左到右排列植物；所有逐株字段使用同一顺序，
        # 确保角度、初始力与坐标仍属于同一株植物。
        order = sorted(
            range(len(mirrored["positions"])),
            key=lambda i: (-float(mirrored["positions"][i, 0]), -float(mirrored["positions"][i, 1])),
        )
        for field in ("positions", "plant_angles", "single_drag", "plant_mask"):
            mirrored[field] = mirrored[field][order]

        # 37 位构型按每行反转实现左右镜像，不改变上下游所在行。
        # 简化的训练测试 Dataset 可能没有构型元数据，因此仅在存在时同步更新。
        if "vegetation_layout" in mirrored:
            layout = parse_layout(mirrored["vegetation_layout"])
            rows = []
            offset = 0
            for row_length in ROW_LENGTHS:
                rows.extend(reversed(layout[offset:offset + row_length]))
                offset += row_length
            mirrored["vegetation_layout"] = "".join(str(value) for value in rows)
        return mirrored


def collate_hydro_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把不同水草数量的 sample 动态 padding 为一个 batch。

    参数：
        samples: ``HydroDataset`` 返回的一个或多个 sample。

    返回值：
        ``positions`` 为 ``[B,Nmax,2]``；``single_drag``、``plant_angles`` 和
        ``plant_mask`` 为 ``[B,Nmax]``。其中 ``plant_angles`` 只用于审计和输出，
        不会传给神经网络。
    """

    if not samples:
        raise ValueError("不能对空 sample 列表执行 collate。")

    batch_size = len(samples)
    maximum_plant_count = max(int(sample["positions"].shape[0]) for sample in samples)
    dtype = samples[0]["positions"].dtype
    device = samples[0]["positions"].device

    positions = torch.zeros((batch_size, maximum_plant_count, 2), dtype=dtype, device=device)
    single_drag = torch.zeros((batch_size, maximum_plant_count), dtype=dtype, device=device)
    plant_angles = torch.zeros(
        (batch_size, maximum_plant_count), dtype=torch.long, device=device
    )
    plant_mask = torch.zeros(
        (batch_size, maximum_plant_count), dtype=torch.bool, device=device
    )

    for batch_index, sample in enumerate(samples):
        plant_count = int(sample["positions"].shape[0])
        positions[batch_index, :plant_count] = sample["positions"]
        single_drag[batch_index, :plant_count] = sample["single_drag"]
        plant_angles[batch_index, :plant_count] = sample["plant_angles"]
        plant_mask[batch_index, :plant_count] = sample["plant_mask"]

    return {
        "positions": positions,
        "single_drag": single_drag,
        "plant_angles": plant_angles,
        "plant_mask": plant_mask,
        "global_features": torch.stack([sample["global_features"] for sample in samples]),
        "target_drag": torch.stack([sample["target_drag"] for sample in samples]),
        "raw_target_drag": torch.stack([sample["raw_target_drag"] for sample in samples]),
        "sample_id": [str(sample["sample_id"]) for sample in samples],
        "model_id": torch.tensor([sample["model_id"] for sample in samples], dtype=torch.long),
        "angle": torch.tensor([sample["angle"] for sample in samples], dtype=torch.long),
        "flow_speed": torch.tensor([sample["flow_speed"] for sample in samples], dtype=dtype),
        "source_index": torch.tensor(
            [sample["source_index"] for sample in samples], dtype=torch.long
        ),
    }


# 使用常见名称作为兼容别名，方便 PyTorch DataLoader 直接引用。
hydro_collate_fn = collate_hydro_samples
