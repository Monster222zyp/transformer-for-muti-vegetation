"""HydroTransformer 使用的数据集和动态 padding 批处理函数。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .geometry import build_layout_index, layout_to_positions, parse_layout


DATASET_COLUMNS = (
    "TX",
    "TY",
    "TZ",
    "FX_0",
    "FY_0",
    "FZ",
    "flow_speed",
    "vegetation_layout",
)

# 第一版训练协议只允许把微小负阻力截断到零。把允许值集中定义，后续若增加
# 其他经过审查的策略，可以同时扩展参数校验和目标变换逻辑，避免配置被静默忽略。
SUPPORTED_NEGATIVE_TARGET_POLICIES = ("clamp_to_zero",)


class HydroDataset(Dataset[dict[str, Any]]):
    """读取八列总 CSV，并把每行转换为一张可变长度的水草集合。

    参数：
        csv_path: ``prepare_dataset.py`` 生成的八列总 CSV。
        input_csv_path: Experiment/input.csv，用于从构型反查 ``model_id`` 和角度。
        dtype: 浮点张量类型，默认使用训练稳定且常见的 ``torch.float32``。
        negative_target_policy: 负标签处理策略；当前仅支持 ``clamp_to_zero``，即把
            负 ``FX_0`` 的训练标签设为零，同时完整保留原始标签用于审计。

    每个样本保留原始 ``FX_0``，同时提供归零负值后的训练标签 ``target_drag``。
    """

    def __init__(
        self,
        csv_path: str | Path,
        input_csv_path: str | Path,
        dtype: torch.dtype = torch.float32,
        negative_target_policy: str = "clamp_to_zero",
    ) -> None:
        # 策略属于训练语义，必须先于任何文件访问进行校验。这样配置拼写错误不会
        # 被后续的路径错误掩盖，更不会在用户不知情时退回某个默认处理方式。
        if negative_target_policy not in SUPPORTED_NEGATIVE_TARGET_POLICIES:
            supported = ", ".join(SUPPORTED_NEGATIVE_TARGET_POLICIES)
            raise ValueError(
                f"不支持 negative_target_policy={negative_target_policy!r}；"
                f"当前允许值：{supported}。"
            )

        self.csv_path = Path(csv_path).resolve()
        self.input_csv_path = Path(input_csv_path).resolve()
        self.dtype = dtype
        self.negative_target_policy = negative_target_policy
        self.layout_index = build_layout_index(self.input_csv_path)
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict[str, Any]]:
        """解析全部 CSV 行，失败时报告准确的数据行号。"""
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"找不到总数据集：{self.csv_path}")

        samples: list[dict[str, Any]] = []
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames != list(DATASET_COLUMNS):
                raise ValueError(
                    f"总 CSV 表头应为 {list(DATASET_COLUMNS)}，实际为 {reader.fieldnames}。"
                )
            for source_index, row in enumerate(reader):
                csv_line_number = source_index + 2
                layout = row["vegetation_layout"].strip()
                parse_layout(layout)
                if layout not in self.layout_index:
                    raise ValueError(f"总 CSV 第 {csv_line_number} 行的构型无法反查 model/angle。")
                model_id, angle = self.layout_index[layout]

                try:
                    numeric = {column: float(row[column]) for column in DATASET_COLUMNS[:-1]}
                except (TypeError, ValueError) as error:
                    raise ValueError(f"总 CSV 第 {csv_line_number} 行包含非数值字段。") from error
                if any(value != value or abs(value) == float("inf") for value in numeric.values()):
                    raise ValueError(f"总 CSV 第 {csv_line_number} 行包含 NaN 或无穷大。")

                positions = torch.tensor(layout_to_positions(layout), dtype=self.dtype)
                plant_count = positions.shape[0]
                if plant_count == 0:
                    raise ValueError(f"总 CSV 第 {csv_line_number} 行没有任何水草。")

                raw_target = numeric["FX_0"]
                # 构造函数已经拒绝全部未知策略，因此此处可以明确执行当前唯一协议。
                # raw_target_drag 始终使用变换前的值，保证异常值仍可在预测表中追踪。
                if self.negative_target_policy == "clamp_to_zero":
                    target = max(raw_target, 0.0)
                else:  # pragma: no cover - 防御未来修改绕过构造函数校验。
                    raise AssertionError("负标签策略未经实现。")
                samples.append(
                    {
                        "positions": positions,
                        "single_drag": torch.ones(plant_count, dtype=self.dtype),
                        "plant_mask": torch.ones(plant_count, dtype=torch.bool),
                        "global_features": torch.tensor([numeric["flow_speed"]], dtype=self.dtype),
                        "target_drag": torch.tensor(target, dtype=self.dtype),
                        "raw_target_drag": torch.tensor(raw_target, dtype=self.dtype),
                        "model_id": model_id,
                        "angle": angle,
                        "flow_speed": numeric["flow_speed"],
                        "source_index": source_index,
                    }
                )
        if not samples:
            raise ValueError(f"总数据集没有样本：{self.csv_path}")
        return samples

    def __len__(self) -> int:
        """返回样本行数。"""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回指定行转换后的水草集合样本。"""
        return self.samples[index]


def collate_hydro_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Tensor]:
    """把可变植株数样本动态 padding 为一个批次。

    参数：
        samples: ``HydroDataset`` 返回的一个或多个样本。

    返回：
        ``positions`` 为 ``[B,Nmax,2]``，``single_drag`` 和 ``plant_mask`` 为
        ``[B,Nmax]``；其他浮点特征和可追踪元数据均按批次堆叠。
    """
    if not samples:
        raise ValueError("不能对空样本列表执行 collate。")

    batch_size = len(samples)
    maximum_plant_count = max(int(sample["positions"].shape[0]) for sample in samples)
    dtype = samples[0]["positions"].dtype
    device = samples[0]["positions"].device

    positions = torch.zeros((batch_size, maximum_plant_count, 2), dtype=dtype, device=device)
    single_drag = torch.zeros((batch_size, maximum_plant_count), dtype=dtype, device=device)
    plant_mask = torch.zeros((batch_size, maximum_plant_count), dtype=torch.bool, device=device)

    for batch_index, sample in enumerate(samples):
        plant_count = int(sample["positions"].shape[0])
        positions[batch_index, :plant_count] = sample["positions"]
        single_drag[batch_index, :plant_count] = sample["single_drag"]
        plant_mask[batch_index, :plant_count] = sample["plant_mask"]

    return {
        "positions": positions,
        "single_drag": single_drag,
        "plant_mask": plant_mask,
        "global_features": torch.stack([sample["global_features"] for sample in samples]),
        "target_drag": torch.stack([sample["target_drag"] for sample in samples]),
        "raw_target_drag": torch.stack([sample["raw_target_drag"] for sample in samples]),
        "model_id": torch.tensor([sample["model_id"] for sample in samples], dtype=torch.long),
        "angle": torch.tensor([sample["angle"] for sample in samples], dtype=torch.long),
        "flow_speed": torch.tensor([sample["flow_speed"] for sample in samples], dtype=dtype),
        "source_index": torch.tensor([sample["source_index"] for sample in samples], dtype=torch.long),
    }


# 使用常见名称作为兼容别名，方便训练入口直接传给 PyTorch DataLoader。
hydro_collate_fn = collate_hydro_samples
