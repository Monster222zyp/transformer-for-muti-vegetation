"""数据整合、几何坐标与 HydroDataset 的回归测试。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch

from Experiment.generator import ROTATION_MAPPING
from model.hydro.data import HydroDataset, collate_hydro_samples
from model.hydro.geometry import build_hex_coordinates, build_layout_index
from model.prepare_dataset import EXPECTED_MISSING_CONDITIONS, OUTPUT_COLUMNS, prepare_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "summarized_data"
INPUT_CSV = PROJECT_ROOT / "Experiment" / "input.csv"


@pytest.fixture(scope="module")
def generated_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """在 pytest 临时目录生成数据，避免测试覆盖正式产物。"""
    temporary_directory = tmp_path_factory.mktemp("prepared_data")
    output_csv = temporary_directory / "all_models.csv"
    audit_json = temporary_directory / "audit.json"
    prepare_dataset(SOURCE_ROOT, INPUT_CSV, output_csv, audit_json, strict=True)
    return output_csv, audit_json


def test_prepare_dataset_has_expected_rows_and_known_missing_conditions(
    generated_dataset: tuple[Path, Path],
) -> None:
    """当前真实数据必须恰好为 332 行，并只缺失四个已知工况。"""
    output_csv, audit_json = generated_dataset
    with output_csv.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.reader(csv_file))
    assert rows[0] == list(OUTPUT_COLUMNS)
    assert len(rows) - 1 == 332
    assert all(len(row) == 8 for row in rows[1:])
    assert all(len(row[-1]) == 37 and set(row[-1]) <= {"0", "1"} for row in rows[1:])

    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    actual_missing = tuple(
        (item["model_id"], item["angle"], item["flow_speed"])
        for item in audit["missing_conditions"]
    )
    assert actual_missing == EXPECTED_MISSING_CONDITIONS
    assert audit["rows_by_flow_speed"] == {"0.1": 83, "0.2": 83, "0.3": 82, "0.4": 84}


def test_layout_index_uniquely_recovers_all_model_angle_pairs() -> None:
    """17 个基础构型的六个方向都必须能够唯一反查。"""
    layout_index = build_layout_index(INPUT_CSV)
    assert len(layout_index) == 17 * 6
    assert set(layout_index.values()) == {
        (model_id, angle)
        for model_id in range(1, 18)
        for angle in range(0, 360, 60)
    }


def test_hex_coordinates_follow_physical_axes_and_unit_spacing() -> None:
    """坐标必须满足向上 +x、向左 +y、中心为原点和相邻距离为 1。"""
    coordinates = build_hex_coordinates()
    assert len(coordinates) == 37
    assert coordinates[18] == pytest.approx((0.0, 0.0))
    assert coordinates[0][0] > coordinates[33][0]
    assert coordinates[0][1] > coordinates[3][1]

    distances = sorted(
        math.dist(coordinates[first], coordinates[second])
        for first in range(37)
        for second in range(first + 1, 37)
    )
    assert distances[0] == pytest.approx(1.0)
    assert all(distance >= 1.0 - 1e-12 for distance in distances)


def test_experiment_rotation_mapping_is_clockwise_in_physical_coordinates() -> None:
    """实验顺时针映射必须与 ``+x`` 向上、``+y`` 向左的坐标约定一致。"""
    coordinates = build_hex_coordinates()
    cosine = math.cos(math.radians(60.0))
    sine = math.sin(math.radians(60.0))

    for source_index, target_index in ROTATION_MAPPING.items():
        source_x, source_y = coordinates[source_index]
        # 在当前物理轴中，俯视顺时针 60° 对应 (x', y')=(x cos+y sin, -x sin+y cos)。
        expected_target = (
            source_x * cosine + source_y * sine,
            -source_x * sine + source_y * cosine,
        )
        assert coordinates[target_index] == pytest.approx(expected_target, abs=1e-12)


def test_dataset_clamps_negative_targets_and_preserves_raw_values(
    generated_dataset: tuple[Path, Path],
) -> None:
    """负 FX_0 仅在训练标签归零，原始值和流速特征必须保留。"""
    output_csv, _ = generated_dataset
    dataset = HydroDataset(
        output_csv,
        INPUT_CSV,
        negative_target_policy="clamp_to_zero",
    )
    assert len(dataset) == 332
    assert dataset.negative_target_policy == "clamp_to_zero"

    negative_samples = [sample for sample in dataset if sample["raw_target_drag"].item() < 0.0]
    assert len(negative_samples) == 4
    assert all(sample["target_drag"].item() == 0.0 for sample in negative_samples)
    assert all(sample["global_features"].item() == pytest.approx(sample["flow_speed"]) for sample in dataset)


def test_dataset_rejects_unknown_negative_target_policy_before_file_access() -> None:
    """未知策略必须立即报错，不能因路径缺失而掩盖配置问题。"""
    with pytest.raises(ValueError, match="negative_target_policy"):
        HydroDataset(
            "不存在的数据.csv",
            "不存在的构型.csv",
            negative_target_policy="keep_negative",
        )


def test_collate_dynamically_pads_with_false_mask(generated_dataset: tuple[Path, Path]) -> None:
    """不同植株数进入同一批次后，padding 力为零且 mask 为 False。"""
    output_csv, _ = generated_dataset
    dataset = HydroDataset(output_csv, INPUT_CSV)
    first = dataset[0]
    sample_with_more_plants = next(
        sample for sample in dataset if sample["positions"].shape[0] > first["positions"].shape[0]
    )
    batch = collate_hydro_samples([first, sample_with_more_plants])

    assert batch["positions"].shape[0] == 2
    assert batch["plant_mask"].dtype == torch.bool
    first_count = first["positions"].shape[0]
    assert not batch["plant_mask"][0, first_count:].any()
    assert torch.count_nonzero(batch["single_drag"][0, first_count:]) == 0
