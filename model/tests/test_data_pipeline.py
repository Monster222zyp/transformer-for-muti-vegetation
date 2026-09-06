"""数据整合、几何坐标与 HydroDataset 的回归测试。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch

from Experiment.generator import ROTATION_MAPPING
from Experiment.summarize_sensor_data import (
    EXPECTED_MISSING_CONDITIONS,
    OUTPUT_COLUMNS,
    calculate_trimmed_column_means,
    load_rotated_layouts,
    read_numeric_sensor_csv,
    rotate_mean_force,
    summarize_sensor_tree,
)
from model.hydro.data import HydroDataset, collate_hydro_samples
from model.hydro.geometry import build_hex_coordinates
from model.hydro.physics import load_physical_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "filtered_data"
INPUT_CSV = PROJECT_ROOT / "Experiment" / "input.csv"


@pytest.fixture(scope="module")
def generated_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """在 pytest 临时目录同时生成 CSV 和 JSONL，避免覆盖正式产物。"""
    temporary_directory = tmp_path_factory.mktemp("summarized_data")
    output_csv = temporary_directory / "summarized_data.csv"
    output_jsonl = temporary_directory / "dataset.jsonl"
    summarize_sensor_tree(
        SOURCE_ROOT,
        output_csv,
        output_jsonl,
        INPUT_CSV,
        0.10,
        True,
        True,
    )
    return output_csv, output_jsonl


def test_summarized_dataset_has_expected_rows_and_known_missing_conditions(
    generated_dataset: tuple[Path, Path],
) -> None:
    """当前汇总 CSV 必须恰好为 332 行，不再保存派生 state。"""
    output_csv, _ = generated_dataset
    with output_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    assert reader.fieldnames == list(OUTPUT_COLUMNS)
    assert "state" not in reader.fieldnames
    assert len(rows) == 332
    assert all(len(row["vegetation_layout"]) == 37 and set(row["vegetation_layout"]) <= {"0", "1"} for row in rows)
    assert len({row["sample_id"] for row in rows}) == len(rows)
    actual_conditions = {(int(row["model_id"]), int(row["angle"]), float(row["flow_speed"])) for row in rows}
    assert set(EXPECTED_MISSING_CONDITIONS).isdisjoint(actual_conditions)
    assert [(int(row["model_id"]), int(row["angle"]), float(row["flow_speed"])) for row in rows] == sorted(actual_conditions)


def test_summarized_row_preserves_rotated_layout_and_flow_order(
    generated_dataset: tuple[Path, Path],
) -> None:
    """一条 sample 必须同时携带旋转力、旋转排布和对应流速。"""

    output_csv, _ = generated_dataset
    with output_csv.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    row = next(
        item
        for item in rows
        if int(item["model_id"]) == 1
        and int(item["angle"]) == 60
        and float(item["flow_speed"]) == pytest.approx(0.1)
    )
    source_file = SOURCE_ROOT / "model_1" / "sensor_1_ang60_lowpass_filtered.csv"
    expected_means = rotate_mean_force(
        calculate_trimmed_column_means(read_numeric_sensor_csv(source_file), 0.10), 60
    )
    expected_layout = load_rotated_layouts(INPUT_CSV)[(1, 60)]

    assert row["vegetation_layout"] == expected_layout
    assert row["sample_id"] == "model_001_angle_060_flow_0.1"
    assert [float(row[column]) for column in ("TX", "TY", "TZ", "FX_0", "FY_0", "FZ")] == pytest.approx(expected_means)


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


def test_dataset_jsonl_contains_rotated_layout_and_each_plant_geometry(
    generated_dataset: tuple[Path, Path],
) -> None:
    """JSONL 必须一行一个 sample，并显式保存每根水草的坐标和原始角度。"""

    output_csv, output_jsonl = generated_dataset
    with output_csv.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]

    assert len(records) == len(csv_rows) == 332
    assert [record["sample_id"] for record in records] == [row["sample_id"] for row in csv_rows]
    record = next(
        item
        for item in records
        if item["model_id"] == 1 and item["angle"] == 60 and item["flow_speed"] == 0.1
    )
    assert record["vegetation_layout"] == "0001000000000100000000001000000001000"
    assert all(plant["angle"] == 60 for plant in record["plants"])
    expected_grid_indices = [
        index for index, value in enumerate(record["vegetation_layout"]) if value == "1"
    ]
    assert [plant["grid_index"] for plant in record["plants"]] == expected_grid_indices
    coordinates = build_hex_coordinates()
    assert [
        (plant["x"], plant["y"]) for plant in record["plants"]
    ] == pytest.approx([coordinates[index] for index in expected_grid_indices])


def test_same_model_uses_six_distinct_rotated_layouts(
    generated_dataset: tuple[Path, Path],
) -> None:
    """model_1 的六个角度必须写入六条明确不同的已旋转 01 序列。"""

    output_csv, _ = generated_dataset
    with output_csv.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    actual = {
        int(row["angle"]): row["vegetation_layout"]
        for row in rows
        if int(row["model_id"]) == 1 and float(row["flow_speed"]) == 0.1
    }
    assert actual == {
        0: "1000001000000000000000000100000000001",
        60: "0001000000000100000000001000000001000",
        120: "0000000000000001010001000010000000000",
        180: "1000000000010000000000000000001000001",
        240: "0001000000001000000000010000000001000",
        300: "0000000000100001000101000000000000000",
    }


def test_dataset_clamps_negative_targets_and_preserves_raw_values(
    generated_dataset: tuple[Path, Path],
) -> None:
    """负 FX_0 仅在训练标签归零，原始值和流速特征必须保留。"""
    _, output_jsonl = generated_dataset
    dataset = HydroDataset(output_jsonl, negative_target_policy="clamp_to_zero")
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
            negative_target_policy="keep_negative",
        )


def test_collate_dynamically_pads_with_false_mask(
    generated_dataset: tuple[Path, Path],
) -> None:
    """不同植株数进入同一批次后，padding 力为零且 mask 为 False。"""
    _, output_jsonl = generated_dataset
    dataset = HydroDataset(output_jsonl)
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
    assert batch["plant_angles"].dtype == torch.long
    assert torch.count_nonzero(batch["plant_angles"][0, first_count:]) == 0


def test_dataset_uses_raw_angle_and_speed_to_select_single_drag(
    generated_dataset: tuple[Path, Path],
) -> None:
    """Dataset 必须直接按逐株原始角度和流速选择物理单株阻力。"""

    physical_payload = {
        "version": 1,
        "drag_unit": "N",
        "states": {
            "1": {
                "angles_deg": [0, 120, 240],
                "single_drag_by_flow_speed": {
                    "0.1": 1.1,
                    "0.2": 1.2,
                    "0.3": 1.3,
                    "0.4": 1.4,
                },
            },
            "2": {
                "angles_deg": [60, 180, 300],
                "single_drag_by_flow_speed": {
                    "0.1": 2.1,
                    "0.2": 2.2,
                    "0.3": 2.3,
                    "0.4": 2.4,
                },
            },
        },
    }
    _, output_jsonl = generated_dataset
    dataset = HydroDataset(output_jsonl, physical_config=physical_payload)

    observed_angles: set[int] = set()
    for sample in dataset:
        angle = int(sample["angle"])
        observed_angles.add(angle)
        drag_group = 1 if angle in {0, 120, 240} else 2
        expected_drag = drag_group + float(sample["flow_speed"])
        assert torch.unique(sample["plant_angles"]).tolist() == [angle]
        torch.testing.assert_close(
            sample["single_drag"],
            torch.full_like(sample["single_drag"], expected_drag),
        )

    assert observed_angles == {0, 60, 120, 180, 240, 300}


@pytest.mark.parametrize(
    "mutation, expected_message",
    [
        (lambda payload: payload["states"]["1"]["single_drag_by_flow_speed"].pop("0.4"), "完整配置流速"),
        (lambda payload: payload["states"]["2"]["single_drag_by_flow_speed"].update({"0.3": 0.0}), "有限正数"),
        (lambda payload: payload["states"]["2"].update({"angles_deg": [0, 180, 300]}), "重复角度"),
    ],
)
def test_physical_config_rejects_incomplete_or_invalid_values(
    mutation, expected_message: str,
) -> None:
    """缺失流速、非正阻力和跨状态重复角度必须在训练前报错。"""

    payload = {
        "version": 1,
        "drag_unit": "N",
        "states": {
            "1": {
                "angles_deg": [0, 120, 240],
                "single_drag_by_flow_speed": {
                    "0.1": 1.0,
                    "0.2": 1.0,
                    "0.3": 1.0,
                    "0.4": 1.0,
                },
            },
            "2": {
                "angles_deg": [60, 180, 300],
                "single_drag_by_flow_speed": {
                    "0.1": 1.0,
                    "0.2": 1.0,
                    "0.3": 1.0,
                    "0.4": 1.0,
                },
            },
        },
    }
    mutation(payload)
    with pytest.raises(ValueError, match=expected_message):
        load_physical_config(payload)
