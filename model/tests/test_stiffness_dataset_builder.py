"""分类建集与完整实验重建回归：所有输出只写入 pytest 临时目录。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from Experiment.generator import ROTATION_MAPPING
from Experiment.summarize_sensor_data import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_LAYOUT_PATH,
    DEFAULT_NON_SIX_ANGLE_PATH,
    DEFAULT_REPEATABILITY_PATH,
    DEFAULT_SIX_ANGLE_PATH,
    MEASUREMENT_COLUMNS,
    build_dataset_records,
    build_parser,
    calculate_repeatability,
    calculate_trimmed_column_means,
    load_rotated_classifications,
    load_rotated_layouts,
    load_stiffness_config,
    parse_sensor_filename,
    summarize_sensor_tree,
)
from model.hydro.geometry import build_hex_coordinates


# 测试文件定位和所有辅助函数的默认值集中在这里；真实产物永不作为输出目标。
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_MAPPING = {1: 0.5, 3: 2.0, 7: 4.2}
TEST_MODEL_COUNT = 1
TEST_SENSOR_ROW_COUNT = 13
EXPECTED_REJECTED_PAIRS = {(15, 0, 0.2), (15, 165, 0.2), (16, 0, 0.2), (16, 30, 0.2), (17, 0, 0.2)}


def _write_config(path: Path, mapping: dict[int, float]) -> Path:
    """向临时 path 写材料 mapping，返回该 YAML 路径。"""
    path.write_text(yaml.safe_dump({"stiffness_by_code": mapping, "excluded_flow_speeds": [0.1], "repeatability_threshold_percent": 15.0}), encoding="utf-8")
    return path


def _write_layout(path: Path, row: list[int], count: int = TEST_MODEL_COUNT) -> Path:
    """在临时 path 写 count 个构型；row 保存一行的 37 个类别。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([row] * count)
    return path


def _read_jsonl(path: Path) -> list[dict]:
    """读取 path 中的一行一个 JSON；返回记录列表供内容比较。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cli_defaults_are_safe_and_use_full17() -> None:
    """默认 CLI 应重建完整 17 组到新目录，且不自动覆盖既有结果。"""
    arguments = build_parser().parse_args([])
    assert arguments.source_mode == "full17"
    assert not arguments.overwrite
    assert arguments.config == DEFAULT_CONFIG_PATH
    assert arguments.input_csv == DEFAULT_LAYOUT_PATH
    assert arguments.dataset_path == DEFAULT_DATASET_PATH
    assert arguments.dataset_path.parent.name == "rebuilt_all"
    assert parse_sensor_filename(Path("sensor_2_round1_ang75_lowpass_filtered.csv")) == (2, 75.0)
    assert parse_sensor_filename(Path("sensor_4_ang300_lowpass_filtered.csv")) == (4, 300.0)
    assert parse_sensor_filename(Path("gather_4_round1_ang300.csv")) is None


def test_arbitrary_material_codes_rotate_with_each_plant(tmp_path: Path) -> None:
    """类别 7 与 3 的 EI 必须随网格位置转到 60° 锚点，再旋转余角 15°。"""
    # 使用项目外的 CSV，确保 generator 不再从 CSV 的父目录寻找。
    row = [0] * 37
    row[0], row[36] = 7, 3
    external_csv = _write_layout(tmp_path / "external_input.csv", row)
    mapping = load_stiffness_config(_write_config(tmp_path / "custom.yaml", TEST_MAPPING))["stiffness_by_code"]
    classifications = load_rotated_classifications(external_csv, mapping)
    binary_layouts = load_rotated_layouts(external_csv, mapping)
    summary = {"sample_id": "model_001_angle_075_flow_0.2", "model_id": 1, "angle": 75, "flow_speed": 0.2, "vegetation_layout": binary_layouts[(1, 60)], **{column: 1.0 for column in MEASUREMENT_COLUMNS}}
    record = build_dataset_records([summary], rotated_classifications=classifications, stiffness_by_code=mapping)[0]
    assert record["schema_version"] == 3
    by_index = {plant["grid_index"]: plant for plant in record["plants"]}
    cosine, sine = math.cos(math.radians(15)), math.sin(math.radians(15))
    coordinates = build_hex_coordinates()
    for source_index, code in ((0, 7), (36, 3)):
        # 首先用 generator 的离散映射找到网格锚点，EI 必须仍来自原植株的类别。
        anchor_index = ROTATION_MAPPING[source_index]
        plant = by_index[anchor_index]
        anchor_x, anchor_y = coordinates[anchor_index]
        assert plant["stiffness"] == TEST_MAPPING[code]
        assert (plant["x"], plant["y"]) == pytest.approx((anchor_x * cosine + anchor_y * sine, -anchor_x * sine + anchor_y * cosine), abs=1e-12)
        assert plant["angle"] == 75
    assert set(record["vegetation_layout"]) <= {"0", "1"}
    assert len(record["vegetation_layout"]) == 37


@pytest.mark.parametrize("mapping", [{1: 0.0}, {1: -0.5}, {1: math.inf}, {1: math.nan}, {1: True}, {-1: 2.0}, {0: 2.0}, {1.5: 2.0}, {True: 2.0}])
def test_invalid_ei_or_category_is_rejected_before_building(tmp_path: Path, mapping: dict) -> None:
    """非法 EI、负数或非整数类别都应在 YAML 加载阶段失败。"""
    with pytest.raises(ValueError):
        load_stiffness_config(_write_config(tmp_path / "invalid.yaml", mapping))


@pytest.mark.parametrize("code", [-1, 99])
def test_unknown_or_negative_input_category_is_rejected(tmp_path: Path, code: int) -> None:
    """input.csv 不允许负数或未映射类别，不能静默用默认 EI 填补。"""
    row = [0] * 37
    row[0] = code
    with pytest.raises(ValueError):
        load_rotated_classifications(_write_layout(tmp_path / "invalid.csv", row), TEST_MAPPING)


def test_unknown_category_fails_before_sensor_discovery_and_output(tmp_path: Path) -> None:
    """非法类别必须先报错；不能被不存在的 sensor 根目录掩盖，也不能写产物。"""
    row = [0] * 37
    row[0] = 99
    layout = _write_layout(tmp_path / "unknown.csv", row)
    with pytest.raises(ValueError, match="未配置"):
        summarize_sensor_tree(tmp_path / "absent_source", tmp_path / "summary.csv", tmp_path / "all.jsonl", layout, 0.1, False, True, source_mode="full17")
    assert not (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "all.jsonl").exists()


def test_binary_helper_explicitly_interprets_one_as_material_category() -> None:
    """旧 helper 的纯 01 输入使用类别 1 的映射值，不隐式读取其它分类表。"""
    summary = {"sample_id": "custom_binary", "model_id": 100, "angle": 0, "flow_speed": 0.2, "vegetation_layout": "1" + "0" * 36, **{column: 0.0 for column in MEASUREMENT_COLUMNS}}
    records = build_dataset_records([summary], stiffness_by_code={1: 3.5})
    assert records[0]["plants"][0]["stiffness"] == 3.5


def test_trim_floor_and_column_independence() -> None:
    """13 行每端应删除 floor(13×10%)=1 行，且每列独立排序。"""
    values = np.column_stack([np.arange(13, dtype=float), np.arange(13, dtype=float)[::-1]] * 3)
    # 将极端值放在不同原始行，只有逐列排序才会正确去除各列异常点。
    values[0, 0], values[-1, 0] = -1000.0, 1000.0
    assert calculate_trimmed_column_means(values, 0.10) == pytest.approx([6.0] * 6)


def test_symmetric_repeatability_zero_and_threshold() -> None:
    """两轮零力应为 0%；37 与 43 的对称差率恰为 15%，交换轮次相同。"""
    assert calculate_repeatability(0.0, 0.0) == (0.0, 0.0, 0.0)
    assert calculate_repeatability(37.0, 43.0) == (6.0, 40.0, 15.0)
    assert calculate_repeatability(43.0, 37.0) == calculate_repeatability(37.0, 43.0)
    assert calculate_repeatability(-37.0, -43.0) == calculate_repeatability(37.0, 43.0)


def test_pair_filtering_includes_exact_threshold_and_averages_six_fields(tmp_path: Path) -> None:
    """阈值等于 15% 保留、超过则整对删除，所有六个字段在逐轮处理后平均。"""
    row = [0] * 37
    row[0], row[36] = 7, 3
    layout = _write_layout(tmp_path / "materials.csv", row, count=15)
    config = _write_config(tmp_path / "strategy.yaml", TEST_MAPPING)
    model_dir = tmp_path / "source" / "model_15"
    model_dir.mkdir(parents=True)
    # 第一工况差率恰为 15%；第二工况比 15% 略高，用于核实整对丢弃。
    for angle, second_force in ((0, 43.0), (15, 43.1)):
        for round_number, force in ((1, 37.0), (2, second_force)):
            radians = math.radians(angle)
            measurement = [round_number, 2 * round_number, 3 * round_number, force * math.cos(radians), force * math.sin(radians), 6 * round_number]
            with (model_dir / f"sensor_2_round{round_number}_ang{angle}_lowpass_filtered.csv").open("w", newline="") as handle:
                csv.writer(handle).writerows([measurement] * TEST_SENSOR_ROW_COUNT)
    output_dir = tmp_path / "rebuilt"
    rows = summarize_sensor_tree(tmp_path / "source", output_dir / "summary.csv", output_dir / "all.jsonl", layout, 0.1, False, False, source_mode="full17", config_path=config)
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "model_015_angle_000_flow_0.2"
    assert [rows[0][column] for column in MEASUREMENT_COLUMNS] == pytest.approx([1.5, 3.0, 4.5, 40.0, 0.0, 9.0])
    with (output_dir / DEFAULT_REPEATABILITY_PATH.name).open(newline="") as handle:
        audit = list(csv.DictReader(handle))
    assert [entry["retained"] for entry in audit] == ["True", "False"]
    assert len(_read_jsonl(output_dir / DEFAULT_NON_SIX_ANGLE_PATH.name)) == 0


@pytest.fixture(scope="module")
def rebuilt_full17(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """真实 CSV 重建只能写入 pytest 临时目录，严格检查完整旧、新源工况。"""
    output_dir = tmp_path_factory.mktemp("full17_stiffness_rebuild")
    summarize_sensor_tree(REPOSITORY_ROOT / "filtered_data", output_dir / "summary.csv", output_dir / "all.jsonl", DEFAULT_LAYOUT_PATH, 0.1, False, True, source_mode="full17")
    return output_dir


def test_real_rebuild_counts_rejections_and_sort_order(rebuilt_full17: Path) -> None:
    """完整输出 460、六角 300、非六角 160；审计 216 对且只删除已知五对。"""
    records = _read_jsonl(rebuilt_full17 / "all.jsonl")
    assert len(records) == 460
    assert len(_read_jsonl(rebuilt_full17 / DEFAULT_SIX_ANGLE_PATH.name)) == 300
    assert len(_read_jsonl(rebuilt_full17 / DEFAULT_NON_SIX_ANGLE_PATH.name)) == 160
    actual = [(record["model_id"], record["angle"], record["flow_speed"]) for record in records]
    assert actual == sorted(actual)
    assert len({record["sample_id"] for record in records}) == 460
    assert all("round" not in record["sample_id"] for record in records)
    assert all(record["flow_speed"] != 0.1 and record["schema_version"] == 3 for record in records)
    with (rebuilt_full17 / DEFAULT_REPEATABILITY_PATH.name).open(newline="") as handle:
        audit = list(csv.DictReader(handle))
    assert len(audit) == 216
    rejected = {(int(entry["model_id"]), int(entry["angle_deg"]), float(entry["flow_speed_m_s"])) for entry in audit if entry["retained"] == "False"}
    assert rejected == EXPECTED_REJECTED_PAIRS


def test_real_rebuild_preserves_measurements_geometry_and_ids(rebuilt_full17: Path) -> None:
    """对照既有所有.jsonl，新增 schema/EI 以外的实验测量和几何保持一致。"""
    reference = {record["sample_id"]: record for record in _read_jsonl(REPOSITORY_ROOT / "datasets" / "所有.jsonl")}
    actual = {record["sample_id"]: record for record in _read_jsonl(rebuilt_full17 / "all.jsonl")}
    assert actual.keys() == reference.keys()
    mappings = load_stiffness_config()["stiffness_by_code"]
    classifications = load_rotated_classifications(DEFAULT_LAYOUT_PATH, mappings)
    for sample_id, record in actual.items():
        original = reference[sample_id]
        for key in ("sample_id", "model_id", "angle", "flow_speed", "vegetation_layout"):
            assert record[key] == original[key]
        assert record["measurements"] == pytest.approx(original["measurements"], rel=1e-12, abs=1e-12)
        assert len(record["plants"]) == len(original["plants"])
        codes = classifications[(record["model_id"], record["angle"] - record["angle"] % 60)]
        for plant, original_plant in zip(record["plants"], original["plants"]):
            for key in ("plant_index", "grid_index", "angle"):
                assert plant[key] == original_plant[key]
            assert (plant["x"], plant["y"]) == pytest.approx((original_plant["x"], original_plant["y"]), abs=1e-12)
            assert plant["stiffness"] == mappings[codes[plant["grid_index"]]]


def test_existing_output_is_protected_before_any_write(rebuilt_full17: Path) -> None:
    """没有 overwrite 授权时所有已有输出保持原始字节。"""
    output_paths = list(rebuilt_full17.iterdir())
    before = {path: path.read_bytes() for path in output_paths}
    with pytest.raises(FileExistsError):
        summarize_sensor_tree(REPOSITORY_ROOT / "filtered_data", rebuilt_full17 / "summary.csv", rebuilt_full17 / "all.jsonl", DEFAULT_LAYOUT_PATH, 0.1, False, True, source_mode="full17")
    assert {path: path.read_bytes() for path in output_paths} == before


def test_real_rebuilt_mixed_ei_and_angles_reach_trainable_model(rebuilt_full17: Path, tmp_path: Path) -> None:
    """真实全量新格式直接接入默认物理计算、三维标准化、模型训练及逐株评估。"""

    import numpy as np
    import torch
    from model.hydro.data import HydroDataset, collate_hydro_samples
    from model.models import HydroTransformer
    from model.training.checkpoint import CHECKPOINT_VERSION, load_checkpoint, resolved_model_config
    from model.training.trainer import GlobalFeatureScaler, fit_fixed_epochs, predict_dataset

    dataset = HydroDataset(rebuilt_full17 / "all.jsonl")
    assert len(dataset) == 460
    actual_ei = {float(value) for sample in dataset for value in sample["plant_stiffness"]}
    assert actual_ei == {0.5, 1.0, 2.0}
    # 用真实非六角和六角混合样本做短训练，产物仍全部放在 pytest 临时目录。
    indices = np.array([index for index, sample in enumerate(dataset)
                        if sample["flow_speed"] == 0.3 and (
                            (sample["model_id"] <= 2 and sample["angle"] == 60)
                            or (sample["model_id"] >= 15 and sample["angle"] in (15, 75)))])
    assert len(indices) == 8
    scaler = GlobalFeatureScaler.fit(dataset, indices)
    batch = collate_hydro_samples([dataset[int(index)] for index in indices])
    model = HydroTransformer(d_model=16, n_heads=2, n_layers=1, ffn_dim=32,
                             plant_encoder_hidden_dim=12, dropout=0.0,
                             relative_hidden_dim=8, coefficient_hidden_dim=8)
    with torch.no_grad():
        output = model(batch["positions"], batch["single_drag"],
                       scaler.transform(batch["global_features"]), batch["plant_mask"],
                       plant_angles=batch["plant_angles"], plant_stiffness=batch["plant_stiffness"])
    torch.testing.assert_close(output["total_drag"], batch["single_drag"].sum(dim=1))
    before = {name: value.detach().clone() for name, value in model.plant_encoder.named_parameters()}
    settings = {"batch_size": 4, "learning_rate": 3e-4, "weight_decay": 0.0,
                "gradient_clip": 1.0, "num_workers": 0, "device": "cpu"}
    checkpoint_path = fit_fixed_epochs(
        model, dataset, indices, collate_hydro_samples, settings, scaler,
        tmp_path, resolved_model_config(model), epochs=2, relative_floor=0.1, seed=7,
        checkpoint_metadata={"physical_config": dataset.physical_config.to_dict()},
    )
    assert any(not torch.equal(value.detach(), before[name])
               for name, value in model.plant_encoder.named_parameters())
    predictions = predict_dataset(model, dataset, indices, collate_hydro_samples,
                                  4, 0, scaler, torch.device("cpu"))
    assert len(predictions.predictions) == len(indices)
    assert predictions.plant_coefficients
    for row in predictions.plant_coefficients:
        assert row["plant_stiffness_nm2"] in actual_ei
        assert math.isfinite(row["latent_coefficient"])
    # 新权重、三维 scaler 和物理v3可一起恢复，并复现相同逐样本预测。
    restored_model = HydroTransformer(**resolved_model_config(model))
    checkpoint = load_checkpoint(checkpoint_path, restored_model)
    assert checkpoint["checkpoint_version"] == CHECKPOINT_VERSION
    assert checkpoint["checkpoint_metadata"]["physical_config"] == dataset.physical_config.to_dict()
    restored_predictions = predict_dataset(
        restored_model, dataset, indices, collate_hydro_samples, 4, 0,
        GlobalFeatureScaler.from_dict(checkpoint["scaler"]), torch.device("cpu"),
    )
    assert restored_predictions.predictions == predictions.predictions
