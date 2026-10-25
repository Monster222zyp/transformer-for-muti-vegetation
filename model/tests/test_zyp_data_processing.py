"""检查新批次的秒窗口、运动诊断和双轮筛选；所有产物只写入 pytest 临时目录。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from Experiment import filter_sensor_data as filtering
from Experiment import summarize_sensor_data as summary
from Experiment import process_zyp_data as processing
from model.hydro.data import HydroDataset


# 人造实验的默认值全部集中在这里，不引用或改动真实原始实验文件。
TEST_SENSOR_COUNT = 240
TEST_GATHER_COUNT = 24
TEST_START = 0.05
TEST_END = 0.20
TEST_FIRST_FORCE = 2.0
TEST_SECOND_FORCE = 2.1
TEST_REJECTED_FORCE = 4.0
TEST_STIFFNESS = {1: 8.0e-7, 2: 5.907e-6, 3: 2.08e-5}
TEST_REJECTED_KEY = (1, 0, 0.2)
TEST_MOTION_COUNT = 600
TEST_SLOPE = 0.3
TEST_ACCELERATION = 0.1
TEST_NUMERIC_TOLERANCE = 1.0e-10


def test_time_windows_use_independent_sample_rates() -> None:
    """同一秒窗口在两台设备上对应不同索引，截止时刻本身不能被收入。"""
    assert processing.time_window_indices(12200, 1000.0, 3.0, 4.99) == (3000, 4990)
    assert processing.time_window_indices(1207, 100.0, 3.0, 4.99) == (300, 499)
    assert processing.time_window_indices(12200, 1000.0, 3.0002, 4.9902) == (3001, 4991)
    with pytest.raises(ValueError, match="完整"):
        processing.time_window_indices(1207, 1000.0, 3.0, 4.99)


def test_gather_detects_acceleration_and_stationary_records(tmp_path: Path) -> None:
    """直线位移通过，持续加速和静止失败，避免把拟合直线本身当作匀速证据。"""
    times = np.arange(TEST_MOTION_COUNT) / processing.DEFAULT_GATHER_FS
    path = tmp_path / "gather.csv"
    cases = (
        (TEST_SLOPE * times, True),
        (TEST_SLOPE * times + TEST_ACCELERATION * times ** 2, False),
        (np.zeros_like(times), False),
    )
    for positions, expected in cases:
        # 只设置设备计数和 X 位移，其余 17 个通道不影响此项运动检查。
        values = np.zeros((len(times), processing.GATHER_COLUMN_COUNT))
        values[:, 0] = np.arange(len(times))
        values[:, processing.GATHER_X_COLUMN] = positions / processing.GATHER_DISPLACEMENT_TO_METERS
        np.savetxt(path, values, delimiter=" ")
        metrics = processing.inspect_gather(path, processing.DEFAULT_GATHER_FS, 3.0, 4.99,
                                            TEST_SLOPE, processing.DEFAULT_SPEED_CV_PERCENT,
                                            processing.DEFAULT_SPEED_MAX_DEVIATION_PERCENT)
        assert metrics["uniform_motion"] is expected
        if expected:
            assert metrics["fitted_velocity_m_s"] == pytest.approx(TEST_SLOPE)
            assert metrics["maximum_position_residual_m"] < TEST_NUMERIC_TOLERANCE


def _write_batch(tmp_path: Path) -> tuple[Path, Path, Path]:
    """在 tmp_path 创建一个完整 model 的人造实验，返回 raw、构型表和 YAML 路径。"""
    root = tmp_path / "raw"
    model_dir = root / "model1"
    model_dir.mkdir(parents=True)
    layout_csv = tmp_path / "input.csv"
    layout_csv.write_text(",".join(map(str, [1, 2, 3] + [0] * (summary.LAYOUT_POINT_COUNT - 3))) + "\n", encoding="utf-8")
    config = tmp_path / "stiffness.yaml"
    config.write_text(yaml.safe_dump({"stiffness_by_code": TEST_STIFFNESS,
                                     "repeatability_threshold_percent": summary.DEFAULT_REPEATABILITY_THRESHOLD_PERCENT}), encoding="utf-8")
    sensor_times = np.arange(TEST_SENSOR_COUNT) / processing.DEFAULT_SENSOR_FS
    gather_times = np.arange(TEST_GATHER_COUNT) / processing.DEFAULT_GATHER_FS
    for angle in range(0, processing.ANGLE_CYCLE_DEGREES, processing.ANGLE_STEP_DEGREES):
        for sensor in processing.SENSOR_IDS:
            for round_id in processing.ROUND_IDS:
                force = TEST_FIRST_FORCE if round_id == 1 else TEST_SECOND_FORCE
                if (1, angle, summary.FLOW_SPEED_BY_SENSOR[sensor]) == TEST_REJECTED_KEY and round_id == 2:
                    force = TEST_REJECTED_FORCE
                # 在参考方向添加小幅随时间变化，能检测错误的时间截取及处理顺序。
                reference_force = force + sensor_times ** 2
                radians = np.deg2rad(angle)
                values = np.zeros((TEST_SENSOR_COUNT, len(summary.SENSOR_COLUMNS)))
                values[:, 0] = sensor_times
                values[:, 3] = reference_force * np.cos(radians)
                values[:, 4] = reference_force * np.sin(radians)
                np.savetxt(model_dir / f"sensor_{sensor}_round{round_id}_ang{angle}.csv", values, delimiter=",")
                gather = np.zeros((TEST_GATHER_COUNT, processing.GATHER_COLUMN_COUNT))
                gather[:, 0] = np.arange(TEST_GATHER_COUNT)
                gather[:, processing.GATHER_X_COLUMN] = summary.FLOW_SPEED_BY_SENSOR[sensor] * gather_times / processing.GATHER_DISPLACEMENT_TO_METERS
                np.savetxt(model_dir / f"gather_{sensor}_round{round_id}_ang{angle}.csv", gather, delimiter=" ")
    return root, layout_csv, config


def test_batch_filters_averages_preserves_materials_and_loads(tmp_path: Path) -> None:
    """完整测试 72 对中剔除一对、合格两轮平均、材料随排布保留及模型读取兼容。"""
    root, layout_csv, config = _write_batch(tmp_path)
    output, audit = tmp_path / "output", tmp_path / "audit"
    arguments = processing.build_parser().parse_args([
        "--input-root", str(root), "--layout-csv", str(layout_csv), "--config", str(config),
        "--output-dir", str(output), "--audit-dir", str(audit),
        "--start-seconds", str(TEST_START), "--end-seconds", str(TEST_END),
    ])
    manifest = processing.process_batch(arguments)
    records = [json.loads(line) for line in (output / processing.DATASET_FILENAME).read_text(encoding="utf-8").splitlines()]
    comparisons = list(csv.DictReader((output / processing.REPEATABILITY_FILENAME).open(encoding="utf-8-sig", newline="")))
    assert manifest["repeatability_pair_count"] == 72
    assert len(records) == manifest["retained_pair_count"] == 71
    assert manifest["rejected_pair_count"] == 1
    assert manifest["uniform_motion_file_count"] == 144
    rejected = [row for row in comparisons if row["retained"] == "False"]
    assert len(rejected) == 1
    assert (int(rejected[0]["model_id"]), int(rejected[0]["angle_deg"]), float(rejected[0]["flow_speed_m_s"])) == TEST_REJECTED_KEY
    for record in records:
        assert sorted(plant["stiffness"] for plant in record["plants"]) == sorted(TEST_STIFFNESS.values())

    # 用旧脚本完整的 filter_frame 入口独立核对 15° 的两轮输出，验证逐列处理顺序。
    expected_means = []
    for round_id in processing.ROUND_IDS:
        source = root / "model1" / f"sensor_2_round{round_id}_ang15.csv"
        frame = pd.read_csv(source, header=None)
        first, stop = processing.time_window_indices(len(frame), arguments.sensor_fs, TEST_START, TEST_END)
        filter_config = filtering.FilterConfig(
            input_path=root, output_dir=output, mode="lowpass", sample_rate=arguments.sensor_fs,
            cutoff=arguments.cutoff_hz, band_low=filtering.bandpass_low, band_high=filtering.bandpass_high,
            order=arguments.filter_order, numtaps=filtering.fir_numtaps,
            columns=tuple(range(len(summary.SENSOR_COLUMNS))), trim_head=first, trim_tail=stop,
            max_nan_gap=0, make_plots=False, overwrite=False,
        )
        filtered = filtering.filter_frame(frame.iloc[first:stop].reset_index(drop=True), filter_config)
        expected_means.append(summary.rotate_mean_force(summary.calculate_trimmed_column_means(filtered.to_numpy(), arguments.trim_fraction), 15))
    selected = next(record for record in records if record["angle"] == 15 and record["flow_speed"] == 0.2)
    assert list(selected["measurements"].values()) == pytest.approx(np.mean(expected_means, axis=0))
    assert len(HydroDataset(output / processing.DATASET_FILENAME)) == len(records)

    # 同目录默认拒绝覆盖；缺少 gather 时也须报错，不能少产出一轮而不告知。
    with pytest.raises(FileExistsError):
        processing.process_batch(arguments)

    # 故意截短一轮：默认中止，显式选择后整对删除，缺失力留空并给出高误差标记。
    short_source = root / "model1" / "sensor_3_round1_ang15.csv"
    short_source.write_text(short_source.read_text().splitlines()[0] + "\n", encoding="utf-8")
    arguments.output_dir, arguments.audit_dir = tmp_path / "short_output", tmp_path / "short_audit"
    with pytest.raises(ValueError, match="sensor_3_round1_ang15"):
        processing.process_batch(arguments)
    require_path = arguments.output_dir / processing.DATASET_FILENAME
    assert not require_path.exists()
    arguments.short_sensor_policy = "drop-pair"
    short_manifest = processing.process_batch(arguments)
    short_comparisons = list(csv.DictReader((arguments.output_dir / processing.REPEATABILITY_FILENAME).open(encoding="utf-8-sig", newline="")))
    unavailable = [row for row in short_comparisons if row["comparison_status"] == "incomplete_sensor_window"]
    assert short_manifest["noncomparable_pair_count"] == len(unavailable) == 1
    assert short_manifest["retained_pair_count"] == 70
    assert unavailable[0]["round1_FX_0_N"] == unavailable[0]["absolute_difference_N"] == ""
    assert float(unavailable[0]["difference_ratio_percent"]) == processing.DEFAULT_INCOMPLETE_ERROR_PERCENT
    assert "非实测差率" in unavailable[0]["exclusion_reason"]
    assert unavailable[0]["retained"] == "False"
    (root / "model1" / "gather_2_round1_ang0.csv").unlink()
    with pytest.raises(ValueError, match="缺少"):
        processing.discover_sources(root, (1,))
