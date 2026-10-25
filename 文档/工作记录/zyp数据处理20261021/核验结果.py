"""独立核验本批次的重复性、两轮平均、材料几何和原始序列抽查，不改实验数据。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal


# 所有路径和核验参数集中在开头；命令行可覆盖数据位置和报告位置。
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO_ROOT / "datasets" / "zyp_20261021" / "dataset.jsonl"
DEFAULT_REPEATABILITY = DEFAULT_DATASET.parent / "repeatability.csv"
DEFAULT_AUDIT = Path(__file__).resolve().parent / "逐轮处理与运动诊断.csv"
DEFAULT_REPORT = Path(__file__).resolve().parent / "核验摘要.json"
DEFAULT_LAYOUT = REPO_ROOT / "Experiment" / "input.csv"
DEFAULT_CONFIG = REPO_ROOT / "Experiment" / "stiffness_config.yaml"
DEFAULT_CHECK_MODEL_LOADER = True
EXPECTED_MODELS = set(range(1, 18))
EXPECTED_PAIRS = 1224
EXPECTED_RUNS = 2448
EXPECTED_SENSOR_POINTS = 1990
EXPECTED_GATHER_POINTS = 199
EXPECTED_SENSOR_LAST_TIME = 4.989
EXPECTED_GATHER_LAST_TIME = 4.98
EXPECTED_INCOMPLETE_ERROR_PERCENT = 10000.0
SENSOR_FS = 1000.0
START_ROW = 3000
STOP_ROW = 4990
CUTOFF_HZ = 2.0
FILTER_ORDER = 5
TRIM_FRACTION = 0.10
COORDINATE_DECIMALS = 10
NUMERIC_ATOL = 1.0e-11
NUMERIC_RTOL = 1.0e-11
MICROMETERS_PER_METER = 1.0e6
SPOT_CHECK_KEYS = ((1, 0, 0.2, 1), (4, 45, 0.3, 2), (8, 90, 0.4, 1),
                   (12, 195, 0.3, 2), (17, 345, 0.4, 2), (17, 0, 0.2, 1))
MEASUREMENT_COLUMNS = ("TX", "TY", "TZ", "FX_0", "FY_0", "FZ")

# 这里只复用固定格点定义及材料配置解析，不复用待核验的筛选和平均实现。
sys.path.insert(0, str(REPO_ROOT))
from Experiment.summarize_sensor_data import load_stiffness_config
from model.hydro.geometry import build_hex_coordinates


def require(condition: bool, message: str) -> None:
    """condition 不满足时用 message 终止核验；不依赖可被 Python -O 禁用的 assert。"""
    if not condition:
        raise ValueError(message)


def close(actual: object, expected: object, message: str) -> None:
    """用指定浮点容差比较 actual 与 expected；不相等时报告 message。"""
    require(bool(np.allclose(actual, expected, atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL)), message)


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取带表头的 path，兼容 UTF-8 BOM，返回保留原始精度的字符串字典列表。"""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify(arguments: argparse.Namespace) -> dict[str, object]:
    """核验 CLI 指定的产物并返回统计摘要；任何不一致都停止，不修补或覆盖数据。"""
    records = [json.loads(line) for line in arguments.dataset.read_text(encoding="utf-8").splitlines()]
    comparisons, runs = read_csv(arguments.repeatability), read_csv(arguments.audit)
    configuration = load_stiffness_config(arguments.config)
    threshold = configuration["repeatability_threshold_percent"]
    mapping = configuration["stiffness_by_code"]
    with arguments.layout.open(encoding="utf-8-sig", newline="") as handle:
        layouts = [list(map(int, row)) for row in csv.reader(handle) if row]
    samples = {(record["model_id"], record["angle"], record["flow_speed"]): record for record in records}
    by_run = {(int(row["model_id"]), int(row["angle_deg"]), float(row["flow_speed_m_s"]), int(row["round"])): row for row in runs}
    by_pair = {(int(row["model_id"]), int(row["angle_deg"]), float(row["flow_speed_m_s"])): row for row in comparisons}
    require(len(comparisons) == len(by_pair) == EXPECTED_PAIRS, "重复性工况数量或唯一性不符")
    require(len(runs) == len(by_run) == EXPECTED_RUNS, "逐轮记录数量或唯一性不符")
    require(len(records) == len(samples) == len({record["sample_id"] for record in records}), "JSONL 工况或 ID 重复")
    require({record["model_id"] for record in records} == EXPECTED_MODELS, "保留数据未覆盖全部 model")

    # 独立重算每一对的差率，再核对入选集合以及全部六个测量值的两轮平均。
    retained_keys = set()
    noncomparable = []
    for key, row in by_pair.items():
        if row["comparison_status"] == "incomplete_sensor_window":
            require(row["retained"] == "False" and key not in samples, f"不完整工况被收入：{key}")
            close(float(row["difference_ratio_percent"]), EXPECTED_INCOMPLETE_ERROR_PERCENT, f"短记录高误差标记不符：{key}")
            require(row["absolute_difference_N"] == ""
                    and row["average_absolute_force_N"] == "", f"不完整工况伪造了统计量：{key}")
            require("非实测差率" in row["exclusion_reason"], f"未说明短记录标记含义：{key}")
            for round_id in (1, 2):
                require(row[f"round{round_id}_FX_0_N"] == by_run[(*key, round_id)]["FX_0"], f"短记录来源不符：{key}")
            require(any(by_run[(*key, round_id)]["sensor_window_available"] == "False" for round_id in (1, 2)), f"不完整标记无依据：{key}")
            noncomparable.append(key)
            continue
        require(row["comparison_status"] == "compared", f"未知比较状态：{key}")
        first, second = float(row["round1_FX_0_N"]), float(row["round2_FX_0_N"])
        denominator = (abs(first) + abs(second)) / 2
        ratio = abs(first - second) / denominator * 100 if denominator else 0.0
        close(float(row["difference_ratio_percent"]), ratio, f"差率不符：{key}")
        close(float(row["absolute_difference_N"]), abs(first - second), f"绝对差不符：{key}")
        close(float(row["average_absolute_force_N"]), denominator, f"平均绝对力不符：{key}")
        require((row["retained"] == "True") == (ratio <= threshold), f"保留标记不符：{key}")
        close(first, float(by_run[(*key, 1)]["FX_0"]), f"round1 来源不符：{key}")
        close(second, float(by_run[(*key, 2)]["FX_0"]), f"round2 来源不符：{key}")
        if ratio <= threshold:
            retained_keys.add(key)
            require(key in samples, f"合格工况缺失：{key}")
            for column in MEASUREMENT_COLUMNS:
                expected = (float(by_run[(*key, 1)][column]) + float(by_run[(*key, 2)][column])) / 2
                close(samples[key]["measurements"][column], expected, f"两轮平均不符：{key}/{column}")
    require(set(samples) == retained_keys, "JSONL 包含不合格工况")

    # 从初始排布直接旋转完整角度，独立验证分段锚点旋转的真实坐标和逐株材料。
    coordinates = build_hex_coordinates()
    for key, record in samples.items():
        model, angle, _ = key
        cosine, sine = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        expected = sorted((round(x * cosine + y * sine, COORDINATE_DECIMALS),
                           round(-x * sine + y * cosine, COORDINATE_DECIMALS), mapping[code])
                          for (x, y), code in zip(coordinates, layouts[model - 1]) if code)
        actual = sorted((round(plant["x"], COORDINATE_DECIMALS), round(plant["y"], COORDINATE_DECIMALS), plant["stiffness"])
                        for plant in record["plants"])
        require(actual == expected, f"真实坐标或材料对应不符：{key}")
        require(record["schema_version"] == 3, f"schema 版本不符：{key}")
        require(all(plant["angle"] == angle for plant in record["plants"]), f"逐株角度不符：{key}")

    # 核对全部实际采样边界；gather 比 sensor 少十倍采样点是独立采样率的结果。
    for row in runs:
        if row["sensor_window_available"] == "True":
            require(int(row["sensor_window_rows"]) == EXPECTED_SENSOR_POINTS, "sensor 窗口点数不符")
            close(float(row["sensor_first_time_s"]), 3.0, "sensor 起点不符")
            close(float(row["sensor_last_time_s"]), EXPECTED_SENSOR_LAST_TIME, "sensor 终点不符")
        else:
            require(int(row["sensor_window_rows"]) == 0 and all(row[column] == "" for column in MEASUREMENT_COLUMNS), "短记录被补造均值")
            require(int(row["sensor_rows"]) < STOP_ROW, "短记录判定不符")
        require(int(row["gather_window_rows"]) == EXPECTED_GATHER_POINTS, "gather 窗口点数不符")
        close(float(row["gather_first_time_s"]), 3.0, "gather 起点不符")
        close(float(row["gather_last_time_s"]), EXPECTED_GATHER_LAST_TIME, "gather 终点不符")

    # 六份原始文件用 SciPy 二维滤波接口直接重算，与处理脚本的逐列流程交叉检查。
    sos = signal.butter(FILTER_ORDER, CUTOFF_HZ, fs=SENSOR_FS, output="sos")
    for key in SPOT_CHECK_KEYS:
        row = by_run[key]
        raw = np.loadtxt(REPO_ROOT / row["sensor_file"], delimiter=",")
        filtered = signal.sosfiltfilt(sos, raw[START_ROW:STOP_ROW], axis=0)
        remove = int(np.floor(len(filtered) * TRIM_FRACTION))
        means = np.sort(filtered, axis=0)[remove:len(filtered) - remove].mean(axis=0)
        cosine, sine = np.cos(np.deg2rad(key[1])), np.sin(np.deg2rad(key[1]))
        force_x, force_y = means[3], means[4]
        means[3], means[4] = force_x * cosine + force_y * sine, -force_x * sine + force_y * cosine
        close([float(row[column]) for column in MEASUREMENT_COLUMNS], means, f"原始数据重算不符：{key}")

    # 使用真实模型加载器完整读取，验证输出字段与训练入口兼容；不启动训练。
    loaded_count = None
    if arguments.check_model_loader:
        from model.hydro.data import HydroDataset
        loaded_count = len(HydroDataset(arguments.dataset))
        require(loaded_count == len(records), "模型加载数量不符")
    return {
        "dataset_records": len(records), "repeatability_pairs": len(comparisons), "retained_pairs": len(retained_keys),
        "rejected_pairs": len(comparisons) - len(retained_keys), "audited_runs": len(runs),
        "noncomparable_pairs": noncomparable,
        "repeatability_rejected_pairs": len(comparisons) - len(retained_keys) - len(noncomparable),
        "independently_recomputed_raw_files": len(SPOT_CHECK_KEYS), "model_loader_records": loaded_count,
        "uniform_motion_passed": sum(row["uniform_motion"] == "True" for row in runs),
        "maximum_velocity_cv_percent": max(float(row["velocity_cv_percent"]) for row in runs),
        "maximum_velocity_deviation_percent": max(float(row["maximum_velocity_deviation_percent"]) for row in runs),
        "maximum_position_residual_um": max(float(row["maximum_position_residual_m"]) for row in runs) * MICROMETERS_PER_METER,
        "speed_ranges_m_s": {str(speed): [min(float(row["fitted_velocity_m_s"]) for row in runs if float(row["flow_speed_m_s"]) == speed),
                                            max(float(row["fitted_velocity_m_s"]) for row in runs if float(row["flow_speed_m_s"]) == speed)]
                             for speed in (0.2, 0.3, 0.4)},
        "sensor_total_row_counts": sorted({int(row["sensor_rows"]) for row in runs}),
        "gather_total_row_counts": sorted({int(row["gather_rows"]) for row in runs}),
        "negative_FX_0_record_count": sum(record["measurements"]["FX_0"] < 0 for record in records),
        "repeatability_ratio_percentiles": dict(zip(("min", "median", "p95", "max"),
                                                   map(float, np.percentile([float(row["difference_ratio_percent"]) for row in comparisons if row["comparison_status"] == "compared"], (0, 50, 95, 100))))),
    }


def main() -> None:
    """支持命令行路径覆盖；核验全部成功后保存 JSON 摘要并打印简要结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="待核验 JSONL")
    parser.add_argument("--repeatability", type=Path, default=DEFAULT_REPEATABILITY, help="待核验重复性 CSV")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT, help="逐轮运动与处理审计 CSV")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="核验摘要输出位置")
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT, help="初始 37 格点材料类别表")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="材料 EI 和重复性阈值配置")
    parser.add_argument("--check-model-loader", action=argparse.BooleanOptionalAction, default=DEFAULT_CHECK_MODEL_LOADER, help="是否完整通过模型加载器读取")
    arguments = parser.parse_args()
    result = verify(arguments)
    arguments.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
