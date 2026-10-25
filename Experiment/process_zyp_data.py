"""处理 zyp 的 17 组双轮原始实验，生成 JSONL 和重复性 CSV。

沿用既有脚本的低通、逐列截尾平均、力旋转和 schema 3 构建函数。
sensor 与 gather 起点对齐，但采样率分别是 1000 Hz 和 100 Hz。
gather 仅用于检查匀速运动，不把实测运动速度替换为实验流速标签。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# 路径和命令行默认值集中在开头；从任意工作目录运行都定位到同一仓库。
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "zyp"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "zyp_20261021"
DEFAULT_AUDIT_DIR = REPO_ROOT / "文档" / "工作记录" / "zyp数据处理20261021"
DEFAULT_LAYOUT_CSV = SCRIPT_DIR / "input.csv"
DEFAULT_CONFIG = SCRIPT_DIR / "stiffness_config.yaml"
DEFAULT_SENSOR_FS = 1000.0  # sensor 每秒采样点数。
DEFAULT_GATHER_FS = 100.0  # gather 每秒采样点数，已由用户确认。
DEFAULT_START_SECONDS = 3.0  # 包含这个时刻。
DEFAULT_END_SECONDS = 4.99  # 不包含这个时刻，避开 5 s 的边界。
DEFAULT_CUTOFF_HZ = 2.0  # 沿用历史 Butterworth 低通截止频率。
DEFAULT_FILTER_ORDER = 5  # Butterworth 阶数，沿用旧流程。
DEFAULT_TRIM_FRACTION = 0.10  # 每列两端分别去除的比例。
DEFAULT_REPEATABILITY_THRESHOLD = None  # None 表示读取 YAML 中的百分数阈值。
DEFAULT_SPEED_CV_PERCENT = 1.0  # 差分速度的变异系数上限，仅作运动诊断。
DEFAULT_SPEED_MAX_DEVIATION_PERCENT = 5.0  # 单步速度相对均速偏差上限，仅作诊断。
DEFAULT_OVERWRITE = False
DEFAULT_SHORT_SENSOR_POLICY = "error"  # 默认中止；drop-pair 显式排除窗口不足的整对。
DEFAULT_INCOMPLETE_ERROR_PERCENT = 10000.0  # 用户要求短记录给很高误差；此值是标记，不是实测值。
MAX_MEASURED_SYMMETRIC_DIFFERENCE_PERCENT = 200.0  # 真实对称差率的数学上限。

# 固定的本批次实验矩阵、单位换算和输出名称，也集中声明以便核对。
ANGLE_STEP_DEGREES = 15
ANGLE_CYCLE_DEGREES = 360
SENSOR_IDS = (2, 3, 4)
ROUND_IDS = (1, 2)
GATHER_COLUMN_COUNT = 19
GATHER_X_COLUMN = 1  # 从 0 开始计数：第 2 列为 X 位移。
GATHER_DISPLACEMENT_TO_METERS = 1.0e-6  # 旧工具除 1000 得 mm，再除 1000 得 m。
FX_COLUMN = 3  # TX,TY,TZ,FX,FY,FZ 中 FX 的位置。
PERCENT_SCALE = 100.0
INDEX_ROUNDING_TOLERANCE = 1.0e-9  # 防止十进制时间的浮点误差多取一个点。
HASH_BLOCK_BYTES = 1024 * 1024
DATASET_FILENAME = "dataset.jsonl"
REPEATABILITY_FILENAME = "repeatability.csv"
AUDIT_FILENAME = "逐轮处理与运动诊断.csv"
MANIFEST_FILENAME = "处理摘要.json"
FILE_PATTERN = re.compile(
    r"^(?P<kind>sensor|gather)_(?P<sensor>\d+)_round(?P<round>\d+)_ang(?P<angle>\d+)\.csv$"
)
MODEL_PATTERN = re.compile(r"^model(?P<model>\d+)$")

# 加入仓库根目录后，同时支持直接执行与 pytest 的包导入。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from Experiment import filter_sensor_data as filtering
from Experiment import gather_data as gathering
from Experiment import summarize_sensor_data as summary


def file_sha256(path: Path) -> str:
    """分块计算 path 文件的 SHA-256 摘要，用于确定复现时是否为相同原始文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def time_window_indices(count: int, fs: float, start: float, end: float) -> tuple[int, int]:
    """返回左闭右开窗口的行索引；count 是总行数，fs 为 Hz，start/end 为秒。

    第一行视为 t=0。非整数采样边界向上取整，保证实际时刻满足 start<=t<end。
    数据不足时明确报错，不把短文件悄悄当成完整时间窗口。
    """
    if not all(np.isfinite(value) for value in (fs, start, end)) or fs <= 0 or not 0 <= start < end:
        raise ValueError("采样率必须为正有限数，时间需满足 0 <= start < end。")
    first = int(np.ceil(start * fs - INDEX_ROUNDING_TOLERANCE))
    stop = int(np.ceil(end * fs - INDEX_ROUNDING_TOLERANCE))
    if first < 0 or stop > count or stop - first < 2:
        raise ValueError(f"{count} 行、{fs} Hz 无法提供完整的 [{start}, {end}) s 窗口。")
    return first, stop


def discover_sources(input_root: Path, model_ids: Sequence[int]) -> dict[tuple[int, int, int, int], dict[str, Path]]:
    """检查原始文件完整性，返回 (model, angle, sensor, round) 对应的文件对。

    input_root 是含 modelN 的父目录；model_ids 来自构型表，不依赖旧批次的
    model1～14 单轮限制。每个条件必须同时有 sensor 和 gather，两轮必须齐全。
    """
    if not input_root.is_dir():
        raise FileNotFoundError(f"原始目录不存在：{input_root}")
    found: dict[tuple[int, int, int, int], dict[str, Path]] = {}
    for path in input_root.rglob("*.csv"):
        model_match = MODEL_PATTERN.fullmatch(path.parent.name)
        file_match = FILE_PATTERN.fullmatch(path.name)
        if path.parent.parent != input_root or model_match is None or file_match is None:
            raise ValueError(f"原始 CSV 目录或命名不符合本批次约定：{path}")
        key = (int(model_match["model"]), int(file_match["angle"]), int(file_match["sensor"]), int(file_match["round"]))
        kind = file_match["kind"]
        if kind in found.setdefault(key, {}):
            raise ValueError(f"同一工况重复出现 {kind} 文件：{key}")
        found[key][kind] = path

    # 对照完整笛卡尔积检查，避免缺失工况在后续分组中被静默忽略。
    expected = {
        (model, angle, sensor, round_id)
        for model in model_ids
        for angle in range(0, ANGLE_CYCLE_DEGREES, ANGLE_STEP_DEGREES)
        for sensor in SENSOR_IDS
        for round_id in ROUND_IDS
    }
    if set(found) != expected:
        raise ValueError(f"实验矩阵不完整：缺失={sorted(expected - set(found))}，额外={sorted(set(found) - expected)}")
    incomplete = [key for key, files in found.items() if set(files) != {"sensor", "gather"}]
    if incomplete:
        raise ValueError(f"以下工况缺少 sensor 或 gather：{incomplete}")
    return dict(sorted(found.items()))


def inspect_gather(path: Path, fs: float, start: float, end: float, nominal_speed: float,
                   cv_limit: float, deviation_limit: float) -> dict[str, object]:
    """检查 gather 位移，返回匀速指标；参数速度单位 m/s，两个阈值单位 %。

    fs/start/end 定义与 sensor 同步的秒窗口；nominal_speed 是文件名代表的流速。
    cv_limit 限制差分速度的标准差/平均绝对速度；deviation_limit 限制最大偏差。
    这里只标记运动情况，最终 JSONL 的筛选规则仍是原流程的 FX 重复性阈值。
    """
    # gather 虽使用 .csv 扩展名，实际按空白分隔；复用旧解析器避免混同 sensor。
    values = gathering.read_gather_csv(path).to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] != GATHER_COLUMN_COUNT or not np.isfinite(values).all():
        raise ValueError(f"gather 必须为有限数值的 {GATHER_COLUMN_COUNT} 列：{path}")
    first, stop = time_window_indices(len(values), fs, start, end)
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError(f"gather 第一列记录计数不严格递增：{path}")

    # 位移先减均值，时间先居中，再拟合 x=v*t+b，减少大绝对坐标的舍入影响。
    times = np.arange(first, stop, dtype=float) / fs
    displacement = values[first:stop, GATHER_X_COLUMN] * GATHER_DISPLACEMENT_TO_METERS
    centered_time = times - times.mean()
    centered_displacement = displacement - displacement.mean()
    slope = float(np.dot(centered_time, centered_displacement) / np.dot(centered_time, centered_time))
    residual = centered_displacement - slope * centered_time
    velocities = np.diff(displacement) * fs
    mean_velocity = float(velocities.mean())

    # 静止记录不能作为匀速拖曳通过；None 在 CSV 中留空，不伪造为零波动。
    moving = mean_velocity != 0.0
    cv = float(velocities.std(ddof=0) / abs(mean_velocity) * PERCENT_SCALE) if moving else None
    maximum_deviation = float(np.max(np.abs(velocities - mean_velocity)) / abs(mean_velocity) * PERCENT_SCALE) if moving else None
    reversals = int(np.sum(velocities * mean_velocity <= 0)) if moving else len(velocities)
    uniform = moving and cv <= cv_limit and maximum_deviation <= deviation_limit and reversals == 0
    tick_steps = np.diff(values[:, 0])
    return {
        "gather_rows": len(values), "gather_window_rows": stop - first,
        "gather_first_time_s": float(times[0]), "gather_last_time_s": float(times[-1]),
        "gather_tick_step_min": float(tick_steps.min()), "gather_tick_step_max": float(tick_steps.max()),
        "fitted_velocity_m_s": slope, "speed_error_percent": (abs(slope) / nominal_speed - 1) * PERCENT_SCALE,
        "velocity_cv_percent": cv, "maximum_velocity_deviation_percent": maximum_deviation,
        "maximum_position_residual_m": float(np.max(np.abs(residual))),
        "nonforward_steps": reversals, "uniform_motion": bool(uniform),
    }


def write_audit_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    """将 rows 的完整逐轮测量与运动指标原子写入 path；字段顺序取第一条记录。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def process_batch(arguments: argparse.Namespace) -> dict[str, object]:
    """执行整批处理并返回摘要；arguments 由 build_parser 定义，路径可用 CLI 覆盖。"""
    input_root = arguments.input_root.resolve()
    output_dir = arguments.output_dir.resolve()
    audit_dir = arguments.audit_dir.resolve()
    configuration = summary.load_stiffness_config(arguments.config)
    mapping = configuration["stiffness_by_code"]
    classifications = summary.load_rotated_classifications(arguments.layout_csv, mapping)
    model_ids = sorted({model for model, _ in classifications})
    threshold = configuration["repeatability_threshold_percent"] if arguments.repeatability_threshold is None else arguments.repeatability_threshold

    # 参数错误在批处理前暴露；gather 和 sensor 的窗口分别计算，不能共用行索引。
    positive_parameters = (arguments.sensor_fs, arguments.gather_fs, arguments.speed_cv_limit, arguments.speed_deviation_limit)
    if any(not np.isfinite(value) or value <= 0 for value in positive_parameters):
        raise ValueError("采样率和匀速诊断阈值必须为正有限数。")
    if not np.isfinite(threshold) or threshold < 0 or not np.isfinite(arguments.trim_fraction) or not 0 <= arguments.trim_fraction < 0.5:
        raise ValueError("重复性阈值须非负有限；每端截尾比例须满足 0<=fraction<0.5。")
    if not np.isfinite(arguments.incomplete_error_percent) or arguments.incomplete_error_percent <= max(threshold, MAX_MEASURED_SYMMETRIC_DIFFERENCE_PERCENT):
        raise ValueError("短记录误差标记必须是有限数，且高于重复性阈值和真实对称差率上限 200%。")
    time_window_indices(int(np.ceil(arguments.end_seconds * arguments.sensor_fs)), arguments.sensor_fs, arguments.start_seconds, arguments.end_seconds)
    if output_dir.is_relative_to(input_root) or audit_dir.is_relative_to(input_root):
        raise ValueError("处理结果不得写进原始数据目录。")
    outputs = [output_dir / DATASET_FILENAME, output_dir / REPEATABILITY_FILENAME,
               audit_dir / AUDIT_FILENAME, audit_dir / MANIFEST_FILENAME]
    if len(set(outputs)) != len(outputs):
        raise ValueError("各个输出文件必须使用不同路径。")
    for target in outputs:
        if target.exists() and not arguments.overwrite:
            raise FileExistsError(f"输出已存在；确认要替换后使用 --overwrite：{target}")
    sources = discover_sources(input_root, model_ids)

    # 未使用的带通参数沿用原脚本，仅为构造同一 FilterConfig；六列均用低通。
    filter_config = filtering.FilterConfig(
        input_path=input_root, output_dir=output_dir, mode="lowpass",
        sample_rate=arguments.sensor_fs, cutoff=arguments.cutoff_hz,
        band_low=filtering.bandpass_low, band_high=filtering.bandpass_high,
        order=arguments.filter_order, numtaps=filtering.fir_numtaps,
        columns=tuple(range(len(summary.SENSOR_COLUMNS))), trim_head=0, trim_tail=0,
        max_nan_gap=0, make_plots=False, overwrite=arguments.overwrite,
    )
    filtering.validate_filter_config(filter_config)
    grouped: dict[tuple[int, int, int], dict[int, np.ndarray | None]] = {}
    audit_rows: list[dict[str, object]] = []
    for key, paths in sources.items():
        model, angle, sensor, round_id = key
        speed = summary.FLOW_SPEED_BY_SENSOR[sensor]
        values = summary.read_numeric_sensor_csv(paths["sensor"])
        # 短记录只能显式排除整对，不能改时间窗、补零或拿另一轮充作双轮平均。
        short_reason = ""
        try:
            first, stop = time_window_indices(len(values), arguments.sensor_fs, arguments.start_seconds, arguments.end_seconds)
        except ValueError as error:
            if arguments.short_sensor_policy == "error":
                raise ValueError(f"{paths['sensor']}：{error}") from error
            first = stop = None
            short_reason = str(error)
        means = None
        if not short_reason:
            # 与旧流程顺序一致：先截取，再按列低通，再各端截尾，最后旋转 FX/FY。
            cropped = values[first:stop]
            filtered = np.column_stack([filtering.lowpass_filter(cropped[:, column], filter_config)
                                        for column in filter_config.columns])
            means = summary.rotate_mean_force(summary.calculate_trimmed_column_means(filtered, arguments.trim_fraction), angle)
            if not np.isfinite(means).all():
                raise ValueError(f"滤波、截尾或旋转后出现非有限测量值：{paths['sensor']}")
        grouped.setdefault((model, angle, sensor), {})[round_id] = means
        motion = inspect_gather(paths["gather"], arguments.gather_fs, arguments.start_seconds,
                                arguments.end_seconds, speed, arguments.speed_cv_limit, arguments.speed_deviation_limit)

        # 每轮保留来源摘要、实际时间、六轴均值和运动诊断，便于追溯被筛除的记录。
        audit_rows.append({
            "model_id": model, "angle_deg": angle, "flow_speed_m_s": speed, "round": round_id,
            "sensor_file": paths["sensor"].relative_to(REPO_ROOT).as_posix() if paths["sensor"].is_relative_to(REPO_ROOT) else str(paths["sensor"]),
            "gather_file": paths["gather"].relative_to(REPO_ROOT).as_posix() if paths["gather"].is_relative_to(REPO_ROOT) else str(paths["gather"]),
            "sensor_sha256": file_sha256(paths["sensor"]), "gather_sha256": file_sha256(paths["gather"]),
            "sensor_rows": len(values), "sensor_window_rows": stop - first if not short_reason else 0,
            "sensor_first_time_s": first / arguments.sensor_fs if not short_reason else None,
            "sensor_last_time_s": (stop - 1) / arguments.sensor_fs if not short_reason else None,
            "sensor_window_available": not bool(short_reason), "sensor_exclusion_reason": short_reason,
            **{column: float(means[index]) if means is not None else None
               for index, column in enumerate(summary.MEASUREMENT_COLUMNS)}, **motion,
        })
        if angle == ANGLE_CYCLE_DEGREES - ANGLE_STEP_DEGREES and sensor == SENSOR_IDS[-1] and round_id == ROUND_IDS[-1]:
            print(f"已处理 model{model}：144 份 sensor 和 144 份 gather", flush=True)

    rows: list[dict[str, object]] = []
    repeatability_rows: list[dict[str, object]] = []
    for (model, angle, sensor), rounds in sorted(grouped.items()):
        speed = summary.FLOW_SPEED_BY_SENSOR[sensor]
        codes = classifications[(model, angle - angle % summary.ANCHOR_STEP_DEGREES)]
        incomplete = any(rounds[round_id] is None for round_id in ROUND_IDS)
        if incomplete:
            # 用户要求以很高误差标记短记录；绝对差和缺失力值仍留空，不伪造测量。
            difference = average_force = None
            ratio = arguments.incomplete_error_percent
            retained = False
            status = "incomplete_sensor_window"
            reason = "；".join(f"round{round_id} sensor 不足以覆盖指定时间窗口" for round_id in ROUND_IDS if rounds[round_id] is None)
            reason += f"；{ratio:g}% 为无效记录标记，非实测差率"
        else:
            difference, average_force, ratio = summary.calculate_repeatability(float(rounds[1][FX_COLUMN]), float(rounds[2][FX_COLUMN]))
            retained = ratio <= threshold
            status = "compared"
            reason = "" if retained else f"FX_0 对称差率超过 {threshold:g}%"
        repeatability_rows.append({
            "model_id": model, "plant_count": sum(code > 0 for code in codes), "angle_deg": angle,
            "flow_speed_m_s": speed, "round1_FX_0_N": float(rounds[1][FX_COLUMN]) if rounds[1] is not None else None,
            "round2_FX_0_N": float(rounds[2][FX_COLUMN]) if rounds[2] is not None else None, "absolute_difference_N": difference,
            "average_absolute_force_N": average_force, "difference_ratio_percent": ratio, "retained": retained,
            "comparison_status": status, "exclusion_reason": reason,
        })
        if retained:
            # 仅合格两轮等权平均；每个工况生成一条记录，不把轮次复制成独立样本。
            means = np.mean([rounds[round_id] for round_id in ROUND_IDS], axis=0)
            rows.append({
                "sample_id": summary.build_sample_id(model, angle, speed), "model_id": model,
                "angle": angle, "flow_speed": speed,
                "vegetation_layout": "".join("1" if code else "0" for code in codes),
                **dict(zip(summary.MEASUREMENT_COLUMNS, map(float, means))),
            })
    if not rows:
        raise ValueError("没有通过重复性阈值的工况，未写入数据集。")
    records = summary.build_dataset_records(rows, rotated_classifications=classifications, stiffness_by_code=mapping)

    # 全部计算成功后才创建产物；JSONL 严格沿用 schema 3，不添加训练端不认识的字段。
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    summary.write_dataset_jsonl(records, outputs[0], arguments.overwrite)
    # 前十列沿用历史重复性格式；另加状态和原因，并用 UTF-8 BOM 方便 Excel 读取中文。
    write_audit_csv(repeatability_rows, outputs[1])
    write_audit_csv(audit_rows, outputs[2])
    manifest = {
        "input_root": str(input_root), "output_dir": str(output_dir),
        "parameters": {name: str(value) if isinstance(value, Path) else value for name, value in vars(arguments).items()},
        "time_alignment": "sensor 与 gather 第 0 行均为 t=0，已由用户确认；截取采用左闭右开区间",
        "stiffness_by_code_N_m2": mapping, "repeatability_threshold_percent": threshold,
        "uniform_motion_rule": "速度非零、无反向或停步、差分速度CV及最大偏差均不超过参数阈值；仅诊断，不参与筛选",
        "gather_displacement_to_meters": GATHER_DISPLACEMENT_TO_METERS,
        "source_file_count": len(sources) * 2, "repeatability_pair_count": len(repeatability_rows),
        "retained_pair_count": len(rows), "rejected_pair_count": len(repeatability_rows) - len(rows),
        "noncomparable_pair_count": sum(row["comparison_status"] == "incomplete_sensor_window" for row in repeatability_rows),
        "incomplete_error_percent_marker": arguments.incomplete_error_percent,
        "repeatability_rejected_pair_count": sum(row["comparison_status"] == "compared" and not row["retained"] for row in repeatability_rows),
        "short_sensor_files": [row["sensor_file"] for row in audit_rows if not row["sensor_window_available"]],
        "uniform_motion_file_count": sum(row["uniform_motion"] for row in audit_rows),
        "nonuniform_motion_files": [row["gather_file"] for row in audit_rows if not row["uniform_motion"]],
        "by_model": {str(model): {"pairs": sum(row["model_id"] == model for row in repeatability_rows),
                                  "retained": sum(row["model_id"] == model for row in rows)} for model in model_ids},
        "by_speed": {str(summary.FLOW_SPEED_BY_SENSOR[sensor]): {
            "pairs": sum(row["flow_speed_m_s"] == summary.FLOW_SPEED_BY_SENSOR[sensor] for row in repeatability_rows),
            "retained": sum(row["flow_speed"] == summary.FLOW_SPEED_BY_SENSOR[sensor] for row in rows),
        } for sensor in SENSOR_IDS},
        "provenance_sha256": {str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path): file_sha256(path)
                              for path in (arguments.layout_csv.resolve(), arguments.config.resolve(), Path(__file__).resolve(),
                                           Path(filtering.__file__), Path(gathering.__file__), Path(summary.__file__), SCRIPT_DIR / "rotate_force.py",
                                           SCRIPT_DIR / "generator.py", REPO_ROOT / "model" / "hydro" / "geometry.py")},
        "output_sha256": {path.name: file_sha256(path) for path in outputs[:3]},
        "library_versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__},
    }
    temporary = outputs[3].with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(outputs[3])
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行入口；路径、采样率、时间窗口和处理参数均有开头的默认值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="modelN 原始目录的父目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="JSONL 与重复性 CSV 的独立输出目录")
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR, help="逐轮审计 CSV 和参数摘要所在的工作记录目录")
    parser.add_argument("--layout-csv", type=Path, default=DEFAULT_LAYOUT_CSV, help="每个 model 一行、37 列的初始材料排布")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="EI 材料映射与重复性阈值 YAML")
    parser.add_argument("--sensor-fs", type=float, default=DEFAULT_SENSOR_FS, help="sensor 采样率，Hz")
    parser.add_argument("--gather-fs", type=float, default=DEFAULT_GATHER_FS, help="gather 采样率，Hz")
    parser.add_argument("--start-seconds", type=float, default=DEFAULT_START_SECONDS, help="包含的起始时刻，s")
    parser.add_argument("--end-seconds", type=float, default=DEFAULT_END_SECONDS, help="不包含的截止时刻，s")
    parser.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ, help="低通截止频率，Hz")
    parser.add_argument("--filter-order", type=int, default=DEFAULT_FILTER_ORDER, help="Butterworth 低通阶数")
    parser.add_argument("--trim-fraction", type=float, default=DEFAULT_TRIM_FRACTION, help="每列每端去除的比例")
    parser.add_argument("--repeatability-threshold", type=float, default=DEFAULT_REPEATABILITY_THRESHOLD, help="FX 对称差率阈值，百分数；省略读取 YAML")
    parser.add_argument("--speed-cv-limit", type=float, default=DEFAULT_SPEED_CV_PERCENT, help="仅诊断：差分速度 CV 上限，百分数")
    parser.add_argument("--speed-deviation-limit", type=float, default=DEFAULT_SPEED_MAX_DEVIATION_PERCENT, help="仅诊断：单步速度最大偏差上限，百分数")
    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE, help="允许覆盖指定输出目录中本脚本的四个产物")
    parser.add_argument("--short-sensor-policy", choices=("error", "drop-pair"), default=DEFAULT_SHORT_SENSOR_POLICY,
                        help="sensor 不足窗口时中止，或显式排除整对并在 CSV 中记录原因")
    parser.add_argument("--incomplete-error-percent", type=float, default=DEFAULT_INCOMPLETE_ERROR_PERCENT,
                        help="用户指定的短记录高误差标记，百分数；不是实测差率")
    return parser


def main() -> int:
    """解析 CLI 并处理数据；成功返回 0，输入或写入错误返回 1。"""
    try:
        manifest = process_batch(build_parser().parse_args())
    except (OSError, ValueError) as error:
        print(f"处理失败：{error}", file=sys.stderr)
        return 1
    print(f"完成：{manifest['repeatability_pair_count']} 对，保留 {manifest['retained_pair_count']} 条，"
          f"删除 {manifest['rejected_pair_count']} 对；匀速检查通过 {manifest['uniform_motion_file_count']} 份。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
