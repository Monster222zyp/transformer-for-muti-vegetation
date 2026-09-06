"""将滤波后的传感器序列整理为实验汇总 CSV 和模型数据 JSONL。

每个输出行代表一个 ``model_id × angle × flow_speed`` 实验 sample。脚本先对
六轴传感器序列进行逐列截尾平均，再把 ``FX/FY`` 旋转到 0° 坐标系，最后补齐
模型编号和旋转后的 37 位水草排布。``summarized_data.csv`` 用于人工审计；
``dataset.jsonl`` 额外展开每根水草的固定坐标和原始角度，供 ``HydroDataset``
直接读取。角度只用于物理默认阻力查表，不作为神经网络输入。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

try:
    # 直接执行脚本时，Experiment 目录会位于模块搜索路径中。
    from rotate_force import rotate_force_xy_to_zero_frame
except ModuleNotFoundError:  # pragma: no cover - 由包方式导入时使用。
    # pytest 等场景以 ``Experiment.summarize_sensor_data`` 导入，需要相对导入。
    from .rotate_force import rotate_force_xy_to_zero_frame


# 当前脚本和项目根目录用于构造与工作目录无关的默认路径。
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# 直接执行 Experiment 下的脚本时，Python 默认只把 Experiment 放入搜索路径。
# 显式加入仓库根目录后，可以复用模型侧唯一的六边形坐标定义，避免两边各维护
# 一套坐标公式而逐渐产生偏差。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.hydro.geometry import build_hex_coordinates, parse_layout


# 所有命令行参数的默认值集中硬编码在代码开头。需要临时处理其他数据时，可以
# 使用命令行覆盖；需要长期调整默认行为时，只修改这里即可。
DEFAULT_INPUT_PATH = REPO_ROOT / "filtered_data"
DEFAULT_OUTPUT_DIRECTORY = SCRIPT_DIR /"output"
DEFAULT_SUMMARIZED_PATH = DEFAULT_OUTPUT_DIRECTORY / "summarized_data.csv"
DEFAULT_DATASET_PATH = DEFAULT_OUTPUT_DIRECTORY / "dataset.jsonl"
DEFAULT_LAYOUT_PATH = SCRIPT_DIR / "input.csv"
DEFAULT_TRIM_FRACTION = 0.10
DEFAULT_OVERWRITE_EXISTING = True
DEFAULT_STRICT_VALIDATION = True
DATASET_SCHEMA_VERSION = 1

# sensor 编号与实验流速的固定对应关系，单位为 m/s。
FLOW_SPEED_BY_SENSOR: dict[int, float] = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
SENSOR_COLUMNS = ("TX", "TY", "TZ", "FX", "FY", "FZ")
OUTPUT_COLUMNS = (
    "sample_id",
    "model_id",
    "angle",
    "vegetation_layout",
    "TX",
    "TY",
    "TZ",
    "FX_0",
    "FY_0",
    "FZ",
    "flow_speed",
)
VALID_ANGLES = (0, 60, 120, 180, 240, 300)
EXPECTED_MODEL_IDS = tuple(range(1, 15))
EXPECTED_MISSING_CONDITIONS = ((3, 60, 0.2), (3, 240, 0.3), (9, 240, 0.1), (13, 240, 0.3))
MODEL_DIRECTORY_PATTERN = re.compile(r"^model_(?P<model_id>\d+)$", re.IGNORECASE)
SENSOR_FILE_RE = re.compile(
    r"^sensor_(?P<sensor>\d+)_ang(?P<angle>-?\d+(?:\.\d+)?)"
    r"(?:_(?:lowpass|bandpass)_filtered)?\.csv$",
    re.IGNORECASE,
)


def parse_sensor_filename(path: Path) -> tuple[int, float] | None:
    """从 ``sensor_A_angB`` 文件名提取 sensor 编号和角度。"""

    match = SENSOR_FILE_RE.fullmatch(path.name)
    return None if match is None else (int(match.group("sensor")), float(match.group("angle")))


def discover_sensor_files(input_root: Path, output_file: Path) -> list[Path]:
    """递归返回符合命名规则的传感器 CSV，并排除最终输出文件。"""

    if not input_root.is_dir():
        raise ValueError(f"输入路径必须是包含 model_N 子目录的目录：{input_root}")
    files = [
        candidate
        for candidate in input_root.rglob("*.csv")
        if candidate.is_file() and candidate.resolve() != output_file.resolve() and parse_sensor_filename(candidate)
    ]

    def sort_key(path: Path) -> tuple[str, float, int, str]:
        """按模型目录、角度、sensor 编号建立稳定排序键。"""

        parsed_name = parse_sensor_filename(path)
        assert parsed_name is not None
        sensor_number, angle = parsed_name
        return (path.parent.relative_to(input_root).as_posix().casefold(), angle, sensor_number, path.name.casefold())

    return sorted(files, key=sort_key)


def read_numeric_sensor_csv(path: Path) -> np.ndarray:
    """读取无表头六列 sensor CSV，并拒绝空值、文本与非有限数值。"""

    frame = pd.read_csv(path, header=None, sep=",")
    if frame.empty:
        raise ValueError(f"文件为空：{path}")
    if frame.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError(f"{path} 有 {frame.shape[1]} 列；sensor CSV 必须恰好有 6 列。")
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"文件包含文本、NaN 或 Inf：{path}")
    return values


def calculate_trimmed_column_means(values: np.ndarray, fraction: float) -> np.ndarray:
    """对六列数据分别去除两端比例后，返回对应的平均值。"""

    if values.ndim != 2 or values.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError("截尾平均输入必须是具有 6 列的二维数组。")
    if not 0 <= fraction < 0.5:
        raise ValueError("trim_fraction 必须满足 0 <= fraction < 0.5。")
    sample_count = values.shape[0]
    removed_per_side = int(np.floor(sample_count * fraction))
    if sample_count - 2 * removed_per_side <= 0:
        raise ValueError(f"{sample_count} 个样本在截尾后没有剩余数据。")
    sorted_values = np.sort(values, axis=0)
    end_index = sample_count - removed_per_side if removed_per_side else sample_count
    return sorted_values[removed_per_side:end_index, :].mean(axis=0)


def rotate_mean_force(means: np.ndarray, angle: float) -> np.ndarray:
    """将六列均值中的 ``FX/FY`` 转换到 0° 力坐标系。"""

    rotated = means.astype(float, copy=True)
    rotated[3], rotated[4] = rotate_force_xy_to_zero_frame(float(rotated[3]), float(rotated[4]), angle)
    return rotated


def _load_rotation_function(experiment_dir: Path) -> Callable[[list[int]], list[list[int]]]:
    """按路径加载实验侧的 ``get_rotated_groups``，保证排布旋转规则唯一。"""

    generator_path = experiment_dir / "generator.py"
    if not generator_path.is_file():
        raise FileNotFoundError(f"找不到旋转函数文件：{generator_path}")
    module_spec = importlib.util.spec_from_file_location("experiment_generator", generator_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"无法加载旋转函数文件：{generator_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    rotation_function = getattr(module, "get_rotated_groups", None)
    if not callable(rotation_function):
        raise AttributeError(f"{generator_path} 未提供 get_rotated_groups。")
    return rotation_function


def _parse_layout(row: Sequence[str], row_number: int) -> list[int]:
    """验证 ``input.csv`` 的一行是恰好 37 位的 0/1 基础排布。"""

    try:
        layout = [int(cell.strip()) for cell in row]
    except ValueError as error:
        raise ValueError(f"构型文件第 {row_number} 行包含非整数。") from error
    if len(layout) != 37 or any(value not in (0, 1) for value in layout):
        raise ValueError(f"构型文件第 {row_number} 行必须是 37 位 0/1 排布。")
    return layout


def load_rotated_layouts(input_csv: Path) -> dict[tuple[int, int], str]:
    """加载基础排布并返回 ``(model_id, angle) -> 旋转后排布字符串`` 映射。"""

    if not input_csv.is_file():
        raise FileNotFoundError(f"找不到构型文件：{input_csv}")
    base_layouts: list[list[int]] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if row and any(cell.strip() for cell in row):
                base_layouts.append(_parse_layout(row, row_number))
    rotation_function = _load_rotation_function(input_csv.parent)
    rotated_layouts: dict[tuple[int, int], str] = {}
    for model_id, base_layout in enumerate(base_layouts, start=1):
        rotations = rotation_function(base_layout)
        if len(rotations) != len(VALID_ANGLES):
            raise ValueError(f"model_{model_id} 的旋转函数未返回六个方向。")
        serialized_rotations: list[str] = []
        for angle, layout in zip(VALID_ANGLES, rotations):
            parsed_layout = _parse_layout([str(value) for value in layout], model_id)
            serialized_layout = "".join(str(value) for value in parsed_layout)
            rotated_layouts[(model_id, angle)] = serialized_layout
            serialized_rotations.append(serialized_layout)

        # 当前实验要求一个 model 的六个角度分别保存旋转后的排布。若六次旋转中
        # 出现重复，通常意味着输入构型具有旋转对称性，无法满足“不同角度使用不同
        # 01 序列”的数据契约，因此在汇总阶段直接拒绝，而不是静默写入相同排布。
        if len(set(serialized_rotations)) != len(VALID_ANGLES):
            raise ValueError(
                f"model_{model_id} 的六个旋转角度没有产生六个不同的 37 位排布。"
            )
    return rotated_layouts


def _model_id_for_source(source_file: Path, input_root: Path) -> int:
    """从直接父目录 ``model_N`` 提取模型编号，拒绝模糊的嵌套路径。"""

    relative_parent = source_file.parent.relative_to(input_root)
    if len(relative_parent.parts) != 1:
        raise ValueError(f"sensor 文件必须直接位于 model_N 目录：{source_file}")
    match = MODEL_DIRECTORY_PATTERN.fullmatch(relative_parent.name)
    if match is None:
        raise ValueError(f"sensor 文件父目录必须命名为 model_N：{source_file}")
    return int(match.group("model_id"))


def _validate_strict_baseline(rows: Sequence[dict[str, object]]) -> None:
    """验证当前正式实验的 332 行基线，阻止缺失或误混入工况。"""

    expected = {
        (model_id, angle, speed)
        for model_id in EXPECTED_MODEL_IDS
        for angle in VALID_ANGLES
        for speed in FLOW_SPEED_BY_SENSOR.values()
    }
    expected.difference_update(EXPECTED_MISSING_CONDITIONS)
    actual = {(int(row["model_id"]), int(row["angle"]), float(row["flow_speed"])) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"严格基线验证失败：应有 {len(expected)} 行，实际 {len(rows)} 行；"
            f"缺失={missing}，额外={unexpected}。可用 --no-strict 处理非正式批次。"
        )


def build_sample_id(model_id: int, angle: int, flow_speed: float) -> str:
    """构造不依赖行号的稳定 sample 标识。

    参数：
        model_id: 实验构型编号。
        angle: 当前实验原始旋转角度，单位为 degree。
        flow_speed: 当前实验流速，单位为 m/s。

    返回值：
        例如 ``model_001_angle_060_flow_0.2`` 的唯一字符串。即使以后 CSV 增删
        其他 sample，这个标识也不会因为行号变化而改变。
    """

    return f"model_{model_id:03d}_angle_{angle:03d}_flow_{flow_speed:.1f}"


def build_dataset_records(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """把 sample 汇总行转换为带逐株坐标和角度的 JSONL 记录。

    参数：
        rows: 已经完成排序和实验完整性校验的汇总行。

    返回值：
        与 ``rows`` 一一对应的字典列表；每个字典的 ``plants`` 是变长数组，
        每一项保存 ``plant_index、grid_index、x、y、angle``。
    """

    all_coordinates = build_hex_coordinates()
    records: list[dict[str, object]] = []
    seen_sample_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        if sample_id in seen_sample_ids:
            raise ValueError(f"第 {row_number} 条汇总记录包含重复 sample_id：{sample_id}")
        seen_sample_ids.add(sample_id)

        model_id = int(row["model_id"])
        angle = int(row["angle"])
        layout = str(row["vegetation_layout"])
        parsed_layout = parse_layout(layout)

        # grid_index 使用完整 37 点网格中的固定编号；plant_index 只在当前 sample
        # 内从 0 连续编号。两者同时保存可以兼顾模型顺序和实验位置追溯。
        plants: list[dict[str, object]] = []
        for grid_index, occupied in enumerate(parsed_layout):
            if not occupied:
                continue
            x_coordinate, y_coordinate = all_coordinates[grid_index]
            plants.append(
                {
                    "plant_index": len(plants),
                    "grid_index": grid_index,
                    "x": float(x_coordinate),
                    "y": float(y_coordinate),
                    # 当前实验中，同一 sample 内所有水草都采用该 sample 的原始角度。
                    "angle": angle,
                }
            )
        if not plants:
            raise ValueError(f"sample {sample_id} 的 vegetation_layout 中没有水草。")

        records.append(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "sample_id": sample_id,
                "model_id": model_id,
                "angle": angle,
                "flow_speed": float(row["flow_speed"]),
                # 仍以字符串保存 37 位 01 序列，避免前导 0 丢失或超过浮点有效位数。
                "vegetation_layout": layout,
                "plants": plants,
                "measurements": {
                    column: float(row[column])
                    for column in ("TX", "TY", "TZ", "FX_0", "FY_0", "FZ")
                },
            }
        )
    return records


def write_summarized_csv(
    rows: Sequence[dict[str, object]], output_file: Path, overwrite: bool
) -> None:
    """带表头原子写入 sample 级汇总 CSV。"""

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"输出已存在；使用 --overwrite 可覆盖：{output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    # 使用 UTF-8 写入标准表头；读取端仍兼容带 BOM 的外部 CSV。
    with temporary_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_file.replace(output_file)


def write_dataset_jsonl(
    records: Sequence[dict[str, object]], output_file: Path, overwrite: bool
) -> None:
    """以 UTF-8 JSONL 格式原子写入模型数据文件。

    参数：
        records: 一行一个 sample 的完整 JSON 可序列化记录。
        output_file: ``dataset.jsonl`` 的输出路径。
        overwrite: 为 ``False`` 且目标已存在时拒绝覆盖。
    """

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"输出已存在；使用 --overwrite 可覆盖：{output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary_file.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            # separators 去除无意义空格，使一条 sample 严格占据一个物理文本行。
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary_file.replace(output_file)


def summarize_sensor_tree(
    input_root: Path,
    output_file: Path,
    dataset_file: Path,
    input_csv: Path,
    fraction: float,
    overwrite: bool,
    strict: bool,
) -> list[dict[str, object]]:
    """汇总完整模型目录树，并同时写出 CSV 与 JSONL。"""

    if not 0 <= fraction < 0.5:
        raise ValueError("trim_fraction 必须满足 0 <= fraction < 0.5。")
    if output_file.resolve() == dataset_file.resolve():
        raise ValueError("summarized CSV 与 dataset JSONL 不能使用同一个输出路径。")
    files = discover_sensor_files(input_root, output_file)
    if not files:
        raise FileNotFoundError(f"没有找到符合 sensor_A_angB 规则的 CSV：{input_root}")
    rotated_layouts = load_rotated_layouts(input_csv.resolve())
    grouped_rows: dict[tuple[int, int], dict[int, np.ndarray]] = defaultdict(dict)

    for source_file in files:
        parsed_name = parse_sensor_filename(source_file)
        assert parsed_name is not None
        sensor_number, raw_angle = parsed_name
        try:
            model_id = _model_id_for_source(source_file, input_root)
            if not raw_angle.is_integer() or int(raw_angle) not in VALID_ANGLES:
                raise ValueError(f"不支持的角度 {raw_angle}；允许值为 {VALID_ANGLES}。")
            angle = int(raw_angle)
            if sensor_number not in FLOW_SPEED_BY_SENSOR:
                raise KeyError(f"sensor_{sensor_number} 没有流速映射。")
            sensor_rows = grouped_rows[(model_id, angle)]
            if sensor_number in sensor_rows:
                raise ValueError(f"model_{model_id}/ang{angle} 存在重复 sensor_{sensor_number} 文件。")
            sensor_rows[sensor_number] = rotate_mean_force(
                calculate_trimmed_column_means(read_numeric_sensor_csv(source_file), fraction), angle
            )
            print(f"[成功] 输入 {source_file} -> 截尾平均完成（流速 {FLOW_SPEED_BY_SENSOR[sensor_number]:g}）")
        except Exception as error:
            print(f"[失败] 输入 {source_file}：{error}；已跳过")

    rows: list[dict[str, object]] = []
    expected_sensors = set(FLOW_SPEED_BY_SENSOR)
    for (model_id, angle), sensor_rows in sorted(grouped_rows.items()):
        missing_sensors = sorted(expected_sensors - set(sensor_rows))
        if missing_sensors:
            speeds = "、".join(f"{FLOW_SPEED_BY_SENSOR[sensor]:g}" for sensor in missing_sensors)
            print(f"[错误] model_{model_id}/ang{angle} 缺少流速 {speeds}；仍将保留已有 sample。")
        layout = rotated_layouts.get((model_id, angle))
        if layout is None:
            print(f"[失败] model_{model_id}/ang{angle} 没有对应构型；已跳过该工况。")
            continue
        for sensor_number, means in sorted(sensor_rows.items()):
            flow_speed = FLOW_SPEED_BY_SENSOR[sensor_number]
            row = {
                "sample_id": build_sample_id(model_id, angle, flow_speed),
                "model_id": model_id,
                "angle": angle,
                "vegetation_layout": layout,
                "TX": float(means[0]),
                "TY": float(means[1]),
                "TZ": float(means[2]),
                "FX_0": float(means[3]),
                "FY_0": float(means[4]),
                "FZ": float(means[5]),
                "flow_speed": flow_speed,
            }
            numeric_columns = (
                "model_id",
                "angle",
                "TX",
                "TY",
                "TZ",
                "FX_0",
                "FY_0",
                "FZ",
                "flow_speed",
            )
            if not all(np.isfinite(float(row[column])) for column in numeric_columns):
                raise ValueError(f"model_{model_id}/ang{angle} 产生了非有限数值。")
            rows.append(row)
    rows.sort(key=lambda row: (int(row["model_id"]), int(row["angle"]), float(row["flow_speed"])))
    if not rows:
        raise ValueError("没有可写入的有效 sample。")
    if strict:
        _validate_strict_baseline(rows)

    # 禁止覆盖时必须在写第一个文件之前同时检查两个目标，避免只写出其中一份。
    if not overwrite:
        existing_outputs = [
            path for path in (output_file, dataset_file) if path.exists()
        ]
        if existing_outputs:
            raise FileExistsError(
                "输出已存在；使用 --overwrite 可覆盖："
                + "、".join(str(path) for path in existing_outputs)
            )

    # 在落盘之前构建并完整验证 JSONL 记录。这样任何逐株坐标或排布错误都会在
    # 两个正式文件被替换前暴露，不会写出一份新 CSV 配一份旧 JSONL。
    dataset_records = build_dataset_records(rows)
    if len(dataset_records) != len(rows):
        raise AssertionError("JSONL sample 数量与 summarized CSV 行数不一致。")
    write_summarized_csv(rows, output_file, overwrite)
    write_dataset_jsonl(dataset_records, dataset_file, overwrite)
    return rows


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器，默认值均来自代码开头的常量。"""

    parser = argparse.ArgumentParser(description="将 sensor CSV 汇总为 CSV 和 JSONL。")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="包含 model_N 子目录的输入根目录。",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_SUMMARIZED_PATH,
        help="sample 级 summarized_data.csv 输出路径。",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="带逐株 x、y、angle 的 dataset.jsonl 输出路径。",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_LAYOUT_PATH,
        help="Experiment/input.csv 构型路径。",
    )
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=DEFAULT_TRIM_FRACTION,
        help="每端截尾比例，默认 0.10。",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERWRITE_EXISTING,
        help="是否允许覆盖两个输出文件。",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_STRICT_VALIDATION,
        help="是否验证当前 332 行正式实验基线。",
    )
    return parser


def main() -> int:
    """解析参数、执行汇总，并返回适合命令行的状态码。"""

    arguments = build_parser().parse_args()
    try:
        rows = summarize_sensor_tree(
            arguments.input_path.expanduser().resolve(),
            arguments.output_path.expanduser().resolve(),
            arguments.dataset_path.expanduser().resolve(),
            arguments.input_csv.expanduser().resolve(),
            arguments.trim_fraction,
            arguments.overwrite,
            arguments.strict,
        )
    except Exception as error:
        print(f"[失败] {error}")
        return 2
    print(f"处理结束：生成 {len(rows)} 个 sample。")
    print(f"汇总 CSV：{arguments.output_path.expanduser().resolve()}")
    print(f"模型 JSONL：{arguments.dataset_path.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
