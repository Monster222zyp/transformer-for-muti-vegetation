"""将滤波后的传感器序列整理为实验汇总 CSV 和模型数据 JSONL。

每个输出行代表一个 ``model_id × angle × flow_speed`` 实验 sample。脚本先对
六轴传感器序列进行逐列截尾平均，再把 ``FX/FY`` 旋转到 0° 坐标系，最后补齐
模型编号和旋转后的 37 位水草排布。input.csv 的正整数是材料类别，必须通过
YAML 映射为实测抗弯刚度 EI（单位 N·m²），不会把类别编号当作物理量。
新批次先逐轮计算测量值，再按主阻力重复性筛选并平均；schema 3 保存每株
真实坐标、完整角度和 stiffness，二进制 vegetation_layout 只说明位置是否占用。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import yaml

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
DEFAULT_OUTPUT_DIRECTORY = SCRIPT_DIR / "output" / "rebuilt_all"
DEFAULT_SUMMARIZED_PATH = DEFAULT_OUTPUT_DIRECTORY / "summarized_data.csv"
DEFAULT_DATASET_PATH = DEFAULT_OUTPUT_DIRECTORY / "所有.jsonl"
DEFAULT_SIX_ANGLE_PATH = DEFAULT_OUTPUT_DIRECTORY / "所有60度的.jsonl"
DEFAULT_NON_SIX_ANGLE_PATH = DEFAULT_OUTPUT_DIRECTORY / "新的非60度的.jsonl"
DEFAULT_REPEATABILITY_PATH = DEFAULT_OUTPUT_DIRECTORY / "repeatability_new15_17.csv"
DEFAULT_LAYOUT_PATH = SCRIPT_DIR / "input.csv"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "stiffness_config.yaml"
DEFAULT_TRIM_FRACTION = 0.10
DEFAULT_OVERWRITE_EXISTING = False
DEFAULT_STRICT_VALIDATION = True
DEFAULT_SOURCE_MODE = "full17"
LEGACY_SOURCE_MODE = "legacy14"
DEFAULT_STIFFNESS_BY_CODE = {1: 0.5, 2: 1.0, 3: 2.0}
DEFAULT_EXCLUDED_FLOW_SPEEDS = (0.1,)
DEFAULT_REPEATABILITY_THRESHOLD_PERCENT = 15.0
DATASET_SCHEMA_VERSION = 3
LAYOUT_POINT_COUNT = 37
ANGLE_CYCLE_DEGREES = 360
ANCHOR_STEP_DEGREES = 60
NEW_ANGLE_STEP_DEGREES = 15
PERCENT_SCALE = 100.0
REPEAT_ROUNDS = (1, 2)

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
NEW_VALID_ANGLES = tuple(range(0, ANGLE_CYCLE_DEGREES, NEW_ANGLE_STEP_DEGREES))
EXPECTED_MODEL_IDS = tuple(range(1, 15))
NEW_MODEL_IDS = (15, 16, 17)
EXPECTED_MISSING_CONDITIONS = ((3, 60, 0.2), (3, 240, 0.3), (9, 240, 0.1), (13, 240, 0.3))
MEASUREMENT_COLUMNS = ("TX", "TY", "TZ", "FX_0", "FY_0", "FZ")
REPEATABILITY_COLUMNS = (
    "model_id", "plant_count", "angle_deg", "flow_speed_m_s",
    "round1_FX_0_N", "round2_FX_0_N", "absolute_difference_N",
    "average_absolute_force_N", "difference_ratio_percent", "retained",
)
MODEL_DIRECTORY_PATTERN = re.compile(r"^model_(?P<model_id>\d+)$", re.IGNORECASE)
SENSOR_FILE_RE = re.compile(
    r"^sensor_(?P<sensor>\d+)(?:_round(?P<round>\d+))?_ang(?P<angle>-?\d+(?:\.\d+)?)"
    r"(?:_(?:lowpass|bandpass)_filtered)?\.csv$",
    re.IGNORECASE,
)


def parse_sensor_filename(path: Path) -> tuple[int, float] | None:
    """提取流速编号与角度；兼容 round 文件且保持旧两元素返回值。"""

    match = SENSOR_FILE_RE.fullmatch(path.name)
    return None if match is None else (int(match.group("sensor")), float(match.group("angle")))


def load_stiffness_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    """读取并验证建集策略，返回规范化后的字典。

    参数 config_path 是 YAML 文件路径；stiffness_by_code 的键是任意正整数类别，
    值是正且有限的实测 EI。excluded_flow_speeds 是排除的流速列表（m/s），
    repeatability_threshold_percent 是允许的两轮对称差率上限（百分数）。
    """
    # safe_load 只读取 YAML 数据，不允许配置文件执行 Python 对象构造。
    with Path(config_path).open("r", encoding="utf-8-sig") as handle:
        configuration = yaml.safe_load(handle)
    if not isinstance(configuration, dict):
        raise ValueError("刚度配置必须是 YAML 字典。")
    raw_mapping = configuration.get("stiffness_by_code")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError("stiffness_by_code 必须是非空的类别到实测 EI 映射。")

    # 显式拒绝 bool 和小数类别，防止 True 被 Python 当作类别 1。
    stiffness_by_code: dict[int, float] = {}
    for code, raw_stiffness in raw_mapping.items():
        if isinstance(code, bool) or not isinstance(code, int) or code <= 0:
            raise ValueError(f"stiffness_by_code 的类别必须是正整数：{code!r}。")
        if isinstance(raw_stiffness, bool) or not isinstance(raw_stiffness, (int, float)):
            raise ValueError(f"类别 {code} 的实测 EI 必须是正有限数值。")
        stiffness = float(raw_stiffness)
        if not np.isfinite(stiffness) or stiffness <= 0:
            raise ValueError(f"类别 {code} 的实测 EI 必须是正有限数值。")
        stiffness_by_code[code] = stiffness

    # 检查策略值后统一转为 float，避免后续字符串或 NaN 比较产生静默行为。
    raw_excluded = configuration.get("excluded_flow_speeds", list(DEFAULT_EXCLUDED_FLOW_SPEEDS))
    if not isinstance(raw_excluded, (list, tuple)):
        raise ValueError("excluded_flow_speeds 必须是流速列表。")
    excluded = []
    for speed in raw_excluded:
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not np.isfinite(speed) or speed <= 0:
            raise ValueError("excluded_flow_speeds 必须只含正有限流速。")
        excluded.append(float(speed))
    threshold = configuration.get("repeatability_threshold_percent", DEFAULT_REPEATABILITY_THRESHOLD_PERCENT)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("repeatability_threshold_percent 必须是非负有限百分数。")
    return {
        "stiffness_by_code": stiffness_by_code,
        "excluded_flow_speeds": tuple(excluded),
        "repeatability_threshold_percent": float(threshold),
    }


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


def _parse_layout(
    row: Sequence[str], row_number: int, stiffness_by_code: dict[int, float] | None = None
) -> list[int]:
    """验证一行 37 列非负整数类别；row_number 用于指出错误行。

    stiffness_by_code 为类别对应的实测 EI；省略时使用代码开头的默认映射。
    返回完整类别列表，保留类别与位置的关联，0 表示该位置没有植株。
    """

    try:
        layout = [int(cell.strip()) for cell in row]
    except ValueError as error:
        raise ValueError(f"构型文件第 {row_number} 行包含非整数。") from error
    if len(layout) != LAYOUT_POINT_COUNT or any(value < 0 for value in layout):
        raise ValueError(f"构型文件第 {row_number} 行必须是 37 列非负整数类别。")
    mapping = DEFAULT_STIFFNESS_BY_CODE if stiffness_by_code is None else stiffness_by_code
    unknown_codes = sorted(set(layout) - {0} - set(mapping))
    if unknown_codes:
        raise ValueError(f"构型文件第 {row_number} 行有未配置的刚度类别：{unknown_codes}。")
    if not any(layout):
        raise ValueError(f"构型文件第 {row_number} 行没有水草。")
    return layout


def load_rotated_classifications(
    input_csv: Path, stiffness_by_code: dict[int, float] | None = None
) -> dict[tuple[int, int], tuple[int, ...]]:
    """返回 ``(model_id, 六角锚点) -> 37 个材料类别`` 的映射。

    参数 input_csv 可以位于项目外；stiffness_by_code 说明允许的类别及 EI。
    generator 从脚本目录加载，因此外部 CSV 不需要携带 generator.py。
    """

    if not input_csv.is_file():
        raise FileNotFoundError(f"找不到构型文件：{input_csv}")
    base_layouts: list[list[int]] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if row and any(cell.strip() for cell in row):
                base_layouts.append(_parse_layout(row, row_number, stiffness_by_code))
    if not base_layouts:
        raise ValueError(f"构型文件没有有效行：{input_csv}")
    rotation_function = _load_rotation_function(SCRIPT_DIR)
    rotated_layouts: dict[tuple[int, int], tuple[int, ...]] = {}
    for model_id, base_layout in enumerate(base_layouts, start=1):
        rotations = rotation_function(base_layout)
        if len(rotations) != len(VALID_ANGLES):
            raise ValueError(f"model_{model_id} 的旋转函数未返回六个方向。")
        for angle, layout in zip(VALID_ANGLES, rotations):
            # get_rotated_groups 只移动数值，不把类别压成 01，因此 EI 随植株旋转。
            parsed_layout = _parse_layout([str(value) for value in layout], model_id, stiffness_by_code)
            rotated_layouts[(model_id, angle)] = tuple(parsed_layout)
    return rotated_layouts


def load_rotated_layouts(
    input_csv: Path, stiffness_by_code: dict[int, float] | None = None
) -> dict[tuple[int, int], str]:
    """保持旧公开接口：将旋转后的正类别压成 1，返回 37 位二进制排布。

    参数 input_csv 与 stiffness_by_code 同 load_rotated_classifications；类别
    信息仅在这份兼容视图中压缩，建集过程使用完整分类生成逐株 stiffness。
    """
    classifications = load_rotated_classifications(input_csv, stiffness_by_code)
    return {key: "".join("1" if code else "0" for code in values) for key, values in classifications.items()}


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
    *,
    rotated_classifications: dict[tuple[int, int], tuple[int, ...]] | None = None,
    stiffness_by_code: dict[int, float] | None = None,
) -> list[dict[str, object]]:
    """把 sample 汇总行转换为带逐株坐标和角度的 JSONL 记录。

    参数：
        rows: 已经完成排序和实验完整性校验的汇总行。
        rotated_classifications: 每个 model 的六角锚点类别，逐株 EI 由此定位。
            省略时，调用者的纯 01 排布明确按类别 0/1 解释；仍通过类别 1
            查表得到 EI，不读取其它构型表。包含多类材料的主建集链始终传入此映射。
        stiffness_by_code: 材料类别到实测 EI（N·m²）的对应关系。

    返回值：
        与 ``rows`` 一一对应的字典列表；每个字典的 ``plants`` 是变长数组，
        每一项保存 ``plant_index、grid_index、x、y、angle、stiffness``。
    """

    all_coordinates = build_hex_coordinates()
    mapping = DEFAULT_STIFFNESS_BY_CODE if stiffness_by_code is None else stiffness_by_code
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
        # 非六角工况沿用较小的六角排布作为网格锚点，再旋转剩余角度。
        anchor_angle = angle - angle % ANCHOR_STEP_DEGREES
        codes = (
            rotated_classifications[(model_id, anchor_angle)]
            if rotated_classifications is not None else parsed_layout
        )
        if tuple(int(code > 0) for code in codes) != parsed_layout:
            raise ValueError(f"sample {sample_id} 的材料类别与二进制排布不一致。")

        # grid_index 使用完整 37 点网格中的固定编号；plant_index 只在当前 sample
        # 内从 0 连续编号。两者同时保存可以兼顾模型顺序和实验位置追溯。
        plants: list[dict[str, object]] = []
        for grid_index, occupied in enumerate(parsed_layout):
            if not occupied:
                continue
            x_coordinate, y_coordinate = all_coordinates[grid_index]
            if angle % ANCHOR_STEP_DEGREES:
                x_coordinate, y_coordinate = rotate_force_xy_to_zero_frame(
                    x_coordinate, y_coordinate, angle % ANCHOR_STEP_DEGREES
                )
            stiffness = mapping.get(codes[grid_index])
            if stiffness is None or not np.isfinite(stiffness) or stiffness <= 0:
                raise ValueError(f"sample {sample_id} 的类别 {codes[grid_index]} 没有正有限 EI。")
            plants.append(
                {
                    "plant_index": len(plants),
                    "grid_index": grid_index,
                    "x": float(x_coordinate),
                    "y": float(y_coordinate),
                    # 当前实验中，同一 sample 内所有水草都采用该 sample 的原始角度。
                    "angle": angle,
                    "stiffness": float(stiffness),
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
                    for column in MEASUREMENT_COLUMNS
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


def calculate_repeatability(first_force: float, second_force: float) -> tuple[float, float, float]:
    """返回绝对差、平均绝对力、对称差率（%）；参数是两轮 FX_0（N）。"""
    # 分母使用两轮绝对值的平均，交换轮次不会影响结果；两轮都为零时定义为 0%。
    difference = abs(first_force - second_force)
    average_absolute_force = (abs(first_force) + abs(second_force)) / len(REPEAT_ROUNDS)
    ratio = PERCENT_SCALE * difference / average_absolute_force if average_absolute_force else 0.0
    return difference, average_absolute_force, ratio


def _validate_source_baseline(
    actual: set[tuple[int, int, float, int]], source_mode: str, excluded_flow_speeds: Sequence[float]
) -> None:
    """严格验证原始工况：actual 为模型、角度、流速、轮次；旧单轮记作 0。

    source_mode 选择 legacy14 或 full17；excluded_flow_speeds 中的工况不参加检查。
    旧四个已知缺失允许保留，文件行数没有硬编码下限，短 CSV 也合法。
    """
    # 旧批次每个工况只有一个文件；允许的历史缺失从期望集合中删除。
    expected = {
        (model, angle, speed, 0)
        for model in EXPECTED_MODEL_IDS
        for angle in VALID_ANGLES
        for speed in FLOW_SPEED_BY_SENSOR.values()
        if speed not in excluded_flow_speeds and (model, angle, speed) not in EXPECTED_MISSING_CONDITIONS
    }
    if source_mode == DEFAULT_SOURCE_MODE:
        expected.update(
            (model, angle, FLOW_SPEED_BY_SENSOR[sensor], round_number)
            for model in NEW_MODEL_IDS
            for angle in NEW_VALID_ANGLES
            for sensor in FLOW_SPEED_BY_SENSOR if sensor != 1
            for round_number in REPEAT_ROUNDS
            if FLOW_SPEED_BY_SENSOR[sensor] not in excluded_flow_speeds
        )
    if actual != expected:
        raise ValueError(
            f"严格源数据验证失败：应有 {len(expected)} 份，实际 {len(actual)} 份；"
            f"缺失={sorted(expected - actual)}，额外={sorted(actual - expected)}。"
            "可用 --no-strict 处理非正式批次。"
        )


def write_repeatability_csv(rows: Sequence[dict[str, object]], output_file: Path, overwrite: bool) -> None:
    """原子写入重复性审计 CSV；参数 rows 含所有两轮工况，包括被删除的整对。"""
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"输出已存在；使用 --overwrite 可覆盖：{output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPEATABILITY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_file.replace(output_file)


def summarize_sensor_tree(
    input_root: Path,
    output_file: Path,
    dataset_file: Path,
    input_csv: Path,
    fraction: float,
    overwrite: bool,
    strict: bool,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    source_mode: str = LEGACY_SOURCE_MODE,
    excluded_flow_speeds: Sequence[float] | None = None,
    repeatability_threshold_percent: float | None = None,
    six_angle_file: Path | None = None,
    non_six_angle_file: Path | None = None,
    repeatability_file: Path | None = None,
) -> list[dict[str, object]]:
    """从滤波 CSV 重建测量、几何和逐株 EI，并返回已排序的汇总行。

    参数：input_root 是 model_N 目录的父目录；output_file 与 dataset_file
    是汇总 CSV 和全量 JSONL；input_csv 是 37 列类别表；fraction 是每端截尾
    比例；overwrite 显式允许替换输出；strict 验证正式源工况是否完整。
    config_path 是材料与筛选 YAML。旧七个位置参数调用继续默认 legacy14，
    命令行默认 full17。excluded_flow_speeds 和 repeatability_threshold_percent
    可覆盖 YAML 策略；legacy14 默认保留全部流速以兼容 332 条历史基线。
    三个额外输出路径只用于 full17，省略时在 dataset_file 同目录生成。
    """
    if not np.isfinite(fraction) or not 0 <= fraction < 0.5:
        raise ValueError("trim_fraction 必须满足 0 <= fraction < 0.5。")
    if source_mode not in (LEGACY_SOURCE_MODE, DEFAULT_SOURCE_MODE):
        raise ValueError("source_mode 只允许 legacy14 或 full17。")

    # 首先验证全部类别和 EI。即使某行没有对应实验文件，也不能隐藏非法材料。
    configuration = load_stiffness_config(config_path)
    stiffness_by_code = configuration["stiffness_by_code"]
    assert isinstance(stiffness_by_code, dict)
    classifications = load_rotated_classifications(input_csv.resolve(), stiffness_by_code)
    if excluded_flow_speeds is None:
        excluded_flow_speeds = () if source_mode == LEGACY_SOURCE_MODE else configuration["excluded_flow_speeds"]
    excluded = tuple(float(speed) for speed in excluded_flow_speeds)
    if any(not np.isfinite(speed) or speed <= 0 for speed in excluded):
        raise ValueError("排除流速必须是正有限数值。")
    threshold = float(configuration["repeatability_threshold_percent"] if repeatability_threshold_percent is None else repeatability_threshold_percent)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("重复性阈值必须是非负有限百分数。")

    # 所有目标先同时检查，避免发现第二个目标已存在时第一份结果已经被写入。
    output_paths = [output_file, dataset_file]
    if source_mode == DEFAULT_SOURCE_MODE:
        six_angle_file = six_angle_file or dataset_file.parent / DEFAULT_SIX_ANGLE_PATH.name
        non_six_angle_file = non_six_angle_file or dataset_file.parent / DEFAULT_NON_SIX_ANGLE_PATH.name
        repeatability_file = repeatability_file or dataset_file.parent / DEFAULT_REPEATABILITY_PATH.name
        output_paths.extend([six_angle_file, non_six_angle_file, repeatability_file])
    resolved_paths = [path.resolve() for path in output_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("CSV、JSONL、角度子集及重复性输出必须使用不同路径。")
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError("输出已存在；使用 --overwrite 可覆盖：" + "、".join(str(path) for path in existing_outputs))

    # 文件按完整工况分组；轮次键 0 表示旧单轮，1/2 表示新批次的两次独立实验。
    files = discover_sensor_files(input_root, output_file)
    if not files:
        raise FileNotFoundError(f"没有找到符合命名规则的 sensor CSV：{input_root}")
    grouped: dict[tuple[int, int, float], dict[int, np.ndarray]] = {}
    actual_sources: set[tuple[int, int, float, int]] = set()
    for source_file in files:
        model_id = _model_id_for_source(source_file, input_root)
        if source_mode == LEGACY_SOURCE_MODE and model_id not in EXPECTED_MODEL_IDS:
            continue
        match = SENSOR_FILE_RE.fullmatch(source_file.name)
        assert match is not None
        sensor_number, raw_angle = int(match.group("sensor")), float(match.group("angle"))
        round_number = int(match.group("round")) if match.group("round") is not None else 0
        is_new = model_id in NEW_MODEL_IDS
        valid_angles = NEW_VALID_ANGLES if is_new else VALID_ANGLES
        if model_id not in (*EXPECTED_MODEL_IDS, *NEW_MODEL_IDS):
            raise ValueError(f"不支持的模型编号：model_{model_id}。")
        if not raw_angle.is_integer() or int(raw_angle) not in valid_angles:
            raise ValueError(f"model_{model_id} 不支持角度 {raw_angle}；允许值为 {valid_angles}。")
        if sensor_number not in FLOW_SPEED_BY_SENSOR or (is_new and sensor_number == 1):
            raise ValueError(f"model_{model_id} 不支持 sensor_{sensor_number}。")
        if (is_new and round_number not in REPEAT_ROUNDS) or (not is_new and round_number != 0):
            raise ValueError(f"model_{model_id} 的轮次命名不合法：{source_file.name}。")
        angle, speed = int(raw_angle), FLOW_SPEED_BY_SENSOR[sensor_number]
        if speed in excluded:
            continue
        source_key = (model_id, angle, speed, round_number)
        if source_key in actual_sources:
            raise ValueError(f"重复的 sensor 工况：{source_key}。")
        anchor_angle = angle - angle % ANCHOR_STEP_DEGREES
        if (model_id, anchor_angle) not in classifications:
            raise ValueError(f"model_{model_id} 没有对应的构型行。")
        actual_sources.add(source_key)
        # 每轮逐列 floor(n×fraction) 两端截尾；仅 FX/FY 按完整实验角度旋转。
        means = rotate_mean_force(
            calculate_trimmed_column_means(read_numeric_sensor_csv(source_file), fraction), angle
        )
        if not np.isfinite(means).all():
            raise ValueError(f"输入 {source_file} 在截尾或旋转后产生非有限测量值。")
        grouped.setdefault((model_id, angle, speed), {})[round_number] = means
    if strict:
        _validate_source_baseline(actual_sources, source_mode, excluded)

    rows: list[dict[str, object]] = []
    repeatability_rows: list[dict[str, object]] = []
    for (model_id, angle, flow_speed), rounds in sorted(grouped.items()):
        anchor_angle = angle - angle % ANCHOR_STEP_DEGREES
        codes = classifications[(model_id, anchor_angle)]
        if model_id in NEW_MODEL_IDS:
            if set(rounds) != set(REPEAT_ROUNDS):
                raise ValueError(f"model_{model_id}/ang{angle}/{flow_speed} 必须同时存在 round1 与 round2。")
            first_force, second_force = float(rounds[1][3]), float(rounds[2][3])
            difference, average_force, ratio = calculate_repeatability(first_force, second_force)
            retained = ratio <= threshold
            repeatability_rows.append({
                "model_id": model_id, "plant_count": sum(code > 0 for code in codes),
                "angle_deg": angle, "flow_speed_m_s": flow_speed,
                "round1_FX_0_N": first_force, "round2_FX_0_N": second_force,
                "absolute_difference_N": difference, "average_absolute_force_N": average_force,
                "difference_ratio_percent": ratio, "retained": retained,
            })
            if not retained:
                continue
            # 合格整对在截尾及旋转之后，对全部六个字段进行等权平均。
            means = (rounds[1] + rounds[2]) / len(REPEAT_ROUNDS)
        else:
            means = rounds[0]
        row = {
            "sample_id": build_sample_id(model_id, angle, flow_speed),
            "model_id": model_id, "angle": angle, "flow_speed": flow_speed,
            "vegetation_layout": "".join("1" if code else "0" for code in codes),
            **{column: float(value) for column, value in zip(MEASUREMENT_COLUMNS, means)},
        }
        if not all(np.isfinite(float(row[column])) for column in MEASUREMENT_COLUMNS):
            raise ValueError(f"model_{model_id}/ang{angle} 产生了非有限测量值。")
        rows.append(row)
    if not rows:
        raise ValueError("没有可写入的有效 sample。")

    # 落盘前构建全部 schema 3 记录；索引、真实坐标和材料对应关系先完整验证。
    records = build_dataset_records(rows, rotated_classifications=classifications, stiffness_by_code=stiffness_by_code)
    write_summarized_csv(rows, output_file, overwrite)
    write_dataset_jsonl(records, dataset_file, overwrite)
    if source_mode == DEFAULT_SOURCE_MODE:
        assert six_angle_file is not None and non_six_angle_file is not None and repeatability_file is not None
        write_dataset_jsonl([record for record in records if int(record["angle"]) % ANCHOR_STEP_DEGREES == 0], six_angle_file, overwrite)
        write_dataset_jsonl([record for record in records if int(record["angle"]) % ANCHOR_STEP_DEGREES != 0], non_six_angle_file, overwrite)
        write_repeatability_csv(repeatability_rows, repeatability_file, overwrite)
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
        help="带逐株 x、y、angle、stiffness 的 schema 3 全量 JSONL 输出路径。",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_LAYOUT_PATH,
        help="37 列数字类别表路径；第 N 行对应 model_N。",
    )
    # 配置与新增策略只使用带名称的参数，保持原七个位置参数函数调用兼容。
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="实测 EI 映射与筛选策略 YAML。")
    parser.add_argument("--source-mode", choices=(DEFAULT_SOURCE_MODE, LEGACY_SOURCE_MODE), default=DEFAULT_SOURCE_MODE, help="full17 重建全部批次；legacy14 仅重建旧 332 条基线。")
    parser.add_argument("--six-angle-path", type=Path, default=DEFAULT_SIX_ANGLE_PATH, help="60° 整数倍角度的 JSONL 输出路径。")
    parser.add_argument("--non-six-angle-path", type=Path, default=DEFAULT_NON_SIX_ANGLE_PATH, help="非 60° 整数倍角度的 JSONL 输出路径。")
    parser.add_argument("--repeatability-path", type=Path, default=DEFAULT_REPEATABILITY_PATH, help="全部新批次两轮工况的重复性 CSV 输出路径。")
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
        help="显式允许覆盖输出文件；默认拒绝覆盖。",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_STRICT_VALIDATION,
        help="验证正式源工况完整性，允许旧四个已知缺失，不限制 CSV 行数。",
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
            config_path=arguments.config.expanduser().resolve(),
            source_mode=arguments.source_mode,
            six_angle_file=arguments.six_angle_path.expanduser().resolve(),
            non_six_angle_file=arguments.non_six_angle_path.expanduser().resolve(),
            repeatability_file=arguments.repeatability_path.expanduser().resolve(),
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
