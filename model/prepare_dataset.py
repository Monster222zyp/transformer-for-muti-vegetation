"""把按构型和角度保存的实验汇总数据合并为单个建模 CSV。"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence


# 默认路径统一由当前脚本反推项目根目录，避免命令从不同工作目录启动时失效。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "summarized_data"
DEFAULT_INPUT_CSV = PROJECT_ROOT / "Experiment" / "input.csv"
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parent / "data" / "all_models.csv"
DEFAULT_AUDIT_JSON = Path(__file__).resolve().parent / "data" / "all_models.audit.json"

# 汇总数据前七列的顺序是实验处理链约定，最后一列由本脚本追加。
SOURCE_COLUMNS = ("TX", "TY", "TZ", "FX_0", "FY_0", "FZ", "flow_speed")
OUTPUT_COLUMNS = (*SOURCE_COLUMNS, "vegetation_layout")
VALID_ANGLES = (0, 60, 120, 180, 240, 300)
VALID_FLOW_SPEEDS = (0.1, 0.2, 0.3, 0.4)
EXPECTED_ROW_COUNT = 332
EXPECTED_MODEL_IDS = tuple(range(1, 15))
EXPECTED_MISSING_CONDITIONS = (
    (3, 60, 0.2),
    (3, 240, 0.3),
    (9, 240, 0.1),
    (13, 240, 0.3),
)

# 文件夹和文件名必须严格匹配，避免把备份文件或临时文件悄悄混入数据集。
MODEL_DIRECTORY_PATTERN = re.compile(r"model_(\d+)$")
ANGLE_FILE_PATTERN = re.compile(r"ang(\d+)\.csv$")


def _load_rotation_function(experiment_dir: Path) -> Callable[[list[int]], list[list[int]]]:
    """从指定 Experiment 目录加载项目已有的 ``get_rotated_groups`` 函数。

    参数：
        experiment_dir: 包含 ``generator.py`` 的 Experiment 目录。

    返回：
        generator.py 中定义的六方向旋转函数，确保建模数据与实验工具使用同一规则。
    """
    generator_path = experiment_dir / "generator.py"
    if not generator_path.is_file():
        raise FileNotFoundError(f"找不到旋转函数文件：{generator_path}")

    # 使用文件路径加载而不是依赖当前工作目录，使命令行入口在任意目录均可运行。
    module_spec = importlib.util.spec_from_file_location("experiment_generator", generator_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"无法创建 generator.py 的导入描述：{generator_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    rotation_function = getattr(module, "get_rotated_groups", None)
    if not callable(rotation_function):
        raise AttributeError(f"{generator_path} 未提供可调用的 get_rotated_groups。")
    return rotation_function


def load_base_layouts(input_csv: Path) -> list[list[int]]:
    """读取 ``Experiment/input.csv``，并验证每行是 37 位 0/1 构型。

    参数：
        input_csv: 无表头构型 CSV；第 N 行严格对应 ``model_N``。

    返回：
        按文件行顺序排列的构型列表。
    """
    if not input_csv.is_file():
        raise FileNotFoundError(f"找不到构型文件：{input_csv}")

    layouts: list[list[int]] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row_number, row in enumerate(csv.reader(csv_file), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 37:
                raise ValueError(f"构型文件第 {row_number} 行应为 37 列，实际为 {len(row)} 列。")
            try:
                layout = [int(cell.strip()) for cell in row]
            except ValueError as error:
                raise ValueError(f"构型文件第 {row_number} 行包含非整数。") from error
            if any(value not in (0, 1) for value in layout):
                raise ValueError(f"构型文件第 {row_number} 行包含 0/1 以外的值。")
            layouts.append(layout)

    if not layouts:
        raise ValueError(f"构型文件没有有效数据：{input_csv}")
    return layouts


def encode_layout(layout: Sequence[int]) -> str:
    """把 37 个整数编码成不带分隔符的固定长度 01 字符串。"""
    if len(layout) != 37 or any(value not in (0, 1) for value in layout):
        raise ValueError("水草构型必须恰好包含 37 个 0/1 值。")
    return "".join(str(value) for value in layout)


def _numeric_source_paths(source_root: Path) -> list[tuple[int, int, Path]]:
    """发现源文件并按 ``model_id``、角度进行数值排序。"""
    if not source_root.is_dir():
        raise FileNotFoundError(f"找不到汇总数据目录：{source_root}")

    discovered: list[tuple[int, int, Path]] = []
    for model_dir in source_root.iterdir():
        model_match = MODEL_DIRECTORY_PATTERN.fullmatch(model_dir.name)
        if not model_dir.is_dir() or model_match is None:
            continue
        model_id = int(model_match.group(1))
        for angle_file in model_dir.iterdir():
            angle_match = ANGLE_FILE_PATTERN.fullmatch(angle_file.name)
            if not angle_file.is_file() or angle_match is None:
                continue
            discovered.append((model_id, int(angle_match.group(1)), angle_file))

    discovered.sort(key=lambda item: (item[0], item[1]))
    if not discovered:
        raise ValueError(f"在 {source_root} 中没有找到 model_N/angB.csv。")
    return discovered


def _read_source_rows(source_file: Path) -> list[list[str]]:
    """读取一个角度文件，并严格验证七列数值和允许的流速。"""
    rows: list[list[str]] = []
    seen_speeds: set[float] = set()
    with source_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row_number, row in enumerate(csv.reader(csv_file), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(SOURCE_COLUMNS):
                raise ValueError(
                    f"{source_file} 第 {row_number} 行应为 7 列，实际为 {len(row)} 列。"
                )
            cleaned = [cell.strip() for cell in row]
            try:
                numeric_values = [float(cell) for cell in cleaned]
            except ValueError as error:
                raise ValueError(f"{source_file} 第 {row_number} 行包含非数值内容。") from error
            if any(value != value or abs(value) == float("inf") for value in numeric_values):
                raise ValueError(f"{source_file} 第 {row_number} 行包含 NaN 或无穷大。")

            flow_speed = numeric_values[-1]
            matched_speed = next(
                (allowed for allowed in VALID_FLOW_SPEEDS if abs(flow_speed - allowed) < 1e-9),
                None,
            )
            if matched_speed is None:
                raise ValueError(f"{source_file} 第 {row_number} 行流速 {flow_speed} 不在允许范围。")
            if matched_speed in seen_speeds:
                raise ValueError(f"{source_file} 中流速 {matched_speed} 重复。")
            seen_speeds.add(matched_speed)
            rows.append(cleaned)

    if not rows:
        raise ValueError(f"角度文件没有有效数据：{source_file}")
    rows.sort(key=lambda row: float(row[-1]))
    return rows


def prepare_dataset(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    audit_json: Path = DEFAULT_AUDIT_JSON,
    strict: bool = True,
) -> dict[str, object]:
    """生成总 CSV 和审计 JSON，并返回审计内容。

    参数：
        source_root: ``summarized_data`` 根目录。
        input_csv: 实验构型文件，第 N 行对应 ``model_N``。
        output_csv: 带表头的八列总数据集输出路径。
        audit_json: 记录覆盖范围、缺测工况和校验结果的 JSON 路径。
        strict: 为 ``True`` 时要求当前数据精确符合 14 个模型和 332 行基线。

    返回：
        同步写入 JSON 的审计字典。
    """
    source_root = Path(source_root).resolve()
    input_csv = Path(input_csv).resolve()
    output_csv = Path(output_csv).resolve()
    audit_json = Path(audit_json).resolve()

    base_layouts = load_base_layouts(input_csv)
    get_rotated_groups = _load_rotation_function(input_csv.parent)
    source_paths = _numeric_source_paths(source_root)

    output_rows: list[list[str]] = []
    observed_conditions: set[tuple[int, int, float]] = set()
    source_summaries: list[dict[str, object]] = []
    model_counter: Counter[int] = Counter()
    angle_counter: Counter[int] = Counter()
    speed_counter: Counter[str] = Counter()

    for model_id, angle, source_file in source_paths:
        if model_id > len(base_layouts):
            raise ValueError(f"{source_file} 对应 model_{model_id}，但 input.csv 只有 {len(base_layouts)} 行。")
        if angle not in VALID_ANGLES:
            raise ValueError(f"{source_file} 的角度 {angle} 不是 60 度的允许角度。")

        # 第 N 个模型使用 input.csv 第 N 行，旋转列表下标与角度除以 60 一一对应。
        rotated_layouts = get_rotated_groups(base_layouts[model_id - 1])
        layout = rotated_layouts[angle // 60]
        if sum(layout) != sum(base_layouts[model_id - 1]):
            raise AssertionError(f"model_{model_id} 在 {angle}° 旋转后植株数量发生变化。")
        layout_code = encode_layout(layout)

        source_rows = _read_source_rows(source_file)
        source_summaries.append(
            {
                "model_id": model_id,
                "angle": angle,
                "source_file": source_file.relative_to(source_root).as_posix(),
                "row_count": len(source_rows),
                "flow_speeds": [float(row[-1]) for row in source_rows],
            }
        )
        for source_row in source_rows:
            speed = float(source_row[-1])
            condition = (model_id, angle, speed)
            if condition in observed_conditions:
                raise ValueError(f"发现重复工况：model_{model_id}, {angle}°, {speed} m/s。")
            observed_conditions.add(condition)
            output_rows.append([*source_row, layout_code])
            model_counter[model_id] += 1
            angle_counter[angle] += 1
            speed_counter[f"{speed:.1f}"] += 1

    observed_model_ids = tuple(sorted(model_counter))
    missing_conditions = sorted(
        (model_id, angle, speed)
        for model_id in observed_model_ids
        for angle in VALID_ANGLES
        for speed in VALID_FLOW_SPEEDS
        if (model_id, angle, speed) not in observed_conditions
    )

    if strict:
        if observed_model_ids != EXPECTED_MODEL_IDS:
            raise ValueError(f"严格模式要求 model_1～model_14，实际为 {observed_model_ids}。")
        if len(source_paths) != len(EXPECTED_MODEL_IDS) * len(VALID_ANGLES):
            raise ValueError(f"严格模式要求 84 个角度文件，实际为 {len(source_paths)} 个。")
        if len(output_rows) != EXPECTED_ROW_COUNT:
            raise ValueError(f"严格模式要求 {EXPECTED_ROW_COUNT} 行，实际为 {len(output_rows)} 行。")
        if tuple(missing_conditions) != EXPECTED_MISSING_CONDITIONS:
            raise ValueError(f"缺测工况与已知记录不一致：{missing_conditions}")

    # 所有校验通过后才创建和覆盖输出，避免错误运行留下半成品。
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(output_rows)

    audit: dict[str, object] = {
        "source_root": str(source_root),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "columns": list(OUTPUT_COLUMNS),
        "row_count": len(output_rows),
        "source_file_count": len(source_paths),
        "model_ids": list(observed_model_ids),
        "rows_by_model": {str(key): model_counter[key] for key in sorted(model_counter)},
        "rows_by_angle": {str(key): angle_counter[key] for key in sorted(angle_counter)},
        "rows_by_flow_speed": {key: speed_counter[key] for key in sorted(speed_counter)},
        "missing_conditions": [
            {"model_id": model_id, "angle": angle, "flow_speed": speed}
            for model_id, angle, speed in missing_conditions
        ],
        "source_files": source_summaries,
        "checks": {
            "strict_mode": strict,
            "all_rows_have_eight_columns": all(len(row) == 8 for row in output_rows),
            "all_layouts_are_37_bit_binary": all(
                len(row[-1]) == 37 and set(row[-1]) <= {"0", "1"} for row in output_rows
            ),
            "rotation_preserves_plant_count": True,
        },
    }
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    with audit_json.open("w", encoding="utf-8", newline="\n") as json_file:
        json.dump(audit, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")
    return audit


def build_argument_parser() -> argparse.ArgumentParser:
    """创建数据准备命令的参数解析器。"""
    parser = argparse.ArgumentParser(description="合并多水草实验汇总数据并生成审计报告。")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="summarized_data 目录。")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Experiment/input.csv 路径。")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="总 CSV 输出路径。")
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON, help="审计 JSON 输出路径。")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="允许未来新增模型或缺测模式变化；默认严格验证当前 332 行基线。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """解析命令行参数，生成数据集并打印简短摘要。"""
    args = build_argument_parser().parse_args(argv)
    audit = prepare_dataset(
        source_root=args.source_root,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        audit_json=args.audit_json,
        strict=not args.no_strict,
    )
    print(f"已生成 {audit['row_count']} 行数据：{Path(args.output_csv).resolve()}")
    print(f"审计报告：{Path(args.audit_json).resolve()}")


if __name__ == "__main__":
    main()
