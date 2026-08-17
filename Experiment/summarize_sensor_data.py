"""把滤波后的 sensor CSV 汇总为按 model 和角度组织的训练数据。

处理流程如下：
1. 递归寻找 ``sensor_A_angB`` 原始文件或滤波结果文件。
2. 对六个数据列分别排序，各去掉最小 10% 和最大 10% 的样本。
3. 对每一列剩余的 80% 样本求平均。
4. 调用 rotate_force.py，把平均后的 FX、FY 转换到 0° 坐标系。
5. 在同一个 model 内按角度合并 sensor 1～4；某个流速缺失时输出错误提示，
   但仍用其余流速生成 ``angB.csv``。

单个输入文件或单个输出文件出错时，脚本会打印失败原因并继续处理其他文件。

输出 CSV 没有表头，每行七列依次为：
TX、TY、TZ、FX_0、FY_0、FZ、flow_speed。
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from rotate_force import rotate_force_xy_to_zero_frame


# =============================================================================
# 用户可直接修改的默认配置；命令行参数可以临时覆盖这些值。
# =============================================================================

# SCRIPT_DIR 是当前脚本所在的 Experiment 文件夹。
SCRIPT_DIR = Path(__file__).resolve().parent

# REPO_ROOT 是仓库根目录；滤波数据和汇总数据统一存放在仓库根目录。
REPO_ROOT = SCRIPT_DIR.parent

# 默认读取 filter_sensor_data.py 生成的镜像目录。
inputpath = REPO_ROOT / "filtered_data"

# 输出继续保留 model_1、model_2 等相对目录。
outputpath = REPO_ROOT / "summarized_data"

# 每一列两端各丢弃 10%；最终参与平均的是中间 80% 的样本。
trim_fraction = 0.10

# sensor 编号 A 到流速的显式映射表，单位沿用项目原始数据的流速单位。
# 以后若某个编号对应的流速变化，只需修改这里，不必改动汇总逻辑。
FLOW_SPEED_BY_SENSOR: dict[int, float] = {
    1: 0.1,
    2: 0.2,
    3: 0.3,
    4: 0.4,
}

# 是否默认覆盖已经存在的 angB.csv。
overwrite_existing = True

SENSOR_COLUMNS = ("TX", "TY", "TZ", "FX", "FY", "FZ")

# 同时接受原始名称 sensor_1_ang60.csv 和过滤脚本生成的
# sensor_1_ang60_lowpass_filtered.csv / bandpass_filtered.csv。
SENSOR_FILE_RE = re.compile(
    r"^sensor_(?P<sensor>\d+)_ang(?P<angle>-?\d+(?:\.\d+)?)"
    r"(?:_(?:lowpass|bandpass)_filtered)?\.csv$",
    re.IGNORECASE,
)


def parse_sensor_filename(path: Path) -> tuple[int, float] | None:
    """解析文件名中的 sensor 编号 A 和角度 B。

    Args:
        path: 待检查的 CSV 路径；函数只读取文件名，不打开文件。

    Returns:
        匹配成功时返回 ``(sensor 编号, 角度)``；其他文件返回 ``None``。
    """

    match = SENSOR_FILE_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group("sensor")), float(match.group("angle"))


def discover_sensor_files(input_root: Path, output_root: Path) -> list[Path]:
    """递归寻找可汇总的 sensor CSV，并明确排除输出目录。

    Args:
        input_root: 单个 CSV，或需要递归扫描的根目录。
        output_root: 汇总结果的根目录；若它位于输入树中则不能被再次读取。

    Returns:
        按相对目录、角度、sensor 编号稳定排序的文件路径列表。
    """

    if not input_root.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_root}")

    if input_root.is_file():
        if parse_sensor_filename(input_root) is None:
            raise ValueError(f"文件名不符合 sensor_A_angB 规则：{input_root.name}")
        return [input_root]

    files: list[Path] = []
    for candidate in input_root.rglob("*.csv"):
        if candidate == output_root or output_root in candidate.parents:
            continue
        if candidate.is_file() and parse_sensor_filename(candidate) is not None:
            files.append(candidate)

    def sort_key(path: Path) -> tuple[str, float, int, str]:
        """生成便于人工核对的稳定排序键。"""

        parsed_name = parse_sensor_filename(path)
        assert parsed_name is not None  # files 中只保存已经通过正则校验的文件。
        sensor_number, angle = parsed_name
        relative_parent = path.parent.relative_to(input_root).as_posix().casefold()
        return relative_parent, angle, sensor_number, path.name.casefold()

    return sorted(files, key=sort_key)


def read_numeric_sensor_csv(path: Path) -> np.ndarray:
    """读取无表头六列 CSV，并确保所有单元格都是有限数值。

    Args:
        path: 一个 sensor CSV 文件。

    Returns:
        形状为 ``(样本数, 6)`` 的浮点 NumPy 数组。
    """

    frame = pd.read_csv(path, header=None, sep=",")
    if frame.empty:
        raise ValueError(f"文件为空：{path}")
    if frame.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError(
            f"{path} 有 {frame.shape[1]} 列；sensor CSV 必须恰好有 6 列"
        )

    # errors='coerce' 会把不能转换的文本变成 NaN，随后统一给出明确错误。
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric_frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"文件包含文本、NaN 或 Inf，无法计算截尾平均：{path}")
    return values


def calculate_trimmed_column_means(
    values: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """对每一列独立去掉两端样本，再计算列平均值。

    Args:
        values: 二维数值数组；行是样本，列是 TX～FZ。
        fraction: 每一端要去掉的比例，例如 0.10 表示最小和最大各去掉 10%。

    Returns:
        每一列的截尾平均值，一共六个浮点数。

    Notes:
        去掉的数量使用 ``floor(样本数 * fraction)``。例如 101 行数据时，
        每一端去掉 10 行，剩余 81 行。每列独立排序，因此某一列的极端值
        不会导致同一行在其他列也被删除。
    """

    if values.ndim != 2 or values.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError("截尾平均输入必须是具有 6 列的二维数组")
    if not 0 <= fraction < 0.5:
        raise ValueError("trim_fraction 必须满足 0 <= fraction < 0.5")

    sample_count = values.shape[0]
    removed_per_side = int(np.floor(sample_count * fraction))
    if sample_count - 2 * removed_per_side <= 0:
        raise ValueError(f"{sample_count} 个样本在截尾后没有剩余数据")

    sorted_values = np.sort(values, axis=0)
    end_index = sample_count - removed_per_side if removed_per_side else sample_count
    retained_values = sorted_values[removed_per_side:end_index, :]
    return retained_values.mean(axis=0)


def rotate_mean_force(means: np.ndarray, angle: float) -> np.ndarray:
    """调用 rotate_force.py，把六列平均值中的 FX、FY 转到 0° 坐标系。

    Args:
        means: 按 TX、TY、TZ、FX、FY、FZ 排列的六个平均值。
        angle: 文件名中的角度 B，单位为 degree（度）。

    Returns:
        新的六元素数组；只有 FX、FY 被替换为旋转后的 FX_0、FY_0。
    """

    rotated = means.astype(float, copy=True)
    rotated_fx, rotated_fy = rotate_force_xy_to_zero_frame(
        float(rotated[3]),
        float(rotated[4]),
        angle,
    )
    rotated[3] = rotated_fx
    rotated[4] = rotated_fy
    return rotated


def format_angle(angle: float) -> str:
    """把数值角度转换成简洁且稳定的文件名片段。"""

    if angle.is_integer():
        return str(int(angle))
    return format(angle, "g")


def write_group_csv(
    rows: list[list[float]],
    output_file: Path,
    overwrite: bool,
) -> None:
    """以无表头格式原子写入一个角度的四行汇总结果。

    Args:
        rows: 每个 sensor 一行的七列数值。
        output_file: 目标 ``angB.csv`` 路径。
        overwrite: 为 ``False`` 时，已有文件会触发错误而不是被覆盖。
    """

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"输出已存在；使用 --overwrite 可覆盖：{output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(
        temporary_file,
        index=False,
        header=False,
        float_format="%.10f",
    )
    temporary_file.replace(output_file)


def summarize_sensor_tree(
    input_root: Path,
    output_root: Path,
    fraction: float,
    overwrite: bool,
) -> list[Path]:
    """完成整棵目录树的截尾平均、坐标旋转、分组与写入。

    Args:
        input_root: 滤波结果或原始 sensor CSV 的输入根目录。
        output_root: angB.csv 的输出根目录。
        fraction: 每列两端各丢弃的样本比例。
        overwrite: 是否允许覆盖已有 angB.csv。

    Returns:
        本次成功写出的所有 CSV 路径。
    """

    # fraction 是整个批次共用的参数。这个参数非法时，所有文件都不可能被
    # 正确处理，因此在进入逐文件容错循环前统一拒绝，而不是重复打印错误。
    if not 0 <= fraction < 0.5:
        raise ValueError("trim_fraction 必须满足 0 <= fraction < 0.5")

    files = discover_sensor_files(input_root, output_root)
    if not files:
        raise FileNotFoundError(f"没有找到符合 sensor_A_angB 规则的 CSV：{input_root}")

    # 第一层键是相对于 inputpath 的 model 目录，第二层键是角度，第三层键
    # 是 sensor 编号。显式保存三层键可以可靠检查重复文件和缺失流速。
    grouped_rows: dict[Path, dict[float, dict[int, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for source_file in files:
        parsed_name = parse_sensor_filename(source_file)
        assert parsed_name is not None  # discover_sensor_files 已经完成校验。
        sensor_number, angle = parsed_name
        relative_parent = (
            Path()
            if input_root.is_file()
            else source_file.parent.relative_to(input_root)
        )

        # 在读取 CSV 之前先建立角度分组。这样即使本文件损坏，该角度仍会在
        # 输出阶段报告“缺少某个流速”，而不会悄悄消失。
        sensor_rows = grouped_rows[relative_parent][angle]

        # 每个源文件都有独立的异常边界：其中一个 CSV 格式错误、存在 NaN、
        # 缺少流速映射或发生重复时，只跳过这个文件，后面的文件继续处理。
        try:
            if sensor_number not in FLOW_SPEED_BY_SENSOR:
                raise KeyError(
                    f"sensor_{sensor_number} 没有流速映射；"
                    "请在 FLOW_SPEED_BY_SENSOR 中补充"
                )
            if sensor_number in sensor_rows:
                raise ValueError(
                    f"同一目录和角度存在重复 sensor_{sensor_number} 文件"
                )

            values = read_numeric_sensor_csv(source_file)
            means = calculate_trimmed_column_means(values, fraction)
            sensor_rows[sensor_number] = rotate_mean_force(means, angle)
            print(
                f"[成功] 输入 {source_file} -> "
                f"截尾平均完成（流速 {FLOW_SPEED_BY_SENSOR[sensor_number]:g}）"
            )
        except Exception as error:  # 批处理必须继续，所以在单文件粒度捕获异常。
            print(f"[失败] 输入 {source_file}：{error}；已跳过")

    expected_sensors = set(FLOW_SPEED_BY_SENSOR)
    output_files: list[Path] = []
    for relative_parent in sorted(grouped_rows, key=lambda item: item.as_posix().casefold()):
        for angle in sorted(grouped_rows[relative_parent]):
            sensor_rows = grouped_rows[relative_parent][angle]
            actual_sensors = set(sensor_rows)
            missing_sensors = sorted(expected_sensors - actual_sensors)
            if missing_sensors:
                missing_speeds = [
                    FLOW_SPEED_BY_SENSOR[sensor_number]
                    for sensor_number in missing_sensors
                ]
                missing_speed_text = "、".join(
                    format(speed, "g") for speed in missing_speeds
                )
                print(
                    f"[错误] {relative_parent or Path('.')} 的 "
                    f"ang{format_angle(angle)} 缺少流速 {missing_speed_text}；"
                    f"仍将使用已有流速生成 {len(sensor_rows)} 行"
                )

            # 如果这个角度的所有输入文件都失败，就没有任何数值可写。此时
            # 跳过空表，但继续处理后面的角度和 model。
            if not sensor_rows:
                print(
                    f"[失败] 输出 {relative_parent or Path('.')} / "
                    f"ang{format_angle(angle)}.csv：没有可用行；已跳过"
                )
                continue

            # 只遍历实际成功的 sensor，避免缺失流速时访问不存在的数据。
            # 按 sensor 编号排列后，已有流速仍保持从小到大的稳定顺序。
            rows: list[list[float]] = []
            for sensor_number in sorted(sensor_rows):
                row = sensor_rows[sensor_number].tolist()
                row.append(FLOW_SPEED_BY_SENSOR[sensor_number])
                rows.append(row)

            output_file = output_root / relative_parent / f"ang{format_angle(angle)}.csv"

            # 写入失败也只影响当前角度文件。例如 --no-overwrite 遇到已有文件
            # 时，其他 model 和角度仍然可以继续生成。
            try:
                write_group_csv(rows, output_file, overwrite)
                output_files.append(output_file)
                print(f"[成功] 输出 {output_file}（{len(rows)} 行）")
            except Exception as error:  # 输出文件之间彼此独立，可安全继续。
                print(f"[失败] 输出 {output_file}：{error}；已跳过")

    return output_files


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器，并使用文件顶部变量作为默认值。"""

    parser = argparse.ArgumentParser(
        description="对 sensor CSV 做逐列截尾平均，并按 model/角度合并四种流速"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=inputpath,
        help="单个 sensor CSV，或要递归扫描的输入根目录",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=outputpath,
        help="保存 model/angB.csv 的输出根目录",
    )
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=trim_fraction,
        help="每列最小端和最大端各丢弃的比例，默认 0.10",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=overwrite_existing,
        help="是否允许覆盖已有的 angB.csv",
    )
    return parser


def main() -> int:
    """解析参数，执行批量汇总，并返回适合命令行使用的状态码。"""

    arguments = build_parser().parse_args()
    resolved_input = arguments.input_path.expanduser().resolve()
    resolved_output = arguments.output_path.expanduser().resolve()

    try:
        output_files = summarize_sensor_tree(
            input_root=resolved_input,
            output_root=resolved_output,
            fraction=arguments.trim_fraction,
            overwrite=arguments.overwrite,
        )
    except Exception as error:
        print(f"[失败] {error}")
        return 2

    print(f"处理结束：生成 {len(output_files)} 个角度汇总文件")
    print(f"输出目录：{resolved_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
