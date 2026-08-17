"""统一的 sensor 原始数据低通/带通滤波脚本。

支持两种模式：
1. lowpass：4 阶 Butterworth 低通，对应旧 filter_data.py 的主要功能。
2. bandpass：Hamming 窗 FIR 带通，对应旧 sensor_data_filter_frequency.py。

项目原始 sensor CSV 为无表头、逗号分隔的 6 列，依次映射为
TX、TY、TZ、FX、FY、FZ。脚本默认递归处理 data 文件夹中的
sensor_编号_ang角度.csv，并在仓库根目录的 filtered_data 中保留原目录结构。
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

# 该脚本只需要批量保存 PNG，不需要弹出绘图窗口；Agg 后端也能在缺少 Tk/Tcl 的环境中工作。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


# =============================================================================
# 用户可直接修改的默认配置；命令行参数可以临时覆盖这些值。
# =============================================================================

# SCRIPT_DIR 是当前脚本所在的 Experiment 文件夹。
SCRIPT_DIR = Path(__file__).resolve().parent

# REPO_ROOT 是仓库根目录；正式数据目录统一放在这里，而不是放进 Experiment。
REPO_ROOT = SCRIPT_DIR.parent

# 默认从仓库根目录的 data 读取原始 sensor 数据。
inputpath = REPO_ROOT / "data"

# 过滤结果与原始 data 完全分离，避免递归扫描时再次读到生成的文件。
# 例如 data/model_1/sensor_1_ang0.csv 会写入
# filtered_data/model_1/sensor_1_ang0_lowpass_filtered.csv。
# 默认把滤波结果写到仓库根目录的 filtered_data。
outputpath = REPO_ROOT / "filtered_data"

# data/datanew 是 rotate_force.py 生成的旧旋转结果，sensor_filtered 是旧版
# filter_sensor_data.py 的输出；它们都不是这次应再次滤波的原始数据。
# 如项目以后增加其他生成目录，可继续把目录名加入这个集合。
excluded_directory_names = frozenset({"datanew", "sensor_filtered"})

filter_mode = "lowpass"
fs = 500.0
lowpass_cutoff = 2
bandpass_low = 0.05
bandpass_high = 1.0
butterworth_order = 5
fir_numtaps = 401
selected_columns = "all"
# 截取区间采用左闭右开索引：[trim_head, trim_tail)；trim_tail=0 表示一直取到末尾。
trim_head = 1200
trim_tail = 2800
max_nan_gap = 5

# 是否默认生成只包含滤波结果的曲线图。
# False 表示默认不画图；改为 True 后，不传命令行参数也会生成图片。
make_plots = True

# 是否默认允许覆盖已经存在的滤波结果。
# False 可以防止误覆盖；需要反复处理同一文件时可以改为 True。
overwrite_existing = True

SENSOR_COLUMNS = ("TX", "TY", "TZ", "FX", "FY", "FZ")
SENSOR_FILE_RE = re.compile(
    r"^sensor_(?P<index>\d+)_ang(?P<angle>-?\d+(?:\.\d+)?)\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FilterConfig:
    """保存滤波参数，所有文件共用同一份经过校验的配置。"""

    input_path: Path
    output_dir: Path
    mode: str
    sample_rate: float
    cutoff: float
    band_low: float
    band_high: float
    order: int
    numtaps: int
    columns: tuple[int, ...]
    trim_head: int
    trim_tail: int
    max_nan_gap: int
    make_plots: bool
    overwrite: bool


def parse_sensor_filename(path: Path) -> tuple[int, float] | None:
    """从文件名读取 sensor 编号和角度；不是原始 sensor 文件时返回 None。"""

    match = SENSOR_FILE_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group("index")), float(match.group("angle"))


def discover_sensor_files(
    path: Path,
    excluded_names: frozenset[str] = excluded_directory_names,
    excluded_roots: tuple[Path, ...] = (),
) -> list[Path]:
    """递归发现原始 sensor CSV，并跳过已知的生成目录。

    Args:
        path: 单个 sensor CSV，或需要递归检查的输入根目录。
        excluded_names: 不应进入的目录名称；比较时不区分大小写。
        excluded_roots: 不应读取的完整目录路径，主要用于排除输出目录。

    Returns:
        稳定排序后的文件路径列表。目录路径优先，其次依次按 sensor 编号、
        角度和文件名排序，所以重复运行时处理顺序一致。
    """

    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在：{path}")
    if path.is_file():
        if parse_sensor_filename(path) is None:
            raise ValueError(f"文件名不符合 sensor_编号_ang角度.csv：{path.name}")
        return [path]

    normalized_excluded_names = {name.casefold() for name in excluded_names}
    files: list[Path] = []

    # rglob 会检查输入根目录下任意深度的 CSV；严格文件名正则会自动排除
    # gather CSV、PNG 以及名称带有 _filtered 后缀的旧输出文件。
    for item in path.rglob("*.csv"):
        relative_path = item.relative_to(path)
        if any(part.casefold() in normalized_excluded_names for part in relative_path.parts[:-1]):
            continue
        if any(item == root or root in item.parents for root in excluded_roots):
            continue
        if item.is_file() and parse_sensor_filename(item) is not None:
            files.append(item)

    def sort_key(item: Path) -> tuple[str, int, float, str]:
        """生成跨目录也稳定的排序键。"""

        parsed_name = parse_sensor_filename(item)
        assert parsed_name is not None  # files 列表只保存已经通过正则校验的路径。
        sensor_index, angle = parsed_name
        relative_parent = item.parent.relative_to(path).as_posix().casefold()
        return relative_parent, sensor_index, angle, item.name.casefold()

    return sorted(files, key=sort_key)


def read_sensor_csv(path: Path) -> pd.DataFrame:
    """读取原始无表头 6 列 CSV，并明确拒绝 gather 等其他结构。"""

    frame = pd.read_csv(path, header=None, sep=",")
    if frame.empty:
        raise ValueError("文件为空")
    if frame.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError(
            f"检测到 {frame.shape[1]} 列；sensor 原始数据必须为 6 列，"
            "gather CSV 需要使用 data/gather_data.py"
        )

    # errors='coerce' 将无法解析的文本变为 NaN，随后由统一缺失值规则处理。
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric.columns = SENSOR_COLUMNS
    return numeric


def parse_columns(text: str) -> tuple[int, ...]:
    """解析 all、轴名或从 0 开始的列号，例如 all、FX、0,3,5。"""

    if text.strip().lower() == "all":
        return tuple(range(len(SENSOR_COLUMNS)))

    parsed: list[int] = []
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        if item.upper() in SENSOR_COLUMNS:
            index = SENSOR_COLUMNS.index(item.upper())
        else:
            try:
                index = int(item)
            except ValueError as error:
                raise ValueError(f"未知列 {item!r}；请使用 TX..FZ 或 0..5") from error
        if index < 0 or index >= len(SENSOR_COLUMNS):
            raise ValueError(f"列号 {index} 越界；合法范围是 0..5")
        if index not in parsed:
            parsed.append(index)

    if not parsed:
        raise ValueError("至少需要选择一列")
    return tuple(parsed)


def trim_frame(frame: pd.DataFrame, head: int, tail: int) -> pd.DataFrame:
    """截取从 ``head`` 到 ``tail`` 的数据；``tail`` 超出总长度时截取到末尾。"""

    if head < 0 or tail < 0:
        raise ValueError("trim_head 和 trim_tail 不能为负数")

    # trim_tail=0 保留为“截取到文件末尾”的便捷写法，确保脚本顶部的默认配置仍可直接运行。
    # 当 tail 大于文件总行数时，使用总行数作为结束位置，以满足“数据不足则取到最后”的需求。
    end = len(frame) if tail == 0 else min(tail, len(frame))
    if head >= end:
        raise ValueError(
            f"截取区间无有效数据：head={head}，tail={tail}，原始行数={len(frame)}"
        )

    # pandas 的 iloc 使用左闭右开区间，即包含索引 head、不包含索引 end。
    return frame.iloc[head:end].reset_index(drop=True).copy()


def fill_small_gaps(values: np.ndarray, max_gap: int, column_name: str) -> np.ndarray:
    """只插值较短的内部 NaN/Inf 缺口，避免悄悄修复严重损坏的数据。"""

    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    missing = series.isna().to_numpy()
    if not missing.any():
        return series.to_numpy()
    if missing[0] or missing[-1]:
        raise ValueError(f"{column_name} 的首尾存在 NaN/Inf，无法安全插值")

    # 找出连续缺失区间长度；任一区间过长就停止处理该文件。
    padded = np.concatenate(([False], missing, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    longest_gap = int(np.max(ends - starts))
    if longest_gap > max_gap:
        raise ValueError(
            f"{column_name} 最长缺失区间为 {longest_gap}，超过允许值 {max_gap}"
        )
    return series.interpolate(method="linear").to_numpy()


def validate_filter_config(config: FilterConfig) -> None:
    """在读取大量文件前检查所有与文件长度无关的滤波参数。"""

    nyquist = config.sample_rate / 2.0
    if config.sample_rate <= 0:
        raise ValueError("fs 必须大于 0")
    if config.order <= 0:
        raise ValueError("Butterworth 阶数必须大于 0")
    if not 0 < config.cutoff < nyquist:
        raise ValueError(f"cutoff 必须位于 0 和 Nyquist 频率 {nyquist} Hz 之间")
    if not 0 < config.band_low < config.band_high < nyquist:
        raise ValueError(
            f"带通范围必须满足 0 < low < high < {nyquist} Hz"
        )
    if config.numtaps < 3 or config.numtaps % 2 == 0:
        raise ValueError("FIR numtaps 必须是大于等于 3 的奇数")
    if config.max_nan_gap < 0:
        raise ValueError("max_nan_gap 不能为负数")


def lowpass_filter(values: np.ndarray, config: FilterConfig) -> np.ndarray:
    """使用数值稳定的 SOS 形式进行零相位 Butterworth 低通滤波。"""

    coefficients = signal.butter(
        config.order,
        config.cutoff,
        btype="lowpass",
        fs=config.sample_rate,
        output="sos",
    )
    return signal.sosfiltfilt(coefficients, values)


def bandpass_filter(values: np.ndarray, config: FilterConfig) -> np.ndarray:
    """使用 Hamming 窗 FIR 系数进行零相位带通滤波。"""

    if len(values) <= 3 * (config.numtaps - 1):
        raise ValueError(
            f"数据只有 {len(values)} 行，无法使用 {config.numtaps} taps 的零相位 FIR；"
            "请减小 --numtaps 或使用更长记录"
        )

    duration = len(values) / config.sample_rate
    if config.band_low < 1.0 / duration:
        warnings.warn(
            f"记录时长仅 {duration:.3f}s，低截止频率 {config.band_low}Hz "
            "低于约一个完整周期，结果可能不可靠。",
            RuntimeWarning,
            stacklevel=2,
        )

    coefficients = signal.firwin(
        config.numtaps,
        (config.band_low, config.band_high),
        pass_zero=False,
        fs=config.sample_rate,
        window="hamming",
    )
    return signal.filtfilt(coefficients, [1.0], values)


def filter_frame(frame: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    """仅替换用户选择的列，未选择列保持裁剪后的原值。"""

    result = frame.copy()
    for column_index in config.columns:
        column_name = SENSOR_COLUMNS[column_index]
        clean_values = fill_small_gaps(
            frame.iloc[:, column_index].to_numpy(),
            config.max_nan_gap,
            column_name,
        )
        if config.mode == "lowpass":
            filtered_values = lowpass_filter(clean_values, config)
        else:
            filtered_values = bandpass_filter(clean_values, config)
        result.iloc[:, column_index] = filtered_values
    return result


def write_csv_atomically(frame: pd.DataFrame, output_file: Path) -> None:
    """先写临时文件再替换目标，避免程序中断后留下半个 CSV。"""

    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    frame.to_csv(temporary_file, index=False, header=False, float_format="%.10f")
    temporary_file.replace(output_file)


def save_filtered_plot(
    filtered: pd.DataFrame,
    source_file: Path,
    output_dir: Path,
    config: FilterConfig,
) -> None:
    """只绘制所选列的滤波结果，并用数据组编号作为横轴。"""

    # 数据组编号从 1 开始，使横轴与用户阅读 CSV 时常用的“第 1 组、第 2 组”一致。
    sample_numbers = np.arange(1, len(filtered) + 1)

    # 较短数据每 200 组显示一个刻度，较长数据每 500 组显示一个刻度，避免标签过密。
    tick_interval = 200 if len(filtered) <= 2000 else 500
    # 除起点 1 外，其余刻度使用整齐的 200/500 倍数，例如 1、500、1000……4000。
    tick_positions = [1, *range(tick_interval, len(filtered) + 1, tick_interval)]
    # 如果最后一组不恰好落在固定间隔上，也把末尾编号显示出来，明确图中数据范围。
    if tick_positions[-1] != len(filtered):
        tick_positions.append(len(filtered))

    figure, axes = plt.subplots(
        len(config.columns),
        1,
        figsize=(12, max(3.2, 2.6 * len(config.columns))),
        sharex=True,
        squeeze=False,
    )
    for row, column_index in enumerate(config.columns):
        axis = axes[row, 0]
        column_name = SENSOR_COLUMNS[column_index]
        # 图片中只保留滤波后的曲线，不再绘制任何原始数据点或原始数据曲线。
        axis.plot(
            sample_numbers,
            filtered.iloc[:, column_index],
            linewidth=1.2,
            label="filtered",
        )
        axis.set_ylabel(column_name)
        axis.set_xlim(1, len(filtered))
        axis.set_xticks(tick_positions)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")
    axes[-1, 0].set_xlabel(f"Data group number (1-{len(filtered)})")
    figure.suptitle(f"{source_file.name} - {config.mode}")
    figure.tight_layout()
    figure.savefig(output_dir / f"{source_file.stem}_{config.mode}.png", dpi=160)
    plt.close(figure)


def process_sensor_file(path: Path, config: FilterConfig) -> Path:
    """读取、裁剪、滤波并保存一个 sensor 文件。"""

    # 目录输入时，先计算源文件相对于 inputpath 的父目录，再把它拼到
    # outputpath 后面。单文件输入没有可复制的目录树，直接写到 outputpath。
    relative_parent = (
        Path()
        if config.input_path.is_file()
        else path.parent.relative_to(config.input_path)
    )
    file_output_dir = config.output_dir / relative_parent
    file_output_dir.mkdir(parents=True, exist_ok=True)

    output_file = file_output_dir / f"{path.stem}_{config.mode}_filtered.csv"
    if output_file.exists() and not config.overwrite:
        raise FileExistsError(f"输出已存在；使用 --overwrite 可覆盖：{output_file}")

    raw_frame = read_sensor_csv(path)
    cropped_frame = trim_frame(raw_frame, config.trim_head, config.trim_tail)
    filtered_frame = filter_frame(cropped_frame, config)
    write_csv_atomically(filtered_frame, output_file)
    if config.make_plots:
        # 图片与对应的滤波 CSV 使用同一个目录，从而保留 model_1 等输入子目录结构。
        save_filtered_plot(filtered_frame, path, file_output_dir, config)
    return output_file


def build_parser() -> argparse.ArgumentParser:
    """创建命令行接口，并让顶部变量继续作为可编辑默认值。"""

    parser = argparse.ArgumentParser(description="批量滤波原始 6 轴 sensor CSV")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=inputpath,
        help="单个文件或要递归检查的输入文件夹",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=outputpath,
        help="镜像保存目录结构的输出根目录",
    )
    parser.add_argument("--mode", choices=("lowpass", "bandpass"), default=filter_mode)
    parser.add_argument("--fs", type=float, default=fs, help="采样率（Hz）")
    parser.add_argument("--cutoff", type=float, default=lowpass_cutoff, help="低通截止频率（Hz）")
    parser.add_argument("--low", type=float, default=bandpass_low, help="带通低截止频率（Hz）")
    parser.add_argument("--high", type=float, default=bandpass_high, help="带通高截止频率（Hz）")
    parser.add_argument("--order", type=int, default=butterworth_order, help="Butterworth 阶数")
    parser.add_argument("--numtaps", type=int, default=fir_numtaps, help="Hamming FIR 奇数长度")
    parser.add_argument("--columns", default=selected_columns, help="all、TX..FZ 或 0..5，逗号分隔")
    parser.add_argument("--trim-head", type=int, default=trim_head, help="截取区间的起始索引（包含）")
    parser.add_argument(
        "--trim-tail",
        type=int,
        default=trim_tail,
        help="截取区间的结束索引（不包含）；0 或超过总行数时取到末尾",
    )
    parser.add_argument("--max-nan-gap", type=int, default=max_nan_gap)
    # 未提供命令行开关时，使用代码顶部 make_plots 的默认值。
    # BooleanOptionalAction 同时提供 --plot 和 --no-plot，方便临时双向覆盖。
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=make_plots,
        help="是否生成只包含滤波结果的曲线图",
    )

    # overwrite_existing 控制默认覆盖策略；命令行可用 --overwrite 或 --no-overwrite 覆盖。
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=overwrite_existing,
        help="是否允许覆盖已有结果",
    )
    return parser


def main() -> int:
    """构建配置并批量处理；坏文件只计入失败，不中断整个文件夹。"""

    arguments = build_parser().parse_args()
    resolved_input = arguments.input_path.expanduser().resolve()
    resolved_output = (
        arguments.output_path.expanduser().resolve()
        if arguments.output_path is not None
        else (resolved_input.parent if resolved_input.is_file() else resolved_input) / "sensor_filtered"
    )
    resolved_output.mkdir(parents=True, exist_ok=True)

    config = FilterConfig(
        input_path=resolved_input,
        output_dir=resolved_output,
        mode=arguments.mode,
        sample_rate=arguments.fs,
        cutoff=arguments.cutoff,
        band_low=arguments.low,
        band_high=arguments.high,
        order=arguments.order,
        numtaps=arguments.numtaps,
        columns=parse_columns(arguments.columns),
        trim_head=arguments.trim_head,
        trim_tail=arguments.trim_tail,
        max_nan_gap=arguments.max_nan_gap,
        make_plots=arguments.plot,
        overwrite=arguments.overwrite,
    )
    validate_filter_config(config)

    files = discover_sensor_files(config.input_path, excluded_roots=(config.output_dir,))
    if not files:
        print(f"未找到合法文件：{config.input_path}/sensor_编号_ang角度.csv")
        return 1

    success_count = 0
    for path in files:
        try:
            output_file = process_sensor_file(path, config)
            success_count += 1
            print(f"[成功] {path.name} -> {output_file}")
        except Exception as error:  # 批处理需要继续，因此在文件粒度捕获异常。
            print(f"[失败] {path.name}：{error}")

    print(f"处理结束：成功 {success_count}/{len(files)} 个文件")
    return 0 if success_count == len(files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
