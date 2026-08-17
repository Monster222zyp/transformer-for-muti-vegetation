"""批量读取、导出并绘制 gather 原始数据。

原始 gather CSV 没有表头，使用空白字符分隔。每行的结构为：
时间戳 + N 组（位移、速度、附加值）。不同数据文件可能有不同的 N，
因此本脚本会根据实际列数动态识别通道，不再固定要求 6 个通道。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 用户可直接修改的默认配置
# =============================================================================

# SCRIPT_DIR 是当前脚本所在的 Experiment 文件夹，REPO_ROOT 是仓库根目录。
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# inputpath 指定需要处理的文件夹。gather 不属于正式数据链，但保留工具时默认指向真实目录。
inputpath = REPO_ROOT / "data" / "model_1"

# outputpath 为 None 时，结果写入 inputpath/gather_figure。
# 也可以改成 Path(r"D:\your\output") 之类的绝对路径。
outputpath: Path | None = None

# fs 是采样率，单位为 Hz；它用于由样本序号生成时间轴。
fs = 100.0

# 位移数据的单位换算系数。保留旧脚本行为：原始值除以 1000。
displacement_scale = 1.0 / 1000.0

# 速度数据的单位换算系数。保留旧脚本行为：原始值乘以 2.5。
velocity_scale = 2.5

# initial：每个通道减去第一个样本；fixed：减去固定值；none：不减偏移。
offset_mode = "initial"
fixed_offset_value = 5.0

# 文件名必须完整匹配此规则。兼容历史拼写错误 gatehr。
GATHER_FILE_RE = re.compile(
    r"^(?:gather|gatehr)_(?P<index>\d+)_ang(?P<angle>-?\d+(?:\.\d+)?)\.csv$",
    re.IGNORECASE,
)

# 前 6 个通道沿用旧脚本名称；通道更多时自动使用 channel_7 等通用名称。
LEGACY_CHANNEL_NAMES = ("x", "y", "xx", "z", "b", "c")


@dataclass(frozen=True)
class GatherConfig:
    """保存一次处理所需的全部配置，避免函数依赖零散的全局变量。"""

    input_dir: Path
    output_dir: Path
    sample_rate: float
    displacement_scale: float
    velocity_scale: float
    offset_mode: str
    fixed_offset: float
    make_plots: bool


def parse_gather_filename(path: Path) -> tuple[int, float] | None:
    """解析 gather 文件名，返回（实验编号，角度）；不匹配时返回 None。"""

    match = GATHER_FILE_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group("index")), float(match.group("angle"))


def discover_gather_files(folder: Path) -> list[Path]:
    """只发现指定文件夹顶层的合法 gather CSV，并按编号和角度数值排序。"""

    if not folder.exists():
        raise FileNotFoundError(f"输入文件夹不存在：{folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"inputpath 必须是文件夹：{folder}")

    # 只扫描顶层，避免把输出文件夹中的结果再次当成输入。
    files = [path for path in folder.iterdir() if path.is_file() and parse_gather_filename(path)]
    return sorted(files, key=lambda path: (*parse_gather_filename(path), path.name.lower()))


def read_gather_csv(path: Path) -> pd.DataFrame:
    """读取无表头、空白分隔的 gather CSV，并验证其动态三列分组结构。"""

    # header=None 很重要：原始文件第一行就是数据，不是列名。
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if frame.empty:
        raise ValueError("文件为空")

    # 强制检查每个单元格都是数值；错误位置会由 pandas 放入异常信息中。
    frame = frame.apply(pd.to_numeric, errors="raise")
    column_count = frame.shape[1]
    if column_count < 4 or (column_count - 1) % 3 != 0:
        raise ValueError(
            f"列数为 {column_count}，但 gather 应为 1 + N×3 列"
        )
    return frame


def channel_name(channel_index: int) -> str:
    """把从 0 开始的通道序号转换为稳定、可读的输出列名。"""

    if channel_index < len(LEGACY_CHANNEL_NAMES):
        return LEGACY_CHANNEL_NAMES[channel_index]
    return f"channel_{channel_index + 1}"


def calculate_offset(values: pd.Series, config: GatherConfig) -> float:
    """根据 offset_mode 计算一个通道需要减去的偏移量。"""

    if config.offset_mode == "initial":
        return float(values.iloc[0])
    if config.offset_mode == "fixed":
        return config.fixed_offset
    return 0.0


def process_gather_file(path: Path, config: GatherConfig) -> tuple[int, Path]:
    """处理一个 gather CSV，返回识别出的通道数和导出 CSV 路径。"""

    frame = read_gather_csv(path)
    channel_count = (frame.shape[1] - 1) // 3
    time_seconds = np.arange(len(frame), dtype=float) / config.sample_rate

    # 每组的第 1 列是位移，第 2 列是速度；第 3 列暂不参与旧功能的导出和绘图。
    displacement_indices = [1 + 3 * index for index in range(channel_count)]
    velocity_indices = [2 + 3 * index for index in range(channel_count)]

    # 导出归零或固定偏移后的位移，保持旧脚本的 CSV 功能。
    exported = pd.DataFrame({"t": time_seconds})
    for channel_index, column_index in enumerate(displacement_indices):
        values = frame.iloc[:, column_index].astype(float) * config.displacement_scale
        exported[channel_name(channel_index)] = values - calculate_offset(values, config)

    output_csv = config.output_dir / f"{path.stem}_disp_data.csv"
    exported.to_csv(output_csv, index=False, float_format="%.6f")

    # --no-plot 可只导出 CSV，适合在服务器或批处理环境中运行。
    if config.make_plots:
        figure, (displacement_axis, velocity_axis) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True
        )

        for channel_index, column_index in enumerate(displacement_indices):
            raw_values = frame.iloc[:, column_index].astype(float) * config.displacement_scale
            displacement_axis.plot(
                time_seconds,
                raw_values,
                label=f"{channel_name(channel_index)} displacement",
            )
        displacement_axis.set_title("Gather displacement")
        displacement_axis.set_ylabel("Displacement (mm)")
        displacement_axis.grid(True)
        displacement_axis.legend(loc="upper right")

        for channel_index, column_index in enumerate(velocity_indices):
            raw_values = frame.iloc[:, column_index].astype(float) * config.velocity_scale
            velocity_axis.plot(
                time_seconds,
                raw_values,
                label=f"{channel_name(channel_index)} velocity",
            )
        velocity_axis.set_title("Gather velocity")
        velocity_axis.set_xlabel("Time (s)")
        velocity_axis.set_ylabel("Velocity (mm/s)")
        velocity_axis.grid(True)
        velocity_axis.legend(loc="upper right")

        figure.tight_layout()
        figure.savefig(
            config.output_dir / f"{path.stem}_disp_vel_subplots.png",
            dpi=200,
        )
        plt.close(figure)

    return channel_count, output_csv


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器；命令行参数会覆盖文件顶部的默认配置。"""

    parser = argparse.ArgumentParser(description="批量处理原始 gather CSV")
    parser.add_argument("--input-path", type=Path, default=inputpath, help="原始 CSV 文件夹")
    parser.add_argument("--output-path", type=Path, default=outputpath, help="输出文件夹")
    parser.add_argument("--fs", type=float, default=fs, help="采样率（Hz）")
    parser.add_argument(
        "--offset-mode",
        choices=("initial", "fixed", "none"),
        default=offset_mode,
        help="位移偏移处理方式",
    )
    parser.add_argument("--fixed-offset", type=float, default=fixed_offset_value)
    parser.add_argument("--no-plot", action="store_true", help="不生成 PNG 图")
    return parser


def main() -> int:
    """批量处理输入文件夹；单个文件失败不会中断其他文件。"""

    arguments = build_parser().parse_args()
    if arguments.fs <= 0:
        raise ValueError("--fs 必须大于 0")

    resolved_input = arguments.input_path.expanduser().resolve()
    resolved_output = (
        arguments.output_path.expanduser().resolve()
        if arguments.output_path is not None
        else resolved_input / "gather_figure"
    )
    resolved_output.mkdir(parents=True, exist_ok=True)

    config = GatherConfig(
        input_dir=resolved_input,
        output_dir=resolved_output,
        sample_rate=arguments.fs,
        displacement_scale=displacement_scale,
        velocity_scale=velocity_scale,
        offset_mode=arguments.offset_mode,
        fixed_offset=arguments.fixed_offset,
        make_plots=not arguments.no_plot,
    )

    files = discover_gather_files(config.input_dir)
    if not files:
        print(f"未找到合法文件：{config.input_dir}/gather_编号_ang角度.csv")
        return 1

    success_count = 0
    for path in files:
        try:
            channel_count, output_csv = process_gather_file(path, config)
            success_count += 1
            print(f"[成功] {path.name}：识别 {channel_count} 个通道 -> {output_csv}")
        except Exception as error:  # 批处理时记录坏文件，然后继续处理其余文件。
            print(f"[失败] {path.name}：{error}")

    print(f"处理结束：成功 {success_count}/{len(files)} 个文件")
    return 0 if success_count == len(files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
