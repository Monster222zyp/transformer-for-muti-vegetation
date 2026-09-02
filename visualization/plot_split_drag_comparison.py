"""将模型 prediction CSV 按排序后的样本比例切分，并绘制多张阻力对比图。

这个脚本复用训练阶段生成 ``test_drag_comparison.png`` 时采用的排序规则：

1. 首先按照 ``isolated_drag``（单株阻力，也就是用户所说的 ``iso_drag``）升序排列；
2. ``isolated_drag`` 相同时，再按照 ``target_drag`` 升序排列；
3. 如果前两项仍相同，则使用 ``source_index`` 保证排序结果稳定且可复现。

切分发生在排序之后。例如共有 100 个样本，传入 ``--cuts 0.3 0.6``，会得到
排序区间 [1, 30]、[31, 60] 和 [61, 100] 对应的三张图片。每张图片都会单独
建立 Matplotlib 坐标轴，因此会根据本段数据自动采用各自的纵轴比例尺。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

# 使用不依赖桌面窗口的绘图后端，使脚本能够在终端、服务器和 CI 环境中运行。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 当前文件位于仓库根目录下的 visualization 文件夹中。将仓库根目录加入
# Python 模块搜索路径后，无论从哪个工作目录启动脚本，都可以导入 model 包。
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# 所有可配置参数的默认值均集中写在代码开头。直接运行脚本时会使用这些值，
# 同时也可以通过对应的命令行参数临时覆盖，而不需要修改代码。
DEFAULT_INPUT_PATH = REPOSITORY_ROOT / "artifacts" / "fold_0" / "test_predictions.csv"
DEFAULT_CUTS = (0.3, 0.6)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "visualization" 
DEFAULT_TITLE = "Fold 0 test drag comparison"
DEFAULT_DPI = 180

from model.training.visualization import sort_drag_predictions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 需要解析的参数序列。传入 ``None`` 时读取终端中的命令行参数；
            测试代码也可以传入一个字符串列表来直接调用本函数。

    Returns:
        argparse.Namespace: 包含输入 CSV、切分比例、输出目录、标题和图片 DPI
        的参数对象。
    """

    parser = argparse.ArgumentParser(
        description=(
            "按照 isolated_drag、target_drag 的顺序排列 prediction CSV，"
            "再按比例切成若干张阻力对比图。"
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=(
            "prediction CSV 文件路径。省略时使用代码开头的 "
            f"DEFAULT_INPUT_PATH：{DEFAULT_INPUT_PATH}。"
        ),
    )
    parser.add_argument(
        "--cuts",
        type=float,
        nargs="+",
        default=list(DEFAULT_CUTS),
        metavar="RATIO",
        help=(
            "升序排列的累计切分比例，必须位于 0 和 1 之间且严格递增。"
            "例如 --cuts 0.3 0.6 会生成三张图；默认值为 0.3 0.6。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "图片输出目录。默认在输入 CSV 旁创建 "
            "<CSV名称>_drag_comparison_parts 文件夹。"
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=DEFAULT_TITLE,
        help="图片标题前缀；默认根据输入 CSV 文件名自动生成。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"输出图片分辨率，必须为正整数；默认值为 {DEFAULT_DPI}。",
    )
    return parser.parse_args(argv)


def read_prediction_csv(input_path: Path) -> list[dict[str, str]]:
    """读取 prediction CSV，并保留每一行的字段名称和值。

    Args:
        input_path: prediction CSV 的文件路径。CSV 至少需要包含
            ``isolated_drag``、``target_drag`` 和 ``predicted_drag``；
            ``source_index`` 用于在阻力相同时维持稳定顺序。

    Returns:
        list[dict[str, str]]: CSV 中的所有非空数据行，每一行表示一个 sample。

    Raises:
        FileNotFoundError: 输入路径不存在或不是文件。
        ValueError: CSV 没有表头或没有任何 sample。
    """

    resolved_path = input_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"找不到 prediction CSV 文件：{resolved_path}")

    # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 CSV 文件。
    with resolved_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 缺少表头：{resolved_path}")
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"CSV 中没有可绘制的 sample：{resolved_path}")
    return rows


def calculate_boundaries(sample_count: int, cuts: Sequence[float]) -> list[int]:
    """把累计切分比例转换为 Python 切片所需的整数边界。

    Args:
        sample_count: 排序后的 sample 总数，必须大于 0。
        cuts: 累计切分比例。例如 ``(0.3, 0.6)`` 表示在总长度的 30% 和
            60% 位置切分。

    Returns:
        list[int]: 包含开头 0 和结尾 ``sample_count`` 的完整边界列表。
        例如 100 个样本和 ``(0.3, 0.6)`` 会返回 ``[0, 30, 60, 100]``。

    Raises:
        ValueError: 样本数无效、比例不是有限数值、比例越界、没有严格递增，
            或某两个比例落到同一个整数边界而产生空分段。
    """

    if sample_count <= 0:
        raise ValueError("sample_count 必须大于 0。")

    normalized_cuts = [float(cut) for cut in cuts]
    for index, cut in enumerate(normalized_cuts):
        if not math.isfinite(cut):
            raise ValueError(f"第 {index + 1} 个切分比例不是有限数值：{cut}")
        if not 0.0 < cut < 1.0:
            raise ValueError(f"切分比例必须位于 0 和 1 之间：{cut}")

    if any(left >= right for left, right in zip(normalized_cuts, normalized_cuts[1:])):
        raise ValueError("切分比例必须严格递增，例如：--cuts 0.3 0.6。")

    # 向下取整意味着比例 p 对应前 floor(N*p) 个排序样本，定义明确且可复现。
    integer_cuts = [math.floor(sample_count * cut) for cut in normalized_cuts]
    boundaries = [0, *integer_cuts, sample_count]
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError(
            "当前切分比例在该 CSV 的样本数量下产生了空分段；"
            "请减少切分点，或让相邻比例之间的距离更大。"
        )
    return boundaries


def default_output_directory(input_path: Path) -> Path:
    """根据输入 CSV 名称生成默认输出目录。

    Args:
        input_path: prediction CSV 文件路径。

    Returns:
        Path: 位于输入文件旁边的输出目录路径。文件名末尾若为
        ``_predictions``，会先移除该后缀，使名称更加简洁。
    """

    base_name = input_path.stem
    if base_name.endswith("_predictions"):
        base_name = base_name.removesuffix("_predictions")
    return input_path.parent / f"{base_name}_drag_comparison_parts"


def default_title(input_path: Path) -> str:
    """由输入 CSV 文件名生成可读的默认标题。

    Args:
        input_path: prediction CSV 文件路径。

    Returns:
        str: 将下划线替换为空格后的标题，例如 ``test_predictions.csv``
        会得到 ``Test drag comparison``。
    """

    base_name = input_path.stem
    if base_name.endswith("_predictions"):
        base_name = base_name.removesuffix("_predictions")
    return f"{base_name.replace('_', ' ').title()} drag comparison"


def plot_segment(
    rows: Sequence[dict[str, object]],
    *,
    start_index: int,
    end_index: int,
    part_number: int,
    part_count: int,
    output_path: Path,
    title_prefix: str,
    dpi: int,
) -> None:
    """绘制一个连续排序区间中的阻力对比图。

    Args:
        rows: 已经完成全局排序、并切出的当前 sample 分段。
        start_index: 当前分段在完整排序中的起始下标，使用 Python 的 0-based
            规则且包含该位置。
        end_index: 当前分段的结束下标，使用 Python 切片规则且不包含该位置。
        part_number: 当前图片编号，从 1 开始。
        part_count: 所有图片的总数量。
        output_path: 当前 PNG 图片的保存路径。
        title_prefix: 图片标题前缀。
        dpi: 输出图片分辨率。

    Returns:
        None: 函数直接将图片写入 ``output_path``。
    """

    # 横坐标保留完整 CSV 中的全局排序编号，而不是每一张图都重新从 1 开始。
    # 因此不同图片首尾相接，仍能直观看出它们在原始排序中的位置。
    x_values = list(range(start_index + 1, end_index + 1))
    target_drag = [float(row["target_drag"]) for row in rows]
    predicted_drag = [float(row["predicted_drag"]) for row in rows]
    isolated_drag = [float(row["isolated_drag"]) for row in rows]

    # 图片宽度随当前分段的样本数变化，并沿用原始对比图的绘图风格。
    figure_width = min(18.0, max(10.0, len(rows) * 0.08))
    figure, axis = plt.subplots(figsize=(figure_width, 6.0))
    axis.plot(
        x_values,
        target_drag,
        label="target_drag",
        linewidth=1.4,
        marker="o",
        markersize=2.5,
    )
    axis.plot(
        x_values,
        predicted_drag,
        label="predicted_drag",
        linewidth=1.4,
        marker="o",
        markersize=2.5,
    )
    axis.plot(
        x_values,
        isolated_drag,
        label="isolated_drag",
        linewidth=1.4,
        marker="o",
        markersize=2.5,
    )

    # 每一张图拥有独立的 axis，因此 Matplotlib 会针对本段数据自动计算纵轴范围。
    axis.set_title(
        f"{title_prefix} — part {part_number}/{part_count} "
        f"(sorted samples {start_index + 1}–{end_index})"
    )
    axis.set_xlabel("Sorted sample index")
    axis.set_ylabel("Drag")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(loc="best")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def generate_split_plots(
    input_path: Path,
    cuts: Sequence[float],
    output_dir: Path | None = None,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """读取、排序并切分 prediction CSV，随后生成所有对比图。

    Args:
        input_path: prediction CSV 文件路径。
        cuts: 严格递增的累计切分比例。
        output_dir: 图片输出目录；传入 ``None`` 时使用自动生成的默认目录。
        title: 图片标题前缀；传入 ``None`` 时根据 CSV 文件名自动生成。
        dpi: 图片分辨率，必须为正整数。

    Returns:
        list[Path]: 按分段顺序排列的所有 PNG 输出路径。

    Raises:
        ValueError: DPI 无效，或者输入 CSV/切分比例不符合要求。
    """

    if dpi <= 0:
        raise ValueError(f"dpi 必须为正整数，当前值为：{dpi}")

    resolved_input = input_path.expanduser().resolve()
    rows = read_prediction_csv(resolved_input)

    # 调用训练模块中的同一个排序函数，避免训练图和切分图出现排序规则漂移。
    sorted_rows = sort_drag_predictions(rows)
    boundaries = calculate_boundaries(len(sorted_rows), cuts)

    resolved_output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else default_output_directory(resolved_input)
    )
    title_prefix = title if title is not None else default_title(resolved_input)
    part_count = len(boundaries) - 1
    output_paths: list[Path] = []

    # 相邻边界构成一个分段；所有分段互不重叠且合起来恰好覆盖全部 sample。
    for part_index, (start_index, end_index) in enumerate(
        zip(boundaries, boundaries[1:]),
        start=1,
    ):
        output_path = resolved_output_dir / (
            f"{resolved_input.stem}_drag_comparison_part_{part_index:02d}.png"
        )
        plot_segment(
            sorted_rows[start_index:end_index],
            start_index=start_index,
            end_index=end_index,
            part_number=part_index,
            part_count=part_count,
            output_path=output_path,
            title_prefix=title_prefix,
            dpi=dpi,
        )
        output_paths.append(output_path)

    return output_paths


def main(argv: Sequence[str] | None = None) -> int:
    """执行命令行入口，并向终端报告每张图片的输出位置。

    Args:
        argv: 可选的命令行参数序列；传入 ``None`` 时读取系统命令行。

    Returns:
        int: 成功返回 0；发生输入、数据或绘图错误时返回 1。
    """

    args = parse_args(argv)
    try:
        output_paths = generate_split_plots(
            input_path=args.input_path,
            cuts=args.cuts,
            output_dir=args.output_dir,
            title=args.title,
            dpi=args.dpi,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1

    print(f"已生成 {len(output_paths)} 张阻力对比图：")
    for output_path in output_paths:
        print(f"- {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
