"""训练与评估阶段使用的轻量可视化工具。

本模块只处理 ``predict_dataset`` 返回的逐样本预测字典，不依赖具体模型。
图表使用 matplotlib 的非交互 ``Agg`` 后端，因此也能在没有桌面的服务器上生成 PNG。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

# 必须在导入 pyplot 前选择非交互后端，防止服务器训练时尝试打开图形窗口。
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  # 后端必须先于 pyplot 配置。

from .metrics import EVALUATION_FLOW_SPEEDS, group_predictions_by_flow_speed


def _finite_number(row: Mapping[str, Any], field: str, row_index: int) -> float:
    """读取并校验一个用于排序或绘图的有限数值。

    参数:
        row: 单条预测记录，通常来自 ``PredictionResult.predictions``。
        field: 要读取的字段名称。
        row_index: 该记录在原始输入中的位置，仅用于生成易定位的错误信息。

    返回:
        转换为 Python ``float`` 的有限数值。

    异常:
        ValueError: 字段缺失、不能转换为数值，或数值为 NaN/无穷大时抛出。
    """

    if field not in row:
        raise ValueError(f"第 {row_index} 条预测缺少必需字段：{field}")

    # float() 同时兼容 Python 数值、NumPy 标量及零维 Tensor 等常见结果类型。
    try:
        value = float(row[field])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"第 {row_index} 条预测的 {field} 必须是数值，实际为 {row[field]!r}"
        ) from error

    if not math.isfinite(value):
        raise ValueError(
            f"第 {row_index} 条预测的 {field} 必须是有限数值，实际为 {value!r}"
        )
    return value


def sort_drag_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """按 isolated drag、target drag 和 source index 排序预测记录。

    排序优先级依次为：

    1. ``isolated_drag`` 从小到大；
    2. ``isolated_drag`` 相同时，``target_drag`` 从小到大；
    3. 前两项都相同时，``source_index`` 从小到大；
    4. 三项仍完全相同时，保留输入中的原始先后顺序。

    参数:
        predictions: ``predict_dataset`` 返回的 ``predictions`` 行列表。

    返回:
        排序后的新字典列表。函数不会修改输入列表或其中的原始字典。

    异常:
        ValueError: 输入为空、字段缺失，或所需字段不是有限数值时抛出。
    """

    if not predictions:
        raise ValueError("无法绘制 drag 对比图：predictions 不能为空。")

    # 将已校验的排序键和记录放在一起，避免排序时反复转换同一个字段。
    validated_rows: list[tuple[float, float, float, int, dict[str, Any]]] = []
    for original_index, row in enumerate(predictions):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"第 {original_index} 条预测必须是字段字典，实际为 {type(row).__name__}"
            )

        # 除排序字段外也提前校验 predicted_drag，确保画图阶段不会得到半成品。
        isolated_drag = _finite_number(row, "isolated_drag", original_index)
        target_drag = _finite_number(row, "target_drag", original_index)
        source_index = _finite_number(row, "source_index", original_index)
        _finite_number(row, "predicted_drag", original_index)

        validated_rows.append(
            (
                isolated_drag,
                target_drag,
                source_index,
                original_index,
                dict(row),
            )
        )

    # Python 的排序本身是稳定的；original_index 让完全相同记录的规则更加明确。
    validated_rows.sort(key=lambda item: item[:4])
    return [item[4] for item in validated_rows]


def write_drag_comparison_plot(
    predictions: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Drag prediction comparison",
) -> Path:
    """把 target、predicted 与 isolated drag 绘制为三条折线并保存 PNG。

    参数:
        predictions: ``predict_dataset`` 返回的逐样本预测行列表。
        output_path: PNG 输出路径；父目录不存在时会自动创建。
        title: 图表标题，默认使用英文标题以避免服务器缺少中文字体。

    返回:
        已完成写入的 ``Path`` 对象。

    异常:
        ValueError: 预测为空、字段缺失/非法，或输出扩展名不是 ``.png`` 时抛出。
    """

    destination = Path(output_path)
    if destination.suffix.lower() != ".png":
        raise ValueError(f"drag 对比图必须输出为 .png 文件：{destination}")

    sorted_rows = sort_drag_predictions(predictions)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # 横坐标从 1 开始，使图中的 sample 序号符合日常计数习惯。
    sample_numbers = list(range(1, len(sorted_rows) + 1))
    target_drag = [float(row["target_drag"]) for row in sorted_rows]
    predicted_drag = [float(row["predicted_drag"]) for row in sorted_rows]
    isolated_drag = [float(row["isolated_drag"]) for row in sorted_rows]

    # 宽度随样本量缓慢增长并设置上限，兼顾少量点的可读性与大量点的文件大小。
    figure_width = min(18.0, max(10.0, len(sorted_rows) * 0.08))
    figure, axis = plt.subplots(figsize=(figure_width, 6.0))
    try:
        # marker 确保每条线确实显示 N 个 sample 点；较小尺寸避免点多时互相遮挡。
        axis.plot(
            sample_numbers,
            target_drag,
            label="target_drag",
            linewidth=1.4,
            marker="o",
            markersize=2.5,
        )
        axis.plot(
            sample_numbers,
            predicted_drag,
            label="predicted_drag",
            linewidth=1.4,
            marker="o",
            markersize=2.5,
        )
        axis.plot(
            sample_numbers,
            isolated_drag,
            label="isolated_drag",
            linewidth=1.4,
            marker="o",
            markersize=2.5,
        )

        axis.set_title(title)
        axis.set_xlabel("Sorted sample index")
        axis.set_ylabel("Drag")
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination, dpi=180, format="png")
    finally:
        # 主动释放 Figure，防止交叉验证循环中累计大量 matplotlib 对象占用内存。
        plt.close(figure)

    return destination


def _draw_drag_lines(
    axis: Any,
    sorted_rows: Sequence[Mapping[str, Any]],
) -> None:
    """在指定 matplotlib 坐标轴上绘制三条阻力折线。

    参数:
        axis: matplotlib 的 ``Axes`` 对象，负责承载当前子图。
        sorted_rows: 已由 ``sort_drag_predictions`` 排序并校验的预测记录。
    """

    # 横坐标从 1 开始，使每个速度面板中的 sample 编号符合日常计数习惯。
    sample_numbers = list(range(1, len(sorted_rows) + 1))
    line_settings = {
        "linewidth": 1.3,
        "marker": "o",
        "markersize": 2.4,
    }
    axis.plot(
        sample_numbers,
        [float(row["target_drag"]) for row in sorted_rows],
        label="target_drag",
        **line_settings,
    )
    axis.plot(
        sample_numbers,
        [float(row["predicted_drag"]) for row in sorted_rows],
        label="predicted_drag",
        **line_settings,
    )
    axis.plot(
        sample_numbers,
        [float(row["isolated_drag"]) for row in sorted_rows],
        label="isolated_drag",
        **line_settings,
    )


def write_drag_comparison_by_flow_speed_plot(
    predictions: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Drag prediction comparison by flow speed",
) -> Path:
    """把四档流速的真实、预测和单株叠加阻力绘制为 2×2 子图。

    参数:
        predictions: ``predict_dataset`` 返回的逐样本预测记录；除总体图字段外，
            每条记录还必须包含 ``flow_speed``。
        output_path: PNG 输出路径；父目录不存在时会自动创建。
        title: 整张 2×2 图的总标题。

    返回:
        已完成写入的 ``Path`` 对象。

    说明:
        每个面板内部继续按照 isolated drag、target drag、source index 排序。
        某个 fold 缺少某档速度时仍保留对应面板，并明确显示 ``No samples``，
        从而保证四张子图的位置始终是 0.1、0.2、0.3、0.4 m/s。
    """

    destination = Path(output_path)
    if destination.suffix.lower() != ".png":
        raise ValueError(f"按流速 drag 对比图必须输出为 .png 文件：{destination}")

    grouped_rows = group_predictions_by_flow_speed(predictions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(16.0, 10.0))
    try:
        for axis, speed in zip(axes.flat, EVALUATION_FLOW_SPEEDS):
            rows = grouped_rows[speed]
            axis.set_title(f"{speed:.1f} m/s (n={len(rows)})")
            axis.set_xlabel("Sorted sample index")
            axis.set_ylabel("Drag")
            axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)

            if rows:
                sorted_rows = sort_drag_predictions(rows)
                _draw_drag_lines(axis, sorted_rows)
                axis.legend()
            else:
                # 空面板仍清楚表明该 fold 中没有对应速度，而不是静默漏画。
                axis.text(
                    0.5,
                    0.5,
                    "No samples",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )

        figure.suptitle(title)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        figure.savefig(destination, dpi=180, format="png")
    finally:
        # 交叉验证会连续生成多张图，必须及时释放内存中的 Figure。
        plt.close(figure)
    return destination


__all__ = [
    "sort_drag_predictions",
    "write_drag_comparison_by_flow_speed_plot",
    "write_drag_comparison_plot",
]
