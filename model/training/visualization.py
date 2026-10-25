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
import numpy as np

# 必须在导入 pyplot 前选择非交互后端，防止服务器训练时尝试打开图形窗口。
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  # 后端必须先于 pyplot 配置。

from .config import DEFAULT_DRAG_PARITY_AXIS_MODE, DRAG_PARITY_AXIS_MODES
from .metrics import EVALUATION_FLOW_SPEEDS, group_predictions_by_flow_speed


# 散点图的默认参数集中在这里，便于统一调整论文图片的尺寸、配色和统计格式。
DEFAULT_PARITY_PREFIX = "test"
DEFAULT_PARITY_TITLE = "Test-set drag parity"
PARITY_MEASUREMENT_FIELD = "raw_target_drag"  # 原始实测值，保留训练归零前的负测量。
PARITY_FIGURE_SIZE = (6.6, 6.6)  # 每张图的宽、高，单位 inch。
PARITY_DPI = 300  # PNG 输出分辨率，单位 dots per inch。
PARITY_AXIS_PADDING = 0.05  # 在当前模式确定的数据范围两端各增加 5% 留白。
PARITY_AXIS_NOTE_POSITION = (0.5, -0.16)  # 范围说明位于横轴标签下方，使用轴相对坐标。
PARITY_AXIS_NOTE_FONT_SIZE = 8
PARITY_AXIS_NOTES = {
    "independent": "Axis range: independent for each plot",
    "shared": "Axis range: shared between both plots",
}
PARITY_ZERO_SPAN = 1.0  # 所有数值均为零时，用于计算非零坐标留白的参考跨度。
PARITY_POINT_SIZE = 26.0  # 散点面积，单位 point²。
PARITY_POINT_ALPHA = 0.65  # 散点透明度，便于观察重叠样本。
PARITY_REFERENCE_COLOR = "#555555"
PARITY_REFERENCE_LINEWIDTH = 1.1
PARITY_GRID_ALPHA = 0.25
PARITY_GRID_LINEWIDTH = 0.6
PARITY_STATS_POSITIONS = {
    # 每项为坐标轴内的相对位置和水平对齐方式；基线框放右侧，避开左上方极值。
    "model": ((0.04, 0.96), "left"),
    "baseline": ((0.96, 0.96), "right"),
}
PARITY_STATS_FONT_SIZE = 10
PARITY_STATS_BOX = {"facecolor": "white", "alpha": 0.85, "edgecolor": "none"}
PARITY_METRIC_SIGNIFICANT_DIGITS = 4
PARITY_R2_DECIMAL_PLACES = 3
PARITY_IDENTITY_LABEL = "1:1 reference"
PARITY_MEASUREMENT_LABEL = "Measured drag (N)"
PARITY_SERIES = (
    # 每项依次为文件后缀、预测表字段、标题、纵轴标签和色盲友好配色。
    ("model", "predicted_drag", "Model prediction", "Predicted drag (N)", "#0072B2"),
    (
        "baseline", "isolated_drag", "Summed single-plant baseline",
        "Summed single-plant drag (N)", "#D55E00",
    ),
)


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


def write_drag_parity_plots(
    predictions: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    prefix: str = DEFAULT_PARITY_PREFIX,
    title: str = DEFAULT_PARITY_TITLE,
    axis_mode: str = DEFAULT_DRAG_PARITY_AXIS_MODE,
) -> dict[str, Path]:
    """为同一批测试记录分别保存模型和逐株叠加基线的预测—测量散点图。

    参数:
        predictions: 完整测试集的逐样本预测记录；每条需包含原始实测
            ``raw_target_drag``、模型预测 ``predicted_drag`` 和逐株力之和
            ``isolated_drag``。每条记录在每张图中各对应一个点，不排序或抽样。
        output_dir: 两张 PNG 的保存目录，不存在时自动创建。
        prefix: 文件名前缀；输出 ``{prefix}_drag_parity_model.png`` 与
            ``{prefix}_drag_parity_baseline.png``。
        title: 两张图共用的标题前缀，后接模型或基线名称；英文标签便于论文排版，
            也避免无中文字体的服务器出现缺字。
        axis_mode: ``independent`` 按每张图的实测值与该图预测值独立取范围；
            ``shared`` 按实测值与两组预测值统一取范围。两种模式均保留全部数据，
            每张图内部的横纵轴范围始终相同，且每单位长度代表相同的 N 数值。

    返回:
        ``model``、``baseline`` 分别对应已生成 PNG 路径的字典。

    说明:
        横轴保留原始实验测量，包括训练时被归零的负测量；图内 MAE、RMSE、
        R² 也使用相同原始测量口径，不直接复用基于 ``target_drag`` 的训练指标。
        默认两图分别取范围，防止极大的基线预测压缩模型散点；图下注明范围模式，
        跨图比较绝对误差时应读取坐标刻度及 MAE、RMSE。R² 未定义时显示 undefined。

    异常:
        ValueError: 坐标模式无效、输入为空、记录不是映射，或必需数值缺失/非有限时抛出。
    """

    if not predictions:
        raise ValueError("无法绘制 drag 散点图：predictions 不能为空。")
    if axis_mode not in DRAG_PARITY_AXIS_MODES:
        raise ValueError("axis_mode 必须是 independent 或 shared。")

    # 一次性校验两张图要使用的全部数据，防止第二张图才发现输入错误。
    fields = (PARITY_MEASUREMENT_FIELD, *(series[1] for series in PARITY_SERIES))
    values: dict[str, list[float]] = {field: [] for field in fields}
    for row_index, row in enumerate(predictions):
        if not isinstance(row, Mapping):
            raise ValueError(f"第 {row_index} 条预测必须是字段字典，实际为 {type(row).__name__}")
        for field in fields:
            values[field].append(_finite_number(row, field, row_index))

    # 使用 float64 计算图内指标；不修改预测表，也不改变训练标签的归零策略。
    measured = np.asarray(values[PARITY_MEASUREMENT_FIELD], dtype=np.float64)
    # 只有 shared 模式才合并两组预测；独立模式下基线极值不会参与模型图的范围计算。
    shared_values = (
        np.concatenate([np.asarray(values[field], dtype=np.float64) for field in fields])
        if axis_mode == "shared" else None
    )
    centered_measurement = measured - measured.mean()
    total_sum_squares = float(np.sum(centered_measurement**2))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for name, field, series_title, ylabel, color in PARITY_SERIES:
        # 两张图分别计算各自的误差；常量实测值或单样本没有可定义的 R²。
        predicted = np.asarray(values[field], dtype=np.float64)
        # 每张图的 x、y 同时覆盖本图全部数据与零点，再在两端留白；不裁剪离群点。
        range_values = (
            shared_values if shared_values is not None
            else np.concatenate((measured, predicted))
        )
        lower = min(0.0, float(range_values.min()))
        upper = max(0.0, float(range_values.max()))
        span = upper - lower if upper > lower else PARITY_ZERO_SPAN
        padding = span * PARITY_AXIS_PADDING
        limits = (lower - padding, upper + padding)
        error = predicted - measured
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        r2 = 1.0 - float(np.sum(error**2)) / total_sum_squares if total_sum_squares > 0.0 else math.nan
        r2_label = f"{r2:.{PARITY_R2_DECIMAL_PLACES}f}" if math.isfinite(r2) else "undefined"
        statistics = (
            f"n = {len(measured)}\n"
            f"MAE = {mae:.{PARITY_METRIC_SIGNIFICANT_DIGITS}g} N\n"
            f"RMSE = {rmse:.{PARITY_METRIC_SIGNIFICANT_DIGITS}g} N\n"
            f"$R^2$ = {r2_label}"
        )
        output_path = destination / f"{prefix}_drag_parity_{name}.png"
        figure, axis = plt.subplots(figsize=PARITY_FIGURE_SIZE)
        try:
            # 原始每条测试记录都保留；透明散点允许读者看到密集重叠区域。
            axis.scatter(
                measured, predicted, s=PARITY_POINT_SIZE,
                alpha=PARITY_POINT_ALPHA, color=color, edgecolors="none",
            )
            axis.plot(
                limits, limits, linestyle="--", color=PARITY_REFERENCE_COLOR,
                linewidth=PARITY_REFERENCE_LINEWIDTH, label=PARITY_IDENTITY_LABEL,
            )
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel(PARITY_MEASUREMENT_LABEL)
            axis.set_ylabel(ylabel)
            axis.set_title(f"{title}\n{series_title}")
            axis.grid(True, linewidth=PARITY_GRID_LINEWIDTH, alpha=PARITY_GRID_ALPHA)
            axis.set_axisbelow(True)
            # 模型统计框在左上，基线统计框在右上，避免基线高估点被文字框覆盖。
            statistics_position, statistics_alignment = PARITY_STATS_POSITIONS[name]
            axis.text(
                *statistics_position, statistics, transform=axis.transAxes,
                ha=statistics_alignment, va="top", fontsize=PARITY_STATS_FONT_SIZE,
                bbox=PARITY_STATS_BOX,
            )
            axis.legend(loc="lower right")
            # 明示两图是否共用范围，提醒读者结合刻度和误差指标比较模型与基线。
            axis.text(
                *PARITY_AXIS_NOTE_POSITION, PARITY_AXIS_NOTES[axis_mode],
                transform=axis.transAxes, ha="center", va="top",
                fontsize=PARITY_AXIS_NOTE_FONT_SIZE, color=PARITY_REFERENCE_COLOR,
            )
            figure.tight_layout()
            figure.savefig(output_path, dpi=PARITY_DPI, format="png")
        finally:
            # 正常写图或写入失败时都释放 Figure，避免连续训练时积累图形对象。
            plt.close(figure)
        output_paths[name] = output_path

    return output_paths


__all__ = [
    "sort_drag_predictions",
    "write_drag_comparison_by_flow_speed_plot",
    "write_drag_comparison_plot",
    "write_drag_parity_plots",
]
