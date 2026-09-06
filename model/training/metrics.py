"""总阻力和相互作用系数的评估指标。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


# 评估数据固定使用实验中的四档流速。把硬编码集中放在模块开头，便于后续检查。
EVALUATION_FLOW_SPEEDS = (0.1, 0.2, 0.3, 0.4)
"""按流速汇总指标和绘图时支持的速度，单位为 m/s。"""

FLOW_SPEED_METRIC_MATCH_RTOL = 0.0
"""匹配流速时的相对容差；固定为零，避免容差随数值大小变化。"""

FLOW_SPEED_METRIC_MATCH_ATOL = 1.0e-8
"""匹配流速时的绝对容差，用于吸收 float32 转换产生的微小尾差。"""


def _as_1d_float(values: Any) -> np.ndarray:
    """把 Tensor、列表或数组转换为一维 ``float64`` NumPy 数组。"""

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def compute_regression_metrics(
    target_drag: Any,
    predicted_drag: Any,
    isolated_drag: Any,
    mape_threshold: float = 1.0e-6,
) -> dict[str, float]:
    """计算计划约定的阻力与相互作用系数指标。

    Args:
        target_drag: 已归零后的真实总阻力。
        predicted_drag: 模型预测的总阻力。
        isolated_drag: 每个样本的 ``sum(D_i^(0))``。
        mape_threshold: 仅当 ``target_drag`` 大于此值时计入 MAPE。

    Returns:
        包含 ``MAE_D``、``RMSE_D``、``R2``、``MAE_C``、``RMSE_C``、
        ``MAPE_D``、``MAPE_coverage`` 和 ``sMAPE_D`` 的字典。MAPE 没有
        有效样本时返回 ``NaN``，覆盖率仍返回零。
    """

    target = _as_1d_float(target_drag)
    predicted = _as_1d_float(predicted_drag)
    isolated = _as_1d_float(isolated_drag)
    if not (target.size == predicted.size == isolated.size):
        raise ValueError("target_drag、predicted_drag 与 isolated_drag 长度必须一致。")
    if target.size == 0:
        raise ValueError("无法计算空数据集的指标。")
    if np.any(isolated <= 0.0):
        raise ValueError("isolated_drag 必须全部大于零。")

    error = predicted - target
    target_c = target / isolated
    predicted_c = predicted / isolated
    error_c = predicted_c - target_c

    # 常量标签的总平方和为零，此时 R² 没有数学定义，显式记作 NaN。
    total_sum_squares = np.sum((target - target.mean()) ** 2)
    r2 = (
        float(1.0 - np.sum(error**2) / total_sum_squares)
        if total_sum_squares > 0.0
        else float("nan")
    )

    mape_mask = target > mape_threshold
    mape = (
        float(np.mean(np.abs(error[mape_mask] / target[mape_mask])) * 100.0)
        if np.any(mape_mask)
        else float("nan")
    )
    smape_denominator = np.abs(target) + np.abs(predicted)
    # 当标签与预测同时为零时，该项按零误差处理，而不是产生 0/0。
    smape_terms = np.divide(
        2.0 * np.abs(error),
        smape_denominator,
        out=np.zeros_like(error),
        where=smape_denominator > 0.0,
    )

    return {
        "MAE_D": float(np.mean(np.abs(error))),
        "RMSE_D": float(np.sqrt(np.mean(error**2))),
        "R2": r2,
        "MAE_C": float(np.mean(np.abs(error_c))),
        "RMSE_C": float(np.sqrt(np.mean(error_c**2))),
        "MAPE_D": mape,
        "MAPE_coverage": float(np.mean(mape_mask)),
        "sMAPE_D": float(np.mean(smape_terms) * 100.0),
    }


def match_evaluation_flow_speed(value: Any, row_index: int) -> float:
    """把预测记录中的流速匹配到固定的实验速度。

    参数:
        value: 预测记录里的 ``flow_speed`` 值。
        row_index: 该记录在输入中的位置，仅用于生成便于定位的错误信息。

    返回:
        ``0.1``、``0.2``、``0.3`` 或 ``0.4`` 中匹配到的标准 Python 浮点数。

    异常:
        ValueError: 流速无法转换为有限数值，或不属于项目支持的四档速度。
    """

    try:
        speed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"第 {row_index} 条预测的 flow_speed 必须是数值，实际为 {value!r}"
        ) from error
    if not math.isfinite(speed):
        raise ValueError(
            f"第 {row_index} 条预测的 flow_speed 必须是有限数值，实际为 {speed!r}"
        )

    for supported_speed in EVALUATION_FLOW_SPEEDS:
        if np.isclose(
            speed,
            supported_speed,
            rtol=FLOW_SPEED_METRIC_MATCH_RTOL,
            atol=FLOW_SPEED_METRIC_MATCH_ATOL,
        ):
            return supported_speed
    supported_text = ", ".join(str(item) for item in EVALUATION_FLOW_SPEEDS)
    raise ValueError(
        f"第 {row_index} 条预测的 flow_speed={speed!r} 不受支持；"
        f"仅支持 {supported_text} m/s。"
    )


def group_predictions_by_flow_speed(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[float, list[Mapping[str, Any]]]:
    """按四档实验流速对逐样本预测分桶，并保持每个桶内的原始顺序。

    参数:
        predictions: ``predict_dataset`` 返回的逐样本预测记录。

    返回:
        键为标准流速，值为该流速全部记录的字典。没有样本的速度仍保留空列表，
        方便 2×2 图固定显示四个面板。

    异常:
        ValueError: 输入为空、记录不是映射、缺少 ``flow_speed`` 或流速非法。
    """

    if not predictions:
        raise ValueError("无法按流速分组：predictions 不能为空。")

    grouped: dict[float, list[Mapping[str, Any]]] = {
        speed: [] for speed in EVALUATION_FLOW_SPEEDS
    }
    for row_index, row in enumerate(predictions):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"第 {row_index} 条预测必须是字段字典，实际为 {type(row).__name__}"
            )
        if "flow_speed" not in row:
            raise ValueError(f"第 {row_index} 条预测缺少必需字段：flow_speed")
        speed = match_evaluation_flow_speed(row["flow_speed"], row_index)
        grouped[speed].append(row)
    return grouped


def compute_metrics_by_flow_speed(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """按流速分别计算完整回归指标、样本数和真实阻力均值。

    参数:
        predictions: 逐样本预测记录。每条记录必须包含 ``flow_speed``、
            ``target_drag``、``predicted_drag`` 和 ``isolated_drag``。

    返回:
        按数值速度排序的嵌套字典。JSON 键固定为 ``"0.1"`` 等稳定字符串；
        没有样本的速度不会生成伪指标。每个速度包含 ``sample_count``、
        ``mean_target_D`` 以及 ``compute_regression_metrics`` 的全部八项指标。
    """

    grouped_rows = group_predictions_by_flow_speed(predictions)
    grouped_metrics: dict[str, dict[str, float | int]] = {}
    for speed, rows in grouped_rows.items():
        if not rows:
            continue

        # 先显式提取三组基础量，再交给统一指标函数，确保总体与分组口径一致。
        target_drag = [row["target_drag"] for row in rows]
        metrics = compute_regression_metrics(
            target_drag,
            [row["predicted_drag"] for row in rows],
            [row["isolated_drag"] for row in rows],
        )
        grouped_metrics[f"{speed:.1f}"] = {
            "sample_count": len(rows),
            "mean_target_D": float(np.mean(_as_1d_float(target_drag))),
            **metrics,
        }
    return grouped_metrics


__all__ = [
    "EVALUATION_FLOW_SPEEDS",
    "FLOW_SPEED_METRIC_MATCH_ATOL",
    "FLOW_SPEED_METRIC_MATCH_RTOL",
    "compute_metrics_by_flow_speed",
    "compute_regression_metrics",
    "group_predictions_by_flow_speed",
    "match_evaluation_flow_speed",
]
