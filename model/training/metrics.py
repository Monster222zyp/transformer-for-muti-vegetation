"""总阻力和相互作用系数的评估指标。"""

from __future__ import annotations

from typing import Any

import numpy as np


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
