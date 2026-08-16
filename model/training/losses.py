"""训练损失函数与相对误差分母拟合工具。

当前训练目标监督实验能够直接提供的最终总阻力 ``D``，但不再使用绝对误差。
每条样本先把“预测值与真实值之差”除以稳定化分母，再计算平方误差。这样较小
阻力样本不会因为物理量数值小而在 batch 平均值中失去作用。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# 即使训练集分位数意外非常接近零，分母也不能小于该数值安全边界。
MINIMUM_RELATIVE_FLOOR = 1.0e-8
# 默认只使用训练子集中的正标签，并取其 5% 分位数作为相对误差分母下限。
DEFAULT_RELATIVE_FLOOR_QUANTILE = 0.05
RELATIVE_FLOOR_STRATEGY = "train_positive_quantile"


def interaction_coefficient(
    total_drag: torch.Tensor,
    single_drag: torch.Tensor,
    plant_mask: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """计算总体相互作用系数 ``C = D_total / sum(D_i^(0))``。

    Args:
        total_drag: 每个样本的总阻力，形状为 ``[B]``。
        single_drag: 每株水草的孤立阻力，形状为 ``[B, N]``。
        plant_mask: 有效植株掩码，形状为 ``[B, N]``。
        epsilon: 防止分母为零的最小正数。

    Returns:
        每个样本的相互作用系数，形状为 ``[B]``。

    Raises:
        ValueError: 某个样本没有有效的孤立阻力时抛出。
    """

    # padding 的 single_drag 理论上已经是零；这里再次应用 mask，避免上游数据错误。
    isolated_drag = (single_drag * plant_mask.to(single_drag.dtype)).sum(dim=1)
    if torch.any(isolated_drag <= epsilon):
        raise ValueError("每个样本至少需要一株具有正 single_drag 的有效水草。")
    return total_drag / isolated_drag.clamp_min(epsilon)


def fit_relative_drag_floor(
    dataset: Dataset,
    indices: Sequence[int],
    quantile: float = DEFAULT_RELATIVE_FLOOR_QUANTILE,
    minimum: float = MINIMUM_RELATIVE_FLOOR,
) -> float:
    """只根据训练子集拟合稳定化相对误差的分母下限。

    Args:
        dataset: 能通过整数索引返回 ``target_drag`` 的完整数据集。
        indices: 当前训练子集索引。调用者不得传入 validation 或 test 索引。
        quantile: 正标签分位数，``0.05`` 表示 5% 分位数。
        minimum: 数值安全下限，防止拟合结果过小或等于零。

    Returns:
        大于零且有限的 Python ``float``，供训练、验证和 checkpoint 共用。

    Raises:
        ValueError: 分位数、minimum、索引集合或标签不合法，或者训练子集没有
            任何正标签时抛出。
    """

    quantile = float(quantile)
    minimum = float(minimum)
    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("relative loss 分位数必须是 [0, 1] 内的有限数。")
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("relative loss 的数值安全下限必须是有限正数。")

    # 逐条读取给定索引，确保统计量只来自当前训练子集，避免 validation/test 泄漏。
    positive_targets: list[float] = []
    observed_count = 0
    for index in indices:
        observed_count += 1
        target: Any = dataset[int(index)]["target_drag"]
        if torch.is_tensor(target):
            if target.numel() != 1:
                raise ValueError("dataset 中每条 target_drag 必须是标量。")
            target_value = float(target.detach().cpu().item())
        else:
            target_array = np.asarray(target, dtype=np.float64)
            if target_array.size != 1:
                raise ValueError("dataset 中每条 target_drag 必须是标量。")
            target_value = float(target_array.reshape(-1)[0])
        if not math.isfinite(target_value):
            raise ValueError("训练子集 target_drag 包含 NaN 或无穷大。")
        if target_value > 0.0:
            positive_targets.append(target_value)

    if observed_count == 0:
        raise ValueError("无法从空训练集合拟合 relative loss 分母下限。")
    if not positive_targets:
        raise ValueError("训练子集没有正 target_drag，无法拟合 relative loss 分母下限。")

    fitted_quantile = float(np.quantile(positive_targets, quantile))
    return max(fitted_quantile, minimum)


def relative_total_drag_mse_loss(
    predicted_drag: torch.Tensor,
    target_drag: torch.Tensor,
    relative_floor: float | torch.Tensor,
) -> torch.Tensor:
    """计算最终预测总阻力的稳定化目标相对均方误差。

    Args:
        predicted_drag: 模型预测的总阻力，形状为 ``[B]``。
        target_drag: 已归零负值后的目标总阻力，形状为 ``[B]``。
        relative_floor: 分母下限，必须是有限正标量。正式训练中应由
            :func:`fit_relative_drag_floor` 仅使用训练子集拟合。

    Returns:
        ``mean(((predicted-target)/max(abs(target), floor))**2)``，是一个
        保持梯度的标量张量。

    Raises:
        ValueError: 输入形状不一致、包含非有限值，或分母下限不是有限正标量时
            抛出，避免 broadcasting（广播）或除零悄悄产生错误 loss。
    """

    if predicted_drag.shape != target_drag.shape:
        raise ValueError(
            "predicted_drag 与 target_drag 必须具有相同形状；"
            f"实际分别为 {tuple(predicted_drag.shape)} 和 {tuple(target_drag.shape)}。"
        )
    if not torch.all(torch.isfinite(predicted_drag)):
        raise ValueError("predicted_drag 包含 NaN 或无穷大。")
    if not torch.all(torch.isfinite(target_drag)):
        raise ValueError("target_drag 包含 NaN 或无穷大。")

    # 把 Python float 或标量 Tensor 转到标签所在设备和 dtype，避免 CPU/CUDA 混用。
    floor = torch.as_tensor(
        relative_floor,
        dtype=target_drag.dtype,
        device=target_drag.device,
    )
    if floor.numel() != 1:
        raise ValueError("relative_floor 必须是标量。")
    if not bool(torch.isfinite(floor).item()) or float(floor.item()) <= 0.0:
        raise ValueError("relative_floor 必须是有限正数。")

    # 零标签和极小正标签统一使用训练集拟合的 floor，避免除零和极端梯度。
    denominator = torch.clamp(torch.abs(target_drag), min=floor)
    relative_error = (predicted_drag - target_drag) / denominator
    return torch.mean(relative_error**2)
