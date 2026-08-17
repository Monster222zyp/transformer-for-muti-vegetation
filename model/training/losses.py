"""训练损失函数。

本项目用总阻力与孤立单株阻力之和的比值 ``C`` 训练，使不同植株数量的
样本在损失中具有接近的权重。
"""

from __future__ import annotations

import torch


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


def interaction_mse_loss(
    predicted_drag: torch.Tensor,
    target_drag: torch.Tensor,
    single_drag: torch.Tensor,
    plant_mask: torch.Tensor,
) -> torch.Tensor:
    """计算预测与标签的相互作用系数均方误差。

    Args:
        predicted_drag: 模型预测的总阻力，形状为 ``[B]``。
        target_drag: 已归零负值后的目标总阻力，形状为 ``[B]``。
        single_drag: 每株孤立阻力，形状为 ``[B, N]``。
        plant_mask: 有效植株掩码，形状为 ``[B, N]``。

    Returns:
        标量 MSE loss。
    """

    predicted_c = interaction_coefficient(
        predicted_drag, single_drag, plant_mask
    )
    target_c = interaction_coefficient(target_drag, single_drag, plant_mask)
    return torch.mean((predicted_c - target_c) ** 2)
