"""二维旋转位置编码（2D RoPE）。"""

from typing import Tuple

import torch
from torch import Tensor, nn


def _apply_axis_rope(values: Tensor, coordinate: Tensor, base: float) -> Tensor:
    """给一个坐标轴对应的偶数维通道应用成对旋转。

    参数:
        values: ``[B, H, N, axis_dim]``，其中 ``axis_dim`` 必须是偶数。
        coordinate: ``[B, N]``，每个植物在当前轴上的无量纲坐标。
        base: RoPE 频率底数。

    返回:
        与 ``values`` 同形状的旋转结果。
    """

    axis_dim = values.shape[-1]
    # 每两个相邻通道组成一个二维旋转对，因此这里只为每一对计算一个角频率。
    pair_indices = torch.arange(0, axis_dim, 2, device=values.device, dtype=values.dtype)
    inverse_frequency = 1.0 / (base ** (pair_indices / axis_dim))
    angles = coordinate.to(dtype=values.dtype).unsqueeze(1).unsqueeze(-1) * inverse_frequency
    cosine = angles.cos()
    sine = angles.sin()

    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class RotaryPositionEmbedding2D(nn.Module):
    """将每个 attention head 的前半通道编码 x，后半通道编码 y。

    参数:
        head_dim: 单个 attention head 的维度；启用 RoPE 时必须能被 4 整除。
        base: 旋转频率底数，默认采用标准值 10000。
        enabled: 为 ``False`` 时原样返回 Q、K，用于 RoPE 消融。
    """

    def __init__(self, head_dim: int, base: float = 10_000.0, enabled: bool = True) -> None:
        super().__init__()
        # 关闭 RoPE 后不会拆分 x/y 通道和旋转对，因此不应限制 head_dim。
        # 这使不满足四倍数的普通 attention 配置也能用于无 RoPE 消融。
        if enabled and head_dim % 4 != 0:
            raise ValueError("2D RoPE 要求 head_dim 能被 4 整除。")
        self.head_dim = head_dim
        self.base = base
        self.enabled = enabled

    def forward(self, query: Tensor, key: Tensor, positions: Tensor) -> Tuple[Tensor, Tensor]:
        """对形状为 ``[B,H,N,Dh]`` 的 Q、K 应用二维 RoPE。"""

        if not self.enabled:
            return query, key

        half = self.head_dim // 2

        def rotate(values: Tensor) -> Tensor:
            x_part = _apply_axis_rope(values[..., :half], positions[..., 0], self.base)
            y_part = _apply_axis_rope(values[..., half:], positions[..., 1], self.base)
            return torch.cat((x_part, y_part), dim=-1)

        return rotate(query), rotate(key)
