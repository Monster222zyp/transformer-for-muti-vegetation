"""面向二维水草耦合问题的多头自注意力。"""

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .conditional_norm import FeatureWiseLinearModulation
from .relative_geometry import RelativeGeometryEncoder
from .rope_2d import RotaryPositionEmbedding2D


def _safe_masked_softmax(scores: Tensor, source_mask: Tensor, target_mask: Tensor) -> Tensor:
    """执行不会在“整行都被 mask”时产生 NaN 的 masked softmax。

    标准 ``softmax([-inf, ...])`` 会产生 NaN。这里先使用有限最小值，再将非法位置
    归零并重新归一化；若样本没有任何有效 source，最终 attention 保持全零。
    """

    expanded_source_mask = source_mask[:, None, None, :]
    scores = scores.masked_fill(~expanded_source_mask, torch.finfo(scores.dtype).min)
    attention = torch.softmax(scores, dim=-1)
    attention = attention * expanded_source_mask.to(dtype=attention.dtype)
    denominator = attention.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).tiny)
    attention = attention / denominator
    return attention * target_mask[:, None, :, None].to(dtype=attention.dtype)


class HydroMultiHeadAttention(nn.Module):
    """标准 QK softmax 加上 pair-dependent relative Value 的 attention。

    参数:
        d_model: token 隐藏维度。
        n_heads: attention head 数量。
        dropout: attention 权重与输出投影后的 dropout 概率。
        condition_dim: 全局条件向量维度。
        relative_hidden_dim: 相对几何编码器的隐藏维度。
        use_rope: 是否启用 2D RoPE。
        use_relative_value: 是否启用 relative Value。
        condition_value_on_global: 是否用全局量 FiLM 调制普通 Value。
        condition_relative_value_on_global: 是否用全局量 FiLM 调制 relative Value。
        rope_base: RoPE 频率底数。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        condition_dim: int,
        relative_hidden_dim: int = 64,
        use_rope: bool = True,
        use_relative_value: bool = True,
        condition_value_on_global: bool = True,
        condition_relative_value_on_global: bool = True,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除。")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # 只有 2D RoPE 需要将单 head 切成 x/y 两半并组成旋转对。
        # 关闭 RoPE 的消融模型仍是合法的标准 attention，不应受此约束。
        if use_rope and self.head_dim % 4 != 0:
            raise ValueError("为了使用 2D RoPE，head_dim 必须能被 4 整除。")

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.rope = RotaryPositionEmbedding2D(self.head_dim, base=rope_base, enabled=use_rope)
        self.condition_value_on_global = condition_value_on_global
        # FiLM 始终实例化，开关仅在 forward 中旁路。由此，不同消融配置在同 seed 下
        # 消耗完全相同的初始化随机数，Q/K/V、FFN 等公共层可以逐元素公平比较。
        self.value_modulation = FeatureWiseLinearModulation(condition_dim, d_model)
        self.relative_geometry = RelativeGeometryEncoder(
            n_heads=n_heads,
            head_dim=self.head_dim,
            hidden_dim=relative_hidden_dim,
            condition_dim=condition_dim,
            enabled=use_relative_value,
            condition_on_global=condition_relative_value_on_global,
        )
        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, features: Tensor) -> Tensor:
        """将 ``[B,N,D]`` 转成 ``[B,H,N,Dh]``。"""

        batch_size, plant_count, _ = features.shape
        return features.reshape(batch_size, plant_count, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        condition: Tensor,
        plant_mask: Tensor,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """计算水草之间的消息传递。

        返回:
            默认返回 ``[B,N,D]``。当 ``return_attention=True`` 时，返回
            ``(output, attention)``，其中 attention 为 ``[B,H,N,N]``。
        """

        batch_size, plant_count, _ = hidden_states.shape
        query = self._split_heads(self.query_projection(hidden_states))
        key = self._split_heads(self.key_projection(hidden_states))
        raw_value = self.value_projection(hidden_states)
        if self.condition_value_on_global:
            raw_value = self.value_modulation(raw_value, condition)
        value = self._split_heads(raw_value)

        query, key = self.rope(query, key, positions)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = _safe_masked_softmax(scores, plant_mask, plant_mask)
        dropped_attention = self.attention_dropout(attention)

        # 内容项沿 source j 求和，不显式展开成巨大的 pair-dependent V 张量。
        content_message = torch.einsum("bhij,bhjd->bhid", dropped_attention, value)
        relative_value = self.relative_geometry(positions, condition)
        relative_message = torch.einsum("bhij,bhijd->bhid", dropped_attention, relative_value)
        message = content_message + relative_message

        merged = message.transpose(1, 2).reshape(batch_size, plant_count, self.d_model)
        # attention 子层只负责输出投影。残差分支的 dropout 统一由外层 block 执行，
        # 避免同一 attention message 在进入 residual 前连续经历两次 dropout。
        output = self.output_projection(merged)
        output = output * plant_mask.unsqueeze(-1).to(dtype=output.dtype)
        if return_attention:
            return output, attention
        return output
