"""面向二维水草耦合问题的多头自注意力。"""

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .conditional_norm import FeatureWiseLinearModulation
from .relative_geometry import RelativeGeometryEncoder
from .rope_2d import RotaryPositionEmbedding2D


# Directional attention 的合法模式和默认参数集中定义在模块开头，避免模型、
# Transformer block 和训练配置各自硬编码不同的字符串或数值。
DIRECTIONAL_ATTENTION_MODES = frozenset({"none", "soft", "hard"})
DIRECTIONAL_SOFT_ACTIVATIONS = frozenset({"relu"})
DEFAULT_DIRECTIONAL_ATTENTION_MODE = "none"
DEFAULT_DIRECTIONAL_SOFT_ACTIVATION = "relu"
DEFAULT_DIRECTIONAL_SOFT_STRENGTH = 1.0
DEFAULT_DIRECTIONAL_TOLERANCE = 1.0e-6


def compute_downstream_offsets(positions: Tensor) -> Tensor:
    """计算每个 target-source 对的下游偏移量 ``x_i-x_j``。

    参数:
        positions: 植物二维坐标，形状为 ``[B,N,2]``。本项目约定 ``x`` 越大越
            靠上游，因此水流从 ``x+`` 一侧流向 ``x-`` 一侧。

    返回:
        ``[B,N_target,N_source]``。正值表示 source ``j`` 位于 target ``i`` 的
        下游，负值表示 source 位于上游，零表示同一横截面或 self-edge。
    """

    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions 必须是 [B,N,2]，才能计算上下游关系。")
    x_coordinates = positions[..., 0]
    target_x = x_coordinates.unsqueeze(-1)
    source_x = x_coordinates.unsqueeze(-2)
    return target_x - source_x


def compute_directional_soft_penalty(
    downstream_offsets: Tensor,
    activation: str = DEFAULT_DIRECTIONAL_SOFT_ACTIVATION,
    tolerance: float = DEFAULT_DIRECTIONAL_TOLERANCE,
) -> Tensor:
    """把下游偏移转换为可从 attention logits 中扣除的非负惩罚。

    参数:
        downstream_offsets: ``[B,N_target,N_source]``，通常来自
            :func:`compute_downstream_offsets`。
        activation: 惩罚函数名称。第一版只支持 ``"relu"``；显式保留该参数，
            使未来增加其他函数时能够继续复现旧 checkpoint 的 ReLU 语义。
        tolerance: 非负坐标容差。偏移不超过该值时不施加惩罚。

    返回:
        与 ``downstream_offsets`` 同形状的非负张量。当前计算公式为
        ``ReLU(downstream_offsets - tolerance)``。
    """

    if activation not in DIRECTIONAL_SOFT_ACTIVATIONS:
        raise ValueError(
            "directional_soft_activation 必须是 "
            f"{sorted(DIRECTIONAL_SOFT_ACTIVATIONS)}，当前收到 {activation!r}。"
        )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("directional_tolerance 必须是有限非负数。")

    shifted_offsets = downstream_offsets - tolerance
    if activation == "relu":
        return torch.relu(shifted_offsets)
    # 上方集合校验保证当前分支不可达；保留断言可在未来扩展集合时防止漏实现。
    raise AssertionError(f"尚未实现 directional soft activation: {activation!r}")


def _safe_masked_softmax(
    scores: Tensor,
    source_mask: Tensor,
    target_mask: Tensor,
    pair_mask: Optional[Tensor] = None,
) -> Tensor:
    """执行不会在“整行都被 mask”时产生 NaN 的 masked softmax。

    标准 ``softmax([-inf, ...])`` 会产生 NaN。这里先使用有限最小值，再将非法位置
    归零并重新归一化；若样本没有任何有效 source，最终 attention 保持全零。
    """

    expanded_source_mask = source_mask[:, None, None, :]
    expanded_valid_mask = expanded_source_mask
    if pair_mask is not None:
        expected_shape = (scores.shape[0], scores.shape[-2], scores.shape[-1])
        if pair_mask.shape != expected_shape or pair_mask.dtype != torch.bool:
            raise ValueError(
                "pair_mask 必须是与 attention target/source 轴对齐的 bool 张量 "
                f"{expected_shape}。"
            )
        expanded_valid_mask = expanded_valid_mask & pair_mask[:, None, :, :]

    scores = scores.masked_fill(~expanded_valid_mask, torch.finfo(scores.dtype).min)
    attention = torch.softmax(scores, dim=-1)
    attention = attention * expanded_valid_mask.to(dtype=attention.dtype)
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
        directional_attention_mode: ``none`` 保持双向 attention；``soft`` 对下游
            source 施加连续惩罚；``hard`` 完全屏蔽下游 source。
        directional_soft_activation: soft 惩罚函数名称；当前只支持 ``relu``。
        directional_soft_strength: soft 惩罚强度；0 表示不改变 attention scores。
        directional_tolerance: 判断同一横截面和上下游关系时使用的非负坐标容差。
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
        directional_attention_mode: str = DEFAULT_DIRECTIONAL_ATTENTION_MODE,
        directional_soft_activation: str = DEFAULT_DIRECTIONAL_SOFT_ACTIVATION,
        directional_soft_strength: float = DEFAULT_DIRECTIONAL_SOFT_STRENGTH,
        directional_tolerance: float = DEFAULT_DIRECTIONAL_TOLERANCE,
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
        if directional_attention_mode not in DIRECTIONAL_ATTENTION_MODES:
            raise ValueError(
                "directional_attention_mode 必须是 "
                f"{sorted(DIRECTIONAL_ATTENTION_MODES)}，"
                f"当前收到 {directional_attention_mode!r}。"
            )
        if directional_soft_activation not in DIRECTIONAL_SOFT_ACTIVATIONS:
            raise ValueError(
                "directional_soft_activation 必须是 "
                f"{sorted(DIRECTIONAL_SOFT_ACTIVATIONS)}，"
                f"当前收到 {directional_soft_activation!r}。"
            )
        if not math.isfinite(directional_soft_strength) or directional_soft_strength < 0.0:
            raise ValueError("directional_soft_strength 必须是有限非负数。")
        if not math.isfinite(directional_tolerance) or directional_tolerance < 0.0:
            raise ValueError("directional_tolerance 必须是有限非负数。")

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.rope = RotaryPositionEmbedding2D(self.head_dim, base=rope_base, enabled=use_rope)
        self.condition_value_on_global = condition_value_on_global
        self.directional_attention_mode = directional_attention_mode
        self.directional_soft_activation = directional_soft_activation
        self.directional_soft_strength = directional_soft_strength
        self.directional_tolerance = directional_tolerance
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

        # 正值代表 source 位于 target 下游。soft 模式按距离降低该方向的 logit；
        # hard 模式则完全禁止该消息边。self-edge 和同一 x 横截面的边均被保留。
        directional_pair_mask: Optional[Tensor] = None
        if self.directional_attention_mode != "none":
            downstream_offsets = compute_downstream_offsets(positions)
            if self.directional_attention_mode == "soft":
                # strength=0 时直接旁路，保证 soft(0) 与 none 逐元素一致。
                if self.directional_soft_strength > 0.0:
                    soft_penalty = compute_directional_soft_penalty(
                        downstream_offsets,
                        activation=self.directional_soft_activation,
                        tolerance=self.directional_tolerance,
                    ).to(dtype=scores.dtype)
                    scores = scores - self.directional_soft_strength * soft_penalty[:, None, :, :]
            elif self.directional_attention_mode == "hard":
                directional_pair_mask = downstream_offsets <= self.directional_tolerance

        attention = _safe_masked_softmax(
            scores,
            plant_mask,
            plant_mask,
            pair_mask=directional_pair_mask,
        )
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
