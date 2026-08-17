"""HydroTransformer 的 Pre-LN 残差块。"""

from typing import Optional, Tuple, Union

from torch import Tensor, nn

from .attention import HydroMultiHeadAttention
from .conditional_norm import ConditionalLayerNorm


class HydroTransformerBlock(nn.Module):
    """组合条件 LayerNorm、HydroAttention 和标准前馈网络。

    所有归一化都放在子层之前，因此这是 Pre-LN（预归一化）结构。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        condition_dim: int,
        relative_hidden_dim: int = 64,
        use_rope: bool = True,
        use_relative_value: bool = True,
        use_conditional_layernorm: bool = True,
        condition_value_on_global: bool = True,
        condition_relative_value_on_global: bool = True,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.norm1 = ConditionalLayerNorm(
            d_model, condition_dim, enabled=use_conditional_layernorm
        )
        self.attention = HydroMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            condition_dim=condition_dim,
            relative_hidden_dim=relative_hidden_dim,
            use_rope=use_rope,
            use_relative_value=use_relative_value,
            condition_value_on_global=condition_value_on_global,
            condition_relative_value_on_global=condition_relative_value_on_global,
            rope_base=rope_base,
        )
        self.norm2 = ConditionalLayerNorm(
            d_model, condition_dim, enabled=use_conditional_layernorm
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        condition: Tensor,
        plant_mask: Tensor,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """更新植物 token，并可选返回本层 attention 权重。"""

        normalized = self.norm1(hidden_states, condition)
        attention_result = self.attention(
            normalized,
            positions,
            condition,
            plant_mask,
            return_attention=return_attention,
        )
        if return_attention:
            attention_output, attention_weights = attention_result
        else:
            attention_output = attention_result
            attention_weights = None

        hidden_states = hidden_states + self.residual_dropout(attention_output)
        hidden_states = hidden_states * plant_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        normalized = self.norm2(hidden_states, condition)
        hidden_states = hidden_states + self.feed_forward(normalized)
        hidden_states = hidden_states * plant_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)

        if return_attention:
            return hidden_states, attention_weights
        return hidden_states

