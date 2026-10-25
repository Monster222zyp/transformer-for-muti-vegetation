"""完整 HydroTransformer 模型。"""

from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor, nn

from .attention import (
    DEFAULT_DIRECTIONAL_ATTENTION_MODE,
    DEFAULT_DIRECTIONAL_SOFT_ACTIVATION,
    DEFAULT_DIRECTIONAL_SOFT_STRENGTH,
    DEFAULT_DIRECTIONAL_TOLERANCE,
)
from .coefficient_head import CoefficientHead
from .global_encoder import GlobalEncoder
from .plant_encoder import (
    DEFAULT_PLANT_ENCODER_HIDDEN_DIM, DEFAULT_STIFFNESS_REFERENCE_NM2, PlantEncoder,
)
from .transformer_block import HydroTransformerBlock


@dataclass(frozen=True)
class HydroTransformerConfig:
    """HydroTransformer 的可序列化配置。

    默认值对应项目确认的第一版：256 维、8 heads、4 blocks、1024 维 FFN。
    五个布尔开关允许独立进行 RoPE、relative Value 和三条全局条件路径的消融；
    directional attention 字段控制下游 source 到上游 target 的消息约束。
    """

    global_input_dim: int = 3
    global_hidden_dim: int = 64
    plant_encoder_hidden_dim: int = DEFAULT_PLANT_ENCODER_HIDDEN_DIM
    stiffness_reference_nm2: float = DEFAULT_STIFFNESS_REFERENCE_NM2
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.05
    relative_hidden_dim: int = 64
    coefficient_hidden_dim: int = 128
    rope_base: float = 10_000.0
    log_coefficient_min: float = -5.0
    log_coefficient_max: float = 5.0
    use_rope: bool = True
    use_relative_value: bool = True
    use_conditional_layernorm: bool = True
    condition_value_on_global: bool = True
    condition_relative_value_on_global: bool = True
    directional_attention_mode: str = DEFAULT_DIRECTIONAL_ATTENTION_MODE
    directional_soft_activation: str = DEFAULT_DIRECTIONAL_SOFT_ACTIVATION
    directional_soft_strength: float = DEFAULT_DIRECTIONAL_SOFT_STRENGTH
    directional_tolerance: float = DEFAULT_DIRECTIONAL_TOLERANCE


class HydroTransformer(nn.Module):
    """预测多水草总阻力和不可直接物理解读的逐株 latent coefficient。

    参数:
        config: ``HydroTransformerConfig``；省略时使用项目默认配置。
        **overrides: 按字段名覆盖配置，方便测试和命令行构造小模型。

    注意:
        若同时给出 ``config`` 和 ``overrides``，overrides 优先。这种接口既适合训练脚本
        从 YAML 展开参数，也适合直接传入一个已保存的配置对象。
    """

    def __init__(
        self,
        config: Optional[HydroTransformerConfig] = None,
        **overrides: Union[int, float, bool, str],
    ) -> None:
        super().__init__()
        base_values = asdict(config or HydroTransformerConfig())
        valid_names = {field.name for field in fields(HydroTransformerConfig)}
        unknown_names = set(overrides) - valid_names
        if unknown_names:
            raise TypeError(f"未知 HydroTransformer 配置项: {sorted(unknown_names)}")
        base_values.update(overrides)
        self.config = HydroTransformerConfig(**base_values)
        cfg = self.config

        if cfg.n_layers < 1:
            raise ValueError("n_layers 必须至少为 1。")
        if cfg.log_coefficient_min >= cfg.log_coefficient_max:
            raise ValueError("log_coefficient_min 必须小于 log_coefficient_max。")

        if cfg.global_input_dim != 3:
            raise ValueError("新模型 global_input_dim 必须为 3：[流速, sin角度, cos角度]。")
        # 角度和实测 EI 通过共享 MLP 得到逐株 Token；不再复制同一个可学习向量。
        self.plant_encoder = PlantEncoder(
            cfg.d_model, cfg.plant_encoder_hidden_dim, cfg.stiffness_reference_nm2
        )
        self.global_encoder = GlobalEncoder(
            cfg.global_input_dim, cfg.global_hidden_dim, cfg.d_model
        )
        self.blocks = nn.ModuleList(
            [
                HydroTransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ffn_dim=cfg.ffn_dim,
                    dropout=cfg.dropout,
                    condition_dim=cfg.d_model,
                    relative_hidden_dim=cfg.relative_hidden_dim,
                    use_rope=cfg.use_rope,
                    use_relative_value=cfg.use_relative_value,
                    use_conditional_layernorm=cfg.use_conditional_layernorm,
                    condition_value_on_global=cfg.condition_value_on_global,
                    condition_relative_value_on_global=cfg.condition_relative_value_on_global,
                    rope_base=cfg.rope_base,
                    directional_attention_mode=cfg.directional_attention_mode,
                    directional_soft_activation=cfg.directional_soft_activation,
                    directional_soft_strength=cfg.directional_soft_strength,
                    directional_tolerance=cfg.directional_tolerance,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.coefficient_head = CoefficientHead(cfg.d_model, cfg.coefficient_hidden_dim)

    def _validate_inputs(
        self,
        positions: Tensor,
        single_drag: Tensor,
        global_features: Tensor,
        plant_mask: Tensor,
    ) -> None:
        """在矩阵运算前报告容易理解的输入形状和类型错误。"""

        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("positions 必须是 [B,N,2]。")
        expected_plant_shape = positions.shape[:2]
        if single_drag.shape != expected_plant_shape:
            raise ValueError("single_drag 必须是 [B,N]，并与 positions 对齐。")
        if plant_mask.shape != expected_plant_shape or plant_mask.dtype != torch.bool:
            raise ValueError("plant_mask 必须是与 positions 对齐的 bool 张量 [B,N]。")
        if global_features.shape != (positions.shape[0], self.config.global_input_dim):
            raise ValueError("global_features 必须是 [B,3]，顺序为标准化流速、sin角度、cos角度。")
        if not torch.isfinite(global_features).all():
            raise ValueError("global_features 必须为有限数值。")

        if torch.any(single_drag[plant_mask] <= 0):
            raise ValueError("有效水草的 single_drag 必须大于 0。")
        if torch.any(single_drag[~plant_mask] != 0):
            raise ValueError("padding 位置的 single_drag 必须为 0。")

    def forward(
        self,
        positions: Tensor,
        single_drag: Tensor,
        global_features: Tensor,
        plant_mask: Tensor,
        return_attention: bool = False,
        *,
        plant_angles: Tensor | None = None,
        plant_stiffness: Tensor | None = None,
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """执行总阻力预测。

        参数:
            positions: ``[B,N,2]``，有效植物的二维无量纲坐标。
            single_drag: ``[B,N]``，孤立单株阻力；padding 位置应为 0。
            global_features: ``[B,3]``，标准化流速、共同角度的 sin 和 cos。
            plant_mask: ``[B,N]`` bool；``True`` 代表真实植物。
            return_attention: 是否额外返回每一层的 attention 权重。
            plant_angles: ``[B,N]``，每株的转盘角（degree），作为关键词传入。
            plant_stiffness: ``[B,N]``，每株实测 EI（N·m²），作为关键词传入。

        返回:
            字典包含 ``total_drag [B]``、``coefficient [B,N]`` 和
            ``log_coefficient [B,N]``。启用 ``return_attention`` 时还包含长度等于
            block 数的 ``attention`` 列表，每项形状为 ``[B,H,N,N]``。
        """

        self._validate_inputs(positions, single_drag, global_features, plant_mask)
        if plant_angles is None or plant_stiffness is None:
            raise ValueError("新模型需要 plant_angles 和 plant_stiffness，不能使用旧共享 Token 输入。")
        if plant_angles.shape != plant_mask.shape or plant_stiffness.shape != plant_mask.shape:
            raise ValueError("plant_angles 和 plant_stiffness 必须与 positions 对齐为 [B,N]。")
        mask_as_float = plant_mask.to(dtype=positions.dtype)

        # 起始向量直接包含每株角度和 EI；几何仍由位置注意力机制读取。
        hidden_states = self.plant_encoder(plant_angles, plant_stiffness, plant_mask)
        condition = self.global_encoder(global_features)

        collected_attention: List[Tensor] = []
        for block in self.blocks:
            if return_attention:
                hidden_states, attention = block(
                    hidden_states,
                    positions,
                    condition,
                    plant_mask,
                    return_attention=True,
                )
                collected_attention.append(attention)
            else:
                hidden_states = block(
                    hidden_states,
                    positions,
                    condition,
                    plant_mask,
                    return_attention=False,
                )

        raw_log_coefficient = self.coefficient_head(hidden_states)
        log_coefficient = raw_log_coefficient.clamp(
            min=self.config.log_coefficient_min,
            max=self.config.log_coefficient_max,
        )
        # padding 的系数必须为 0；log 值本身没有物理意义，也统一归零方便保存和分析。
        coefficient = torch.exp(log_coefficient) * mask_as_float
        log_coefficient = log_coefficient * mask_as_float
        total_drag = (coefficient * single_drag * mask_as_float).sum(dim=1)

        outputs: Dict[str, Union[Tensor, List[Tensor]]] = {
            "total_drag": total_drag,
            "coefficient": coefficient,
            "log_coefficient": log_coefficient,
        }
        if return_attention:
            outputs["attention"] = collected_attention
        return outputs
