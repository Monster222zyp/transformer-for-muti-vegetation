"""把每株水草的朝向和实测弯曲刚度 EI 编码成 Transformer 起始向量。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


# 默认参数集中列出；正式训练由 model/configs/base.yaml 覆盖并保存到 checkpoint。
DEFAULT_PLANT_ENCODER_HIDDEN_DIM = 128  # 两层 MLP 的中间向量维度。
DEFAULT_STIFFNESS_REFERENCE_NM2 = 1.0  # EI 的固定参考量，单位 N·m²，不从测试集拟合。
PLANT_FEATURE_DIM = 3  # 固定顺序：sin(转盘角)、cos(转盘角)、log(EI/EI_ref)。
FULL_ROTATION_DEGREES = 360.0  # 周期角归一化，保证 0° 和 360° 相同。


class PlantEncoder(nn.Module):
    """用共享 MLP 逐株编码连续物理属性，不以刚度类别编号作为 Embedding 索引。

    参数：
        output_dim: 每株输出向量长度，必须等于 Transformer 的 d_model。
        hidden_dim: 中间层宽度，默认 128。
        stiffness_reference_nm2: 实测 EI 取对数前的参考量，默认 1 N·m²。
    属性 network：Linear(3, hidden_dim) → SiLU → Linear(hidden_dim, output_dim)。
    所有植株共用网络参数，但根据自己的角度和 EI 得到不同的输入向量。
    """

    def __init__(
        self, output_dim: int, hidden_dim: int = DEFAULT_PLANT_ENCODER_HIDDEN_DIM,
        stiffness_reference_nm2: float = DEFAULT_STIFFNESS_REFERENCE_NM2,
    ) -> None:
        super().__init__()
        # 维度和参考量必须提前验证，避免在 log 或矩阵乘法处才出现难定位的错误。
        for name, value in (("output_dim", output_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"PlantEncoder 的 {name} 必须为正整数。")
        if (isinstance(stiffness_reference_nm2, bool)
                or not math.isfinite(stiffness_reference_nm2) or stiffness_reference_nm2 <= 0):
            raise ValueError("stiffness_reference_nm2 必须为有限正数，单位 N·m²。")
        self.stiffness_reference_nm2 = float(stiffness_reference_nm2)
        self.network = nn.Sequential(
            nn.Linear(PLANT_FEATURE_DIM, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def physical_features(self, angles: Tensor, stiffness: Tensor, mask: Tensor) -> Tensor:
        """返回 [B,N,3] 输入特征；角度单位 degree，stiffness 单位 N·m²。

        mask 的 True 是真实植株。只验证真实位置，padding 先替换成安全角度和
        参考 EI 再取 log，不能对 padding 的 0 刚度直接计算 log(0)。
        """

        if angles.shape != stiffness.shape or mask.shape != stiffness.shape or mask.dtype != torch.bool:
            raise ValueError("plant_angles、plant_stiffness、plant_mask 必须为对齐的 [B,N]。")
        if angles.ndim != 2:
            raise ValueError("逐株角度和 EI 必须为 [B,N]。")
        if not torch.isfinite(angles[mask]).all():
            raise ValueError("有效水草的 plant_angles 必须为有限数值。")
        if not torch.isfinite(stiffness[mask]).all() or torch.any(stiffness[mask] <= 0):
            raise ValueError("有效水草的 plant_stiffness 必须为有限正 EI，单位 N·m²。")

        # 使用编码器权重的浮点类型，整数角也能安全计算；padding 不产生非有限中间量。
        dtype = self.network[0].weight.dtype
        angles = torch.where(mask, angles, torch.zeros_like(angles)).to(dtype=dtype)
        safe_stiffness = torch.where(mask, stiffness, stiffness.new_full((), self.stiffness_reference_nm2))
        safe_stiffness = safe_stiffness.to(dtype=dtype)
        radians = torch.deg2rad(angles.remainder(FULL_ROTATION_DEGREES))
        features = torch.stack(
            (torch.sin(radians), torch.cos(radians),
             torch.log(safe_stiffness / self.stiffness_reference_nm2)), dim=-1,
        )
        return features * mask.unsqueeze(-1).to(dtype=features.dtype)

    def forward(self, angles: Tensor, stiffness: Tensor, mask: Tensor) -> Tensor:
        """编码 [B,N] 的原始物理属性，返回 [B,N,output_dim]；padding 向量归零。"""

        features = self.physical_features(angles, stiffness, mask)
        tokens = self.network(features)
        # Linear 的 bias 会让零特征产生非零向量，因此必须在编码后再次屏蔽 padding。
        return tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)
