"""由全局物理条件控制的归一化和特征调制模块。"""

from typing import Tuple

from torch import Tensor, nn


class FeatureWiseLinearModulation(nn.Module):
    """使用 FiLM 对特征执行 ``(1 + gamma) * x + beta``。

    FiLM（Feature-wise Linear Modulation，逐特征线性调制）让流速等全局量改变
    网络内部特征。投影层采用全零初始化，因此模型刚创建时严格执行恒等映射，
    不会在训练开始前凭空改变 Value 或 relative Value。

    参数:
        condition_dim: 全局条件向量 ``g`` 的最后一维。
        feature_dim: 被调制特征的最后一维。
    """

    def __init__(self, condition_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.to_scale_shift = nn.Linear(condition_dim, 2 * feature_dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def parameters_from_condition(self, condition: Tensor) -> Tuple[Tensor, Tensor]:
        """返回调制参数 ``gamma`` 和 ``beta``，二者形状均为 ``[B, feature_dim]``。"""

        gamma, beta = self.to_scale_shift(condition).chunk(2, dim=-1)
        return gamma, beta

    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        """调制任意以 batch 开头、以 ``feature_dim`` 结尾的特征张量。

        中间的空间维度可能是植物维 ``N``，也可能是成对植物维 ``N x N``；
        这里自动补入广播维度，因此同一个模块可以服务普通 Value 和 relative Value。
        """

        gamma, beta = self.parameters_from_condition(condition)
        broadcast_shape = (condition.shape[0],) + (1,) * (features.ndim - 2) + (self.feature_dim,)
        gamma = gamma.reshape(broadcast_shape)
        beta = beta.reshape(broadcast_shape)
        return (1.0 + gamma) * features + beta


class ConditionalLayerNorm(nn.Module):
    """先做 LayerNorm，再根据全局条件执行零初始化的 FiLM。

    参数:
        feature_dim: Transformer 隐藏向量维度。
        condition_dim: 全局条件向量维度。
        enabled: 为 ``False`` 时退化为普通 LayerNorm，用于消融实验。
        eps: LayerNorm 防止除零所用的小常数。
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        enabled: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.norm = nn.LayerNorm(feature_dim, eps=eps)
        # 无论当前消融实验是否启用 Conditional LayerNorm，都实例化同一个 FiLM。
        # 这样开关不会改变后续层消耗随机数的顺序；使用相同 seed 构造的不同消融模型，
        # 其公共参数便具有完全相同的初始值，比较结果不会混入初始化差异。
        self.modulation = FeatureWiseLinearModulation(condition_dim, feature_dim)

    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        """归一化 ``features``，并在启用时用 ``condition`` 调制结果。"""

        normalized = self.norm(features)
        if not self.enabled:
            return normalized
        return self.modulation(normalized, condition)
