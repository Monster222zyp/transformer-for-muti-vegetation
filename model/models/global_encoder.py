"""全局物理量编码器。"""

from torch import Tensor, nn


class GlobalEncoder(nn.Module):
    """把流速等全局物理量映射到 Transformer 的条件向量空间。

    参数:
        input_dim: 每条样本包含的全局物理量个数。当前数据只使用流速，因此默认为 1。
        hidden_dim: 两层感知机中间层的维度。
        output_dim: 条件向量维度，通常等于 ``d_model``。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, global_features: Tensor) -> Tensor:
        """编码 ``[batch, input_dim]``，返回 ``[batch, output_dim]``。"""

        return self.network(global_features)

