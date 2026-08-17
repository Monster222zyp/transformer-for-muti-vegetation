"""把每株水草隐藏向量转换为正的 latent coefficient。"""

from torch import Tensor, nn


class CoefficientHead(nn.Module):
    """所有植物共享的 ``d_model → 128 → 1`` 系数头。

    最后一层采用全零初始化，使模型初始时 ``log(c_i)=0``、``c_i=1``，对应“尚未
    学到耦合作用”的物理先验。
    """

    def __init__(self, d_model: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden = nn.Linear(d_model, hidden_dim)
        self.activation = nn.SiLU()
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """返回尚未截断的逐株 ``log_coefficient``，形状为 ``[B,N]``。"""

        return self.output(self.activation(self.hidden(hidden_states))).squeeze(-1)

