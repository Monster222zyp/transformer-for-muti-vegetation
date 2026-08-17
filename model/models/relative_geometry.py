"""植物两两相对几何的构造与编码。"""

import torch
from torch import Tensor, nn

from .conditional_norm import FeatureWiseLinearModulation


def compute_relative_positions(positions: Tensor) -> Tensor:
    """计算 ``source - target`` 方向的相对位置。

    参数:
        positions: ``[B,N,2]``，最后一维依次为 x、y。

    返回:
        ``[B,N,N,2]``。元素 ``[b,i,j]`` 严格等于 ``p_j - p_i``，其中
        ``i`` 是接收信息的 target plant，``j`` 是提供信息的 source plant。
    """

    target = positions[:, :, None, :]
    source = positions[:, None, :, :]
    return source - target


class RelativeGeometryEncoder(nn.Module):
    """将 ``[dx, dy, distance]`` 编码为每个 head 的 relative Value。

    参数:
        n_heads: attention head 数量。
        head_dim: 单个 head 的通道数。
        hidden_dim: 几何多层感知机的隐藏维度。
        condition_dim: 全局条件向量维度。
        enabled: 为 ``False`` 时返回全零 relative Value。
        condition_on_global: 是否用全局物理条件 FiLM 调制 relative Value。
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        hidden_dim: int,
        condition_dim: int,
        enabled: bool = True,
        condition_on_global: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.output_dim = n_heads * head_dim
        self.enabled = enabled
        self.condition_on_global = condition_on_global
        self.geometry_network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )
        # FiLM 始终存在，消融开关只控制 forward 是否经过它。这样切换条件路径时，
        # 不会改变后续模块的随机初始化顺序，保证同 seed 消融实验公平。
        self.modulation = FeatureWiseLinearModulation(condition_dim, self.output_dim)

    def forward(self, positions: Tensor, condition: Tensor) -> Tensor:
        """返回 ``[B,H,N,N,Dh]``，并保证所有 self-edge ``i==j`` 精确为零。"""

        batch_size, plant_count, _ = positions.shape
        if not self.enabled:
            return positions.new_zeros(
                batch_size, self.n_heads, plant_count, plant_count, self.head_dim
            )

        relative_position = compute_relative_positions(positions)
        distance = torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True)
        geometry_features = torch.cat((relative_position, distance), dim=-1)
        relative_value = self.geometry_network(geometry_features)
        if self.condition_on_global:
            relative_value = self.modulation(relative_value, condition)

        # 即使 FiLM 的 beta 后续学成非零，self-edge 也必须固定为零，避免重复表示 V_i。
        self_edge = torch.eye(plant_count, device=positions.device, dtype=torch.bool)
        relative_value = relative_value.masked_fill(self_edge[None, :, :, None], 0.0)
        return relative_value.reshape(
            batch_size, plant_count, plant_count, self.n_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)
