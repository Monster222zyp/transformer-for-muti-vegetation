"""线性 warmup 后接 cosine decay 的学习率调度器。"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineScheduler(LambdaLR):
    """按优化器 step 调整学习率。

    Args:
        optimizer: 需要调度的 PyTorch 优化器。
        total_steps: 整个训练阶段的优化器更新次数。
        warmup_steps: 线性 warmup 次数；会自动限制在 ``total_steps - 1``。
        last_epoch: PyTorch scheduler 的恢复游标，默认从头训练。
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps 必须大于零。")
        self.total_steps = int(total_steps)
        self.warmup_steps = min(max(int(warmup_steps), 0), total_steps - 1)

        def lr_multiplier(current_step: int) -> float:
            """返回当前 step 相对于基础学习率的倍率。"""

            if self.warmup_steps > 0 and current_step < self.warmup_steps:
                return float(current_step + 1) / float(self.warmup_steps)
            cosine_steps = max(1, self.total_steps - self.warmup_steps)
            progress = (current_step - self.warmup_steps) / cosine_steps
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda=lr_multiplier, last_epoch=last_epoch)


def choose_warmup_steps(total_steps: int) -> int:
    """按计划选择 ``min(1000, 10% 总步数)``，短训练至少 warmup 1 step。"""

    if total_steps <= 1:
        return 0
    return min(1000, max(1, int(round(total_steps * 0.10))))
