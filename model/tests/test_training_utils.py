"""训练基础设施的快速单元测试；不运行耗时的完整交叉验证。"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from model.training.checkpoint import load_checkpoint, save_checkpoint
from model.training.metrics import compute_regression_metrics
from model.training.scheduler import WarmupCosineScheduler, choose_warmup_steps
from model.training.splits import build_group_kfold_splits


def test_group_split_has_no_model_leakage_and_is_deterministic() -> None:
    """同一 model_id 不得跨训练、验证、测试集合。"""

    groups = np.repeat(np.arange(10), 4)
    first = build_group_kfold_splits(groups, n_splits=5, seed=123)
    second = build_group_kfold_splits(groups, n_splits=5, seed=123)

    assert len(first) == 5
    for split_a, split_b in zip(first, second):
        train_groups = set(groups[split_a.train_indices])
        validation_groups = set(groups[split_a.validation_indices])
        test_groups = set(groups[split_a.test_indices])
        assert train_groups.isdisjoint(validation_groups)
        assert train_groups.isdisjoint(test_groups)
        assert validation_groups.isdisjoint(test_groups)
        assert np.array_equal(split_a.train_indices, split_b.train_indices)
        assert np.array_equal(split_a.validation_indices, split_b.validation_indices)
        assert np.array_equal(split_a.test_indices, split_b.test_indices)


def test_metrics_handle_zero_targets_and_report_mape_coverage() -> None:
    """零标签不进入 MAPE，但必须安全进入 MAE、RMSE 和 sMAPE。"""

    metrics = compute_regression_metrics(
        target_drag=[0.0, 2.0],
        predicted_drag=[0.0, 1.0],
        isolated_drag=[1.0, 2.0],
    )

    assert metrics["MAE_D"] == pytest.approx(0.5)
    assert metrics["RMSE_D"] == pytest.approx(math.sqrt(0.5))
    assert metrics["MAE_C"] == pytest.approx(0.25)
    assert metrics["RMSE_C"] == pytest.approx(math.sqrt(0.125))
    assert metrics["MAPE_D"] == pytest.approx(50.0)
    assert metrics["MAPE_coverage"] == pytest.approx(0.5)
    assert metrics["sMAPE_D"] == pytest.approx(100.0 / 3.0)


def test_scheduler_and_checkpoint_round_trip(tmp_path) -> None:
    """scheduler 能变化学习率，checkpoint 能恢复模型和优化器状态。"""

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    total_steps = 10
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps, choose_warmup_steps(total_steps)
    )
    # 执行两个 step，确保已越过单步 warmup，进入 cosine 衰减区间。
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
    saved_weight = model.weight.detach().clone()
    checkpoint_path = tmp_path / "smoke.pt"
    save_checkpoint(
        checkpoint_path,
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": 0,
        },
    )

    with torch.no_grad():
        model.weight.zero_()
    state = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    assert state["epoch"] == 0
    assert torch.allclose(model.weight, saved_weight)
    assert optimizer.param_groups[0]["lr"] < 3.0e-4
