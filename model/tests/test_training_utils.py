"""训练基础设施的快速单元测试；不运行耗时的完整交叉验证。"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import model.training.checkpoint as checkpoint_module
from model.training.checkpoint import load_checkpoint, save_checkpoint
from model.training.losses import (
    fit_relative_drag_floor,
    relative_total_drag_mse_loss,
)
from model.training.metrics import (
    compute_metrics_by_flow_speed,
    compute_regression_metrics,
)
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


def test_metrics_by_flow_speed_report_complete_sorted_groups() -> None:
    """分流速指标应稳定排序，并包含表格信息和原有全部回归指标。"""

    predictions = [
        {
            "flow_speed": float(np.float32(0.1)),
            "target_drag": 1.0,
            "predicted_drag": 1.0,
            "isolated_drag": 1.0,
        },
        {
            "flow_speed": 0.1,
            "target_drag": 3.0,
            "predicted_drag": 2.0,
            "isolated_drag": 2.0,
        },
        {
            "flow_speed": 0.4,
            "target_drag": 10.0,
            "predicted_drag": 10.0,
            "isolated_drag": 5.0,
        },
        {
            "flow_speed": 0.4,
            "target_drag": 20.0,
            "predicted_drag": 20.0,
            "isolated_drag": 10.0,
        },
    ]

    grouped = compute_metrics_by_flow_speed(predictions)

    assert list(grouped) == ["0.1", "0.4"]
    assert set(grouped["0.1"]) == {
        "sample_count",
        "mean_target_D",
        "MAE_D",
        "RMSE_D",
        "R2",
        "MAE_C",
        "RMSE_C",
        "MAPE_D",
        "MAPE_coverage",
        "sMAPE_D",
    }
    assert grouped["0.1"]["sample_count"] == 2
    assert grouped["0.1"]["mean_target_D"] == pytest.approx(2.0)
    assert grouped["0.1"]["MAE_D"] == pytest.approx(0.5)
    assert grouped["0.1"]["RMSE_D"] == pytest.approx(math.sqrt(0.5))
    assert grouped["0.1"]["R2"] == pytest.approx(0.5)
    assert grouped["0.4"]["R2"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("row_update", "error_message"),
    [
        ({}, "flow_speed"),
        ({"flow_speed": float("nan")}, "有限数值"),
        ({"flow_speed": 0.5}, "不受支持"),
    ],
)
def test_metrics_by_flow_speed_reject_invalid_speed(
    row_update: dict[str, float],
    error_message: str,
) -> None:
    """缺失、非有限或协议外的流速不能被静默分到错误分组。"""

    row = {
        "target_drag": 1.0,
        "predicted_drag": 1.0,
        "isolated_drag": 1.0,
        **row_update,
    }

    with pytest.raises(ValueError, match=error_message):
        compute_metrics_by_flow_speed([row])


class TargetOnlyDataset:
    """仅提供标量 target_drag，用于验证 train-only 分母统计。"""

    def __init__(self, targets: list[float]) -> None:
        """把输入数值转换为与正式 Dataset 一致的标量 Tensor。"""

        self.targets = [torch.tensor(value) for value in targets]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """返回指定行的目标字典。"""

        return {"target_drag": self.targets[index]}


def test_relative_total_drag_loss_matches_manual_calculation() -> None:
    """稳定化相对 MSE 必须与逐条手工公式完全一致。"""

    predicted_drag = torch.tensor([0.05, 2.4])
    target_drag = torch.tensor([0.0, 2.0])

    # 第一条使用 floor=0.1，第二条使用真实值 2.0；相对误差均为 0.5 和 0.2。
    loss = relative_total_drag_mse_loss(predicted_drag, target_drag, 0.1)
    assert loss.item() == pytest.approx((0.5**2 + 0.2**2) / 2.0)


def test_relative_loss_gives_equal_weight_to_equal_percentage_errors() -> None:
    """不同阻力量级的相同百分比误差应产生相同的逐样本贡献。"""

    predicted_drag = torch.tensor([1.1, 11.0])
    target_drag = torch.tensor([1.0, 10.0])

    loss = relative_total_drag_mse_loss(predicted_drag, target_drag, 0.1)
    assert loss.item() == pytest.approx(0.1**2)


def test_relative_loss_rejects_invalid_shapes_and_floor() -> None:
    """预测与标签形状不同时必须报错，避免广播产生看似正常的错误 loss。"""

    with pytest.raises(ValueError, match="相同形状"):
        relative_total_drag_mse_loss(torch.ones(2, 1), torch.ones(2), 0.1)
    with pytest.raises(ValueError, match="有限正数"):
        relative_total_drag_mse_loss(torch.ones(2), torch.ones(2), 0.0)


def test_relative_floor_uses_only_positive_training_targets() -> None:
    """分母统计必须忽略零值，并且不能读取未传入的 validation/test 标签。"""

    # 最后一条模拟 validation/test 中的巨大标签；训练索引故意不包含它。
    dataset = TargetOnlyDataset([0.0, 1.0, 2.0, 3.0, 1.0e9])
    floor = fit_relative_drag_floor(dataset, [0, 1, 2, 3], quantile=0.05)

    assert floor == pytest.approx(float(np.quantile([1.0, 2.0, 3.0], 0.05)))


def test_relative_floor_rejects_training_subset_without_positive_target() -> None:
    """全零训练子集没有可解释的相对尺度，必须给出明确异常。"""

    dataset = TargetOnlyDataset([0.0, 0.0])
    with pytest.raises(ValueError, match="没有正 target_drag"):
        fit_relative_drag_floor(dataset, [0, 1], quantile=0.05)


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


def test_checkpoint_retries_transient_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """前四次原子替换被拒绝时，应按既定间隔重试并最终成功。"""

    destination = tmp_path / "retry.pt"
    real_replace = Path.replace
    replacement_attempts = 0
    observed_delays: list[float] = []

    def flaky_replace(source: Path, target: str | Path) -> Path:
        """模拟 Windows 短暂占用目标文件，前四次抛出 PermissionError。"""

        nonlocal replacement_attempts
        replacement_attempts += 1
        if replacement_attempts <= 4:
            raise PermissionError(5, "模拟目标文件被占用")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(
        checkpoint_module.time, "sleep", observed_delays.append
    )
    save_checkpoint(destination, {"epoch": 7})

    assert replacement_attempts == 5
    assert observed_delays == [0.2, 0.5, 1.0, 2.0]
    assert torch.load(destination, weights_only=False)["epoch"] == 7
    assert list(tmp_path.glob("retry.pt.*.tmp")) == []


def test_checkpoint_preserves_unique_temporary_after_all_retries_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """持续文件锁必须给出明确路径，并保留可正常加载的唯一临时文件。"""

    destination = tmp_path / "locked.pt"

    def always_locked(_source: Path, _target: str | Path) -> Path:
        """模拟目标 checkpoint 在全部替换尝试中始终被占用。"""

        raise PermissionError(5, "模拟持续文件锁")

    monkeypatch.setattr(Path, "replace", always_locked)
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda _delay: None)
    with pytest.raises(PermissionError, match="临时文件已保留"):
        save_checkpoint(destination, {"epoch": 11})

    temporary_files = list(tmp_path.glob("locked.pt.*.tmp"))
    assert len(temporary_files) == 1
    temporary = temporary_files[0]
    assert f".{os.getpid()}." in temporary.name
    assert torch.load(temporary, weights_only=False)["epoch"] == 11
    terminal_error = capsys.readouterr().err
    assert str(destination) in terminal_error
    assert str(temporary) in terminal_error
