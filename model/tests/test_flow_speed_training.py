"""``flow_speed`` 固定划分在训练总入口中的专项回归测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import model.train as train_module
from model.training.trainer import PredictionResult


FLOW_SPEEDS = (0.1, 0.2, 0.3, 0.4, 0.4)
"""测试数据的固定流速；前三条训练，后两条同时用于 validation/test。"""


class FlowSpeedDataset(Dataset):
    """提供划分、manifest 和 scaler 所需字段的最小确定性数据集。"""

    def __init__(self) -> None:
        """按 ``FLOW_SPEEDS`` 创建五条结构完整、数值简单的样本。"""

        self.samples: list[dict[str, Any]] = []
        for source_index, flow_speed in enumerate(FLOW_SPEEDS):
            # 两株水草足以让 ``plant_count`` 元数据和训练标签收集逻辑正常工作；
            # 坐标值本身不会进入本测试中被替换掉的模型训练函数。
            positions = torch.tensor(
                [[0.0, 0.0], [1.0, float(source_index)]],
                dtype=torch.float32,
            )
            self.samples.append(
                {
                    "positions": positions,
                    "global_features": torch.tensor(
                        [flow_speed], dtype=torch.float32
                    ),
                    "target_drag": torch.tensor(
                        1.0 + source_index, dtype=torch.float32
                    ),
                    "model_id": source_index + 1,
                    "state_id": 1,
                    "angle": 0,
                    "flow_speed": flow_speed,
                    "source_index": source_index,
                }
            )

    def __len__(self) -> int:
        """返回固定样本数。"""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """按整数索引返回样本字典。

        Args:
            index: 从 0 开始的数据集索引。

        Returns:
            包含训练划分所需字段的单条样本。
        """

        return self.samples[index]


def _flow_speed_config() -> dict[str, Any]:
    """创建 ``run_cross_validation`` 所需的最小配置字典。"""

    return {
        "seed": 17,
        "data": {
            "csv_path": "synthetic.csv",
            "negative_target_policy": "clamp_to_zero",
        },
        "model": {},
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "device": "cpu",
            "max_epochs": 3,
        },
        "cross_validation": {
            "split_mode": "flow_speed",
            # 这两个普通 CV 参数应由固定速度模式忽略。
            "n_splits": 5,
            "validation_fraction": 0.2,
        },
        "resolved_physical_config": {"states": {}},
    }


def test_flow_speed_training_uses_only_train_speeds_and_skips_final_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """训练入口应只执行 fold_0，并完整记录重叠评估和跳过重训的语义。"""

    dataset = FlowSpeedDataset()
    recorded: dict[str, Any] = {"prediction_indices": []}

    def fake_relative_floor(
        received_dataset: Dataset,
        indices: np.ndarray,
        settings: dict[str, Any],
    ) -> float:
        """记录 loss floor 收到的索引并返回固定正数。"""

        assert received_dataset is dataset
        assert settings is _flow_speed_config_settings
        recorded["floor_indices"] = np.asarray(indices).copy()
        return 0.25

    def fake_fit_with_early_stopping(**kwargs: Any) -> SimpleNamespace:
        """代替耗时训练，记录 train/validation 与 checkpoint metadata。"""

        recorded["train_indices"] = np.asarray(kwargs["train_indices"]).copy()
        recorded["validation_indices"] = np.asarray(
            kwargs["validation_indices"]
        ).copy()
        recorded["checkpoint_metadata"] = kwargs["checkpoint_metadata"]
        # 创建轻量占位文件，以验证 fold checkpoint 仍按原目录契约产生。
        best_checkpoint = Path(kwargs["artifact_dir"]) / "best.pt"
        best_checkpoint.write_bytes(b"flow-speed-test")
        return SimpleNamespace(
            best_epoch=2,
            best_validation_loss=0.125,
            best_checkpoint=best_checkpoint,
            history=[],
        )

    def fake_predict_dataset(
        model: object,
        received_dataset: Dataset,
        indices: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> PredictionResult:
        """为 validation/test 返回可汇总的确定性预测，并记录索引。"""

        del model, args, kwargs
        assert received_dataset is dataset
        copied_indices = np.asarray(indices).copy()
        recorded["prediction_indices"].append(copied_indices)
        predictions: list[dict[str, Any]] = []
        coefficients: list[dict[str, Any]] = []
        for index in copied_indices:
            numeric_index = int(index)
            target_drag = float(numeric_index + 1)
            predictions.append(
                {
                    "source_index": numeric_index,
                    "target_drag": target_drag,
                    "predicted_drag": target_drag,
                    "isolated_drag": 1.0,
                }
            )
            coefficients.append(
                {"source_index": numeric_index, "plant_index": 0, "coefficient": 1.0}
            )
        return PredictionResult(
            metrics={"RMSE_D": 0.0},
            predictions=predictions,
            plant_coefficients=coefficients,
        )

    def fail_if_final_retraining_runs(*args: Any, **kwargs: Any) -> None:
        """若速度模式错误进入全量重训分支，立即使测试失败。"""

        del args, kwargs
        raise AssertionError("flow_speed 模式不应调用 fit_fixed_epochs")

    config = _flow_speed_config()
    _flow_speed_config_settings = config["training"]
    monkeypatch.setattr(train_module, "_fit_training_relative_floor", fake_relative_floor)
    monkeypatch.setattr(train_module, "_new_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        train_module, "fit_with_early_stopping", fake_fit_with_early_stopping
    )
    monkeypatch.setattr(train_module, "predict_dataset", fake_predict_dataset)
    monkeypatch.setattr(train_module, "fit_fixed_epochs", fail_if_final_retraining_runs)
    # 文件写入函数各有独立单元测试；这里替换它们，避免要求假预测具备完整生产字段。
    monkeypatch.setattr(train_module, "write_prediction_result", lambda *args: None)
    monkeypatch.setattr(train_module, "write_drag_comparison_plot", lambda *args, **kwargs: None)

    train_module.run_cross_validation(dataset, config, tmp_path)

    expected_train = np.asarray([0, 1, 2], dtype=np.int64)
    expected_validation_test = np.asarray([3, 4], dtype=np.int64)
    assert np.array_equal(recorded["train_indices"], expected_train)
    assert np.array_equal(recorded["floor_indices"], expected_train)
    assert np.array_equal(recorded["validation_indices"], expected_validation_test)
    assert len(recorded["prediction_indices"]) == 2
    assert all(
        np.array_equal(indices, expected_validation_test)
        for indices in recorded["prediction_indices"]
    )

    # 真实 scaler 保持启用，因此其均值可以直接证明只读取了 0.1/0.2/0.3。
    scaler_data = json.loads((tmp_path / "fold_0" / "scaler.json").read_text("utf-8"))
    assert scaler_data["mean"][0] == pytest.approx(0.2)
    assert (tmp_path / "fold_0" / "best.pt").is_file()
    assert not (tmp_path / "final_scaler.json").exists()
    assert not (tmp_path / "final_model.pt").exists()

    metrics = json.loads((tmp_path / "cv_metrics.json").read_text("utf-8"))
    assert metrics["split_mode"] == "flow_speed"
    assert metrics["validation_test_overlap"] is True
    assert metrics["final_retraining_performed"] is False
    metadata = recorded["checkpoint_metadata"]
    assert metadata["split_mode"] == "flow_speed"
    assert metadata["validation_test_overlap"] is True
    assert metadata["evaluation_source_indices"] == [3, 4]

    # manifest 中 0.4 m/s 样本应分别出现一次 validation 和一次 test 角色。
    with (tmp_path / "fold_assignments.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        manifest_rows = list(csv.DictReader(handle))
    validation_indices = {
        int(row["dataset_index"])
        for row in manifest_rows
        if row["role"] == "validation"
    }
    test_indices = {
        int(row["dataset_index"])
        for row in manifest_rows
        if row["role"] == "test"
    }
    assert validation_indices == {3, 4}
    assert test_indices == validation_indices
