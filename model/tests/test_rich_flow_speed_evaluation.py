"""普通交叉验证的分流速指标与 2×2 图表产物契约测试。"""

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
from model.training.metrics import compute_regression_metrics
from model.training.splits import GroupSplit
from model.training.trainer import PredictionResult


TEST_FLOW_SPEEDS = (0.1, 0.2, 0.3, 0.4)
"""测试用的四档固定流速，单位为 m/s。"""


class RichEvaluationDataset(Dataset):
    """提供 train、validation、test 各四档流速的轻量数据集。"""

    def __init__(self) -> None:
        """创建十二条结构完整、无需真实模型前向传播的样本。"""

        self.samples: list[dict[str, Any]] = []
        for source_index in range(12):
            flow_speed = TEST_FLOW_SPEEDS[source_index % len(TEST_FLOW_SPEEDS)]
            self.samples.append(
                {
                    "positions": torch.tensor(
                        [[0.0, 0.0], [1.0, float(source_index)]],
                        dtype=torch.float32,
                    ),
                    "global_features": torch.tensor(
                        [flow_speed], dtype=torch.float32
                    ),
                    "target_drag": torch.tensor(
                        1.0 + source_index, dtype=torch.float32
                    ),
                    "sample_id": f"sample_{source_index}",
                    "model_id": source_index + 1,
                    "angle": 0,
                    "flow_speed": flow_speed,
                    "source_index": source_index,
                }
            )

    def __len__(self) -> int:
        """返回固定的十二条样本。"""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回指定位置的样本。"""

        return self.samples[index]


def _ordinary_cv_config() -> dict[str, Any]:
    """创建普通 sample 模式入口所需的最小配置。"""

    return {
        "seed": 23,
        "data": {
            "dataset_path": "synthetic.jsonl",
            "negative_target_policy": "clamp_to_zero",
        },
        "model": {},
        "training": {
            "batch_size": 4,
            "num_workers": 0,
            "device": "cpu",
            "max_epochs": 2,
        },
        "cross_validation": {
            "split_mode": "sample",
            "n_splits": 3,
            "validation_fraction": 0.2,
        },
        "resolved_physical_config": {"states": {}},
    }


def test_ordinary_cv_writes_metrics_and_plots_by_flow_speed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """普通 CV 应保留总体结果，并新增 fold/root 两级的分流速产物。"""

    dataset = RichEvaluationDataset()
    split = GroupSplit(
        fold=0,
        train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        validation_indices=np.asarray([4, 5, 6, 7], dtype=np.int64),
        test_indices=np.asarray([8, 9, 10, 11], dtype=np.int64),
    )

    def fake_fit_with_early_stopping(**kwargs: Any) -> SimpleNamespace:
        """跳过耗时训练，并提供正常的最佳 epoch 信息。"""

        checkpoint = Path(kwargs["artifact_dir"]) / "best.pt"
        checkpoint.write_bytes(b"rich-evaluation-test")
        return SimpleNamespace(
            best_epoch=2,
            best_validation_loss=0.125,
            best_checkpoint=checkpoint,
            history=[],
        )

    def fake_predict_dataset(
        model: object,
        received_dataset: Dataset,
        indices: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> PredictionResult:
        """按给定索引返回包含四档流速的确定性预测。"""

        del model, args, kwargs
        assert received_dataset is dataset
        predictions: list[dict[str, Any]] = []
        coefficients: list[dict[str, Any]] = []
        for index in np.asarray(indices):
            numeric_index = int(index)
            sample = dataset[numeric_index]
            target_drag = float(sample["target_drag"])
            isolated_drag = target_drag + 1.0
            predictions.append(
                {
                    "source_index": numeric_index,
                    "model_id": int(sample["model_id"]),
                    "sample_id": str(sample["sample_id"]),
                    "angle": 0,
                    "flow_speed": float(sample["flow_speed"]),
                    "raw_target_drag": target_drag,
                    "target_drag": target_drag,
                    "predicted_drag": target_drag + 0.1,
                    "isolated_drag": isolated_drag,
                    "target_C": target_drag / isolated_drag,
                    "predicted_C": (target_drag + 0.1) / isolated_drag,
                }
            )
            coefficients.append(
                {
                    "source_index": numeric_index,
                    "plant_index": 0,
                    "coefficient": 1.0,
                }
            )
        metrics = compute_regression_metrics(
            [row["target_drag"] for row in predictions],
            [row["predicted_drag"] for row in predictions],
            [row["isolated_drag"] for row in predictions],
        )
        return PredictionResult(metrics, predictions, coefficients)

    def fake_fit_fixed_epochs(
        model: object,
        received_dataset: Dataset,
        indices: np.ndarray,
        collate_fn: object,
        settings: dict[str, Any],
        scaler: object,
        artifact_dir: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Path:
        """模拟普通 CV 的全量重训并生成最终 checkpoint 占位文件。"""

        del model, indices, collate_fn, settings, scaler, args, kwargs
        assert received_dataset is dataset
        checkpoint = Path(artifact_dir) / "final_model.pt"
        checkpoint.write_bytes(b"final-model-test")
        return checkpoint

    monkeypatch.setattr(train_module, "_new_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        train_module, "fit_with_early_stopping", fake_fit_with_early_stopping
    )
    monkeypatch.setattr(train_module, "predict_dataset", fake_predict_dataset)
    monkeypatch.setattr(train_module, "fit_fixed_epochs", fake_fit_fixed_epochs)

    train_module.run_cross_validation(
        dataset,
        _ordinary_cv_config(),
        tmp_path,
        prepared_splits=("sample", [split]),
    )

    fold_dir = tmp_path / "fold_0"
    for prefix in ("validation", "test"):
        assert (fold_dir / f"{prefix}_drag_comparison.png").is_file()
        assert (
            fold_dir / f"{prefix}_drag_comparison_by_flow_speed.png"
        ).is_file()
        assert (fold_dir / f"{prefix}_metrics_by_flow_speed.json").is_file()
        assert (fold_dir / f"{prefix}_metrics_by_flow_speed.csv").is_file()
    assert (tmp_path / "cv_drag_comparison.png").is_file()
    assert (tmp_path / "cv_drag_comparison_by_flow_speed.png").is_file()
    assert (tmp_path / "cv_metrics_by_flow_speed.json").is_file()
    assert (tmp_path / "cv_metrics_by_flow_speed.csv").is_file()
    assert (tmp_path / "final_model.pt").is_file()

    metrics = json.loads((tmp_path / "cv_metrics.json").read_text("utf-8"))
    assert list(metrics["aggregate_by_flow_speed"]) == ["0.1", "0.2", "0.3", "0.4"]
    assert list(metrics["folds"][0]["validation_metrics_by_flow_speed"]) == [
        "0.1",
        "0.2",
        "0.3",
        "0.4",
    ]
    assert list(metrics["folds"][0]["test_metrics_by_flow_speed"]) == [
        "0.1",
        "0.2",
        "0.3",
        "0.4",
    ]
    assert metrics["final_retraining_performed"] is True

    with (tmp_path / "cv_metrics_by_flow_speed.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["flow_speed"] for row in rows] == ["0.1", "0.2", "0.3", "0.4"]
    assert all(int(row["sample_count"]) == 1 for row in rows)
    assert "按流速汇总的 held-out test 指标" in capsys.readouterr().out
