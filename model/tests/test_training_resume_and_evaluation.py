"""训练恢复、模型配置校验和评估范围语义的回归测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from model.evaluate import _select_evaluation_indices
from model.hydro.data import collate_hydro_samples
from model.models import HydroTransformer
from model.train import _new_model, _resolve_config_paths
from model.training.checkpoint import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    resolved_model_config,
    save_checkpoint,
)
from model.training.trainer import (
    GlobalFeatureScaler,
    fit_with_early_stopping,
    predict_dataset,
)


SMALL_MODEL_CONFIG = {
    "d_model": 16,
    "n_heads": 2,
    "n_layers": 1,
    "ffn_dim": 32,
    "dropout": 0.0,
    "relative_hidden_dim": 8,
    "coefficient_hidden_dim": 8,
}


class TinyHydroDataset(Dataset):
    """只用于训练恢复测试的两个确定性水草样本。"""

    def __init__(self) -> None:
        self.samples = []
        for source_index, target in enumerate((1.5, 1.8)):
            self.samples.append(
                {
                    "positions": torch.tensor(
                        [[0.0, 0.0], [1.0, float(source_index)]],
                        dtype=torch.float32,
                    ),
                    "single_drag": torch.ones(2),
                    "plant_mask": torch.ones(2, dtype=torch.bool),
                    "global_features": torch.tensor([0.1 + source_index * 0.1]),
                    "target_drag": torch.tensor(target),
                    "raw_target_drag": torch.tensor(target),
                    "model_id": source_index + 1,
                    "angle": 0,
                    "flow_speed": 0.1 + source_index * 0.1,
                    "source_index": source_index,
                }
            )

    def __len__(self) -> int:
        """返回固定的两个样本。"""

        return len(self.samples)

    def __getitem__(self, index: int):
        """返回指定测试样本。"""

        return self.samples[index]


class SourceIndexDataset(Dataset):
    """只提供稳定 source_index 的轻量评估范围测试数据集。"""

    def __init__(self, source_indices: list[int]) -> None:
        self.source_indices = source_indices

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int):
        return {"source_index": self.source_indices[index]}


def _settings(max_epochs: int = 1) -> dict[str, object]:
    """返回一次 CPU 单 batch 训练所需的最小配置。"""

    return {
        "batch_size": 2,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "max_epochs": max_epochs,
        "early_stopping_patience": 2,
        "gradient_clip": 1.0,
        "num_workers": 0,
        "device": "cpu",
    }


def test_model_is_initialized_after_setting_fold_seed() -> None:
    """外部 RNG 状态变化不能改变同一 seed 创建的模型参数。"""

    first = _new_model(SMALL_MODEL_CONFIG, seed=123)
    torch.randn(100)
    second = _new_model(SMALL_MODEL_CONFIG, seed=123)

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters()
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_resume_to_new_directory_recreates_best_checkpoint(tmp_path: Path) -> None:
    """跨目录恢复且无剩余 epoch 时，新目录仍必须拥有可加载的 best.pt。"""

    dataset = TinyHydroDataset()
    indices = np.arange(len(dataset), dtype=np.int64)
    scaler = GlobalFeatureScaler.fit(dataset, indices)
    original_dir = tmp_path / "original"
    resumed_dir = tmp_path / "resumed"
    first_model = _new_model(SMALL_MODEL_CONFIG, seed=7)
    fit_with_early_stopping(
        first_model,
        dataset,
        indices,
        indices,
        collate_hydro_samples,
        _settings(max_epochs=1),
        scaler,
        original_dir,
        SMALL_MODEL_CONFIG,
        seed=7,
    )

    resumed_model = _new_model(SMALL_MODEL_CONFIG, seed=999)
    result = fit_with_early_stopping(
        resumed_model,
        dataset,
        indices,
        indices,
        collate_hydro_samples,
        _settings(max_epochs=1),
        scaler,
        resumed_dir,
        SMALL_MODEL_CONFIG,
        seed=7,
        resume_from=original_dir / "last.pt",
    )

    assert result.best_checkpoint == resumed_dir / "best.pt"
    assert result.best_checkpoint.is_file()
    load_checkpoint(result.best_checkpoint, resumed_model)


def test_checkpoint_validates_complete_model_config(tmp_path: Path) -> None:
    """恢复到配置不同的模型时，应在 state_dict shape 错误前给出明确配置错误。"""

    model = HydroTransformer(**SMALL_MODEL_CONFIG)
    checkpoint_path = tmp_path / "configured.pt"
    save_checkpoint(
        checkpoint_path,
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_state": model.state_dict(),
            "model_config": resolved_model_config(model),
        },
    )
    mismatched = HydroTransformer(**{**SMALL_MODEL_CONFIG, "n_layers": 2})

    with pytest.raises(ValueError, match="model_config"):
        load_checkpoint(checkpoint_path, mismatched)


def test_prediction_keeps_source_index_first_and_rounds_flow_speed() -> None:
    """预测表优先保留稳定行标识，流速恢复为实验的一位小数精度。"""

    dataset = TinyHydroDataset()
    indices = np.arange(len(dataset), dtype=np.int64)
    scaler = GlobalFeatureScaler.fit(dataset, indices)
    model = _new_model(SMALL_MODEL_CONFIG, seed=5)
    result = predict_dataset(
        model,
        dataset,
        indices,
        collate_hydro_samples,
        batch_size=2,
        num_workers=0,
        scaler=scaler,
        device=torch.device("cpu"),
    )

    first_row = result.predictions[0]
    assert next(iter(first_row)) == "source_index"
    assert str(first_row["flow_speed"]) == "0.1"


def test_fold_final_and_external_evaluation_scopes() -> None:
    """fold 只取 held-out source，final 标记 in-sample，external 使用全部外部行。"""

    dataset = SourceIndexDataset([10, 20, 30])
    fold_indices, fold_scope = _select_evaluation_indices(
        dataset,
        {"checkpoint_role": "fold", "evaluation_source_indices": [30, 10]},
        external_data=False,
    )
    final_indices, final_scope = _select_evaluation_indices(
        dataset,
        {"checkpoint_role": "final", "evaluation_source_indices": None},
        external_data=False,
    )
    external_indices, external_scope = _select_evaluation_indices(
        dataset,
        {"checkpoint_role": "fold", "evaluation_source_indices": [30]},
        external_data=True,
    )

    assert fold_indices.tolist() == [2, 0]
    assert fold_scope == "held_out"
    assert final_indices.tolist() == [0, 1, 2]
    assert final_scope == "in_sample"
    assert external_indices.tolist() == [0, 1, 2]
    assert external_scope == "external"


def test_relative_config_paths_resolve_from_project_root() -> None:
    """YAML 中的相对默认路径不依赖命令执行目录。"""

    config = {
        "data": {
            "csv_path": "model/data/all_models.csv",
            "input_csv_path": "Experiment/input.csv",
        },
        "output": {"artifact_dir": "model/artifacts"},
    }
    _resolve_config_paths(config)

    assert Path(config["data"]["csv_path"]).is_absolute()
    assert Path(config["data"]["input_csv_path"]).is_absolute()
    assert Path(config["output"]["artifact_dir"]).is_absolute()
