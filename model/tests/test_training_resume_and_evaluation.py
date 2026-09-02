"""训练恢复、模型配置校验和评估范围语义的回归测试。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from model.evaluate import _select_evaluation_indices
from model.hydro.data import collate_hydro_samples
from model.models import HydroTransformer
from model.train import (
    _fit_training_relative_floor,
    _new_model,
    _resolve_config_paths,
)
from model.training.checkpoint import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    resolved_model_config,
    save_checkpoint,
)
from model.training.trainer import (
    GlobalFeatureScaler,
    _build_adamw,
    _sample_loss_history,
    fit_fixed_epochs,
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
                    "plant_state": torch.full(
                        (2,), 1 + source_index, dtype=torch.long
                    ),
                    "plant_mask": torch.ones(2, dtype=torch.bool),
                    "global_features": torch.tensor([0.1 + source_index * 0.1]),
                    "target_drag": torch.tensor(target),
                    "raw_target_drag": torch.tensor(target),
                    "model_id": source_index + 1,
                    "state_id": 1 + source_index,
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


def test_each_training_scope_fits_its_own_relative_floor() -> None:
    """Overfit、fold 和 final 传入不同 train 索引时必须得到各自的尺度。"""

    dataset = TinyHydroDataset()
    settings = _settings()
    # 单样本训练集合的任意分位数都等于该正标签本身。
    first_only = _fit_training_relative_floor(
        dataset, np.asarray([0], dtype=np.int64), settings
    )
    second_only = _fit_training_relative_floor(
        dataset, np.asarray([1], dtype=np.int64), settings
    )
    # 全量 final 的 5% 分位数只由两条训练标签 1.5 和 1.8 计算。
    all_samples = _fit_training_relative_floor(
        dataset, np.asarray([0, 1], dtype=np.int64), settings
    )

    assert first_only == pytest.approx(1.5)
    assert second_only == pytest.approx(1.8)
    assert all_samples == pytest.approx(float(np.quantile([1.5, 1.8], 0.05)))


def test_adamw_uses_fused_kernel_only_on_cuda() -> None:
    """CUDA 使用 fused AdamW，CPU 必须自动回退到普通实现。"""

    cpu_model = torch.nn.Linear(2, 1)
    cpu_optimizer = _build_adamw(
        cpu_model, _settings(), torch.device("cpu")
    )
    assert cpu_optimizer.defaults["fused"] is False

    if torch.cuda.is_available():
        cuda_model = torch.nn.Linear(2, 1).cuda()
        cuda_optimizer = _build_adamw(
            cuda_model, _settings(), torch.device("cuda")
        )
        assert cuda_optimizer.defaults["fused"] is True


def test_training_reports_every_ten_epochs_and_writes_loss_curve(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """第10个 epoch 必须打印 loss，并生成训练/验证双折线 PNG。"""

    dataset = TinyHydroDataset()
    indices = np.arange(len(dataset), dtype=np.int64)
    scaler = GlobalFeatureScaler.fit(dataset, indices)
    settings = _settings(max_epochs=10)
    # 测试必须完整运行到第10个 epoch，因此把 patience 设置得大于训练轮数。
    settings["early_stopping_patience"] = 11
    settings["progress_interval_epochs"] = 10
    settings["loss_plot_interval_epochs"] = 10

    result = fit_with_early_stopping(
        _new_model(SMALL_MODEL_CONFIG, seed=19),
        dataset,
        indices,
        indices,
        collate_hydro_samples,
        settings,
        scaler,
        tmp_path,
        SMALL_MODEL_CONFIG,
        seed=19,
        relative_floor=0.1,
        progress_label="Fold test",
    )

    terminal_output = capsys.readouterr().out
    assert "[Fold test] Epoch 10/10" in terminal_output
    assert "train_relative_mse=" in terminal_output
    assert "validation_relative_mse=" in terminal_output
    assert (tmp_path / "loss_curve.png").stat().st_size > 0
    assert (tmp_path / "history.csv").is_file()
    with (tmp_path / "history.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        history_rows = list(csv.DictReader(handle))
    assert len(history_rows) == 10
    assert set(history_rows[0]) == {
        "epoch",
        "train_relative_mse",
        "validation_relative_mse",
        "learning_rate",
    }

    sampled = _sample_loss_history(result.history, interval_epochs=10)
    assert [int(item["epoch"]) for item in sampled] == [10]

    # 提前停止在非10倍数时，图中还应保留最后一个 epoch 作为结束点。
    partial_history = [
        {"epoch": float(epoch), "train_loss": 1.0, "validation_loss": 1.0}
        for epoch in range(1, 24)
    ]
    partial_sampled = _sample_loss_history(partial_history, interval_epochs=10)
    assert [int(item["epoch"]) for item in partial_sampled] == [10, 20, 23]


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
        relative_floor=0.1,
    )

    # best 只服务评估，last 才服务续训；两者都不能重复保存 best_model_state。
    best_state = torch.load(
        original_dir / "best.pt", map_location="cpu", weights_only=False
    )
    last_state = torch.load(
        original_dir / "last.pt", map_location="cpu", weights_only=False
    )
    assert best_state["checkpoint_type"] == "best"
    assert "optimizer_state" not in best_state
    assert "scheduler_state" not in best_state
    assert "history" not in best_state
    assert "best_model_state" not in best_state
    assert best_state["loss_name"] == "stabilized_relative_mse"
    assert best_state["relative_floor"] == pytest.approx(0.1)
    assert best_state["relative_floor_quantile"] == pytest.approx(0.05)
    assert best_state["checkpoint_metadata"]["loss_name"] == (
        "stabilized_relative_mse"
    )
    assert best_state["checkpoint_metadata"]["relative_floor"] == pytest.approx(
        0.1
    )
    assert last_state["checkpoint_type"] == "last"
    assert "optimizer_state" in last_state
    assert "scheduler_state" in last_state
    assert "history" in last_state
    assert "best_model_state" not in last_state

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
        relative_floor=0.1,
        resume_from=original_dir / "last.pt",
    )

    assert result.best_checkpoint == resumed_dir / "best.pt"
    assert result.best_checkpoint.is_file()
    load_checkpoint(result.best_checkpoint, resumed_model)


def test_resume_rejects_old_absolute_loss_but_inference_loads_it(
    tmp_path: Path,
) -> None:
    """旧 absolute-MSE 权重可推理，但不能混入新的相对目标继续训练。"""

    dataset = TinyHydroDataset()
    indices = np.arange(len(dataset), dtype=np.int64)
    scaler = GlobalFeatureScaler.fit(dataset, indices)
    source_dir = tmp_path / "source"
    model = _new_model(SMALL_MODEL_CONFIG, seed=31)
    fit_with_early_stopping(
        model,
        dataset,
        indices,
        indices,
        collate_hydro_samples,
        _settings(max_epochs=1),
        scaler,
        source_dir,
        SMALL_MODEL_CONFIG,
        seed=31,
        relative_floor=0.1,
    )

    # 模拟同一 v4 模型结构、但训练目标仍为旧绝对 MSE 的 checkpoint。
    old_state = torch.load(
        source_dir / "last.pt", map_location="cpu", weights_only=False
    )
    old_state["loss_name"] = "total_drag_mse"
    old_checkpoint = tmp_path / "old_absolute.pt"
    save_checkpoint(old_checkpoint, old_state)

    # 通用加载器服务推理，不检查 loss_name，因此旧权重仍可正常加载。
    inference_model = _new_model(SMALL_MODEL_CONFIG, seed=999)
    load_checkpoint(old_checkpoint, inference_model)

    # 只有续训入口会比较训练目标，并拒绝把两种 loss 混在同一次实验中。
    with pytest.raises(ValueError, match="不同训练目标"):
        fit_with_early_stopping(
            _new_model(SMALL_MODEL_CONFIG, seed=31),
            dataset,
            indices,
            indices,
            collate_hydro_samples,
            _settings(max_epochs=1),
            scaler,
            tmp_path / "rejected_resume",
            SMALL_MODEL_CONFIG,
            seed=31,
            relative_floor=0.1,
            resume_from=old_checkpoint,
        )


def test_resume_rejects_changed_relative_floor(tmp_path: Path) -> None:
    """训练子集或分位数变化导致 floor 改变时，必须拒绝断点续训。"""

    dataset = TinyHydroDataset()
    indices = np.arange(len(dataset), dtype=np.int64)
    scaler = GlobalFeatureScaler.fit(dataset, indices)
    source_dir = tmp_path / "source"
    fit_with_early_stopping(
        _new_model(SMALL_MODEL_CONFIG, seed=41),
        dataset,
        indices,
        indices,
        collate_hydro_samples,
        _settings(max_epochs=1),
        scaler,
        source_dir,
        SMALL_MODEL_CONFIG,
        seed=41,
        relative_floor=0.1,
    )

    with pytest.raises(ValueError, match="relative_floor"):
        fit_with_early_stopping(
            _new_model(SMALL_MODEL_CONFIG, seed=41),
            dataset,
            indices,
            indices,
            collate_hydro_samples,
            _settings(max_epochs=1),
            scaler,
            tmp_path / "changed_floor",
            SMALL_MODEL_CONFIG,
            seed=41,
            relative_floor=0.2,
            resume_from=source_dir / "last.pt",
        )


def test_final_checkpoint_records_relative_objective(tmp_path: Path) -> None:
    """全量固定 epoch 重训也必须保存实际 floor 和完整相对目标 metadata。"""

    dataset = TinyHydroDataset()
    indices = np.arange(len(dataset), dtype=np.int64)
    scaler = GlobalFeatureScaler.fit(dataset, indices)
    checkpoint_path = fit_fixed_epochs(
        _new_model(SMALL_MODEL_CONFIG, seed=51),
        dataset,
        indices,
        collate_hydro_samples,
        _settings(max_epochs=1),
        scaler,
        tmp_path,
        SMALL_MODEL_CONFIG,
        seed=51,
        epochs=1,
        relative_floor=0.25,
    )

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert state["loss_name"] == "stabilized_relative_mse"
    assert state["relative_floor"] == pytest.approx(0.25)
    assert state["checkpoint_metadata"]["relative_floor"] == pytest.approx(0.25)


def test_last_checkpoint_uses_interval_and_forces_final_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """last.pt 按间隔保存，并在不落在间隔上的最终 epoch 强制保存。"""

    import model.training.trainer as trainer_module

    dataset = TinyHydroDataset()
    indices = np.arange(len(dataset), dtype=np.int64)
    scaler = GlobalFeatureScaler.fit(dataset, indices)
    settings = _settings(max_epochs=4)
    settings["early_stopping_patience"] = 5
    settings["checkpoint_interval_epochs"] = 3
    saved_last_epochs: list[int] = []
    original_save = trainer_module.save_checkpoint

    def recording_save(path: str | Path, state: dict[str, object]) -> None:
        """记录 last.pt 的 epoch，同时执行真正的原子保存。"""

        if Path(path).name == "last.pt":
            saved_last_epochs.append(int(state["epoch"]))
        original_save(path, state)

    monkeypatch.setattr(trainer_module, "save_checkpoint", recording_save)
    fit_with_early_stopping(
        _new_model(SMALL_MODEL_CONFIG, seed=23),
        dataset,
        indices,
        indices,
        collate_hydro_samples,
        settings,
        scaler,
        tmp_path,
        SMALL_MODEL_CONFIG,
        seed=23,
        relative_floor=0.1,
    )

    # 零基 epoch 2 对应第3轮间隔保存；零基 epoch 3 对应第4轮最终强制保存。
    assert saved_last_epochs == [2, 3]


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


def test_checkpoint_preserves_physics_and_rejects_old_single_token_version(
    tmp_path: Path,
) -> None:
    """双状态物理表必须随 checkpoint 保存，旧架构版本必须明确拒绝。"""

    model = HydroTransformer(**SMALL_MODEL_CONFIG)
    physical_config = {
        "version": 1,
        "drag_unit": "N",
        "states": {
            "1": {
                "angles_deg": [0, 120, 240],
                "single_drag_by_flow_speed": {
                    "0.1": 1.0,
                    "0.2": 1.1,
                    "0.3": 1.2,
                    "0.4": 1.3,
                },
            },
            "2": {
                "angles_deg": [60, 180, 300],
                "single_drag_by_flow_speed": {
                    "0.1": 2.0,
                    "0.2": 2.1,
                    "0.3": 2.2,
                    "0.4": 2.3,
                },
            },
        },
    }
    checkpoint_path = tmp_path / "two_state.pt"
    state = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "model_config": resolved_model_config(model),
        "checkpoint_metadata": {"physical_config": physical_config},
    }
    save_checkpoint(checkpoint_path, state)
    restored = load_checkpoint(checkpoint_path, model)
    assert restored["checkpoint_metadata"]["physical_config"] == physical_config

    old_path = tmp_path / "old_single_token.pt"
    save_checkpoint(old_path, {**state, "checkpoint_version": 3})
    with pytest.raises(ValueError, match="旧版单 Token"):
        load_checkpoint(old_path, model)


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
    assert first_row["state_id"] == 1
    assert result.plant_coefficients[0]["state_id"] == 1
    assert str(first_row["flow_speed"]) == "0.1"


def test_fold_final_and_external_evaluation_scopes() -> None:
    """不同 checkpoint 和重叠标记必须得到准确且不混淆的评估范围。"""

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
    overlap_indices, overlap_scope = _select_evaluation_indices(
        dataset,
        {
            "checkpoint_role": "fold",
            "evaluation_source_indices": [20, 30],
            "validation_test_overlap": True,
        },
        external_data=False,
    )

    assert fold_indices.tolist() == [2, 0]
    assert fold_scope == "held_out"
    assert final_indices.tolist() == [0, 1, 2]
    assert final_scope == "in_sample"
    assert external_indices.tolist() == [0, 1, 2]
    assert external_scope == "external"
    assert overlap_indices.tolist() == [1, 2]
    assert overlap_scope == "validation_test_overlap"


def test_relative_config_paths_resolve_from_project_root() -> None:
    """YAML 中的相对默认路径不依赖命令执行目录。"""

    config = {
        "data": {
            "csv_path": "summarized_data.csv",
            "physics_config_path": "model/configs/physical.yaml",
        },
        "output": {"artifact_dir": "model/artifacts"},
    }
    _resolve_config_paths(config)

    assert Path(config["data"]["csv_path"]).is_absolute()
    assert Path(config["data"]["physics_config_path"]).is_absolute()
    assert Path(config["output"]["artifact_dir"]).is_absolute()
