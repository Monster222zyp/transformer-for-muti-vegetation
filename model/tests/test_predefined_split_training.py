"""用户显式提供 train/validation/test 三份 JSONL 时的训练回归测试。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import model.evaluate as evaluate_module
import model.train as train_module
from model.hydro.physics import load_physical_config
from model.training.splits import GroupSplit
from model.training.trainer import PredictionResult


SMALL_MODEL_CONFIG = {
    "global_input_dim": 3,
    "d_model": 16,
    "n_heads": 4,
    "n_layers": 1,
    "ffn_dim": 32,
    "dropout": 0.0,
    "use_rope": True,
    "use_relative_value": True,
    "use_conditional_layernorm": True,
    "condition_value_on_global": True,
    "condition_relative_value_on_global": True,
    "directional_attention_mode": "none",
    "directional_soft_activation": "relu",
    "directional_soft_strength": 1.0,
    "directional_tolerance": 1.0e-6,
}
"""端到端烟雾测试使用的小型网络，避免测试依赖生产模型的计算量。"""


class PredefinedDataset(Dataset):
    """提供 train/validation/test 编排所需字段的最小确定性数据集。"""

    def __init__(self) -> None:
        """创建三条训练、两条验证和两条测试样本。

        ``source_index`` 故意按每个角色从零重新开始，模拟三个独立 JSONL 各自的
        本地行号；``sample_id`` 则保持全局唯一，用于验证 checkpoint 的稳定映射。
        """

        role_local_indices = (0, 1, 2, 0, 1, 0, 1)
        flow_speeds = (0.1, 0.2, 0.3, 0.2, 0.3, 0.3, 0.4)
        self.samples: list[dict[str, Any]] = []
        for dataset_index, (source_index, flow_speed) in enumerate(
            zip(role_local_indices, flow_speeds)
        ):
            self.samples.append(
                {
                    "positions": torch.tensor(
                        [[0.0, 0.0], [1.0, float(dataset_index)]],
                        dtype=torch.float32,
                    ),
                    "global_features": torch.tensor(
                        [flow_speed, 0.0, 1.0], dtype=torch.float32
                    ),
                    "target_drag": torch.tensor(
                        1.0 + dataset_index, dtype=torch.float32
                    ),
                    "sample_id": f"predefined_{dataset_index}",
                    "model_id": dataset_index + 1,
                    "angle": 0,
                    "flow_speed": flow_speed,
                    "source_index": source_index,
                }
            )

    def __len__(self) -> int:
        """返回固定样本数。"""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """按拼接后的整数索引返回样本。"""

        return self.samples[index]


class SampleIdDataset(Dataset):
    """只提供跨文件重复检查所需 ``sample_id`` 的轻量数据集。"""

    def __init__(self, sample_ids: list[str]) -> None:
        """保存调用方给出的样本 ID 顺序。"""

        self.sample_ids = sample_ids

    def __len__(self) -> int:
        """返回样本 ID 数量。"""

        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, str]:
        """返回只含稳定样本 ID 的字典。"""

        return {"sample_id": self.sample_ids[index]}


def _predefined_config(tmp_path: Path) -> dict[str, Any]:
    """创建单次显式三数据集训练所需的最小配置。"""

    return {
        "seed": 23,
        "data": {
            "dataset_path": "unused-in-predefined.jsonl",
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
            "split_mode": train_module.PREDEFINED_SPLIT_MODE,
            "n_splits": 5,
            "validation_fraction": 0.2,
        },
        "resolved_physical_config": {"states": {}},
        "predefined_split": {
            "train_dataset_path": str(tmp_path / "train.jsonl"),
            "validation_dataset_path": str(tmp_path / "validation.jsonl"),
            "test_dataset_path": str(tmp_path / "test.jsonl"),
            "sample_counts": {"train": 3, "validation": 2, "test": 2},
            "sha256": {
                "train": "train-sha",
                "validation": "validation-sha",
                "test": "test-sha",
            },
        },
    }


def _predefined_split() -> GroupSplit:
    """返回与 :class:`PredefinedDataset` 三段边界一致的唯一划分。"""

    return GroupSplit(
        fold=0,
        train_indices=np.asarray([0, 1, 2], dtype=np.int64),
        validation_indices=np.asarray([3, 4], dtype=np.int64),
        test_indices=np.asarray([5, 6], dtype=np.int64),
    )


def _valid_jsonl_record(sample_id: str, flow_speed: float) -> dict[str, Any]:
    """创建满足原有 schema v1 的两株水草样本。

    参数：
        sample_id: 跨 train/validation/test 保持唯一的样本标识。
        flow_speed: ``physical.yaml`` 已定义的流速，单位为 m/s。

    返回值：
        可直接写成单行 JSONL 的完整记录。
    """

    layout = "1" + "0" * 35 + "1"
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "model_id": 1,
        "angle": 0,
        "flow_speed": flow_speed,
        "vegetation_layout": layout,
        "plants": [
            {
                "plant_index": 0,
                "grid_index": 0,
                "x": 2.598076211353316,
                "y": 1.5,
                "angle": 0,
            },
            {
                "plant_index": 1,
                "grid_index": 36,
                "x": -2.598076211353316,
                "y": -1.5,
                "angle": 0,
            },
        ],
        "measurements": {
            "TX": 0.0,
            "TY": 0.0,
            "TZ": 0.0,
            "FX_0": 1.0,
            "FY_0": 0.0,
            "FZ": 0.0,
        },
    }


def test_prepare_predefined_datasets_loads_original_schema_and_offsets(
    tmp_path: Path,
) -> None:
    """三份原格式 JSONL 应通过同一 loader，并形成确定的连续索引。"""

    paths = {
        "train": tmp_path / "train.jsonl",
        "validation": tmp_path / "validation.jsonl",
        "test": tmp_path / "test.jsonl",
    }
    for role, speed in (("train", 0.1), ("validation", 0.2), ("test", 0.3)):
        paths[role].write_text(
            json.dumps(
                _valid_jsonl_record(f"{role}_sample", speed),
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    config = {
        "data": {"negative_target_policy": "clamp_to_zero"},
        "predefined_split": {
            "train_dataset_path": str(paths["train"]),
            "validation_dataset_path": str(paths["validation"]),
            "test_dataset_path": str(paths["test"]),
        },
    }
    physical_config = load_physical_config(
        train_module.PROJECT_ROOT / "model" / "configs" / "physical.yaml"
    )
    combined, prepared, datasets = train_module._prepare_predefined_datasets(
        config, physical_config
    )

    split_mode, splits = prepared
    assert split_mode == "predefined"
    assert len(combined) == 3
    assert {role: len(dataset) for role, dataset in datasets.items()} == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert splits[0].train_indices.tolist() == [0]
    assert splits[0].validation_indices.tolist() == [1]
    assert splits[0].test_indices.tolist() == [2]
    assert config["predefined_split"]["sample_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert config["predefined_split"]["sha256"]["test"] == (
        train_module._sha256_file(paths["test"])
    )


def test_predefined_cli_requires_exactly_three_role_datasets() -> None:
    """YAML 未填写时 CLI 必须补全三路径，旧模式不得误接收这些参数。"""

    valid_args = train_module.parse_args(
        [
            "--mode",
            "predefined",
            "--train-data",
            "train.jsonl",
            "--validation-data",
            "validation.jsonl",
            "--test-data",
            "test.jsonl",
        ]
    )
    train_module._validate_mode_arguments(valid_args)

    missing_test = train_module.parse_args(
        [
            "--mode",
            "predefined",
            "--train-data",
            "train.jsonl",
            "--validation-data",
            "validation.jsonl",
        ]
    )
    with pytest.raises(ValueError, match="--test-data"):
        train_module._validate_mode_arguments(missing_test)

    cv_with_manual_data = train_module.parse_args(
        ["--mode", "cv", "--train-data", "train.jsonl"]
    )
    with pytest.raises(ValueError, match="cv 模式不能使用"):
        train_module._validate_mode_arguments(cv_with_manual_data)


def test_yaml_mode_and_paths_are_defaults_without_changing_cv() -> None:
    """YAML 决定默认模式和三路径；CLI 临时切换回 cv 时不误用它们。"""

    config = {
        "mode": "predefined",
        "predefined_split": {
            "train_dataset_path": "train.jsonl",
            "validation_dataset_path": "validation.jsonl",
            "test_dataset_path": "test.jsonl",
        },
    }
    predefined_args = train_module.parse_args([])
    train_module._validate_mode_arguments(predefined_args, config)
    assert predefined_args.mode == "predefined"
    assert predefined_args.train_data == "train.jsonl"
    assert predefined_args.validation_data == "validation.jsonl"
    assert predefined_args.test_data == "test.jsonl"

    cv_args = train_module.parse_args(["--mode", "cv"])
    train_module._validate_mode_arguments(cv_args, config)
    assert cv_args.mode == "cv"
    assert cv_args.train_data is None
    assert cv_args.validation_data is None
    assert cv_args.test_data is None

    override_args = train_module.parse_args(["--test-data", "new_test.jsonl"])
    train_module._validate_mode_arguments(override_args, config)
    assert override_args.test_data == "new_test.jsonl"


def test_base_yaml_declares_training_mode_and_three_dataset_paths() -> None:
    """仓库主配置显式展示运行模式和三份 JSONL 的填写位置。"""

    config = train_module.load_config(train_module.DEFAULT_CONFIG_PATH)
    assert config["mode"] == "cv"
    assert config["predefined_split"] == {
        "train_dataset_path": None,
        "validation_dataset_path": None,
        "test_dataset_path": None,
    }


def test_yaml_predefined_paths_are_relative_to_project_root() -> None:
    """配置文件中的相对路径沿用项目根目录约定，而不是终端当前目录。"""

    config = train_module.load_config(train_module.DEFAULT_CONFIG_PATH)
    config["predefined_split"] = {
        "train_dataset_path": "data/train.jsonl",
        "validation_dataset_path": "data/validation.jsonl",
        "test_dataset_path": "data/test.jsonl",
    }
    train_module._resolve_config_paths(config)

    for role in ("train", "validation", "test"):
        assert config["predefined_split"][f"{role}_dataset_path"] == str(
            (train_module.PROJECT_ROOT / "data" / f"{role}.jsonl").resolve()
        )


def test_predefined_rejects_sample_id_overlap_between_roles() -> None:
    """跨角色重复样本必须在训练和清理旧产物之前被拒绝。"""

    datasets = {
        "train": SampleIdDataset(["train_only", "leaked"]),
        "validation": SampleIdDataset(["validation_only"]),
        "test": SampleIdDataset(["leaked"]),
    }
    with pytest.raises(ValueError, match="数据泄漏"):
        train_module._validate_predefined_sample_ids(datasets)  # type: ignore[arg-type]


def test_predefined_training_uses_declared_roles_and_skips_final_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """训练、early stopping、测试和 checkpoint 元数据必须严格遵循三文件角色。"""

    dataset = PredefinedDataset()
    split = _predefined_split()
    config = _predefined_config(tmp_path)
    recorded: dict[str, Any] = {"prediction_indices": []}

    def fake_relative_floor(
        received_dataset: Dataset,
        indices: np.ndarray,
        settings: dict[str, Any],
    ) -> float:
        """记录 loss 分母只读取训练索引。"""

        assert received_dataset is dataset
        assert settings is config["training"]
        recorded["floor_indices"] = np.asarray(indices).copy()
        return 0.5

    def fake_fit_with_early_stopping(**kwargs: Any) -> SimpleNamespace:
        """替代耗时训练并保存 checkpoint metadata。"""

        recorded["train_indices"] = np.asarray(kwargs["train_indices"]).copy()
        recorded["validation_indices"] = np.asarray(
            kwargs["validation_indices"]
        ).copy()
        recorded["checkpoint_metadata"] = kwargs["checkpoint_metadata"]
        best_checkpoint = Path(kwargs["artifact_dir"]) / "best.pt"
        best_checkpoint.write_bytes(b"predefined-test")
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
        """返回足以生成聚合指标的确定性预测，并记录评估索引。"""

        del model, args, kwargs
        assert received_dataset is dataset
        copied_indices = np.asarray(indices).copy()
        recorded["prediction_indices"].append(copied_indices)
        predictions: list[dict[str, Any]] = []
        coefficients: list[dict[str, Any]] = []
        for index in copied_indices:
            sample = dataset[int(index)]
            target_drag = float(int(index) + 1)
            predictions.append(
                {
                    "source_index": int(sample["source_index"]),
                    "sample_id": str(sample["sample_id"]),
                    "flow_speed": float(sample["flow_speed"]),
                    "raw_target_drag": target_drag,
                    "target_drag": target_drag,
                    "predicted_drag": target_drag,
                    "isolated_drag": 1.0,
                }
            )
            coefficients.append(
                {
                    "source_index": int(sample["source_index"]),
                    "plant_index": 0,
                    "latent_coefficient": 1.0,
                }
            )
        return PredictionResult(
            metrics={"RMSE_D": 0.0},
            predictions=predictions,
            plant_coefficients=coefficients,
        )

    def fail_if_final_retraining_runs(*args: Any, **kwargs: Any) -> None:
        """predefined 若错误把 test 加入 final 重训，应立即失败。"""

        del args, kwargs
        raise AssertionError("predefined 模式不应调用 fit_fixed_epochs")

    monkeypatch.setattr(
        train_module, "_fit_training_relative_floor", fake_relative_floor
    )
    monkeypatch.setattr(train_module, "_new_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        train_module, "fit_with_early_stopping", fake_fit_with_early_stopping
    )
    monkeypatch.setattr(train_module, "predict_dataset", fake_predict_dataset)
    monkeypatch.setattr(
        train_module, "fit_fixed_epochs", fail_if_final_retraining_runs
    )
    monkeypatch.setattr(train_module, "write_prediction_result", lambda *args: None)
    monkeypatch.setattr(
        train_module, "write_drag_comparison_plot", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        train_module,
        "write_drag_comparison_by_flow_speed_plot",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        train_module, "_write_metrics_by_flow_speed", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        train_module, "_print_metrics_by_flow_speed_table", lambda *args: None
    )

    train_module.run_cross_validation(
        dataset,
        config,
        tmp_path,
        prepared_splits=(train_module.PREDEFINED_SPLIT_MODE, [split]),
    )

    assert np.array_equal(recorded["train_indices"], split.train_indices)
    assert np.array_equal(recorded["floor_indices"], split.train_indices)
    assert np.array_equal(
        recorded["validation_indices"], split.validation_indices
    )
    assert len(recorded["prediction_indices"]) == 2
    assert np.array_equal(
        recorded["prediction_indices"][0], split.validation_indices
    )
    assert np.array_equal(recorded["prediction_indices"][1], split.test_indices)

    scaler = json.loads(
        (tmp_path / "predefined" / "scaler.json").read_text("utf-8")
    )
    assert scaler["mean"][0] == pytest.approx(0.2)
    assert (tmp_path / "predefined" / "best.pt").is_file()
    assert not (tmp_path / "final_model.pt").exists()
    assert not (tmp_path / "final_scaler.json").exists()

    metrics = json.loads(
        (tmp_path / "predefined_metrics.json").read_text("utf-8")
    )
    assert metrics["split_mode"] == "predefined"
    assert metrics["validation_test_overlap"] is False
    assert metrics["final_retraining_performed"] is False
    assert metrics["dataset_sample_counts"] == {
        "train": 3,
        "validation": 2,
        "test": 2,
    }
    assert (tmp_path / "predefined_predictions.csv").is_file()
    assert (tmp_path / "predefined_drag_parity_model.png").is_file()
    assert (tmp_path / "predefined_drag_parity_baseline.png").is_file()
    assert (tmp_path / "predefined_plant_coefficients.csv").is_file()

    metadata = recorded["checkpoint_metadata"]
    assert metadata["checkpoint_role"] == "predefined"
    assert metadata["dataset_path"] == str(tmp_path / "test.jsonl")
    assert metadata["evaluation_source_indices"] == [0, 1]
    assert metadata["evaluation_sample_ids"] == ["predefined_5", "predefined_6"]
    assert metadata["final_retraining_performed"] is False

    with (tmp_path / "predefined_split_assignments.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["role"] for row in rows] == [
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
    ]
    assert [int(row["source_index"]) for row in rows[-2:]] == [0, 1]


def test_main_predefined_validates_before_cleaning_and_reports_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """总入口应先准备三数据集，再清理旧结果并提示 predefined/best.pt。"""

    managed_root = tmp_path / "managed"
    output_dir = managed_root / "run"
    output_dir.mkdir(parents=True)
    old_checkpoint = output_dir / "fold_4" / "best.pt"
    old_checkpoint.parent.mkdir()
    old_checkpoint.write_bytes(b"old")

    config = _predefined_config(tmp_path)
    config["data"]["physics_config_path"] = str(tmp_path / "physical.yaml")
    config["output"] = {"artifact_dir": str(output_dir)}
    args = argparse.Namespace(
        mode="predefined",
        config=str(tmp_path / "config.yaml"),
        data=None,
        train_data=str(tmp_path / "train.jsonl"),
        validation_data=str(tmp_path / "validation.jsonl"),
        test_data=str(tmp_path / "test.jsonl"),
        physics_config=None,
        artifact_dir=None,
        device=None,
        batch_size=None,
        max_epochs=None,
        seed=None,
        resume_checkpoint=None,
    )
    dataset = PredefinedDataset()
    prepared = (train_module.PREDEFINED_SPLIT_MODE, [_predefined_split()])
    events: list[str] = []

    monkeypatch.setattr(train_module, "parse_args", lambda: args)
    monkeypatch.setattr(train_module, "load_config", lambda path: config)
    monkeypatch.setattr(train_module, "_resolve_config_paths", lambda value: None)
    monkeypatch.setattr(train_module, "_apply_cli_overrides", lambda value, cli: None)
    monkeypatch.setattr(
        train_module,
        "load_physical_config",
        lambda path: SimpleNamespace(to_dict=lambda: {"states": {}}),
    )

    def fake_prepare_datasets(
        received_config: dict[str, Any], physical_config: object
    ) -> tuple[Dataset, tuple[str, list[GroupSplit]], dict[str, Any]]:
        """确认三数据集验证发生时旧 checkpoint 仍存在。"""

        assert received_config is config
        assert physical_config is not None
        assert old_checkpoint.is_file()
        events.append("datasets")
        return dataset, prepared, {}

    def fake_run(
        received_dataset: Dataset,
        received_config: dict[str, Any],
        received_output: Path,
        prepared_splits: tuple[str, list[GroupSplit]] | None = None,
    ) -> None:
        """确认 fresh 清理完成，并创建推荐 checkpoint 占位文件。"""

        assert received_dataset is dataset
        assert received_config is config
        assert prepared_splits is prepared
        assert not old_checkpoint.exists()
        checkpoint = received_output / "predefined" / "best.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"new")
        events.append("run")

    monkeypatch.setattr(
        train_module, "_prepare_predefined_datasets", fake_prepare_datasets
    )
    monkeypatch.setattr(train_module, "run_cross_validation", fake_run)
    monkeypatch.setattr(
        train_module, "DEFAULT_MANAGED_ARTIFACT_ROOT", managed_root
    )

    train_module.main()

    assert events == ["datasets", "run"]
    output = capsys.readouterr().out
    assert f"推荐 checkpoint：{output_dir.resolve() / 'predefined' / 'best.pt'}" in output


def test_predefined_evaluation_prefers_stable_sample_ids() -> None:
    """predefined checkpoint 应按 sample_id 恢复 test 顺序而非依赖当前行号。"""

    dataset = SampleIdDataset(["test_b", "unused", "test_a"])
    # 补充 evaluator 只在 source-index fallback 中需要的字段；本测试会优先走 ID。
    dataset_with_sources = [
        {"sample_id": dataset[index]["sample_id"], "source_index": index}
        for index in range(len(dataset))
    ]

    class EvaluationDataset(Dataset):
        """为评估范围选择提供 sample_id 和 source_index。"""

        def __len__(self) -> int:
            """返回三条样本。"""

            return len(dataset_with_sources)

        def __getitem__(self, index: int) -> dict[str, Any]:
            """按当前位置返回评估元数据。"""

            return dataset_with_sources[index]

    indices, scope = evaluate_module._select_evaluation_indices(
        EvaluationDataset(),  # type: ignore[arg-type]
        {
            "checkpoint_role": "predefined",
            "evaluation_source_indices": [0, 1],
            "evaluation_sample_ids": ["test_a", "test_b"],
        },
        external_data=False,
    )
    assert indices.tolist() == [2, 0]
    assert scope == "held_out"


def test_evaluate_cli_supports_top_level_defaults_and_validates_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """评估脚本的顶部变量可直接配置，CLI 数值仍需通过边界检查。"""

    checkpoint = tmp_path / "best.pt"
    dataset = tmp_path / "test.jsonl"
    monkeypatch.setattr(evaluate_module, "DEFAULT_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(evaluate_module, "DEFAULT_DATA_PATH", dataset)
    monkeypatch.setattr(evaluate_module, "DEFAULT_EXTERNAL_DATA", True)

    args = evaluate_module.parse_args([])
    assert args.checkpoint == str(checkpoint)
    assert args.data == str(dataset)
    assert args.external_data is True

    with pytest.raises(ValueError, match="batch-size"):
        evaluate_module.parse_args(
            ["--checkpoint", str(checkpoint), "--batch-size", "0"]
        )
    with pytest.raises(ValueError, match="num-workers"):
        evaluate_module.parse_args(
            ["--checkpoint", str(checkpoint), "--num-workers", "-1"]
        )


def test_predefined_test_hash_mismatch_requires_external_mode(
    tmp_path: Path,
) -> None:
    """自动复评保存的 test 文件时必须发现内容变化，外部模式则允许新文件。"""

    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text('{"changed": true}\n', encoding="utf-8")
    metadata = {
        "checkpoint_role": "predefined",
        "dataset_sha256": {"test": "not-the-actual-hash"},
    }
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_module._validate_predefined_test_fingerprint(
            dataset_path, metadata, external_data=False
        )

    actual_hash = evaluate_module._validate_predefined_test_fingerprint(
        dataset_path, metadata, external_data=True
    )
    assert actual_hash == train_module._sha256_file(dataset_path)


def test_predefined_end_to_end_train_and_evaluate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用真实小模型完成一轮训练，并由 evaluate 自动复评保存的 test JSONL。"""

    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    for path, sample_id, speed in (
        (train_path, "e2e_train", 0.1),
        (validation_path, "e2e_validation", 0.2),
        (test_path, "e2e_test", 0.3),
    ):
        path.write_text(
            json.dumps(
                _valid_jsonl_record(sample_id, speed), ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )

    managed_root = tmp_path / "artifacts"
    output_dir = managed_root / "run"
    config = {
        "mode": "predefined",
        "seed": 31,
        "data": {
            "dataset_path": str(tmp_path / "unused.jsonl"),
            "physics_config_path": str(
                train_module.PROJECT_ROOT
                / "model"
                / "configs"
                / "physical.yaml"
            ),
            "negative_target_policy": "clamp_to_zero",
        },
        "model": dict(SMALL_MODEL_CONFIG),
        "training": {
            "loss": "stabilized_relative_mse",
            "relative_loss_floor_strategy": "train_positive_quantile",
            "relative_loss_floor_quantile": 0.05,
            "batch_size": 1,
            "mirror_augmentation": False,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "gradient_clip": 1.0,
            "num_workers": 0,
            "device": "cpu",
            "progress_interval_epochs": 1,
            "loss_plot_interval_epochs": 1,
            "checkpoint_interval_epochs": 1,
        },
        "cross_validation": {
            "split_mode": "plant_count",
            "n_splits": 5,
            "validation_fraction": 0.2,
        },
        "predefined_split": {
            "train_dataset_path": str(train_path),
            "validation_dataset_path": str(validation_path),
            "test_dataset_path": str(test_path),
        },
        "output": {"artifact_dir": str(output_dir)},
    }
    train_args = argparse.Namespace(
        mode=None,
        config=str(tmp_path / "config.yaml"),
        data=None,
        train_data=None,
        validation_data=None,
        test_data=None,
        physics_config=None,
        artifact_dir=str(output_dir),
        device="cpu",
        batch_size=1,
        max_epochs=1,
        seed=31,
        resume_checkpoint=None,
    )
    monkeypatch.setattr(train_module, "parse_args", lambda: train_args)
    monkeypatch.setattr(train_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        train_module, "DEFAULT_MANAGED_ARTIFACT_ROOT", managed_root
    )

    train_module.main()

    best_checkpoint = output_dir / "predefined" / "best.pt"
    assert best_checkpoint.is_file()
    assert (output_dir / "predefined_metrics.json").is_file()
    assert not (output_dir / "final_model.pt").exists()

    evaluation_dir = output_dir / "evaluation_test"
    evaluate_args = argparse.Namespace(
        checkpoint=str(best_checkpoint),
        data=None,
        external_data=False,
        output_dir=str(evaluation_dir),
        device="cpu",
        batch_size=1,
        num_workers=0,
    )
    monkeypatch.setattr(evaluate_module, "parse_args", lambda: evaluate_args)
    evaluate_module.main()

    context = json.loads(
        (evaluation_dir / "evaluation_context.json").read_text("utf-8")
    )
    metrics = json.loads(
        (evaluation_dir / "evaluation_metrics.json").read_text("utf-8")
    )
    assert context["evaluation_scope"] == "held_out"
    assert context["checkpoint_role"] == "predefined"
    assert context["dataset_path"] == str(test_path.resolve())
    assert context["sample_count"] == 1
    assert metrics["sample_count"] == 1

    # 同一 checkpoint 也应接受额外的、与测试集不同的 JSONL，
    # 并将这次运行标记为 external，而不是误认为又评估了一次原测试集。
    external_path = tmp_path / "external.jsonl"
    external_path.write_text(
        json.dumps(_valid_jsonl_record("e2e_external", 0.4), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    external_dir = output_dir / "evaluation_external"
    evaluate_args.data = str(external_path)
    evaluate_args.external_data = True
    evaluate_args.output_dir = str(external_dir)
    evaluate_module.main()

    external_context = json.loads(
        (external_dir / "evaluation_context.json").read_text("utf-8")
    )
    external_metrics = json.loads(
        (external_dir / "evaluation_metrics.json").read_text("utf-8")
    )
    assert external_context["evaluation_scope"] == "external"
    assert external_context["dataset_path"] == str(external_path.resolve())
    assert external_context["sample_count"] == 1
    assert external_metrics["sample_count"] == 1
