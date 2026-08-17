"""HydroTransformer 训练命令行入口。

示例：
    python -m model.train --mode cv
    python -m model.train --mode overfit --max-epochs 300
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.hydro.data import HydroDataset, collate_hydro_samples
from model.models import HydroTransformer
from model.training.config import load_config, save_config_snapshot
from model.training.metrics import compute_regression_metrics
from model.training.splits import GroupSplit, build_group_kfold_splits
from model.training.trainer import (
    GlobalFeatureScaler,
    fit_fixed_epochs,
    fit_with_early_stopping,
    predict_dataset,
    resolve_device,
    set_reproducible_seed,
    write_prediction_result,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "model" / "configs" / "base.yaml"


def parse_args() -> argparse.Namespace:
    """定义并解析训练命令行参数。"""

    parser = argparse.ArgumentParser(description="训练多水草 HydroTransformer。")
    parser.add_argument(
        "--mode",
        choices=("cv", "overfit"),
        default="cv",
        help="cv 执行5折评估和全量重训；overfit 仅检查前32条样本。",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML 配置。")
    parser.add_argument("--data", help="覆盖配置中的总 CSV 路径。")
    parser.add_argument("--input-csv", help="覆盖 Experiment/input.csv 路径。")
    parser.add_argument("--artifact-dir", help="覆盖产物目录。")
    parser.add_argument("--device", help="auto、cpu 或 cuda。")
    parser.add_argument("--batch-size", type=int, help="覆盖 batch size。")
    parser.add_argument("--max-epochs", type=int, help="覆盖最大 epoch。")
    parser.add_argument("--seed", type=int, help="覆盖全局随机种子。")
    parser.add_argument(
        "--resume-checkpoint",
        help="仅 overfit 模式使用，从 last.pt 或同结构 checkpoint 恢复。",
    )
    return parser.parse_args()


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """将用户明确给出的 CLI 参数写回最终配置。"""

    if args.data:
        config["data"]["csv_path"] = str(Path(args.data).resolve())
    if args.input_csv:
        config["data"]["input_csv_path"] = str(Path(args.input_csv).resolve())
    if args.artifact_dir:
        config["output"]["artifact_dir"] = str(Path(args.artifact_dir).resolve())
    if args.device:
        config["training"]["device"] = args.device
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.seed is not None:
        config["seed"] = args.seed


def _resolve_config_paths(config: dict[str, Any]) -> None:
    """将 YAML 中的相对默认路径统一解释为相对于项目根目录。

    CLI 覆盖在此函数之后应用，因此用户命令行显式传入的相对路径仍按当前工作目录
    解析。最终写入配置快照的三个路径均为绝对路径。
    """

    for section, key in (
        ("data", "csv_path"),
        ("data", "input_csv_path"),
        ("output", "artifact_dir"),
    ):
        configured_path = Path(config[section][key])
        if not configured_path.is_absolute():
            configured_path = PROJECT_ROOT / configured_path
        config[section][key] = str(configured_path.resolve())


def _collect_groups(dataset: HydroDataset) -> np.ndarray:
    """从数据集元数据收集每行 ``model_id``。"""

    return np.asarray(
        [int(dataset[index]["model_id"]) for index in range(len(dataset))],
        dtype=np.int64,
    )


def _write_scaler(path: Path, scaler: GlobalFeatureScaler) -> None:
    """保存人类可读的训练集标准化统计。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scaler.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_fold_manifest(
    path: Path, dataset: HydroDataset, splits: list[GroupSplit]
) -> None:
    """保存每个 fold 的样本角色，方便检查 model 泄漏。"""

    rows: list[dict[str, Any]] = []
    for split in splits:
        for role, indices in (
            ("train", split.train_indices),
            ("validation", split.validation_indices),
            ("test", split.test_indices),
        ):
            for index in indices:
                sample = dataset[int(index)]
                rows.append(
                    {
                        "fold": split.fold,
                        "role": role,
                        "dataset_index": int(index),
                        "source_index": int(sample["source_index"]),
                        "model_id": int(sample["model_id"]),
                        "angle": int(sample["angle"]),
                        "flow_speed": float(sample["flow_speed"]),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _new_model(model_config: dict[str, Any], seed: int) -> HydroTransformer:
    """先固定随机种子，再创建独立模型，保证初始化可复现。"""

    set_reproducible_seed(seed)
    return HydroTransformer(**model_config)


def _checkpoint_metadata(
    config: dict[str, Any],
    checkpoint_role: str,
    evaluation_source_indices: list[int] | None = None,
) -> dict[str, Any]:
    """构造评估恢复所需的数据来源、标签策略和 held-out 索引。"""

    return {
        "checkpoint_role": checkpoint_role,
        "dataset_path": config["data"]["csv_path"],
        "input_csv_path": config["data"]["input_csv_path"],
        "negative_target_policy": config["data"]["negative_target_policy"],
        "evaluation_source_indices": evaluation_source_indices,
    }


def run_overfit(
    dataset: HydroDataset, config: dict[str, Any], output_dir: Path, resume: str | None
) -> None:
    """在最多32条样本上同时训练和验证，检查实现是否有能力拟合。"""

    indices = np.arange(min(32, len(dataset)), dtype=np.int64)
    if indices.size == 0:
        raise ValueError("数据集为空，无法执行 overfit 检查。")
    scaler = GlobalFeatureScaler.fit(dataset, indices)
    _write_scaler(output_dir / "overfit" / "scaler.json", scaler)
    settings = dict(config["training"])
    # overfit 是诊断，不应因 patience 提前中断；允许用户用 max_epochs 控制耗时。
    settings["early_stopping_patience"] = int(settings["max_epochs"]) + 1
    overfit_seed = int(config["seed"])
    model = _new_model(config["model"], overfit_seed)
    source_indices = [
        int(dataset[int(index)]["source_index"]) for index in indices
    ]
    result = fit_with_early_stopping(
        model=model,
        dataset=dataset,
        train_indices=indices,
        validation_indices=indices,
        collate_fn=collate_hydro_samples,
        settings=settings,
        scaler=scaler,
        artifact_dir=output_dir / "overfit",
        model_config=config["model"],
        seed=overfit_seed,
        resume_from=resume,
        checkpoint_metadata=_checkpoint_metadata(
            config, "overfit", source_indices
        ),
    )
    prediction = predict_dataset(
        model,
        dataset,
        indices,
        collate_hydro_samples,
        int(settings["batch_size"]),
        int(settings["num_workers"]),
        scaler,
        resolve_device(str(settings["device"])),
    )
    write_prediction_result(prediction, output_dir / "overfit", "overfit")
    print(
        f"Overfit 完成：best_epoch={result.best_epoch}, "
        f"C-MSE={result.best_validation_loss:.6g}"
    )


def run_cross_validation(
    dataset: HydroDataset, config: dict[str, Any], output_dir: Path
) -> None:
    """执行无 model_id 泄漏的5折评估，随后按中位最佳 epoch 全量重训。"""

    groups = _collect_groups(dataset)
    cv_config = config["cross_validation"]
    splits = build_group_kfold_splits(
        groups,
        n_splits=int(cv_config["n_splits"]),
        validation_fraction=float(cv_config["validation_fraction"]),
        seed=int(config["seed"]),
    )
    _write_fold_manifest(output_dir / "fold_assignments.csv", dataset, splits)

    all_prediction_rows: list[dict[str, Any]] = []
    all_coefficient_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    best_epochs: list[int] = []
    for split in splits:
        fold_dir = output_dir / f"fold_{split.fold}"
        scaler = GlobalFeatureScaler.fit(dataset, split.train_indices)
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_scaler(fold_dir / "scaler.json", scaler)

        fold_seed = int(config["seed"]) + split.fold
        model = _new_model(config["model"], fold_seed)
        test_source_indices = [
            int(dataset[int(index)]["source_index"])
            for index in split.test_indices
        ]
        fit_result = fit_with_early_stopping(
            model=model,
            dataset=dataset,
            train_indices=split.train_indices,
            validation_indices=split.validation_indices,
            collate_fn=collate_hydro_samples,
            settings=config["training"],
            scaler=scaler,
            artifact_dir=fold_dir,
            model_config=config["model"],
            seed=fold_seed,
            checkpoint_metadata=_checkpoint_metadata(
                config, "fold", test_source_indices
            ),
        )
        test_result = predict_dataset(
            model,
            dataset,
            split.test_indices,
            collate_hydro_samples,
            int(config["training"]["batch_size"]),
            int(config["training"]["num_workers"]),
            scaler,
            resolve_device(str(config["training"]["device"])),
        )
        for row in test_result.predictions:
            row["fold"] = split.fold
        for row in test_result.plant_coefficients:
            row["fold"] = split.fold
        write_prediction_result(test_result, fold_dir, "test")
        all_prediction_rows.extend(test_result.predictions)
        all_coefficient_rows.extend(test_result.plant_coefficients)
        fold_summaries.append(
            {
                "fold": split.fold,
                "best_epoch": fit_result.best_epoch,
                "best_validation_C_MSE": fit_result.best_validation_loss,
                **test_result.metrics,
            }
        )
        best_epochs.append(fit_result.best_epoch)
        print(
            f"Fold {split.fold} 完成：best_epoch={fit_result.best_epoch}, "
            f"test_RMSE_D={test_result.metrics['RMSE_D']:.6g}"
        )

    aggregate_metrics = compute_regression_metrics(
        [row["target_drag"] for row in all_prediction_rows],
        [row["predicted_drag"] for row in all_prediction_rows],
        [row["isolated_drag"] for row in all_prediction_rows],
    )
    aggregate_result = {
        "aggregate": aggregate_metrics,
        "folds": fold_summaries,
    }
    (output_dir / "cv_metrics.json").write_text(
        json.dumps(aggregate_result, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    _write_rows(output_dir / "cv_predictions.csv", all_prediction_rows)
    _write_rows(output_dir / "cv_plant_coefficients.csv", all_coefficient_rows)

    # 中位数若为 x.5，round 采用银行家舍入并不直观，因此显式四舍五入。
    final_epochs = int(np.floor(statistics.median(best_epochs) + 0.5))
    all_indices = np.arange(len(dataset), dtype=np.int64)
    full_scaler = GlobalFeatureScaler.fit(dataset, all_indices)
    _write_scaler(output_dir / "final_scaler.json", full_scaler)
    final_seed = int(config["seed"])
    final_model = _new_model(config["model"], final_seed)
    final_checkpoint = fit_fixed_epochs(
        final_model,
        dataset,
        all_indices,
        collate_hydro_samples,
        config["training"],
        full_scaler,
        output_dir,
        config["model"],
        int(config["seed"]),
        final_epochs,
        checkpoint_metadata=_checkpoint_metadata(config, "final"),
    )
    print(f"交叉验证及全量重训完成：{final_checkpoint}")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """写合并后的 CSV 明细。"""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """加载配置和数据，派发 overfit 或 CV 流程。"""

    args = parse_args()
    if args.mode == "cv" and args.resume_checkpoint:
        raise ValueError("--resume-checkpoint 目前仅用于 overfit 模式。")
    config = load_config(args.config)
    _resolve_config_paths(config)
    _apply_cli_overrides(config, args)
    output_dir = Path(config["output"]["artifact_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_dir / "resolved_config.json")
    dataset = HydroDataset(
        config["data"]["csv_path"],
        input_csv_path=config["data"]["input_csv_path"],
        negative_target_policy=config["data"]["negative_target_policy"],
    )
    if args.mode == "overfit":
        run_overfit(dataset, config, output_dir, args.resume_checkpoint)
    else:
        run_cross_validation(dataset, config, output_dir)


if __name__ == "__main__":
    main()
