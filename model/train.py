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
from model.hydro.physics import load_physical_config
from model.models import HydroTransformer
from model.training.config import load_config, save_config_snapshot
from model.training.losses import (
    DEFAULT_RELATIVE_FLOOR_QUANTILE,
    RELATIVE_FLOOR_STRATEGY,
    fit_relative_drag_floor,
)
from model.training.metrics import compute_regression_metrics
from model.training.splits import (
    FLOW_SPEED_SPLIT_MODE,
    GroupSplit,
    build_cross_validation_splits,
)
from model.training.trainer import (
    LOSS_NAME,
    GlobalFeatureScaler,
    fit_fixed_epochs,
    fit_with_early_stopping,
    predict_dataset,
    resolve_device,
    set_reproducible_seed,
    write_prediction_result,
)
from model.training.visualization import write_drag_comparison_plot


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "model" / "configs" / "base.yaml"


def parse_args() -> argparse.Namespace:
    """定义并解析训练命令行参数。"""

    parser = argparse.ArgumentParser(description="训练多水草 HydroTransformer。")
    parser.add_argument(
        "--mode",
        choices=("cv", "overfit"),
        default="cv",
        help=(
            "cv 按配置执行数据划分与评估；flow_speed 固定运行一折且不全量重训，"
            "其他模式执行多折评估和全量重训。overfit 仅检查前32条样本。"
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML 配置。")
    parser.add_argument("--data", help="覆盖配置中的总 CSV 路径。")
    parser.add_argument("--physics-config", help="覆盖双状态物理参数 YAML 路径。")
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
    if args.physics_config:
        config["data"]["physics_config_path"] = str(
            Path(args.physics_config).resolve()
        )
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
        ("data", "physics_config_path"),
        ("output", "artifact_dir"),
    ):
        configured_path = Path(config[section][key])
        if not configured_path.is_absolute():
            configured_path = PROJECT_ROOT / configured_path
        config[section][key] = str(configured_path.resolve())


def _collect_split_labels(
    dataset: HydroDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """收集四种划分模式需要的构型、水草根数和流速标签。

    参数：
        dataset: 完整的 :class:`HydroDataset`。每条样本必须包含 ``model_id``、
            只保留有效植株的 ``positions`` 和以 m/s 为单位的 ``flow_speed``。

    返回值：
        三元组 ``(model_ids, plant_counts, flow_speeds)``。三个 NumPy 数组都与
        数据集等长，第 ``i`` 个元素分别表示第 ``i`` 条样本的构型编号、有效水草
        根数与流速。
    """

    model_ids: list[int] = []
    plant_counts: list[int] = []
    flow_speeds: list[float] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        model_ids.append(int(sample["model_id"]))
        # Dataset 的 positions 已经移除了 input.csv 中值为 0 的空位，所以第一维
        # 就是该样本的真实水草根数，不会把 padding 位置误算为水草。
        plant_counts.append(int(sample["positions"].shape[0]))
        # Dataset 已把 CSV 的 flow_speed 解析为 Python float；这里保留 float64
        # 标签供划分逻辑做带容差比较，不使用模型输入 Tensor 的 dtype。
        flow_speeds.append(float(sample["flow_speed"]))
    return (
        np.asarray(model_ids, dtype=np.int64),
        np.asarray(plant_counts, dtype=np.int64),
        np.asarray(flow_speeds, dtype=np.float64),
    )


def _write_scaler(path: Path, scaler: GlobalFeatureScaler) -> None:
    """保存人类可读的训练集标准化统计。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scaler.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_fold_manifest(
    path: Path,
    dataset: HydroDataset,
    splits: list[GroupSplit],
    split_mode: str,
) -> None:
    """保存每个 fold 的样本角色和分组字段，方便人工检查数据泄漏。

    参数：
        path: 输出 CSV 路径。
        dataset: 提供样本元数据的完整数据集。
        splits: 每折的 train、validation、test 索引。
        split_mode: 本次使用的 ``sample``、``model``、``plant_count`` 或
            ``flow_speed`` 模式。
    """

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
                        "plant_count": int(sample["positions"].shape[0]),
                        "split_mode": split_mode,
                        "state_id": int(sample["state_id"]),
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


def _fit_training_relative_floor(
    dataset: HydroDataset,
    train_indices: np.ndarray,
    settings: dict[str, Any],
) -> float:
    """仅使用当前训练索引拟合稳定化相对 loss 的分母下限。

    参数：
        dataset: 包含 ``target_drag`` 的完整数据集。
        train_indices: 当前 overfit、fold 或 final 的训练样本索引。
        settings: 训练配置，提供分母策略与正标签分位数。

    返回值：
        当前训练任务固定使用的有限正分母下限。
    """

    strategy = str(
        settings.get("relative_loss_floor_strategy", RELATIVE_FLOOR_STRATEGY)
    )
    if strategy != RELATIVE_FLOOR_STRATEGY:
        raise ValueError(
            "当前只支持 relative_loss_floor_strategy="
            f"{RELATIVE_FLOOR_STRATEGY!r}，实际收到 {strategy!r}。"
        )
    quantile = float(
        settings.get(
            "relative_loss_floor_quantile", DEFAULT_RELATIVE_FLOOR_QUANTILE
        )
    )
    return fit_relative_drag_floor(dataset, train_indices, quantile=quantile)


def _checkpoint_metadata(
    config: dict[str, Any],
    checkpoint_role: str,
    relative_floor: float,
    evaluation_source_indices: list[int] | None = None,
) -> dict[str, Any]:
    """构造评估恢复所需的数据来源、标签策略和 held-out 索引。"""

    split_mode = str(config["cross_validation"].get("split_mode", "model"))
    validation_test_overlap = split_mode == FLOW_SPEED_SPLIT_MODE
    return {
        "checkpoint_role": checkpoint_role,
        "dataset_path": config["data"]["csv_path"],
        "negative_target_policy": config["data"]["negative_target_policy"],
        # 保存训练目标的完整语义；评估可以忽略，断点续训必须严格核对。
        "loss_name": LOSS_NAME,
        "relative_floor": float(relative_floor),
        "relative_floor_strategy": config["training"].get(
            "relative_loss_floor_strategy", RELATIVE_FLOOR_STRATEGY
        ),
        "relative_floor_quantile": float(
            config["training"].get(
                "relative_loss_floor_quantile",
                DEFAULT_RELATIVE_FLOOR_QUANTILE,
            )
        ),
        # 保存完整解析结果而不只保存文件路径，避免物理 YAML 后续修改影响历史模型。
        "physical_config": config["resolved_physical_config"],
        # 保存划分语义用于科研审计。普通模式的 evaluation_source_indices 是独立
        # held-out test；flow_speed 模式则明确标记 validation/test 完全重叠。
        "split_mode": split_mode,
        "validation_test_overlap": validation_test_overlap,
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
    relative_floor = _fit_training_relative_floor(dataset, indices, settings)
    overfit_seed = int(config["seed"])
    model = _new_model(config["model"], overfit_seed)
    print(
        f"开始 Overfit 训练：samples={indices.size}, "
        f"max_epochs={int(settings['max_epochs'])}, "
        f"relative_floor={relative_floor:.6g}",
        flush=True,
    )
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
        relative_floor=relative_floor,
        resume_from=resume,
        checkpoint_metadata=_checkpoint_metadata(
            config, "overfit", relative_floor, source_indices
        ),
        progress_label="Overfit",
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
        f"Relative-MSE={result.best_validation_loss:.6g}"
    )


def run_cross_validation(
    dataset: HydroDataset, config: dict[str, Any], output_dir: Path
) -> None:
    """按配置执行划分与评估，并按模式决定是否在全量数据上重训。"""

    model_ids, plant_counts, flow_speeds = _collect_split_labels(dataset)
    cv_config = config["cross_validation"]
    split_mode = str(cv_config.get("split_mode", "model"))
    validation_test_overlap = split_mode == FLOW_SPEED_SPLIT_MODE
    if validation_test_overlap:
        print(
            "flow_speed 模式：0.1/0.2/0.3 m/s 用于训练，完整 0.4 m/s "
            "同时用于 validation 和 test；test 指标不是独立 held-out 指标，"
            "并且本次不会生成 final_model.pt。",
            flush=True,
        )
    splits = build_cross_validation_splits(
        split_mode=split_mode,
        n_samples=len(dataset),
        model_ids=model_ids,
        plant_counts=plant_counts,
        flow_speeds=flow_speeds,
        n_splits=int(cv_config["n_splits"]),
        validation_fraction=float(cv_config["validation_fraction"]),
        seed=int(config["seed"]),
    )
    _write_fold_manifest(
        output_dir / "fold_assignments.csv", dataset, splits, split_mode
    )

    all_prediction_rows: list[dict[str, Any]] = []
    all_coefficient_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    best_epochs: list[int] = []
    for split in splits:
        fold_dir = output_dir / f"fold_{split.fold}"
        scaler = GlobalFeatureScaler.fit(dataset, split.train_indices)
        relative_floor = _fit_training_relative_floor(
            dataset, split.train_indices, config["training"]
        )
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_scaler(fold_dir / "scaler.json", scaler)

        fold_seed = int(config["seed"]) + split.fold
        model = _new_model(config["model"], fold_seed)
        print(
            f"开始 Fold {split.fold}：split_mode={split_mode}, "
            f"train={split.train_indices.size}, "
            f"validation={split.validation_indices.size}, "
            f"test={split.test_indices.size}, "
            f"max_epochs={int(config['training']['max_epochs'])}, "
            f"relative_floor={relative_floor:.6g}",
            flush=True,
        )
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
            relative_floor=relative_floor,
            checkpoint_metadata=_checkpoint_metadata(
                config, "fold", relative_floor, test_source_indices
            ),
            progress_label=f"Fold {split.fold}",
        )
        # 两个角色始终使用相同的最优模型、train-only scaler 和 relative_floor。
        # 普通模式的 test 是独立 held-out 集合；flow_speed 模式则按实验协议让
        # validation/test 复用完整 0.4 m/s 数据，因此不能作独立泛化解释。
        validation_result = predict_dataset(
            model,
            dataset,
            split.validation_indices,
            collate_hydro_samples,
            int(config["training"]["batch_size"]),
            int(config["training"]["num_workers"]),
            scaler,
            resolve_device(str(config["training"]["device"])),
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
        for row in validation_result.predictions:
            row["fold"] = split.fold
        for row in validation_result.plant_coefficients:
            row["fold"] = split.fold
        for row in test_result.predictions:
            row["fold"] = split.fold
        for row in test_result.plant_coefficients:
            row["fold"] = split.fold
        write_prediction_result(validation_result, fold_dir, "validation")
        write_prediction_result(test_result, fold_dir, "test")
        write_drag_comparison_plot(
            validation_result.predictions,
            fold_dir / "validation_drag_comparison.png",
            title=f"Fold {split.fold} validation drag comparison",
        )
        write_drag_comparison_plot(
            test_result.predictions,
            fold_dir / "test_drag_comparison.png",
            title=f"Fold {split.fold} test drag comparison",
        )
        all_prediction_rows.extend(test_result.predictions)
        all_coefficient_rows.extend(test_result.plant_coefficients)
        fold_summaries.append(
            {
                "fold": split.fold,
                "best_epoch": fit_result.best_epoch,
                "best_validation_relative_MSE": fit_result.best_validation_loss,
                "relative_floor": relative_floor,
                "split_mode": split_mode,
                # 保留原有顶层 test 指标，兼容已经读取 cv_metrics.json 的分析脚本；
                # 同时增加两个具名对象，使 validation/test 的含义更直观。
                "validation_metrics": validation_result.metrics,
                "test_metrics": test_result.metrics,
                **test_result.metrics,
            }
        )
        best_epochs.append(fit_result.best_epoch)
        print(
            f"Fold {split.fold} 完成：best_epoch={fit_result.best_epoch}, "
            f"validation_RMSE_D={validation_result.metrics['RMSE_D']:.6g}, "
            f"test_RMSE_D={test_result.metrics['RMSE_D']:.6g}"
        )

    aggregate_metrics = compute_regression_metrics(
        [row["target_drag"] for row in all_prediction_rows],
        [row["predicted_drag"] for row in all_prediction_rows],
        [row["isolated_drag"] for row in all_prediction_rows],
    )
    aggregate_result = {
        "split_mode": split_mode,
        "validation_test_overlap": validation_test_overlap,
        # 先写 false，只有普通模式的全量重训成功完成后才更新为 true。这样即使
        # 重训中断，已落盘的指标也不会错误声称 final_model 已经生成。
        "final_retraining_performed": False,
        "aggregate": aggregate_metrics,
        "folds": fold_summaries,
    }
    metrics_path = output_dir / "cv_metrics.json"
    metrics_path.write_text(
        json.dumps(aggregate_result, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    _write_rows(output_dir / "cv_predictions.csv", all_prediction_rows)
    _write_rows(output_dir / "cv_plant_coefficients.csv", all_coefficient_rows)

    # 固定流速实验只需要 fold_0 的跨速度模型。0.4 m/s 已参与 early stopping，
    # 因此既不能再把它加入训练，也不应创建声称使用全量数据的 final checkpoint。
    if validation_test_overlap:
        print(
            "flow_speed 固定划分完成：已保存 fold_0 及汇总产物；"
            "已按配置跳过全量重训。"
        )
        return

    # 中位数若为 x.5，round 采用银行家舍入并不直观，因此显式四舍五入。
    final_epochs = int(np.floor(statistics.median(best_epochs) + 0.5))
    all_indices = np.arange(len(dataset), dtype=np.int64)
    full_scaler = GlobalFeatureScaler.fit(dataset, all_indices)
    final_relative_floor = _fit_training_relative_floor(
        dataset, all_indices, config["training"]
    )
    _write_scaler(output_dir / "final_scaler.json", full_scaler)
    final_seed = int(config["seed"])
    final_model = _new_model(config["model"], final_seed)
    print(
        f"开始全量重训：samples={all_indices.size}, epochs={final_epochs}, "
        f"relative_floor={final_relative_floor:.6g}",
        flush=True,
    )
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
        final_relative_floor,
        checkpoint_metadata=_checkpoint_metadata(
            config, "final", final_relative_floor
        ),
        progress_label="Final retraining",
    )
    # 只有 checkpoint 已成功写出后才把指标元数据更新为“已完成全量重训”。
    aggregate_result["final_retraining_performed"] = True
    metrics_path.write_text(
        json.dumps(aggregate_result, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
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
    print("CUDA available:", torch.cuda.is_available())
    args = parse_args()
    if args.mode == "cv" and args.resume_checkpoint:
        raise ValueError("--resume-checkpoint 目前仅用于 overfit 模式。")
    config = load_config(args.config)
    _resolve_config_paths(config)
    _apply_cli_overrides(config, args)
    output_dir = Path(config["output"]["artifact_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    physical_config = load_physical_config(config["data"]["physics_config_path"])
    config["resolved_physical_config"] = physical_config.to_dict()
    save_config_snapshot(config, output_dir / "resolved_config.json")
    dataset = HydroDataset(
        config["data"]["csv_path"],
        negative_target_policy=config["data"]["negative_target_policy"],
        physical_config=physical_config,
    )
    if args.mode == "overfit":
        run_overfit(dataset, config, output_dir, args.resume_checkpoint)
    else:
        run_cross_validation(dataset, config, output_dir)


if __name__ == "__main__":
    main()
