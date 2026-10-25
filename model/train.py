"""HydroTransformer 训练命令行入口。

示例：
    # 无参数时读取 model/configs/base.yaml 的 mode 和数据路径。
    python -m model.train
    python -m model.train --mode cv
    python -m model.train --mode overfit --max-epochs 300
    python -m model.train --mode predefined --train-data train.jsonl \
        --validation-data validation.jsonl --test-data test.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from model.hydro.data import HydroDataset, collate_hydro_samples
from model.hydro.physics import load_physical_config
from model.models import HydroTransformer
from model.training.artifacts import (
    prepare_fresh_artifact_directory,
    prepare_resume_artifact_directory,
)
from model.training.config import (
    DEFAULT_DRAG_PARITY_AXIS_MODE,
    DRAG_PARITY_AXIS_MODES,
    load_config,
    save_config_snapshot,
)
from model.training.losses import (
    DEFAULT_RELATIVE_FLOOR_QUANTILE,
    RELATIVE_FLOOR_STRATEGY,
    fit_relative_drag_floor,
)
from model.training.metrics import (
    compute_metrics_by_flow_speed,
    compute_regression_metrics,
)
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
from model.training.visualization import (
    write_drag_comparison_by_flow_speed_plot,
    write_drag_comparison_plot,
    write_drag_parity_plots,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "model" / "configs" / "base.yaml"
DEFAULT_MANAGED_ARTIFACT_ROOT = PROJECT_ROOT / "model" / "artifacts"
"""允许 fresh run 自动清理的项目默认产物根目录。"""

PREDEFINED_SPLIT_MODE = "predefined"
"""用户显式提供 train/validation/test 三份 JSONL 时使用的划分模式名。"""

MODEL_SOURCE_ROOT = PROJECT_ROOT / "model"
"""训练源码目录；任何自动清理目标都不能等于或包含该目录。"""

FLOW_SPEED_METRIC_COLUMNS = (
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
)
"""按流速输出 CSV 时使用的固定列顺序。"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """定义并解析训练命令行参数。"""

    parser = argparse.ArgumentParser(description="训练多水草 HydroTransformer。")
    parser.add_argument(
        "--mode",
        choices=("cv", "overfit", PREDEFINED_SPLIT_MODE),
        default=None,
        help=(
            "省略时使用 YAML 的 mode；cv 按配置执行数据划分与评估；"
            "flow_speed 固定运行一折且不全量重训，"
            "其他模式执行多折评估和全量重训。overfit 仅检查前32条样本；"
            "predefined 使用用户显式提供的 train/validation/test 三份 JSONL，"
            "只训练一个模型且不执行 final 重训。"
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML 配置。")
    parser.add_argument("--data", help="覆盖配置中的 dataset.jsonl 路径。")
    parser.add_argument(
        "--train-data",
        default=None,
        help="predefined 模式的训练集 JSONL。",
    )
    parser.add_argument(
        "--validation-data",
        default=None,
        help="predefined 模式的验证集 JSONL。",
    )
    parser.add_argument(
        "--test-data",
        default=None,
        help="predefined 模式的测试集 JSONL。",
    )
    parser.add_argument("--physics-config", help="覆盖角度阻力分组 YAML 路径。")
    parser.add_argument("--artifact-dir", help="覆盖产物目录。")
    parser.add_argument(
        "--drag-parity-axis-mode",
        choices=DRAG_PARITY_AXIS_MODES,
        default=None,
        help="覆盖散点图范围模式：independent 各图独立取范围，shared 两图共用范围。",
    )
    parser.add_argument("--device", help="auto、cpu 或 cuda。")
    parser.add_argument("--batch-size", type=int, help="覆盖 batch size。")
    parser.add_argument("--max-epochs", type=int, help="覆盖最大 epoch。")
    parser.add_argument("--seed", type=int, help="覆盖全局随机种子。")
    parser.add_argument(
        "--resume-checkpoint",
        help="仅 overfit 模式使用，从 last.pt 或同结构 checkpoint 恢复。",
    )
    return parser.parse_args(argv)


def _validate_mode_arguments(
    args: argparse.Namespace, config: dict[str, Any] | None = None
) -> None:
    """合并 YAML/CLI 的运行模式和数据路径，再验证组合是否有效。

    ``predefined`` 要求 train、validation、test 三份 JSONL 全部存在于 YAML
    或命令行中；命令行覆盖 YAML。若 CLI 未指定 mode，则以 YAML 的 mode 为准。
    其他模式继续只使用单一 ``--data``。显式三分集目前不支持断点续训，避免把
    checkpoint 的单数据集恢复语义误用于三个独立文件。
    """

    configured = config or {}
    args.mode = args.mode or configured.get("mode", "cv")
    if args.mode not in ("cv", "overfit", PREDEFINED_SPLIT_MODE):
        raise ValueError(f"未知训练 mode：{args.mode!r}。")

    # 只把命令行显式传入的三路径视为其他模式的非法参数。YAML 可以预存
    # predefined 路径而继续运行 cv，便于维护者在两种实验协议间切换。
    predefined_values = {
        "--train-data": getattr(args, "train_data", None),
        "--validation-data": getattr(args, "validation_data", None),
        "--test-data": getattr(args, "test_data", None),
    }
    supplied_predefined = [name for name, value in predefined_values.items() if value]

    if args.mode == PREDEFINED_SPLIT_MODE:
        split_config = configured.get("predefined_split") or {}
        args.train_data = args.train_data or split_config.get("train_dataset_path")
        args.validation_data = args.validation_data or split_config.get(
            "validation_dataset_path"
        )
        args.test_data = args.test_data or split_config.get("test_dataset_path")
        predefined_values = {
            "--train-data": args.train_data,
            "--validation-data": args.validation_data,
            "--test-data": args.test_data,
        }
        missing = [name for name, value in predefined_values.items() if not value]
        if missing:
            raise ValueError(
                "predefined 模式必须同时提供 --train-data、--validation-data 和 "
                f"--test-data；当前缺少：{', '.join(missing)}。"
            )
        if getattr(args, "data", None):
            raise ValueError("predefined 模式不能同时使用 --data。")
        if args.resume_checkpoint:
            raise ValueError("--resume-checkpoint 目前仅用于 overfit 模式。")
        return

    if supplied_predefined:
        raise ValueError(
            f"{args.mode} 模式不能使用预定义三数据集参数："
            f"{', '.join(supplied_predefined)}。"
        )
    if args.mode == "cv" and args.resume_checkpoint:
        raise ValueError("--resume-checkpoint 目前仅用于 overfit 模式。")


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """将用户明确给出的 CLI 参数写回最终配置。"""

    if args.data:
        config["data"]["dataset_path"] = str(Path(args.data).resolve())
    if args.physics_config:
        config["data"]["physics_config_path"] = str(
            Path(args.physics_config).resolve()
        )
    if args.artifact_dir:
        config["output"]["artifact_dir"] = str(Path(args.artifact_dir).resolve())
    # 绘图参数写回配置，确保训练配置快照记录最终生效的坐标模式。
    # 兼容外部脚本或旧测试手动构造、尚未包含新增字段的 Namespace。
    parity_axis_mode = getattr(args, "drag_parity_axis_mode", None)
    if parity_axis_mode is not None:
        config["output"]["drag_parity_axis_mode"] = parity_axis_mode
    if args.device:
        config["training"]["device"] = args.device
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.seed is not None:
        config["seed"] = args.seed
    # 把最终生效的 mode 写入快照；CLI 未指定时也明确记录从 YAML 取得的值。
    config["mode"] = args.mode
    if args.mode == PREDEFINED_SPLIT_MODE:
        # YAML 路径已在 _resolve_config_paths 中按项目根目录解析；命令行显式路径
        # 则按当前工作目录解析。最终快照和 checkpoint 都记录绝对路径。
        config["predefined_split"] = {
            "train_dataset_path": str(Path(args.train_data).resolve()),
            "validation_dataset_path": str(Path(args.validation_data).resolve()),
            "test_dataset_path": str(Path(args.test_data).resolve()),
        }
        config["cross_validation"]["split_mode"] = PREDEFINED_SPLIT_MODE


def _resolve_config_paths(config: dict[str, Any]) -> None:
    """将 YAML 中的相对默认路径统一解释为相对于项目根目录。

    CLI 覆盖在此函数之后应用，因此用户命令行显式传入的相对路径仍按当前工作目录
    解析。最终写入配置快照的三个路径均为绝对路径。
    """

    for section, key in (
        ("data", "dataset_path"),
        ("data", "physics_config_path"),
        ("output", "artifact_dir"),
    ):
        configured_path = Path(config[section][key])
        if not configured_path.is_absolute():
            configured_path = PROJECT_ROOT / configured_path
        config[section][key] = str(configured_path.resolve())

    # 三份显式数据路径可以保持 null，直到维护者真正切换到 predefined。
    # 非空 YAML 路径沿用其他配置路径的约定，相对于项目根目录解释。
    split_config = config.get("predefined_split") or {}
    for key in (
        "train_dataset_path",
        "validation_dataset_path",
        "test_dataset_path",
    ):
        if not split_config.get(key):
            continue
        configured_path = Path(split_config[key])
        if not configured_path.is_absolute():
            configured_path = PROJECT_ROOT / configured_path
        split_config[key] = str(configured_path.resolve())


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


def _prepare_cross_validation_splits(
    dataset: HydroDataset,
    config: dict[str, Any],
) -> tuple[str, list[GroupSplit]]:
    """在清理旧产物前验证配置并构建本次 CV 划分。

    参数：
        dataset: 已完整加载并通过数据契约校验的数据集。
        config: 合并默认值且已应用 CLI 覆盖的训练配置。

    返回值：
        二元组 ``(split_mode, splits)``；第二项可直接交给
        :func:`run_cross_validation`，避免清理后再次构建产生差异。

    异常：
        ValueError: 模式、折数、验证比例或划分标签不满足要求时，由划分模块抛出。
    """

    model_ids, plant_counts, flow_speeds = _collect_split_labels(dataset)
    cv_config = config["cross_validation"]
    split_mode = str(cv_config.get("split_mode", "model"))
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
    return split_mode, splits


def _sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256，避免一次把大型 JSONL 全部读入内存。

    参数：
        path: 需要形成不可变内容摘要的文件路径。

    返回值：
        64 位小写十六进制 SHA-256 字符串。
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_predefined_sample_ids(
    datasets: dict[str, HydroDataset],
) -> None:
    """拒绝 train、validation、test 之间重复的 ``sample_id``。

    每个 :class:`HydroDataset` 已经保证单文件内部 ID 唯一；这里补充跨文件检查。
    相同 ID 出现在多个角色中通常表示数据泄漏，即测试样本已经被训练或 early
    stopping 间接使用。
    """

    owner_by_sample_id: dict[str, str] = {}
    duplicates: list[str] = []
    for role, dataset in datasets.items():
        for index in range(len(dataset)):
            sample_id = str(dataset[index]["sample_id"])
            previous_role = owner_by_sample_id.get(sample_id)
            if previous_role is not None and previous_role != role:
                duplicates.append(f"{sample_id} ({previous_role}/{role})")
            else:
                owner_by_sample_id[sample_id] = role
    if duplicates:
        preview = "、".join(duplicates[:10])
        suffix = "……" if len(duplicates) > 10 else ""
        raise ValueError(
            "预定义 train/validation/test 之间存在重复 sample_id，可能造成数据泄漏："
            f"{preview}{suffix}"
        )


def _prepare_predefined_datasets(
    config: dict[str, Any],
    physical_config: Any,
) -> tuple[Dataset, tuple[str, list[GroupSplit]], dict[str, HydroDataset]]:
    """加载、校验并拼接用户指定的 train/validation/test JSONL。

    三份文件分别使用严格 schema v1/v2 数据加载器。拼接只为复用训练器已经稳定
    的 ``Dataset + indices`` 接口；每个样本的 ``source_index`` 仍表示它在所属
    JSONL 中的本地行号，不会被伪装成跨文件全局行号。

    返回值：
        ``(combined_dataset, prepared_splits, datasets_by_role)``。其中划分只有 fold 0，
        并严格按三个文件的边界构造，不包含随机过程。
    """

    split_config = config["predefined_split"]
    dataset_paths = {
        "train": Path(split_config["train_dataset_path"]),
        "validation": Path(split_config["validation_dataset_path"]),
        "test": Path(split_config["test_dataset_path"]),
    }
    datasets = {
        role: HydroDataset(
            path,
            negative_target_policy=config["data"]["negative_target_policy"],
            physical_config=physical_config,
        )
        for role, path in dataset_paths.items()
    }
    _validate_predefined_sample_ids(datasets)

    train_count = len(datasets["train"])
    validation_count = len(datasets["validation"])
    test_count = len(datasets["test"])
    # HydroDataset 会拒绝空 JSONL；这里保留明确的角色级错误，便于以后替换 Dataset
    # 实现时仍维持三集合必须非空的训练契约。
    for role, count in (
        ("train", train_count),
        ("validation", validation_count),
        ("test", test_count),
    ):
        if count <= 0:
            raise ValueError(f"predefined 模式的 {role} 数据集不能为空。")

    validation_start = train_count
    test_start = train_count + validation_count
    split = GroupSplit(
        fold=0,
        train_indices=np.arange(0, train_count, dtype=np.int64),
        validation_indices=np.arange(
            validation_start, test_start, dtype=np.int64
        ),
        test_indices=np.arange(
            test_start, test_start + test_count, dtype=np.int64
        ),
    )
    combined_dataset: Dataset = ConcatDataset(
        [datasets["train"], datasets["validation"], datasets["test"]]
    )

    # 把可复现所需的数据内容摘要写回配置快照。路径只能说明文件位置；SHA-256
    # 才能发现同一路径下的数据后来被替换或重新排序。
    split_config["sample_counts"] = {
        role: len(dataset) for role, dataset in datasets.items()
    }
    split_config["sha256"] = {
        role: _sha256_file(path) for role, path in dataset_paths.items()
    }
    return (
        combined_dataset,
        (PREDEFINED_SPLIT_MODE, [split]),
        datasets,
    )


def _write_scaler(path: Path, scaler: GlobalFeatureScaler) -> None:
    """保存人类可读的训练集标准化统计。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scaler.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_metrics_by_flow_speed(
    output_dir: Path,
    prefix: str,
    metrics_by_flow_speed: dict[str, dict[str, float | int]],
) -> None:
    """把按流速分组的指标同时写成 JSON 和 CSV。

    参数:
        output_dir: 当前根产物目录或 fold 目录。
        prefix: 文件名前缀，例如 ``cv``、``validation`` 或 ``test``。
        metrics_by_flow_speed: ``compute_metrics_by_flow_speed`` 返回的嵌套字典。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}_metrics_by_flow_speed.json"
    json_path.write_text(
        json.dumps(metrics_by_flow_speed, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    # CSV 第一列显式保存流速，后续列固定排序，方便 Excel 和分析脚本稳定读取。
    csv_path = output_dir / f"{prefix}_metrics_by_flow_speed.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["flow_speed", *FLOW_SPEED_METRIC_COLUMNS],
        )
        writer.writeheader()
        for speed, metrics in metrics_by_flow_speed.items():
            writer.writerow({"flow_speed": speed, **metrics})


def _print_metrics_by_flow_speed_table(
    metrics_by_flow_speed: dict[str, dict[str, float | int]],
) -> None:
    """在训练结束时打印便于快速比较的分流速指标表。

    完整十列数据保存在 JSON/CSV；控制台只展示最常用的样本数、真实 D 均值、
    MAE、RMSE、R² 和 sMAPE，避免一行过宽而难以阅读。
    """

    print("按流速汇总的 held-out test 指标：", flush=True)
    print(
        "flow(m/s) | samples | mean_target_D | MAE_D | RMSE_D | R2 | sMAPE_D(%)",
        flush=True,
    )
    for speed, metrics in metrics_by_flow_speed.items():
        print(
            f"{speed:>9} | "
            f"{int(metrics['sample_count']):>7d} | "
            f"{float(metrics['mean_target_D']):>13.6f} | "
            f"{float(metrics['MAE_D']):>5.6f} | "
            f"{float(metrics['RMSE_D']):>6.6f} | "
            f"{float(metrics['R2']):>8.6f} | "
            f"{float(metrics['sMAPE_D']):>11.6f}",
            flush=True,
        )


def _write_fold_manifest(
    path: Path,
    dataset: Dataset,
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
                        "sample_id": str(sample["sample_id"]),
                        "model_id": int(sample["model_id"]),
                        "plant_count": int(sample["positions"].shape[0]),
                        "split_mode": split_mode,
                        "angle": int(sample["angle"]),
                        "flow_speed": float(sample["flow_speed"]),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_predefined_manifest(
    path: Path,
    dataset: Dataset,
    split: GroupSplit,
    config: dict[str, Any],
) -> None:
    """保存显式三数据集的角色、来源路径和本地/拼接索引。

    参数：
        path: 输出 CSV 路径。
        dataset: 按 train、validation、test 顺序拼接的逻辑数据集。
        split: 与三个文件边界完全一致的唯一划分。
        config: 包含三条规范化数据路径的完整运行配置。
    """

    split_config = config["predefined_split"]
    path_by_role = {
        "train": split_config["train_dataset_path"],
        "validation": split_config["validation_dataset_path"],
        "test": split_config["test_dataset_path"],
    }
    rows: list[dict[str, Any]] = []
    for role, indices in (
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("test", split.test_indices),
    ):
        for index in indices:
            sample = dataset[int(index)]
            rows.append(
                {
                    "role": role,
                    "dataset_path": path_by_role[role],
                    "dataset_index": int(index),
                    "source_index": int(sample["source_index"]),
                    "sample_id": str(sample["sample_id"]),
                    "model_id": int(sample["model_id"]),
                    "plant_count": int(sample["positions"].shape[0]),
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
    dataset: Dataset,
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
    evaluation_sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    """构造评估恢复所需的数据来源、标签策略和 held-out 索引。"""

    split_mode = str(config["cross_validation"].get("split_mode", "model"))
    validation_test_overlap = split_mode == FLOW_SPEED_SPLIT_MODE
    metadata: dict[str, Any] = {
        "checkpoint_role": checkpoint_role,
        "dataset_path": (
            config["predefined_split"]["test_dataset_path"]
            if split_mode == PREDEFINED_SPLIT_MODE
            else config["data"]["dataset_path"]
        ),
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
        "evaluation_sample_ids": evaluation_sample_ids,
    }
    if split_mode == PREDEFINED_SPLIT_MODE:
        split_config = config["predefined_split"]
        metadata["dataset_paths"] = {
            "train": split_config["train_dataset_path"],
            "validation": split_config["validation_dataset_path"],
            "test": split_config["test_dataset_path"],
        }
        metadata["dataset_sample_counts"] = dict(split_config["sample_counts"])
        metadata["dataset_sha256"] = dict(split_config["sha256"])
        metadata["final_retraining_performed"] = False
    return metadata


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
    dataset: Dataset,
    config: dict[str, Any],
    output_dir: Path,
    prepared_splits: tuple[str, list[GroupSplit]] | None = None,
) -> None:
    """按配置执行划分与评估，并按模式决定是否在全量数据上重训。

    参数：
        dataset: 已加载的完整数据集。
        config: 当前运行的完整配置。
        output_dir: 已安全初始化的产物目录。
        prepared_splits: 可选的预验证 ``(split_mode, splits)``。训练总入口会在清理旧
            产物前传入它；直接调用本函数时省略即可按原行为即时构建。
    """

    if prepared_splits is None:
        split_mode, splits = _prepare_cross_validation_splits(dataset, config)
    else:
        split_mode, splits = prepared_splits
    validation_test_overlap = split_mode == FLOW_SPEED_SPLIT_MODE
    predefined_split = split_mode == PREDEFINED_SPLIT_MODE
    if validation_test_overlap:
        print(
            "flow_speed 模式：0.1/0.2/0.3 m/s 用于训练，完整 0.4 m/s "
            "同时用于 validation 和 test；test 指标不是独立 held-out 指标，"
            "并且本次不会生成 final_model.pt。",
            flush=True,
        )
    if predefined_split:
        print(
            "predefined 模式：train 仅用于参数更新，validation 仅用于 early "
            "stopping，test 仅在最佳 checkpoint 上评估；本次不会执行 final 重训。",
            flush=True,
        )
        _write_predefined_manifest(
            output_dir / "predefined_split_assignments.csv",
            dataset,
            splits[0],
            config,
        )
    else:
        _write_fold_manifest(
            output_dir / "fold_assignments.csv", dataset, splits, split_mode
        )

    all_prediction_rows: list[dict[str, Any]] = []
    all_coefficient_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    best_epochs: list[int] = []
    for split in splits:
        fold_dir = (
            output_dir / PREDEFINED_SPLIT_MODE
            if predefined_split
            else output_dir / f"fold_{split.fold}"
        )
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
        test_sample_ids = [
            str(dataset[int(index)]["sample_id"])
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
                config,
                PREDEFINED_SPLIT_MODE if predefined_split else "fold",
                relative_floor,
                test_source_indices,
                test_sample_ids,
            ),
            progress_label=(
                "Predefined split" if predefined_split else f"Fold {split.fold}"
            ),
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

        # 普通 CV 的 validation/test 通常同时包含四档流速，因此在总体图旁增加
        # 2×2 分流速图，并把每档完整指标独立落盘。flow_speed 模式的两个集合
        # 只有 0.4 m/s，继续使用总体图即可，避免生成三个没有统计意义的空面板。
        validation_metrics_by_flow_speed = None
        test_metrics_by_flow_speed = None
        if not validation_test_overlap:
            validation_metrics_by_flow_speed = compute_metrics_by_flow_speed(
                validation_result.predictions
            )
            test_metrics_by_flow_speed = compute_metrics_by_flow_speed(
                test_result.predictions
            )
            _write_metrics_by_flow_speed(
                fold_dir,
                "validation",
                validation_metrics_by_flow_speed,
            )
            _write_metrics_by_flow_speed(
                fold_dir,
                "test",
                test_metrics_by_flow_speed,
            )
            write_drag_comparison_by_flow_speed_plot(
                validation_result.predictions,
                fold_dir / "validation_drag_comparison_by_flow_speed.png",
                title=f"Fold {split.fold} validation drag comparison by flow speed",
            )
            write_drag_comparison_by_flow_speed_plot(
                test_result.predictions,
                fold_dir / "test_drag_comparison_by_flow_speed.png",
                title=f"Fold {split.fold} test drag comparison by flow speed",
            )

        all_prediction_rows.extend(test_result.predictions)
        all_coefficient_rows.extend(test_result.plant_coefficients)
        fold_summary = {
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
        if not validation_test_overlap:
            fold_summary["validation_metrics_by_flow_speed"] = (
                validation_metrics_by_flow_speed
            )
            fold_summary["test_metrics_by_flow_speed"] = test_metrics_by_flow_speed
        fold_summaries.append(fold_summary)
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
    if predefined_split:
        aggregate_result["dataset_paths"] = {
            "train": config["predefined_split"]["train_dataset_path"],
            "validation": config["predefined_split"]["validation_dataset_path"],
            "test": config["predefined_split"]["test_dataset_path"],
        }
        aggregate_result["dataset_sample_counts"] = dict(
            config["predefined_split"]["sample_counts"]
        )
        aggregate_result["dataset_sha256"] = dict(
            config["predefined_split"]["sha256"]
        )

    # 普通 CV 的合并 test 预测是 out-of-fold 结果：每条样本只在未参与该 fold
    # 训练与 early stopping 时被预测一次。直接在逐样本行上重新计算分组指标，
    # 避免错误地平均各 fold 的 RMSE 或 R²。
    aggregate_metrics_by_flow_speed = None
    if not validation_test_overlap:
        aggregate_metrics_by_flow_speed = compute_metrics_by_flow_speed(
            all_prediction_rows
        )
        aggregate_result["aggregate_by_flow_speed"] = (
            aggregate_metrics_by_flow_speed
        )
    root_prefix = PREDEFINED_SPLIT_MODE if predefined_split else "cv"
    metrics_path = output_dir / f"{root_prefix}_metrics.json"
    metrics_path.write_text(
        json.dumps(aggregate_result, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    _write_rows(
        output_dir / f"{root_prefix}_predictions.csv", all_prediction_rows
    )
    _write_rows(
        output_dir / f"{root_prefix}_plant_coefficients.csv",
        all_coefficient_rows,
    )

    if aggregate_metrics_by_flow_speed is not None:
        _write_metrics_by_flow_speed(
            output_dir,
            root_prefix,
            aggregate_metrics_by_flow_speed,
        )
        # 根目录图直接展示所有 fold 合并后的 held-out test 预测；这比任意单折图
        # 更适合作为整次交叉验证的总体结果。
        write_drag_comparison_plot(
            all_prediction_rows,
            output_dir / f"{root_prefix}_drag_comparison.png",
            title=(
                "Predefined test drag comparison"
                if predefined_split
                else "Cross-validation held-out drag comparison"
            ),
        )
        write_drag_comparison_by_flow_speed_plot(
            all_prediction_rows,
            output_dir / f"{root_prefix}_drag_comparison_by_flow_speed.png",
            title=(
                "Predefined test drag comparison by flow speed"
                if predefined_split
                else "Cross-validation held-out drag comparison by flow speed"
            ),
        )
        # 仅使用各 fold 的 test 预测生成全测试集散点图，不混入 validation，
        # 也不使用随后全量重训模型的训练内预测。两图使用相同的原始测量，
        # 默认各自取范围，图内仍保持 x、y 等比例；配置也可切回共用范围。
        write_drag_parity_plots(
            all_prediction_rows,
            output_dir,
            prefix=root_prefix,
            axis_mode=config.get("output", {}).get(
                "drag_parity_axis_mode", DEFAULT_DRAG_PARITY_AXIS_MODE,
            ),
            title=(
                "Predefined test drag parity"
                if predefined_split
                else "Cross-validation held-out test drag parity"
            ),
        )
        _print_metrics_by_flow_speed_table(aggregate_metrics_by_flow_speed)

    # 固定流速实验只需要 fold_0 的跨速度模型。0.4 m/s 已参与 early stopping，
    # 因此既不能再把它加入训练，也不应创建声称使用全量数据的 final checkpoint。
    if validation_test_overlap:
        print(
            "flow_speed 固定划分完成：已保存 fold_0 及汇总产物；"
            "已按配置跳过全量重训。"
        )
        return

    # 用户已经明确指定 test 集时，任何自动全量重训都会改变三集合语义；若把 test
    # 加入重训还会形成直接数据泄漏。因此 predefined 只保留最佳验证 checkpoint。
    if predefined_split:
        print(
            "predefined 固定三数据集训练完成：已保存 validation/test 产物；"
            "已按设计跳过 final 重训。"
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
    """验证输入、安全初始化输出目录，再派发 overfit 或 CV 流程。"""
    print("CUDA available:", torch.cuda.is_available())
    args = parse_args()
    config = load_config(args.config)
    _resolve_config_paths(config)
    _validate_mode_arguments(args, config)
    _apply_cli_overrides(config, args)

    # 先完整加载物理表和 JSONL。若输入无效，程序会在触碰旧训练结果前失败，避免因为
    # 拼错数据路径或 YAML 内容而丢失上一次可用实验。
    physical_config = load_physical_config(config["data"]["physics_config_path"])
    config["resolved_physical_config"] = physical_config.to_dict()
    prepared_splits: tuple[str, list[GroupSplit]] | None = None
    if args.mode == PREDEFINED_SPLIT_MODE:
        # 三份 JSONL 的 schema、跨集合 sample_id、样本数和内容哈希全部在清理旧产物
        # 之前验证。任何一项失败都不会破坏上一次训练结果。
        dataset, prepared_splits, _ = _prepare_predefined_datasets(
            config, physical_config
        )
    else:
        dataset = HydroDataset(
            config["data"]["dataset_path"],
            negative_target_policy=config["data"]["negative_target_policy"],
            physical_config=physical_config,
        )

    # CV 划分同样在清理前完成验证。这样 split_mode 拼写错误、分组不足或固定速度
    # 缺档时都会保留旧结果；返回的索引随后直接复用，不会重复随机划分。
    if args.mode == "cv":
        prepared_splits = _prepare_cross_validation_splits(dataset, config)

    configured_output_dir = Path(config["output"]["artifact_dir"])
    print(f"训练产物目录：{configured_output_dir.resolve()}", flush=True)
    if args.resume_checkpoint:
        output_dir = prepare_resume_artifact_directory(configured_output_dir)
        print(
            "检测到 --resume-checkpoint：保留产物目录中的已有 checkpoint 和 history，"
            "跳过自动清理。",
            flush=True,
        )
    else:
        # fresh run 只允许清理最终解析出的 artifact_dir。保护列表覆盖项目源码、当前
        # 工作目录、用户目录以及本次已经验证过的全部输入文件。
        config_path = Path(args.config).expanduser().resolve(strict=False)
        protected_paths: tuple[Path, ...] = (
            Path.home(),
            Path.cwd(),
            PROJECT_ROOT,
            MODEL_SOURCE_ROOT,
            config_path,
            Path(config["data"]["dataset_path"]),
            Path(config["data"]["physics_config_path"]),
        )
        if args.mode == PREDEFINED_SPLIT_MODE:
            protected_paths += tuple(
                Path(config["predefined_split"][key])
                for key in (
                    "train_dataset_path",
                    "validation_dataset_path",
                    "test_dataset_path",
                )
            )
        output_dir = prepare_fresh_artifact_directory(
            configured_output_dir,
            managed_root=DEFAULT_MANAGED_ARTIFACT_ROOT,
            protected_paths=protected_paths,
        )
        print("已清空该产物目录中的旧训练结果。", flush=True)

    # 将安全初始化后的规范化路径写回快照，保证日志、配置和实际文件位置一致。
    config["output"]["artifact_dir"] = str(output_dir)
    save_config_snapshot(config, output_dir / "resolved_config.json")
    if args.mode == "overfit":
        run_overfit(dataset, config, output_dir, args.resume_checkpoint)
        recommended_checkpoint = output_dir / "overfit" / "best.pt"
    else:
        run_cross_validation(
            dataset,
            config,
            output_dir,
            prepared_splits=prepared_splits,
        )
        split_mode = str(config["cross_validation"].get("split_mode", "model"))
        if split_mode == PREDEFINED_SPLIT_MODE:
            recommended_checkpoint = (
                output_dir / PREDEFINED_SPLIT_MODE / "best.pt"
            )
        elif split_mode == FLOW_SPEED_SPLIT_MODE:
            recommended_checkpoint = output_dir / "fold_0" / "best.pt"
        else:
            recommended_checkpoint = output_dir / "final_model.pt"
    print(
        f"训练结果已保存至：{output_dir}\n推荐 checkpoint：{recommended_checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
