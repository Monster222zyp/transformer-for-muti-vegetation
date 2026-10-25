"""从 checkpoint 恢复数据语义并评估 HydroTransformer。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.hydro.data import HydroDataset, collate_hydro_samples
from model.hydro.physics import load_physical_config
from model.models import HydroTransformer
from model.training.checkpoint import CHECKPOINT_VERSION, load_checkpoint
from model.training.trainer import (
    GlobalFeatureScaler,
    predict_dataset,
    resolve_device,
    write_prediction_result,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 维护者可以直接修改下面的默认变量后无参数运行，也可以继续用同名 CLI 覆盖。
# checkpoint 默认留空是为了避免意外评估一个过时模型；至少应在这里或命令行明确指定。
DEFAULT_CHECKPOINT_PATH: Path | None = Path(    r"C:\Users\Mayn\Desktop\myfiles\Transformer project\transformer-for-muti-vegetation\model\artifacts\final_model.pt"
)
DEFAULT_DATA_PATH=PROJECT_ROOT / "datasets" /"non_60.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model" / "artifacts" / "evaluation"
DEFAULT_EXTERNAL_DATA = False
DEFAULT_DEVICE = "auto"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0


def _optional_path_default(path: Path | None) -> str | None:
    """把文件开头的可选 Path 默认值转换为 argparse 可使用的字符串。"""

    return None if path is None else str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析评估入口参数。

    ``--data`` 和 ``--output-dir`` 一旦由用户显式提供，相对路径
    都按命令执行时的当前工作目录解析；省略时采用 checkpoint 或项目根默认路径。
    """

    parser = argparse.ArgumentParser(description="评估已训练的 HydroTransformer。")
    parser.add_argument(
        "--checkpoint",
        default=_optional_path_default(DEFAULT_CHECKPOINT_PATH),
        help="best.pt、last.pt 或 final_model.pt；也可在文件开头设置默认路径。",
    )
    parser.add_argument(
        "--data",
        default=_optional_path_default(DEFAULT_DATA_PATH),
        help="评估 dataset.jsonl；省略时使用 checkpoint 保存的原数据路径。",
    )
    parser.add_argument(
        "--external-data",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_EXTERNAL_DATA,
        help="声明 --data 是外部数据；没有此开关时禁止替换 checkpoint 原数据。",
    )
    parser.add_argument("--output-dir", help="输出目录；默认位于项目 model/artifacts。")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="auto、cpu 或 cuda。")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args(argv)
    if not args.checkpoint:
        raise ValueError(
            "请通过 --checkpoint 指定模型，或在 evaluate.py 开头设置 "
            "DEFAULT_CHECKPOINT_PATH。"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须是大于 0 的整数。")
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能小于 0。")
    return args


def _sha256_file(path: Path) -> str:
    """分块计算评估数据的 SHA-256 内容摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_evaluation_paths(
    args: argparse.Namespace, metadata: dict[str, Any]
) -> tuple[Path, Path]:
    """按“checkpoint 默认、CLI 按 cwd”规则解析数据和输出路径。"""

    saved_dataset = metadata.get("dataset_path")
    if args.external_data and not args.data:
        raise ValueError("--external-data 必须与显式 --data 一起使用。")
    if args.data:
        dataset_path = Path(args.data).resolve()
    elif saved_dataset:
        dataset_path = Path(saved_dataset).resolve()
    else:
        raise ValueError("checkpoint 未记录 dataset_path，请显式提供 --data。")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_DIR.resolve()
    )

    # 普通评估必须复用原数据；改变数据来源需要显式声明，避免误把外部数据当 held-out。
    if (
        args.data
        and saved_dataset
        and dataset_path != Path(saved_dataset).resolve()
        and not args.external_data
    ):
        raise ValueError("替换 checkpoint 原数据时必须显式添加 --external-data。")
    return dataset_path, output_dir


def _indices_from_source_indices(
    dataset: HydroDataset, requested_source_indices: list[int]
) -> np.ndarray:
    """把 checkpoint 保存的稳定 ``source_index`` 映射回当前 Dataset 索引。"""

    source_to_dataset: dict[int, int] = {}
    for dataset_index in range(len(dataset)):
        source_index = int(dataset[dataset_index]["source_index"])
        if source_index in source_to_dataset:
            raise ValueError(f"数据集包含重复 source_index：{source_index}")
        source_to_dataset[source_index] = dataset_index
    missing = sorted(set(requested_source_indices) - set(source_to_dataset))
    if missing:
        raise ValueError(f"数据集缺少 checkpoint 要求的 source_index：{missing[:10]}")
    return np.asarray(
        [source_to_dataset[int(source)] for source in requested_source_indices],
        dtype=np.int64,
    )


def _indices_from_sample_ids(
    dataset: HydroDataset, requested_sample_ids: list[str]
) -> np.ndarray:
    """按稳定 ``sample_id`` 把 checkpoint 的测试样本映射回当前数据集。"""

    sample_to_dataset: dict[str, int] = {}
    for dataset_index in range(len(dataset)):
        sample_id = str(dataset[dataset_index]["sample_id"])
        if sample_id in sample_to_dataset:
            raise ValueError(f"数据集包含重复 sample_id：{sample_id}")
        sample_to_dataset[sample_id] = dataset_index
    missing = sorted(set(requested_sample_ids) - set(sample_to_dataset))
    if missing:
        raise ValueError(f"数据集缺少 checkpoint 要求的 sample_id：{missing[:10]}")
    return np.asarray(
        [sample_to_dataset[str(sample_id)] for sample_id in requested_sample_ids],
        dtype=np.int64,
    )


def _select_evaluation_indices(
    dataset: HydroDataset,
    metadata: dict[str, Any],
    external_data: bool,
) -> tuple[np.ndarray, str]:
    """依据 checkpoint 角色和重叠标记选择准确的评估范围语义。"""

    if external_data:
        return np.arange(len(dataset), dtype=np.int64), "external"

    role = metadata.get("checkpoint_role")
    saved_indices = metadata.get("evaluation_source_indices")
    if role in ("fold", "predefined"):
        if not saved_indices:
            raise ValueError(f"{role} checkpoint 缺少 held-out test source indices。")
        saved_sample_ids = metadata.get("evaluation_sample_ids")
        indices = (
            _indices_from_sample_ids(dataset, list(saved_sample_ids))
            if saved_sample_ids
            else _indices_from_source_indices(dataset, saved_indices)
        )
        # flow_speed 的 test 索引也参与过 early stopping，不能沿用普通 fold 的
        # held_out 标签；独立 scope 可防止下游表格或报告误读这个指标。
        if bool(metadata.get("validation_test_overlap", False)):
            return indices, "validation_test_overlap"
        return indices, "held_out"
    if role == "final":
        return np.arange(len(dataset), dtype=np.int64), "in_sample"
    if role == "overfit":
        if not saved_indices:
            raise ValueError("overfit checkpoint 缺少诊断样本 source indices。")
        return _indices_from_source_indices(dataset, saved_indices), "overfit_diagnostic"
    raise ValueError(f"未知 checkpoint_role：{role!r}")


def _validate_predefined_test_fingerprint(
    dataset_path: Path,
    metadata: dict[str, Any],
    external_data: bool,
) -> str:
    """验证 predefined checkpoint 自动恢复的 test JSONL 没有被替换。

    外部评估允许用户主动选择任意同 schema 数据，因此只计算并返回实际摘要；只有
    checkpoint 自动恢复自己保存的 predefined test 文件时才执行严格比对。
    """

    actual_sha256 = _sha256_file(dataset_path)
    if external_data or metadata.get("checkpoint_role") != "predefined":
        return actual_sha256
    saved_hashes = metadata.get("dataset_sha256")
    expected_sha256 = (
        saved_hashes.get("test") if isinstance(saved_hashes, dict) else None
    )
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            "predefined checkpoint 保存的 test JSONL 与当前文件 SHA-256 不一致；"
            "请恢复原测试集，或显式使用 --data 和 --external-data 评估新数据。"
        )
    return actual_sha256


def main() -> None:
    """恢复模型/scaler，并按 checkpoint 角色生成不混淆数据范围的评估产物。"""

    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint 文件：{checkpoint_path}")
    device = resolve_device(args.device)
    # 先在 CPU 读取构造模型、定位数据所需的轻量元数据。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("checkpoint_version", -1)) < CHECKPOINT_VERSION:
        raise ValueError(
            "checkpoint 来自旧版共享 Token 或角度状态架构；当前模型使用逐株角度/EI"
            "编码器和三维全局输入，请使用当前代码重新训练。"
        )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint 缺少 resolved model_config。")
    metadata = checkpoint.get("checkpoint_metadata", {})
    physical_payload = metadata.get("physical_config")
    if not isinstance(physical_payload, dict):
        raise ValueError(
            "checkpoint 缺少按角度查询阻力所需的 physical_config；"
            "请使用当前代码重新训练。"
        )
    physical_config = load_physical_config(physical_payload)
    dataset_path, output_dir = _resolve_evaluation_paths(
        args, metadata
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"找不到评估 JSONL：{dataset_path}")
    dataset_sha256 = _validate_predefined_test_fingerprint(
        dataset_path, metadata, args.external_data
    )

    model = HydroTransformer(**model_config)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    scaler = GlobalFeatureScaler.from_dict(checkpoint["scaler"])
    negative_target_policy = metadata.get(
        "negative_target_policy", "clamp_to_zero"
    )
    dataset = HydroDataset(
        dataset_path,
        negative_target_policy=negative_target_policy,
        physical_config=physical_config,
    )
    indices, evaluation_scope = _select_evaluation_indices(
        dataset, metadata, args.external_data
    )
    result = predict_dataset(
        model,
        dataset,
        indices,
        collate_hydro_samples,
        args.batch_size,
        args.num_workers,
        scaler,
        device,
    )
    result.metrics["sample_count"] = int(indices.size)
    # 明细表逐行标记评估语义，防止离开 JSON 上下文后误读 held-out、重叠评估
    # 与 in-sample 数据。
    for row in result.predictions:
        row["evaluation_scope"] = evaluation_scope
    for row in result.plant_coefficients:
        row["evaluation_scope"] = evaluation_scope
    write_prediction_result(result, output_dir, "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "evaluation_scope": evaluation_scope,
        "checkpoint_role": metadata.get("checkpoint_role"),
        "checkpoint_version": int(checkpoint.get("checkpoint_version", -1)),
        "split_mode": metadata.get("split_mode"),
        "validation_test_overlap": bool(
            metadata.get("validation_test_overlap", False)
        ),
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "negative_target_policy": negative_target_policy,
        "physical_config": physical_config.to_dict(),
        "sample_count": int(indices.size),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "source_indices": [
            int(dataset[int(index)]["source_index"]) for index in indices
        ],
    }
    (output_dir / "evaluation_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"评估完成：scope={evaluation_scope}, samples={indices.size}, "
        f"RMSE_D={result.metrics['RMSE_D']:.6g}"
    )


if __name__ == "__main__":
    main()
