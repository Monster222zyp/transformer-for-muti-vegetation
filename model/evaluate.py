"""从 checkpoint 恢复数据语义并评估 HydroTransformer。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.hydro.data import HydroDataset, collate_hydro_samples
from model.models import HydroTransformer
from model.training.checkpoint import load_checkpoint
from model.training.trainer import (
    GlobalFeatureScaler,
    predict_dataset,
    resolve_device,
    write_prediction_result,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model" / "artifacts" / "evaluation"


def parse_args() -> argparse.Namespace:
    """解析评估入口参数。

    ``--data``、``--input-csv`` 和 ``--output-dir`` 一旦由用户显式提供，相对路径
    都按命令执行时的当前工作目录解析；省略时采用 checkpoint 或项目根默认路径。
    """

    parser = argparse.ArgumentParser(description="评估已训练的 HydroTransformer。")
    parser.add_argument("--checkpoint", required=True, help="best.pt 或 final_model.pt。")
    parser.add_argument(
        "--data",
        help="评估 CSV；省略时使用 checkpoint 保存的原数据路径。",
    )
    parser.add_argument(
        "--input-csv",
        help="构型定义；省略时使用 checkpoint 保存的 Experiment/input.csv。",
    )
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="声明 --data 是外部数据；没有此开关时禁止替换 checkpoint 原数据。",
    )
    parser.add_argument("--output-dir", help="输出目录；默认位于项目 model/artifacts。")
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda。")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _resolve_evaluation_paths(
    args: argparse.Namespace, metadata: dict[str, Any]
) -> tuple[Path, Path, Path]:
    """按“checkpoint 默认、CLI 按 cwd”规则解析数据、构型和输出路径。"""

    saved_dataset = metadata.get("dataset_path")
    saved_input = metadata.get("input_csv_path")
    if args.external_data and not args.data:
        raise ValueError("--external-data 必须与显式 --data 一起使用。")
    if args.data:
        dataset_path = Path(args.data).resolve()
    elif saved_dataset:
        dataset_path = Path(saved_dataset).resolve()
    else:
        raise ValueError("checkpoint 未记录 dataset_path，请显式提供 --data。")

    if args.input_csv:
        input_csv_path = Path(args.input_csv).resolve()
    elif saved_input:
        input_csv_path = Path(saved_input).resolve()
    else:
        input_csv_path = (PROJECT_ROOT / "Experiment" / "input.csv").resolve()
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
    return dataset_path, input_csv_path, output_dir


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


def _select_evaluation_indices(
    dataset: HydroDataset,
    metadata: dict[str, Any],
    external_data: bool,
) -> tuple[np.ndarray, str]:
    """依据 checkpoint 角色选择 held-out、in-sample 或外部评估范围。"""

    if external_data:
        return np.arange(len(dataset), dtype=np.int64), "external"

    role = metadata.get("checkpoint_role")
    saved_indices = metadata.get("evaluation_source_indices")
    if role == "fold":
        if not saved_indices:
            raise ValueError("fold checkpoint 缺少 held-out test source indices。")
        return _indices_from_source_indices(dataset, saved_indices), "held_out"
    if role == "final":
        return np.arange(len(dataset), dtype=np.int64), "in_sample"
    if role == "overfit":
        if not saved_indices:
            raise ValueError("overfit checkpoint 缺少诊断样本 source indices。")
        return _indices_from_source_indices(dataset, saved_indices), "overfit_diagnostic"
    raise ValueError(f"未知 checkpoint_role：{role!r}")


def main() -> None:
    """恢复模型/scaler，并按 checkpoint 角色生成不混淆数据范围的评估产物。"""

    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    device = resolve_device(args.device)
    # 先在 CPU 读取构造模型、定位数据所需的轻量元数据。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint 缺少 resolved model_config。")
    metadata = checkpoint.get("checkpoint_metadata", {})
    dataset_path, input_csv_path, output_dir = _resolve_evaluation_paths(
        args, metadata
    )

    model = HydroTransformer(**model_config)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    scaler = GlobalFeatureScaler.from_dict(checkpoint["scaler"])
    negative_target_policy = metadata.get(
        "negative_target_policy", "clamp_to_zero"
    )
    dataset = HydroDataset(
        dataset_path,
        input_csv_path=input_csv_path,
        negative_target_policy=negative_target_policy,
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
    # 明细表逐行标记评估语义，防止离开 JSON 上下文后误读 held-out 与 in-sample。
    for row in result.predictions:
        row["evaluation_scope"] = evaluation_scope
    for row in result.plant_coefficients:
        row["evaluation_scope"] = evaluation_scope
    write_prediction_result(result, output_dir, "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "evaluation_scope": evaluation_scope,
        "checkpoint_role": metadata.get("checkpoint_role"),
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "input_csv_path": str(input_csv_path),
        "negative_target_policy": negative_target_policy,
        "sample_count": int(indices.size),
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
