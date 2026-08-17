"""单折训练、固定 epoch 重训与预测产物生成。"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .checkpoint import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    resolved_model_config,
    save_checkpoint,
)
from .losses import interaction_mse_loss
from .metrics import compute_regression_metrics
from .scheduler import WarmupCosineScheduler, choose_warmup_steps


@dataclass(frozen=True)
class GlobalFeatureScaler:
    """按训练集合统计的全局特征标准化参数。"""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def fit(cls, dataset: Dataset, indices: Sequence[int]) -> "GlobalFeatureScaler":
        """只读取给定训练索引，计算逐特征均值和标准差。"""

        values = []
        for index in indices:
            feature = dataset[int(index)]["global_features"]
            if hasattr(feature, "detach"):
                feature = feature.detach().cpu().numpy()
            values.append(np.asarray(feature, dtype=np.float64).reshape(-1))
        if not values:
            raise ValueError("无法从空训练集合拟合 global feature scaler。")
        matrix = np.stack(values, axis=0)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        # 单一流速或常量物理特征的标准差设为 1，保证输出有限。
        std = np.where(std < 1.0e-12, 1.0, std)
        return cls(tuple(mean.tolist()), tuple(std.tolist()))

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """使用与输入相同的 device/dtype 标准化 batch。"""

        mean = values.new_tensor(self.mean)
        std = values.new_tensor(self.std)
        return (values - mean) / std

    def to_dict(self) -> dict[str, list[float]]:
        """转换为可写入 JSON/checkpoint 的字典。"""

        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GlobalFeatureScaler":
        """从 checkpoint 中恢复 scaler。"""

        return cls(tuple(payload["mean"]), tuple(payload["std"]))


@dataclass(frozen=True)
class FitResult:
    """一次训练完成后的关键信息。"""

    best_epoch: int
    best_validation_loss: float
    best_checkpoint: Path
    history: list[dict[str, float]]


@dataclass(frozen=True)
class PredictionResult:
    """评估指标与两个可直接写 CSV 的明细表。"""

    metrics: dict[str, float]
    predictions: list[dict[str, Any]]
    plant_coefficients: list[dict[str, Any]]


def set_reproducible_seed(seed: int) -> None:
    """设置 Python、NumPy 与 PyTorch 随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """把 ``auto`` 解析为 CUDA（若可用）或 CPU。"""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置请求 CUDA，但当前 PyTorch 无法使用 CUDA。")
    return device


def _move_and_normalize_batch(
    batch: dict[str, Any],
    device: torch.device,
    scaler: GlobalFeatureScaler,
) -> dict[str, Any]:
    """将模型所需张量移动到设备，并标准化唯一的全局物理输入。"""

    moved = dict(batch)
    for key in (
        "positions",
        "single_drag",
        "plant_mask",
        "global_features",
        "target_drag",
        "raw_target_drag",
    ):
        if key in moved and torch.is_tensor(moved[key]):
            moved[key] = moved[key].to(device)
    moved["global_features"] = scaler.transform(
        moved["global_features"].to(torch.float32)
    )
    return moved


def _make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    collate_fn: Callable[[list[Any]], Any],
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """创建只覆盖指定索引的 DataLoader。"""

    return DataLoader(
        Subset(dataset, [int(index) for index in indices]),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def _forward_loss(
    model: torch.nn.Module, batch: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """执行模型 forward 并计算 C-space MSE。"""

    output = model(
        positions=batch["positions"],
        single_drag=batch["single_drag"],
        global_features=batch["global_features"],
        plant_mask=batch["plant_mask"],
    )
    loss = interaction_mse_loss(
        output["total_drag"],
        batch["target_drag"],
        batch["single_drag"],
        batch["plant_mask"],
    )
    return loss, output


def _mean_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: GlobalFeatureScaler,
) -> float:
    """在无梯度模式计算样本加权的平均验证 loss。"""

    model.eval()
    weighted_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_and_normalize_batch(raw_batch, device, scaler)
            loss, _ = _forward_loss(model, batch)
            current_count = int(batch["target_drag"].shape[0])
            weighted_loss += float(loss.item()) * current_count
            sample_count += current_count
    if sample_count == 0:
        raise ValueError("验证集合为空。")
    return weighted_loss / sample_count


def fit_with_early_stopping(
    model: torch.nn.Module,
    dataset: Dataset,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    collate_fn: Callable[[list[Any]], Any],
    settings: dict[str, Any],
    scaler: GlobalFeatureScaler,
    artifact_dir: str | Path,
    model_config: dict[str, Any] | None,
    seed: int,
    resume_from: str | Path | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> FitResult:
    """训练一个 fold，并以验证 C-MSE early stop。

    ``last.pt`` 每个 epoch 都会更新，可用于恢复；``best.pt`` 只保存最优验证状态。
    """

    set_reproducible_seed(seed)
    device = resolve_device(str(settings["device"]))
    model = model.to(device=device, dtype=torch.float32)
    train_loader = _make_loader(
        dataset,
        train_indices,
        int(settings["batch_size"]),
        collate_fn,
        True,
        int(settings["num_workers"]),
    )
    validation_loader = _make_loader(
        dataset,
        validation_indices,
        int(settings["batch_size"]),
        collate_fn,
        False,
        int(settings["num_workers"]),
    )
    if len(train_loader) == 0:
        raise ValueError("训练集合为空。")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    total_steps = int(settings["max_epochs"]) * len(train_loader)
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps, choose_warmup_steps(total_steps)
    )
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    best_path = destination / "best.pt"
    last_path = destination / "last.pt"

    start_epoch = 0
    best_loss = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    # resolved 配置来自已完成默认值补全的模型实例，避免 checkpoint 依赖不完整 YAML。
    complete_model_config = resolved_model_config(model) or model_config
    best_model_state: dict[str, torch.Tensor] | None = None
    if resume_from is not None:
        state = load_checkpoint(
            resume_from, model, optimizer, scheduler, map_location=device
        )
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state.get("best_validation_loss", math.inf))
        best_epoch = int(state.get("best_epoch", -1))
        epochs_without_improvement = int(
            state.get("epochs_without_improvement", 0)
        )
        history = list(state.get("history", []))
        best_model_state = state.get("best_model_state")
        if best_model_state is None and best_epoch >= 0:
            # 兼容 version 1 checkpoint：旧 last.pt 没有内嵌最优权重，尝试读取其同目录 best.pt。
            sibling_best = Path(resume_from).resolve().with_name("best.pt")
            if sibling_best.is_file():
                sibling_state = torch.load(
                    sibling_best, map_location="cpu", weights_only=False
                )
                best_model_state = sibling_state.get("model_state")
            elif int(state.get("epoch", -1)) + 1 == best_epoch:
                best_model_state = state["model_state"]
        if best_epoch >= 0 and best_model_state is None:
            raise ValueError("恢复 checkpoint 未包含最优模型，且同目录找不到 best.pt。")
        if best_model_state is not None:
            # 即使恢复到另一个 artifact 目录，也立即建立本地 best.pt，保证零剩余 epoch 可结束。
            local_best_state = dict(state)
            # best.pt 的 epoch 应描述其 model_state，而不是恢复来源 last.pt 的较晚 epoch。
            local_best_state["epoch"] = best_epoch - 1
            local_best_state["model_state"] = best_model_state
            local_best_state["best_model_state"] = best_model_state
            local_best_state["model_config"] = complete_model_config
            local_best_state["checkpoint_version"] = CHECKPOINT_VERSION
            if checkpoint_metadata is not None:
                local_best_state["checkpoint_metadata"] = checkpoint_metadata
            save_checkpoint(best_path, local_best_state)

    for epoch in range(start_epoch, int(settings["max_epochs"])):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for raw_batch in train_loader:
            batch = _move_and_normalize_batch(raw_batch, device, scaler)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _forward_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(settings["gradient_clip"])
            )
            optimizer.step()
            scheduler.step()
            current_count = int(batch["target_drag"].shape[0])
            train_loss_sum += float(loss.item()) * current_count
            train_sample_count += current_count

        train_loss = train_loss_sum / train_sample_count
        validation_loss = _mean_loss(model, validation_loader, device, scaler)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            # CPU clone 与继续更新的模型参数解耦，可安全内嵌到后续 last.pt。
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        state = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler": scaler.to_dict(),
            "model_config": complete_model_config,
            "best_model_state": best_model_state,
            "checkpoint_metadata": checkpoint_metadata or {},
            "settings": settings,
            "history": history,
        }
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(best_path, state)
        if epochs_without_improvement >= int(settings["early_stopping_patience"]):
            break

    if best_epoch < 0:
        raise RuntimeError("训练未产生有效 checkpoint。")
    load_checkpoint(best_path, model, map_location=device)
    _write_json(destination / "history.json", history)
    return FitResult(best_epoch, best_loss, best_path, history)


def fit_fixed_epochs(
    model: torch.nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    collate_fn: Callable[[list[Any]], Any],
    settings: dict[str, Any],
    scaler: GlobalFeatureScaler,
    artifact_dir: str | Path,
    model_config: dict[str, Any],
    seed: int,
    epochs: int,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> Path:
    """在全部数据上训练固定 epoch，生成最终模型 checkpoint。"""

    if epochs <= 0:
        raise ValueError("最终重训 epochs 必须大于零。")
    set_reproducible_seed(seed)
    device = resolve_device(str(settings["device"]))
    model = model.to(device=device, dtype=torch.float32)
    loader = _make_loader(
        dataset,
        indices,
        int(settings["batch_size"]),
        collate_fn,
        True,
        int(settings["num_workers"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    total_steps = epochs * len(loader)
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps, choose_warmup_steps(total_steps)
    )
    for _epoch in range(epochs):
        model.train()
        for raw_batch in loader:
            batch = _move_and_normalize_batch(raw_batch, device, scaler)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _forward_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(settings["gradient_clip"])
            )
            optimizer.step()
            scheduler.step()

    destination = Path(artifact_dir) / "final_model.pt"
    complete_model_config = resolved_model_config(model) or model_config
    save_checkpoint(
        destination,
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "epoch": epochs - 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler": scaler.to_dict(),
            "model_config": complete_model_config,
            "checkpoint_metadata": checkpoint_metadata or {},
            "settings": settings,
            "trained_on_all_data": True,
        },
    )
    return destination


def predict_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    collate_fn: Callable[[list[Any]], Any],
    batch_size: int,
    num_workers: int,
    scaler: GlobalFeatureScaler,
    device: torch.device,
) -> PredictionResult:
    """评估指定索引，并保留逐样本预测和逐株 latent coefficient。"""

    loader = _make_loader(
        dataset, indices, batch_size, collate_fn, False, num_workers
    )
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    all_targets: list[float] = []
    all_predictions: list[float] = []
    all_isolated: list[float] = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_and_normalize_batch(raw_batch, device, scaler)
            _, output = _forward_loss(model, batch)
            predicted = output["total_drag"].detach().cpu()
            coefficients = output["coefficient"].detach().cpu()
            masks = batch["plant_mask"].detach().cpu()
            positions = batch["positions"].detach().cpu()
            single_drag = batch["single_drag"].detach().cpu()
            targets = batch["target_drag"].detach().cpu()
            raw_targets = batch.get("raw_target_drag", targets).detach().cpu()
            isolated = (single_drag * masks.to(single_drag.dtype)).sum(dim=1)

            for row_index in range(predicted.shape[0]):
                metadata = {
                    key: _metadata_value(raw_batch, key, row_index)
                    for key in ("source_index", "model_id", "angle", "flow_speed")
                }
                # collate 会把 Python float 转为 float32；回写前恢复实验固定的一位小数精度。
                metadata["flow_speed"] = round(float(metadata["flow_speed"]), 1)
                target_value = float(targets[row_index])
                prediction_value = float(predicted[row_index])
                isolated_value = float(isolated[row_index])
                prediction_rows.append(
                    {
                        **metadata,
                        "raw_target_drag": float(raw_targets[row_index]),
                        "target_drag": target_value,
                        "predicted_drag": prediction_value,
                        "isolated_drag": isolated_value,
                        "target_C": target_value / isolated_value,
                        "predicted_C": prediction_value / isolated_value,
                    }
                )
                valid_count = int(masks[row_index].sum().item())
                for plant_index in range(valid_count):
                    coefficient_rows.append(
                        {
                            **metadata,
                            "plant_index": plant_index,
                            "x": float(positions[row_index, plant_index, 0]),
                            "y": float(positions[row_index, plant_index, 1]),
                            "single_drag": float(single_drag[row_index, plant_index]),
                            "latent_coefficient": float(
                                coefficients[row_index, plant_index]
                            ),
                        }
                    )
                all_targets.append(target_value)
                all_predictions.append(prediction_value)
                all_isolated.append(isolated_value)

    metrics = compute_regression_metrics(
        all_targets, all_predictions, all_isolated
    )
    return PredictionResult(metrics, prediction_rows, coefficient_rows)


def _metadata_value(batch: dict[str, Any], key: str, index: int) -> Any:
    """从 collate 后的 tensor/list 中读取一个 Python 标量。"""

    if key not in batch:
        return None
    value = batch[key][index]
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_prediction_result(
    result: PredictionResult,
    output_dir: str | Path,
    prefix: str,
) -> None:
    """把指标、逐样本预测和逐株系数写入 artifacts。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / f"{prefix}_metrics.json", result.metrics)
    _write_csv(destination / f"{prefix}_predictions.csv", result.predictions)
    _write_csv(
        destination / f"{prefix}_plant_coefficients.csv",
        result.plant_coefficients,
    )


def _write_json(path: Path, payload: Any) -> None:
    """以 UTF-8 和缩进格式写 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写字典列表；空列表时仍创建一个空文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
