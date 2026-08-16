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
from .losses import (
    DEFAULT_RELATIVE_FLOOR_QUANTILE,
    RELATIVE_FLOOR_STRATEGY,
    relative_total_drag_mse_loss,
)
from .metrics import compute_regression_metrics
from .scheduler import WarmupCosineScheduler, choose_warmup_steps


# 默认每训练 10 个 epoch 向终端报告一次进度。
# 该默认值也用于 loss 图采样；用户可在 YAML 的 training 配置中覆盖。
DEFAULT_EPOCH_REPORT_INTERVAL = 10
DEFAULT_LOSS_PLOT_INTERVAL = 10
DEFAULT_CHECKPOINT_INTERVAL = 100
# checkpoint 和配置使用这个稳定名称记录训练目标，防止误用旧 absolute-MSE
# checkpoint 续训；推理加载不会检查该字段，因此旧权重仍可安全用于预测。
LOSS_NAME = "stabilized_relative_mse"


def _relative_objective_state(
    relative_floor: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """构造可写入 checkpoint 的稳定化相对 loss 参数。

    参数：
        relative_floor: 当前训练子集拟合出的分母下限。
        settings: 当前训练配置，提供分母策略与分位数。

    返回值：
        包含 loss 名称、实际分母下限、拟合策略和分位数的字典。
    """

    floor = float(relative_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("relative_floor 必须是有限正数。")
    strategy = str(
        settings.get("relative_loss_floor_strategy", RELATIVE_FLOOR_STRATEGY)
    )
    if strategy != RELATIVE_FLOOR_STRATEGY:
        raise ValueError(
            "当前训练器只支持 relative_loss_floor_strategy="
            f"{RELATIVE_FLOOR_STRATEGY!r}，实际收到 {strategy!r}。"
        )
    quantile = float(
        settings.get(
            "relative_loss_floor_quantile", DEFAULT_RELATIVE_FLOOR_QUANTILE
        )
    )
    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("relative_loss_floor_quantile 必须是 [0, 1] 内的有限数。")
    return {
        "loss_name": LOSS_NAME,
        "relative_floor": floor,
        "relative_floor_strategy": strategy,
        "relative_floor_quantile": quantile,
    }


def _checkpoint_metadata_with_objective(
    checkpoint_metadata: dict[str, Any] | None,
    objective_state: dict[str, Any],
) -> dict[str, Any]:
    """把训练目标写入 checkpoint 审计 metadata，并覆盖冲突的旧字段。"""

    merged_metadata = dict(checkpoint_metadata or {})
    merged_metadata.update(objective_state)
    return merged_metadata


def _build_adamw(
    model: torch.nn.Module,
    settings: dict[str, Any],
    device: torch.device,
) -> torch.optim.AdamW:
    """根据训练设备创建 AdamW optimizer。

    参数：
        model: 提供待更新参数的 PyTorch 模型。
        settings: 包含 ``learning_rate`` 和 ``weight_decay`` 的训练配置。
        device: 模型实际训练设备。

    返回值：
        CUDA 上启用 fused kernel、CPU 上使用普通实现的 AdamW。
    """

    # fused AdamW 把多次逐参数更新融合为更少的 CUDA kernel；CPU 不支持该模式。
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        fused=device.type == "cuda",
    )


def _lightweight_best_checkpoint(
    *,
    model_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_validation_loss: float,
    scaler: "GlobalFeatureScaler",
    model_config: dict[str, Any] | None,
    checkpoint_metadata: dict[str, Any] | None,
    settings: dict[str, Any],
    relative_floor: float,
) -> dict[str, Any]:
    """构造只供推理和评估使用的轻量 ``best.pt`` 内容。

    该状态不会保存 optimizer、scheduler、history 或重复的最佳权重，因此不能直接
    用于完整断点续训。训练恢复应使用同目录的 ``last.pt``。
    """

    objective_state = _relative_objective_state(relative_floor, settings)
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "best",
        **objective_state,
        # checkpoint 内部沿用从零开始的 epoch 索引；best_epoch 是对用户展示的一基索引。
        "epoch": best_epoch - 1,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "model_state": model_state,
        "scaler": scaler.to_dict(),
        "model_config": model_config,
        "checkpoint_metadata": _checkpoint_metadata_with_objective(
            checkpoint_metadata, objective_state
        ),
        "settings": settings,
    }


def _read_positive_interval(
    settings: dict[str, Any], key: str, default: int
) -> int:
    """从训练配置读取正整数间隔。

    参数：
        settings: 当前训练配置字典。
        key: 要读取的配置键。
        default: 配置中缺少该键时使用的默认值。

    返回值：
        大于零的 epoch 间隔。

    异常：
        配置值不能转换为正整数时抛出 ``ValueError``。
    """

    interval = int(settings.get(key, default))
    if interval <= 0:
        raise ValueError(f"{key} 必须是大于零的整数。")
    return interval


def _sample_loss_history(
    history: Sequence[dict[str, float]], interval_epochs: int
) -> list[dict[str, float]]:
    """按固定 epoch 间隔提取 loss 点，并保留非整周期的最后一点。

    常规采样点为第 ``interval_epochs``、``2*interval_epochs``……个 epoch。
    如果 early stopping 发生在非整周期位置，则额外保留最后一个 epoch，方便图中
    看出训练实际停止在哪里。
    """

    if interval_epochs <= 0:
        raise ValueError("loss 图采样间隔必须大于零。")
    if not history:
        return []

    sampled = [
        item
        for item in history
        if int(round(float(item["epoch"]))) % interval_epochs == 0
    ]
    last_item = history[-1]
    last_epoch = int(round(float(last_item["epoch"])))
    if not sampled or int(round(float(sampled[-1]["epoch"]))) != last_epoch:
        sampled.append(last_item)
    return sampled


def write_loss_curve(
    history: Sequence[dict[str, float]],
    output_path: str | Path,
    interval_epochs: int = DEFAULT_LOSS_PLOT_INTERVAL,
    title: str = "Training and Validation Loss",
) -> Path:
    """把训练历史绘制成训练/验证双折线 PNG。

    参数：
        history: 每个 epoch 的历史记录；每项必须包含 ``epoch``、
            ``train_loss`` 和 ``validation_loss``。
        output_path: PNG 输出路径。
        interval_epochs: 常规采样间隔，默认每 10 个 epoch 一个点。
        title: 图表标题。

    返回值：
        已写入的 PNG 绝对或相对 ``Path``。

    异常：
        history 为空、缺少字段或间隔非法时抛出 ``ValueError``/``KeyError``。

    副作用：
        创建输出目录并覆盖同名 PNG。使用 ``Agg`` 后端，因此无图形界面的服务器
        或终端环境也可以生成图片。
    """

    sampled = _sample_loss_history(history, interval_epochs)
    if not sampled:
        raise ValueError("训练历史为空，无法绘制 loss 曲线。")

    # matplotlib 采用延迟导入，数据准备和纯推理场景不会提前加载绘图库。
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    epochs = [int(round(float(item["epoch"]))) for item in sampled]
    train_losses = [float(item["train_loss"]) for item in sampled]
    validation_losses = [float(item["validation_loss"]) for item in sampled]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(epochs, train_losses, marker="o", label="Training relative MSE")
    axis.plot(
        epochs,
        validation_losses,
        marker="o",
        label="Validation relative MSE",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Stabilized relative MSE loss")
    axis.set_title(title)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def write_training_history_csv(
    history: Sequence[dict[str, float]], output_path: str | Path
) -> Path:
    """把逐 epoch 的相对 loss 历史写成人类可读 CSV。

    参数：
        history: 每项包含 ``epoch``、``train_loss``、``validation_loss`` 和
            ``learning_rate`` 的训练历史。
        output_path: 要写入的 CSV 路径。

    返回值：
        已写入文件的 ``Path``。
    """

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "epoch",
        "train_relative_mse",
        "validation_relative_mse",
        "learning_rate",
    )
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            writer.writerow(
                {
                    "epoch": int(round(float(item["epoch"]))),
                    "train_relative_mse": float(item["train_loss"]),
                    "validation_relative_mse": float(
                        item["validation_loss"]
                    ),
                    "learning_rate": float(item["learning_rate"]),
                }
            )
    return destination


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
        "plant_state",
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


def _forward_model(
    model: torch.nn.Module, batch: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """使用标准 batch 字段执行模型 forward，供训练和纯推理共用。"""

    return model(
        positions=batch["positions"],
        single_drag=batch["single_drag"],
        global_features=batch["global_features"],
        plant_mask=batch["plant_mask"],
        plant_state=batch["plant_state"],
    )


def _forward_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    relative_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """执行模型 forward，并计算稳定化目标相对 MSE。"""

    output = _forward_model(model, batch)
    loss = relative_total_drag_mse_loss(
        output["total_drag"],
        batch["target_drag"],
        relative_floor,
    )
    return loss, output


def _mean_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: GlobalFeatureScaler,
    relative_floor: float,
) -> float:
    """在无梯度模式计算样本加权的平均验证 loss。"""

    model.eval()
    weighted_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_and_normalize_batch(raw_batch, device, scaler)
            loss, _ = _forward_loss(model, batch, relative_floor)
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
    relative_floor: float,
    resume_from: str | Path | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
    progress_label: str | None = None,
) -> FitResult:
    """训练一个 fold，并以验证集稳定化相对 MSE 执行 early stopping。

    ``last.pt`` 每隔固定 epoch 更新，并在训练结束时强制保存；``best.pt`` 只保存
    最优模型推理所需的轻量状态。
    """

    configured_loss = str(settings.get("loss", LOSS_NAME))
    if configured_loss != LOSS_NAME:
        raise ValueError(
            f"当前训练器只支持 loss={LOSS_NAME!r}，实际收到 {configured_loss!r}。"
        )
    objective_state = _relative_objective_state(relative_floor, settings)
    complete_checkpoint_metadata = _checkpoint_metadata_with_objective(
        checkpoint_metadata, objective_state
    )

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

    optimizer = _build_adamw(model, settings, device)
    total_steps = int(settings["max_epochs"]) * len(train_loader)
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps, choose_warmup_steps(total_steps)
    )
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    best_path = destination / "best.pt"
    last_path = destination / "last.pt"
    report_interval = _read_positive_interval(
        settings,
        "progress_interval_epochs",
        DEFAULT_EPOCH_REPORT_INTERVAL,
    )
    plot_interval = _read_positive_interval(
        settings,
        "loss_plot_interval_epochs",
        DEFAULT_LOSS_PLOT_INTERVAL,
    )
    checkpoint_interval = _read_positive_interval(
        settings,
        "checkpoint_interval_epochs",
        DEFAULT_CHECKPOINT_INTERVAL,
    )
    display_label = progress_label or destination.name

    start_epoch = 0
    best_loss = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    # resolved 配置来自已完成默认值补全的模型实例，避免 checkpoint 依赖不完整 YAML。
    complete_model_config = resolved_model_config(model) or model_config
    if resume_from is not None:
        state = load_checkpoint(
            resume_from, model, optimizer, scheduler, map_location=device
        )
        # 断点续训必须继续使用同一份物理基准阻力。若用户在训练中途修改
        # physical.yaml，直接拒绝恢复比把两种物理语义混入一次实验更安全。
        current_physical = complete_checkpoint_metadata.get("physical_config")
        saved_physical = state.get("checkpoint_metadata", {}).get(
            "physical_config"
        )
        if current_physical is not None and saved_physical != current_physical:
            raise ValueError(
                "resume checkpoint 的 physical_config 与当前训练配置不一致；"
                "请恢复原物理参数或开始新的训练目录。"
            )
        if state.get("checkpoint_type") == "best":
            raise ValueError(
                "best.pt 是轻量评估 checkpoint，不含 optimizer/scheduler；"
                "请使用 last.pt 进行断点续训。"
            )
        resumed_loss = state.get("loss_name")
        if resumed_loss != LOSS_NAME:
            raise ValueError(
                "不能使用不同训练目标的 checkpoint 继续训练："
                f"当前 loss={LOSS_NAME!r}，checkpoint loss={resumed_loss!r}。"
            )
        saved_floor = state.get("relative_floor")
        if saved_floor is None or not math.isclose(
            float(saved_floor),
            objective_state["relative_floor"],
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "resume checkpoint 的 relative_floor 与当前训练子集不一致；"
                f"当前为 {objective_state['relative_floor']!r}，"
                f"checkpoint 为 {saved_floor!r}。"
            )
        for objective_key in (
            "loss_name",
            "relative_floor",
            "relative_floor_strategy",
            "relative_floor_quantile",
        ):
            saved_metadata_value = state.get("checkpoint_metadata", {}).get(
                objective_key
            )
            expected_metadata_value = complete_checkpoint_metadata[objective_key]
            if objective_key == "relative_floor":
                values_match = saved_metadata_value is not None and math.isclose(
                    float(saved_metadata_value),
                    float(expected_metadata_value),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            else:
                values_match = saved_metadata_value == expected_metadata_value
            if not values_match:
                raise ValueError(
                    "resume checkpoint metadata 中的 "
                    f"{objective_key} 与当前训练目标不一致。"
                )
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state.get("best_validation_loss", math.inf))
        best_epoch = int(state.get("best_epoch", -1))
        epochs_without_improvement = int(
            state.get("epochs_without_improvement", 0)
        )
        history = list(state.get("history", []))
        best_model_state: dict[str, torch.Tensor] | None = None
        if best_epoch >= 0:
            # version 3 的 last.pt 不再重复嵌入最佳权重，最佳状态来自同目录 best.pt。
            sibling_best = Path(resume_from).resolve().with_name("best.pt")
            if sibling_best.is_file():
                sibling_state = torch.load(
                    sibling_best, map_location="cpu", weights_only=False
                )
                best_model_state = sibling_state.get("model_state")
            # 兼容此前曾生成的 checkpoint；只读取旧字段，新 last.pt 不再写入它。
            elif state.get("best_model_state") is not None:
                best_model_state = state["best_model_state"]
            elif int(state.get("epoch", -1)) + 1 == best_epoch:
                best_model_state = state["model_state"]
        if best_epoch >= 0 and best_model_state is None:
            raise ValueError("恢复 checkpoint 未包含最优模型，且同目录找不到 best.pt。")
        if best_model_state is not None:
            # 即使恢复到另一个 artifact 目录，也立即建立本地 best.pt，保证零剩余 epoch 可结束。
            save_checkpoint(
                best_path,
                _lightweight_best_checkpoint(
                    model_state=best_model_state,
                    best_epoch=best_epoch,
                    best_validation_loss=best_loss,
                    scaler=scaler,
                    model_config=complete_model_config,
                    checkpoint_metadata=complete_checkpoint_metadata,
                    settings=settings,
                    relative_floor=relative_floor,
                ),
            )

    for epoch in range(start_epoch, int(settings["max_epochs"])):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for raw_batch in train_loader:
            batch = _move_and_normalize_batch(raw_batch, device, scaler)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _forward_loss(model, batch, relative_floor)
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
        validation_loss = _mean_loss(
            model, validation_loader, device, scaler, relative_floor
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        epoch_number = epoch + 1
        if epoch_number % report_interval == 0:
            print(
                f"[{display_label}] Epoch {epoch_number}/{int(settings['max_epochs'])} | "
                f"train_relative_mse={train_loss:.6g} | "
                f"validation_relative_mse={validation_loss:.6g} | "
                f"lr={float(optimizer.param_groups[0]['lr']):.6g}",
                flush=True,
            )
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            # best.pt 使用 CPU 权重，便于跨设备评估；其中不保存 optimizer 等续训状态。
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            save_checkpoint(
                best_path,
                _lightweight_best_checkpoint(
                    model_state=best_model_state,
                    best_epoch=best_epoch,
                    best_validation_loss=best_loss,
                    scaler=scaler,
                    model_config=complete_model_config,
                    checkpoint_metadata=complete_checkpoint_metadata,
                    settings=settings,
                    relative_floor=relative_floor,
                ),
            )
        else:
            epochs_without_improvement += 1

        should_stop = epochs_without_improvement >= int(
            settings["early_stopping_patience"]
        )
        is_final_epoch = epoch_number == int(settings["max_epochs"])
        should_save_last = (
            epoch_number % checkpoint_interval == 0
            or should_stop
            or is_final_epoch
        )
        if should_save_last:
            # last.pt 只保存当前续训状态，不再重复嵌入 best_model_state。
            save_checkpoint(
                last_path,
                {
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "checkpoint_type": "last",
                    **objective_state,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "epochs_without_improvement": epochs_without_improvement,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler": scaler.to_dict(),
                    "model_config": complete_model_config,
                    "checkpoint_metadata": complete_checkpoint_metadata,
                    "settings": settings,
                    "history": history,
                },
            )
        if should_stop:
            # 非 10 倍数处提前停止时额外输出停止位置，避免长时间运行后只有旧进度。
            if epoch_number % report_interval != 0:
                print(
                    f"[{display_label}] Early stopping at Epoch {epoch_number} | "
                    f"train_relative_mse={train_loss:.6g} | "
                    f"validation_relative_mse={validation_loss:.6g}",
                    flush=True,
                )
            break

    if best_epoch < 0:
        raise RuntimeError("训练未产生有效 checkpoint。")
    load_checkpoint(best_path, model, map_location=device)
    _write_json(destination / "history.json", history)
    write_training_history_csv(history, destination / "history.csv")
    if history:
        write_loss_curve(
            history,
            destination / "loss_curve.png",
            interval_epochs=plot_interval,
            title=f"{display_label} Loss Curve",
        )
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
    relative_floor: float,
    checkpoint_metadata: dict[str, Any] | None = None,
    progress_label: str | None = None,
) -> Path:
    """在全部数据上训练固定 epoch，生成最终模型 checkpoint。"""

    if epochs <= 0:
        raise ValueError("最终重训 epochs 必须大于零。")
    configured_loss = str(settings.get("loss", LOSS_NAME))
    if configured_loss != LOSS_NAME:
        raise ValueError(
            f"当前训练器只支持 loss={LOSS_NAME!r}，实际收到 {configured_loss!r}。"
        )
    objective_state = _relative_objective_state(relative_floor, settings)
    complete_checkpoint_metadata = _checkpoint_metadata_with_objective(
        checkpoint_metadata, objective_state
    )
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
    optimizer = _build_adamw(model, settings, device)
    total_steps = epochs * len(loader)
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps, choose_warmup_steps(total_steps)
    )
    report_interval = _read_positive_interval(
        settings,
        "progress_interval_epochs",
        DEFAULT_EPOCH_REPORT_INTERVAL,
    )
    display_label = progress_label or "Final retraining"
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for raw_batch in loader:
            batch = _move_and_normalize_batch(raw_batch, device, scaler)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _forward_loss(model, batch, relative_floor)
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
        epoch_number = epoch + 1
        if epoch_number % report_interval == 0:
            print(
                f"[{display_label}] Epoch {epoch_number}/{epochs} | "
                f"train_relative_mse={train_loss:.6g} | "
                "validation_relative_mse=N/A | "
                f"lr={float(optimizer.param_groups[0]['lr']):.6g}",
                flush=True,
            )

    destination = Path(artifact_dir) / "final_model.pt"
    complete_model_config = resolved_model_config(model) or model_config
    save_checkpoint(
        destination,
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            **objective_state,
            "epoch": epochs - 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler": scaler.to_dict(),
            "model_config": complete_model_config,
            "checkpoint_metadata": complete_checkpoint_metadata,
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
            # 纯预测不需要 target 或 relative_floor，因此直接执行模型 forward。
            output = _forward_model(model, batch)
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
                    for key in (
                        "source_index",
                        "model_id",
                        "state_id",
                        "angle",
                        "flow_speed",
                    )
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
