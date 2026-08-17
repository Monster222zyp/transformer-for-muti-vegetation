"""训练 checkpoint 的保存与恢复。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VERSION = 2


def resolved_model_config(model: torch.nn.Module) -> dict[str, Any] | None:
    """读取模型实例实际采用的完整配置。

    ``HydroTransformer`` 会把默认值补全后保存在 dataclass ``model.config`` 中；
    checkpoint 应保存这份 resolved 配置，而不是可能缺字段的原始 YAML 片段。
    普通 PyTorch 测试模型没有 ``config`` 时返回 ``None``。
    """

    config = getattr(model, "config", None)
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("model.config 必须是 dataclass 或字典，才能写入 checkpoint。")


def _validate_model_config(
    checkpoint: dict[str, Any], model: torch.nn.Module
) -> None:
    """在加载权重前检查 checkpoint 与目标模型的完整配置是否一致。"""

    actual = resolved_model_config(model)
    if actual is None:
        return
    expected = checkpoint.get("model_config")
    if expected is None:
        raise ValueError("checkpoint 缺少完整 model_config，无法安全恢复 HydroTransformer。")
    if expected != actual:
        differing_keys = sorted(
            key
            for key in set(expected) | set(actual)
            if expected.get(key) != actual.get(key)
        )
        raise ValueError(
            "checkpoint 的 model_config 与当前模型不一致；差异字段："
            f"{differing_keys}"
        )


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    """原子保存 checkpoint，避免训练中断留下半个文件。

    Args:
        path: 最终 checkpoint 路径。
        state: 需要交给 :func:`torch.save` 的状态字典。
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """加载模型，并按需恢复优化器和 scheduler。

    Args:
        path: checkpoint 文件路径。
        model: 接收 ``model_state`` 的模型。
        optimizer: 若提供，则恢复 ``optimizer_state``。
        scheduler: 若提供，则恢复 ``scheduler_state``。
        map_location: checkpoint 张量要加载到的设备。

    Returns:
        checkpoint 中的完整状态字典，调用者可读取 epoch、scaler 等信息。
    """

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    _validate_model_config(checkpoint, model)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint
