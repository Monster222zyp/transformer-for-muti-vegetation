"""训练 checkpoint 的保存与恢复。"""

from __future__ import annotations

import os
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


# version 5 移除角度状态 Token：角度只用于物理默认阻力查表，所有水草共享一个
# 可学习 Token。旧 checkpoint 的参数结构和输入语义都不同，因此不自动迁移。
CHECKPOINT_VERSION = 5
# 首次替换失败后依次等待这些秒数；因此总共最多尝试 5 次原子替换。
CHECKPOINT_RETRY_DELAYS_SECONDS = (0.2, 0.5, 1.0, 2.0)
# Directional attention 没有新增权重。保留这些默认值可读取同一模型架构中缺少
# 显式方向字段的早期配置快照；跨 v5 的旧角度状态 Token checkpoint 仍会先被拒绝。
LEGACY_DIRECTIONAL_ATTENTION_DEFAULTS = {
    "directional_attention_mode": "none",
    "directional_soft_activation": "relu",
    "directional_soft_strength": 1.0,
    "directional_tolerance": 1.0e-6,
}


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
    if int(checkpoint.get("checkpoint_version", -1)) < CHECKPOINT_VERSION:
        raise ValueError(
            "checkpoint 来自旧版角度状态 Token 架构；当前模型不再把角度状态输入"
            "神经网络，请使用当前代码重新训练。"
        )
    expected = checkpoint.get("model_config")
    if not isinstance(expected, dict):
        raise ValueError("checkpoint 缺少完整 model_config，无法安全恢复 HydroTransformer。")
    # 旧 v4 checkpoint 保存于 directional attention 引入之前。仅当目标模型也采用
    # 兼容默认值时，这些缺失字段才会在补全后匹配；尝试用 soft/hard 续训仍会被拒绝。
    normalized_expected = dict(expected)
    for key, default_value in LEGACY_DIRECTIONAL_ATTENTION_DEFAULTS.items():
        normalized_expected.setdefault(key, default_value)

    if normalized_expected != actual:
        differing_keys = sorted(
            key
            for key in set(normalized_expected) | set(actual)
            if normalized_expected.get(key) != actual.get(key)
        )
        raise ValueError(
            "checkpoint 的 model_config 与当前模型不一致；差异字段："
            f"{differing_keys}"
        )


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    """原子保存 checkpoint，并容忍 Windows 上短暂的文件占用。

    Args:
        path: 最终 checkpoint 路径。
        state: 需要交给 :func:`torch.save` 的状态字典。
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # PID 区分不同训练进程，UUID 区分同一进程的多次保存。即使上次失败留下
    # 临时文件，下一次保存也不会覆盖那份可恢复的完整 checkpoint。
    temporary = destination.with_name(
        f"{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    torch.save(state, temporary)

    # torch.save 已经完整结束后才进入替换阶段。Windows Defender、索引服务或其他
    # 读取进程可能短暂锁定 destination，因此仅对 PermissionError 做有限重试。
    maximum_attempts = len(CHECKPOINT_RETRY_DELAYS_SECONDS) + 1
    for attempt_index in range(maximum_attempts):
        try:
            temporary.replace(destination)
            return
        except PermissionError as error:
            if attempt_index == maximum_attempts - 1:
                message = (
                    f"checkpoint 已完整写入临时文件，但连续 {maximum_attempts} 次无法替换"
                    f"目标文件 {destination}。临时文件已保留在 {temporary}，请勿删除；"
                    "请检查杀毒软件、同步程序、文件预览器或其他训练进程是否占用目标文件。"
                )
                print(message, file=sys.stderr, flush=True)
                raise PermissionError(message) from error

            delay = CHECKPOINT_RETRY_DELAYS_SECONDS[attempt_index]
            print(
                f"checkpoint 目标文件暂时被占用：{destination}；"
                f"将在 {delay:g} 秒后重试原子替换 "
                f"({attempt_index + 2}/{maximum_attempts})。",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


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
