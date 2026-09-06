"""YAML 配置读取、默认值补全与快照工具。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TRAINING_MIRROR_AUGMENTATION = True


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 20260814,
    "data": {
        "dataset_path": "Experiment/output/dataset.jsonl",
        "physics_config_path": "model/configs/physical.yaml",
        "negative_target_policy": "clamp_to_zero",
    },
    "model": {
        "global_input_dim": 1,
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 4,
        "ffn_dim": 1024,
        "dropout": 0.05,
        "use_rope": True,
        "use_relative_value": True,
        "use_conditional_layernorm": True,
        "condition_value_on_global": True,
        "condition_relative_value_on_global": True,
        # 正式训练默认使用可微的方向惩罚；HydroTransformer 直接构造时仍默认 none，
        # 以便旧 checkpoint 和独立调用保留原有双向 attention 语义。
        "directional_attention_mode": "soft",
        "directional_soft_activation": "relu",
        "directional_soft_strength": 1.0,
        "directional_tolerance": 1.0e-6,
    },
    "training": {
        # 用训练集正标签 5% 分位数稳定分母，监督最终总阻力的相对平方误差。
        "loss": "stabilized_relative_mse",
        "relative_loss_floor_strategy": "train_positive_quantile",
        "relative_loss_floor_quantile": 0.05,
        "batch_size": 32,
        # 仅为训练样本追加 y 取负、angle 取负并归一化的镜像，验证和测试不增强。
        "mirror_augmentation": DEFAULT_TRAINING_MIRROR_AUGMENTATION,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "max_epochs": 500,
        "early_stopping_patience": 50,
        "gradient_clip": 1.0,
        "num_workers": 0,
        "device": "auto",
        # 每 10 个 epoch 在终端打印一次 train/validation loss。
        "progress_interval_epochs": 10,
        # loss_curve.png 常规采样点间隔；early stop 的最后一点会额外保留。
        "loss_plot_interval_epochs": 10,
        # 完整续训 checkpoint 每 100 个 epoch 保存一次；结束和 early stop 时强制保存。
        "checkpoint_interval_epochs": 100,
    },
    "cross_validation": {
        # 默认保持同一 model_id 完整，避免相近角度和流速跨集合造成数据泄漏。
        "split_mode": "model",
        "n_splits": 5,
        "validation_fraction": 0.2,
    },
    "output": {"artifact_dir": "model/artifacts"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并字典；标量和列表由用户配置整体覆盖。"""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    """读取 YAML，并用 :data:`DEFAULT_CONFIG` 补全遗漏字段。"""

    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    with Path(path).open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    if not isinstance(user_config, dict):
        raise ValueError("配置文件顶层必须是键值映射。")
    config = _deep_merge(DEFAULT_CONFIG, user_config)
    # 必须使用 YAML 布尔值，避免字符串 "false" 被 Python 当成开启。
    if not isinstance(config["training"]["mirror_augmentation"], bool):
        raise ValueError("training.mirror_augmentation 必须是布尔值 true 或 false。")
    return config


def save_config_snapshot(config: dict[str, Any], path: str | Path) -> None:
    """把实际使用的配置保存为稳定、便于审计的 JSON。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
