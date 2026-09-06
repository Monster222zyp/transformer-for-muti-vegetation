"""HydroTransformer 的训练工具包。"""

from .checkpoint import load_checkpoint, resolved_model_config, save_checkpoint
from .losses import (
    fit_relative_drag_floor,
    interaction_coefficient,
    relative_total_drag_mse_loss,
)
from .metrics import compute_metrics_by_flow_speed, compute_regression_metrics
from .scheduler import WarmupCosineScheduler
from .splits import (
    SUPPORTED_SPLIT_MODES,
    GroupSplit,
    build_cross_validation_splits,
    build_group_kfold_splits,
)
from .visualization import (
    sort_drag_predictions,
    write_drag_comparison_by_flow_speed_plot,
    write_drag_comparison_plot,
)

__all__ = [
    "GroupSplit",
    "WarmupCosineScheduler",
    "SUPPORTED_SPLIT_MODES",
    "build_cross_validation_splits",
    "build_group_kfold_splits",
    "compute_metrics_by_flow_speed",
    "compute_regression_metrics",
    "fit_relative_drag_floor",
    "interaction_coefficient",
    "relative_total_drag_mse_loss",
    "load_checkpoint",
    "resolved_model_config",
    "save_checkpoint",
    "sort_drag_predictions",
    "write_drag_comparison_by_flow_speed_plot",
    "write_drag_comparison_plot",
]
