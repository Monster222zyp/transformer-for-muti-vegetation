"""多水草 HydroTransformer 的核心数据与模型组件。"""

from .data import HydroDataset, collate_hydro_samples, hydro_collate_fn
from .geometry import build_hex_coordinates, build_layout_index, layout_to_positions
from .physics import (
    DEFAULT_PHYSICAL_CONFIG_PATH,
    PhysicalConfig,
    load_physical_config,
    physical_config_from_mapping,
)

__all__ = [
    "HydroDataset",
    "PhysicalConfig",
    "DEFAULT_PHYSICAL_CONFIG_PATH",
    "build_hex_coordinates",
    "build_layout_index",
    "collate_hydro_samples",
    "hydro_collate_fn",
    "layout_to_positions",
    "load_physical_config",
    "physical_config_from_mapping",
]
