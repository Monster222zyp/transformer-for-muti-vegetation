"""从配置文件读取 checkpoint 与 sample CSV，绘制逐株预测系数热力图。

运行方式（在仓库任意位置均可）：

    py -3.11 interpretability/plot_plant_coefficients.py

也可以显式指定另一份配置：

    py -3.11 interpretability/plot_plant_coefficients.py --config path/to/config.yaml

脚本不会在代码中保存 checkpoint 路径、输入 sample 或绘图参数；所有可调整参数
都位于独立的 ``config.yaml``。输入 CSV 必须采用 ``summarized_data.csv`` 的 11 列
格式，每行代表一个 sample。

模型输出的逐株 ``c`` 是 latent coefficient（潜在系数）。总预测阻力满足
``predicted_drag = Σ(c_i × single_drag_i)``，因此这里绘制的 ``c`` 不等同于可以
直接测量的单株阻力，也不同于整体阻力比值 ``C``。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

# 解释性脚本只输出 PNG，不依赖图形桌面；必须在导入 pyplot 前设置无窗口后端。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import Normalize
from matplotlib.path import Path as MatplotlibPath


# SCRIPT_DIRECTORY 仅用于定位默认配置和仓库模块，不属于用户可调绘图参数。
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIRECTORY / "config.yaml"

# 从 interpretability 子目录直接执行时，Python 默认看不到仓库根目录下的 model 包。
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# 设置中文字体回退，避免中文标题在常见 Windows/Linux 环境中显示为方框。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Noto Sans SC",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# 项目模块在仓库根目录加入 sys.path 后导入，保证脚本从任意工作目录运行都一致。
from model.hydro.data import HydroDataset, collate_hydro_samples
from model.hydro.geometry import build_hex_coordinates
from model.hydro.physics import load_physical_config
from model.models import HydroTransformer
from model.training.checkpoint import load_checkpoint
from model.training.trainer import GlobalFeatureScaler


@dataclass(frozen=True)
class ScriptConfig:
    """经过完整校验、路径已解析的脚本配置。

    属性：
        checkpoint_path: 训练产生的 ``.pt`` checkpoint。
        samples_csv_path: 使用 summarized_data.csv 契约的输入 CSV。
        output_directory: PNG 输出目录。
        device_name: ``auto``、``cpu`` 或 ``cuda``。
        gaussian_sigma: 高斯影响标准差，坐标单位为相邻格点距离。
        gaussian_boundary_padding_rings: 连续图在最外层水草之外扩展的圈数。
        heatmap_grid_size: 连续图每个方向的采样点数。
        contour_levels: 连续热力图的等值填色层级数量。
        image_dpi: PNG 分辨率。
        figure_size_inches: Matplotlib 图像宽、高，单位为 inch。
        point_colormap: 离散格点图 colormap 名称。
        gaussian_colormap: 连续高斯图 colormap 名称。
        annotate_coefficients: 是否在水草格点显示 c 数值。
        show_plant_centers: 是否在连续图叠加水草中心。
        shared_color_scale: 是否让 CSV 中的所有 sample 共用色标。
    """

    checkpoint_path: Path
    samples_csv_path: Path
    output_directory: Path
    device_name: str
    gaussian_sigma: float
    gaussian_boundary_padding_rings: float
    heatmap_grid_size: int
    contour_levels: int
    image_dpi: int
    figure_size_inches: tuple[float, float]
    point_colormap: str
    gaussian_colormap: str
    annotate_coefficients: bool
    show_plant_centers: bool
    shared_color_scale: bool


@dataclass(frozen=True)
class SamplePrediction:
    """一条 CSV sample 的模型推理结果。

    属性：
        csv_row_index: CSV 中从零开始、不含表头的数据行索引。
        model_id: 实验模型编号。
        angle: 实验旋转角度，单位为 degree。
        state_id: 水草状态编号 1 或 2。
        flow_speed: 流速，单位为 m/s。
        positions: 真实水草坐标，形状为 ``[N, 2]``。
        coefficients: 与 positions 对齐的逐株系数 c，形状为 ``[N]``。
        predicted_drag: 模型预测的总阻力，单位为 N。
    """

    csv_row_index: int
    model_id: int
    angle: int
    state_id: int
    flow_speed: float
    positions: np.ndarray
    coefficients: np.ndarray
    predicted_drag: float


@dataclass(frozen=True)
class GaussianField:
    """一条 sample 对应的高斯叠加网格及六边形边界。"""

    grid_x: np.ndarray
    grid_y: np.ndarray
    values: np.ma.MaskedArray
    boundary: np.ndarray


def require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """读取必需的 YAML mapping，缺失或类型错误时指出完整字段名。

    参数：
        parent: 包含目标字段的上级 mapping。
        key: 目标字段名称。

    返回值：
        对应字段的 mapping 值。
    """

    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.yaml 的 {key} 必须是键值映射。")
    return value


def require_value(parent: Mapping[str, Any], section: str, key: str) -> Any:
    """读取必需配置值，不允许脚本静默使用隐藏默认值。

    参数：
        parent: 当前 YAML section。
        section: section 名称，用于错误信息。
        key: 字段名称。

    返回值：
        YAML 中的原始配置值。
    """

    if key not in parent:
        raise ValueError(f"config.yaml 缺少必需字段 {section}.{key}。")
    return parent[key]


def resolve_config_path(raw_path: Any, config_directory: Path, field_name: str) -> Path:
    """将配置中的相对路径统一解释为相对于 config.yaml 所在目录。

    参数：
        raw_path: YAML 中的路径字符串。
        config_directory: config.yaml 的父目录。
        field_name: 完整字段名，用于报告空路径。

    返回值：
        绝对规范化路径。
    """

    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"config.yaml 的 {field_name} 必须是非空路径字符串。")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = config_directory / candidate
    return candidate.resolve()


def load_script_config(config_path: Path) -> ScriptConfig:
    """读取并严格验证独立 YAML 配置文件。

    参数：
        config_path: 用户指定或脚本默认的 config.yaml 路径。

    返回值：
        所有路径已解析、所有数值已校验的不可变配置对象。
    """

    resolved_config_path = config_path.expanduser().resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{resolved_config_path}")
    with resolved_config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    if not isinstance(raw_config, Mapping):
        raise ValueError("config.yaml 顶层必须是键值映射。")

    paths = require_mapping(raw_config, "paths")
    runtime = require_mapping(raw_config, "runtime")
    visualization = require_mapping(raw_config, "visualization")
    config_directory = resolved_config_path.parent

    device_name = str(require_value(runtime, "runtime", "device")).casefold()
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("runtime.device 只能为 auto、cpu 或 cuda。")

    gaussian_sigma = float(require_value(visualization, "visualization", "gaussian_sigma"))
    gaussian_boundary_padding_rings = float(
        require_value(
            visualization,
            "visualization",
            "gaussian_boundary_padding_rings",
        )
    )
    heatmap_grid_size = int(require_value(visualization, "visualization", "heatmap_grid_size"))
    contour_levels = int(require_value(visualization, "visualization", "contour_levels"))
    image_dpi = int(require_value(visualization, "visualization", "image_dpi"))
    if gaussian_sigma <= 0:
        raise ValueError("visualization.gaussian_sigma 必须大于 0。")
    if gaussian_boundary_padding_rings < 0:
        raise ValueError("visualization.gaussian_boundary_padding_rings 不能小于 0。")
    if heatmap_grid_size < 32:
        raise ValueError("visualization.heatmap_grid_size 必须至少为 32。")
    if contour_levels < 2:
        raise ValueError("visualization.contour_levels 必须至少为 2。")
    if image_dpi <= 0:
        raise ValueError("visualization.image_dpi 必须大于 0。")

    raw_figure_size = require_value(visualization, "visualization", "figure_size_inches")
    if not isinstance(raw_figure_size, list) or len(raw_figure_size) != 2:
        raise ValueError("visualization.figure_size_inches 必须是 [宽, 高]。")
    figure_size = (float(raw_figure_size[0]), float(raw_figure_size[1]))
    if min(figure_size) <= 0:
        raise ValueError("visualization.figure_size_inches 的宽和高都必须大于 0。")

    point_colormap = str(require_value(visualization, "visualization", "point_colormap"))
    gaussian_colormap = str(require_value(visualization, "visualization", "gaussian_colormap"))
    for field_name, colormap_name in (
        ("point_colormap", point_colormap),
        ("gaussian_colormap", gaussian_colormap),
    ):
        if colormap_name not in matplotlib.colormaps:
            raise ValueError(f"visualization.{field_name} 不是有效 Matplotlib colormap：{colormap_name}")

    boolean_values: dict[str, bool] = {}
    for boolean_key in ("annotate_coefficients", "show_plant_centers", "shared_color_scale"):
        raw_boolean = require_value(visualization, "visualization", boolean_key)
        if not isinstance(raw_boolean, bool):
            raise ValueError(f"visualization.{boolean_key} 必须为 true 或 false。")
        boolean_values[boolean_key] = raw_boolean

    return ScriptConfig(
        checkpoint_path=resolve_config_path(
            require_value(paths, "paths", "checkpoint_path"),
            config_directory,
            "paths.checkpoint_path",
        ),
        samples_csv_path=resolve_config_path(
            require_value(paths, "paths", "samples_csv_path"),
            config_directory,
            "paths.samples_csv_path",
        ),
        output_directory=resolve_config_path(
            require_value(paths, "paths", "output_directory"),
            config_directory,
            "paths.output_directory",
        ),
        device_name=device_name,
        gaussian_sigma=gaussian_sigma,
        gaussian_boundary_padding_rings=gaussian_boundary_padding_rings,
        heatmap_grid_size=heatmap_grid_size,
        contour_levels=contour_levels,
        image_dpi=image_dpi,
        figure_size_inches=figure_size,
        point_colormap=point_colormap,
        gaussian_colormap=gaussian_colormap,
        annotate_coefficients=boolean_values["annotate_coefficients"],
        show_plant_centers=boolean_values["show_plant_centers"],
        shared_color_scale=boolean_values["shared_color_scale"],
    )


def resolve_device(device_name: str) -> torch.device:
    """把已验证的设备名称转换为实际 PyTorch device。"""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config 指定 cuda，但当前 PyTorch 未检测到可用 CUDA。")
    return torch.device(device_name)


def load_model_bundle(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[HydroTransformer, GlobalFeatureScaler, Mapping[str, Any]]:
    """严格恢复模型、训练期 scaler 和 checkpoint metadata。

    参数：
        checkpoint_path: config 指定的 .pt 文件。
        device: 模型推理设备。

    返回值：
        ``(eval 模型, 训练期 scaler, checkpoint metadata)``。
    """

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")
    raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = raw_checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint 缺少完整 model_config。")

    model = HydroTransformer(**model_config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    metadata = checkpoint.get("checkpoint_metadata")
    scaler_payload = checkpoint.get("scaler")
    if not isinstance(metadata, Mapping) or "physical_config" not in metadata:
        raise ValueError("checkpoint 缺少 checkpoint_metadata.physical_config。")
    if not isinstance(scaler_payload, dict):
        raise ValueError("checkpoint 缺少训练期 scaler，不能重新拟合输入 sample。")
    model.eval()
    return model, GlobalFeatureScaler.from_dict(scaler_payload), metadata


def predict_all_samples(
    config: ScriptConfig,
    model: HydroTransformer,
    scaler: GlobalFeatureScaler,
    metadata: Mapping[str, Any],
    device: torch.device,
) -> list[SamplePrediction]:
    """直接从 config 指定的 CSV 读取并逐行预测全部 sample。

    参数：
        config: 完整脚本配置。
        model: 已加载 checkpoint 的模型。
        scaler: checkpoint 中训练期全局特征 scaler。
        metadata: checkpoint metadata，提供冻结的物理参数与标签策略。
        device: 推理设备。

    返回值：
        与输入 CSV 数据行顺序完全一致的预测结果。
    """

    if not config.samples_csv_path.is_file():
        raise FileNotFoundError(f"找不到输入 sample CSV：{config.samples_csv_path}")
    physical_config = load_physical_config(metadata["physical_config"])
    negative_target_policy = str(metadata.get("negative_target_policy", "clamp_to_zero"))

    # HydroDataset 负责严格验证 11 列表头、数值、37 位排布、state/angle 和流速物理表，
    # 因而解释脚本与训练入口共享完全相同的输入契约。
    dataset = HydroDataset(
        config.samples_csv_path,
        negative_target_policy=negative_target_policy,
        physical_config=physical_config,
    )
    predictions: list[SamplePrediction] = []

    # 一次运行固定遍历 CSV 的每一行，不提供抽样或跳过选项；因此每个 sample 都会
    # 在后续绘图阶段产生一张格点图和一张连续高斯图。
    for csv_row_index in range(len(dataset)):
        sample = dataset[csv_row_index]
        batch = collate_hydro_samples([sample])
        positions = batch["positions"].to(device)
        single_drag = batch["single_drag"].to(device)
        plant_mask = batch["plant_mask"].to(device)
        plant_state = batch["plant_state"].to(device)
        global_features = scaler.transform(batch["global_features"].to(device))

        with torch.inference_mode():
            outputs = model(
                positions=positions,
                single_drag=single_drag,
                global_features=global_features,
                plant_mask=plant_mask,
                plant_state=plant_state,
            )

        plant_count = int(plant_mask[0].sum().item())
        predictions.append(
            SamplePrediction(
                csv_row_index=csv_row_index,
                model_id=int(sample["model_id"]),
                angle=int(sample["angle"]),
                state_id=int(sample["state_id"]),
                flow_speed=float(sample["flow_speed"]),
                positions=positions[0, :plant_count].detach().cpu().numpy(),
                coefficients=outputs["coefficient"][0, :plant_count].detach().cpu().numpy(),
                predicted_drag=float(outputs["total_drag"][0].item()),
            )
        )
    return predictions


def to_display_coordinates(physical_coordinates: np.ndarray) -> np.ndarray:
    """把模型物理坐标 ``(x,y)`` 转为图像坐标 ``(-y,x)``。

    训练约定 +x 指向上方、+y 指向左方；转换后仍保持 +x 在图像上方，同时让
    Matplotlib 的横轴向右增长。
    """

    return np.column_stack((-physical_coordinates[:, 1], physical_coordinates[:, 0]))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """使用单调链算法计算 37 点阵的正六边形凸包。"""

    ordered_points = sorted({(float(x), float(y)) for x, y in points})
    if len(ordered_points) < 3:
        raise ValueError("计算六边形边界至少需要三个不同点。")

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        """返回三个二维点形成的转向叉积。"""

        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in ordered_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def expand_hexagon_boundary(boundary: np.ndarray, padding_rings: float) -> np.ndarray:
    """将六边形边界沿中心向外扩展指定圈数。

    当前 37 点阵是半径为 3 的六边形：最外层水草中心正好位于凸包边界上。
    相邻格点距离为 1，因此扩大一圈等价于把六边形顶点半径从 3 增加到 4。

    参数：
        boundary: 原始 37 点凸包的六个顶点，形状为 ``[6,2]``。
        padding_rings: 需要向外增加的圈数；允许使用小数，0 表示不扩展。

    返回值：
        中心和方向保持不变、半径扩展后的六边形顶点。
    """

    if padding_rings < 0:
        raise ValueError("padding_rings 不能小于 0。")
    center = boundary.mean(axis=0)
    centered_vertices = boundary - center
    vertex_radii = np.linalg.norm(centered_vertices, axis=1)
    original_radius = float(vertex_radii.mean())
    if original_radius <= 0:
        raise ValueError("原始六边形边界半径必须大于 0。")

    # 正六边形的顶点半径等于边长，也等于沿点阵增加一圈所增加的长度单位。
    scale = (original_radius + padding_rings) / original_radius
    return center + centered_vertices * scale


def build_gaussian_field(
    prediction: SamplePrediction,
    all_display_coordinates: np.ndarray,
    sigma: float,
    boundary_padding_rings: float,
    grid_size: int,
) -> GaussianField:
    """将所有水草的 ``c × Gaussian`` 直接求和并裁剪到完整六边形。

    参数：
        prediction: 一条 sample 的坐标与逐株 c。
        all_display_coordinates: 完整 37 点展示坐标。
        sigma: 二维高斯标准差。
        boundary_padding_rings: 原始 37 点凸包之外需要扩展的圈数。
        grid_size: 每个方向的连续网格采样数。

    返回值：
        六边形内有效、六边形外掩膜的连续影响场。
    """

    # 原始凸包经过最外侧水草中心。向外扩大一圈后，边缘水草的高斯尾部仍位于
    # 可视区域内，不会刚到中心位置就被六边形裁剪掉。
    boundary = expand_hexagon_boundary(
        convex_hull(all_display_coordinates),
        boundary_padding_rings,
    )
    horizontal_values = np.linspace(boundary[:, 0].min(), boundary[:, 0].max(), grid_size)
    vertical_values = np.linspace(boundary[:, 1].min(), boundary[:, 1].max(), grid_size)
    grid_x, grid_y = np.meshgrid(horizontal_values, vertical_values)
    plant_display_coordinates = to_display_coordinates(prediction.positions)

    # 广播一次计算 N 株水草到整个网格的距离，然后执行用户要求的高斯影响直接求和。
    squared_distance = (
        (grid_x[None] - plant_display_coordinates[:, 0, None, None]) ** 2
        + (grid_y[None] - plant_display_coordinates[:, 1, None, None]) ** 2
    )
    gaussian_weights = np.exp(-squared_distance / (2.0 * sigma**2))
    influence = np.sum(prediction.coefficients[:, None, None] * gaussian_weights, axis=0)

    boundary_path = MatplotlibPath(boundary)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    inside_boundary = boundary_path.contains_points(grid_points, radius=1.0e-10).reshape(grid_x.shape)
    return GaussianField(
        grid_x=grid_x,
        grid_y=grid_y,
        values=np.ma.array(influence, mask=~inside_boundary),
        boundary=boundary,
    )


def safe_upper_bound(values: np.ndarray | np.ma.MaskedArray) -> float:
    """返回适合 Normalize 的正上界，避免全零数组造成退化色标。"""

    maximum = float(np.ma.max(values))
    return maximum if maximum > 0 else 1.0e-12


def make_point_normalizations(
    predictions: Sequence[SamplePrediction],
    shared: bool,
) -> list[Normalize]:
    """创建离散逐株 c 图的色标范围。

    参数：
        predictions: CSV 全部 sample 的推理结果。
        shared: 为 true 时所有 sample 使用统一色标，否则每张图自适应。

    返回值：
        与 predictions 一一对应的 Normalize 列表。
    """

    if shared:
        coefficient_maximum = max(safe_upper_bound(item.coefficients) for item in predictions)
        return [Normalize(vmin=0.0, vmax=coefficient_maximum)] * len(predictions)
    return [
        Normalize(vmin=0.0, vmax=safe_upper_bound(item.coefficients))
        for item in predictions
    ]


def find_shared_gaussian_upper_bound(
    predictions: Sequence[SamplePrediction],
    all_display_coordinates: np.ndarray,
    config: ScriptConfig,
) -> float:
    """逐条计算连续场最大值，获得统一色标上界且不缓存全部大网格。

    参数：
        predictions: CSV 全部 sample 的预测结果。
        all_display_coordinates: 完整 37 点展示坐标。
        config: 提供高斯标准差和网格尺寸。

    返回值：
        所有 sample 连续高斯影响的最大值。
    """

    maximum = 0.0
    for prediction in predictions:
        field = build_gaussian_field(
            prediction,
            all_display_coordinates,
            config.gaussian_sigma,
            config.gaussian_boundary_padding_rings,
            config.heatmap_grid_size,
        )
        maximum = max(maximum, safe_upper_bound(field.values))
    return maximum


def output_stem(prediction: SamplePrediction) -> str:
    """用 CSV 行号和实验条件生成唯一、可排序的输出文件名前缀。"""

    flow_text = f"{prediction.flow_speed:g}".replace(".", "p")
    return (
        f"row_{prediction.csv_row_index:04d}_model_{prediction.model_id:02d}_"
        f"angle_{prediction.angle:03d}_flow_{flow_text}"
    )


def plot_point_coefficients(
    prediction: SamplePrediction,
    all_display_coordinates: np.ndarray,
    normalization: Normalize,
    config: ScriptConfig,
    output_path: Path,
) -> None:
    """绘制第一类图：只在真实水草格点按逐株 c 填色。"""

    figure, axis = plt.subplots(figsize=config.figure_size_inches, constrained_layout=True)
    plant_display_coordinates = to_display_coordinates(prediction.positions)

    # 白色六边形显示全部 37 个可用格点；随后只覆盖真实水草，空格点不会被误认为低 c。
    axis.scatter(
        all_display_coordinates[:, 0],
        all_display_coordinates[:, 1],
        marker="h",
        s=730,
        facecolors="white",
        edgecolors="#9aa0a6",
        linewidths=0.8,
        zorder=1,
    )
    colored_points = axis.scatter(
        plant_display_coordinates[:, 0],
        plant_display_coordinates[:, 1],
        c=prediction.coefficients,
        cmap=config.point_colormap,
        norm=normalization,
        marker="h",
        s=730,
        edgecolors="black",
        linewidths=0.9,
        zorder=2,
    )
    if config.annotate_coefficients:
        for point, coefficient in zip(plant_display_coordinates, prediction.coefficients):
            axis.text(
                point[0],
                point[1],
                f"{coefficient:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                zorder=3,
            )

    colorbar = figure.colorbar(colored_points, ax=axis, pad=0.02)
    colorbar.set_label("latent coefficient c")
    axis.set_title(
        "逐株预测系数 c（仅有水草的格点）\n"
        f"CSV row={prediction.csv_row_index}, model={prediction.model_id}, "
        f"angle={prediction.angle}°, state={prediction.state_id}, "
        f"flow={prediction.flow_speed:g} m/s, predicted drag={prediction.predicted_drag:.4f} N"
    )
    axis.set_aspect("equal")
    axis.set_axis_off()
    figure.savefig(output_path, dpi=config.image_dpi, bbox_inches="tight")
    plt.close(figure)


def plot_gaussian_influence(
    prediction: SamplePrediction,
    field: GaussianField,
    normalization: Normalize,
    config: ScriptConfig,
    output_path: Path,
) -> None:
    """绘制第二类图：完整六边形内的高斯叠加影响场。"""

    figure, axis = plt.subplots(figsize=config.figure_size_inches, constrained_layout=True)
    contour = axis.contourf(
        field.grid_x,
        field.grid_y,
        field.values,
        # 显式使用 Normalize 的上下界生成层级，保证共享色标时每张图的颜色边界一致。
        levels=np.linspace(
            float(normalization.vmin),
            float(normalization.vmax),
            config.contour_levels,
        ),
        cmap=config.gaussian_colormap,
        norm=normalization,
    )

    # 显式闭合六边形边界，连续颜色只保留在该边界内部。
    closed_boundary = np.vstack((field.boundary, field.boundary[0]))
    axis.plot(closed_boundary[:, 0], closed_boundary[:, 1], color="white", linewidth=1.2)
    if config.show_plant_centers:
        plant_display_coordinates = to_display_coordinates(prediction.positions)
        axis.scatter(
            plant_display_coordinates[:, 0],
            plant_display_coordinates[:, 1],
            s=24,
            c="white",
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    colorbar = figure.colorbar(contour, ax=axis, pad=0.02)
    colorbar.set_label("summed Gaussian c influence")
    axis.set_title(
        "完整六边形的高斯叠加影响\n"
        f"CSV row={prediction.csv_row_index}, σ={config.gaussian_sigma:g}, "
        f"model={prediction.model_id}, angle={prediction.angle}°, "
        f"flow={prediction.flow_speed:g} m/s"
    )
    axis.set_aspect("equal")
    axis.set_axis_off()
    figure.savefig(output_path, dpi=config.image_dpi, bbox_inches="tight")
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    """创建只负责指定独立 config 文件位置的命令行解析器。"""

    parser = argparse.ArgumentParser(
        description="从 config 指定的 CSV 读取 sample，并绘制模型逐株 c 热力图。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="独立 YAML 配置文件；默认 interpretability/config.yaml。",
    )
    return parser


def main() -> int:
    """执行配置读取、CSV 推理和两类热力图输出。"""

    arguments = build_argument_parser().parse_args()
    try:
        config = load_script_config(arguments.config)
        device = resolve_device(config.device_name)
        model, scaler, metadata = load_model_bundle(config.checkpoint_path, device)
        predictions = predict_all_samples(config, model, scaler, metadata, device)

        all_physical_coordinates = np.asarray(build_hex_coordinates(), dtype=float)
        all_display_coordinates = to_display_coordinates(all_physical_coordinates)
        point_normalizations = make_point_normalizations(
            predictions,
            config.shared_color_scale,
        )

        # 共享连续色标时先逐条扫描最大值，但不保存 332 个大网格；正式绘图时仍逐条
        # 构建、保存并释放，峰值内存基本不随 CSV 行数增长。
        shared_gaussian_upper_bound = (
            find_shared_gaussian_upper_bound(
                predictions,
                all_display_coordinates,
                config,
            )
            if config.shared_color_scale
            else None
        )

        config.output_directory.mkdir(parents=True, exist_ok=True)
        for prediction, point_norm in zip(
            predictions,
            point_normalizations,
        ):
            field = build_gaussian_field(
                prediction,
                all_display_coordinates,
                config.gaussian_sigma,
                config.gaussian_boundary_padding_rings,
                config.heatmap_grid_size,
            )
            gaussian_upper_bound = (
                shared_gaussian_upper_bound
                if shared_gaussian_upper_bound is not None
                else safe_upper_bound(field.values)
            )
            gaussian_norm = Normalize(vmin=0.0, vmax=gaussian_upper_bound)
            stem = output_stem(prediction)
            point_path = config.output_directory / f"{stem}_point_coefficients.png"
            gaussian_path = config.output_directory / f"{stem}_gaussian_influence.png"
            plot_point_coefficients(
                prediction,
                all_display_coordinates,
                point_norm,
                config,
                point_path,
            )
            plot_gaussian_influence(
                prediction,
                field,
                gaussian_norm,
                config,
                gaussian_path,
            )
            print(f"[完成] {point_path}")
            print(f"[完成] {gaussian_path}")
        return 0
    except Exception as error:
        print(f"[失败] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
