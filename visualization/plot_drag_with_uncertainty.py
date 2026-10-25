"""从交叉验证预测与两轮重复实验绘制阻力均值 ± 1 个样本 SD。

保留现有对比图的三条曲线和排序，不加载 checkpoint，也不重新进行模型预测。
误差棒仅描述实验重复性：每个工况有两轮测量，采用 ddof=1 的样本标准差。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image


# 路径及全部命令行参数的默认值集中在这里，避免依赖启动时的工作目录。
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULT_DIR = REPO_ROOT.parent / "multiR_60deg_sample"
DEFAULT_PREDICTIONS = RESULT_DIR / "cv_predictions.csv"
DEFAULT_REPEATS = REPO_ROOT / "datasets" / "multi_stiffness" / "repeatability.csv"
DEFAULT_OUTPUT_STEM = RESULT_DIR / "cv_drag_comparison_with_uncertainty"
DEFAULT_AUDIT_DIR = REPO_ROOT / "文档" / "工作记录" / "60度误差棒绘图20261021"
DEFAULT_FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
DEFAULT_DPI = 600
DEFAULT_WIDTH_INCHES = 18.0
DEFAULT_HEIGHT_INCHES = 6.0
DEFAULT_MARKER_SIZE = 0.45  # 样本点直径，单位 pt；同时取消点的边框，避免边框放大点。
DEFAULT_LINE_WIDTH = 0.50  # 减细连线，避免覆盖短误差棒。
DEFAULT_ERROR_LINE_WIDTH = 0.90  # 误差棒比连线更粗，使重复性范围优先显示。
DEFAULT_CAP_SIZE = 1.5  # 误差棒横帽的半宽，单位 pt。
DEFAULT_EXPECTED_SAMPLES = 287
DEFAULT_OVERWRITE = False

# 数学定义、数值核对容差和绘图外观固定参数也集中声明，便于复现。
REPEAT_COUNT = 2
SD_DDOF = 1  # 样本标准差：分母为 n-1，不是描述总体的分母 n。
ANGLE_STEP_DEGREES = 60
MEAN_ABSOLUTE_TOLERANCE_N = 1.0e-6  # 模型 CSV 中的标签经过 float32 转换。
MEAN_RELATIVE_TOLERANCE = 1.0e-6
SD_NUMERICAL_TOLERANCE = 1.0e-12
PREVIEW_DPI = 160
COLORS = {"target": "#1f77b4", "prediction": "#ff7f0e", "isolated": "#2ca02c", "error": "#123b63"}
DETAIL_Y_HEADROOM = 1.08  # 细节图在实验上界和预测最大值之上留出 8% 空间。
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE = 10
LEGEND_SIZE = 10
NOTE_SIZE = 9
TITLE = "交叉验证测试样本阻力对比"
X_LABEL = "排序后的样本序号"
Y_LABEL = "阻力（N）"
NOTE = "实验误差棒：均值 ± 1 个样本标准差（SD），每个工况 n=2；样本按单株阻力叠加基线升序排列。"
DETAIL_TITLE = "实验与预测细节：均值 ± 1 SD（n=2）"
DETAIL_NOTE = "与全图使用相同的 287 个样本、排序和误差棒；本图单独显示实验与预测，以便看清实验 SD。"

# 使用无窗口绘图后端，避免批处理过程中弹出绘图窗口。
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt

# 复用原图的唯一排序定义，使新旧图上同一横坐标始终对应同一个样本。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from model.training.visualization import sort_drag_predictions


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 path 的带表头 CSV，兼容 UTF-8 BOM；返回每行的字符串字段字典。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV 没有数据行：{path}")
    return rows


def condition_key(model: object, angle: object, speed: object) -> tuple[int, int, float]:
    """将 model、角度（degree）和流速（m/s）转换为可匹配两份 CSV 的工况键。"""
    values = np.array([float(model), float(angle), float(speed)])
    if not np.isfinite(values).all() or not values[0].is_integer() or not values[1].is_integer():
        raise ValueError(f"model/角度/流速非法：{model}, {angle}, {speed}")
    return int(values[0]), int(values[1]), float(values[2])


def prepare_rows(predictions_path: Path, repeats_path: Path, expected_samples: int) -> list[dict[str, object]]:
    """匹配两份输入并返回原图顺序的数据。

    predictions_path 是合并各折测试集的预测 CSV；repeats_path 含两轮 FX_0；
    expected_samples 用于阻止遗漏样本。每条返回记录含原预测字段和实验均值、SD。
    """
    predictions = read_csv(predictions_path)
    repeats = read_csv(repeats_path)
    if len(predictions) != expected_samples:
        raise ValueError(f"预期 {expected_samples} 个预测样本，实际 {len(predictions)} 个。")
    if len({row["sample_id"] for row in predictions}) != len(predictions):
        raise ValueError("预测 CSV 存在重复 sample_id，不能把重复预测当成独立实验。")

    # 全量工况索引必须唯一；不完整记录和高误差标记只在选中后明确拒绝，不补值。
    repeated_by_key = {}
    for row in repeats:
        key = condition_key(row["model_id"], row["angle_deg"], row["flow_speed_m_s"])
        if key in repeated_by_key:
            raise ValueError(f"重复性 CSV 的工况键重复：{key}")
        repeated_by_key[key] = row

    combined = []
    seen_conditions = set()
    for prediction in predictions:
        key = condition_key(prediction["model_id"], prediction["angle"], prediction["flow_speed"])
        if key[1] % ANGLE_STEP_DEGREES or key in seen_conditions:
            raise ValueError(f"预测含非 60° 整数倍或重复工况：{key}")
        seen_conditions.add(key)
        if key not in repeated_by_key:
            raise ValueError(f"预测样本找不到对应的两轮实验：{prediction['sample_id']}")
        repeat = repeated_by_key[key]
        if repeat["retained"].lower() != "true" or repeat["comparison_status"] != "compared":
            raise ValueError(f"样本未通过重复性筛选或实验记录不完整：{prediction['sample_id']}")
        measurements = np.array([float(repeat["round1_FX_0_N"]), float(repeat["round2_FX_0_N"])])
        if not np.isfinite(measurements).all():
            raise ValueError(f"实验值不是有限数：{prediction['sample_id']}")

        # 使用 CSV 的完整精度计算均值和样本 SD；n=2 时 SD=|F1-F2|/sqrt(2)。
        mean = float(measurements.mean())
        sd = float(measurements.std(ddof=SD_DDOF))
        expected_sd = float(abs(measurements[0] - measurements[1]) / np.sqrt(REPEAT_COUNT))
        if not np.isclose(sd, expected_sd, atol=SD_NUMERICAL_TOLERANCE, rtol=SD_NUMERICAL_TOLERANCE):
            raise ValueError(f"样本 SD 计算不一致：{prediction['sample_id']}")
        raw_target, target = float(prediction["raw_target_drag"]), float(prediction["target_drag"])
        if not np.allclose([raw_target, target], mean, atol=MEAN_ABSOLUTE_TOLERANCE_N, rtol=MEAN_RELATIVE_TOLERANCE):
            raise ValueError(f"预测标签不是这两轮实验的均值：{prediction['sample_id']}")

        # 原预测字段保留不变；绘图实验中心值使用原始重复性 CSV 的双精度均值。
        combined.append({
            **prediction,
            "round1_FX_0_N": float(measurements[0]), "round2_FX_0_N": float(measurements[1]),
            "experiment_mean_N": mean, "experiment_sd_N": sd,
            "experiment_lower_1sd_N": mean - sd, "experiment_upper_1sd_N": mean + sd,
            "target_minus_experiment_mean_N": target - mean,
        })

    # 同时核对 CSV 中全部合格的六角工况，保证此次不是仅绘制预测文件中的一部分。
    eligible = {key for key, row in repeated_by_key.items() if key[1] % ANGLE_STEP_DEGREES == 0
                and row["retained"].lower() == "true" and row["comparison_status"] == "compared"}
    if eligible != seen_conditions:
        raise ValueError("预测工况与全部合格六角实验不一致，不能声明已绘制全部样本。")
    return sort_drag_predictions(combined)


def draw_plot(rows: list[dict[str, object]], arguments: argparse.Namespace, png: Path, svg: Path,
              preview: Path, detail: bool) -> dict[str, int]:
    """以 rows 绘图；detail 决定绘制三曲线全图，还是聚焦实验与预测的细节图。

    arguments 提供样式；png/svg/preview 分别指定高清图、矢量图和预览输出位置。
    """
    if not arguments.font_path.is_file():
        raise FileNotFoundError(f"中文字体不存在，请指定 --font-path：{arguments.font_path}")
    font_manager.fontManager.addfont(str(arguments.font_path))
    font_name = font_manager.FontProperties(fname=arguments.font_path).get_name()
    style = {"font.family": font_name, "axes.unicode_minus": False, "svg.fonttype": "path",
             "font.size": TICK_SIZE, "axes.labelsize": LABEL_SIZE, "xtick.labelsize": TICK_SIZE,
             "ytick.labelsize": TICK_SIZE, "legend.fontsize": LEGEND_SIZE}

    # 样本序号从 1 开始，与现有 cv_drag_comparison.png 的横坐标约定一致。
    x = np.arange(1, len(rows) + 1)
    mean = np.array([row["experiment_mean_N"] for row in rows])
    sd = np.array([row["experiment_sd_N"] for row in rows])
    predicted = np.array([float(row["predicted_drag"]) for row in rows])
    isolated = np.array([float(row["isolated_drag"]) for row in rows])
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(arguments.width, arguments.height))
        try:
            # 深色误差棒显示于圆点之上；仅取消数据点边框，不取消误差棒的横帽。
            bars = axis.errorbar(x, mean, yerr=sd, fmt="o-", color=COLORS["target"],
                                 ecolor=COLORS["error"], markersize=arguments.marker_size,
                                 linewidth=arguments.line_width, elinewidth=arguments.error_line_width,
                                 capsize=arguments.cap_size, capthick=arguments.error_line_width,
                                 barsabove=True, label="实验均值 ± 1 SD（n=2）", zorder=4)
            bars.lines[0].set_markeredgewidth(0.0)
            axis.plot(x, predicted, "o-", color=COLORS["prediction"],
                      markersize=arguments.marker_size, linewidth=arguments.line_width,
                      markeredgewidth=0.0,
                      label="模型预测", zorder=3)
            if not detail:
                axis.plot(x, isolated, "o-", color=COLORS["isolated"],
                          markersize=arguments.marker_size, linewidth=arguments.line_width,
                          markeredgewidth=0.0, label="单株阻力叠加基线", zorder=2)
            axis.set_title(DETAIL_TITLE if detail else TITLE, fontsize=TITLE_SIZE, pad=12)
            axis.set_xlabel(X_LABEL)
            axis.set_ylabel(Y_LABEL)
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
            axis.margins(x=0.045, y=0.07)
            if detail:
                # 完整基线最高约 33 N，会压缩 SD 的视觉高度；细节图单独展示两条测量/预测曲线。
                # 仅改变显示范围，仍绘制全部样本，且不放大或缩放任何 SD 数值。
                axis.set_ylim(0.0, float(max(np.max(mean + sd), np.max(predicted))) * DETAIL_Y_HEADROOM)

            # 手动指定图例次序，以实验、预测、基线的顺序与原图对应。
            handles, labels = axis.get_legend_handles_labels()
            legend_by_label = dict(zip(labels, handles))
            legend_order = ("实验均值 ± 1 SD（n=2）", "模型预测") if detail else (
                "实验均值 ± 1 SD（n=2）", "模型预测", "单株阻力叠加基线")
            axis.legend([legend_by_label[label] for label in legend_order], legend_order,
                        loc="upper left", framealpha=0.95, markerscale=1.8)
            figure.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.17)
            figure.text(0.055, 0.045, DETAIL_NOTE if detail else NOTE, fontsize=NOTE_SIZE, color="#444444")

            # 每个样本应有一条竖向误差棒，避免漏画或只绘制部分样本。
            error_bar_count = len(bars.lines[2][0].get_segments())
            if error_bar_count != len(rows):
                raise ValueError("误差棒数量与样本数不一致。")
            figure.savefig(png, dpi=arguments.dpi, format="png", facecolor="white")
            figure.savefig(svg, format="svg", facecolor="white")
            figure.savefig(preview, dpi=PREVIEW_DPI, format="png", facecolor="white")
        finally:
            plt.close(figure)
    with Image.open(png) as saved:
        width, height = saved.size
        saved.verify()
    return {"samples": len(rows), "error_bars": error_bar_count, "width_pixels": width, "height_pixels": height}


def main() -> None:
    """解析命令行、检查输出路径、保存图与数据并记录复现摘要；不改输入文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS, help="合并交叉验证测试集预测的 CSV")
    parser.add_argument("--repeats", type=Path, default=DEFAULT_REPEATS, help="含两轮 FX_0 的重复性 CSV")
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM, help="PNG/SVG/绘图数据的公共路径，不含扩展名")
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR, help="参数摘要和低分辨率预览的工作记录目录")
    parser.add_argument("--font-path", type=Path, default=DEFAULT_FONT_PATH, help="中文字体文件路径")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PNG 每英寸像素数")
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH_INCHES, help="图宽，inch")
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT_INCHES, help="图高，inch")
    parser.add_argument("--marker-size", type=float, default=DEFAULT_MARKER_SIZE, help="每个样本点直径，pt")
    parser.add_argument("--line-width", type=float, default=DEFAULT_LINE_WIDTH, help="折线宽度，pt")
    parser.add_argument("--error-line-width", type=float, default=DEFAULT_ERROR_LINE_WIDTH, help="误差棒线宽，pt")
    parser.add_argument("--cap-size", type=float, default=DEFAULT_CAP_SIZE, help="误差棒横帽半宽，pt")
    parser.add_argument("--expected-samples", type=int, default=DEFAULT_EXPECTED_SAMPLES, help="预期样本总数，发现遗漏时中止")
    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE, help="允许替换本脚本生成的图和绘图数据")
    arguments = parser.parse_args()
    sizes = (arguments.dpi, arguments.width, arguments.height, arguments.marker_size,
             arguments.line_width, arguments.error_line_width, arguments.cap_size, arguments.expected_samples)
    if any(not np.isfinite(value) or value <= 0 for value in sizes):
        raise ValueError("分辨率、图形尺寸、线宽、点大小及预期样本数必须为正有限数。")

    # 所有输出先统一检查，避免已经写完第一张图才发现另一产物会覆盖已有文件。
    png = arguments.output_stem.with_suffix(".png")
    svg = arguments.output_stem.with_suffix(".svg")
    data_csv = arguments.output_stem.parent / (arguments.output_stem.name + "_data.csv")
    preview = arguments.audit_dir / "绘图预览.png"
    detail_png = arguments.output_stem.parent / (arguments.output_stem.name + "_detail.png")
    detail_svg = arguments.output_stem.parent / (arguments.output_stem.name + "_detail.svg")
    detail_preview = arguments.audit_dir / "误差棒细节预览.png"
    manifest_path = arguments.audit_dir / "绘图摘要.json"
    outputs = (png, svg, data_csv, preview, detail_png, detail_svg, detail_preview, manifest_path)
    input_paths = {arguments.predictions.resolve(), arguments.repeats.resolve()}
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.resolve() in input_paths for path in outputs):
        raise ValueError("各输出路径必须唯一，且不能覆盖输入 CSV。")
    if not arguments.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("输出已存在；确认要重画时使用 --overwrite。")
    input_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    rows = prepare_rows(arguments.predictions, arguments.repeats, arguments.expected_samples)
    arguments.output_stem.parent.mkdir(parents=True, exist_ok=True)
    arguments.audit_dir.mkdir(parents=True, exist_ok=True)
    image_summary = draw_plot(rows, arguments, png, svg, preview, detail=False)
    detail_summary = draw_plot(rows, arguments, detail_png, detail_svg, detail_preview, detail=True)

    # 绘图数据按横坐标顺序保存，方便逐个样本复核均值、SD 和对应 fold。
    with data_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sorted_sample_index", *rows[0].keys()))
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow({"sorted_sample_index": index, **row})
    for path in input_paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != input_hashes[str(path)]:
            raise ValueError(f"绘图过程中输入文件发生改变：{path}")
    manifest = {
        **image_summary, "detail_image": detail_summary,
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()},
        "uncertainty_definition": "两轮独立实验 FX_0 的均值 ±1 个样本SD，ddof=1，n=2；SD=abs(F1-F2)/sqrt(2)",
        "sort_order": ["isolated_drag", "target_drag", "source_index"],
        "fold_counts": dict(Counter(str(row["fold"]) for row in rows)),
        "angle_counts": dict(Counter(str(row["angle"]) for row in rows)),
        "sd_range_N": [min(row["experiment_sd_N"] for row in rows), max(row["experiment_sd_N"] for row in rows)],
        "maximum_label_mean_difference_N": max(abs(row["target_minus_experiment_mean_N"]) for row in rows),
        "input_sha256": input_hashes,
        "output_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs if path != manifest_path},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({**image_summary, "png": str(png.resolve()), "svg": str(svg.resolve()),
                      "detail_png": str(detail_png.resolve()),
                      "sd_range_N": manifest["sd_range_N"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
