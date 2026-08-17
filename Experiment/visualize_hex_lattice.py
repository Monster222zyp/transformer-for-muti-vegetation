import argparse
import csv
import math
from pathlib import Path

from generator import get_rotated_groups


ROW_LENGTHS = [4, 5, 6, 7, 6, 5, 4]
TOTAL_POINTS = sum(ROW_LENGTHS)  # 37

# =============================================================================
# 用户可以直接修改的默认参数
# =============================================================================

# SCRIPT_DIR 是当前脚本所在的 Experiment 文件夹。
SCRIPT_DIR = Path(__file__).resolve().parent

# 输入 CSV：每个样本包含 37 个六边形点位数值。
DEFAULT_INPUT_PATH = SCRIPT_DIR / "input.csv"

# 所有样本组成的大图输出路径。
DEFAULT_OVERVIEW_OUTPUT_PATH = SCRIPT_DIR / "hex_lattice.svg"

# 每组样本的六张旋转图输出根目录。
DEFAULT_ROTATIONS_OUTPUT_ROOT = SCRIPT_DIR / "rotated_plots"

# 大图中每行显示的样本数量。
DEFAULT_COLUMNS = 4


def build_hex_coordinates():
    coords = []
    for r, row_len in enumerate(ROW_LENGTHS):
        x_start = -(row_len - 1) / 2.0
        y = -r * (math.sqrt(3) / 2.0)
        for c in range(row_len):
            x = x_start + c
            coords.append((x, y))
    return coords


def read_input_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for raw_row in reader:
            cleaned = [cell.strip() for cell in raw_row if cell.strip() != ""]
            if not cleaned:
                continue
            try:
                values = [float(cell) for cell in cleaned]
            except ValueError:
                continue

            if len(values) == TOTAL_POINTS:
                rows.append(values)
            elif len(values) == 1:
                rows.append(values)
            else:
                raise ValueError(
                    f"Found a row with length {len(values)}; expected {TOTAL_POINTS} values."
                )

    if not rows:
        raise ValueError("No valid numeric rows found in input CSV.")

    # Support single-column 37-row format.
    if all(len(r) == 1 for r in rows):
        if len(rows) != TOTAL_POINTS:
            raise ValueError(
                f"Single-column input detected, but row count is {len(rows)}; expected {TOTAL_POINTS}."
            )
        return [[r[0] for r in rows]]

    if any(len(r) != TOTAL_POINTS for r in rows):
        raise ValueError("Inconsistent row lengths in CSV.")

    return rows


def build_neighbor_pairs(coords):
    threshold = 1.001
    pairs = []
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        for j in range(i + 1, n):
            x2, y2 = coords[j]
            if math.hypot(x1 - x2, y1 - y2) <= threshold:
                pairs.append((i, j))
    return pairs


def lerp(a, b, t):
    return a + (b - a) * t


def value_to_hex_color(v, vmin, vmax):
    # 对于 0、1、2、3 这四个离散类别，固定使用用户指定的颜色。
    # SVG 颜色使用十六进制格式：0 是白色，1 是灰色，2 是黄色，3 是黑色。
    discrete_colors = {
        0: "#ffffff",
        1: "#808080",
        2: "#ffff00",
        3: "#000000",
    }

    # CSV 读取后的数值通常是 float，例如 1 会变成 1.0。
    # 这里先四舍五入并检查误差，确保 1.0 这类值可以正确匹配到离散颜色。
    rounded_value = int(round(v))
    if rounded_value in discrete_colors and abs(v - rounded_value) < 1e-9:
        return discrete_colors[rounded_value]

    # 如果后续输入里出现了 0、1、2、3 之外的连续数值，保留原来的 Viridis-like 渐变逻辑作为兜底。
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    if vmax <= vmin:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))

    segment = t * (len(stops) - 1)
    i = int(math.floor(segment))
    if i >= len(stops) - 1:
        i = len(stops) - 2
    local_t = segment - i

    r = int(round(lerp(stops[i][0], stops[i + 1][0], local_t)))
    g = int(round(lerp(stops[i][1], stops[i + 1][1], local_t)))
    b = int(round(lerp(stops[i][2], stops[i + 1][2], local_t)))
    return f"#{r:02x}{g:02x}{b:02x}"


def format_legend_value(value):
    # 如果数值是整数，则使用整数格式显示，避免出现 1.0 这种不必要的小数形式。
    if float(value).is_integer():
        return str(int(value))

    # 非整数使用两位小数，保证图例文本简洁且可读。
    return f"{value:.2f}"


def build_legend_values(samples):
    # 先把所有样本中的数值拍平成一个列表，方便统一分析。
    all_values = [value for sample in samples for value in sample]
    unique_values = sorted(set(all_values))

    # 如果样本是 0/1/2/3 这种离散整数值，则固定展示 0、1、2、3 四个图例项，
    # 这样即使某个数值在当前样本里没出现，图例也仍然完整。
    if unique_values and all(float(int(v)) == v for v in unique_values):
        integer_values = [int(v) for v in unique_values]
        if all(0 <= v <= 3 for v in integer_values):
            return [0.0, 1.0, 2.0, 3.0]

    # 当唯一值较少时，直接把唯一值作为图例项，方便逐一查看映射关系。
    if len(unique_values) <= 10:
        return unique_values

    # 当唯一值过多时，抽样 6 个刻度值，避免图例过长。
    tick_count = 6
    vmin = min(unique_values)
    vmax = max(unique_values)
    if vmax <= vmin:
        return [vmin]
    step = (vmax - vmin) / (tick_count - 1)
    return [vmin + i * step for i in range(tick_count)]


def visualize_svg(samples, output_path, show_ids=False, columns=4):
    coords = build_hex_coordinates()
    pairs = build_neighbor_pairs(coords)
    sample_count = len(samples)

    vmin = min(min(s) for s in samples)
    vmax = max(max(s) for s in samples)

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    scale = 86.0
    pad_x = 58.0
    pad_y = 42.0
    title_h = 28.0
    radius = 18.0

    tile_w = (max_x - min_x) * scale + 2 * pad_x
    tile_h = (max_y - min_y) * scale + 2 * pad_y + title_h

    cols = max(1, columns)
    rows = math.ceil(sample_count / cols)
    legend_values = build_legend_values(samples)
    legend_w = 190.0 if legend_values else 0.0
    canvas_w = cols * tile_w + legend_w
    canvas_h = rows * tile_h

    elements = []
    elements.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w:.0f}" height="{canvas_h:.0f}" '
        f'viewBox="0 0 {canvas_w:.0f} {canvas_h:.0f}">'
    )
    elements.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')

    for i, values in enumerate(samples):
        col_idx = i % cols
        row_idx = i // cols
        ox = col_idx * tile_w
        oy = row_idx * tile_h
        top_y = oy + title_h

        elements.append(f'<g id="sample-{i}">')
        elements.append(
            f'<text x="{ox + tile_w / 2:.2f}" y="{oy + 20:.2f}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="16" fill="#222">Sample {i}</text>'
        )

        pixel_coords = []
        for x, y in coords:
            px = ox + pad_x + (x - min_x) * scale
            py = top_y + pad_y + (max_y - y) * scale
            pixel_coords.append((px, py))

        for a, b in pairs:
            x1, y1 = pixel_coords[a]
            x2, y2 = pixel_coords[b]
            elements.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="#D0D0D0" stroke-width="2"/>'
            )

        for idx, (px, py) in enumerate(pixel_coords):
            color = value_to_hex_color(values[idx], vmin, vmax)
            elements.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" fill="{color}" '
                f'stroke="#2f2f2f" stroke-width="1.5"/>'
            )
            if show_ids:
                elements.append(
                    f'<text x="{px:.2f}" y="{py + 4:.2f}" text-anchor="middle" '
                    f'font-family="Arial, sans-serif" font-size="11" fill="#ffffff" '
                    f'font-weight="bold">{idx}</text>'
                )

        elements.append("</g>")

    # 在右侧新增图例区域，明确每种颜色对应的数值。
    if legend_values:
        legend_x = cols * tile_w + 18.0
        legend_y = 28.0
        legend_item_gap = 34.0
        legend_box_w = legend_w - 28.0
        legend_box_h = 56.0 + len(legend_values) * legend_item_gap

        elements.append(
            f'<rect x="{legend_x:.2f}" y="{legend_y:.2f}" width="{legend_box_w:.2f}" '
            f'height="{legend_box_h:.2f}" rx="12" ry="12" fill="#fafafa" stroke="#d8d8d8" stroke-width="1.2"/>'
        )
        elements.append(
            f'<text x="{legend_x + 14:.2f}" y="{legend_y + 24:.2f}" '
            f'font-family="Arial, sans-serif" font-size="14" fill="#222" font-weight="bold">Legend</text>'
        )
        elements.append(
            f'<text x="{legend_x + 14:.2f}" y="{legend_y + 42:.2f}" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#666">Color -> Value</text>'
        )

        for idx, value in enumerate(legend_values):
            cy = legend_y + 64.0 + idx * legend_item_gap
            color = value_to_hex_color(value, vmin, vmax)
            label = format_legend_value(value)
            elements.append(
                f'<circle cx="{legend_x + 18:.2f}" cy="{cy:.2f}" r="10" fill="{color}" '
                f'stroke="#2f2f2f" stroke-width="1.2"/>'
            )
            elements.append(
                f'<text x="{legend_x + 38:.2f}" y="{cy + 4:.2f}" '
                f'font-family="Arial, sans-serif" font-size="13" fill="#222">{label}</text>'
            )

    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def write_single_sample_svg(
    values: list[float],
    output_path: Path,
    title: str,
    show_ids: bool = False,
) -> None:
    """将一个旋转后的 37 点样本保存为一张独立 SVG。"""

    # 使用与大图相同的六边形坐标和相邻点连接关系，保持原有绘图效果。
    coords = build_hex_coordinates()
    pairs = build_neighbor_pairs(coords)

    # 计算六边形点阵的坐标范围，用于确定 SVG 画布大小。
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 当前旋转样本自身的最小值和最大值用于颜色映射。
    vmin = min(values)
    vmax = max(values)

    # 以下绘图尺寸沿用原 export_group_rotations.py 的配置。
    scale = 90.0
    pad_x = 62.0
    pad_y = 44.0
    title_h = 34.0
    radius = 18.0

    canvas_w = (max_x - min_x) * scale + 2 * pad_x
    canvas_h = (max_y - min_y) * scale + 2 * pad_y + title_h

    # 把逻辑坐标转换为 SVG 中使用的像素坐标。
    pixel_coords = []
    top_y = title_h
    for x, y in coords:
        px = pad_x + (x - min_x) * scale
        py = top_y + pad_y + (max_y - y) * scale
        pixel_coords.append((px, py))

    # 构建 SVG 的背景和标题。
    elements = []
    elements.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w:.0f}" height="{canvas_h:.0f}" '
        f'viewBox="0 0 {canvas_w:.0f} {canvas_h:.0f}">'
    )
    elements.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    elements.append(
        f'<text x="{canvas_w / 2:.2f}" y="22" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="17" fill="#222">{title}</text>'
    )

    # 先画相邻点之间的连线，使圆点显示在线条上方。
    for first_index, second_index in pairs:
        x1, y1 = pixel_coords[first_index]
        x2, y2 = pixel_coords[second_index]
        elements.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="#D0D0D0" stroke-width="2"/>'
        )

    # 根据每个点的数值画圆，并按需在圆点上显示 0～36 的点位编号。
    for point_index, (px, py) in enumerate(pixel_coords):
        color = value_to_hex_color(values[point_index], vmin, vmax)
        elements.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" fill="{color}" '
            f'stroke="#2f2f2f" stroke-width="1.5"/>'
        )
        if show_ids:
            elements.append(
                f'<text x="{px:.2f}" y="{py + 4:.2f}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="11" fill="#ffffff" '
                f'font-weight="bold">{point_index}</text>'
            )

    # 创建父目录并写入完整 SVG 文本。
    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def export_all_rotations(
    input_csv: Path,
    output_root: Path,
    show_ids: bool = False,
) -> None:
    """为输入 CSV 中的每组样本导出 0～300 度的六张旋转图。"""

    # 重新读取完整输入，确保 --row 只影响大图，不改变原旋转导出功能。
    samples = read_input_csv(input_csv)

    # 沿用原脚本行为：按非零点数量从少到多排列组编号。
    samples = sorted(
        samples,
        key=lambda row: sum(1 for value in row if float(value) != 0.0),
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # 每组建立独立文件夹，并分别导出六个 60 度间隔的旋转结果。
    for sample_index, sample in enumerate(samples, start=1):
        # 项目结构数据是 0/1 整数；旋转函数也要求整数列表。
        base_values = [int(value) for value in sample]
        rotations = get_rotated_groups(base_values)

        group_dir = output_root / f"group_{sample_index:03d}"
        group_dir.mkdir(parents=True, exist_ok=True)

        for rotation_index, rotation_values in enumerate(rotations):
            angle = rotation_index * 60
            filename = f"rot_{angle:03d}.svg"
            title = f"Group {sample_index} | Rotate {angle} deg"
            write_single_sample_svg(
                values=rotation_values,
                output_path=group_dir / filename,
                title=title,
                show_ids=show_ids,
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize all 37-point samples in one SVG and export six rotated SVGs "
            "for every sample."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OVERVIEW_OUTPUT_PATH,
        help=f"Overview SVG path (default: {DEFAULT_OVERVIEW_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_ROTATIONS_OUTPUT_ROOT,
        help=(
            "Output root directory for rotated SVGs "
            f"(default: {DEFAULT_ROTATIONS_OUTPUT_ROOT})"
        ),
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Visualize one sample row only (0-based index).",
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Show point IDs (0..36) on each dot.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=DEFAULT_COLUMNS,
        help=(
            "Number of subplots per row for multi-sample input "
            f"(default: {DEFAULT_COLUMNS})."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    samples = read_input_csv(input_path)

    if args.row is not None:
        if args.row < 0 or args.row >= len(samples):
            raise IndexError(
                f"--row out of range. Got {args.row}; valid range is 0..{len(samples)-1}."
            )
        samples = [samples[args.row]]

    output_path = Path(args.output)
    if output_path.suffix.lower() != ".svg":
        output_path = output_path.with_suffix(".svg")
        print(f"Changed output extension to SVG: {output_path.name}")

    visualize_svg(
        samples=samples,
        output_path=output_path,
        show_ids=args.show_ids,
        columns=args.columns,
    )
    print(f"Saved visualization to: {output_path.resolve()}")

    # 大图生成完成后，继续执行原 export_group_rotations.py 的全部旋转导出功能。
    rotations_output_root = Path(args.output_root)
    export_all_rotations(
        input_csv=input_path,
        output_root=rotations_output_root,
        show_ids=args.show_ids,
    )
    print(f"Saved rotated plots under: {rotations_output_root.resolve()}")


if __name__ == "__main__":
    main()
