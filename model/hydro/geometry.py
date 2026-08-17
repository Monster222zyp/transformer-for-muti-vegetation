"""37 点正六边形的坐标、构型解析与旋转反查工具。"""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
from typing import Callable, Sequence


# 七行长度与 Experiment 的点位编号完全一致，元素总数为 37。
ROW_LENGTHS = (4, 5, 6, 7, 6, 5, 4)
POINT_COUNT = sum(ROW_LENGTHS)
VALID_ANGLES = (0, 60, 120, 180, 240, 300)


def build_hex_coordinates() -> list[tuple[float, float]]:
    """返回按点位编号排列的 37 个实验物理坐标 ``(x, y)``。

    ``+x`` 指向点阵上方，也就是主阻力正方向；``+y`` 指向点阵左侧。
    相邻点的中心距离归一化为 1，中心点 18 的坐标为 ``(0, 0)``。
    """
    coordinates: list[tuple[float, float]] = []
    for row_index, row_length in enumerate(ROW_LENGTHS):
        # 竖直相邻行的距离是正三角形高度；越靠上，物理 x 越大。
        x_coordinate = (3 - row_index) * math.sqrt(3.0) / 2.0
        for column_index in range(row_length):
            # 每行以中心对齐，列号越小越靠左，因此物理 y 越大。
            y_coordinate = (row_length - 1) / 2.0 - column_index
            coordinates.append((x_coordinate, y_coordinate))
    if len(coordinates) != POINT_COUNT:
        raise AssertionError("内部坐标生成错误：点位数量不是 37。")
    return coordinates


def parse_layout(layout: str | Sequence[int]) -> tuple[int, ...]:
    """把字符串或整数序列严格转换为可哈希的 37 位构型。"""
    if isinstance(layout, str):
        cleaned = layout.strip()
        if len(cleaned) != POINT_COUNT or set(cleaned) - {"0", "1"}:
            raise ValueError("vegetation_layout 必须是长度为 37 的 01 字符串。")
        return tuple(int(character) for character in cleaned)

    parsed = tuple(int(value) for value in layout)
    if len(parsed) != POINT_COUNT or any(value not in (0, 1) for value in parsed):
        raise ValueError("水草构型必须恰好包含 37 个 0/1 值。")
    return parsed


def layout_to_positions(
    layout: str | Sequence[int],
    coordinates: Sequence[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """筛选构型中值为 1 的固定网格坐标，不执行额外旋转。

    参数：
        layout: 37 位 01 字符串或整数序列。
        coordinates: 可选的 37 点坐标；默认使用实验物理坐标。

    返回：
        按原点位编号排序的有效水草坐标列表。
    """
    parsed_layout = parse_layout(layout)
    all_coordinates = list(coordinates) if coordinates is not None else build_hex_coordinates()
    if len(all_coordinates) != POINT_COUNT:
        raise ValueError(f"coordinates 应包含 {POINT_COUNT} 个坐标。")
    return [all_coordinates[index] for index, occupied in enumerate(parsed_layout) if occupied]


def _load_rotation_function(generator_path: Path) -> Callable[[list[int]], list[list[int]]]:
    """按文件路径加载实验代码中的六方向旋转函数。"""
    if not generator_path.is_file():
        raise FileNotFoundError(f"找不到旋转函数文件：{generator_path}")
    module_spec = importlib.util.spec_from_file_location("hydro_experiment_generator", generator_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"无法加载旋转函数文件：{generator_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    rotation_function = getattr(module, "get_rotated_groups", None)
    if not callable(rotation_function):
        raise AttributeError(f"{generator_path} 未提供 get_rotated_groups。")
    return rotation_function


def build_layout_index(input_csv_path: str | Path) -> dict[str, tuple[int, int]]:
    """建立 ``旋转后构型 -> (model_id, angle)`` 的唯一反向索引。

    参数：
        input_csv_path: Experiment/input.csv 路径；旋转函数从同目录 generator.py 复用。

    返回：
        键为 37 位 01 字符串，值为从 1 开始的模型编号和角度。
    """
    input_path = Path(input_csv_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到构型文件：{input_path}")
    get_rotated_groups = _load_rotation_function(input_path.parent / "generator.py")

    base_layouts: list[list[int]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row_number, row in enumerate(csv.reader(csv_file), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            try:
                layout = list(parse_layout([int(cell.strip()) for cell in row]))
            except (ValueError, TypeError) as error:
                raise ValueError(f"构型文件第 {row_number} 行格式无效。") from error
            base_layouts.append(layout)

    layout_index: dict[str, tuple[int, int]] = {}
    for model_id, base_layout in enumerate(base_layouts, start=1):
        rotations = get_rotated_groups(base_layout)
        if len(rotations) != len(VALID_ANGLES):
            raise ValueError(f"model_{model_id} 的旋转函数没有返回六个方向。")
        for angle, rotated_layout in zip(VALID_ANGLES, rotations):
            layout_key = "".join(str(value) for value in parse_layout(rotated_layout))
            if layout_key in layout_index:
                previous = layout_index[layout_key]
                raise ValueError(
                    f"构型反查不唯一：model_{model_id}/{angle}° 与 "
                    f"model_{previous[0]}/{previous[1]}° 相同。"
                )
            layout_index[layout_key] = (model_id, angle)
    return layout_index
