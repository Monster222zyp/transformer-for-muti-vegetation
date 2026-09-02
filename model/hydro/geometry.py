"""37 点正六边形的坐标与构型解析工具。"""

from __future__ import annotations

import math
from typing import Sequence


# 七行长度与 Experiment 的点位编号完全一致，元素总数为 37。
ROW_LENGTHS = (4, 5, 6, 7, 6, 5, 4)
POINT_COUNT = sum(ROW_LENGTHS)


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
