"""把多份同一 schema 的 JSONL 数据集合并为一份 JSONL。

每行保留原记录内容和输入顺序；若 sample_id 重复或 schema_version 不一致，
脚本会在写出目标文件前报错，避免生成无法追溯的混合数据集。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


# 无参数运行时使用这里列出的数据集。默认留空，防止误把某次实验结果再次合并。
DEFAULT_INPUT_PATHS: tuple[Path, ...] = ()
# 未指定 --output 时的目标文件；相对路径按运行命令时的目录解释。
DEFAULT_OUTPUT_PATH = Path("merged_dataset.jsonl")
# 默认拒绝覆盖已经存在的输出；需要重建时显式传 --overwrite。
DEFAULT_OVERWRITE = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析数据集路径和覆盖开关；命令行输入优先于文件顶部默认值。"""

    parser = argparse.ArgumentParser(description="合并多份 JSONL 数据集。")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="按指定顺序合并的输入 JSONL；也可在脚本顶部填写默认路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="合并后的 JSONL 路径。",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERWRITE,
        help="是否允许替换已存在的输出文件。",
    )
    return parser.parse_args(argv)


def load_records(input_paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    """按输入顺序读取记录并检查 ID、版本及 JSONL 的基本结构。

    参数：
        input_paths: 至少两份源数据集；每个文件每行必须是一个 JSON 对象。

    返回值：
        保留输入文件与行内顺序的记录列表。
    """

    if len(input_paths) < 2:
        raise ValueError("请至少指定两份输入 JSONL。")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    expected_version: int | None = None
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"找不到输入数据集：{path}")
        # utf-8-sig 同时允许普通 UTF-8 与带 BOM 的 JSONL。
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{path} 第 {line_number} 行为空。")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path} 第 {line_number} 行不是合法 JSON：{error.msg}。"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象。")

                # 合并不改变 sample_id；重复 ID 通常表示同一实验被加入两次。
                sample_id = record.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"{path} 第 {line_number} 行缺少有效 sample_id。")
                if sample_id in seen_ids:
                    raise ValueError(f"发现重复 sample_id：{sample_id}")
                seen_ids.add(sample_id)

                # 同一输出中的 schema_version 必须一致，避免把不同字段语义混用。
                version = record.get("schema_version")
                if type(version) is not int:
                    raise ValueError(f"{path} 第 {line_number} 行缺少整数 schema_version。")
                if expected_version is None:
                    expected_version = version
                elif version != expected_version:
                    raise ValueError(
                        f"schema_version 不一致：期望 {expected_version}，"
                        f"{path} 第 {line_number} 行为 {version}。"
                    )
                # schema 3 把弯曲刚度 EI 放在每株植物上；合并时保留原值，
                # 但拒绝缺失、布尔值、非有限数和非正数，避免产生不可训练的文件。
                if version == 3:
                    plants = record.get("plants")
                    if not isinstance(plants, list) or not plants:
                        raise ValueError(f"{path} 第 {line_number} 行需要非空 plants 列表。")
                    for plant_number, plant in enumerate(plants, start=1):
                        stiffness = plant.get("stiffness") if isinstance(plant, dict) else None
                        if (
                            isinstance(stiffness, bool)
                            or not isinstance(stiffness, (int, float))
                            or not math.isfinite(stiffness)
                            or stiffness <= 0
                        ):
                            raise ValueError(
                                f"{path} 第 {line_number} 行第 {plant_number} 株的 stiffness "
                                "必须是正的有限数，单位为 N·m²。"
                            )
                records.append(record)
    if not records:
        raise ValueError("所有输入数据集均为空。")
    return records


def main(argv: list[str] | None = None) -> int:
    """完成路径检查、读取校验和原子写出，成功返回 0。"""

    args = parse_args(argv)
    input_paths = tuple(path.expanduser().resolve() for path in (args.inputs or DEFAULT_INPUT_PATHS))
    output_path = args.output.expanduser().resolve()
    if output_path in input_paths:
        raise ValueError("输出路径不能与任何输入路径相同。")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在；如需替换请加 --overwrite：{output_path}")

    # 先完整验证所有输入，再开始写出。临时文件替换避免中途中断留下半份数据。
    records = load_records(input_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        raise FileExistsError(f"发现已有临时文件，请先核对其内容：{temporary_path}")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"已合并 {len(input_paths)} 份数据集，共 {len(records)} 条：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
