"""原地过滤 dataset.jsonl，删除 flow_speed 等于 0.1 的完整样本。"""

import argparse
import json
import os
import tempfile
from pathlib import Path


# 默认处理脚本同目录的 dataset.jsonl；也可以通过命令行传入其他 JSONL 路径。
DEFAULT_JSONL = Path(__file__).with_name("dataset.jsonl")
REMOVED_FLOW_SPEED = 0.1


def remove_speed_01(jsonl_path: Path) -> tuple[int, int]:
    """删除指定文件中的 0.1 m/s 样本，并返回（保留数量，删除数量）。"""

    temporary_path = None
    try:
        # 临时文件与原文件位于同一目录；处理全部成功后才替换原文件。
        with jsonl_path.open("r", encoding="utf-8-sig") as source:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=jsonl_path.parent,
                delete=False,
                suffix=".jsonl",
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                kept_count = 0
                removed_count = 0

                for line in source:
                    # 跳过空行，避免把空行写回结果文件。
                    if not line.strip():
                        continue

                    sample = json.loads(line)
                    if float(sample["flow_speed"]) == REMOVED_FLOW_SPEED:
                        removed_count += 1
                        continue

                    # 直接保存原始 JSON 行，完整保留 plants、measurements 等字段。
                    temporary_file.write(line.rstrip("\r\n") + "\n")
                    kept_count += 1

        os.replace(temporary_path, jsonl_path)
        return kept_count, removed_count
    finally:
        # JSON 解析或写入失败时删除临时文件，原始文件不会被替换。
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="原地过滤 JSONL，删除 flow_speed 为 0.1 的样本"
    )
    parser.add_argument(
        "jsonl_path",
        nargs="?",
        type=Path,
        default=DEFAULT_JSONL,
        help="待原地过滤的 dataset.jsonl 路径",
    )
    arguments = parser.parse_args()

    kept, removed = remove_speed_01(arguments.jsonl_path)
    print(f"完成：保留 {kept} 行，删除 {removed} 行。")
