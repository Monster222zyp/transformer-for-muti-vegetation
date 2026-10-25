"""原地过滤 dataset.jsonl，只保留 flow_speed 等于 0.4 的完整样本。"""

import argparse
import json
import os
import tempfile
from pathlib import Path


# 默认处理脚本同目录的 dataset.jsonl；命令行可传入其他路径。
DEFAULT_JSONL = Path(__file__).with_name("dataset.jsonl")
TARGET_FLOW_SPEED = 0.4


def filter_jsonl(jsonl_path: Path) -> tuple[int, int]:
    """jsonl_path 为待原地过滤的文件；返回（保留样本数，删除样本数）。"""
    temp_path = None
    try:
        # 在同目录写临时文件；完整处理成功后才替换原文件。
        with jsonl_path.open("r", encoding="utf-8-sig") as source:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=jsonl_path.parent,
                delete=False, suffix=".jsonl",
            ) as temp:
                temp_path = Path(temp.name)
                kept = removed = 0
                for line in source:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    # 保留完整原始记录，包括嵌套的 plants 和 measurements。
                    if float(sample["flow_speed"]) == TARGET_FLOW_SPEED:
                        temp.write(line.rstrip("\r\n") + "\n")
                        kept += 1
                    else:
                        removed += 1
        os.replace(temp_path, jsonl_path)
        return kept, removed
    finally:
        # 解析失败时仅删除临时文件，原始数据保持不变。
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="原地过滤 JSONL，只保留 flow_speed 为 0.4 的样本")
    parser.add_argument("jsonl_path", nargs="?", type=Path, default=DEFAULT_JSONL, help="待原地过滤的 dataset.jsonl 路径")
    args = parser.parse_args()

    kept_count, removed_count = filter_jsonl(args.jsonl_path)
    print(f"完成：保留 {kept_count} 行，删除 {removed_count} 行。")
