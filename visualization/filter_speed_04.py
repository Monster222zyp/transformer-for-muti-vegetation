"""只保留 CSV 中 flow_speed 等于 0.4 的行。"""

import argparse
import csv
import os
import tempfile
from pathlib import Path


# 默认处理与本脚本位于同一目录的 summarized_data.csv。
DEFAULT_CSV = Path(__file__).with_name("summarized_data.csv")


def filter_csv(csv_path: Path) -> tuple[int, int]:
    """原地过滤 CSV，并返回（保留行数，删除行数）。"""
    # 临时文件放在原 CSV 所在目录，确保最终替换操作可靠。
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "flow_speed" not in reader.fieldnames:
            raise ValueError("CSV 中没有 flow_speed 列")

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=csv_path.parent,
            delete=False,
            suffix=".csv",
        ) as temp:
            temp_path = Path(temp.name)
            writer = csv.DictWriter(temp, fieldnames=reader.fieldnames)
            writer.writeheader()

            kept = removed = 0
            for row in reader:
                # 转成浮点数后比较，可同时接受 0.4、0.40 等写法。
                if float(row["flow_speed"]) == 0.4:
                    writer.writerow(row)
                    kept += 1
                else:
                    removed += 1

    # 只有完整写完临时文件后才替换原文件，避免处理中断损坏数据。
    os.replace(temp_path, csv_path)
    return kept, removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只保留 flow_speed 为 0.4 的 CSV 行")
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_CSV, help="待处理的 CSV 路径")
    args = parser.parse_args()

    kept_count, removed_count = filter_csv(args.csv_path)
    print(f"完成：保留 {kept_count} 行，删除 {removed_count} 行。")
