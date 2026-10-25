"""读取已保存的预测 CSV，重新绘制散点图，无需加载权重或重新训练。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


# 所有可调整默认值集中在顶部；相对输入、输出路径按命令执行时的工作目录解析。
PROJECT_ROOT_PARENT_INDEX = 3  # 本脚本位于项目根目录下的“文档/工作记录/本次任务”中。
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[PROJECT_ROOT_PARENT_INDEX]
DEFAULT_INPUT_CSV = DEFAULT_PROJECT_ROOT / "model" / "artifacts" / "cv_predictions.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_CSV.parent
DEFAULT_PREFIX = "cv"
DEFAULT_TITLE = "Cross-validation held-out test drag parity"
DEFAULT_AXIS_MODE = "independent"
AXIS_MODE_CHOICES = ("independent", "shared")
CSV_ENCODING = "utf-8-sig"  # 同时兼容普通 UTF-8 和带 BOM 的 CSV。

# 将项目根目录加入导入路径，允许从任意工作目录直接运行这个独立维护脚本。
sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))
from model.training.visualization import write_drag_parity_plots  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数；返回包含输入路径、输出目录和绘图设置的 Namespace。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="已有预测 CSV。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="两张 PNG 的输出目录。")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="输出文件名前缀。")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="图表标题前缀。")
    parser.add_argument(
        "--axis-mode", choices=AXIS_MODE_CHOICES, default=DEFAULT_AXIS_MODE,
        help="independent 各图独立范围；shared 两图共用范围。",
    )
    return parser.parse_args()


def main() -> None:
    """读取所有预测记录，调用项目绘图函数，并打印生成文件的绝对路径。"""

    args = parse_args()
    # DictReader 把每行转换为字段字典；绘图函数负责数值转换及有限值校验。
    with args.input_csv.open("r", encoding=CSV_ENCODING, newline="") as handle:
        predictions = list(csv.DictReader(handle))
    paths = write_drag_parity_plots(
        predictions, args.output_dir, prefix=args.prefix,
        title=args.title, axis_mode=args.axis_mode,
    )
    # 只打印两张输出图的位置，便于在终端中定位，不输出整份预测数据。
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
