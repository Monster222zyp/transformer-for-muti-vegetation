"""按用户指定的总根数配方，多次调用 generator.py 构建最终 input.csv。"""

# argparse 用于接收输出路径、覆盖许可和随机种子等命令行参数。
import argparse
# csv 用于独立验证最终 CSV 中的二进制序列。
import csv
# subprocess 用于真正以命令行方式反复调用 generator.py。
import subprocess
# sys 用于获取当前 Python 解释器路径，保证子进程使用同一运行环境。
import sys
# Path 用于处理输出文件和 generator.py 的绝对路径。
from pathlib import Path
# Sequence 用于标注可选命令行参数列表类型。
from typing import Sequence


# 这是用户给定的 17 组总根数；每个数对应一次 generator.py 调用和最终的一行数据。
ROOT_COUNT_RECIPE = (4, 4, 4, 5, 5, 6, 6, 8, 8, 8, 10, 10, 12, 12, 18, 20, 22)

# 每一个生成序列的固定长度，与 generator.py 中的六边形点位数一致。
SEQUENCE_LENGTH = 37

# 构型数据与本脚本同属 Experiment，默认输出固定为脚本旁边的 input.csv。
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().with_name("input.csv")


def build_argument_parser() -> argparse.ArgumentParser:
    """创建批量构建脚本的命令行参数解析器。"""
    # description 会显示在 python build_input_groups.py --help 中。
    parser = argparse.ArgumentParser(
        description="按预设的 17 个总根数逐次调用 generator.py，生成二进制 input.csv。"
    )

    # --output 允许把最终结果写到自定义文件；默认仍写入 input.csv。
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help=f"最终 CSV 路径，默认值为 {DEFAULT_OUTPUT_PATH}。",
    )

    # --overwrite 是显式保护开关，防止批量脚本无提示地删掉已有输入数据。
    parser.add_argument(
        "--overwrite", action="store_true",
        help="允许删除已有输出文件后，重新生成完整的 17 行数据。",
    )

    # --seed 为整个批次提供可复现的基础随机种子。
    parser.add_argument(
        "--seed", type=int, default=None,
        help="可选基础随机种子；第 N 次调用会使用基础值加 N。",
    )

    # 返回配置完成的参数解析器。
    return parser


def read_binary_groups(csv_path: Path) -> list[list[int]]:
    """读取最终 CSV，并验证每行都是 37 位的 0/1 序列。"""
    # groups 保存所有通过基础格式验证的数据行。
    groups: list[list[int]] = []

    # utf-8-sig 可兼容 generator.py 写出的 UTF-8 BOM。
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        # 创建 CSV 读取器。
        reader = csv.reader(csv_file)

        # 按行读取并进行格式检查。
        for row_index, row in enumerate(reader, start=1):
            # 忽略纯空行，避免手工编辑时的末尾空行影响结果。
            if not row or all(cell.strip() == "" for cell in row):
                continue

            # 每行必须完整包含 37 个六边形点位。
            if len(row) != SEQUENCE_LENGTH:
                raise ValueError(f"第 {row_index} 行不是 {SEQUENCE_LENGTH} 列。")

            # 将文本值转换成整数；异常内容会由 int 给出明确错误。
            group = [int(cell.strip()) for cell in row]

            # 最终文件只能包含 0 和 1。
            if any(value not in (0, 1) for value in group):
                raise ValueError(f"第 {row_index} 行包含非 0/1 数值。")

            # 保存通过检查的一行。
            groups.append(group)

    # 返回全部有效行。
    return groups


def validate_output(csv_path: Path) -> None:
    """验证最终 CSV 的 17 行数据是否严格匹配总根数配方。"""
    # 读取并完成每行长度与二进制取值的基础验证。
    groups = read_binary_groups(csv_path)

    # 行数必须与配方长度一致，确保生成器确实调用了 17 次。
    if len(groups) != len(ROOT_COUNT_RECIPE):
        raise ValueError(
            f"最终 CSV 应有 {len(ROOT_COUNT_RECIPE)} 行，实际有 {len(groups)} 行。"
        )

    # 逐行检查 1 的数量是否严格等于对应位置的总根数。
    for row_index, (group, expected_count) in enumerate(
        zip(groups, ROOT_COUNT_RECIPE), start=1
    ):
        # 二进制序列中的 1 的总和就是这一行的总根数。
        actual_count = sum(group)

        # 根数不一致时立即停止并指出具体行。
        if actual_count != expected_count:
            raise ValueError(
                f"第 {row_index} 行应有 {expected_count} 个 1，实际有 {actual_count} 个。"
            )


def main(argv: Sequence[str] | None = None) -> None:
    """逐次调用 generator.py，随后验证生成完成的二进制 CSV。"""
    # argv 为 None 时 argparse 读取真实命令行参数；测试时可传入列表。
    args = build_argument_parser().parse_args(argv)

    # 将输出路径转换为绝对路径，使每一次子进程调用都操作同一个文件。
    output_path = args.output.resolve()

    # 目标已经存在但没有 --overwrite 时，停止执行以保护已有数据。
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"输出文件已存在：{output_path}。确认重建请添加 --overwrite。"
        )

    # 用户明确允许覆盖时，先删除旧文件，确保最终结果恰好为 17 行。
    if output_path.exists():
        output_path.unlink()

    # generator.py 与本批量脚本位于同一目录，因此使用 __file__ 可以避免工作目录影响。
    generator_path = Path(__file__).with_name("generator.py")

    # 按配方顺序逐次调用生成器，每次 append-count 都固定为 1。
    for call_index, point_count in enumerate(ROOT_COUNT_RECIPE):
        # 组装本次真正要运行的 Python 命令。
        command = [
            sys.executable,
            str(generator_path),
            "--point-count",
            str(point_count),
            "--append-count",
            "1",
            "--input",
            str(output_path),
            "--output",
            str(output_path),
        ]

        # 第一次调用时没有 input.csv，所以明确允许生成器从空列表开始。
        if call_index == 0:
            command.append("--allow-missing-input")

        # 用户提供基础种子时，每一次调用使用不同种子，同时整个批次仍可复现。
        if args.seed is not None:
            command.extend(["--seed", str(args.seed + call_index)])

        # 先输出本次的根数，方便在终端观察 17 次独立调用。
        print(f"第 {call_index + 1} 次调用：总根数 = {point_count}")

        # check=True 确保任意一次生成失败时立即停止，不产生看似成功的不完整结果。
        subprocess.run(command, check=True)

    # 全部调用完成后，再独立验证行数、二进制取值和每一行的总根数。
    validate_output(output_path)

    # 打印批量生成完成后的结果摘要。
    print(f"生成完成：{output_path}")
    print(f"共 {len(ROOT_COUNT_RECIPE)} 行，根数配方：{' '.join(map(str, ROOT_COUNT_RECIPE))}")


# 只有直接执行本文件时才运行批量生成；被导入时不会改动任何 CSV。
if __name__ == "__main__":
    main()
