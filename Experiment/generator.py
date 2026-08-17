"""生成 37 位二进制六边形序列，并排除 60 度旋转后的重复序列。"""

# argparse 用于把脚本变成可从命令行传参的工具。
import argparse
# csv 用于读取和写入无表头的 CSV 序列文件。
import csv
# random 用于随机选择应设为 1 的点位。
import random
# Path 用于可靠地处理 Windows 和其他系统上的文件路径。
from pathlib import Path
# Sequence 用于标注可选的命令行参数列表类型。
from typing import Sequence


# 每个六边形序列固定包含 37 个位置。
SEQUENCE_LENGTH = 37

# 构型文件属于 Experiment，因此默认输入和输出都指向脚本旁边的 input.csv。
DEFAULT_INPUT_CSV_PATH = Path(__file__).resolve().with_name("input.csv")

# 最终数据要求为二进制序列，因此只允许 0 和 1 两种数值。
BINARY_VALUES = (0, 1)

# 第 1 和第 37 个点位必须被选中；Python 下标从 0 开始，所以对应 0 和 36。
FORCE_NONZERO_INDEXES = (0, 36)

# 当连续生成大量旋转重复的候选序列时，使用此上限防止程序无限循环。
MAX_FAILED_ATTEMPTS = 100_000

# 此映射表示顺时针旋转 60 度后，每个旧位置的值应移动到哪个新位置。
# 映射覆盖了全部 37 个位置，中心点 18 旋转后仍然是自身。
ROTATION_MAPPING = {
    # 第 1 行的 4 个位置。
    0: 3, 1: 8, 2: 14, 3: 21,
    # 第 2 行的 5 个位置。
    4: 2, 5: 7, 6: 13, 7: 20, 8: 27,
    # 第 3 行的 6 个位置。
    9: 1, 10: 6, 11: 12, 12: 19, 13: 26, 14: 32,
    # 第 4 行的 7 个位置，其中 18 是中心点。
    15: 0, 16: 5, 17: 11, 18: 18, 19: 25, 20: 31, 21: 36,
    # 第 5 行的 6 个位置。
    22: 4, 23: 10, 24: 17, 25: 24, 26: 30, 27: 35,
    # 第 6 行的 5 个位置。
    28: 9, 29: 16, 30: 23, 31: 29, 32: 34,
    # 第 7 行的 4 个位置。
    33: 15, 34: 22, 35: 28, 36: 33,
}


def generate_random_group(point_count: int) -> list[int]:
    """生成一行包含 point_count 个 1 的 37 位二进制序列。"""
    # 将强制点位转为集合，方便后续快速判断一个位置是否允许随机抽取。
    forced_indexes = set(FORCE_NONZERO_INDEXES)

    # 强制点位数量是本函数可接受的最小根数。
    minimum_count = len(forced_indexes)

    # 根数不足时无法满足“两个强制点均为 1”的要求，因此立即给出明确错误。
    if point_count < minimum_count:
        raise ValueError(f"point_count 必须至少为 {minimum_count}。")

    # 创建全 0 的 37 位初始序列。
    group = [0] * SEQUENCE_LENGTH

    # 将第 1 和第 37 个点位固定为 1。
    for index in forced_indexes:
        group[index] = 1

    # 计算除强制点外还需要随机选中的位置数量。
    remaining_count = point_count - minimum_count

    # 生成所有可随机选择的位置，排除两个已经固定为 1 的点位。
    optional_indexes = [
        index for index in range(SEQUENCE_LENGTH) if index not in forced_indexes
    ]

    # 从可选位置中无重复地随机抽取所需数量。
    selected_indexes = random.sample(optional_indexes, remaining_count)

    # 将被抽中的位置设为 1；其他位置继续保持 0。
    for index in selected_indexes:
        group[index] = 1

    # 返回这一行完整的二进制序列。
    return group


def rotate_once(group: list[int]) -> list[int]:
    """返回 group 顺时针旋转 60 度后的新序列。"""
    # 先创建全 0 的新序列，用来保存旋转后的值。
    rotated_group = [0] * SEQUENCE_LENGTH

    # 按映射逐项移动原序列中的数值。
    for source_index, target_index in ROTATION_MAPPING.items():
        rotated_group[target_index] = group[source_index]

    # 返回新列表，不修改调用者传入的原列表。
    return rotated_group


def get_rotated_groups(group: list[int]) -> list[list[int]]:
    """返回 group 旋转 0、60、120、180、240、300 度后的六个序列。"""
    # rotations 按角度从小到大保存六个独立列表，供可视化脚本和去重逻辑共同使用。
    rotations: list[list[int]] = []

    # current_group 从原始序列开始，每一轮循环结束后再顺时针旋转 60 度。
    current_group = group.copy()

    # 六边形旋转六次会回到原始朝向，因此恰好生成六个方向。
    for _ in range(6):
        # copy 确保返回的每个方向都是独立列表，不会被下一轮旋转修改。
        rotations.append(current_group.copy())

        # 计算下一个 60 度朝向，供下一轮循环使用。
        current_group = rotate_once(current_group)

    # 返回与旧版本相同的“六个列表组成的列表”，维持其他脚本的导入兼容性。
    return rotations


def get_rotation_keys(group: list[int]) -> set[tuple[int, ...]]:
    """返回 group 旋转 0 到 300 度后的六个可哈希序列键。"""
    # keys 保存六种旋转结果；集合可以自动合并具有旋转对称性的重复结果。
    keys: set[tuple[int, ...]] = set()

    # 调用公开的 get_rotated_groups，确保去重逻辑和其他调用者使用完全相同的旋转规则。
    for current_group in get_rotated_groups(group):
        # tuple 可放入 set，用于快速判断是否出现过相同排列。
        keys.add(tuple(current_group))

    # 返回全部旋转键。
    return keys


def load_groups_from_csv(csv_path: Path) -> list[list[int]]:
    """读取并严格验证无表头、每行 37 列的 0/1 CSV。"""
    # 文件不存在时由调用者决定是否允许从空文件开始；此函数只负责读取存在的文件。
    if not csv_path.exists():
        raise FileNotFoundError(f"输入 CSV 文件不存在：{csv_path}")

    # groups 保存每一行转换后的整数列表。
    groups: list[list[int]] = []

    # utf-8-sig 同时兼容带或不带 UTF-8 BOM 的 CSV 文件。
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        # 创建逐行读取的 CSV 读取器。
        reader = csv.reader(csv_file)

        # 从第 1 行开始遍历，便于错误信息指出具体行号。
        for row_index, row in enumerate(reader, start=1):
            # 忽略纯空行，避免手工编辑时的空白行影响结果。
            if not row or all(cell.strip() == "" for cell in row):
                continue

            # 每行必须有 37 列，分别对应六边形上的 37 个位置。
            if len(row) != SEQUENCE_LENGTH:
                raise ValueError(
                    f"输入 CSV 第 {row_index} 行应有 {SEQUENCE_LENGTH} 列，实际有 {len(row)} 列。"
                )

            # 将 CSV 文本转换为整数；出现小数或文字时会给出带行号的错误。
            try:
                group = [int(cell.strip()) for cell in row]
            except ValueError as error:
                raise ValueError(f"输入 CSV 第 {row_index} 行包含非整数内容。") from error

            # 本任务最终只允许 0 和 1，防止旧的 0/1/2/3 数据混入新文件。
            if any(value not in BINARY_VALUES for value in group):
                raise ValueError(f"输入 CSV 第 {row_index} 行包含不是 0/1 的值。")

            # 保存已经通过格式检查的一行。
            groups.append(group)

    # 返回全部有效行；空文件由首次生成参数 --allow-missing-input 以外的逻辑处理。
    return groups


def build_seen_rotation_keys(groups: list[list[int]]) -> set[tuple[int, ...]]:
    """将已有序列及其六种旋转结果转换为去重集合。"""
    # seen_keys 保存历史序列的所有旋转等价形式。
    seen_keys: set[tuple[int, ...]] = set()

    # 逐行加入每个已有序列的六种旋转键。
    for group in groups:
        seen_keys.update(get_rotation_keys(group))

    # 返回用于后续候选序列查重的集合。
    return seen_keys


def append_unique_groups(
    groups: list[list[int]], point_count: int, append_count: int
) -> None:
    """向 groups 原地追加 append_count 个不与已有序列旋转重复的新序列。"""
    # 先将所有既有序列的旋转形式建立为集合，避免新序列与旧序列等价。
    seen_keys = build_seen_rotation_keys(groups)

    # added_count 记录本次调用已经成功追加的行数。
    added_count = 0

    # failed_attempts 记录连续生成到旋转重复候选的次数。
    failed_attempts = 0

    # 达到本次要求的追加行数前持续生成候选。
    while added_count < append_count:
        # 生成符合当前总根数的随机二进制候选序列。
        candidate = generate_random_group(point_count)

        # 获取候选序列在六种朝向下的全部键。
        candidate_keys = get_rotation_keys(candidate)

        # 有任意朝向已出现时，说明候选与历史序列属于同一旋转等价类。
        if candidate_keys & seen_keys:
            # 记录失败次数。
            failed_attempts += 1

            # 连续失败过多时停止，防止可用组合耗尽后无限运行。
            if failed_attempts >= MAX_FAILED_ATTEMPTS:
                raise RuntimeError("连续生成旋转重复序列次数过多，无法继续生成。")

            # 当前候选不可用，开始下一轮随机生成。
            continue

        # 候选通过去重检查后，追加到输出列表。
        groups.append(candidate)

        # 同步更新历史旋转键，确保本次新增的多行之间也不会旋转重复。
        seen_keys.update(candidate_keys)

        # 成功追加一行后增加计数，并重置连续失败计数。
        added_count += 1
        failed_attempts = 0


def write_groups_to_csv(groups: list[list[int]], csv_path: Path) -> None:
    """将二进制序列写入无表头 CSV，并覆盖同名目标文件。"""
    # 当输出路径包含不存在的父目录时，先创建该目录。
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # utf-8-sig 让 Windows Excel 打开文件时可以正确识别 UTF-8 编码。
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        # 创建 CSV 写入器。
        writer = csv.writer(csv_file)

        # 逐行写入 37 个 0/1 值，不添加表头或额外编号。
        writer.writerows(groups)


def build_argument_parser() -> argparse.ArgumentParser:
    """创建 generator.py 的命令行参数解析器。"""
    # description 会显示在 python generator.py --help 的帮助信息中。
    parser = argparse.ArgumentParser(
        description="生成 37 位二进制六边形序列，并按 60 度旋转等价关系去重。"
    )

    # --point-count 是每一行中 1 的数量，即本任务中的总根数。
    parser.add_argument(
        "--point-count", type=int, required=True,
        help="每个新增序列中 1 的数量，范围为 2 到 37。",
    )

    # --append-count 允许一次追加多行，但默认值为 1，适合外部脚本逐次调用。
    parser.add_argument(
        "--append-count", type=int, default=1,
        help="本次追加的序列数量，默认值为 1。",
    )

    # --input 指定读取已有序列的来源文件。
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_CSV_PATH,
        help=f"已有序列的 CSV 路径，默认值为 {DEFAULT_INPUT_CSV_PATH}。",
    )

    # --output 指定保存合并结果的目标文件；默认与输入相同以实现原地追加。
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_INPUT_CSV_PATH,
        help=f"结果 CSV 路径，默认值为 {DEFAULT_INPUT_CSV_PATH}。",
    )

    # --allow-missing-input 仅用于首次调用，以允许从不存在的 input.csv 开始。
    parser.add_argument(
        "--allow-missing-input", action="store_true",
        help="输入文件不存在时从空列表开始生成；适合首次调用。",
    )

    # --seed 是可选随机种子，提供后可复现同一调用的随机选择结果。
    parser.add_argument(
        "--seed", type=int, default=None,
        help="可选随机种子；省略时每次生成结果不同。",
    )

    # 返回全部参数都已定义好的解析器。
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """解析命令行参数，读取、追加并保存二进制序列。"""
    # argv 为 None 时 argparse 自动读取真实命令行；测试时可以传入自定义参数列表。
    args = build_argument_parser().parse_args(argv)

    # 设置随机种子是可选行为，只有用户传入 --seed 才启用。
    if args.seed is not None:
        random.seed(args.seed)

    # 追加数量必须大于 0，否则本次调用没有实际意义。
    if args.append_count <= 0:
        raise ValueError("append_count 必须是正整数。")

    # 总根数至少要包含两个强制点，最多不能超过全部 37 个点。
    min_point_count = len(FORCE_NONZERO_INDEXES)
    if not min_point_count <= args.point_count <= SEQUENCE_LENGTH:
        raise ValueError(
            f"point_count 必须是 {min_point_count} 到 {SEQUENCE_LENGTH} 之间的整数。"
        )

    # 输入文件存在时读取其中的历史序列，确保本次生成结果追加在末尾。
    if args.input.exists():
        groups = load_groups_from_csv(args.input)
    # 首次调用且用户明确允许缺失输入文件时，从空列表开始。
    elif args.allow_missing_input:
        groups = []
    # 其他缺失情况默认报错，避免拼错路径时意外创建新文件。
    else:
        raise FileNotFoundError(
            f"输入 CSV 文件不存在：{args.input}。首次调用请添加 --allow-missing-input。"
        )

    # 记录读取数量，供命令行结果摘要使用。
    input_group_count = len(groups)

    # 将符合当前根数且不旋转重复的新序列追加到列表中。
    append_unique_groups(groups, args.point_count, args.append_count)

    # 将全部旧序列和新序列写入目标 CSV；输入与输出相同时即完成原地追加。
    write_groups_to_csv(groups, args.output)

    # 打印关键结果，方便命令行调用者检查执行是否符合预期。
    print(f"输入 CSV 读取 {input_group_count} 组序列。")
    print(f"本次新增 {args.append_count} 组序列，每组有 {args.point_count} 个 1。")
    print(f"最终输出 {len(groups)} 组序列。")
    print(f"CSV 文件已保存到：{args.output.resolve()}")


# 只有直接执行 generator.py 时才运行 main；被其他脚本导入时不会自动写文件。
if __name__ == "__main__":
    main()
