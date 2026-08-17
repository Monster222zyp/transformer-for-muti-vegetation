import csv
import json
import shutil
from pathlib import Path

# SCRIPT_DIR 是当前脚本所在的 Experiment 文件夹。
SCRIPT_DIR = Path(__file__).resolve().parent

# REPO_ROOT 是仓库根目录；这样从任意工作目录启动脚本都能找到同一个 output。
REPO_ROOT = SCRIPT_DIR.parent

# 运动轨迹、编号副本和映射文件统一保存在仓库根目录的 output 中。
INPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR = INPUT_DIR / "renamed"
MAPPING_PATH = OUTPUT_DIR / "mapping.json"

# motion_profile.py 现在把 x 列写为 mm，因此原来的 2 m 上限改为 2000 mm。
MAX_DISTANCE_MM = 2000.0

# 原有容差是 0.000001 m；换成 mm 后保持同一物理容差，即 0.001 mm。
FLOAT_TOLERANCE_MM = 0.001


def main() -> None:
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = MAPPING_PATH
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"],
        key=lambda p: p.name.lower(),
    )

    if not csv_files:
        raise RuntimeError(f"No CSV files found in input folder: {input_dir}")

    # x_sums 保存每个 CSV 的 x 列总和；当前单位是 mm。
    x_sums = {}
    max_file = None
    max_x_sum = None

    for src_file in csv_files:
        x_sum = 0.0
        with src_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x_sum += float(row["x"])

        x_sums[src_file.name] = x_sum
        if max_x_sum is None or x_sum > max_x_sum:
            max_x_sum = x_sum
            max_file = src_file.name

    # x 列已经是 mm，因此使用 2000 mm 上限和 mm 单位容差进行判定。
    if max_x_sum is not None and max_x_sum > MAX_DISTANCE_MM + FLOAT_TOLERANCE_MM:
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        raise RuntimeError(
            f"Max x_sum exceeded limit: {max_file} has {max_x_sum:.6f} mm, "
            f"limit is {MAX_DISTANCE_MM:.6f} mm + {FLOAT_TOLERANCE_MM:.6f} mm"
        )

    index_to_name = {}
    name_to_index = {}
    for idx, src_file in enumerate(csv_files, start=1):
        stem = src_file.stem
        dst_file = output_dir / f"{idx}.csv"
        shutil.copy2(src_file, dst_file)

        index_key = str(idx)
        index_to_name[index_key] = stem
        name_to_index[stem] = idx

    mapping_data = {
        "index_to_name": index_to_name,
        "name_to_index": name_to_index,
    }

    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(csv_files)} CSV files.")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Mapping file: {mapping_path.resolve()}")
    for filename, x_sum in x_sums.items():
        print(f"{filename}: x_sum_mm={x_sum:.6f}")
    print(f"Max x_sum file: {max_file}, x_sum_mm={max_x_sum:.6f}")


if __name__ == "__main__":
    main()
