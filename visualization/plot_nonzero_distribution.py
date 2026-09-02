"""统计 CSV 每行的非零数值个数，并输出柱状图。"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


# 输入、输出路径直接写在这里；需要更换文件时修改这两个变量即可。
INPUT_CSV = Path(
    r"C:\Users\Mayn\Desktop\myfiles\Transformer project\transformer-for-muti-vegetation\Experiment\input.csv"
)
OUTPUT_IMAGE = Path(__file__).with_name("nonzero_distribution.png")


# 读取每一行，并统计该行中不等于 0 的数值有多少个。
with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file)
    nonzero_counts = [sum(float(value) != 0 for value in row if value.strip()) for row in reader]

# Counter 将“非零数值个数相同的行”汇总起来。
distribution = Counter(nonzero_counts)
x = sorted(distribution)
y = [distribution[count] for count in x]

# 绘制并保存柱状图。
plt.bar(x, y)
plt.xlabel("Number of non-zero values per row")
plt.ylabel("Number of rows")
plt.title("Distribution of non-zero values")
plt.xticks(x)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=200)
plt.close()
