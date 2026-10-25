# Experiment 目录说明

本目录负责实验计划、运动轨迹、原始 sensor 数据处理和条件级数据汇总。正式训练链为：

```text
../data/model_N/sensor_A_angB.csv
    → filter_sensor_data.py
../filtered_data/model_N/*_filtered.csv
    → summarize_sensor_data.py
output/rebuilt_all/summarized_data.csv + output/rebuilt_all/所有.jsonl
```

## 文件导航

| 文件或目录 | 用途 |
|---|---|
| `input.csv` | 建集权威输入；第 N 行对应 model_N，每行 37 个非负整数类别；0 为空位置 |
| `input copy.csv` | 历史二进制构型副本，仅用于核对占用位置，不能代替材料类别输入 |
| `stiffness_config.yaml` | 类别到实测抗弯刚度 EI（N·m²）的映射，以及流速、重复性筛选策略 |
| `generator.py` | 生成或追加指定植株数的构型，排除旋转等价重复 |
| `build_groups.py` | 按固定植株数配方批量重建 17 个构型 |
| `visualize_hex_lattice.py` | 绘制构型总览及六种旋转图 |
| `motion_profile.py` | 生成四档 6 秒拖曳控制轨迹到根目录 `output/` |
| `adapter.py` | 校验并编号运动轨迹，生成设备映射 |
| `filter_sensor_data.py` | 正式 sensor 时间序列截取、缺失处理和滤波 |
| `summarize_sensor_data.py` | 正式条件级截尾平均、力旋转、CSV/JSONL 汇总 |
| `process_zyp_data.py` | zyp 全 17 组、24 角度双轮原始数据处理，同时用 gather 检查匀速 |
| `rotate_force.py` | 历史独立序列旋转工具；正式汇总只复用其变换功能 |
| `gather_data.py` | 历史 gather 运动数据诊断，不进入训练链 |
| `hex_lattice.svg` | 当前构型总览生成物 |
| `rotated_plots/` | 各构型六个角度的 SVG 生成物 |
| `output/` | 历史实验产物；新的完整重建默认写入其下 `rebuilt_all/` |

## 最常用命令

```powershell
# 从仓库根目录执行。
py -3.11 Experiment/filter_sensor_data.py
py -3.11 Experiment/summarize_sensor_data.py
py -3.11 Experiment/visualize_hex_lattice.py --show-ids
```

重建构型与生成运动轨迹不是日常训练前置步骤：

```powershell
# 会重建并覆盖构型，执行前必须确认。
py -3.11 Experiment/build_groups.py --overwrite --seed 20260809

py -3.11 Experiment/motion_profile.py
py -3.11 Experiment/adapter.py
```

## 必须保持的约束

- `input.csv` 第 N 行对应 model N；历史目录为 `data/model_N/`，zyp 目录为 `data/zyp/modelN/`。
- 原始 sensor 文件必须是无表头六列 `TX,TY,TZ,FX,FY,FZ`。
- `sensor_1`～`sensor_4` 分别表示 0.1～0.4 m/s，不是传感器编号。
- 正式汇总结果已经旋转 FX/FY，训练前不要重复旋转。
- `gather` 数据不属于正式模型输入。
- `--strict` 默认检查完整 17 组源工况，允许旧四个已知缺失；不要求每份旧 CSV 都有 1600 行。
- 没有显式 `--overwrite` 时拒绝覆盖任一输出；默认重建目录与现有正式 JSONL 分离。

## 完整重建、材料类别与 schema 3

抗弯刚度 `EI` 表示植株抵抗弯曲的能力，单位为 N·m²。`input.csv` 中的正整数
仅是材料类别编号：当前 YAML 中 `1 → 8.000e-7`、`2 → 5.907e-6`、`3 → 2.080e-5 N·m²`。
同一个编号对应同一份实测 EI，并不意味着类别编号本身就是物理刚度。可以
添加任意正整数类别，例如 `7: 3.6`；未配置类别、负数类别、非正或非有限 EI
都会在读取实验文件之前报错。`0` 表示没有植株，不需要 EI 映射。

CLI 默认 `--source-mode full17`，完整读取旧 model 1～14 的六角单轮文件和新
model 15～17 的 15° 两轮文件。默认输出如下，均位于 `output/rebuilt_all/`：

| 文件 | 当前默认数量 | 内容 |
|---|---:|---|
| `summarized_data.csv` | 460 行 | 保持原列名的条件级测量汇总 |
| `所有.jsonl` | 460 条 | 完整 schema 3 数据 |
| `所有60度的.jsonl` | 300 条 | 0、60、120、180、240、300° 的子集 |
| `新的非60度的.jsonl` | 160 条 | 新三组其余角度的子集 |
| `repeatability_new15_17.csv` | 216 行 | 两轮重复性明细，含所有被删除整对和 `retained` 判断 |

```powershell
# 默认完整重建；不会覆盖既有正式 JSONL。
py -3.11 Experiment/summarize_sensor_data.py

# 可以使用项目外的类别表以及自定义 EI 配置。
py -3.11 Experiment/summarize_sensor_data.py --input-csv 'D:\experiment\input.csv' --config 'D:\experiment\stiffness.yaml'

# 旧 14 组历史兼容模式默认保留四档流速，共 332 条，仍输出带 EI 的 schema 3。
py -3.11 Experiment/summarize_sensor_data.py --source-mode legacy14 --output-path Experiment/output/rebuilt_old14/summary.csv --dataset-path Experiment/output/rebuilt_old14/dataset.jsonl
```

`--config` 默认读取脚本旁的 `stiffness_config.yaml`。该文件的
`stiffness_by_code` 配置 EI；`excluded_flow_speeds` 默认排除 `0.1 m/s`；
`repeatability_threshold_percent` 默认是 `15.0`，表示百分数。`--trim-fraction`
默认 `0.10`，所有路径和参数均可通过命令行覆盖；`--six-angle-path`、
`--non-six-angle-path`、`--repeatability-path` 可以分别指定三个额外产物的路径。
这些路径各有独立默认值，调整 `--dataset-path` 时如需整体迁移输出目录，
应同时指定其它输出路径。确认需要替换自己生成的输出后，再显式添加 `--overwrite`。

每份 CSV 的 `TX,TY,TZ,FX,FY,FZ` 六列分别排序，删除两端各
`floor(行数 × 0.10)` 个值后求均值；短文件也按真实行数截尾。只旋转均值中的
`FX/FY`，使用完整实验角度，`TX/TY/TZ/FZ` 不旋转。新批次必须同时具有两轮：

```text
对称差率（%）= 100 × |round1_FX_0 − round2_FX_0|
                    / ((|round1_FX_0| + |round2_FX_0|) / 2)
```

两轮同时为零时差率是 0%；超过 15% 整对删除，等于 15% 保留。合格两轮的
六个已经截尾、旋转的测量字段分别取平均，不合并原始时间序列。旧 14 组只
保留原单轮值。当前删除的五个 `(model, angle, speed)` 工况为
`(15,0,0.2)`、`(15,165,0.2)`、`(16,0,0.2)`、`(16,30,0.2)`、`(17,0,0.2)`。

schema 3 的 `vegetation_layout` 仍为 37 位二进制字符串，`plants` 中每株新增
正有限的 `stiffness`。类别先与植株一起按 `generator.get_rotated_groups` 移到
60° 网格锚点，再将坐标顺时针旋转 `angle % 60`。例如 75° 使用 60° 的网格
排布和 `grid_index`，再把坐标旋转 15°；EI 始终跟随原植株，`plants.angle`
保存完整的 75°。外部类别表仍使用脚本目录的 `generator.py`。最终 ID 为
`model_001_angle_060_flow_0.2`，不保留轮次后缀，并按 model、角度、流速排序。

Python 公共函数 `summarize_sensor_tree` 保留原七个位置参数；旧调用继续默认
`legacy14`。完整重建需要关键词 `source_mode="full17"`，可用 `config_path`
指定 YAML，用 `excluded_flow_speeds` 和 `repeatability_threshold_percent`
覆盖筛选策略。`six_angle_file`、`non_six_angle_file`、`repeatability_file` 省略时
写入 `dataset_file` 同目录。`load_rotated_layouts` 仍返回二进制排布；需要完整
类别时使用 `load_rotated_classifications`，避免提前丢失每株材料身份。
`build_dataset_records` 的旧调用若只提供纯 01 排布，则明确把 1 解释为材料
类别 1，通过配置映射获得 EI；不会读取当前默认构型表与调用者的排布混合。
多类材料必须显式提供 `rotated_classifications` 和 `stiffness_by_code`，主 CLI
已经始终这样调用。

## zyp 批次：17 组全部重新采集（2026-10-21）

`data/zyp/model1`～`model17` 每组都有 24 个角度、3 档流速和 2 轮实验，
合计 2448 份 sensor 和同数 gather。使用专门入口，避免历史 full17 模式中
model 1～14 只允许六角单轮的限制：

```powershell
& 'C:\Users\Mayn\miniconda3\envs\Transformer\python.exe' Experiment/process_zyp_data.py --short-sensor-policy drop-pair
```

用户已确认 sensor=1000 Hz、gather=100 Hz，二者第 0 行的时刻对齐。默认截取
`[3.000, 4.990)` s：sensor 行索引 `[3000,4990)`，共 1990 点，最后为 4.989 s；
gather 索引 `[300,499)`，共 199 点，最后为 4.98 s。先截取再执行 2 Hz、5 阶
Butterworth 零相位低通，对六列分别两端各删 10% 后平均，再旋转 FX/FY。
全 17 组均按 YAML 的 15% FX 对称差率筛选，合格两轮的六个测量值分别平均。

两份结果写到 `datasets/zyp_20261021/dataset.jsonl` 和 `repeatability.csv`。
构型和 EI 使用本目录的 `input.csv`、`stiffness_config.yaml`；JSONL 为 schema 3。
gather 的第 2 列为 X 位移，原值乘 `1e-6` 换算为 m，以位移拟合速度及差分
速度波动检查匀速。运动诊断不替换 0.2/0.3/0.4 m/s 标签，也不额外剔除工况。
逐轮指标、文件 SHA-256 和参数摘要另存于 `文档/工作记录/zyp数据处理20261021/`。
参数均支持 CLI 覆盖，运行 `--help` 查看；默认拒绝覆盖已有结果。
本批次有 4 份 sensor 过短，用户已确认整对排除。上面的 `drop-pair` 会在 CSV 中
将它们标为 `comparison_status=incomplete_sensor_window`，误差按用户要求记为
`10000%`，并注明这是无效记录标记、不是实测差率；缺失力值和绝对差留空。
省略此参数时，脚本默认遇到过短记录即中止，避免处理其他批次时无意删除工况。

## 历史第 15～17 组批次的滤波记录

本节是旧批次的复现记录；其中 `[2.4,5.6)` s 不适用于上面的 zyp 批次。

原始文件位于项目外的 `raw_data_17/model15`～`model17`。这三组的 sensor 文件名为
`sensor_A_roundR_angB.csv`：`A` 为 2、3、4，`R` 为 1、2，`B` 为 0～345°，
每隔 15° 一档。每组共有 144 份六列 sensor 记录；同目录的 144 份 `gather`
记录是运动数据，不进入此滤波流程。分类 `input.csv` 是后续汇总时的几何及
材料来源；本步时间序列滤波不读取它，历史 `input copy.csv` 仅可核对占用位置。

这批数据以 1000 Hz 采样。为保持与旧批次相同的记录相对时间段 `[2.4, 5.6)` 秒，
截取行索引 `[2400, 5600)`，每份成功输出应有 3200 行。低通保持 2 Hz 截止频率、
5 阶 Butterworth、六列同时处理；采样率从旧批次的 500 Hz 改为 1000 Hz。
例如处理第 15 组：

```powershell
& 'C:\Users\Mayn\miniconda3\envs\Transformer\python.exe' Experiment/filter_sensor_data.py `
  --input-path 'C:\Users\Mayn\Desktop\myfiles\Transformer project\raw_data_17\model15' `
  --output-path 'filtered_data\model_15' `
  --mode lowpass --fs 1000 --cutoff 2 --order 5 --columns all `
  --trim-head 2400 --trim-tail 5600 --no-overwrite
```

第 16、17 组分别将输入 `model15`、输出 `model_15` 改为相应编号。输出文件名保留
`round` 和 15° 角度，因此不会与两轮采集或前 14 组的旧结果混淆。
`summarize_sensor_data.py` 的 full17 模式已按上面的截尾、完整角力旋转、
重复性整对筛选、两轮平均及逐株 EI 规则处理该批次。

更多说明见：

- [项目与物理背景](../文档/维护者指南/项目与物理背景.md)
- [数据字典与文件物理意义](../文档/维护者指南/数据字典与文件物理意义.md)
- [脚本与端到端工作流](../文档/维护者指南/脚本与端到端工作流.md)
- [实验方案与开发者快速上手](实验方案与开发者快速上手.md)
