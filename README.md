# 多水草 HydroTransformer 项目

本项目研究多个水草在平面六边形阵列中的水动力耦合：给定水草的空间排布、整体旋转角度和相对流速，使用实验数据训练 `HydroTransformer`，预测整个构型沿主阻力方向的总阻力。

这份 README 是仓库入口。维护项目时，建议先阅读 [维护者指南](文档/维护者指南/README.md)，再进入对应子目录。

## 当前仓库状态摘要

- 当前构型为 17 个；最新原始实验位于 `data/zyp/model1`～`model17`。
- zyp 实验矩阵：17 个构型 × 24 个角度 × 3 档流速 × 2 轮，共 2448 次实验，每次各有一份 sensor 和 gather。
- 最新处理结果已移动到 `datasets/multi_stiffness/`，使用 `[3.000,4.990)` s 和 15% 重复性筛选；处理方法见 [批次说明](文档/工作记录/zyp数据处理20261021/处理说明.md)。
- 2026-10-22 核对时，根目录 `dataset.jsonl` 为 1197 条最新全角度记录，是 `model/configs/base.yaml` 当前读取的数据；最新六角子集为 `datasets/multi_stiffness/dataset_60deg.jsonl`，共 287 条。
- `Experiment/output/dataset.jsonl` 是 332 条历史基线；`dataset_initial.jsonl` 当前不存在。历史 full17 重建默认写入 `Experiment/output/rebuilt_all/`。
- `model/artifacts/` 保存各次训练产物，使用的数据和划分方式应以对应运行记录为准。

“正式完整基线”和“当前训练默认数据”不是同一个概念。更换数据或复现实验前，务必阅读 [数据字典与文件物理意义](文档/维护者指南/数据字典与文件物理意义.md)。

## 五分钟理解数据流

```text
Experiment/input.csv                         17 个计划构型
        │
data/model_N/sensor_A_angB.csv              原始六轴传感器时间序列
        │  Experiment/filter_sensor_data.py
filtered_data/model_N/*_filtered.csv         截取并滤波后的时间序列
        │  Experiment/summarize_sensor_data.py
Experiment/output/summarized_data.csv        便于人工审计的条件级表格
Experiment/output/dataset.jsonl              模型直接读取的逐条件、逐株数据
        │  python -m model.train
model/artifacts/                              checkpoint、预测、指标和图片
        │
visualization/                               解释性绘图和结果切图
```

上述是历史数据链。最新 zyp 批次通过 `Experiment/process_zyp_data.py` 直接读取原始 sensor/gather，
输出 `datasets/zyp_20261021/dataset.jsonl` 与 `repeatability.csv`，处理参数和运动诊断另存工作记录。
`gather` 用于匀速检查，不是模型输入。单独运行 `Experiment/rotate_force.py` 不属于主链；汇总脚本会在内部完成力坐标旋转。

## 常用命令

以下命令均从仓库根目录执行。推荐 Python 3.11。

```powershell
# 安装模型侧依赖。
py -3.11 -m pip install -r model/requirements.txt

# 安装实验滤波脚本额外需要、但当前 requirements.txt 尚未声明的 SciPy。
py -3.11 -m pip install scipy

# 处理最新 zyp 全 17 组双轮实验；已确认的短记录整对排除并标记高误差。
& 'C:\Users\Mayn\miniconda3\envs\Transformer\python.exe' Experiment/process_zyp_data.py --short-sensor-policy drop-pair

# 复现历史批次的滤波和汇总数据。
py -3.11 Experiment/filter_sensor_data.py
py -3.11 Experiment/summarize_sensor_data.py

# 先做小样本连通性检查，再开始正式交叉验证。
py -3.11 -m model.train --mode overfit `
  --data Experiment/output/dataset.jsonl `
  --artifact-dir model/artifacts/smoke

py -3.11 -m model.train --mode cv `
  --data Experiment/output/dataset.jsonl `
  --artifact-dir model/artifacts/run_001

# 或先在 model/configs/base.yaml 设置 mode: predefined 和三份 JSONL 路径，
# 再运行下列命令；完整配置示例见 model/README.md。
py -3.11 -m model.train --artifact-dir model/artifacts/predefined_run_001

# 运行完整自动化测试。
py -3.11 -m pytest model/tests -q
```

建议每次训练都确认 `base.yaml` 中的 `mode` 和对应数据路径，并使用新的 `--artifact-dir`。`cv`/`overfit` 可用 `--data` 临时覆盖单个数据文件；`predefined` 则使用三条独立路径，不能与 `--data` 混用。不带恢复参数的训练会清理目标产物目录；复用默认目录可能覆盖现有实验结果。

## 目录导航

| 路径 | 维护者应如何理解 |
|---|---|
| `Experiment/` | 实验构型、运动轨迹、sensor 过滤、条件级汇总；见 [Experiment 目录说明](Experiment/README.md) |
| `data/` | 本地原始 sensor/gather 数据；体积较大且被 Git 忽略 |
| `filtered_data/` | sensor 时间序列的派生滤波结果；可由原始数据重建且被 Git 忽略 |
| `model/` | 数据加载、HydroTransformer、训练、评估与测试；见 [模型运行说明](model/README.md) |
| `model/artifacts/` | 当前训练运行的 checkpoint、预测、指标和图；被 Git 忽略 |
| `visualization/` | 逐株 coefficient 图、结果切图及临时数据筛选工具；见 [可视化目录说明](visualization/README.md) |
| `output/` | 运动控制轨迹与编号副本，不是模型训练输出 |
| `文档/维护者指南/` | 面向接手者的项目级说明、数据契约和维护规范 |

## 维护者文档

- [项目与物理背景](文档/维护者指南/项目与物理背景.md)
- [数据字典与文件物理意义](文档/维护者指南/数据字典与文件物理意义.md)
- [脚本与端到端工作流](文档/维护者指南/脚本与端到端工作流.md)
- [训练产物与结果解释](文档/维护者指南/训练产物与结果解释.md)
- [维护规范与已知问题](文档/维护者指南/维护规范与已知问题.md)

现有 [实验方案与开发者快速上手](Experiment/实验方案与开发者快速上手.md) 提供实验侧的深入说明；[模型文件夹脚本详细说明](model/模型文件夹脚本详细说明.md) 面向需要阅读实现的开发者。后者包含大量代码级细节，不是首次接手项目的必读材料。

## 科学解释边界

模型以总阻力 `FX_0` 为监督信号。逐株 `latent_coefficient` 是模型为了重建总阻力而学习的潜在系数，不是传感器直接测得的单株阻力，也不能在没有逐株实验或 CFD 标签时解释为确定的植株间因果作用。

传感器型号、量程、精度、全部六轴的真实单位，水草材料与几何参数，以及 `ang0` 的装置物理参考方向目前没有完整记录。相关量在文档中会明确标注“待确认”，不应凭字段名称自行补全。
