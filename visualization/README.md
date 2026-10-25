# visualization 目录说明

本目录存放模型解释性绘图、预测切图和临时 JSONL 筛选工具。它不负责训练，也不定义正式实验数据。

## 当前文件

| 文件或目录 | 用途 |
|---|---|
| `plot_plant_coefficients.py` | 从 checkpoint 和 JSONL 生成逐株点图与 Gaussian 影响图 |
| `config.yaml` | 上述脚本的 checkpoint、数据、输出与绘图配置；当前含本机绝对路径 |
| `plot_split_drag_comparison.py` | 把 prediction CSV 排序后按比例切成多张图 |
| `plot_drag_with_uncertainty.py` | 匹配双轮实验，绘制实验均值 ±1 个样本 SD 的高分辨率阻力对比图 |
| `plot_nonzero_distribution.py` | 统计构型中非零格点数；当前路径硬编码且导入即执行 |
| `filter_speed_04.py` | 原地保留 JSONL 中 0.4 m/s 样本 |
| `remove_speed_01.py` | 原地删除 JSONL 中 0.1 m/s 样本 |
| `dataset_0.4.jsonl` | 当前 84 条 0.4 m/s 模型就绪可视化子集 |
| `summarized_data_0.4.csv` | 当前 84 条 0.4 m/s 条件表，schema 与正式汇总略有不同 |
| `output/` | 当前 84 个样本 × 2 张图，共 168 张解释图 |
| `output.zip` | 当前解释图目录的压缩归档 |

## 逐株 coefficient 绘图

先检查并修改 `config.yaml` 中的绝对路径，确保输出目录只用于本次绘图：

```powershell
python visualization/plot_plant_coefficients.py `
  --config visualization/config.yaml
```

每个样本输出：

- `*_point_coefficients.png`：每个实际水草位置上的离散 `latent_coefficient`；
- `*_gaussian_influence.png`：把离散值用 Gaussian 核扩散成连续影响图，方便观察空间模式。

Gaussian 图是视觉插值，不代表水槽中被直接测量的连续力场。`latent_coefficient` 也不是实测逐株阻力。

脚本按当前模型接口传入逐株角度 `plant_angles` 和实测弯曲刚度 `plant_stiffness`（EI，单位 N·m²），并与模型使用相同的 CPU/CUDA 设备。两项属性来自 JSONL 中的逐株记录，不需要重新训练模型。

出现低 EI 的 `NumericalStabilityWarning` 时，表示物理求解器提示可能数值不稳定；它本身会继续求解，与“新模型需要 plant_angles 和 plant_stiffness”的输入缺失异常不同。应核对 EI 的实测值和单位，不能通过改大 EI 来修复输入接口问题。

脚本会清空配置的输出目录后重建图片。不要把输出指向仓库根、训练产物目录或混有其他文件的目录。

## 预测对比切图

### 两轮实验的误差棒

`plot_drag_with_uncertainty.py` 读取合并各折测试集的预测 CSV 和重复性 CSV，按
`model_id + angle + flow_speed` 匹配工况。两轮实验均值是蓝色曲线的中心值，
误差棒为样本标准差 SD，采用 `ddof=1`；两次测量时 `SD=abs(F1-F2)/sqrt(2)`。
只接受完整且 `retained=True` 的工况，不把短记录的 10000% 标记当作真实误差。

```powershell
& 'C:\Users\Mayn\miniconda3\envs\Transformer\python.exe' visualization/plot_drag_with_uncertainty.py
```

默认读取同级 `multiR_60deg_sample/cv_predictions.csv` 和本仓库的
`datasets/multi_stiffness/repeatability.csv`，要求包含全部 287 个合格的 60°
整数倍样本。默认输出在预测目录中，与原图并存：

- `cv_drag_comparison_with_uncertainty.png`：10800×3600 像素，600 dpi。
- `cv_drag_comparison_with_uncertainty.svg`：可放大查看的矢量图。
- `cv_drag_comparison_with_uncertainty_detail.png/.svg`：保留全部样本，仅显示实验与预测的细节版，方便看清 SD。
- `cv_drag_comparison_with_uncertainty_data.csv`：按绘图横坐标排序的预测、实验均值和 SD。

点大小默认为 0.45 pt，并取消点边框；连线为 0.50 pt，深色误差棒为 0.90 pt。
原图的点大小为 2.5 pt；排序仍为 `isolated_drag`、`target_drag`、
`source_index` 升序。路径、点大小、线宽、分辨率和尺寸可用 CLI 覆盖，详见 `--help`。
默认拒绝覆盖已有输出，重画可加 `--overwrite`。预览和输入/输出摘要独立保存到
`文档/工作记录/60度误差棒绘图20261021/`。

### 按样本比例切图

当前默认输入路径已过时，建议完整显式指定：

```powershell
py -3.11 visualization/plot_split_drag_comparison.py `
  model/artifacts/run_001/fold_0/test_predictions.csv `
  --cuts 0.3 0.6 `
  --output-dir visualization/test_parts `
  --title "Run 001 Fold 0 test"
```

输入至少需要 `isolated_drag,target_drag,predicted_drag` 三列。切图只是改变展示范围，不改变预测数据。

## JSONL 速度筛选

两个工具都会原地替换输入，默认目标 `visualization/dataset.jsonl` 当前不存在。安全用法是先复制：

```powershell
Copy-Item Experiment/output/dataset.jsonl visualization/dataset_0.4.jsonl
py -3.11 visualization/filter_speed_04.py visualization/dataset_0.4.jsonl
```

或：

```powershell
Copy-Item Experiment/output/dataset.jsonl path/to/working_copy.jsonl
py -3.11 visualization/remove_speed_01.py path/to/working_copy.jsonl
```

不要对唯一完整数据运行筛选器。筛选结果应在文件名或 manifest 中写明规则和源数据版本。

## 相关文档

- [数据字典与文件物理意义](../文档/维护者指南/数据字典与文件物理意义.md)
- [脚本与端到端工作流](../文档/维护者指南/脚本与端到端工作流.md)
- [训练产物与结果解释](../文档/维护者指南/训练产物与结果解释.md)
- [维护规范与已知问题](../文档/维护者指南/维护规范与已知问题.md)

