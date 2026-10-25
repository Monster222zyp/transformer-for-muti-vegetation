# 多水草 HydroTransformer

本目录包含 HydroTransformer 网络、训练与评估代码。模型读取逐株位置、转盘角度、实测弯曲刚度 EI 和真实流速，预测整个构型的总阻力。JSONL 是“一行一个 JSON 对象”的数据文件；新数据采用严格 schema v3，即第 3 版字段与校验规则。2026-10-22 核对时，`configs/base.yaml` 指向根目录 1197 条最新全角度记录的 `dataset.jsonl`；最新数据也位于 `datasets/multi_stiffness/dataset.jsonl`，其六角子集为同目录 287 条的 `dataset_60deg.jsonl`。`datasets/所有.jsonl` 的 460 条属于历史批次。旧批次重建脚本默认输出到 `Experiment/output/rebuilt_all/`。

这里提到的“硬度”明确指弯曲刚度 `EI`，单位 N·m²，不是相对倍数，也不是杨氏模量 `E`。`plants[].stiffness` 是 EI 的数值；例如 `1.0` 就表示 `1.0 N·m²`。EI 同时进入逐株神经网络编码和物理默认力计算。

首次维护本目录时，建议同时阅读 [项目维护者指南](../文档/维护者指南/README.md) 和 [训练产物与结果解释](../文档/维护者指南/训练产物与结果解释.md)。最新的逐株 EI、schema3、角度编码和 checkpoint6 规则以本 README 与 [配置说明](configs/README.md) 为准；早期指南中共享 Token 或旧数据版本的描述属于历史架构。

## 1. 环境安装

批量进行 RoPE/relative value × sample/plant_count 的八工况消融可使用
`python -m model.run_ablation`，详见 [批量消融使用说明](../文档/工作记录/八工况消融实验20261022/使用说明.md)。
默认结果写到仓库父目录的 `ablation_rope_relative_value_20261022/`，每工况独立
保存 artifacts，批次根目录用 `comparison.csv` 和 `comparison.md` 比较 MAPE_D、R²。

建议使用 Python 3.11，在项目根目录执行：

```powershell
py -3.11 -m pip install -r model/requirements.txt
```

`model/requirements.txt` 已包含默认力求解器和实验滤波所需的 SciPy。求解器已经放进项目内部，无需安装 Downloads 中的包或 MATLAB。`PyTorch` 是深度学习框架；如果需要 NVIDIA 显卡加速，请根据本机 CUDA 版本使用 PyTorch 官方安装命令。默认配置的 `device: auto` 会自动检测显卡，没有显卡时使用 CPU，数据与模型统一使用 `float32`。

## 2. 生成实验汇总 CSV 与模型 JSONL

在项目根目录先完成 sensor 数据过滤，再运行实验汇总命令：

```powershell
py -3.11 Experiment/filter_sensor_data.py
py -3.11 Experiment/summarize_sensor_data.py
```

`summarize_sensor_data.py` 默认使用 `full17` 模式，读取 `filtered_data/model_N/`、`Experiment/input.csv` 与 `Experiment/stiffness_config.yaml`，在 `Experiment/output/rebuilt_all/` 生成汇总 CSV、全量/六角/非六角 JSONL 和重复性检验 CSV。`--input-csv` 可指定构型表，`--config` 可指定材料标签和筛选配置：

```powershell
py -3.11 Experiment/summarize_sensor_data.py `
  --input-csv Experiment/input.csv `
  --config Experiment/stiffness_config.yaml
```

CSV 带表头，测量列顺序为：

```text
sample_id,model_id,angle,vegetation_layout,TX,TY,TZ,FX_0,FY_0,FZ,flow_speed
```

每一行是一条“模型 × 角度 × 流速”的实验记录，按 `model_id → angle → flow_speed` 排序。相同 model 的 `60°` 行保存顺时针旋转 60° 后的 37 位排布。默认排除 `0.1 m/s`；新三组共 216 对两轮实验，删除 `FX_0` 对称差距大于 15% 的 5 对，合并剩余 211 对为平均记录。加上旧 14 组保留的 249 条，得到全量 460 条、六角 300 条、非六角 160 条。缺测工况不插值、不补零。公共函数的旧七个位置参数调用保留 `legacy14` 兼容行为，仍得到 332 条历史基线。

每份 JSONL 同样一行一个 sample，并额外包含变长的 `plants` 数组。每根水草保存 `plant_index、grid_index、x、y、angle、stiffness`；当前同一 sample 内每根水草的 angle 都等于 sample angle。CSV 中 `0` 表示空位，`1/2/3` 是材料标签；默认映射分别为 `0.5/1.0/2.0 N·m²`，可在 YAML 中填写实际测量值。JSONL 的 `vegetation_layout` 仍是 37 位的 `0/1` 占位串，逐株 EI 单独写在 `stiffness` 中。

## 3. 输入和标签

当前 17 组构型的完整数据文件是 `datasets/所有.jsonl`；默认配置文件实际读取根目录的六角文件 `dataset.jsonl`。如需训练全角度数据，在 `base.yaml` 改 `data.dataset_path`。数据加载器会直接读取逐株字段，并把一行转换为：

- `positions [N,2]`：N 株有效水草的二维无量纲坐标，相邻格点距离为 1；
- `plant_angles [N]`：每根水草的转盘角 θ，单位 degree，用于网络编码、审计和默认力计算；
- `plant_stiffness [N]`：每根水草的实测弯曲刚度 EI，单位 N·m²；
- `single_drag [N]`：按转盘角度、逐株 EI、当前真实流速及物理参数计算的孤立单株基准阻力；
- `global_features [3]`：数据加载阶段的 `[真实 U, sin(A), cos(A)]`，A 为共同转盘角；
- `target_drag`：用于训练的 `max(FX_0, 0)`；
- `raw_target_drag`：未经截断的原始 `FX_0`，仅供审计；
- `sample_id`、`model_id`、`angle`、`flow_speed`、`source_index`：用于结果追踪的元数据。

一个 batch 内的植株数不同，因此 `collate_hydro_samples` 会做动态 padding（补齐到该 batch 的最大植株数），并用 `plant_mask` 标记真实植株。padding 不参与 attention 或阻力求和。

`model/configs/physical.yaml` 的版本 3 填写圆柱和叶片几何、EI 单位、水密度、
阻力系数和求解参数，不再填写所有植株共享的杨氏模量。物理适配器使用
`I = h × t³ / 12` 和 `E = EI / I`，将每株 EI 换算成求解器所需的 E；
h 和 t 分别为单片叶片的宽度/高度和厚度，E 的单位是 Pa。转盘角 A 对应三列相对来流角
`[(A+60)%360, (A+180)%360, (A+300)%360]`；顺序不可排序，三列各相隔 120°。
Dataset 调用项目内的原 `seagrass_drag` 求解器，直接取其 `total_force_n` 作为
单株默认力。它已经包括每列叶片和一次圆柱力。相同参数、角度、流速和 EI 的结果会缓存。
默认力在 CPU 数据加载阶段预先计算，forward 使用该力乘逐株系数后求和；
不在每个 batch 中重复做梁变形求解，计算使用真实 m/s 流速而非标准化流速。
参数来源、单位及使用方法见 [物理配置说明](configs/README.md)。

逐株 `PlantEncoder` 使用 `[sin(θ), cos(θ), log(EI / EI_ref)]` 生成初始 Token，
即网络中代表该株水草的特征向量。角度在计算三角函数前从 degree 转成 radian；
`EI_ref` 默认 `1.0 N·m²`，除法使 log 的输入没有单位。MLP 是由全连接层组成的
小网络，结构为 `Linear(3,128) → SiLU → Linear(128,d_model)`，默认
`d_model=256`。`Linear` 是可学习的线性变换，`SiLU` 是允许网络表达非线性关系的
激活函数。128 可通过 `plant_encoder_hidden_dim` 调整。

全局条件由现有 `GlobalEncoder` 编码，再送入每层 `ConditionalLayerNorm`；
后者根据流速和共同角度调整该层的归一化参数。全局输入为
`[标准化 U, sin(A), cos(A)]`。只标准化 U，保留两个角特征的物理范围。
位置仍通过 2D RoPE 和 relative Value 进入 attention，即根据位置决定植株之间
的信息交换方式；单株默认力仍在输出时与逐株系数相乘并求和。

schema v1 要求坐标与固定网格一致；schema v2 允许现有非六角数据的真实坐标：
`grid_index` 指向向下取整到 60° 的旋转排布锚点，x/y 再顺时针旋转 `angle%60`。
schema v3 保留连续旋转坐标的语义，允许有限的小数角度，并要求每株显式填写
正且有限的 `stiffness`。缺失、零、负数、NaN 或 Infinity 都会报错。三个版本均
严格校验逐株索引、37 位排布和同向角度。为了读取两个保留的历史副本，v1/v2
会在内存中补 `EI=1.0 N·m²`；读取不会改写原文件。推荐训练使用 v3 数据。

本次已将 `datasets/` 递归的 10 份 JSONL 和根目录 `dataset.jsonl` 迁移为 v3，
共 4,606 条、57,316 株，全部补 `stiffness=1.0`。这是对现有数据的统一占位赋值，
不能据此证明已经掌握不同 EI 下的实验规律：所有训练样本 EI 都相同时，网络
无法学到跨 EI 的变化，必须补充真实且不同 EI 的实验数据。新构建脚本会按
CSV 材料标签和 YAML 映射生成 EI，不会自动继承此次历史迁移的统一赋值。

镜像增强保留 EI，重建镜像后的逐株与全局角特征，并重新计算默认力。原求解器的
列遮蔽处理不保证 `F(A)=F(-A)`，因此镜像后不能沿用原角度的默认力。

默认模型同时启用 2D RoPE 和 relative Value，使网络能够读取植物布局。方向性
attention 约定 `x` 越大越靠上游、水流从 `x+` 一侧流向 `x-` 一侧：`none` 保留
双向 attention，`soft` 使用可配置的 ReLU 连续降低下游 source 的权重，`hard` 则
完全屏蔽下游 source。同一 `x` 横截面和 self-edge 在三种模式中都保持可见。

流速标准化必须只使用当前训练 fold 的均值和标准差。fold 是一次训练/验证/测试划分；每折的统计保存在 `fold_N/scaler.json`。评估时直接从 checkpoint 恢复，禁止使用测试集重新计算；`sin(A)` 和 `cos(A)` 始终不标准化。

## 4. 快速检查与正式训练

先运行32条样本的 overfit（过拟合）检查。它用于确认网络、损失和反向传播确实可以把一个小 batch 拟合下来：

```powershell
py -3.11 -m model.train --mode overfit `
  --data datasets/所有.jsonl `
  --artifact-dir model/artifacts/smoke `
  --max-epochs 300
```

中断后可以从最近一次状态恢复：

```powershell
py -3.11 -m model.train --mode overfit --max-epochs 300 `
  --resume-checkpoint model/artifacts/overfit/last.pt
```

`last.pt` 不再重复内嵌最优模型；恢复时会从它原目录的轻量 `best.pt` 读取最佳权重，
因此也可以恢复到另一个 `--artifact-dir`，并在新目录重建 `best.pt`。恢复时会比较
checkpoint 保存的完整模型配置与当前模型，避免把权重误载入结构不同的网络。带
`--resume-checkpoint` 的运行会保留输出目录中的已有文件，不执行下面的自动清理。

正式划分评估建议显式指定数据和新的产物目录，避免复用并清理当前默认结果：

```powershell
py -3.11 -m model.train --mode cv `
  --data datasets/所有.jsonl `
  --artifact-dir model/artifacts/run_001
```

使用 `--mode cv` 时，通过配置文件中的 `cross_validation.split_mode` 选择自动划分方式：

- `sample`：逐条样本随机分配，适合衡量对同分布工况的插值能力；
- `model`：同一 `model_id` 的六个角度和所有流速只能完整进入 train、validation、test 中的一个集合；
- `plant_count`：水草根数相同的全部样本只能完整进入一个集合，用于验证模型对未见水草根数的泛化能力；
- `flow_speed`：固定把 `0.1/0.2/0.3 m/s` 全部作为 train，把完整 `0.4 m/s` 同时作为 validation 和 test。

`sample`、`model` 和 `plant_count` 由 `seed` 控制并尽量平衡各折样本数；每折完成后，以最佳 epoch 中位数在全部数据上重训 `final_model.pt`。`flow_speed` 是无随机划分的固定单折协议，会忽略 `n_splits`、`validation_fraction` 和划分 seed，只生成 `fold_0`，不会生成 `final_scaler.json` 或 `final_model.pt`。

请特别注意：`flow_speed` 的完整 `0.4 m/s` 数据参与 early stopping，同时又写成 test 结果，因此 test 指标不是独立 held-out 泛化指标。`cv_metrics.json` 会以 `validation_test_overlap: true` 和 `final_retraining_performed: false` 明确记录这两个事实。

### 4.1 显式指定训练集、验证集和测试集

如果已经在仓库外或其他脚本中完成划分，首先修改 `model/configs/base.yaml` 中的运行模式和三份数据路径：

```yaml
mode: predefined
predefined_split:
  train_dataset_path: path/to/train.jsonl
  validation_dataset_path: path/to/validation.jsonl
  test_dataset_path: path/to/test.jsonl
```

这些相对路径以项目根目录为基准。保存后运行：

```powershell
py -3.11 -m model.train --artifact-dir model/artifacts/predefined_run_001
```

临时实验仍可使用命令行覆盖 YAML，而不必改动配置文件：

```powershell
py -3.11 -m model.train --mode predefined `
  --train-data path/to/train.jsonl `
  --validation-data path/to/validation.jsonl `
  --test-data path/to/test.jsonl `
  --artifact-dir model/artifacts/predefined_run_001
```

三份文件的格式必须满足 `HydroDataset` 校验，推荐使用逐株 EI 必填的 schema v3；历史 v1/v2 仅按上述默认 EI 兼容读取。启用 `predefined` 时，三个路径必须在 YAML 或命令行中全部给出，且不能与单数据集参数 `--data` 混用。程序会在清理旧产物前完成以下检查：

- 分别使用原有 `HydroDataset` 校验三份 JSONL；
- 确认三个集合都非空；
- 拒绝跨 train、validation、test 重复的 `sample_id`；
- 保存每个文件的绝对路径、样本数和 SHA-256 内容摘要。

该模式的角色固定为：

- train：只用于梯度更新、流速 scaler 和相对 loss 分母下限拟合；
- validation：只用于 early stopping 和选择 `best.pt`；
- test：只在最佳 checkpoint 上执行最终评估。

`predefined` 不执行 final retraining，因为把 test 加入重训会造成直接泄漏，把 train 与 validation 自动合并也会改变用户明确给出的集合语义。推荐 checkpoint 为：

```text
model/artifacts/predefined_run_001/predefined/best.pt
```

根目录会生成 `predefined_metrics.json`、`predefined_predictions.csv`、`predefined_plant_coefficients.csv`、分流速指标/图片和 `predefined_split_assignments.csv`；不会生成 `final_model.pt`。

切回原有五折模式时，把 `mode` 改回 `cv`；`predefined_split` 中保留已填写的路径也不会影响五折训练。`--mode` 和三条命令行路径只用于临时覆盖 YAML 中对应的值。

常用覆盖参数：

```powershell
py -3.11 -m model.train --mode cv `
  --data datasets/所有.jsonl `
  --artifact-dir model/artifacts/run_001 `
  --device auto --batch-size 32 --max-epochs 500 --seed 20260816
```

当前 `base.yaml` 的默认值是 `plant_count`。若要单独进行跨流速实验，应复制一份新的 YAML，并明确改为：

```yaml
cross_validation:
  split_mode: flow_speed
  n_splits: 5
  validation_fraction: 0.2
```

`flow_speed` 会忽略示例中的折数和验证比例；保留这两个配置是为了切回普通模式时可以直接复用。然后正常运行 `python -m model.train --mode cv`。`split_mode` 不提供命令行覆盖，避免正式实验时命令行与配置文件记录不一致。

训练结果默认保存在项目内的 `model/artifacts/`，不是项目根目录下另一个可能存在的 `artifacts/`。训练开始和成功结束时都会打印解析后的绝对输出目录；`flow_speed` 的推荐模型是 `model/artifacts/fold_0/best.pt`，它按设计不会生成 `final_model.pt`。

每次不带 `--resume-checkpoint` 的 fresh run 会先成功加载配置、物理参数、完整 Dataset 并验证划分，然后清空本次最终 `artifact_dir` 中的旧训练结果。这能防止从五折切换到 `flow_speed` 后残留旧 `fold_1`～`fold_4` 或 `final_model.pt`。使用 `--artifact-dir model/artifacts/run_001` 时只清理 `run_001`，不会影响兄弟实验目录。训练中途失败时旧结果已无法恢复，因此重要实验应先备份或使用新的 run 目录。

为避免误删，目录中会保留 `.hydrotransformer-artifacts` ownership marker。项目默认 `model/artifacts` 及其子目录可直接管理；项目外的自定义非空目录必须已经包含内容正确的 marker。文件系统根、用户主目录、项目根、源码目录、当前工作目录、输入文件的祖先目录以及 symbolic link/Windows junction 都会被拒绝。

更完整的默认值位于 `model/configs/base.yaml`。CLI 参数优先于 YAML。优化器为 AdamW：CUDA 训练启用 fused 实现，CPU 自动回退普通实现。训练使用稳定化目标相对 MSE：

```text
mean(((predicted_drag - target_drag) / max(abs(target_drag), relative_floor))^2)
```

`relative_floor` 是当前训练子集中所有正 `target_drag` 的 5% 分位数。每个 fold 只使用自己的 train 索引拟合一次，validation 和 test 直接复用，不能参与统计；因此 `flow_speed` 只用 `0.1/0.2/0.3 m/s` 计算 scaler 和 `relative_floor`。Overfit 使用前 32 条训练样本拟合，普通模式的 final 重训使用全部训练数据拟合。这样小阻力样本按相对偏差参与优化，同时避免 4 个零标签发生除零。若训练子集没有正标签，训练会直接报错。学习率先线性 warmup，再 cosine 衰减，同时执行梯度范数裁剪。`C` 仍作为评估指标输出，但不参与训练 loss。

训练开始时会先打印当前 overfit/fold 的样本数和实际 `relative_floor`。之后按 `training.progress_interval_epochs` 在 terminal 输出当前 epoch、`train_relative_mse`、`validation_relative_mse` 和学习率；当前 `base.yaml` 设置为每 1 个 epoch 输出一次。若 early stopping 发生在非固定间隔位置，还会额外打印停止时的 loss。

## 5. 独立评估

```powershell
py -3.11 -m model.evaluate `
  --checkpoint model/artifacts/final_model.pt `
  --output-dir model/artifacts/evaluation
```

评估范围由 checkpoint 角色决定：

- 普通模式的 `fold_N/best.pt` 自动只评估该折保存的 held-out test `source_index`；
- `flow_speed` 的 `fold_0/best.pt` 自动评估保存的 `0.4 m/s` 数据，并把范围标记为 `validation_test_overlap`，不能解释为独立 held-out；
- `final_model.pt` 在原训练数据上评估，并明确标记为 `in_sample`，不能当泛化结果；
- `overfit/best.pt` 只评估保存的32条诊断样本。
- `predefined/best.pt` 自动加载训练时保存的 test JSONL、核对 SHA-256，并把全部 test 样本标记为独立 `held_out`。

自动复评显式三数据集训练的原 test 集：

```powershell
py -3.11 -m model.evaluate `
  --checkpoint model/artifacts/predefined_run_001/predefined/best.pt `
  --output-dir model/artifacts/predefined_run_001/evaluation_test
```

评估其他 JSONL（推荐严格 schema v3）必须同时显式声明 `--external-data`：

```powershell
py -3.11 -m model.evaluate `
  --checkpoint path/to/best_or_final_model.pt `
  --data path/to/external_dataset.jsonl `
  --external-data `
  --output-dir model/artifacts/evaluation_external
```

`--external-data` 表示“用户主动替换 checkpoint 原数据，并评估新 JSONL 的全部样本”，不自动保证这份数据与训练集统计独立。外部 JSONL 必须包含真实 `measurements.FX_0`，因此这是有标签评估入口，不是无标签预测脚本。

如果希望把常用参数直接写在脚本中，可以编辑 `model/evaluate.py` 开头的变量：

```python
DEFAULT_CHECKPOINT_PATH = Path(r"model/artifacts/predefined_run_001/predefined/best.pt")
DEFAULT_DATA_PATH = Path(r"path/to/external_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path(r"model/artifacts/evaluation_external")
DEFAULT_EXTERNAL_DATA = True
DEFAULT_DEVICE = "auto"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
```

保存后可直接运行：

```powershell
py -3.11 -m model.evaluate
```

CLI 参数始终可以覆盖这些顶部默认变量；请核对本机脚本顶部保存的路径，正式评估时建议显式传入 `--checkpoint`。一般应使用 `best.pt` 或 `final_model.pt`；`last.pt` 也能读取，但它代表最近保存 epoch 的权重，不一定是验证集最优权重。

训练 checkpoint 保存了数据路径、负标签策略、完整物理计算参数及算法标识、相对 loss 名称、实际 `relative_floor`、分位数策略和 held-out source indices。未显式
提供 `--data` 时，评估入口自动复用该数据集的绝对路径；显式 CLI 相对路径则始终按当前工作目录解释。评估优先使用 checkpoint 内嵌的物理配置，而不是重新读取当前
磁盘上的 `physical.yaml`，因此修改参数不会改变已保存 checkpoint 的含义。
新角度/EI 编码器使用 checkpoint 版本 6；checkpoint 是保存权重及训练配置的
模型状态文件。旧版本没有这套逐株和全局输入结构，训练恢复与独立评估都会拒绝
旧 checkpoint，必须使用新结构重新训练。v1/v2 数据兼容读取不表示旧模型权重兼容。

评估输出包括：

- `evaluation_metrics.json`：样本数、`MAE_D`、`RMSE_D`、`R2`、`MAE_C`、`RMSE_C`、`MAPE_D`、MAPE 覆盖率和 `sMAPE_D`；
- `evaluation_predictions.csv`：逐行原始标签、有效标签、预测值及 C；
- `evaluation_plant_coefficients.csv`：每株水草的位置和 latent coefficient。
- `evaluation_context.json`：评估范围、checkpoint/split 类型、数据路径与 SHA-256、设备、batch、worker 以及实际 source indices。

MAPE（平均绝对百分比误差）不能除以零，所以只统计 `target_drag > 1e-6` 的行并报告覆盖率。sMAPE（对称平均绝对百分比误差）可安全处理零标签。逐株 coefficient 仅是帮助模型完成总阻力预测的潜变量，在没有逐株 CFD 标签前不能解释为真实单株阻力。

## 6. 训练产物与测试

`model/artifacts/` 默认被 Git 忽略，因此只显示 Git 文件的界面可能看不到训练结果，但文件仍保存在磁盘。主要产物有配置快照、带 `plant_count` 和 `split_mode` 的 fold 分配表、scaler、`best.pt`、`last.pt`、`history.json`、`history.csv`、逐样本预测、逐株系数和指标汇总；普通模式还会生成最终 checkpoint，`flow_speed` 不会生成。`history.csv` 的列为 `epoch`、`train_relative_mse`、`validation_relative_mse` 和 `learning_rate`。`best.pt` 是轻量评估 checkpoint，只含模型权重、scaler、配置、最佳 epoch/loss 和评估范围信息；`last.pt` 才含 optimizer、scheduler 和 history，用于断点续训。所有 checkpoint 都记录实际使用的相对分母、划分模式及 validation/test 是否重叠。默认每 100 个 epoch 更新 `last.pt`，最终 epoch 与 early stopping 时强制保存。

checkpoint 先写入带进程 PID 和随机标识的唯一临时文件，再原子替换正式文件。Windows 短暂锁定目标文件时会按 0.2、0.5、1、2 秒自动重试；全部失败时，终端会显示完整临时文件路径，并保留该可恢复文件，不会删除或覆盖它。

每个 `fold_N/` 和 `overfit/` 目录还会生成 `loss_curve.png`：横轴是 epoch，纵轴是稳定化相对 MSE loss，两条折线分别表示训练集和验证集。采样间隔由 `training.loss_plot_interval_epochs` 控制，当前 `base.yaml` 设置为每 1 个 epoch 一个点；若训练在非固定间隔位置提前停止，图中会额外保留最后一个 epoch。全量 final 重训没有验证集，因此只在 terminal 输出训练 loss，不生成伪造的双折线验证图。

每个 `fold_N/` 会为 validation 和 test 各写一套结果：`*_metrics.json`、`*_predictions.csv`、`*_plant_coefficients.csv`，并分别生成 `validation_drag_comparison.png` 与 `test_drag_comparison.png`。预测图先按 `isolated_drag` 从小到大排序；相同值再按 `target_drag` 从小到大排序。横轴是排序后的样本序号，三条带点折线分别表示 `target_drag`、`predicted_drag` 和 `isolated_drag`。

正常交叉验证结束后，产物根目录还会自动生成两张覆盖全部 held-out test（留出测试集）记录的预测—测量散点图：`cv_drag_parity_model.png` 使用模型预测 `predicted_drag`，`cv_drag_parity_baseline.png` 使用逐株孤立阻力之和 `isolated_drag`。每条测试记录各对应一个点，每张图内部保持横纵轴同范围、等比例坐标与 `y=x` 参考线，并标注样本数、MAE（平均绝对误差）、RMSE（均方根误差）和 R²（决定系数）。默认两张图分别根据本图的实测与预测值取范围，包含零点并在两端各留 5%，避免较大的 baseline 预测把模型散点挤在左下角；全部样本和极值均保留。图下注明范围模式，独立范围下跨图比较绝对误差需读取刻度和误差指标。横轴使用原始实验测量 `raw_target_drag`，保留训练时被归零的负测量；图内误差指标也按原始测量重新计算，因此可能与基于 `target_drag` 的训练指标 JSON 不同。单样本或常量测量的 R² 显示为 `undefined`。两图保存为 300 dpi PNG，默认无需额外命令。

范围模式由 `base.yaml` 的 `output.drag_parity_axis_mode` 控制：`independent` 为各图独立范围（默认），`shared` 为两图共用范围。也可用训练参数 `--drag-parity-axis-mode shared` 覆盖，最终配置会写入训练快照。直接调用 `write_drag_parity_plots(..., axis_mode="shared")` 时同样支持共用范围；这一设置仅改变绘图，不影响训练或预测数值。

散点图直接复用各 fold 合并的 test 预测，不包含 validation，不使用随后全量重训模型的训练内预测。`predefined` 模式同样自动输出两图，文件名前缀为 `predefined_`，只包含显式指定的 test 文件。`flow_speed` 模式的 validation/test 重叠，`overfit` 模式没有独立测试集，因此这两类诊断模式不生成上述全测试集散点图。

在 `sample`、`model`、`plant_count` 三种普通模式中，每套 validation/test 结果还会增加 `*_metrics_by_flow_speed.json/csv` 和 `*_drag_comparison_by_flow_speed.png`。分组指标按 `0.1/0.2/0.3/0.4 m/s` 分别报告样本数、真实 D 均值以及完整的 MAE、RMSE、R²、MAPE、sMAPE 和 C 指标；2×2 PNG 固定用四个面板展示四档流速。根目录同时生成合并 held-out test 的 `cv_metrics_by_flow_speed.json/csv`、`cv_drag_comparison.png` 和 `cv_drag_comparison_by_flow_speed.png`，相同内容也写入 `cv_metrics.json` 的 `aggregate_by_flow_speed`。训练结束时 terminal 会打印精简分流速表格。某一 fold 缺少某档速度时，JSON/CSV 不伪造该组指标，图中对应面板显示 `No samples`。

`flow_speed` 模式不生成上述四档分组文件，因为它的 validation/test 本来只有 `0.4 m/s`；其根目录汇总仍是与 validation 重叠的完整 `0.4 m/s` test，并通过 JSON 标记其非独立语义。

运行不含完整训练的快速测试：

```powershell
py -3.11 -m pytest model/tests/test_training_utils.py -q
```

划分与预测图的专项测试分别位于 `test_splits.py` 和 `test_prediction_visualization.py`；它们检查四种模式的索引语义、普通分组无泄漏、固定流速重叠、排序规则、总体 PNG 和四面板 PNG 输出。`test_rich_flow_speed_evaluation.py` 验证普通 CV 的 fold/root 两级分流速指标与图表产物；`test_flow_speed_training.py` 进一步验证固定流速入口只用前三档速度拟合 scaler/loss floor、跳过四档评估和 final 重训；`test_artifact_directory.py` 验证旧结果清理、危险路径保护、resume 保留和输入验证顺序。完整模型的几何、mask、permutation 与 forward/backward 测试位于同一测试目录的其他文件中。
