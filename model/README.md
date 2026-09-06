# 多水草 HydroTransformer

本目录包含 HydroTransformer 网络、训练与评估代码。模型直接读取 `Experiment/output/dataset.jsonl`，根据每根水草的二维位置和流速预测该构型的总阻力。水草原始角度只用于查询物理默认阻力，不作为神经网络输入。

## 1. 环境安装

建议使用 Python 3.11，在项目根目录执行：

```powershell
py -3.11 -m pip install -r model/requirements.txt
```

`PyTorch` 是深度学习框架；如果需要 NVIDIA 显卡加速，请根据本机 CUDA 版本使用 PyTorch 官方安装命令。默认配置的 `device: auto` 会自动检测显卡，没有显卡时使用 CPU，数据与模型统一使用 `float32`。

## 2. 生成实验汇总 CSV 与模型 JSONL

在项目根目录先完成 sensor 数据过滤，再运行实验汇总命令：

```powershell
py -3.11 Experiment/filter_sensor_data.py
py -3.11 Experiment/summarize_sensor_data.py
```

`summarize_sensor_data.py` 读取 `filtered_data/model_N/` 与 `Experiment/input.csv`，在 `Experiment/output/` 同时生成 `summarized_data.csv` 和 `dataset.jsonl`。CSV 带表头，固定列顺序为：

```text
sample_id,model_id,angle,vegetation_layout,TX,TY,TZ,FX_0,FY_0,FZ,flow_speed
```

每一行是一条“模型 × 角度 × 流速”的真实实验记录，按 `model_id → angle → flow_speed` 排序。相同 model 的 `60°` 行保存顺时针旋转 60° 后的 37 位排布，而不是重复 `0°` 排布。当前严格基线为 332 条记录；四个缺测工况不会插值或补零，而是保留已有的三档流速。

`dataset.jsonl` 同样一行一个 sample，并额外包含变长的 `plants` 数组。每根水草保存 `plant_index、grid_index、x、y、angle`；当前同一 sample 内每根水草的 angle 都等于 sample angle。

## 3. 输入和标签

默认数据文件是 `Experiment/output/dataset.jsonl`。数据加载器会直接读取逐株字段，并把一行转换为：

- `positions [N,2]`：N 株有效水草的二维无量纲坐标，相邻格点距离为 1；
- `plant_angles [N]`：每根水草的原始 degree 角度，只保留用于审计和物理查表；
- `single_drag [N]`：按“原始角度 × 当前流速”读取的孤立单株基准阻力；
- `global_features [1]`：流速 U；
- `target_drag`：用于训练的 `max(FX_0, 0)`；
- `raw_target_drag`：未经截断的原始 `FX_0`，仅供审计；
- `sample_id`、`model_id`、`angle`、`flow_speed`、`source_index`：用于结果追踪的元数据。

一个 batch 内的植株数不同，因此 `collate_hydro_samples` 会做动态 padding（补齐到该 batch 的最大植株数），并用 `plant_mask` 标记真实植株。padding 不参与 attention 或阻力求和。

物理测量得到的单根默认阻力写在 `model/configs/physical.yaml`。该文件内部将
`0/120/240°` 和 `60/180/300°` 分为两组，并保存各自在四档流速下的实验阻力，
单位为 `N`。Dataset 通过 `single_drag_for_angle(angle, flow_speed)` 查询阻力；
HydroTransformer 只接收查询结果，所有水草共享同一个可学习初始 Token。

默认模型同时启用 2D RoPE 和 relative Value，使网络能够读取植物布局。方向性
attention 约定 `x` 越大越靠上游、水流从 `x+` 一侧流向 `x-` 一侧：`none` 保留
双向 attention，`soft` 使用可配置的 ReLU 连续降低下游 source 的权重，`hard` 则
完全屏蔽下游 source。同一 `x` 横截面和 self-edge 在三种模式中都保持可见。

流速标准化必须只使用当前训练 fold 的均值和标准差。每折的统计保存在 `fold_N/scaler.json`；评估时直接从 checkpoint 恢复，禁止使用测试集重新计算。

## 4. 快速检查与正式训练

先运行32条样本的 overfit（过拟合）检查。它用于确认网络、损失和反向传播确实可以把一个小 batch 拟合下来：

```powershell
py -3.11 -m model.train --mode overfit --max-epochs 300
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

正式划分评估：

```powershell
py -3.11 -m model.train --mode cv
```

训练入口只通过配置文件中的 `cross_validation.split_mode` 选择划分方式：

- `sample`：逐条样本随机分配，适合衡量对同分布工况的插值能力；
- `model`：同一 `model_id` 的六个角度和所有流速只能完整进入 train、validation、test 中的一个集合；
- `plant_count`：水草根数相同的全部样本只能完整进入一个集合，用于验证模型对未见水草根数的泛化能力；
- `flow_speed`：固定把 `0.1/0.2/0.3 m/s` 全部作为 train，把完整 `0.4 m/s` 同时作为 validation 和 test；这是当前 `base.yaml` 的默认模式。

`sample`、`model` 和 `plant_count` 由 `seed` 控制并尽量平衡各折样本数；每折完成后，以最佳 epoch 中位数在全部数据上重训 `final_model.pt`。`flow_speed` 是无随机划分的固定单折协议，会忽略 `n_splits`、`validation_fraction` 和划分 seed，只生成 `fold_0`，不会生成 `final_scaler.json` 或 `final_model.pt`。

请特别注意：`flow_speed` 的完整 `0.4 m/s` 数据参与 early stopping，同时又写成 test 结果，因此 test 指标不是独立 held-out 泛化指标。`cv_metrics.json` 会以 `validation_test_overlap: true` 和 `final_retraining_performed: false` 明确记录这两个事实。

常用覆盖参数：

```powershell
py -3.11 -m model.train --mode cv `
  --data Experiment/output/dataset.jsonl `
  --artifact-dir model/artifacts/run_001 `
  --device auto --batch-size 32 --max-epochs 500 --seed 20260816
```

默认的跨流速实验配置如下；若需要验证未见水草根数，可把 `split_mode` 改为 `plant_count`：

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

评估其他 CSV 必须同时显式声明 `--external-data`：

```powershell
py -3.11 -m model.evaluate `
  --checkpoint model/artifacts/final_model.pt `
  --data path/to/external.csv `
  --external-data
```

训练 checkpoint 保存了数据路径、负标签策略、完整物理阻力表、相对 loss 名称、实际 `relative_floor`、分位数策略和 held-out source indices。未显式
提供 `--data` 时，评估入口自动复用该数据集的绝对路径；显式 CLI 相对路径则始终按当前工作目录解释。评估优先使用 checkpoint 内嵌的物理表，而不是重新读取当前
磁盘上的 `physical.yaml`，因此修改实测值不会改变历史 checkpoint 的含义。旧单 Token
checkpoint 不支持双状态输入，必须重新训练。同属当前双状态结构、但使用旧绝对 MSE 训练的 checkpoint 仍可加载做推理；由于优化目标不同，训练器会拒绝用它恢复新训练。

评估输出包括：

- `evaluation_metrics.json`：`MAE_D`、`RMSE_D`、`R2`、`MAE_C`、`RMSE_C`、`MAPE_D`、MAPE 覆盖率和 `sMAPE_D`；
- `evaluation_predictions.csv`：逐行原始标签、有效标签、预测值及 C；
- `evaluation_plant_coefficients.csv`：每株水草的位置和 latent coefficient。
- `evaluation_context.json`：`held_out`、`validation_test_overlap`、`in_sample` 或 `external` 范围及实际 source indices。

MAPE（平均绝对百分比误差）不能除以零，所以只统计 `target_drag > 1e-6` 的行并报告覆盖率。sMAPE（对称平均绝对百分比误差）可安全处理零标签。逐株 coefficient 仅是帮助模型完成总阻力预测的潜变量，在没有逐株 CFD 标签前不能解释为真实单株阻力。

## 6. 训练产物与测试

`model/artifacts/` 默认被 Git 忽略，因此只显示 Git 文件的界面可能看不到训练结果，但文件仍保存在磁盘。主要产物有配置快照、带 `plant_count` 和 `split_mode` 的 fold 分配表、scaler、`best.pt`、`last.pt`、`history.json`、`history.csv`、逐样本预测、逐株系数和指标汇总；普通模式还会生成最终 checkpoint，`flow_speed` 不会生成。`history.csv` 的列为 `epoch`、`train_relative_mse`、`validation_relative_mse` 和 `learning_rate`。`best.pt` 是轻量评估 checkpoint，只含模型权重、scaler、配置、最佳 epoch/loss 和评估范围信息；`last.pt` 才含 optimizer、scheduler 和 history，用于断点续训。所有 checkpoint 都记录实际使用的相对分母、划分模式及 validation/test 是否重叠。默认每 100 个 epoch 更新 `last.pt`，最终 epoch 与 early stopping 时强制保存。

checkpoint 先写入带进程 PID 和随机标识的唯一临时文件，再原子替换正式文件。Windows 短暂锁定目标文件时会按 0.2、0.5、1、2 秒自动重试；全部失败时，终端会显示完整临时文件路径，并保留该可恢复文件，不会删除或覆盖它。

每个 `fold_N/` 和 `overfit/` 目录还会生成 `loss_curve.png`：横轴是 epoch，纵轴是稳定化相对 MSE loss，两条折线分别表示训练集和验证集。采样间隔由 `training.loss_plot_interval_epochs` 控制，当前 `base.yaml` 设置为每 1 个 epoch 一个点；若训练在非固定间隔位置提前停止，图中会额外保留最后一个 epoch。全量 final 重训没有验证集，因此只在 terminal 输出训练 loss，不生成伪造的双折线验证图。

每个 `fold_N/` 会为 validation 和 test 各写一套结果：`*_metrics.json`、`*_predictions.csv`、`*_plant_coefficients.csv`，并分别生成 `validation_drag_comparison.png` 与 `test_drag_comparison.png`。预测图先按 `isolated_drag` 从小到大排序；相同值再按 `target_drag` 从小到大排序。横轴是排序后的样本序号，三条带点折线分别表示 `target_drag`、`predicted_drag` 和 `isolated_drag`。

在 `sample`、`model`、`plant_count` 三种普通模式中，每套 validation/test 结果还会增加 `*_metrics_by_flow_speed.json/csv` 和 `*_drag_comparison_by_flow_speed.png`。分组指标按 `0.1/0.2/0.3/0.4 m/s` 分别报告样本数、真实 D 均值以及完整的 MAE、RMSE、R²、MAPE、sMAPE 和 C 指标；2×2 PNG 固定用四个面板展示四档流速。根目录同时生成合并 held-out test 的 `cv_metrics_by_flow_speed.json/csv`、`cv_drag_comparison.png` 和 `cv_drag_comparison_by_flow_speed.png`，相同内容也写入 `cv_metrics.json` 的 `aggregate_by_flow_speed`。训练结束时 terminal 会打印精简分流速表格。某一 fold 缺少某档速度时，JSON/CSV 不伪造该组指标，图中对应面板显示 `No samples`。

`flow_speed` 模式不生成上述四档分组文件，因为它的 validation/test 本来只有 `0.4 m/s`；其根目录汇总仍是与 validation 重叠的完整 `0.4 m/s` test，并通过 JSON 标记其非独立语义。

运行不含完整训练的快速测试：

```powershell
py -3.11 -m pytest model/tests/test_training_utils.py -q
```

划分与预测图的专项测试分别位于 `test_splits.py` 和 `test_prediction_visualization.py`；它们检查四种模式的索引语义、普通分组无泄漏、固定流速重叠、排序规则、总体 PNG 和四面板 PNG 输出。`test_rich_flow_speed_evaluation.py` 验证普通 CV 的 fold/root 两级分流速指标与图表产物；`test_flow_speed_training.py` 进一步验证固定流速入口只用前三档速度拟合 scaler/loss floor、跳过四档评估和 final 重训；`test_artifact_directory.py` 验证旧结果清理、危险路径保护、resume 保留和输入验证顺序。完整模型的几何、mask、permutation 与 forward/backward 测试位于同一测试目录的其他文件中。
