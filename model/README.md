# 多水草 HydroTransformer

本目录包含八列总数据集的生成结果、HydroTransformer 网络、训练与评估代码。模型的任务是根据水草二维排列和流速，预测该构型的总阻力。

## 1. 环境安装

建议使用 Python 3.11，在项目根目录执行：

```powershell
py -3.11 -m pip install -r model/requirements.txt
```

`PyTorch` 是深度学习框架；如果需要 NVIDIA 显卡加速，请根据本机 CUDA 版本使用 PyTorch 官方安装命令。默认配置的 `device: auto` 会自动检测显卡，没有显卡时使用 CPU，数据与模型统一使用 `float32`。

## 2. 生成八列总数据

在项目根目录运行数据整合命令：

```powershell
py -3.11 -m model.prepare_dataset
```

脚本会读取 `summarized_data/model_N/angB.csv` 和 `Experiment/input.csv`，生成
`model/data/all_models.csv` 与 `model/data/all_models.audit.json`。总 CSV 固定为
332 条真实记录和八列；四个缺测工况只写入审计报告，不会插值或补零。重复执行该
命令会按 model、angle、flow speed 的数值顺序重新生成相同的数据集。

## 3. 输入和标签

默认数据文件是 `model/data/all_models.csv`。数据加载器会把一行转换为：

- `positions [N,2]`：N 株有效水草的二维无量纲坐标，相邻格点距离为 1；
- `plant_state [N]`：每株水草的状态编号；`0/120/240°` 为状态 1，`60/180/300°` 为状态 2；
- `single_drag [N]`：按“状态 × 当前流速”读取的孤立单株基准阻力；
- `global_features [1]`：流速 U；
- `target_drag`：用于训练的 `max(FX_0, 0)`；
- `raw_target_drag`：未经截断的原始 `FX_0`，仅供审计；
- `state_id`、`model_id`、`angle`、`flow_speed`、`source_index`：用于状态选择和结果追踪的元数据。

一个 batch 内的植株数不同，因此 `collate_hydro_samples` 会做动态 padding（补齐到该 batch 的最大植株数），并用 `plant_mask` 标记真实植株。padding 不参与 attention 或阻力求和。

物理测量得到的单根默认阻力写在 `model/configs/physical.yaml`。该文件分别保存
状态 1/2 在 `0.1、0.2、0.3、0.4 m/s` 下的八个正式实验阻力值，单位为 `N`。
训练会严格拒绝缺失流速、非正数、重复角度或错误单位。两个状态始终使用不同的
可学习初始 Token，以区分两种角度状态的相互作用特征。

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
checkpoint 保存的完整模型配置与当前模型，避免把权重误载入结构不同的网络。

正式的5折交叉验证与全量重训：

```powershell
py -3.11 -m model.train --mode cv
```

交叉验证的 group 固定为 `model_id`：同一基础排列的六个角度和所有流速只能出现在 train、validation、test 中的一个集合，避免数据泄漏。每个外层训练集合再按固定 seed 抽取20%的 model groups 作为 validation。每折的 early stopping 最优 epoch 完成后，以五折最优 epoch 的中位数在全部数据上重训 `final_model.pt`。

常用覆盖参数：

```powershell
py -3.11 -m model.train --mode cv `
  --data model/data/all_models.csv `
  --artifact-dir model/artifacts/run_001 `
  --device auto --batch-size 32 --max-epochs 500 --seed 20260814
```

更完整的默认值位于 `model/configs/base.yaml`。CLI 参数优先于 YAML。优化器为 AdamW：CUDA 训练启用 fused 实现，CPU 自动回退普通实现。训练使用稳定化目标相对 MSE：

```text
mean(((predicted_drag - target_drag) / max(abs(target_drag), relative_floor))^2)
```

`relative_floor` 是当前训练子集中所有正 `target_drag` 的 5% 分位数。每个 fold 只使用自己的 train 索引拟合一次，validation 和 test 直接复用，不能参与统计；Overfit 使用前 32 条训练样本拟合，final 重训使用全部训练数据拟合。这样小阻力样本按相对偏差参与优化，同时避免 4 个零标签发生除零。若训练子集没有正标签，训练会直接报错。学习率先线性 warmup，再 cosine 衰减，同时执行梯度范数裁剪。`C` 仍作为评估指标输出，但不参与训练 loss。

训练开始时会先打印当前 overfit/fold 的样本数和实际 `relative_floor`。之后每 10 个 epoch 在 terminal 输出一次：当前 epoch、`train_relative_mse`、`validation_relative_mse` 和学习率。若 early stopping 发生在非 10 倍数位置，还会额外打印停止时的 loss。间隔可通过 `training.progress_interval_epochs` 调整。

## 5. 独立评估

```powershell
py -3.11 -m model.evaluate `
  --checkpoint model/artifacts/final_model.pt `
  --output-dir model/artifacts/evaluation
```

评估范围由 checkpoint 角色决定：

- `fold_N/best.pt` 自动只评估该折保存的 held-out test `source_index`；
- `final_model.pt` 在原训练数据上评估，并明确标记为 `in_sample`，不能当泛化结果；
- `overfit/best.pt` 只评估保存的32条诊断样本。

评估其他 CSV 必须同时显式声明 `--external-data`：

```powershell
py -3.11 -m model.evaluate `
  --checkpoint model/artifacts/final_model.pt `
  --data path/to/external.csv `
  --external-data
```

训练 checkpoint 保存了数据、构型文件、负标签策略、完整物理阻力表、相对 loss 名称、实际 `relative_floor`、分位数策略和 held-out source indices。未显式
提供 `--data` 或 `--input-csv` 时，评估入口自动复用这些绝对路径；显式 CLI 相对路径则
始终按当前工作目录解释。评估优先使用 checkpoint 内嵌的物理表，而不是重新读取当前
磁盘上的 `physical.yaml`，因此修改实测值不会改变历史 checkpoint 的含义。旧单 Token
checkpoint 不支持双状态输入，必须重新训练。同属当前双状态结构、但使用旧绝对 MSE 训练的 checkpoint 仍可加载做推理；由于优化目标不同，训练器会拒绝用它恢复新训练。

评估输出包括：

- `evaluation_metrics.json`：`MAE_D`、`RMSE_D`、`R2`、`MAE_C`、`RMSE_C`、`MAPE_D`、MAPE 覆盖率和 `sMAPE_D`；
- `evaluation_predictions.csv`：逐行原始标签、有效标签、预测值及 C；
- `evaluation_plant_coefficients.csv`：每株水草的位置和 latent coefficient。
- `evaluation_context.json`：`held_out`、`in_sample` 或 `external` 范围及实际 source indices。

MAPE（平均绝对百分比误差）不能除以零，所以只统计 `target_drag > 1e-6` 的行并报告覆盖率。sMAPE（对称平均绝对百分比误差）可安全处理零标签。逐株 coefficient 仅是帮助模型完成总阻力预测的潜变量，在没有逐株 CFD 标签前不能解释为真实单株阻力。

## 6. 训练产物与测试

`model/artifacts/` 默认被 Git 忽略。主要产物有配置快照、fold 分配表、scaler、`best.pt`、`last.pt`、最终 checkpoint、`history.json`、`history.csv`、逐样本预测、逐株系数和指标汇总。`history.csv` 的列为 `epoch`、`train_relative_mse`、`validation_relative_mse` 和 `learning_rate`。`best.pt` 是轻量评估 checkpoint，只含模型权重、scaler、配置、最佳 epoch/loss 和评估范围信息；`last.pt` 才含 optimizer、scheduler 和 history，用于断点续训。所有 checkpoint 都记录实际使用的相对分母。默认每 100 个 epoch 更新 `last.pt`，最终 epoch 与 early stopping 时强制保存。

checkpoint 先写入带进程 PID 和随机标识的唯一临时文件，再原子替换正式文件。Windows 短暂锁定目标文件时会按 0.2、0.5、1、2 秒自动重试；全部失败时，终端会显示完整临时文件路径，并保留该可恢复文件，不会删除或覆盖它。

每个 `fold_N/` 和 `overfit/` 目录还会生成 `loss_curve.png`：横轴是 epoch，纵轴是稳定化相对 MSE loss，两条折线分别表示训练集和验证集。常规采样点间隔为 10 个 epoch；若训练在非 10 倍数处提前停止，图中会额外保留最后一个 epoch。采样间隔可通过 `training.loss_plot_interval_epochs` 调整。全量 final 重训没有验证集，因此只在 terminal 输出训练 loss，不生成伪造的双折线验证图。

运行不含完整训练的快速测试：

```powershell
py -3.11 -m pytest model/tests/test_training_utils.py -q
```

该测试检查 group 不泄漏、零标签指标，以及 scheduler/checkpoint 保存恢复。完整模型的几何、mask、permutation 与 forward/backward 测试位于同一测试目录的其他文件中。
