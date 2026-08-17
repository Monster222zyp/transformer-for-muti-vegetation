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
- `single_drag [N]`：当前统一为 1 的孤立单株阻力；
- `global_features [1]`：流速 U；
- `target_drag`：用于训练的 `max(FX_0, 0)`；
- `raw_target_drag`：未经截断的原始 `FX_0`，仅供审计；
- `model_id`、`angle`、`flow_speed`、`source_index`：不送入网络的追踪信息。

一个 batch 内的植株数不同，因此 `collate_hydro_samples` 会做动态 padding（补齐到该 batch 的最大植株数），并用 `plant_mask` 标记真实植株。padding 不参与 attention 或阻力求和。

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

checkpoint 会内嵌当前最优模型，因此也可以把 `last.pt` 恢复到另一个
`--artifact-dir`；新目录会立即生成自己的 `best.pt`。恢复时会比较 checkpoint 保存的
完整模型配置与当前模型，避免把权重误载入结构不同的网络。

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

更完整的默认值位于 `model/configs/base.yaml`。CLI 参数优先于 YAML。优化器为 AdamW，loss 是总体相互作用系数 `C = D_total / sum(single_drag)` 的 MSE；学习率先线性 warmup，再 cosine 衰减，同时执行梯度范数裁剪。

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

训练 checkpoint 保存了数据、构型文件、负标签策略和 held-out source indices。未显式
提供 `--data` 或 `--input-csv` 时，评估入口自动复用这些绝对路径；显式 CLI 相对路径则
始终按当前工作目录解释。

评估输出包括：

- `evaluation_metrics.json`：`MAE_D`、`RMSE_D`、`R2`、`MAE_C`、`RMSE_C`、`MAPE_D`、MAPE 覆盖率和 `sMAPE_D`；
- `evaluation_predictions.csv`：逐行原始标签、有效标签、预测值及 C；
- `evaluation_plant_coefficients.csv`：每株水草的位置和 latent coefficient。
- `evaluation_context.json`：`held_out`、`in_sample` 或 `external` 范围及实际 source indices。

MAPE（平均绝对百分比误差）不能除以零，所以只统计 `target_drag > 1e-6` 的行并报告覆盖率。sMAPE（对称平均绝对百分比误差）可安全处理零标签。逐株 coefficient 仅是帮助模型完成总阻力预测的潜变量，在没有逐株 CFD 标签前不能解释为真实单株阻力。

## 6. 训练产物与测试

`model/artifacts/` 默认被 Git 忽略。主要产物有配置快照、fold 分配表、scaler、`best.pt`、`last.pt`、最终 checkpoint、训练历史、逐样本预测、逐株系数和指标汇总。

运行不含完整训练的快速测试：

```powershell
py -3.11 -m pytest model/tests/test_training_utils.py -q
```

该测试检查 group 不泄漏、零标签指标，以及 scheduler/checkpoint 保存恢复。完整模型的几何、mask、permutation 与 forward/backward 测试位于同一测试目录的其他文件中。
