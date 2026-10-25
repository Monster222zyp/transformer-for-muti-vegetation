# 模型配置、角度/EI 编码与物理默认力

`base.yaml` 决定运行模式、数据路径、网络和训练参数。`physical.yaml` 决定孤立一株水草的理论默认力，两者通过 `data.physics_config_path` 关联。训练入口是 `python -m model.train`，评估入口是 `python -m model.evaluate`。

`output.drag_parity_axis_mode` 控制训练结束后的预测—实测散点图：默认 `independent` 按每张图自己的实测与预测值取范围，避免 baseline（逐株阻力相加的对照预测）过大时压缩模型散点；`shared` 则让两张图共用范围。两种模式的图内 x、y 都保持同范围、等比例并显示 `y=x` 参考线，不裁剪数据。命令行 `--drag-parity-axis-mode shared` 可以覆盖 YAML，实际模式写入配置快照。独立范围下需结合坐标刻度与误差指标比较两图。

这里的“硬度”是实测弯曲刚度 `EI`，单位 N·m²。`E` 是杨氏模量，表示材料抵抗弹性变形的能力，单位 Pa；`I` 是截面二次矩，表示截面形状对弯曲的影响，单位 m⁴。EI 是两者的乘积，既不是无量纲的相对倍数，也不是直接填写 E。每株的 EI 来自 JSONL 的 `plants[].stiffness`。

## base.yaml 中的输入与网络参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `model.global_input_dim` | 3 | 全局输入 `[标准化 U, sin(A), cos(A)]` 的维数 |
| `model.global_hidden_dim` | 64 | 现有 `GlobalEncoder` 全连接网络的中间维数 |
| `model.plant_encoder_hidden_dim` | 128 | 逐株角度/EI 编码网络的中间维数 |
| `model.stiffness_reference_nm2` | 1.0 | `log(EI/EI_ref)` 中的参考 EI，单位 N·m²，必须正且有限 |
| `model.d_model` | 256 | 每株 Token 的维数，即代表该株的特征向量长度 |

单株角度 θ 先从 degree 转成 radian，再计算三角函数。逐株编码器的输入是：

```text
[sin(θ), cos(θ), log(EI / EI_ref)]
    → Linear(3,128) → SiLU → Linear(128,d_model)
```

`Linear` 是可学习的线性变换，`SiLU` 是非线性激活函数，组合起来构成 MLP，也就是一个小型全连接神经网络。以 `EI_ref=1.0 N·m²` 为例，EI 为 `0.5/1/2 N·m²` 时，第三项分别是 `log(0.5)`、`0`、`log(2)`；除以 EI_ref 后，log 的输入没有单位。

数据加载阶段的全局输入是 `[真实 U, sin(A), cos(A)]`，其中 U 的单位是 m/s，A 是共同转盘角。训练时仅使用训练集的 U 均值和标准差进行标准化，两个角特征不参与标准化。评估直接恢复 checkpoint 中的统计。全局向量经现有 `GlobalEncoder` 编码后，送入每层 `ConditionalLayerNorm`；后者根据流速和共同角度调整该层的归一化参数。`global_input_dim` 必须为 3，不能再沿用旧配置的 1。

## 默认力与转盘角

默认力是 `seagrass_drag` 根据叶片变形计算的理论参考阻力，单位 N；它不是实验标签。网络输出逐株修正系数，总阻力等于各株 `默认力 × 修正系数` 之和。实验标签仍为 JSONL 的 `measurements.FX_0`，训练使用原有的非负截断策略。

JSONL 的 angle 是转盘角 A。每株默认有相隔 120° 的三列叶片，传给求解器的相对来流角固定按以下顺序排列：

```text
第一列：(A + 60) % 360
第二列：(A + 180) % 360
第三列：(A + 300) % 360
```

例如 A=0° 对应 `[60,180,300]`，A=75° 对应 `[135,255,15]`。原程序的遮蔽处理与列序有关，不能重新排序，也不能按 120° 周期合并缓存。`total_force_n` 已包含三列叶片及一次圆柱力，直接使用该标量，不额外乘以 3 或每列片数，不重新投影 FX/FY。

## physical.yaml version 3

version 3 使用逐株 EI，`material.stiffness_unit: N*m^2` 声明其单位，删除统一的 `material.young_modulus_pa`。几何与流体参数对整次训练统一生效；EI 随植株变化。`geometry.cylinder_diameter_m` 保留当前实验的 `0.015 m`。

| 配置组 | 内容 | 单位与解释 |
|---|---|---|
| `geometry` | 圆柱直径与高度、叶片长度与厚度与宽度/高度、每列片数 | 长度均为 m；`blade_height_m` 是单片 h，不是整株叶片区高度 |
| `material` | `stiffness_unit` | 固定为 `N*m^2`；每株实际 EI 在 JSONL 中填写 |
| `fluid` | `water_density_kg_m3` | 水密度，kg/m³；真实流速从每条 JSONL 读取 |
| `drag` | 叶片与圆柱阻力系数 | 无量纲，是物理受力公式的输入，不是网络输出的逐株系数 |
| `solver` | 节点数、迭代上限、位移容差和遮蔽参数 | 节点数至少 5；位移容差为 m；遮蔽比例表示保留力的比例 |

矩形叶片截面的计算与单位如下：

```text
I = h × t³ / 12          # h、t 的单位为 m，I 的单位为 m⁴
E = EI / I              # EI 为 N·m²，因此 E 为 N/m²，也就是 Pa
```

适配器用每株 EI 换算 E，交给原求解器；神经网络和物理默认力使用同一份 EI。不会把 stiffness 当作旧 E 的倍数，也不会为 EI=1 自动套用旧材料模量。正式训练应核对实际几何和实测 EI。

`max_iterations: 3000` 保持原柔性模式；`0` 是原脚本的刚性参考分支，同时跳过柔性求解和遮蔽。软材料时，原程序可能将非零迭代上限提高到至少 3000，并将容差收紧到最多 1e-6 m。原 MATLAB 兼容接口的 H_soft/b 不影响默认力，不要求在 YAML 中填写。

不同角度、真实流速或 EI 分别求解；完全相同的物理参数和组合复用进程内缓存，EI 是缓存身份的一部分。求解未达到容差时，保留原程序的 `ConvergenceWarning` 告警与有限计算结果；非法参数或非有限结果会停止加载。默认力在 CPU 的数据加载阶段计算，不在每个训练 batch 中重复求解梁变形。

## 数据、镜像与 checkpoint 兼容边界

schema v3 要求每株显式填写正且有限的 `stiffness`；缺失、零、负数、NaN 和 Infinity 都会报错。v3 允许有限小数角度，并沿用连续旋转坐标校验。历史 v1/v2 可以兼容读取，在内存中补 `EI=1.0 N·m²`，原文件不改写。

当前 `datasets/` 递归全部 10 份 JSONL 和根目录 `dataset.jsonl` 已迁移为 v3，每株统一填 `1.0 N·m²`。`Experiment/output/dataset.jsonl` 与 `visualization/dataset_0.4.jsonl` 保留原字节。历史统一赋值用于明确格式，不能让网络学到跨 EI 规律，后续需要不同实测 EI 的训练数据。

训练镜像增强保留 EI，并重建镜像角的逐株特征、全局 sin/cos 和物理默认力。原求解器不保证 `F(A)=F(-A)`，不能只改角度而沿用原默认力。验证集和测试集不进行镜像增强。

新模型使用 checkpoint 版本 6。checkpoint 是保存权重、输入结构、标准化统计和物理参数的模型状态文件；训练恢复与评估拒绝旧版本，需要重新训练。历史数据的 v1/v2 读取兼容不等于旧权重兼容。评估恢复 checkpoint 中的完整物理配置，不因磁盘上的 YAML 后续改动而改变已训练模型的含义。

## 单独核对一个工况

在项目根目录启动 Python 后执行以下示例。参数集中在示例开头；角度为转盘 degree，流速为 m/s，刚度为 N·m²。

```python
from model.hydro.physics import load_physical_config
from model.hydro.calculated_force import column_angles_for_turntable

# 这四个变量是需要修改的运行参数，物理几何和求解设置仍由 YAML 提供。
ANGLE_DEG = 75.0
FLOW_SPEED_M_S = 0.2
STIFFNESS_NM2 = 1.0
PHYSICAL_CONFIG_PATH = "model/configs/physical.yaml"

# load_physical_config 接收 YAML 路径，返回包含几何、单位和求解器参数的配置对象。
physical = load_physical_config(PHYSICAL_CONFIG_PATH)

# column_angles_for_turntable 接收转盘角，返回按固定列序排列的三个相对来流角。
print(column_angles_for_turntable(ANGLE_DEG))

# single_drag_for_angle 的 angle、flow_speed、stiffness 分别是转盘角、真实流速、EI。
# 返回值为该孤立单株的默认阻力 N；version 3 必须显式提供 stiffness。
print(physical.single_drag_for_angle(ANGLE_DEG, FLOW_SPEED_M_S, stiffness=STIFFNESS_NM2))
```

运行参数应保存在 YAML 或 JSONL 中，无需为了调整材料去修改物理适配器和原求解器。
