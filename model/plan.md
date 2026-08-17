可以。按照你现在确定的方案，我建议把整个工程目标固定成下面这个数学模型：

[
\boxed{
\text{Plant tokens}
\rightarrow
\text{Conditional Transformer}
\rightarrow
c_i
\rightarrow
\hat D=\sum_i c_iD_i^{(0)}
}
]

Transformer 内部坚持标准 softmax attention，但做三件定制化处理：

[
\boxed{
\text{2D RoPE on }Q,K
}
]

[
\boxed{
\text{relative geometry enters Value}
}
]

[
\boxed{
Re\text{ 等 global parameters condition LayerNorm and Value}
}
]

我建议工程上按 **“数据层 → 几何层 → attention → block → output → loss → tests → training”** 八层来实现。最重要的是一开始就保证每个模块可以单独关闭，方便以后做 ablation。

---

# 一、先把最终数学定义固定下来

一条样本包含 (N) 株水草。

第 (i) 株的位置：

[
p_i=(x_i,y_i).
]

已知单株阻力：

[
D_i^{(0)}.
]

全局物理参数：

[
g=[Re,\ldots].
]

Transformer 输出：

[
c_i.
]

最终：

[
\boxed{
\hat D_{\rm total}
==================

\sum_{i=1}^{N}c_iD_i^{(0)}
}
]

来流永远：

[
\mathbf U=U\hat y.
]

因此相对位置定义必须统一：

[
\boxed{
\Delta x_{ij}=x_j-x_i,\qquad
\Delta y_{ij}=y_j-y_i
}
]

其中：

* (i)：target plant；
* (j)：source plant；
* (\Delta y_{ij}<0)：(j) 在 (i) 上游；
* (\Delta y_{ij}>0)：(j) 在 (i) 下游。

这个符号约定要从数据处理、模型、可视化到论文始终不变。

---

# 二、建议的数据 Tensor 设计

对于 mini-batch，因为每个构型中的水草数量 (N) 不同，第一版建议用 padding。

设 batch size 为 (B)，batch 内最大植物数为 (N_{\max})。

输入：

```python
positions:   [B, Nmax, 2]
plant_feat:  [B, Nmax, F]
single_drag: [B, Nmax]
plant_mask:  [B, Nmax]       # bool

global_feat: [B, G]

target_drag: [B]
```

例如：

```text
positions[b, i] = [x_i, y_i]
single_drag[b, i] = D_i^0
global_feat[b] = [Re]
plant_mask[b, i] = True/False
```

padding plant 必须：

```python
plant_mask = False
single_drag = 0
```

这样最终：

[
\hat D
======

\sum_i
m_i c_iD_i^{(0)}
]

其中 (m_i\in{0,1})。

---

# 三、位置在进入网络之前先无量纲化

位置按照一格（相邻两株水草）间距离为1

Re 则建议首先：

[
\boxed{
r=\log Re
}
]

再做训练集统计标准化：

[
\tilde r
========

\frac{\log Re-\mu_{\log Re}}
{\sigma_{\log Re}}.
]

比直接输入 (Re) 数值通常稳定得多。

---

# 四、Plant token 的第一版设计

如果所有水草本身完全一样，我依然建议：

[
\boxed{
h_i^{(0)}=e_{\rm plant}
}
]

其中：

[
e_{\rm plant}\in\mathbb R^{d_{\rm model}}
]

是一个 learnable parameter。

例如：

```python
self.plant_token = nn.Parameter(torch.randn(d_model))
```

forward：

```python
H = self.plant_token[None, None, :].expand(B, N, -1)
```

---

如果不同水草有 intrinsic property，比如：

* (D_i^{(0)})
* diameter
* height
* stiffness
* species

则：

[
h_i^{(0)}
=========

e_{\rm plant}
+
\phi_{\rm plant}(z_i).
]

第一版如果所有植物完全 identical，**不要把 (x,y) concatenate 进 token**。

位置单独由：

1. RoPE；
2. relative-value；

处理。

这样模型结构非常干净。

---

# 五、Global parameter encoder
data中只有流速U，你需要做一个Re的计算器，里面用到的量作为参数。


不要在每个地方直接使用裸 (Re)。

先统一做：

[
g
=

\phi_g(\widetilde{\log Re}).
]

例如：

```python
GlobalEncoder(
    input_dim=G,
    hidden_dim=64,
    output_dim=d_model,
)
```

产生：

```python
g: [B, d_model]
```

推荐：

```text
Linear(G → 64)
SiLU
Linear(64 → d_model)
```

之后整个网络里的 Re conditioning 全部从这个 (g) 来。

这样以后增加：

* Reynolds number
* inlet velocity
* viscosity
* vegetation flexibility

只需要修改 GlobalEncoder。

---

# 六、Conditional LayerNorm 是一个独立模块

使用0初始化的adaLN

---

# 七、你的核心 Attention 模块

这会是整个项目最核心的 class。

建议命名：

```python
HydroMultiHeadAttention
```

输入：

```python
H:        [B, N, D]
positions:[B, N, 2]
g:        [B, Dg]
mask:     [B, N]
```

输出：

```python
M:        [B, N, D]
attention:[B, H, N, N]   # optional
```

---

# 八、标准 Q、K、V

设：

[
D=d_{\rm model}.
]

attention heads：

[
H=n_{\rm heads}.
]

每 head：

[
d_h=\frac{D}{H}.
]

例如第一版：

[
D=128,
\qquad
H=8,
\qquad
d_h=16.
]

不过因为你要做 2D RoPE，我更建议：

[
\boxed{
d_h\text{ 是 }4\text{ 的倍数}
}
]

例如：

[
D=256,\quad H=8,\quad d_h=32.
]

这是一个很舒服的配置。

计算：

[
Q=HW_Q,
]

[
K=HW_K,
]

[
V=HW_V.
]

Tensor：

```python
Q, K, V:
[B, n_heads, N, d_head]
```

---

# 九、2D RoPE

对：

[
Q,K
]

做 RoPE。

不对 V 做 RoPE。

概念上：

[
Q'
==

RoPE_{2D}(Q,x,y),
]

[
K'
==

RoPE_{2D}(K,x,y).
]

然后标准 attention score：

[
\boxed{
S_{ij}^{h}
==========

\frac{
{q_i^{h}}^T k_j^{h}
}{
\sqrt{d_h}
}
}
]

注意这里 (q,k) 已经经过 RoPE。

---

## 2D RoPE 的 channel 分法

比如：

[
d_h=32.
]

可以：

* 前 16 维编码 (x)
* 后 16 维编码 (y)

每一部分再按二维 rotation pair 处理。

因此：

```text
head dim = 32

0:16   → x RoPE
16:32  → y RoPE
```

所以工程上要求：

[
d_h % 4 = 0.
]

直接在初始化：

```python
assert d_head % 4 == 0
```

---

# 十、Relative Geometry Tensor

一次性计算所有 pair：

[
r_{ij}
======

[
\Delta x_{ij},
\Delta y_{ij}
].
]

Tensor shape：

```python
relative_pos: [B, N, N, 2]
```

PyTorch 大概：

```python
target = positions[:, :, None, :]   # B,N,1,2
source = positions[:, None, :, :]   # B,1,N,2

relative_pos = source - target
```

于是：

```python
relative_pos[b, i, j]
```

严格等于：

[
p_j-p_i.
]

一定要写 unit test 检查这个方向。

---

# 十一、Relative Geometry Encoder

建议单独一个 class：

```python
RelativeGeometryEncoder
```

第一版输入我建议：

[
z_{ij}
======

[
\Delta x,
\Delta y,
r
].
]

其中：

[
r=\sqrt{\Delta x^2+\Delta y^2}.
]

虽然理论上 (r) 可以由前两个算出来，但显式给网络通常没有坏处。

甚至可以：

[
z=
[
\Delta x,
\Delta y,
|\Delta x|,
|\Delta y|,
r
].
]

但第一版我会克制一些，只用：

[
\boxed{
[\Delta x,\Delta y,r]
}
]

保留 signed (\Delta y) 是必须的。

---

encoder：

```text
3
↓
Linear(3, 64)
SiLU
Linear(64, 64)
SiLU
Linear(64, n_heads * d_head)
```

输出：

```python
R_v: [B, H, N, N, d_head]
```

也就是：

[
R_{ij}^{V,h}.
]

---

# 十二、一定要处理 self-edge

对于：

[
i=j,
]

有：

[
\Delta x=0,\qquad
\Delta y=0.
]

我建议：

[
\boxed{
R_{ii}^{V}=0
}
]

而不是让模型学习一个任意 self-relative-value。

因为 self 信息本身已经在：

[
V_i
]

里面。

这样：

[
V_i+R_{ii}^V=V_i.
]

代码：

```python
eye = torch.eye(N, device=x.device, dtype=torch.bool)

R_v = R_v.masked_fill(
    eye[None, None, :, :, None],
    0.0
)
```

这是一个非常值得明确固定的设计。

---



# 十四、我甚至建议 Re 同时 condition relative Value

因为实际上水动力 interaction kernel 很可能满足：

[
K=
K(\Delta x,\Delta y,Re),
]

而不是：

[
K=K(\Delta x,\Delta y).
]

所以最终可以做成：

[
R_{ij}^{V}
==========

\phi_R(
\Delta x_{ij},
\Delta y_{ij};
g
).
]

工程上不要第一版直接 concatenate：

```python
[dx, dy, Re]
```

到每一个 pair。

更干净的方法也是 FiLM：

先：

[
R_{ij}^{0}
==========

\phi_{\rm geo}(\Delta x,\Delta y,r).
]

再：

[
\boxed{
R_{ij}^{V}
==========

(1+\gamma_R(g))
\odot R_{ij}^{0}
+
\beta_R(g)
}
]

这样：

```text
geometry → spatial interaction pattern
Re       → modify interaction regime
```

物理解释比较漂亮。

---

# 十五、最终 Value 是 pair-dependent 的

这是最重要的一步：

传统 Transformer：

[
V_j.
]

你的模型：

[
\boxed{
V_{ij}
======

\widetilde V_j
+
R_{ij}^{V}.
}
]

注意现在 Value 已经不是：

```python
[B,H,N,Dh]
```

这么简单。

pair-dependent Value 实际上是：

```python
[B,H,N,N,Dh]
```

其中：

* 第一个 (N)：target (i)
* 第二个 (N)：source (j)

---

# 十六、attention score 仍然完全标准

你已经确定要 standard softmax Transformer。

那么：

[
S_{ij}
======

\frac{q_i^Tk_j}{\sqrt{d_h}}.
]

invalid padding key：

[
S_{ij}=-\infty.
]

然后：

[
\boxed{
A_{ij}
======

\operatorname{softmax}*j(S*{ij})
}
]

没有 relative bias。

没有修改 normalization。

没有 sigmoid。

保持干净。

---

# 十七、最终 Message

你的关键公式就是：

[
\boxed{
M_i^h
=====

\sum_j
A_{ij}^h
\left(
\widetilde V_j^h+
R_{ij}^{V,h}
\right)
}
]

可以拆成：

[
M_i^h
=====

\underbrace{
\sum_jA_{ij}^h\widetilde V_j^h
}*{standard\ attention}
+
\underbrace{
\sum_jA*{ij}^hR_{ij}^{V,h}
}_{relative\ geometry}
.
]

工程上我其实建议就按照这两个 term 分开算。

而不是构造巨大的：

```python
V_pair[B,H,N,N,Dh]
```

因为内存会大很多。

直接：

```python
content_message = einsum(
    "bhij,bhjd->bhid",
    attention,
    V
)

relative_message = einsum(
    "bhij,bhijd->bhid",
    attention,
    R_v
)

message = content_message + relative_message
```

这会清晰很多。

---

# 十八、这一点对显存非常重要

注意：

[
R_v
]

已经是：

[
B\times H\times N^2\times d_h
]

复杂度。

如果：

[
B=32,\quad
H=8,\quad
N=200,\quad
d_h=32,
]

它就已经非常大。

所以从一开始就要记录：

[
N_{\max}.
]

如果以后：

[
N<100
]

基本不用太担心。

如果可能：

[
N=500\sim1000,
]

这个设计后面就必须做 chunking / sparse interaction。

第一版先不用优化，但模块接口要允许以后改。

---

# 十九、一个完整 Transformer Block

我推荐 Pre-LN：

[
X=H^{(l)}.
]

首先：

[
Z=
CLN_1(X,g).
]

attention：

[
M=
HydroAttention(Z,p,g).
]

residual：

[
X'=X+\operatorname{Dropout}(M).
]

然后：

[
Z'=CLN_2(X',g).
]

FFN：

[
F=FFN(Z').
]

最后：

[
\boxed{
H^{(l+1)}
=========

X'+F
}
]

代码概念：

```python
class HydroTransformerBlock(nn.Module):

    def forward(self, x, pos, global_feat, mask):

        z = self.norm1(x, global_feat)

        attn_out = self.attn(
            z,
            pos,
            global_feat,
            mask,
        )

        x = x + self.dropout(attn_out)

        z = self.norm2(x, global_feat)

        x = x + self.ffn(z)

        return x
```

不要修改 FFN。

保持：

```text
Linear(D, 4D)
GELU / SiLU
Dropout
Linear(4D, D)
```

即可。

---

# 二十、第一版超参数

我建议不要上来搞大模型。

第一版：

```yaml
model:
  d_model: 256
  n_heads: 8
  d_head: 32

  n_layers: 4

  ffn_ratio: 4

  dropout: 0.05

  rope:
    enabled: true
    type: 2d

  relative_value:
    enabled: true
    hidden_dim: 64
    use_distance: true
    condition_global: true

  conditional_layernorm:
    enabled: true

  condition_value:
    enabled: true
```

这是一个很合理的起点。

之后再测：

[
L=2,4,6,8.
]

---

# 二十一、Coefficient Head

最后：

[
H^{(L)}
=======

[h_1,\ldots,h_N].
]

使用**同一个 MLP**处理所有水草：

[
z_i=f_{\rm head}(h_i).
]

推荐：

```text
256
↓
128
SiLU
↓
1
```

然后为了保证：

[
c_i>0
]

我第一版推荐：

[
\boxed{
c_i=\exp(z_i)
}
]

但有一个关键初始化：

让最后一层：

```python
weight = 0
bias = 0
```

那么网络刚初始化：

[
z_i=0,
]

因此：

[
\boxed{
c_i=1.
}
]

所以初始模型自动满足：

[
\hat D
======

\sum_iD_i^{(0)}.
]

也就是：

> 默认假定没有水草耦合。

这是非常好的 physics-informed initialization。

---

# 二十二、最终 forward

整个模型 forward 应该非常简单：

```python
def forward(
    positions,
    single_drag,
    global_features,
    plant_mask,
):

    H = self.build_tokens(...)

    g = self.global_encoder(global_features)

    for block in self.blocks:
        H = block(
            H,
            positions,
            g,
            plant_mask,
        )

    log_coeff = self.coeff_head(H).squeeze(-1)

    coeff = torch.exp(log_coeff)

    coeff = coeff * plant_mask

    total_drag = (
        coeff * single_drag
    ).sum(dim=1)

    return {
        "total_drag": total_drag,
        "coefficient": coeff,
        "log_coefficient": log_coeff,
    }
```

一定返回每株：

[
c_i
]

即使训练标签只有 total drag。

以后分析会非常需要它。

---

# 二十三、Loss 不建议直接对 raw total drag 做 MSE

这一点非常重要。

因为水草数量不同：

[
N=5
]

和：

[
N=100
]

总阻力天然不是一个量级。

直接：

[
(\hat D-D)^2
]

会让大 (N) 样本主导训练。

我建议定义：

[
D_{\rm iso}
===========

\sum_iD_i^{(0)}.
]

然后定义总体 interaction coefficient：

[
C
=

\frac{D_{\rm total}}{D_{\rm iso}}.
]

预测：

[
\hat C
======

\frac{\hat D}{D_{\rm iso}}.
]

loss：

[
\boxed{
\mathcal L
==========

(\hat C-C)^2
}
]

这样：

* 无 coupling：(C=1)
* shielding：(C<1)
* drag enhancement：(C>1)

非常容易解释。

---

# 二十四、建议同时监控 5 个 metric

训练过程中至少记录：

[
MAE_D
]

[
RMSE_D
]

[
MAPE_D
]

[
MAE_C
]

以及：

[
R^2.
]

另外分组统计：

```text
metric vs number of plants
metric vs Re
metric vs density
```

这一点以后会非常有用。

平均 test error 很可能掩盖真正的问题。

---

# 二十五、必须提前意识到 coefficient 的 identifiability

如果 supervision 只有：

[
D_{\rm total},
]

那么：

[
c_i
]

不是唯一可识别的。

比如两株相同植物：

[
c_1+c_2
]

确定，不意味着：

[
c_1,c_2
]

分别确定。

所以工程上：

[
c_i
]

可以作为 latent coefficient 使用。

但论文里不能在没有额外验证时直接说：

> 这个就是每株真实阻力 coefficient。

如果 CFD 能提供 individual force：

[
D_i
]

那价值非常高。

你可以增加：

[
c_i^{true}
==========

\frac{D_i}{D_i^{(0)}}.
]

做 auxiliary supervision。

如果拿不到，也没关系，但解释时需要谨慎。

---

# 二十六、项目目录我建议这样设计

```text
hydro_transformer/
│
├── configs/
│   ├── base.yaml
│   ├── ablation_no_rope.yaml
│   ├── ablation_no_relvalue.yaml
│   └── ablation_no_re_condition.yaml
│
├── data/
│   ├── dataset.py
│   ├── collate.py
│   ├── normalization.py
│   └── augmentation.py
│
├── models/
│   ├── model.py
│   │
│   ├── embeddings.py
│   ├── global_encoder.py
│   │
│   ├── conditional_norm.py
│   ├── rope_2d.py
│   ├── relative_geometry.py
│   │
│   ├── attention.py
│   ├── transformer_block.py
│   └── coefficient_head.py
│
├── training/
│   ├── losses.py
│   ├── metrics.py
│   ├── trainer.py
│   └── scheduler.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── ood.py
│   ├── visualize_coeff.py
│   ├── visualize_attention.py
│   └── visualize_relative_value.py
│
├── tests/
│   ├── test_relative_position.py
│   ├── test_rope.py
│   ├── test_attention.py
│   ├── test_masking.py
│   ├── test_permutation.py
│   ├── test_single_plant.py
│   └── test_forward_backward.py
│
├── train.py
└── evaluate.py
```

我会非常推荐这种模块化，而不是一个 `model.py` 写 800 行。

---

# 二十七、第一批 unit tests 比训练模型还重要

这个问题尤其容易发生“模型能训练，但 geometry 方向写反”的 bug。

必须测试这些。

### Test 1：relative position

设置：

[
p_i=(2,5)
]

[
p_j=(3,2).
]

必须得到：

[
\Delta x_{ij}=1
]

[
\Delta y_{ij}=-3.
]

即：

> j 在 i 上游。

---

### Test 2：permutation equivariance

随机排列 plant token 顺序。

总阻力必须：

[
\hat D_{\rm before}
===================

\hat D_{\rm after}
]

到数值误差范围。

每株：

[
c_i
]

应该跟着 permutation 同样排列。

这是一个非常关键的测试。

---

### Test 3：padding invariance

同一个构型：

```text
Nmax = 20
```

和：

```text
Nmax = 100
```

只增加 padding。

预测必须一样。

---

### Test 4：single plant

只有一株水草。

模型初始化时必须：

[
c_1=1.
]

并且：

[
\hat D=D_1^{(0)}.
]

---

### Test 5：constant-token degeneracy

所有初始 token 完全相同。

构造两个不同 geometry。

确认第一层输出：

[
H_A^{(1)}
\neq
H_B^{(1)}.
]

然后关闭 relative Value：

```python
relative_value=False
```

应该观察到这个区别显著消失。

这是直接验证你之前发现的问题。

---

# 二十八、还应该做 physics symmetry test

如果横向边界条件关于 (x) 对称，而来流是 (+\hat y)，那么：

[
(x_i,y_i)
\rightarrow
(-x_i,y_i)
]

理论上总阻力应该不变：

[
D(X)=D(\mathcal R_xX).
]

模型不一定 architecture-exact invariant，但可以测：

[
\Delta_{\rm symmetry}
=====================

\frac{
|\hat D(X)-\hat D(\mathcal R_xX)|
}{
D
}.
]

甚至可以将这种 reflection 加进 data augmentation。

---

# 二十九、Ablation 从工程第一天就预留

所有功能做成 config switch：

```yaml
use_rope: true
use_relative_value: true
use_conditional_ln: true
condition_value_on_global: true
condition_relative_value_on_global: true
```

最终至少跑：

| 模型 | RoPE | Rel-V | Cond-LN | Re→V |
| -- | ---: | ----: | ------: | ---: |
| A  |    ❌ |     ❌ |       ❌ |    ❌ |
| B  |    ✅ |     ❌ |       ❌ |    ❌ |
| C  |    ✅ |     ✅ |       ❌ |    ❌ |
| D  |    ✅ |     ✅ |       ✅ |    ❌ |
| E  |    ✅ |     ✅ |       ✅ |    ✅ |

模型 E 就是你的最终模型。

这样论文中的 architecture justification 会非常清楚。

---

# 三十、训练过程建议分阶段，而不是一上来跑完整数据

### Phase 0：Overfit 一个 batch

例如只拿：

[
32
]

条样本。

训练到：

[
\mathcal L\approx0.
]

如果做不到，先不要继续。

---

### Phase 1：小数据集

例如：

[
1000
]

条数据。

检查：

* loss 是否下降；
* (c_i) 是否数值稳定；
* attention 是否 NaN；
* Re conditioning 是否工作；
* padding 是否正确。

---

### Phase 2：完整训练

之后再跑完整数据。

---

# 三十一、优化器和训练初始设置

我会从：

```yaml
optimizer:
  type: AdamW
  lr: 3e-4
  weight_decay: 1e-4

scheduler:
  warmup_steps: 1000
  type: cosine

gradient_clip:
  max_norm: 1.0
```

开始。

batch size 尽量由：

[
N^2
]

显存决定。

而不是固定追求很大 batch。

---

# 三十二、一定记录 gradient norm

因为你的 relative Value 路径：

[
R_{ij}^V
]

规模比较大。

建议 TensorBoard / WandB 记录：

```text
train/loss
train/grad_norm

coeff/mean
coeff/std
coeff/min
coeff/max

attention/entropy

relative_value/norm

conditioning/gamma_ln_norm
conditioning/gamma_v_norm
```

如果突然：

[
c_i=e^{z_i}
]

爆炸，你会立刻看到。

---

# 三十三、coefficients 最好做安全保护

训练时不要真的：

```python
torch.exp(log_coeff)
```

任其无限增长。

第一版可以：

```python
log_coeff = torch.clamp(
    log_coeff,
    min=-5,
    max=5,
)
```

然后：

[
c_i=e^{z_i}.
]

或者等训练稳定后再去掉 clamp。

如果你知道实际：

[
0<c_i<2
]

甚至可以直接：

[
c_i=2\sigma(z_i).
]

这个要根据 CFD 数据来决定。

---

# 三十四、最重要的 OOD split 不要随机切

你这个项目如果只是随机：

```text
80% train
10% val
10% test
```

很容易高估效果。

我建议至少建立：

### IID test

随机 geometry。

### Density OOD

训练：

[
N\le30
]

测试：

[
N>30.
]

### Geometry OOD

训练 random。

测试：

* aligned
* staggered
* clustered
* channels
* regular lattice

### Reynolds OOD

训练：

[
Re\in[a,b]
]

测试稍微超出：

[
Re>b.
]

真正 AI for Science 的价值主要在这些测试里。

---

# 三十五、relative Value 是你最值得分析的对象

训练完成以后，不要只画 attention。

因为你真正向网络显式提供的 coupling mechanism 是：

[
R^V(\Delta x,\Delta y).
]

建议取固定：

[
Re
]

然后扫描：

[
\Delta x,\Delta y.
]

计算：

[
|R^V_h(\Delta x,\Delta y)|
]

对于不同 head 做二维 heatmap。

然后改变 Re：

[
Re_1,Re_2,Re_3
]

重新画。

如果结构合理，你可能会直接观察到：

* upstream/downstream asymmetry；
* wake-like elongated region；
* cross-stream decay；
* interaction length 随 Re 改变。

这个结果可能会成为项目最有意思的 physics visualization。

---

# 三十六、整个工程我建议分成 7 个里程碑

只有这一份实施顺序我会严格遵守：

1. **数据和约定冻结**：确定 (p_j-p_i) 的符号、无量纲化、mask、train/val/OOD split。
2. **实现基础模块**：`ConditionalLayerNorm`、`RoPE2D`、`RelativeGeometryEncoder`，全部独立 unit test。
3. **实现 HydroAttention**：验证 shape、mask、softmax、relative-value，并专门测试“constant V 不退化”。
4. **完整 HydroTransformer**：4 blocks + coefficient head，初始化满足 (c_i=1)，先 overfit 32 条数据。
5. **正式训练与消融**：Vanilla → +RoPE → +Rel-V → +Cond-LN → +Re-conditioned V。
6. **OOD 和物理验证**：plant number、density、geometry、Re、reflection/permutation consistency。
7. **科学分析**：画 (c_i)、attention、(R^V(\Delta x,\Delta y;Re))，判断模型究竟学到了什么 coupling structure。

---

## 我最终建议的第一版 forward 数据流

```text
                    Re
                     │
                     ▼
             Global Encoder
                     │
                     g
          ┌──────────┼─────────────┐
          │          │             │
          ▼          ▼             ▼
   Conditional    Value FiLM   Rel-Value FiLM
    LayerNorm
          │
          │
Plant ────┴──────► H
positions            │
  │                  │
  │              Q   K   V
  │              │   │   │
  ├──► 2D RoPE ──┘   └───┐
  │                        │
  │                        │
  └─► Δx, Δy ─► RelGeom ─► R_ij^V
                           │
Q_rope K_rope              │
   │     │                 │
   └──┬──┘                 │
      ▼                    │
 QKᵀ / √d                  │
      │                    │
   softmax                 │
      │                    │
      A_ij                 │
      │                    │
      ├────────────┐       │
      ▼            ▼       ▼
   Σ A_ij V_j   Σ A_ij R_ij^V
      │            │
      └──────┬─────┘
             ▼
          Message
             │
         Residual
             │
            FFN
             │
         × 4 blocks
             │
             ▼
        shared MLP
             │
          log c_i
             │
          exp(.)
             │
             c_i
             │
             ▼
       Σ c_i D_i^(0)
             │
             ▼
         Total Drag
```

如果进入实际编码阶段，我建议严格先实现 **`ConditionalLayerNorm → RoPE2D → RelativeGeometryEncoder → HydroMultiHeadAttention`** 这四个最底层组件，再组装 Transformer；不要直接从整个模型开始写。这样调试成本会低很多。
