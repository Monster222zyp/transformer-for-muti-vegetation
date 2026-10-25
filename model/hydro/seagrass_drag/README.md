# 原水草阻力求解器

本目录保存用户提供的 `Matlab-seagrass/python/src/seagrass_drag` v0.1.0
六个运行模块的副本：`__init__.py`、`api.py`、`defaults.py`、`types.py`、
`model.py`、`numerics.py`。它们复刻原 `calculate_drag_coefficient_v2.m`
及辅助函数。集成时未改受力公式、求解流程、遮蔽规则或数值默认值。
没有复制命令行入口、构建产物或原包的实验数据。

使用项目内副本使训练无需依赖个人 Downloads 路径，也不需要 MATLAB。
运行依赖 NumPy 和 SciPy，已列入 `model/requirements.txt`。
项目角度转换和参数校验位于上一级 `calculated_force.py`，修改实际几何、材料
和求解参数应使用 `model/configs/physical.yaml`，不编辑此目录。

`predict_drag` 一次接收三列角度，返回 `DragResult`。其中 `total_force_n`
是三列叶片与一次圆柱的整株力；`blade_force_n` 和 `cylinder_force_n` 是分量；
`columns` 是逐列代表叶片的变形与残差；`converged` 表示迭代是否达到容差。
项目默认直接使用 total_force_n，原收敛告警仍可见。

算法对第二列实施遮蔽，其角差采用原程序的差值绝对值，不改成最短圆周角差。
这会使列序影响结果，也不保证旋转 120° 或镜像后默认力完全相同。
本目录没有复制 MATLAB 主绘图脚本的额外力投影与 F_pre_total，采用的是
原核心求解函数 F_total。

`model/tests/fixtures/seagrass_matlab_force_reference.json` 摘录原包中由
MATLAB R2026a 独立运行生成的 33 个案例的参数、总力和最大变形；不是用
Python 输出生成的期望值。回归测试同时覆盖这些案例和项目转盘角适配。
