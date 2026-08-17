"""按 ``model_id`` 分组的数据划分逻辑。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class GroupSplit:
    """一折训练/验证/测试索引。

    Attributes:
        fold: 从 0 开始的外层折编号。
        train_indices: 用于梯度更新的样本索引。
        validation_indices: 用于 early stopping 的样本索引。
        test_indices: 仅用于该折最终评估的样本索引。
    """

    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def build_group_kfold_splits(
    groups: np.ndarray | list[int],
    n_splits: int = 5,
    validation_fraction: float = 0.2,
    seed: int = 20260814,
) -> list[GroupSplit]:
    """创建外层 GroupKFold 与内层按 group 的固定验证划分。

    Args:
        groups: 每个样本的 ``model_id``；同一值绝不跨训练/验证/测试集合。
        n_splits: 外层交叉验证折数。
        validation_fraction: 从外层训练 groups 中抽出的验证比例。
        seed: 控制内层 group 洗牌的随机种子。

    Returns:
        长度为 ``n_splits`` 的 :class:`GroupSplit` 列表。
    """

    group_array = np.asarray(groups)
    if group_array.ndim != 1 or group_array.size == 0:
        raise ValueError("groups 必须是非空一维数组。")
    unique_groups = np.unique(group_array)
    if unique_groups.size < n_splits:
        raise ValueError("不同 model_id 的数量不能少于 n_splits。")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction 必须在 0 与 1 之间。")

    sample_indices = np.arange(group_array.size)
    outer_splitter = GroupKFold(n_splits=n_splits)
    results: list[GroupSplit] = []
    for fold, (outer_train, test_indices) in enumerate(
        outer_splitter.split(sample_indices, groups=group_array)
    ):
        outer_train_groups = np.unique(group_array[outer_train])
        rng = np.random.default_rng(seed + fold)
        shuffled_groups = rng.permutation(outer_train_groups)
        validation_group_count = max(
            1, int(round(shuffled_groups.size * validation_fraction))
        )
        validation_group_count = min(
            validation_group_count, shuffled_groups.size - 1
        )
        validation_groups = shuffled_groups[:validation_group_count]

        validation_mask = np.isin(group_array[outer_train], validation_groups)
        validation_indices = outer_train[validation_mask]
        train_indices = outer_train[~validation_mask]
        results.append(
            GroupSplit(
                fold=fold,
                train_indices=np.sort(train_indices),
                validation_indices=np.sort(validation_indices),
                test_indices=np.sort(test_indices),
            )
        )
    return results
