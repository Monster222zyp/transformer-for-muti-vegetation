"""四种训练、验证与测试划分模式的专项单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from model.training.splits import (
    build_cross_validation_splits,
    build_group_kfold_splits,
)


def _assert_complete_partition(
    splits: list,
    n_samples: int,
) -> None:
    """验证每折集合互斥、非空，且所有 outer-test 覆盖全部样本一次。

    Args:
        splits: 被测试函数返回的各折划分对象。
        n_samples: 预期被覆盖的样本总数。
    """

    all_test_indices: list[int] = []
    expected_indices = set(range(n_samples))
    for split in splits:
        train = set(split.train_indices.tolist())
        validation = set(split.validation_indices.tolist())
        test = set(split.test_indices.tolist())
        assert train
        assert validation
        assert test
        assert train.isdisjoint(validation)
        assert train.isdisjoint(test)
        assert validation.isdisjoint(test)
        assert train | validation | test == expected_indices
        all_test_indices.extend(split.test_indices.tolist())

    assert sorted(all_test_indices) == list(range(n_samples))


def _assert_same_splits(first: list, second: list) -> None:
    """验证两次调用得到的全部索引完全相同。

    Args:
        first: 第一次调用得到的划分列表。
        second: 第二次调用得到的划分列表。
    """

    assert len(first) == len(second)
    for split_a, split_b in zip(first, second):
        assert np.array_equal(split_a.train_indices, split_b.train_indices)
        assert np.array_equal(
            split_a.validation_indices, split_b.validation_indices
        )
        assert np.array_equal(split_a.test_indices, split_b.test_indices)


def test_sample_mode_is_balanced_deterministic_and_covers_all_samples() -> None:
    """逐样本模式应随机且均衡地分配测试 fold，并可由 seed 完整复现。"""

    parameters = {
        "split_mode": "sample",
        "n_samples": 23,
        "n_splits": 5,
        "validation_fraction": 0.2,
        "seed": 31,
    }
    first = build_cross_validation_splits(**parameters)
    second = build_cross_validation_splits(**parameters)

    _assert_complete_partition(first, n_samples=23)
    _assert_same_splits(first, second)
    test_sizes = [split.test_indices.size for split in first]
    assert max(test_sizes) - min(test_sizes) <= 1


def test_model_mode_keeps_each_model_in_exactly_one_set_per_fold() -> None:
    """model 模式不得让同一个 model_id 跨越训练、验证和测试集合。"""

    model_ids = np.repeat(np.arange(11), [2, 4, 3, 6, 2, 5, 3, 4, 2, 5, 3])
    first = build_cross_validation_splits(
        split_mode="model",
        n_samples=model_ids.size,
        model_ids=model_ids,
        n_splits=5,
        seed=47,
    )
    second = build_cross_validation_splits(
        split_mode="model",
        n_samples=model_ids.size,
        model_ids=model_ids,
        n_splits=5,
        seed=47,
    )

    _assert_complete_partition(first, n_samples=model_ids.size)
    _assert_same_splits(first, second)
    for split in first:
        train_groups = set(model_ids[split.train_indices])
        validation_groups = set(model_ids[split.validation_indices])
        test_groups = set(model_ids[split.test_indices])
        assert train_groups.isdisjoint(validation_groups)
        assert train_groups.isdisjoint(test_groups)
        assert validation_groups.isdisjoint(test_groups)

    # 贪心分配后的测试集差距不应大于最大的单个组，作为“尽量平衡”的边界。
    test_sizes = [split.test_indices.size for split in first]
    largest_group_size = int(np.unique(model_ids, return_counts=True)[1].max())
    assert max(test_sizes) - min(test_sizes) <= largest_group_size


def test_plant_count_mode_keeps_equal_counts_in_one_set_per_fold() -> None:
    """plant_count 模式不得拆散水草根数相同的样本。"""

    plant_counts = np.repeat([4, 5, 6, 8, 10, 12], [7, 3, 5, 8, 4, 6])
    splits = build_cross_validation_splits(
        split_mode="plant_count",
        n_samples=plant_counts.size,
        plant_counts=plant_counts,
        n_splits=5,
        validation_fraction=0.25,
        seed=59,
    )

    _assert_complete_partition(splits, n_samples=plant_counts.size)
    for split in splits:
        train_groups = set(plant_counts[split.train_indices])
        validation_groups = set(plant_counts[split.validation_indices])
        test_groups = set(plant_counts[split.test_indices])
        assert train_groups.isdisjoint(validation_groups)
        assert train_groups.isdisjoint(test_groups)
        assert validation_groups.isdisjoint(test_groups)


def test_flow_speed_mode_uses_fixed_train_and_overlapping_holdout() -> None:
    """flow_speed 应训练前三档，并让完整 0.4 集合同时承担两个评估角色。"""

    flow_speeds = np.asarray(
        [0.1, 0.4, 0.2, 0.3 + 5.0e-9, 0.4 - 5.0e-9, 0.1],
        dtype=np.float64,
    )
    splits = build_cross_validation_splits(
        split_mode="flow_speed",
        n_samples=flow_speeds.size,
        flow_speeds=flow_speeds,
        # 下面三个参数在固定流速协议中均不参与结果，故意传入普通模式不允许的值，
        # 验证它们不会错误阻止 flow_speed 的单折划分。
        n_splits=1,
        validation_fraction=2.0,
        seed=-999,
    )

    assert len(splits) == 1
    split = splits[0]
    assert split.fold == 0
    assert np.array_equal(split.train_indices, np.asarray([0, 2, 3, 5]))
    assert np.array_equal(split.validation_indices, np.asarray([1, 4]))
    assert np.array_equal(split.test_indices, split.validation_indices)
    assert set(split.train_indices).isdisjoint(split.validation_indices)


def test_flow_speed_mode_is_independent_of_cv_parameters() -> None:
    """固定流速划分不应随折数、验证比例或随机种子变化。"""

    flow_speeds = np.asarray([0.4, 0.3, 0.1, 0.2, 0.4], dtype=np.float64)
    first = build_cross_validation_splits(
        split_mode="flow_speed",
        n_samples=flow_speeds.size,
        flow_speeds=flow_speeds,
        n_splits=2,
        validation_fraction=0.1,
        seed=1,
    )
    second = build_cross_validation_splits(
        split_mode="flow_speed",
        n_samples=flow_speeds.size,
        flow_speeds=flow_speeds,
        n_splits=99,
        validation_fraction=0.9,
        seed=999,
    )

    _assert_same_splits(first, second)


@pytest.mark.parametrize(
    ("flow_speeds", "n_samples", "error_message"),
    [
        (None, 4, "必须提供 flow_speeds"),
        ([0.1, 0.2, 0.3, 0.4], 5, "长度必须等于 n_samples"),
        ([0.1, 0.2, 0.3, 0.4, 0.5], 5, "只支持 0.1、0.2、0.3、0.4"),
        ([0.1, 0.2, 0.3, 0.4, np.nan], 5, "NaN 或无穷大"),
        ([0.1, 0.2, 0.3, 0.4, np.inf], 5, "NaN 或无穷大"),
    ],
)
def test_flow_speed_mode_rejects_invalid_labels(
    flow_speeds: list[float] | None,
    n_samples: int,
    error_message: str,
) -> None:
    """缺失、错长、非有限或协议外的流速标签必须立即报错。"""

    with pytest.raises(ValueError, match=error_message):
        build_cross_validation_splits(
            split_mode="flow_speed",
            n_samples=n_samples,
            flow_speeds=flow_speeds,
        )


@pytest.mark.parametrize(
    "flow_speeds",
    [
        [0.1, 0.3, 0.4],
        [0.1, 0.2, 0.3],
        [0.4, 0.4, 0.4],
    ],
)
def test_flow_speed_mode_requires_every_protocol_speed(
    flow_speeds: list[float],
) -> None:
    """四档固定速度缺少任意一档时不能悄悄改变实验协议。"""

    with pytest.raises(ValueError, match="每档至少有一个样本"):
        build_cross_validation_splits(
            split_mode="flow_speed",
            n_samples=len(flow_speeds),
            flow_speeds=flow_speeds,
        )


def test_legacy_group_function_remains_compatible() -> None:
    """旧入口应继续接受原参数，并与统一入口的 model 模式得到相同结果。"""

    model_ids = np.repeat(np.arange(8), 3)
    legacy = build_group_kfold_splits(model_ids, n_splits=4, seed=71)
    unified = build_cross_validation_splits(
        split_mode="model",
        n_samples=model_ids.size,
        model_ids=model_ids,
        n_splits=4,
        seed=71,
    )

    _assert_same_splits(legacy, unified)


@pytest.mark.parametrize("split_mode", ["unknown", "models", ""])
def test_invalid_split_mode_has_clear_error(split_mode: str) -> None:
    """拼错模式名时应列出合法选项，而不是静默回退到某种划分。"""

    with pytest.raises(ValueError, match="未知 split_mode"):
        build_cross_validation_splits(
            split_mode=split_mode,
            n_samples=12,
            n_splits=3,
        )


def test_group_mode_rejects_missing_or_insufficient_groups() -> None:
    """缺少标签、组数少于 folds 或无法拆出验证组时应明确报错。"""

    with pytest.raises(ValueError, match="必须提供 model_ids"):
        build_cross_validation_splits(
            split_mode="model",
            n_samples=12,
            n_splits=3,
        )
    with pytest.raises(ValueError, match="独立分组数"):
        build_cross_validation_splits(
            split_mode="plant_count",
            n_samples=12,
            plant_counts=np.repeat([4, 5], 6),
            n_splits=3,
        )
    with pytest.raises(ValueError, match="outer-train 少于两个独立分组"):
        build_cross_validation_splits(
            split_mode="model",
            n_samples=12,
            model_ids=np.repeat([1, 2], 6),
            n_splits=2,
        )


def test_sample_mode_rejects_too_few_outer_train_samples() -> None:
    """仅两个样本做二折时没有空间再拆验证集，必须提示用户调整参数。"""

    with pytest.raises(ValueError, match="outer-train 少于两个样本"):
        build_cross_validation_splits(
            split_mode="sample",
            n_samples=2,
            n_splits=2,
        )
