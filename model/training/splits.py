"""训练、验证与测试集合的可复现交叉验证划分逻辑。

本模块支持四种划分方式：逐样本、按 ``model_id`` 分组、按水草根数分组，以及
固定按流速划分。前三种模式会保持训练、验证和测试集合互斥；``flow_speed`` 模式
用于特定的跨流速实验，会把 0.4 m/s 的完整数据同时用于 validation 和 test。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FLOW_SPEED_SPLIT_MODE = "flow_speed"
"""固定跨流速实验在配置文件中使用的模式名称。"""

SUPPORTED_SPLIT_MODES = ("sample", "model", "plant_count", FLOW_SPEED_SPLIT_MODE)
"""允许写入配置文件的四种划分模式名称。"""

# ``flow_speed`` 是一个固定实验协议，不从 YAML 接收可变速度列表。把所有硬编码
# 数值集中放在模块开头，可以让数据契约和浮点匹配策略容易审计与修改。
FLOW_SPEED_TRAIN_VALUES = (0.1, 0.2, 0.3)
"""固定进入训练集的流速，单位为 m/s。"""

FLOW_SPEED_VALIDATION_TEST_VALUE = 0.4
"""同时进入 validation 和 test 的固定流速，单位为 m/s。"""

FLOW_SPEED_MATCH_RTOL = 0.0
"""流速匹配的相对容差；设为 0，避免容差随数值大小改变。"""

FLOW_SPEED_MATCH_ATOL = 1.0e-8
"""流速匹配的绝对容差，用于吸收 CSV 浮点解析产生的微小误差。"""


@dataclass(frozen=True)
class GroupSplit:
    """一折训练/验证/测试索引。

    Attributes:
        fold: 从 0 开始的外层折编号。
        train_indices: 用于梯度更新的样本索引。
        validation_indices: 用于 early stopping（提前停止）的样本索引。
        test_indices: 仅用于该折最终评估的样本索引。
    """

    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def _validate_common_arguments(
    n_samples: int,
    n_splits: int,
    validation_fraction: float,
) -> None:
    """检查三种模式共用的数量与比例参数。

    Args:
        n_samples: 数据集中的样本总数。
        n_splits: 外层交叉验证的 fold 数。
        validation_fraction: 从每个 outer-train 中划给验证集的比例。

    Raises:
        ValueError: 参数无法产生非空的训练、验证、测试集合时抛出。
    """

    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples 必须是正整数。")
    if not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError("n_splits 必须是大于或等于 2 的整数。")
    if n_samples < n_splits:
        raise ValueError("样本数不能少于 n_splits，否则无法保证每个测试集非空。")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction 必须在 0 与 1 之间。")


def _as_group_array(
    values: np.ndarray | list[int] | list[float] | None,
    *,
    argument_name: str,
    n_samples: int,
) -> np.ndarray:
    """把分组标签转换为经过长度检查的一维 NumPy 数组。

    Args:
        values: 每个样本对应的分组标签；例如 ``model_id`` 或水草根数。
        argument_name: 用于异常信息的参数名称。
        n_samples: 期望的标签数量，必须与样本总数相同。

    Returns:
        包含 ``n_samples`` 个标签的一维 NumPy 数组。

    Raises:
        ValueError: 标签缺失、不是一维数组，或标签数与样本数不一致时抛出。
    """

    if values is None:
        raise ValueError(f"当前划分模式必须提供 {argument_name}。")
    result = np.asarray(values)
    if result.ndim != 1 or result.size != n_samples:
        raise ValueError(
            f"{argument_name} 必须是一维数组，且长度必须等于 n_samples。"
        )
    return result


def _build_flow_speed_split(
    flow_speeds: np.ndarray | list[float] | None,
    n_samples: int,
) -> list[GroupSplit]:
    """按固定流速协议创建唯一一折划分。

    该实验把 0.1、0.2、0.3 m/s 的全部样本用于梯度更新，把 0.4 m/s 的全部样本
    同时用于 early stopping 和最终指标计算。validation 与 test 因此故意重叠，
    test 指标不能解释为完全独立的 held-out 泛化指标。

    Args:
        flow_speeds: 每条样本的流速，单位为 m/s，长度必须等于 ``n_samples``。
        n_samples: 数据集样本总数，必须是正整数。

    Returns:
        只包含 ``fold=0`` 的列表；训练集与 0.4 m/s 集合互斥，而 validation 和
        test 保存相同的 0.4 m/s 样本索引。

    Raises:
        ValueError: 样本数或流速数组无效、出现协议外流速，或者任一必需集合为空。
    """

    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples 必须是正整数。")
    if flow_speeds is None:
        raise ValueError("flow_speed 模式必须提供 flow_speeds。")

    # 先转换为 float64 再匹配，保证 Python float、NumPy 浮点和 CSV 解析值使用
    # 同一比较精度。字符串等无法解释为数值的输入会转换为清晰的 ValueError。
    try:
        speed_array = np.asarray(flow_speeds, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("flow_speeds 必须是一维有限数值数组。") from error
    if speed_array.ndim != 1 or speed_array.size != n_samples:
        raise ValueError(
            "flow_speeds 必须是一维数组，且长度必须等于 n_samples。"
        )
    if not np.all(np.isfinite(speed_array)):
        raise ValueError("flow_speeds 不能包含 NaN 或无穷大。")

    supported_values = np.asarray(
        (*FLOW_SPEED_TRAIN_VALUES, FLOW_SPEED_VALIDATION_TEST_VALUE),
        dtype=np.float64,
    )
    # 每一行表示一个样本与四个合法速度的匹配结果。容差远小于相邻速度差，正常
    # 数据只会匹配一个值；仍显式检查匹配数量，避免将异常数据静默丢弃或重复归类。
    matches = np.isclose(
        speed_array[:, np.newaxis],
        supported_values[np.newaxis, :],
        rtol=FLOW_SPEED_MATCH_RTOL,
        atol=FLOW_SPEED_MATCH_ATOL,
    )
    match_counts = matches.sum(axis=1)
    if np.any(match_counts != 1):
        invalid_values = np.unique(speed_array[match_counts != 1]).tolist()
        raise ValueError(
            "flow_speed 模式只支持 0.1、0.2、0.3、0.4 m/s；"
            f"发现无法唯一匹配的值：{invalid_values}。"
        )

    # 固定协议要求四档流速都真实存在。只检查“训练集合非空”会让缺少某一档训练
    # 速度的数据悄悄通过，从而改变用户指定的 0.1/0.2/0.3 联合训练实验。
    matched_counts_by_speed = matches.sum(axis=0)
    missing_values = supported_values[matched_counts_by_speed == 0].tolist()
    if missing_values:
        raise ValueError(
            "flow_speed 模式要求 0.1、0.2、0.3、0.4 m/s 每档至少有一个样本；"
            f"当前缺少：{missing_values}。"
        )

    train_mask = np.any(matches[:, : len(FLOW_SPEED_TRAIN_VALUES)], axis=1)
    validation_test_mask = matches[:, -1]
    train_indices = np.flatnonzero(train_mask).astype(np.int64, copy=False)
    validation_test_indices = np.flatnonzero(validation_test_mask).astype(
        np.int64, copy=False
    )
    if train_indices.size == 0:
        raise ValueError("flow_speed 模式的 0.1、0.2、0.3 m/s 训练集不能为空。")
    if validation_test_indices.size == 0:
        raise ValueError("flow_speed 模式的 0.4 m/s validation/test 集合不能为空。")

    return [
        GroupSplit(
            fold=0,
            train_indices=train_indices,
            validation_indices=validation_test_indices.copy(),
            test_indices=validation_test_indices.copy(),
        )
    ]


def _random_balanced_group_folds(
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[np.ndarray]:
    """将完整分组随机且近似均衡地放入外层 folds。

    算法先随机打乱组，随后优先处理样本较多的组，并把每个组放入当前样本数最少
    的 fold。随机打乱负责让 seed 真正影响结果；贪心放置负责避免某个 fold 因大组
    集中而明显过大。

    Args:
        groups: 每个样本的分组标签。
        n_splits: 需要生成的外层 fold 数。
        seed: 控制相同大小组的次序和并列 fold 选择的随机种子。

    Returns:
        长度为 ``n_splits`` 的列表；每个元素是该 fold 持有的完整组标签数组。

    Raises:
        ValueError: 独立分组数少于 ``n_splits`` 时抛出。
    """

    unique_groups, group_counts = np.unique(groups, return_counts=True)
    if unique_groups.size < n_splits:
        raise ValueError(
            f"独立分组数 {unique_groups.size} 少于 n_splits={n_splits}，"
            "无法保证每个测试集都包含完整分组。"
        )

    rng = np.random.default_rng(seed)
    # 先随机排列，再进行稳定的降序排序；样本数相同的组会保留随机顺序。
    randomized_indices = rng.permutation(unique_groups.size)
    ordered_indices = randomized_indices[
        np.argsort(-group_counts[randomized_indices], kind="stable")
    ]

    fold_groups: list[list[object]] = [[] for _ in range(n_splits)]
    fold_sample_counts = np.zeros(n_splits, dtype=np.int64)
    for group_index in ordered_indices:
        # 多个 fold 同为最小负载时随机选择，避免固定偏向编号较小的 fold。
        minimum_count = fold_sample_counts.min()
        candidate_folds = np.flatnonzero(fold_sample_counts == minimum_count)
        selected_fold = int(rng.choice(candidate_folds))
        fold_groups[selected_fold].append(unique_groups[group_index])
        fold_sample_counts[selected_fold] += int(group_counts[group_index])

    return [np.asarray(labels, dtype=groups.dtype) for labels in fold_groups]


def _choose_validation_groups(
    outer_train_groups: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> np.ndarray:
    """从 outer-train 中选择接近目标样本比例的完整验证组。

    Args:
        outer_train_groups: 仅包含 outer-train 样本的分组标签。
        validation_fraction: 希望验证集占 outer-train 的比例。
        seed: 控制候选组并列时优先级的随机种子。

    Returns:
        应完整放入验证集的组标签。

    Raises:
        ValueError: outer-train 少于两个独立组，无法同时生成训练集和验证集时抛出。
    """

    unique_groups, group_counts = np.unique(
        outer_train_groups, return_counts=True
    )
    if unique_groups.size < 2:
        raise ValueError(
            "分组划分后 outer-train 少于两个独立分组，"
            "无法同时生成非空训练集和验证集；请减少 n_splits 或增加分组数。"
        )

    rng = np.random.default_rng(seed)
    randomized_indices = list(rng.permutation(unique_groups.size))
    target_sample_count = outer_train_groups.size * validation_fraction
    selected_indices: list[int] = []
    current_sample_count = 0

    # 每轮选择使验证样本数最接近目标的组；最多选择 m-1 组，以保留非空训练集。
    while randomized_indices and len(selected_indices) < unique_groups.size - 1:
        previous_distance = abs(current_sample_count - target_sample_count)
        best_position = min(
            range(len(randomized_indices)),
            key=lambda position: abs(
                current_sample_count
                + int(group_counts[randomized_indices[position]])
                - target_sample_count
            ),
        )
        candidate_index = randomized_indices[best_position]
        candidate_count = current_sample_count + int(group_counts[candidate_index])
        candidate_distance = abs(candidate_count - target_sample_count)

        # 第一个组必须选中以保证验证集非空；之后仅在更接近目标时继续增加组。
        if selected_indices and candidate_distance >= previous_distance:
            break
        selected_indices.append(candidate_index)
        current_sample_count = candidate_count
        randomized_indices.pop(best_position)

    return unique_groups[np.asarray(selected_indices, dtype=np.int64)]


def _build_sample_splits(
    n_samples: int,
    n_splits: int,
    validation_fraction: float,
    seed: int,
) -> list[GroupSplit]:
    """按独立样本随机生成外层测试 fold 和内层验证集。

    Args:
        n_samples: 数据集样本总数。
        n_splits: 外层交叉验证折数。
        validation_fraction: 验证集占每个 outer-train 的比例。
        seed: 控制外层和各折内层洗牌的随机种子。

    Returns:
        每个样本恰好作为一次测试样本的划分列表。

    Raises:
        ValueError: 某折的 outer-train 少于两个样本时抛出。
    """

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(n_samples)
    test_folds = np.array_split(shuffled_indices, n_splits)
    all_indices = np.arange(n_samples)
    results: list[GroupSplit] = []

    for fold, test_indices in enumerate(test_folds):
        outer_train = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        if outer_train.size < 2:
            raise ValueError(
                "sample 模式的 outer-train 少于两个样本，"
                "无法同时生成非空训练集和验证集。"
            )

        fold_rng = np.random.default_rng(seed + fold + 1)
        shuffled_outer_train = fold_rng.permutation(outer_train)
        validation_count = int(round(outer_train.size * validation_fraction))
        validation_count = max(1, min(validation_count, outer_train.size - 1))
        validation_indices = shuffled_outer_train[:validation_count]
        train_indices = shuffled_outer_train[validation_count:]
        results.append(
            GroupSplit(
                fold=fold,
                train_indices=np.sort(train_indices),
                validation_indices=np.sort(validation_indices),
                test_indices=np.sort(test_indices),
            )
        )
    return results


def _build_group_splits(
    groups: np.ndarray,
    n_splits: int,
    validation_fraction: float,
    seed: int,
) -> list[GroupSplit]:
    """按任意标签保持完整分组，生成训练/验证/测试索引。

    Args:
        groups: 每个样本的分组标签。
        n_splits: 外层交叉验证折数。
        validation_fraction: 验证集占每个 outer-train 的目标比例。
        seed: 控制随机分组放置与验证组选取的随机种子。

    Returns:
        不会拆散任何分组的交叉验证划分列表。
    """

    test_group_folds = _random_balanced_group_folds(groups, n_splits, seed)
    sample_indices = np.arange(groups.size)
    results: list[GroupSplit] = []

    for fold, test_groups in enumerate(test_group_folds):
        test_mask = np.isin(groups, test_groups)
        test_indices = sample_indices[test_mask]
        outer_train = sample_indices[~test_mask]
        validation_groups = _choose_validation_groups(
            groups[outer_train], validation_fraction, seed + fold + 1
        )
        validation_mask = np.isin(groups[outer_train], validation_groups)
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


def build_cross_validation_splits(
    *,
    split_mode: str,
    n_samples: int,
    model_ids: np.ndarray | list[int] | None = None,
    plant_counts: np.ndarray | list[int] | None = None,
    flow_speeds: np.ndarray | list[float] | None = None,
    n_splits: int = 5,
    validation_fraction: float = 0.2,
    seed: int = 20260814,
) -> list[GroupSplit]:
    """根据所选粒度创建可复现的交叉验证划分。

    Args:
        split_mode: 划分方式。``sample`` 表示逐样本随机划分；``model`` 表示
            ``model_id`` 相同的样本不可拆分；``plant_count`` 表示水草根数相同的
            样本不可拆分；``flow_speed`` 表示使用固定的跨流速实验协议。
        n_samples: 数据集中的样本总数。
        model_ids: 每个样本的 ``model_id``。仅 ``model`` 模式必须提供。
        plant_counts: 每个样本的水草根数。仅 ``plant_count`` 模式必须提供。
        flow_speeds: 每个样本的流速。仅 ``flow_speed`` 模式必须提供。
        n_splits: 外层交叉验证折数，默认 5；``flow_speed`` 模式忽略此参数。
        validation_fraction: 从每个 outer-train 中划给验证集的目标比例；
            ``flow_speed`` 模式忽略此参数。
        seed: 控制随机操作的种子；``flow_speed`` 模式不执行随机划分，因此忽略。

    Returns:
        普通模式返回长度为 ``n_splits`` 的列表；``flow_speed`` 固定返回一折。

    Raises:
        ValueError: 模式名称无效、参数无效、标签长度错误，或分组数不足时抛出。
    """

    if split_mode not in SUPPORTED_SPLIT_MODES:
        supported = ", ".join(SUPPORTED_SPLIT_MODES)
        raise ValueError(
            f"未知 split_mode={split_mode!r}；可选值为：{supported}。"
        )

    # 固定流速协议不使用交叉验证折数、验证比例或随机种子，因此必须在普通模式的
    # 共用参数校验之前分派，避免无关参数阻止这一次固定划分。
    if split_mode == FLOW_SPEED_SPLIT_MODE:
        return _build_flow_speed_split(flow_speeds, n_samples)

    _validate_common_arguments(n_samples, n_splits, validation_fraction)

    if split_mode == "sample":
        return _build_sample_splits(
            n_samples, n_splits, validation_fraction, seed
        )

    if split_mode == "model":
        groups = _as_group_array(
            model_ids, argument_name="model_ids", n_samples=n_samples
        )
    else:
        groups = _as_group_array(
            plant_counts, argument_name="plant_counts", n_samples=n_samples
        )
    return _build_group_splits(groups, n_splits, validation_fraction, seed)


def build_group_kfold_splits(
    groups: np.ndarray | list[int],
    n_splits: int = 5,
    validation_fraction: float = 0.2,
    seed: int = 20260814,
) -> list[GroupSplit]:
    """兼容旧调用方式，按 ``model_id`` 完整分组创建划分。

    Args:
        groups: 每个样本的 ``model_id``；同一值不跨训练/验证/测试集合。
        n_splits: 外层交叉验证折数。
        validation_fraction: 从 outer-train 中抽出的验证目标比例。
        seed: 控制外层和内层随机分组的随机种子。

    Returns:
        长度为 ``n_splits`` 的 :class:`GroupSplit` 列表。

    Notes:
        该函数保留原有名称与参数，内部转调统一入口。新代码建议直接调用
        :func:`build_cross_validation_splits`。
    """

    group_array = np.asarray(groups)
    if group_array.ndim != 1 or group_array.size == 0:
        raise ValueError("groups 必须是非空一维数组。")
    return build_cross_validation_splits(
        split_mode="model",
        n_samples=int(group_array.size),
        model_ids=group_array,
        n_splits=n_splits,
        validation_fraction=validation_fraction,
        seed=seed,
    )
