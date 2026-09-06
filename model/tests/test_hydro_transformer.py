"""HydroTransformer 的几何、方向 attention、mask 和数值稳定性单元测试。"""

import pytest
import torch

from model.models import (
    HydroMultiHeadAttention,
    HydroTransformer,
    HydroTransformerConfig,
    RelativeGeometryEncoder,
    compute_directional_soft_penalty,
    compute_downstream_offsets,
    compute_relative_positions,
)
from model.training.config import load_config


def _small_model(**overrides) -> HydroTransformer:
    """构建测试专用的小模型，降低测试时间但保留所有结构路径。"""

    defaults = {
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 2,
        "ffn_dim": 64,
        "dropout": 0.0,
        "relative_hidden_dim": 16,
        "coefficient_hidden_dim": 16,
    }
    defaults.update(overrides)
    return HydroTransformer(**defaults)


def _make_batch(batch_size: int = 2, plant_count: int = 4):
    """生成同时含有效植物和 padding 的确定性测试 batch。"""

    torch.manual_seed(7)
    positions = torch.randn(batch_size, plant_count, 2)
    single_drag = torch.rand(batch_size, plant_count) + 0.2
    global_features = torch.rand(batch_size, 1)
    plant_mask = torch.ones(batch_size, plant_count, dtype=torch.bool)
    if batch_size > 1:
        plant_mask[1, -1] = False
        single_drag[1, -1] = 0.0
    return positions, single_drag, global_features, plant_mask


def _make_coefficients_nontrivial(model: HydroTransformer) -> None:
    """打破系数头的零初始化，使排列不变性测试不会因 c=1 而平凡通过。"""

    torch.manual_seed(13)
    torch.nn.init.normal_(model.coefficient_head.output.weight, std=0.1)
    torch.nn.init.normal_(model.coefficient_head.output.bias, std=0.02)


def test_training_and_direct_construction_use_intended_directional_defaults():
    """正式训练默认启用 soft，直接构造则默认 none 以兼容旧模型。"""

    assert load_config(None)["model"]["directional_attention_mode"] == "soft"
    direct_config = HydroTransformerConfig()
    assert direct_config.directional_attention_mode == "none"
    assert direct_config.directional_soft_activation == "relu"
    assert direct_config.directional_soft_strength == 1.0
    assert direct_config.directional_tolerance == 1.0e-6


def test_forward_shapes_and_attention_shapes():
    """公开 forward 应返回训练和解释所需的全部张量。"""

    model = _small_model().eval()
    batch = _make_batch()
    outputs = model(*batch, return_attention=True)

    assert outputs["total_drag"].shape == (2,)
    assert outputs["coefficient"].shape == (2, 4)
    assert outputs["log_coefficient"].shape == (2, 4)
    assert len(outputs["attention"]) == 2
    assert outputs["attention"][0].shape == (2, 2, 4, 4)


def test_rope_dimension_constraint_only_applies_when_enabled():
    """关闭 RoPE 后，head_dim 不为 4 的倍数仍应支持完整 forward。"""

    # d_model=18、n_heads=3 得到 head_dim=6；它不能执行 2D RoPE，但普通 attention 合法。
    model = _small_model(
        d_model=18,
        n_heads=3,
        ffn_dim=36,
        use_rope=False,
    ).eval()
    positions, single_drag, global_features, plant_mask = _make_batch(batch_size=1)
    outputs = model(positions, single_drag, global_features, plant_mask)
    assert torch.isfinite(outputs["total_drag"]).all()

    # 同一维度组合一旦启用 RoPE，就应尽早给出清晰的配置错误。
    try:
        _small_model(d_model=18, n_heads=3, ffn_dim=36, use_rope=True)
    except ValueError as error:
        assert "head_dim" in str(error) or "RoPE" in str(error)
    else:
        raise AssertionError("启用 2D RoPE 时应拒绝 head_dim=6。")


def test_ablation_switches_preserve_identical_parameter_initialization():
    """同 seed 切换所有消融开关时，全部参数名称和初始值必须完全一致。"""

    torch.manual_seed(20260814)
    full_model = _small_model(
        use_rope=True,
        use_relative_value=True,
        use_conditional_layernorm=True,
        condition_value_on_global=True,
        condition_relative_value_on_global=True,
        directional_attention_mode="none",
    )
    torch.manual_seed(20260814)
    ablated_model = _small_model(
        use_rope=False,
        use_relative_value=False,
        use_conditional_layernorm=False,
        condition_value_on_global=False,
        condition_relative_value_on_global=False,
        directional_attention_mode="hard",
    )

    full_state = full_model.state_dict()
    ablated_state = ablated_model.state_dict()
    assert full_state.keys() == ablated_state.keys()
    for name in full_state:
        torch.testing.assert_close(full_state[name], ablated_state[name], msg=lambda message: f"{name}: {message}")


def test_attention_message_has_only_one_residual_dropout_location():
    """attention 权重可 dropout，但输出投影后只由 block 做一次 residual dropout。"""

    block = _small_model().blocks[0]
    assert hasattr(block.attention, "attention_dropout")
    assert not hasattr(block.attention, "output_dropout")
    assert hasattr(block, "residual_dropout")


def test_relative_position_uses_source_minus_target_direction():
    """确认相对方向是 p_j-p_i，避免把上游与下游写反。"""

    positions = torch.tensor([[[2.0, 5.0], [3.0, 2.0]]])
    relative = compute_relative_positions(positions)

    # target i=0、source j=1 时，应得到 dx=1、dy=-3。
    torch.testing.assert_close(relative[0, 0, 1], torch.tensor([1.0, -3.0]))
    torch.testing.assert_close(relative[0, 1, 0], torch.tensor([-1.0, 3.0]))


def test_downstream_offsets_and_relu_penalty_follow_upstream_sign_convention():
    """x 较大的植物在上游，只有下游 source 应产生 ReLU 惩罚。"""

    positions = torch.tensor([[[2.0, 0.0], [1.0, 3.0], [0.0, -2.0]]])
    downstream_offsets = compute_downstream_offsets(positions)
    expected_offsets = torch.tensor(
        [[[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]]]
    )
    torch.testing.assert_close(downstream_offsets, expected_offsets)

    penalty = compute_directional_soft_penalty(
        downstream_offsets,
        activation="relu",
        tolerance=0.25,
    )
    expected_penalty = torch.tensor(
        [[[0.0, 0.75, 1.75], [0.0, 0.0, 0.75], [0.0, 0.0, 0.0]]]
    )
    torch.testing.assert_close(penalty, expected_penalty)


def test_hard_directional_attention_blocks_only_downstream_sources():
    """hard 模式屏蔽下游 source，同时保留上游、同横截面和 self-edge。"""

    attention_layer = HydroMultiHeadAttention(
        d_model=8,
        n_heads=2,
        dropout=0.0,
        condition_dim=8,
        use_rope=False,
        use_relative_value=False,
        condition_value_on_global=False,
        condition_relative_value_on_global=False,
        directional_attention_mode="hard",
        directional_tolerance=0.0,
    ).eval()
    hidden_states = torch.zeros(1, 4, 8)
    positions = torch.tensor([[[2.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 0.0]]])
    condition = torch.zeros(1, 8)
    plant_mask = torch.ones(1, 4, dtype=torch.bool)

    _, attention = attention_layer(
        hidden_states,
        positions,
        condition,
        plant_mask,
        return_attention=True,
    )

    # 最上游 target 只能读取自己；中间横截面的两株植物可以互相读取。
    assert torch.count_nonzero(attention[:, :, 0, 1:]) == 0
    assert torch.count_nonzero(attention[:, :, 1, 3]) == 0
    assert torch.count_nonzero(attention[:, :, 2, 3]) == 0
    assert torch.all(attention[:, :, 1, 0] > 0)
    assert torch.all(attention[:, :, 1, 2] > 0)
    assert torch.all(attention[:, :, 2, 1] > 0)
    assert torch.all(attention.diagonal(dim1=-2, dim2=-1) > 0)
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(1, 2, 4))


def test_soft_directional_attention_uses_distance_penalty_without_zeroing_edges():
    """soft 模式应连续降低更远的下游权重，但不能像 hard 模式一样置零。"""

    common_arguments = {
        "d_model": 8,
        "n_heads": 2,
        "dropout": 0.0,
        "condition_dim": 8,
        "use_rope": False,
        "use_relative_value": False,
        "condition_value_on_global": False,
        "condition_relative_value_on_global": False,
    }
    none_layer = HydroMultiHeadAttention(
        directional_attention_mode="none", **common_arguments
    ).eval()
    soft_layer = HydroMultiHeadAttention(
        directional_attention_mode="soft",
        directional_soft_activation="relu",
        directional_soft_strength=1.0,
        directional_tolerance=0.0,
        **common_arguments,
    ).eval()
    soft_layer.load_state_dict(none_layer.state_dict())

    hidden_states = torch.zeros(1, 3, 8)
    positions = torch.tensor([[[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]]])
    condition = torch.zeros(1, 8)
    plant_mask = torch.ones(1, 3, dtype=torch.bool)
    _, none_attention = none_layer(
        hidden_states, positions, condition, plant_mask, return_attention=True
    )
    _, soft_attention = soft_layer(
        hidden_states, positions, condition, plant_mask, return_attention=True
    )

    # 对最上游 target，距离为 0/1/2 的 source 权重应依次下降但始终大于零。
    assert torch.all(soft_attention[:, :, 0, 0] > soft_attention[:, :, 0, 1])
    assert torch.all(soft_attention[:, :, 0, 1] > soft_attention[:, :, 0, 2])
    assert torch.all(soft_attention[:, :, 0, :] > 0)
    # 最下游 target 没有位于它下游的 source，因此其 attention 与 none 完全相同。
    torch.testing.assert_close(soft_attention[:, :, 2, :], none_attention[:, :, 2, :])


def test_zero_strength_soft_attention_is_exactly_none():
    """soft_strength=0 必须旁路惩罚计算并精确复现 none 模式。"""

    torch.manual_seed(29)
    none_model = _small_model(directional_attention_mode="none").eval()
    torch.manual_seed(29)
    zero_soft_model = _small_model(
        directional_attention_mode="soft",
        directional_soft_strength=0.0,
    ).eval()
    batch = _make_batch(batch_size=1)

    none_output = none_model(*batch, return_attention=True)
    soft_output = zero_soft_model(*batch, return_attention=True)
    for name in ("total_drag", "coefficient", "log_coefficient"):
        torch.testing.assert_close(soft_output[name], none_output[name], atol=0.0, rtol=0.0)
    for soft_attention, none_attention in zip(
        soft_output["attention"], none_output["attention"]
    ):
        torch.testing.assert_close(soft_attention, none_attention, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"directional_attention_mode": "unknown"}, "directional_attention_mode"),
        ({"directional_soft_activation": "sigmoid"}, "directional_soft_activation"),
        ({"directional_soft_strength": -1.0}, "directional_soft_strength"),
        ({"directional_soft_strength": float("nan")}, "directional_soft_strength"),
        ({"directional_tolerance": -1.0}, "directional_tolerance"),
        ({"directional_tolerance": float("inf")}, "directional_tolerance"),
    ],
)
def test_invalid_directional_attention_configuration_is_rejected(
    overrides: dict[str, object], expected_message: str
) -> None:
    """非法方向配置应在模型构造阶段给出包含字段名的清晰错误。"""

    with pytest.raises(ValueError, match=expected_message):
        _small_model(**overrides)


def test_relative_value_self_edges_are_exactly_zero_after_film():
    """即使全局 FiLM 的 beta 非零，i==j 的 relative Value 也必须严格为零。"""

    encoder = RelativeGeometryEncoder(
        n_heads=2,
        head_dim=4,
        hidden_dim=8,
        condition_dim=8,
        enabled=True,
        condition_on_global=True,
    )
    with torch.no_grad():
        encoder.modulation.to_scale_shift.bias.fill_(0.5)
    positions = torch.randn(2, 3, 2)
    condition = torch.randn(2, 8)
    relative_value = encoder(positions, condition)
    diagonal = relative_value.diagonal(dim1=2, dim2=3)

    assert torch.count_nonzero(diagonal) == 0


@pytest.mark.parametrize("directional_attention_mode", ["none", "soft", "hard"])
def test_permutation_equivariance_and_total_drag_invariance(
    directional_attention_mode: str,
):
    """重排植物顺序后，逐株系数同步重排，而总阻力保持不变。"""

    model = _small_model(directional_attention_mode=directional_attention_mode).eval()
    _make_coefficients_nontrivial(model)
    positions, single_drag, global_features, plant_mask = _make_batch(batch_size=1)
    original = model(positions, single_drag, global_features, plant_mask)

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model(
        positions[:, permutation],
        single_drag[:, permutation],
        global_features,
        plant_mask[:, permutation],
    )

    torch.testing.assert_close(permuted["total_drag"], original["total_drag"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        permuted["coefficient"], original["coefficient"][:, permutation], atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("directional_attention_mode", ["none", "soft", "hard"])
def test_padding_invariance(directional_attention_mode: str):
    """仅增加 padding 植物不能改变真实植物系数或总阻力。"""

    model = _small_model(directional_attention_mode=directional_attention_mode).eval()
    _make_coefficients_nontrivial(model)
    positions, single_drag, global_features, plant_mask = _make_batch(batch_size=1, plant_count=3)
    original = model(positions, single_drag, global_features, plant_mask)

    padded_positions = torch.cat((positions, torch.randn(1, 5, 2)), dim=1)
    padded_drag = torch.cat((single_drag, torch.zeros(1, 5)), dim=1)
    padded_mask = torch.cat((plant_mask, torch.zeros(1, 5, dtype=torch.bool)), dim=1)
    padded = model(padded_positions, padded_drag, global_features, padded_mask)

    torch.testing.assert_close(padded["total_drag"], original["total_drag"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        padded["coefficient"][:, :3], original["coefficient"], atol=1e-5, rtol=1e-5
    )
    assert torch.count_nonzero(padded["coefficient"][:, 3:]) == 0


def test_single_plant_physics_informed_initialization():
    """系数头初始为 c=1，因此单株总阻力应精确等于孤立阻力。"""

    model = _small_model().eval()
    outputs = model(
        positions=torch.tensor([[[0.0, 0.0]]]),
        single_drag=torch.tensor([[2.75]]),
        global_features=torch.tensor([[0.2]]),
        plant_mask=torch.tensor([[True]]),
    )

    torch.testing.assert_close(outputs["coefficient"], torch.ones(1, 1))
    torch.testing.assert_close(outputs["log_coefficient"], torch.zeros(1, 1))
    torch.testing.assert_close(outputs["total_drag"], torch.tensor([2.75]))


def test_relative_value_breaks_constant_token_geometry_degeneracy():
    """constant token 下，relative Value 应让不同几何产生不同消息。"""

    torch.manual_seed(19)
    common_arguments = {
        "d_model": 16,
        "n_heads": 2,
        "dropout": 0.0,
        "condition_dim": 16,
        "relative_hidden_dim": 16,
        "use_rope": True,
        "condition_value_on_global": False,
        "condition_relative_value_on_global": False,
    }
    with_relative = HydroMultiHeadAttention(use_relative_value=True, **common_arguments).eval()
    without_relative = HydroMultiHeadAttention(use_relative_value=False, **common_arguments).eval()
    # 复制公共投影，确保两模型的差异只来自 relative Value 开关。
    without_relative.load_state_dict(with_relative.state_dict(), strict=False)

    hidden_states = torch.ones(1, 3, 16)
    condition = torch.zeros(1, 16)
    plant_mask = torch.ones(1, 3, dtype=torch.bool)
    geometry_a = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    geometry_b = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [0.0, 3.0]]])

    relative_a = with_relative(hidden_states, geometry_a, condition, plant_mask)
    relative_b = with_relative(hidden_states, geometry_b, condition, plant_mask)
    plain_a = without_relative(hidden_states, geometry_a, condition, plant_mask)
    plain_b = without_relative(hidden_states, geometry_b, condition, plant_mask)

    assert not torch.allclose(relative_a, relative_b, atol=1e-6, rtol=1e-6)
    # 普通 V 对所有 source 完全相同，加权和与 attention 权重无关，故几何区别消失。
    torch.testing.assert_close(plain_a, plain_b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("directional_attention_mode", ["none", "soft", "hard"])
def test_all_padding_is_finite_and_returns_zero(directional_attention_mode: str):
    """没有有效植物的 batch 不应产生 NaN，总阻力和系数应为零。"""

    model = _small_model(directional_attention_mode=directional_attention_mode).eval()
    outputs = model(
        positions=torch.randn(2, 4, 2),
        single_drag=torch.zeros(2, 4),
        global_features=torch.randn(2, 1),
        plant_mask=torch.zeros(2, 4, dtype=torch.bool),
        return_attention=True,
    )

    for name in ("total_drag", "coefficient", "log_coefficient"):
        assert torch.isfinite(outputs[name]).all()
        assert torch.count_nonzero(outputs[name]) == 0
    for attention in outputs["attention"]:
        assert torch.isfinite(attention).all()
        assert torch.count_nonzero(attention) == 0


@pytest.mark.parametrize("directional_attention_mode", ["none", "soft", "hard"])
def test_forward_backward_has_no_nan(directional_attention_mode: str):
    """完整前向和反向传播中的输出、损失、梯度均应为有限数。"""

    model = _small_model(directional_attention_mode=directional_attention_mode).train()
    positions, single_drag, global_features, plant_mask = _make_batch()
    outputs = model(positions, single_drag, global_features, plant_mask)
    target = torch.tensor([2.0, 1.5])
    loss = torch.nn.functional.mse_loss(outputs["total_drag"], target)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in outputs.values())
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_all_plants_share_one_trainable_token():
    """模型只包含一个共享 Token，角度状态不能再成为神经网络输入。"""

    model = _small_model().train()
    _make_coefficients_nontrivial(model)
    positions = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ]
    )
    single_drag = torch.ones(2, 2)
    global_features = torch.zeros(2, 1)
    plant_mask = torch.ones(2, 2, dtype=torch.bool)
    assert model.plant_token.shape == (model.config.d_model,)
    outputs = model(positions, single_drag, global_features, plant_mask)
    torch.testing.assert_close(outputs["total_drag"][0], outputs["total_drag"][1])
    outputs["total_drag"].sum().backward()
    assert model.plant_token.grad is not None
    assert torch.count_nonzero(model.plant_token.grad) > 0


def test_forward_signature_does_not_accept_angle_or_state():
    """神经网络 forward 不应接收任何原始角度或派生状态参数。"""

    model = _small_model().eval()
    positions = torch.zeros(1, 2, 2)
    single_drag = torch.tensor([[1.0, 0.0]])
    global_features = torch.zeros(1, 1)
    plant_mask = torch.tensor([[True, False]])

    with pytest.raises(TypeError):
        model(
            positions,
            single_drag,
            global_features,
            plant_mask,
            plant_state=torch.tensor([[1, 0]], dtype=torch.long),
        )
