"""HydroTransformer 的几何、mask、不变性和数值稳定性单元测试。"""

import torch

from model.models import (
    HydroMultiHeadAttention,
    HydroTransformer,
    RelativeGeometryEncoder,
    compute_relative_positions,
)


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
    )
    torch.manual_seed(20260814)
    ablated_model = _small_model(
        use_rope=False,
        use_relative_value=False,
        use_conditional_layernorm=False,
        condition_value_on_global=False,
        condition_relative_value_on_global=False,
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


def test_permutation_equivariance_and_total_drag_invariance():
    """重排植物顺序后，逐株系数同步重排，而总阻力保持不变。"""

    model = _small_model().eval()
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


def test_padding_invariance():
    """仅增加 padding 植物不能改变真实植物系数或总阻力。"""

    model = _small_model().eval()
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


def test_all_padding_is_finite_and_returns_zero():
    """没有有效植物的 batch 不应产生 NaN，总阻力和系数应为零。"""

    model = _small_model().eval()
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


def test_forward_backward_has_no_nan():
    """完整前向和反向传播中的输出、损失、梯度均应为有限数。"""

    model = _small_model().train()
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
