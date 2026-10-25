"""检验连续角度/EI 编码、全局条件传播及新模型完整梯度路径。"""

from __future__ import annotations

import math

import pytest
import torch

from model.models import HydroTransformer, PlantEncoder
from model.training.trainer import GlobalFeatureScaler


# 小维度和固定 seed 仅用于测试；生产默认参数仍在 base.yaml 中。
TEST_SEED = 731  # 复现网络初始化和测试输入。
TEST_HIDDEN_DIM = 12  # 编码器中间维度，降低 CPU 测试开销。
TEST_OUTPUT_DIM = 16  # 同时作为小型 Transformer 的 d_model。
TEST_STIFFNESS_REFERENCE = 1.0  # N·m²；检验 log(EI/EI_ref)。
TEST_ANGLE_DEGREES = 15.5  # 小数、非 60° 工况。
SMALL_MODEL_CONFIG = {
    "d_model": TEST_OUTPUT_DIM, "n_heads": 2, "n_layers": 2,
    "ffn_dim": 32, "dropout": 0.0, "relative_hidden_dim": 8,
    "coefficient_hidden_dim": 8, "plant_encoder_hidden_dim": TEST_HIDDEN_DIM,
}


def _encoder() -> PlantEncoder:
    """用固定 seed 创建逐株编码器，确保回归可重复。"""

    torch.manual_seed(TEST_SEED)
    return PlantEncoder(TEST_OUTPUT_DIM, TEST_HIDDEN_DIM, TEST_STIFFNESS_REFERENCE)


def _model_and_inputs():
    """构造同向但逐株 EI 不同的有效工况，默认力是独立测试输入。"""

    torch.manual_seed(TEST_SEED)
    model = HydroTransformer(**SMALL_MODEL_CONFIG).eval()
    radians = math.radians(TEST_ANGLE_DEGREES)
    inputs = {
        "positions": torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
        "single_drag": torch.tensor([[0.2, 0.4, 0.8]]),
        "global_features": torch.tensor([[0.3, math.sin(radians), math.cos(radians)]]),
        "plant_mask": torch.ones(1, 3, dtype=torch.bool),
        "plant_angles": torch.full((1, 3), TEST_ANGLE_DEGREES),
        "plant_stiffness": torch.tensor([[0.5, 1.0, 2.0]]),
    }
    return model, inputs


def test_periodic_features_and_embeddings_match_for_whole_turns():
    """角度绕一整周不应改变输入特征或 embedding，也不截断小数角。"""

    encoder = _encoder()
    angles = torch.tensor([[TEST_ANGLE_DEGREES, 75.0, 315.0]], dtype=torch.float64)
    stiffness = torch.tensor([[0.5, 1.0, 2.0]])
    mask = torch.ones_like(stiffness, dtype=torch.bool)
    original = encoder(angles, stiffness, mask)
    for turns in (-2, -1, 1, 2):
        torch.testing.assert_close(encoder(angles + 360 * turns, stiffness, mask), original)
    assert not torch.allclose(encoder(angles.floor(), stiffness, mask), original)


def test_input_features_use_sin_cos_and_dimensionless_log_ei():
    """刚度类别不作为离散编号进入模型，连续 EI 先转换为无量纲对数。"""

    encoder = _encoder()
    features = encoder.physical_features(
        torch.tensor([[0.0, 90.0, 180.0]]), torch.tensor([[0.5, 1.0, 2.0]]),
        torch.ones(1, 3, dtype=torch.bool),
    )
    expected = torch.tensor([[[0.0, 1.0, math.log(0.5)],
                              [1.0, 0.0, 0.0], [0.0, -1.0, math.log(2.0)]]])
    torch.testing.assert_close(features, expected, atol=1e-6, rtol=1e-6)


def test_shared_encoder_produces_distinct_vectors_for_distinct_ei():
    """共享的是网络参数，不是所有植物的最终向量。"""

    encoder = _encoder()
    tokens = encoder(torch.full((1, 3), 15.0), torch.tensor([[0.5, 1.0, 2.0]]),
                     torch.ones(1, 3, dtype=torch.bool))
    assert tokens.shape == (1, 3, TEST_OUTPUT_DIM)
    assert not torch.allclose(tokens[:, 0], tokens[:, 1])
    assert not torch.allclose(tokens[:, 1], tokens[:, 2])


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_real_ei_is_rejected(invalid):
    """真实植株必须提供可取对数的正 EI。"""

    with pytest.raises(ValueError, match="有限正 EI"):
        _encoder()(torch.zeros(1, 1), torch.tensor([[invalid]]), torch.ones(1, 1, dtype=torch.bool))


def test_padding_is_safe_before_log_and_after_network_bias():
    """padding 可带零或无效占位值，但不能在中间计算和反向梯度中产生 NaN。"""

    encoder = _encoder()
    angles = torch.tensor([[15.0, float("nan"), float("inf")]])
    stiffness = torch.tensor([[0.5, 0.0, float("nan")]], requires_grad=True)
    mask = torch.tensor([[True, False, False]])
    tokens = encoder(angles, stiffness, mask)
    assert torch.isfinite(tokens).all()
    assert torch.count_nonzero(tokens[:, 1:]) == 0
    tokens.sum().backward()
    assert torch.isfinite(stiffness.grad).all()
    assert torch.count_nonzero(stiffness.grad[:, 1:]) == 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in encoder.parameters())


def test_mixed_ei_permutation_equivariance_and_padding_invariance():
    """坐标、角度、EI、默认力和 mask 必须一起重排；增加 padding 不改变结果。"""

    model, inputs = _model_and_inputs()
    # 系数头初始 c=1；打破零初始化才能真正检验内部向量的排列一致性。
    torch.nn.init.normal_(model.coefficient_head.output.weight, std=0.1)
    original = model(**inputs)
    permutation = torch.tensor([2, 0, 1])
    permuted = {key: value if key == "global_features" else value[:, permutation]
                for key, value in inputs.items()}
    output = model(**permuted)
    torch.testing.assert_close(output["total_drag"], original["total_drag"])
    torch.testing.assert_close(output["coefficient"], original["coefficient"][:, permutation])

    # padding 的 EI=0 不应进入 log；每个逐株张量补一个零占位。
    padded = {key: value if key == "global_features" else
              torch.cat((value, torch.zeros_like(value[:, :1])), dim=1)
              for key, value in inputs.items()}
    padded_output = model(**padded)
    torch.testing.assert_close(padded_output["total_drag"], original["total_drag"])
    torch.testing.assert_close(padded_output["coefficient"][:, :3], original["coefficient"])
    assert padded_output["coefficient"][0, 3] == 0


def test_new_encoder_preserves_physical_initialization_and_receives_gradients():
    """编码器不能破坏 c=1 的物理初始值；训练更新能够经过它传播。"""

    model, inputs = _model_and_inputs()
    initial = model(**inputs)
    torch.testing.assert_close(initial["total_drag"], inputs["single_drag"].sum(dim=1))
    torch.testing.assert_close(initial["coefficient"], torch.ones(1, 3))
    torch.nn.init.normal_(model.coefficient_head.output.weight, std=0.1)
    loss = (model(**inputs)["total_drag"] - 1.1).square().sum()
    loss.backward()
    gradients = [parameter.grad for parameter in model.plant_encoder.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_angle_condition_reaches_both_normalizations_in_every_layer():
    """每层两个 Conditional LayerNorm 均收到同一个由流速和角度生成的条件向量。"""

    model, inputs = _model_and_inputs()
    received = []
    handles = []
    # pre-hook 只记录传入的条件，不改变前向行为；完成后移除所有 hook。
    for block in model.blocks:
        for norm in (block.norm1, block.norm2):
            assert norm.enabled
            handles.append(norm.register_forward_pre_hook(lambda module, args: received.append(args[1].detach())))
    try:
        model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    condition = model.global_encoder(inputs["global_features"])
    assert len(received) == 2 * model.config.n_layers
    for value in received:
        torch.testing.assert_close(value, condition)
    changed_global = inputs["global_features"].clone()
    changed_global[:, 1:] = torch.tensor([[1.0, 0.0]])
    changed_condition = model.global_encoder(changed_global)
    assert not torch.allclose(condition, changed_condition)
    # FiLM（逐特征缩放和平移）初始为零；模拟训练后应能感知不同条件。
    norm = model.blocks[0].norm1
    torch.nn.init.normal_(norm.modulation.to_scale_shift.weight, std=0.1)
    hidden = model.plant_encoder(inputs["plant_angles"], inputs["plant_stiffness"], inputs["plant_mask"])
    assert not torch.allclose(norm(hidden, condition), norm(hidden, changed_condition))


def test_scaler_fits_only_train_flow_and_preserves_angle_features_on_restore():
    """验证/测试流速不参与拟合，角度的 sin/cos 不做二次标准化。"""

    dataset = [{"global_features": torch.tensor(values, dtype=torch.float64)}
               for values in ((0.2, 0.0, 1.0), (0.4, 1.0, 0.0), (9.0, -1.0, 0.0))]
    scaler = GlobalFeatureScaler.fit(dataset, [0, 1])
    assert scaler.mean == pytest.approx((0.3, 0.0, 0.0))
    assert scaler.std == pytest.approx((0.1, 1.0, 1.0))
    assert scaler.standardize_mask == (True, False, False)
    values = torch.stack([record["global_features"] for record in dataset])
    transformed = scaler.transform(values)
    torch.testing.assert_close(transformed[:, 1:], values[:, 1:], rtol=0, atol=0)
    restored = GlobalFeatureScaler.from_dict(scaler.to_dict())
    torch.testing.assert_close(restored.transform(values), transformed, rtol=0, atol=0)


def test_new_model_rejects_one_dimensional_global_contract():
    """旧单流速全局输入不能绕过新架构的固定三维契约。"""

    with pytest.raises(ValueError, match="global_input_dim"):
        HydroTransformer(**SMALL_MODEL_CONFIG, global_input_dim=1)
