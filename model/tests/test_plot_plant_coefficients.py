"""检验逐株系数绘图脚本使用当前模型输入，并在 CPU/CUDA 保持预测一致。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from model.hydro.data import collate_hydro_samples
from model.hydro.physics import load_physical_config
from model.models import HydroTransformer
from model.training.trainer import GlobalFeatureScaler
from visualization import plot_plant_coefficients as plotting_module


# 测试默认值集中声明；小模型和预先构造的单株力用于隔离接口问题，避免重复物理求解。
TEST_SEED = 1022
TEST_DEVICE_NAMES = ("cpu", "cuda")
TEST_MODEL_CONFIG = {
    "d_model": 16, "n_heads": 2, "n_layers": 1, "ffn_dim": 32,
    "dropout": 0.0, "relative_hidden_dim": 8, "coefficient_hidden_dim": 8,
    "plant_encoder_hidden_dim": 12, "global_hidden_dim": 12,
}
TEST_PLANT_INPUTS = (
    {
        "positions": ((0.0, 0.0), (1.0, 0.0)),
        "plant_angles": (15.5, 75.0),
        "plant_stiffness": (8.0e-7, 2.08e-5),
        "single_drag": (0.2, 0.4),
    },
    {
        "positions": ((0.0, 1.0),),
        "plant_angles": (120.0,),
        "plant_stiffness": (5.907e-6,),
        "single_drag": (0.3,),
    },
)
TEST_GLOBAL_FEATURES = (0.3, 0.0, 1.0)
TEST_SCALER_MEAN = (0.2, 0.0, 0.0)
TEST_SCALER_STD = (0.1, 1.0, 1.0)
TEST_SAMPLE_ANGLE = 60
TEST_TARGET_DRAG = 0.5


def _make_samples() -> list[dict[str, Any]]:
    """构造不同株数及不同逐株属性的样本，返回与 HydroDataset 一致的字段字典列表。"""

    samples = []
    for index, plant_inputs in enumerate(TEST_PLANT_INPUTS):
        # 每个逐株字段按相同次序转换为 Tensor；零维标签和元数据供真实 collate 使用。
        sample = {name: torch.tensor(values) for name, values in plant_inputs.items()}
        sample.update(
            plant_mask=torch.ones(len(plant_inputs["positions"]), dtype=torch.bool),
            global_features=torch.tensor(TEST_GLOBAL_FEATURES),
            target_drag=torch.tensor(TEST_TARGET_DRAG),
            raw_target_drag=torch.tensor(TEST_TARGET_DRAG),
            sample_id=f"coefficient_plot_{index}", model_id=index,
            angle=TEST_SAMPLE_ANGLE, flow_speed=TEST_GLOBAL_FEATURES[0], source_index=index,
        )
        samples.append(sample)
    return samples


@pytest.mark.parametrize("device_name", TEST_DEVICE_NAMES)
def test_plot_predictions_match_current_model_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_name: str,
) -> None:
    """tmp_path 隔离文件；monkeypatch 替换数据求解；device_name 检验模型与输入的设备一致。"""

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("当前环境没有 CUDA，跳过 GPU 输入迁移检查。")

    # 保留真实整理输入、标准化和模型推理，只替换与本次接口修复无关的物理求解部分。
    samples = _make_samples()
    source = tmp_path / "samples.jsonl"
    source.touch()
    config = replace(
        plotting_module.load_script_config(plotting_module.DEFAULT_CONFIG_PATH),
        samples_dataset_path=source, output_directory=tmp_path / "output",
    )
    monkeypatch.setattr(plotting_module, "HydroDataset", lambda *args, **kwargs: samples)
    device = torch.device(device_name)
    torch.manual_seed(TEST_SEED)
    model = HydroTransformer(**TEST_MODEL_CONFIG).to(device).eval()
    scaler = GlobalFeatureScaler(mean=TEST_SCALER_MEAN, std=TEST_SCALER_STD)
    metadata = {"physical_config": load_physical_config().to_dict()}

    predictions = plotting_module.predict_all_samples(config, model, scaler, metadata, device)

    # 用显式提供六项正确输入的真实模型输出作参照，防止漏传、错传或重新构造逐株属性。
    assert len(predictions) == len(samples)
    for index, (prediction, sample) in enumerate(zip(predictions, samples)):
        batch = collate_hydro_samples([sample])
        with torch.inference_mode():
            expected = model(
                positions=batch["positions"].to(device),
                single_drag=batch["single_drag"].to(device),
                global_features=scaler.transform(batch["global_features"].to(device)),
                plant_mask=batch["plant_mask"].to(device),
                plant_angles=batch["plant_angles"].to(device),
                plant_stiffness=batch["plant_stiffness"].to(device),
            )
        assert prediction.dataset_row_index == index
        np.testing.assert_allclose(prediction.positions, sample["positions"].numpy())
        np.testing.assert_allclose(prediction.coefficients, expected["coefficient"][0].cpu().numpy())
        assert prediction.predicted_drag == pytest.approx(expected["total_drag"][0].item())
        assert prediction.coefficients.shape == sample["single_drag"].shape
        assert np.isfinite(prediction.coefficients).all()
        # 系数必须仍满足模型原有总阻力定义，确保可视化读取的是逐株输出。
        assert prediction.predicted_drag == pytest.approx(
            float(np.sum(prediction.coefficients * sample["single_drag"].numpy())),
        )
