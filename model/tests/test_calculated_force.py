"""验证原 MATLAB 基准、转盘角适配、参数恢复以及模型实际使用的默认力。"""

from __future__ import annotations

import json
import math
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import ConcatDataset, Subset

from model.hydro.calculated_force import _cached_single_drag, column_angles_for_turntable
from model.hydro.data import HydroDataset, MirrorTrainingDataset, collate_hydro_samples
from model.hydro.geometry import build_hex_coordinates
from model.hydro.physics import load_physical_config
from model.hydro.seagrass_drag import ConvergenceWarning, calculate_drag_coefficient_v2
from model.models import HydroTransformer
from model.training.config import load_config


# 容差沿用原包 MATLAB 对照测试，不能为了通过测试自动放宽。
REFERENCE_PATH = Path(__file__).parent / "fixtures" / "seagrass_matlab_force_reference.json"
FORCE_RELATIVE_TOLERANCE = 1.0e-7
FORCE_ABSOLUTE_TOLERANCE = 1.0e-8
DISPLACEMENT_RELATIVE_TOLERANCE = 1.0e-6
DISPLACEMENT_ABSOLUTE_TOLERANCE = 1.0e-9
TEST_FLOW_SPEED = 0.2  # m/s。
TEST_OCCUPIED_INDICES = (0, 18)  # 用非中心点检验旋转，并用中心点检验动态 mask。
MATLAB_CASES = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", MATLAB_CASES, ids=lambda case: case["name"])
def test_vendored_solver_matches_independent_matlab_reference(case) -> None:
    """33 个原 MATLAB 案例检验柔性/刚性、材料、速度、遮蔽及端点行为。"""

    # 基准含故意未收敛的案例；告警的可见性在下方专门验证。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, force, displacement, _ = calculate_drag_coefficient_v2(*case["args"])
    assert force == pytest.approx(case["F_total"], rel=FORCE_RELATIVE_TOLERANCE,
                                 abs=FORCE_ABSOLUTE_TOLERANCE)
    assert displacement == pytest.approx(
        case["w_max"], rel=DISPLACEMENT_RELATIVE_TOLERANCE,
        abs=DISPLACEMENT_ABSOLUTE_TOLERANCE,
    )


@pytest.mark.parametrize("angle, expected", [
    (0, (60, 180, 300)), (60, (120, 240, 0)), (75, (135, 255, 15)),
    (315, (15, 135, 255)), (-45, (15, 135, 255)), (360, (60, 180, 300)),
    (15.5, (75.5, 195.5, 315.5)),
])
def test_turntable_angles_keep_column_order(angle, expected) -> None:
    """归一化跨越 360° 的角度时仍保留列序，且不截断小数。"""

    assert column_angles_for_turntable(angle) == expected


def _payload(*, rigid: bool = False) -> dict:
    """显式恢复原 MATLAB 材料案例的 v2 参数，不让用户当前配置影响历史基准。"""

    payload = load_physical_config().to_dict()
    payload["version"] = 2
    payload["material"] = {"young_modulus_pa": 1.25e7}
    payload["geometry"]["cylinder_diameter_m"] = 0.025
    if rigid:
        payload["solver"]["max_iterations"] = 0
    return payload


def test_configured_force_is_matlab_total_not_multiplied_again() -> None:
    """默认 PVC 参数必须直接得到 MATLAB 的整株总力，不重复汇总叶片或圆柱。"""

    physical = load_physical_config(_payload())
    reference = next(case for case in MATLAB_CASES if case["name"] == "material_1_height_1")
    assert physical.single_drag_for_angle(0, TEST_FLOW_SPEED) == pytest.approx(
        reference["F_total"], rel=FORCE_RELATIVE_TOLERANCE, abs=FORCE_ABSOLUTE_TOLERANCE,
    )
    assert "states" not in physical.to_dict()


def test_rigid_baseline_uses_all_three_columns_and_one_cylinder() -> None:
    """用可手算的刚性公式独立检查数量、圆柱和三个相对来流角。"""

    payload = _payload(rigid=True)
    geometry, fluid, drag = payload["geometry"], payload["fluid"], payload["drag"]
    angle = 75
    pressure = 0.5 * fluid["water_density_kg_m3"] * TEST_FLOW_SPEED**2
    expected = pressure * (
        drag["cylinder_drag_coefficient"] * geometry["cylinder_diameter_m"]
        * geometry["cylinder_height_m"]
        + drag["blade_drag_coefficient"] * geometry["blade_length_m"]
        * geometry["blade_height_m"] * geometry["blades_per_column"]
        * sum(abs(math.sin(math.radians(value))) for value in column_angles_for_turntable(angle))
    )
    assert load_physical_config(payload).single_drag_for_angle(angle, TEST_FLOW_SPEED) == pytest.approx(expected)


def test_cache_keys_include_physical_parameters_and_normalized_angle() -> None:
    """重复查询命中缓存，改变模量或流速后不会误用前一次的计算结果。"""

    _cached_single_drag.cache_clear()
    payload = _payload()
    physical = load_physical_config(payload)
    force = physical.single_drag_for_angle(0, TEST_FLOW_SPEED)
    assert physical.single_drag_for_angle(360, TEST_FLOW_SPEED) == force
    assert _cached_single_drag.cache_info().hits == 1
    changed = deepcopy(payload)
    changed["material"]["young_modulus_pa"] *= 2
    assert load_physical_config(changed).single_drag_for_angle(0, TEST_FLOW_SPEED) != force
    assert physical.single_drag_for_angle(0, 0.3) != force
    assert _cached_single_drag.cache_info().misses == 3


@pytest.mark.parametrize("group, key, value", [
    ("geometry", "blade_thickness_m", 0), ("material", "young_modulus_pa", math.nan),
    ("solver", "nodes", 4), ("solver", "nodes", True),
    ("solver", "max_iterations", 2.5), ("solver", "enable_shielding", "true"),
    ("solver", "min_shielding", 1.1), ("solver", "shielding_type", "linera"),
])
def test_invalid_parameters_are_rejected_before_solving(group, key, value) -> None:
    """无效材料、数值设置及拼写必须提前报错，而不能静默退回查表。"""

    payload = _payload()
    payload[group][key] = value
    with pytest.raises(ValueError):
        load_physical_config(payload)


def test_missing_parameters_and_invalid_runtime_inputs_are_rejected() -> None:
    """缺少参数不能偷偷用原包默认值；流速和角度必须是有限数值。"""

    payload = _payload()
    del payload["geometry"]["blade_height_m"]
    with pytest.raises(ValueError, match="geometry"):
        load_physical_config(payload)
    physical = load_physical_config(_payload())
    for angle, speed in ((True, 0.2), (math.inf, 0.2), (0, -0.2), (0, math.nan)):
        with pytest.raises(ValueError):
            physical.single_drag_for_angle(angle, speed)


def test_nonconvergence_warning_is_preserved() -> None:
    """原求解器未收敛时仍发出诊断告警，不把有限输出当作已收敛结果。"""

    payload = _payload()
    payload["solver"]["max_iterations"] = 1
    _cached_single_drag.cache_clear()
    with pytest.warns(ConvergenceWarning, match="未收敛"):
        force = load_physical_config(payload).single_drag_for_angle(15, 0.4)
    assert math.isfinite(force) and force > 0


def _record(angle: int, schema: int = 2) -> dict:
    """生成当前同向实验协议的最小样本；v2 坐标旋转六角锚点之后的余角。"""

    coordinates = build_hex_coordinates()
    remainder = angle % 60 if schema == 2 else 0
    cosine, sine = math.cos(math.radians(remainder)), math.sin(math.radians(remainder))
    plants = []
    for plant_index, grid_index in enumerate(TEST_OCCUPIED_INDICES):
        x, y = coordinates[grid_index]
        plants.append({"plant_index": plant_index, "grid_index": grid_index,
                       "x": x * cosine + y * sine, "y": -x * sine + y * cosine, "angle": angle})
    return {
        "schema_version": schema, "sample_id": f"test_angle_{angle}_schema_{schema}",
        "model_id": 1, "angle": angle, "flow_speed": TEST_FLOW_SPEED,
        "vegetation_layout": "".join("1" if index in TEST_OCCUPIED_INDICES else "0" for index in range(37)),
        "plants": plants,
        "measurements": {"TX": 0, "TY": 0, "TZ": 0, "FX_0": 1, "FY_0": 0, "FZ": 0},
    }


def _write_records(path: Path, records: list[dict]) -> None:
    """只写 pytest 临时目录中的测试 JSONL，不触碰正式实验数据。"""

    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_dataset_forward_and_checkpoint_parameter_roundtrip(tmp_path: Path) -> None:
    """混合 v1/v2 数据的默认力进入真正模型正向，参数恢复得到相同力。"""

    path = tmp_path / "samples.jsonl"
    _write_records(path, [_record(0, 1), _record(15), _record(75), _record(315)])
    physical = load_physical_config()
    dataset = HydroDataset(path, physical_config=physical)
    batch = collate_hydro_samples(list(dataset))
    model = HydroTransformer(d_model=16, n_heads=2, n_layers=1, ffn_dim=32,
                             relative_hidden_dim=8, coefficient_hidden_dim=8, dropout=0.0).eval()
    outputs = model(
        batch["positions"], batch["single_drag"], batch["global_features"], batch["plant_mask"],
        plant_angles=batch["plant_angles"], plant_stiffness=batch["plant_stiffness"],
    )
    # 当前系数头初始化为零，exp(0)=1；总力应恰好为各株默认力之和。
    torch.testing.assert_close(outputs["total_drag"], batch["single_drag"].sum(dim=1))
    for sample in dataset:
        expected = physical.single_drag_for_angle(sample["angle"], TEST_FLOW_SPEED, stiffness=1.0)
        torch.testing.assert_close(sample["single_drag"], torch.full_like(sample["single_drag"], expected))

    # 真实 torch 序列化验证新 checkpoint 保存的是参数，不是外部路径或求解缓存。
    checkpoint = tmp_path / "physics.pt"
    torch.save({"physical_config": physical.to_dict()}, checkpoint)
    restored = load_physical_config(torch.load(checkpoint, weights_only=False)["physical_config"])
    assert restored.to_dict() == physical.to_dict()
    assert restored.single_drag_for_angle(75, TEST_FLOW_SPEED, stiffness=1.0) == physical.single_drag_for_angle(75, TEST_FLOW_SPEED, stiffness=1.0)


def test_v2_coordinates_are_validated_against_rotated_anchor(tmp_path: Path) -> None:
    """v2 不能绕过几何契约：坐标与锚点及转盘余角不一致必须被拒绝。"""

    path = tmp_path / "bad_geometry.jsonl"
    record = _record(15)
    record["plants"][0]["x"] += 0.1
    _write_records(path, [record])
    with pytest.raises(ValueError, match="与 x/y 不一致"):
        HydroDataset(path)


def test_mirror_recomputes_force_from_original_dataset_config(tmp_path: Path) -> None:
    """镜像使用所属文件的参数，兼容预定义三集的 ConcatDataset/Subset 嵌套。"""

    path = tmp_path / "mirror.jsonl"
    _write_records(path, [_record(15)])
    dataset = HydroDataset(path)
    subset = Subset(ConcatDataset([dataset, dataset]), [1])
    augmented = MirrorTrainingDataset(subset)
    mirrored = augmented[1]
    expected = dataset.physical_config.single_drag_for_angle(345, TEST_FLOW_SPEED, stiffness=1.0)
    assert mirrored["angle"] == 345
    assert mirrored["target_drag"] == dataset[0]["target_drag"]
    torch.testing.assert_close(mirrored["single_drag"], torch.full_like(mirrored["single_drag"], expected))
    assert mirrored["single_drag"][0] != dataset[0]["single_drag"][0]


def test_v2_training_and_checkpoint_evaluation_use_calculated_physics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 CLI 完成非六角三集训练、镜像增强和 checkpoint 复评，产物仅在临时目录。"""

    from model import evaluate as evaluate_module
    from model import train as train_module

    # 三集使用独立 ID 的非六角工况；配置路径和参数仍通过标准 YAML 传入。
    paths = {}
    for role, angle in (("train", 15), ("validation", 75), ("test", 315)):
        paths[role] = tmp_path / f"{role}.jsonl"
        _write_records(paths[role], [_record(angle)])
    config = load_config(None)
    config["mode"] = "predefined"
    config["data"]["physics_config_path"] = str(
        train_module.PROJECT_ROOT / "model" / "configs" / "physical.yaml"
    )
    config["predefined_split"] = {f"{role}_dataset_path": str(path) for role, path in paths.items()}
    config["model"].update({"d_model": 16, "n_heads": 2, "n_layers": 1,
                             "ffn_dim": 32, "dropout": 0.0})
    config["training"].update({"device": "cpu", "batch_size": 1, "max_epochs": 1,
                                "early_stopping_patience": 1, "num_workers": 0,
                                "mirror_augmentation": True})
    managed_root = tmp_path / "artifacts"
    output_dir = managed_root / "run"
    config["output"]["artifact_dir"] = str(output_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(train_module, "DEFAULT_MANAGED_ARTIFACT_ROOT", managed_root)
    monkeypatch.setattr(sys, "argv", ["model.train", "--config", str(config_path)])
    train_module.main()

    # 新 checkpoint 的默认力来源必须是参数配置，不得保存旧状态查表。
    checkpoint_path = output_dir / "predefined" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload = checkpoint["checkpoint_metadata"]["physical_config"]
    assert payload["version"] == 3 and "states" not in payload
    assert load_physical_config(payload).to_dict() == load_physical_config().to_dict()

    # 复评不传物理 YAML：由 checkpoint 恢复参数并自动选择原测试 JSONL。
    evaluation_dir = output_dir / "evaluation"
    monkeypatch.setattr(sys, "argv", ["model.evaluate", "--checkpoint", str(checkpoint_path),
                                      "--output-dir", str(evaluation_dir), "--device", "cpu"])
    evaluate_module.main()
    context = json.loads((evaluation_dir / "evaluation_context.json").read_text(encoding="utf-8"))
    assert context["sample_count"] == 1
    assert context["dataset_path"] == str(paths["test"].resolve())
