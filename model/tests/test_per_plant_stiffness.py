"""逐株实测 EI、schema3 数据契约与镜像增强的物理/批处理回归测试。"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import ConcatDataset, Subset

from model.hydro.calculated_force import _cached_single_drag, column_angles_for_turntable
from model.hydro.data import HydroDataset, MirrorTrainingDataset, collate_hydro_samples
from model.hydro.geometry import POINT_COUNT, build_hex_coordinates
from model.hydro.physics import load_physical_config
from model.hydro.seagrass_drag import NumericalOptions, calculate_drag_coefficient_v2


# 测试默认值集中定义，临时 JSONL 仅写入 pytest 提供的目录，不改实验数据。
TEST_ANGLE = 15.5
TEST_FLOW_SPEED = 0.2
TEST_STIFFNESS = (0.5, 1.0, 2.0)  # EI，N·m²；不同株具有不同实测刚度。
TEST_GRID_INDICES = (3, 4, 18)  # 旋转后排序会交换前两株，适合检验逐株字段对齐。
TEST_SCHEMA_VERSION = 3
TEST_SAMPLE_ID = "per_plant_stiffness"
TEST_LEGACY_YOUNG_MODULUS = 12500000.0  # 用原脚本常见材料检验 EI 与 E 的换算。
TEST_REFERENCE_COVERED_HEIGHT = 0.2  # 原脚本兼容参数，固定片数时不参与汇总。
TEST_REFERENCE_SPACING = 0.0275  # 原脚本兼容参数，单位 m。
TEST_SECOND_MOMENT_DIVISOR = 12.0
TEST_FULL_ROTATION = 360
TEST_GRID_STEP = 60
TEST_SOLVER_NODES = 40  # 镜像测试的小网格仍走真实求解器，缩短回归测试运行时间。
TEST_FORCE_REL_TOLERANCE = 1.0e-10
TEST_FORCE_ABS_TOLERANCE = 1.0e-12
TEST_SOFT_EI = 0.0002  # 取可明显改变真实柔性变形的 EI，单位 N·m²。
TEST_STIFF_EI = 0.0004
TEST_ALTERNATIVE_CYLINDER_DIAMETER = 0.03  # 镜像配置与默认 D 不同，验证所属配置查找。


def _record(
    *, angle: float = TEST_ANGLE, schema: int = TEST_SCHEMA_VERSION,
    indices: tuple[int, ...] = TEST_GRID_INDICES,
    stiffness: tuple[float, ...] = TEST_STIFFNESS,
    sample_id: str = TEST_SAMPLE_ID,
) -> dict:
    """构造严格对齐的样本；angle 是度，indices 是锚点编号，stiffness 是逐株 EI。"""

    # v2/v3 坐标在六角锚点上顺时针旋转 angle%60；v1 保留固定锚点。
    residual = angle % TEST_GRID_STEP if schema >= 2 else 0
    cosine, sine = math.cos(math.radians(residual)), math.sin(math.radians(residual))
    coordinates = build_hex_coordinates()
    plants = []
    for plant_index, grid_index in enumerate(indices):
        x, y = coordinates[grid_index]
        plant = {
            "plant_index": plant_index, "grid_index": grid_index,
            "x": x * cosine + y * sine, "y": -x * sine + y * cosine,
            "angle": angle,
        }
        if schema == TEST_SCHEMA_VERSION:
            plant["stiffness"] = stiffness[plant_index]
        plants.append(plant)
    return {
        "schema_version": schema, "sample_id": sample_id, "model_id": 1,
        "angle": angle, "flow_speed": TEST_FLOW_SPEED,
        "vegetation_layout": "".join("1" if index in indices else "0" for index in range(POINT_COUNT)),
        "plants": plants,
        "measurements": {"TX": 0, "TY": 0, "TZ": 0, "FX_0": 1, "FY_0": 0, "FZ": 0},
    }


def _write_records(path: Path, records: list[dict]) -> None:
    """把 records 写入测试临时 JSONL；path 由 pytest 临时目录提供。"""

    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _rigid_payload() -> dict:
    """保留全部默认物理量，仅使用刚性参考分支隔离数据契约测试。"""

    payload = load_physical_config().to_dict()
    payload["solver"]["max_iterations"] = 0
    return payload


def test_default_physical_config_roundtrips_without_global_modulus(tmp_path: Path) -> None:
    """真实序列化往返后保留版本、单位、用户几何与全部求解设置。"""

    physical = load_physical_config()
    payload = physical.to_dict()
    assert payload["version"] == TEST_SCHEMA_VERSION
    assert payload["material"] == {"stiffness_unit": "N*m^2"}
    assert payload["geometry"]["cylinder_diameter_m"] == 0.015
    path = tmp_path / "physical.pt"
    torch.save(payload, path)
    restored = load_physical_config(torch.load(path, weights_only=False))
    assert restored.to_dict() == payload
    assert restored.single_drag_for_angle(TEST_ANGLE, TEST_FLOW_SPEED, TEST_STIFFNESS[0]) == physical.single_drag_for_angle(
        TEST_ANGLE, TEST_FLOW_SPEED, TEST_STIFFNESS[0],
    )


def test_ei_conversion_matches_original_matlab_compatible_solver() -> None:
    """EI=E*h*t³/12 适配得到与原脚本接口一致的整株力，没有再次乘片数。"""

    payload = load_physical_config().to_dict()
    geometry, fluid, drag, solver = (payload[key] for key in ("geometry", "fluid", "drag", "solver"))
    measured_ei = (
        TEST_LEGACY_YOUNG_MODULUS * geometry["blade_height_m"]
        * geometry["blade_thickness_m"] ** 3 / TEST_SECOND_MOMENT_DIVISOR
    )
    # 调用原 MATLAB 参数顺序的公共函数，独立检查 EI→E 适配与总力汇总。
    _, original_force, _, _ = calculate_drag_coefficient_v2(
        geometry["cylinder_diameter_m"], geometry["cylinder_height_m"], TEST_LEGACY_YOUNG_MODULUS,
        geometry["blade_length_m"], geometry["blade_thickness_m"], geometry["blade_height_m"],
        TEST_REFERENCE_COVERED_HEIGHT, TEST_REFERENCE_SPACING, TEST_FLOW_SPEED,
        fluid["water_density_kg_m3"], drag["blade_drag_coefficient"], drag["cylinder_drag_coefficient"],
        column_angles_for_turntable(TEST_ANGLE), solver["max_iterations"], solver["tolerance_m"],
        solver["shielding_type"], solver["min_shielding"], solver["max_shielding"],
        options=NumericalOptions(
            nodes=solver["nodes"], blades_per_column=geometry["blades_per_column"],
            initial_angle_difference_deg=solver["initial_angle_difference_deg"],
            enable_shielding=solver["enable_shielding"],
        ),
    )
    actual = load_physical_config(payload).single_drag_for_angle(TEST_ANGLE, TEST_FLOW_SPEED, measured_ei)
    assert actual == pytest.approx(original_force, rel=TEST_FORCE_REL_TOLERANCE, abs=TEST_FORCE_ABS_TOLERANCE)


def test_force_cache_distinguishes_stiffness_and_normalizes_angle() -> None:
    """不同 EI 的真实柔性力不同；同一 EI 和整周等价角则命中同一个缓存键。"""

    _cached_single_drag.cache_clear()
    physical = load_physical_config()
    softer = physical.single_drag_for_angle(0, TEST_FLOW_SPEED, TEST_SOFT_EI)
    assert physical.single_drag_for_angle(TEST_FULL_ROTATION, TEST_FLOW_SPEED, TEST_SOFT_EI) == softer
    stiffer = physical.single_drag_for_angle(0, TEST_FLOW_SPEED, TEST_STIFF_EI)
    assert softer != pytest.approx(stiffer, rel=TEST_FORCE_REL_TOLERANCE)
    assert _cached_single_drag.cache_info().hits == 1
    assert _cached_single_drag.cache_info().misses == 2


@pytest.mark.parametrize("stiffness", [None, True, False, 0, -1, math.nan, math.inf, -math.inf, "1"])
def test_version3_rejects_missing_or_invalid_runtime_ei(stiffness) -> None:
    """新物理配置不允许缺失、布尔、非数值、非有限或非正 EI。"""

    with pytest.raises(ValueError, match="stiffness"):
        load_physical_config().single_drag_for_angle(TEST_ANGLE, TEST_FLOW_SPEED, stiffness)


def test_old_parameter_and_lookup_versions_keep_their_semantics() -> None:
    """v2 无 EI 时保持全局 E，有 EI 时覆盖；v1 无论 EI 多少都查询旧表。"""

    payload = load_physical_config().to_dict()
    payload["version"] = 2
    payload["material"] = {"young_modulus_pa": TEST_LEGACY_YOUNG_MODULUS}
    old_calculated = load_physical_config(payload)
    geometry = payload["geometry"]
    equivalent_ei = TEST_LEGACY_YOUNG_MODULUS * geometry["blade_height_m"] * geometry["blade_thickness_m"] ** 3 / TEST_SECOND_MOMENT_DIVISOR
    assert old_calculated.single_drag_for_angle(TEST_ANGLE, TEST_FLOW_SPEED) == pytest.approx(
        old_calculated.single_drag_for_angle(TEST_ANGLE, TEST_FLOW_SPEED, equivalent_ei),
        rel=TEST_FORCE_REL_TOLERANCE,
    )
    assert old_calculated.single_drag_for_angle(TEST_ANGLE, TEST_FLOW_SPEED, TEST_STIFFNESS[0]) != pytest.approx(
        old_calculated.single_drag_for_angle(TEST_ANGLE, TEST_FLOW_SPEED), rel=TEST_FORCE_REL_TOLERANCE,
    )
    assert load_physical_config(old_calculated.to_dict()).to_dict() == payload
    table = {
        "version": 1, "drag_unit": "N", "states": {
            str(state): {"angles_deg": angles, "single_drag_by_flow_speed": {
                str(speed): float(state) for speed in (0.1, 0.2, 0.3, 0.4)
            }} for state, angles in ((1, [0, 120, 240]), (2, [60, 180, 300]))
        },
    }
    old_table = load_physical_config(table)
    assert old_table.single_drag_for_angle(0, TEST_FLOW_SPEED) == old_table.single_drag_for_angle(
        0, TEST_FLOW_SPEED, TEST_STIFFNESS[2],
    )


def test_schema3_real_angles_features_and_aligned_batch_padding(tmp_path: Path) -> None:
    """v3 小数角不截断，EI 按株输入；padding 为零且三个全局特征完整保留。"""

    path = tmp_path / "mixed_lengths.jsonl"
    _write_records(path, [_record(), _record(indices=(18,), stiffness=(2.0,), sample_id="short")])
    dataset = HydroDataset(path, physical_config=_rigid_payload())
    sample, short = dataset[0], dataset[1]
    batch = collate_hydro_samples([short, sample])
    assert sample["plant_angles"].dtype == torch.float32
    assert sample["plant_angles"].tolist() == [TEST_ANGLE] * len(TEST_STIFFNESS)
    torch.testing.assert_close(sample["plant_stiffness"], torch.tensor(TEST_STIFFNESS))
    torch.testing.assert_close(sample["global_features"], torch.tensor([
        TEST_FLOW_SPEED, math.sin(math.radians(TEST_ANGLE)), math.cos(math.radians(TEST_ANGLE)),
    ]))
    assert batch["global_features"].shape == (2, 3)
    assert batch["angle"][0].item() == TEST_ANGLE
    assert batch["plant_mask"][0].tolist() == [True, False, False]
    for field in ("plant_stiffness", "single_drag", "plant_angles"):
        assert torch.count_nonzero(batch[field][0, 1:]) == 0
    broken = deepcopy(sample)
    broken["plant_stiffness"] = broken["plant_stiffness"][:1]
    with pytest.raises(ValueError, match="逐株对齐"):
        collate_hydro_samples([broken])


@pytest.mark.parametrize("stiffness", [True, False, 0, -1, math.nan, math.inf, "1", None])
def test_schema3_strictly_rejects_invalid_ei(tmp_path: Path, stiffness) -> None:
    """JSONL 的新 schema 不接受无效 EI，不能自动替换为历史默认值。"""

    record = _record()
    record["plants"][0]["stiffness"] = stiffness
    path = tmp_path / "bad_ei.jsonl"
    _write_records(path, [record])
    with pytest.raises(ValueError, match="stiffness"):
        HydroDataset(path, physical_config=_rigid_payload())


@pytest.mark.parametrize("mutation", [
    lambda record: record["plants"][0].pop("stiffness"),
    lambda record: record["plants"][0].update({"stifness": 1}),
    lambda record: record["plants"][0].update({"angle": TEST_ANGLE + 1}),
    lambda record: record["plants"][0].update({"x": 100}),
    lambda record: record.update({"vegetation_layout": "2" * POINT_COUNT}),
    lambda record: record.update({"vegetation_layout": " " + record["vegetation_layout"]}),
    lambda record: record.update({"angle": math.inf}),
])
def test_schema3_retains_strict_geometry_orientation_and_fields(tmp_path: Path, mutation) -> None:
    """新字段不削弱37位布局、同向角、坐标和严格字段集合的原有检查。"""

    record = _record()
    mutation(record)
    path = tmp_path / "bad_contract.jsonl"
    _write_records(path, [record])
    with pytest.raises(ValueError):
        HydroDataset(path, physical_config=_rigid_payload())


def test_legacy_jsonl_defaults_ei_in_memory_and_keeps_integer_angle_dtype(tmp_path: Path) -> None:
    """v1/v2 未迁移副本继续可读；EI=1 的补值不写回源文件，小数角仅 v3 支持。"""

    path = tmp_path / "legacy.jsonl"
    _write_records(path, [_record(angle=0, schema=1, sample_id="v1"), _record(angle=15, schema=2, sample_id="v2")])
    original_text = path.read_text(encoding="utf-8")
    dataset = HydroDataset(path, physical_config=_rigid_payload())
    for sample in dataset:
        assert sample["plant_angles"].dtype == torch.long
        assert sample["stiffness_defaulted"] is True
        assert sample["stiffness_unit"] == "N*m^2"
        torch.testing.assert_close(sample["plant_stiffness"], torch.ones(len(TEST_STIFFNESS)))
    assert collate_hydro_samples(list(dataset))["plant_angles"].dtype == torch.long
    assert path.read_text(encoding="utf-8") == original_text
    # 混合新旧 schema 的一个 batch 必须提升角 dtype，保留新数据的小数角。
    v3_path = tmp_path / "v3.jsonl"
    _write_records(v3_path, [_record()])
    batch = collate_hydro_samples([dataset[0], HydroDataset(v3_path, physical_config=_rigid_payload())[0]])
    assert batch["plant_angles"].dtype == torch.float32
    assert batch["plant_angles"][1, 0].item() == TEST_ANGLE


def test_mirror_preserves_ei_alignment_and_recomputes_angle_features_and_physics(tmp_path: Path) -> None:
    """嵌套 Subset/ConcatDataset 使用所属物理参数，逐株 EI 与镜像角同步求力。"""

    path = tmp_path / "mirror.jsonl"
    _write_records(path, [_record()])
    first = HydroDataset(path, physical_config=_rigid_payload())
    payload = load_physical_config().to_dict()
    payload["geometry"]["cylinder_diameter_m"] = TEST_ALTERNATIVE_CYLINDER_DIAMETER
    payload["solver"]["nodes"] = TEST_SOLVER_NODES
    actual_dataset = HydroDataset(path, physical_config=payload)
    nested = Subset(Subset(ConcatDataset([first, actual_dataset]), [1]), [0])
    mirrored = MirrorTrainingDataset(nested)[1]
    original = actual_dataset[0]
    expected_positions = original["positions"].clone()
    expected_positions[:, 1] *= -1
    order = sorted(range(len(TEST_GRID_INDICES)), key=lambda index: (
        -float(expected_positions[index, 0]), -float(expected_positions[index, 1]),
    ))
    assert order != list(range(len(TEST_GRID_INDICES)))
    torch.testing.assert_close(mirrored["positions"], expected_positions[order])
    torch.testing.assert_close(mirrored["plant_stiffness"], original["plant_stiffness"][order])
    expected_angle = (-TEST_ANGLE) % TEST_FULL_ROTATION
    assert mirrored["angle"] == expected_angle
    torch.testing.assert_close(mirrored["global_features"], torch.tensor([
        TEST_FLOW_SPEED, math.sin(math.radians(expected_angle)), math.cos(math.radians(expected_angle)),
    ]))
    expected_forces = [actual_dataset.physical_config.single_drag_for_angle(
        expected_angle, TEST_FLOW_SPEED, float(stiffness),
    ) for stiffness in mirrored["plant_stiffness"]]
    torch.testing.assert_close(mirrored["single_drag"], torch.tensor(expected_forces))
    assert mirrored["target_drag"] == original["target_drag"]
    torch.testing.assert_close(original["plant_stiffness"], torch.tensor(TEST_STIFFNESS))


def test_version3_rejects_global_modulus_and_wrong_stiffness_unit() -> None:
    """新配置只声明 EI 单位，不能混入全局 E 或把无单位比例当作实测 EI。"""

    payload = load_physical_config().to_dict()
    payload["material"]["young_modulus_pa"] = TEST_LEGACY_YOUNG_MODULUS
    with pytest.raises(ValueError, match="material"):
        load_physical_config(payload)
    payload["material"] = {"stiffness_unit": "Pa"}
    with pytest.raises(ValueError, match="stiffness_unit"):
        load_physical_config(payload)
