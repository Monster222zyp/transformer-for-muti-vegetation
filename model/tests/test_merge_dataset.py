"""schema3 合并保留逐株 EI，并拒绝不可训练的刚度字段。"""

import json

import pytest

from merge_dataset import load_records


# 测试默认 EI 用 N·m²；所有生成文件仅位于 pytest 临时目录。
TEST_STIFFNESS = 0.5
TEST_SCHEMA_VERSION = 3


def _inputs(tmp_path, first_stiffness):
    """创建两个唯一 ID 的最小合并输入，第一份用于覆盖边界值。"""

    paths = (tmp_path / "first.jsonl", tmp_path / "second.jsonl")
    for index, (path, stiffness) in enumerate(zip(paths, (first_stiffness, TEST_STIFFNESS))):
        record = {"schema_version": TEST_SCHEMA_VERSION, "sample_id": f"sample_{index}",
                  "plants": [{"stiffness": stiffness}]}
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return paths


def test_merge_preserves_each_plant_stiffness_and_input_order(tmp_path):
    """合并不是历史迁移，不能把不同 EI 自动改为 1。"""

    records = load_records(_inputs(tmp_path, 2.0))
    assert [record["sample_id"] for record in records] == ["sample_0", "sample_1"]
    assert [record["plants"][0]["stiffness"] for record in records] == [2.0, TEST_STIFFNESS]


@pytest.mark.parametrize("invalid", [None, True, "1", 0.0, -1.0, float("nan"), float("inf")])
def test_schema3_merge_rejects_invalid_per_plant_stiffness(tmp_path, invalid):
    """拒绝缺失、类型错误、非正或非有限的 EI，避免生成错误合并结果。"""

    with pytest.raises(ValueError, match="stiffness.*正的有限数"):
        load_records(_inputs(tmp_path, invalid))
