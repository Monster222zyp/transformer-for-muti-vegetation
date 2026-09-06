"""drag 预测排序与三折线 PNG 输出的专项测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

import model.training.visualization as visualization_module
from model.training.visualization import (
    sort_drag_predictions,
    write_drag_comparison_by_flow_speed_plot,
    write_drag_comparison_plot,
)


def _prediction(
    source_index: int,
    target_drag: float,
    predicted_drag: float,
    isolated_drag: float,
) -> dict[str, float | int]:
    """构造一条最小预测记录，减少不同测试中的重复样板数据。"""

    return {
        "source_index": source_index,
        "target_drag": target_drag,
        "predicted_drag": predicted_drag,
        "isolated_drag": isolated_drag,
    }


def test_sort_drag_predictions_uses_required_priority_without_mutation() -> None:
    """排序必须依次使用 isolated、target、source，且不能改变输入列表。"""

    predictions = [
        _prediction(30, target_drag=1.0, predicted_drag=1.1, isolated_drag=2.0),
        _prediction(20, target_drag=3.0, predicted_drag=2.8, isolated_drag=1.0),
        _prediction(10, target_drag=2.0, predicted_drag=2.1, isolated_drag=1.0),
        _prediction(5, target_drag=2.0, predicted_drag=1.9, isolated_drag=1.0),
    ]
    original_order = [row["source_index"] for row in predictions]

    sorted_rows = sort_drag_predictions(predictions)

    assert [row["source_index"] for row in sorted_rows] == [5, 10, 20, 30]
    assert [row["source_index"] for row in predictions] == original_order
    assert sorted_rows[0] is not predictions[3]


def test_write_drag_comparison_plot_creates_nonempty_png(tmp_path: Path) -> None:
    """有效预测应在新建父目录中生成带 PNG 文件头的非空图像。"""

    predictions = [
        _prediction(2, target_drag=2.0, predicted_drag=2.2, isolated_drag=1.8),
        _prediction(1, target_drag=1.0, predicted_drag=0.9, isolated_drag=1.1),
    ]
    destination = tmp_path / "fold_1" / "validation_drag_comparison.png"

    returned_path = write_drag_comparison_plot(
        predictions,
        destination,
        title="Fold 1 validation drag comparison",
    )

    assert returned_path == destination
    assert destination.stat().st_size > 1_000
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_sort_drag_predictions_rejects_empty_input() -> None:
    """空集合没有可绘制的 sample，应给出直接且清楚的错误。"""

    with pytest.raises(ValueError, match="predictions 不能为空"):
        sort_drag_predictions([])


@pytest.mark.parametrize(
    "missing_field",
    ["source_index", "target_drag", "predicted_drag", "isolated_drag"],
)
def test_sort_drag_predictions_reports_each_missing_field(
    missing_field: str,
) -> None:
    """四个必需字段任一缺失时，错误信息必须指出具体字段。"""

    row = _prediction(1, target_drag=1.0, predicted_drag=1.1, isolated_drag=0.9)
    del row[missing_field]

    with pytest.raises(ValueError, match=missing_field):
        sort_drag_predictions([row])


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), "not-a-number"])
def test_sort_drag_predictions_rejects_non_finite_or_non_numeric_values(
    bad_value: object,
) -> None:
    """非法 drag 不能悄悄进入 matplotlib，否则图中可能出现无提示断线。"""

    row = _prediction(1, target_drag=1.0, predicted_drag=1.1, isolated_drag=0.9)
    row["predicted_drag"] = bad_value  # type: ignore[assignment]

    with pytest.raises(ValueError, match="predicted_drag"):
        sort_drag_predictions([row])


def test_write_drag_comparison_plot_requires_png_suffix(tmp_path: Path) -> None:
    """固定 PNG 格式，防止文件扩展名与实际内容不一致。"""

    with pytest.raises(ValueError, match=r"\.png"):
        write_drag_comparison_plot(
            [_prediction(1, 1.0, 1.1, 0.9)],
            tmp_path / "comparison.svg",
        )


def test_write_drag_comparison_by_flow_speed_creates_four_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """分流速图必须固定生成四个面板，并在每个面板画三条阻力线。"""

    predictions = []
    for source_index, flow_speed in enumerate((0.1, 0.2, 0.3, 0.4), start=1):
        row = _prediction(
            source_index,
            target_drag=float(source_index),
            predicted_drag=float(source_index) + 0.1,
            isolated_drag=float(source_index) + 0.2,
        )
        row["flow_speed"] = flow_speed
        predictions.append(row)

    captured: dict[str, object] = {}
    real_subplots = visualization_module.plt.subplots

    def capture_subplots(*args: object, **kwargs: object) -> tuple[object, object]:
        """保留真实 Figure/Axes，以便在写图后核对子图结构。"""

        figure, axes = real_subplots(*args, **kwargs)
        captured["axes"] = axes
        return figure, axes

    monkeypatch.setattr(visualization_module.plt, "subplots", capture_subplots)
    destination = tmp_path / "test_drag_comparison_by_flow_speed.png"

    returned_path = write_drag_comparison_by_flow_speed_plot(
        predictions,
        destination,
        title="Test drag comparison by flow speed",
    )

    assert returned_path == destination
    assert destination.stat().st_size > 1_000
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    axes = captured["axes"]
    assert getattr(axes, "shape") == (2, 2)
    assert [axis.get_title() for axis in axes.flat] == [
        "0.1 m/s (n=1)",
        "0.2 m/s (n=1)",
        "0.3 m/s (n=1)",
        "0.4 m/s (n=1)",
    ]
    assert all(len(axis.lines) == 3 for axis in axes.flat)


def test_write_drag_comparison_by_flow_speed_marks_missing_panel(
    tmp_path: Path,
) -> None:
    """某档流速缺样本时仍应成功输出固定四格图，而不是伪造曲线。"""

    prediction = _prediction(1, 1.0, 1.1, 0.9)
    prediction["flow_speed"] = 0.1
    destination = tmp_path / "comparison_by_flow_speed.png"

    write_drag_comparison_by_flow_speed_plot([prediction], destination)

    assert destination.is_file()
    assert destination.stat().st_size > 1_000
