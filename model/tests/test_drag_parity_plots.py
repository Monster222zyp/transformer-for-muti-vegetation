"""全测试集模型与逐株叠加散点图的数据、尺度和指标测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.figure import Figure

import model.training.visualization as visualization_module
from model.training.visualization import write_drag_parity_plots


# 测试默认值集中声明；故意包含负实测值，以区分原始测量和训练归零后的标签。
TEST_PREFIX = "cv"
TEST_TITLE = "Held-out test"
TEST_AXIS_MODES = ("independent", "shared")
EXTREME_BASELINE_MULTIPLIER = 1000.0  # 模拟严重高估的基线，验证不会压缩默认模型图。
INVALID_AXIS_MODE = "unknown"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MINIMUM_PNG_BYTES = 1_000
TEST_PREDICTIONS = (
    {"raw_target_drag": -0.2, "target_drag": 0.0, "predicted_drag": 0.1, "isolated_drag": 0.5},
    {"raw_target_drag": 1.0, "target_drag": 1.0, "predicted_drag": 1.2, "isolated_drag": 1.8},
    {"raw_target_drag": 2.0, "target_drag": 2.0, "predicted_drag": 1.8, "isolated_drag": 3.0},
)
CONSTANT_MEASUREMENTS = ((0.0,), (0.0, 0.0), (1.0, 1.0))
INVALID_FIELD_VALUES = (
    ("raw_target_drag", float("nan")),
    ("predicted_drag", float("inf")),
    ("isolated_drag", "not-a-number"),
)


def _capture_axes(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """记录真实坐标轴供断言使用；参数 monkeypatch 负责在测试结束后恢复替换。"""

    axes: list[Any] = []
    original_subplots = visualization_module.plt.subplots

    def capture_subplots(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """参数原样传给 matplotlib；返回真实 Figure 和 Axes，并保留 Axes 引用。"""

        figure, axis = original_subplots(*args, **kwargs)
        axes.append(axis)
        return figure, axis

    monkeypatch.setattr(visualization_module.plt, "subplots", capture_subplots)
    return axes


@pytest.mark.parametrize("axis_mode", TEST_AXIS_MODES)
def test_parity_plots_keep_raw_measurements_and_compare_both_series(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    axis_mode: str,
) -> None:
    """参数 axis_mode 选择范围模式；两种模式均保留原始点、图内等比例与参考线。"""

    predictions = deepcopy(list(TEST_PREDICTIONS))
    original_predictions = deepcopy(predictions)
    axes = _capture_axes(monkeypatch)
    figures_before = visualization_module.plt.get_fignums()
    destination = tmp_path / "new_artifacts"

    paths = write_drag_parity_plots(
        predictions, destination, prefix=TEST_PREFIX, title=TEST_TITLE, axis_mode=axis_mode,
    )

    # 文件必须是真实 PNG；保存完成后不遗留 matplotlib Figure，也不修改输入。
    assert set(paths) == {"model", "baseline"}
    for name, path in paths.items():
        assert path == destination / f"{TEST_PREFIX}_drag_parity_{name}.png"
        assert path.stat().st_size > MINIMUM_PNG_BYTES
        assert path.read_bytes().startswith(PNG_SIGNATURE)
    assert visualization_module.plt.get_fignums() == figures_before
    assert predictions == original_predictions

    # 每条记录必须出现一次，负实测不应被悄悄替换成归零后的 target_drag。
    assert len(axes) == 2
    expected_measurement = [row["raw_target_drag"] for row in TEST_PREDICTIONS]
    for axis, prediction_field in zip(axes, ("predicted_drag", "isolated_drag")):
        points = axis.collections[0].get_offsets()
        assert len(points) == len(TEST_PREDICTIONS)
        np.testing.assert_allclose(points[:, 0], expected_measurement)
        np.testing.assert_allclose(points[:, 1], [row[prediction_field] for row in TEST_PREDICTIONS])
        np.testing.assert_allclose(axis.get_xlim(), axis.get_ylim())
        np.testing.assert_allclose(axis.lines[0].get_xdata(), axis.get_xlim())
        np.testing.assert_allclose(axis.lines[0].get_ydata(), axis.get_ylim())
        assert axis.get_aspect() == 1.0
        assert axis.get_xlabel() == "Measured drag (N)"
        assert "n = 3" in axis.texts[0].get_text()
        assert TEST_TITLE in axis.get_title()
        assert axis_mode in axis.texts[1].get_text()
        # 独立范围也必须覆盖该图全部实测与预测，不能靠隐藏极值改善外观。
        assert axis.get_xlim()[0] < min(expected_measurement)
        assert axis.get_xlim()[1] > max(
            *expected_measurement, *(row[prediction_field] for row in TEST_PREDICTIONS),
        )
    if axis_mode == "shared":
        np.testing.assert_allclose(axes[0].get_xlim(), axes[1].get_xlim())
    else:
        assert axes[0].get_xlim()[1] < axes[1].get_xlim()[1]

    # 直接核对手算的平均绝对误差，防止图内指标误用归零标签或基线指标。
    assert "MAE = 0.2333 N" in axes[0].texts[0].get_text()
    assert "RMSE = 0.238 N" in axes[0].texts[0].get_text()
    assert "$R^2$ = 0.930" in axes[0].texts[0].get_text()
    assert "MAE = 0.8333 N" in axes[1].texts[0].get_text()


def test_default_model_axis_is_unaffected_by_extreme_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认模型范围只由实测与模型预测决定；tmp_path 保存图，monkeypatch 捕获坐标。"""

    axes = _capture_axes(monkeypatch)
    predictions = deepcopy(list(TEST_PREDICTIONS))
    write_drag_parity_plots(predictions, tmp_path / "original")

    # 仅放大基线预测，模型图的坐标范围与误差指标应完全不变，基线极值仍可见。
    for row in predictions:
        row["isolated_drag"] *= EXTREME_BASELINE_MULTIPLIER
    write_drag_parity_plots(predictions, tmp_path / "extreme")
    np.testing.assert_allclose(axes[0].get_xlim(), axes[2].get_xlim())
    assert axes[0].texts[0].get_text() == axes[2].texts[0].get_text()
    assert axes[3].get_ylim()[1] > max(row["isolated_drag"] for row in predictions)


def test_parity_axis_mode_can_be_overridden_by_cli() -> None:
    """YAML 默认独立范围，命令行覆盖会写回配置供绘图和快照共同使用。"""

    from model import train as train_module

    config = train_module.load_config(train_module.DEFAULT_CONFIG_PATH)
    assert config["output"]["drag_parity_axis_mode"] == "independent"
    args = train_module.parse_args(["--mode", "cv", "--drag-parity-axis-mode", "shared"])
    train_module._apply_cli_overrides(config, args)
    assert config["output"]["drag_parity_axis_mode"] == "shared"


def test_parity_plots_reject_invalid_axis_mode_before_writing(tmp_path: Path) -> None:
    """参数 tmp_path 提供临时目录；未知坐标模式必须先报错，不留下部分图片。"""

    destination = tmp_path / "invalid_axis_mode"
    with pytest.raises(ValueError, match="axis_mode"):
        write_drag_parity_plots(TEST_PREDICTIONS, destination, axis_mode=INVALID_AXIS_MODE)
    assert not destination.exists()


@pytest.mark.parametrize("measurements", CONSTANT_MEASUREMENTS)
def test_parity_plots_handle_constant_measurements_and_zero_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    measurements: tuple[float, ...],
) -> None:
    """单样本、常量测量和全零阻力也应写图，R² 显式标为未定义。"""

    axes = _capture_axes(monkeypatch)
    predictions = [
        {"raw_target_drag": value, "predicted_drag": value, "isolated_drag": value}
        for value in measurements
    ]

    paths = write_drag_parity_plots(predictions, tmp_path)

    assert all(path.is_file() for path in paths.values())
    for axis in axes:
        lower, upper = axis.get_xlim()
        assert lower < upper
        assert "undefined" in axis.texts[0].get_text()


@pytest.mark.parametrize("field,bad_value", INVALID_FIELD_VALUES)
def test_parity_plots_validate_both_series_before_writing(
    tmp_path: Path,
    field: str,
    bad_value: Any,
) -> None:
    """两图所需字段有非法值时应先报错，不留下看似成功的部分图片。"""

    predictions = deepcopy(list(TEST_PREDICTIONS))
    predictions[0][field] = bad_value
    destination = tmp_path / "invalid_artifacts"

    with pytest.raises(ValueError, match=field):
        write_drag_parity_plots(predictions, destination)

    assert not destination.exists()


def test_parity_plots_require_raw_measurements(tmp_path: Path) -> None:
    """缺少原始实测时明确报错，避免静默切换为另一种标签口径。"""

    predictions = deepcopy(list(TEST_PREDICTIONS))
    del predictions[0]["raw_target_drag"]

    with pytest.raises(ValueError, match="raw_target_drag"):
        write_drag_parity_plots(predictions, tmp_path)


def test_parity_plots_reject_empty_input(tmp_path: Path) -> None:
    """空测试集不能产生具有误导性的空散点图。"""

    with pytest.raises(ValueError, match="predictions 不能为空"):
        write_drag_parity_plots([], tmp_path)


def test_parity_plots_release_figure_after_save_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """磁盘写入失败时同样释放 Figure，避免训练异常路径泄漏图形对象。"""

    figures_before = visualization_module.plt.get_fignums()

    def fail_savefig(self: Figure, *args: Any, **kwargs: Any) -> None:
        """模拟保存失败；self 是当前 Figure，其余参数是输出路径和绘图设置。"""

        del self, args, kwargs
        raise OSError("模拟磁盘写入失败")

    monkeypatch.setattr(Figure, "savefig", fail_savefig)
    with pytest.raises(OSError, match="模拟磁盘写入失败"):
        write_drag_parity_plots(list(TEST_PREDICTIONS), tmp_path)

    assert visualization_module.plt.get_fignums() == figures_before
