"""训练产物目录清理、安全边界和入口调用顺序的回归测试。"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import model.train as train_module
import model.training.artifacts as artifact_module
from model.training.artifacts import (
    ARTIFACT_OWNERSHIP_MARKER,
    ARTIFACT_OWNERSHIP_MARKER_CONTENT,
    prepare_fresh_artifact_directory,
    prepare_resume_artifact_directory,
)


def _write_valid_marker(directory: Path) -> None:
    """在测试目录写入内容正确的 ownership marker。

    Args:
        directory: 已存在的模拟自定义产物目录。
    """

    (directory / ARTIFACT_OWNERSHIP_MARKER).write_text(
        ARTIFACT_OWNERSHIP_MARKER_CONTENT,
        encoding="utf-8",
    )


def test_managed_artifact_directory_removes_all_old_results(tmp_path: Path) -> None:
    """受管目录应删除旧 fold、final checkpoint 和任意嵌套残留。"""

    managed_root = tmp_path / "model" / "artifacts"
    target = managed_root / "run_001"
    (target / "fold_4").mkdir(parents=True)
    (target / "fold_4" / "last.pt").write_bytes(b"old fold")
    (target / "final_model.pt").write_bytes(b"old final")
    sibling = managed_root / "run_002"
    sibling.mkdir(parents=True)
    (sibling / "keep.txt").write_text("keep", encoding="utf-8")

    result = prepare_fresh_artifact_directory(
        target,
        managed_root=managed_root,
        protected_paths=[tmp_path / "source"],
    )

    assert result == target.resolve()
    assert {path.name for path in target.iterdir()} == {ARTIFACT_OWNERSHIP_MARKER}
    assert (sibling / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_external_directory_requires_marker_only_after_it_becomes_nonempty(
    tmp_path: Path,
) -> None:
    """空的外部目录可初始化，之后必须依靠有效 marker 才能再次清理。"""

    managed_root = tmp_path / "managed"
    external_target = tmp_path / "external" / "run"
    external_target.mkdir(parents=True)

    prepare_fresh_artifact_directory(
        external_target,
        managed_root=managed_root,
        protected_paths=[],
    )
    (external_target / "old.json").write_text("{}", encoding="utf-8")
    prepare_fresh_artifact_directory(
        external_target,
        managed_root=managed_root,
        protected_paths=[],
    )

    assert not (external_target / "old.json").exists()
    assert (external_target / ARTIFACT_OWNERSHIP_MARKER).is_file()


def test_external_nonempty_directory_without_valid_marker_is_rejected(
    tmp_path: Path,
) -> None:
    """任意普通非空目录不能仅因被传入 CLI 就获得递归删除权限。"""

    target = tmp_path / "personal-files"
    target.mkdir()
    important_file = target / "important.txt"
    important_file.write_text("do not delete", encoding="utf-8")

    with pytest.raises(ValueError, match="缺少有效"):
        prepare_fresh_artifact_directory(
            target,
            managed_root=tmp_path / "managed",
            protected_paths=[],
        )

    assert important_file.read_text(encoding="utf-8") == "do not delete"


def test_protected_path_and_filesystem_root_are_rejected(tmp_path: Path) -> None:
    """包含源码/输入的祖先目录和文件系统根目录永远不能成为清理目标。"""

    protected_file = tmp_path / "project" / "input.csv"
    protected_file.parent.mkdir()
    protected_file.write_text("input", encoding="utf-8")

    with pytest.raises(ValueError, match="包含受保护路径"):
        prepare_fresh_artifact_directory(
            tmp_path,
            managed_root=tmp_path / "managed",
            protected_paths=[protected_file],
        )
    with pytest.raises(ValueError, match="文件系统根目录"):
        prepare_fresh_artifact_directory(
            Path(tmp_path.anchor),
            managed_root=tmp_path / "managed",
            protected_paths=[],
        )


def test_cleanup_root_cannot_be_a_symbolic_link(tmp_path: Path) -> None:
    """清理根若是 symbolic link，应拒绝而不是删除链接目标中的内容。"""

    real_directory = tmp_path / "real"
    real_directory.mkdir()
    protected_file = real_directory / "keep.txt"
    protected_file.write_text("keep", encoding="utf-8")
    linked_directory = tmp_path / "linked"
    try:
        linked_directory.symlink_to(real_directory, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"当前 Windows 权限不允许创建 symbolic link：{error}")

    with pytest.raises(ValueError, match="symbolic link 或 junction"):
        prepare_fresh_artifact_directory(
            linked_directory,
            managed_root=tmp_path,
            protected_paths=[],
        )

    assert protected_file.read_text(encoding="utf-8") == "keep"


def test_child_symbolic_link_is_unlinked_without_deleting_its_target(
    tmp_path: Path,
) -> None:
    """目录内部的 symbolic link 只能删除链接自身，不能递归删除链接目标。"""

    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    protected_file = outside_directory / "keep.txt"
    protected_file.write_text("keep", encoding="utf-8")
    linked_child = managed_root / "linked-child"
    try:
        linked_child.symlink_to(outside_directory, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"当前 Windows 权限不允许创建 symbolic link：{error}")

    prepare_fresh_artifact_directory(
        managed_root,
        managed_root=managed_root,
        protected_paths=[],
    )

    assert not linked_child.exists()
    assert protected_file.read_text(encoding="utf-8") == "keep"


def test_marker_cannot_be_a_symbolic_link(tmp_path: Path) -> None:
    """marker 链接必须在清理前被拒绝，防止 write_text 覆盖目录外文件。"""

    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("keep", encoding="utf-8")
    marker = managed_root / ARTIFACT_OWNERSHIP_MARKER
    try:
        marker.symlink_to(outside_file)
    except OSError as error:
        pytest.skip(f"当前 Windows 权限不允许创建 symbolic link：{error}")

    with pytest.raises(ValueError, match="marker 必须是普通文件"):
        prepare_fresh_artifact_directory(
            managed_root,
            managed_root=managed_root,
            protected_paths=[],
        )

    assert outside_file.read_text(encoding="utf-8") == "keep"


def test_cleanup_failure_stops_before_new_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任何旧文件删除失败都必须终止初始化，并保留 marker 供用户重试。"""

    target = tmp_path / "external"
    target.mkdir()
    _write_valid_marker(target)
    old_file = target / "locked.pt"
    old_file.write_bytes(b"locked")

    def fail_removal(path: Path) -> None:
        """模拟 Windows 文件占用导致的删除失败。"""

        raise PermissionError(f"locked: {path}")

    monkeypatch.setattr(artifact_module, "_remove_artifact_entry", fail_removal)
    with pytest.raises(RuntimeError, match="训练尚未开始"):
        prepare_fresh_artifact_directory(
            target,
            managed_root=tmp_path / "managed",
            protected_paths=[],
        )

    assert old_file.exists()
    assert (target / ARTIFACT_OWNERSHIP_MARKER).is_file()


def test_resume_directory_never_removes_existing_files(tmp_path: Path) -> None:
    """resume 初始化只能创建目录，不能删除 checkpoint 或 history。"""

    target = tmp_path / "resume"
    target.mkdir()
    checkpoint = target / "last.pt"
    history = target / "history.json"
    checkpoint.write_bytes(b"checkpoint")
    history.write_text("[]", encoding="utf-8")

    result = prepare_resume_artifact_directory(target)

    assert result == target.resolve()
    assert checkpoint.read_bytes() == b"checkpoint"
    assert history.read_text(encoding="utf-8") == "[]"


def _minimal_main_config(output_dir: Path) -> dict[str, Any]:
    """创建训练入口编排测试所需的最小配置。

    Args:
        output_dir: 临时测试产物目录。

    Returns:
        包含数据、输出和 flow_speed 模式字段的配置字典。
    """

    return {
        "seed": 1,
        "data": {
            "dataset_path": str(output_dir.parent / "input.jsonl"),
            "physics_config_path": str(output_dir.parent / "physical.yaml"),
            "negative_target_policy": "clamp_to_zero",
        },
        "model": {},
        "training": {},
        "cross_validation": {
            "split_mode": "flow_speed",
            "n_splits": 5,
            "validation_fraction": 0.2,
        },
        "output": {"artifact_dir": str(output_dir)},
    }


def _patch_main_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset_factory: Any,
) -> None:
    """替换入口测试不关心的真实配置、物理和数据加载依赖。

    Args:
        monkeypatch: pytest 提供的临时属性替换工具。
        args: ``parse_args`` 应返回的命令行命名空间。
        config: ``load_config`` 应返回的配置对象。
        dataset_factory: 替代 :class:`HydroDataset` 的可调用对象。
    """

    monkeypatch.setattr(train_module, "parse_args", lambda: args)
    monkeypatch.setattr(train_module, "load_config", lambda path: config)
    monkeypatch.setattr(train_module, "_resolve_config_paths", lambda value: None)
    monkeypatch.setattr(train_module, "_apply_cli_overrides", lambda value, cli: None)
    monkeypatch.setattr(
        train_module,
        "load_physical_config",
        lambda path: SimpleNamespace(to_dict=lambda: {"states": {}}),
    )
    monkeypatch.setattr(train_module, "HydroDataset", dataset_factory)


def test_main_validates_inputs_before_cleaning_and_reports_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """fresh flow_speed 应先验证输入，再清旧结果，并报告 fold_0 checkpoint。"""

    managed_root = tmp_path / "managed"
    output_dir = managed_root / "run"
    output_dir.mkdir(parents=True)
    old_final = output_dir / "final_model.pt"
    old_fold = output_dir / "fold_4" / "last.pt"
    old_fold.parent.mkdir()
    old_final.write_bytes(b"old")
    old_fold.write_bytes(b"old")
    events: list[str] = []
    config = _minimal_main_config(output_dir)
    args = argparse.Namespace(
        mode="cv",
        resume_checkpoint=None,
        config=str(tmp_path / "config.yaml"),
    )

    def fake_dataset(*args: Any, **kwargs: Any) -> object:
        """确认 Dataset 构造时旧结果仍存在。"""

        del args, kwargs
        assert old_final.exists()
        events.append("dataset")
        return object()

    def fake_prepare_splits(dataset: object, received_config: dict[str, Any]) -> tuple[str, list]:
        """确认划分预检也发生在清理之前。"""

        assert dataset is not None
        assert received_config is config
        assert old_fold.exists()
        events.append("splits")
        return "flow_speed", []

    def fake_run(
        dataset: object,
        received_config: dict[str, Any],
        received_output: Path,
        prepared_splits: tuple[str, list] | None = None,
    ) -> None:
        """模拟训练，并验证运行时只剩本次 fresh run 的文件。"""

        assert dataset is not None
        assert received_config is config
        assert prepared_splits == ("flow_speed", [])
        assert not old_final.exists()
        assert not old_fold.exists()
        assert (received_output / ARTIFACT_OWNERSHIP_MARKER).is_file()
        checkpoint = received_output / "fold_0" / "best.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"new")
        events.append("run")

    _patch_main_dependencies(
        monkeypatch,
        args=args,
        config=config,
        dataset_factory=fake_dataset,
    )
    monkeypatch.setattr(train_module, "DEFAULT_MANAGED_ARTIFACT_ROOT", managed_root)
    monkeypatch.setattr(train_module, "_prepare_cross_validation_splits", fake_prepare_splits)
    monkeypatch.setattr(train_module, "run_cross_validation", fake_run)

    train_module.main()

    assert events == ["dataset", "splits", "run"]
    output = capsys.readouterr().out
    assert f"训练结果已保存至：{output_dir.resolve()}" in output
    assert f"推荐 checkpoint：{output_dir.resolve() / 'fold_0' / 'best.pt'}" in output


def test_main_keeps_old_results_when_dataset_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CSV 加载失败必须发生在清理前，保证上一次训练结果仍可恢复。"""

    managed_root = tmp_path / "managed"
    output_dir = managed_root / "run"
    output_dir.mkdir(parents=True)
    old_checkpoint = output_dir / "fold_0" / "best.pt"
    old_checkpoint.parent.mkdir()
    old_checkpoint.write_bytes(b"old")
    config = _minimal_main_config(output_dir)
    args = argparse.Namespace(
        mode="cv",
        resume_checkpoint=None,
        config=str(tmp_path / "config.yaml"),
    )

    def invalid_dataset(*args: Any, **kwargs: Any) -> object:
        """模拟 CSV 表头或内容不合法。"""

        del args, kwargs
        raise ValueError("invalid CSV")

    _patch_main_dependencies(
        monkeypatch,
        args=args,
        config=config,
        dataset_factory=invalid_dataset,
    )
    monkeypatch.setattr(train_module, "DEFAULT_MANAGED_ARTIFACT_ROOT", managed_root)

    with pytest.raises(ValueError, match="invalid CSV"):
        train_module.main()

    assert old_checkpoint.read_bytes() == b"old"


def test_main_keeps_old_results_when_split_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CV 划分参数无效时也必须在清理旧结果之前终止。"""

    managed_root = tmp_path / "managed"
    output_dir = managed_root / "run"
    output_dir.mkdir(parents=True)
    old_checkpoint = output_dir / "fold_0" / "best.pt"
    old_checkpoint.parent.mkdir()
    old_checkpoint.write_bytes(b"old")
    config = _minimal_main_config(output_dir)
    args = argparse.Namespace(
        mode="cv",
        resume_checkpoint=None,
        config=str(tmp_path / "config.yaml"),
    )

    _patch_main_dependencies(
        monkeypatch,
        args=args,
        config=config,
        dataset_factory=lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(train_module, "DEFAULT_MANAGED_ARTIFACT_ROOT", managed_root)

    def invalid_splits(*args: Any, **kwargs: Any) -> tuple[str, list]:
        """模拟 split_mode 拼写错误或分组不足。"""

        del args, kwargs
        raise ValueError("invalid split configuration")

    monkeypatch.setattr(train_module, "_prepare_cross_validation_splits", invalid_splits)
    with pytest.raises(ValueError, match="invalid split configuration"):
        train_module.main()

    assert old_checkpoint.read_bytes() == b"old"


def test_main_fresh_overfit_cleans_old_cv_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """fresh overfit 同样应清理目标根中的旧 CV 文件，并提示 overfit best。"""

    managed_root = tmp_path / "managed"
    output_dir = managed_root / "run"
    output_dir.mkdir(parents=True)
    old_metrics = output_dir / "cv_metrics.json"
    old_metrics.write_text("{}", encoding="utf-8")
    config = _minimal_main_config(output_dir)
    args = argparse.Namespace(
        mode="overfit",
        resume_checkpoint=None,
        config=str(tmp_path / "config.yaml"),
    )

    _patch_main_dependencies(
        monkeypatch,
        args=args,
        config=config,
        dataset_factory=lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(train_module, "DEFAULT_MANAGED_ARTIFACT_ROOT", managed_root)

    def fake_overfit(
        dataset: object,
        received_config: dict[str, Any],
        received_output: Path,
        resume: str | None,
    ) -> None:
        """确认旧 CV 汇总已清除，并创建新的 overfit best 占位文件。"""

        assert dataset is not None
        assert received_config is config
        assert resume is None
        assert not old_metrics.exists()
        checkpoint = received_output / "overfit" / "best.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"new")

    monkeypatch.setattr(train_module, "run_overfit", fake_overfit)
    train_module.main()

    output = capsys.readouterr().out
    assert f"推荐 checkpoint：{output_dir.resolve() / 'overfit' / 'best.pt'}" in output


def test_main_resume_skips_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """overfit resume 应保留目标目录，并把 checkpoint 原样传给训练流程。"""

    output_dir = tmp_path / "resume-output"
    output_dir.mkdir()
    old_checkpoint = output_dir / "overfit" / "last.pt"
    old_checkpoint.parent.mkdir()
    old_checkpoint.write_bytes(b"resume")
    config = _minimal_main_config(output_dir)
    args = argparse.Namespace(
        mode="overfit",
        resume_checkpoint=str(old_checkpoint),
        config=str(tmp_path / "config.yaml"),
    )

    _patch_main_dependencies(
        monkeypatch,
        args=args,
        config=config,
        dataset_factory=lambda *args, **kwargs: object(),
    )

    def fake_overfit(
        dataset: object,
        received_config: dict[str, Any],
        received_output: Path,
        resume: str | None,
    ) -> None:
        """确认 resume checkpoint 在调用训练器前仍然存在。"""

        assert dataset is not None
        assert received_config is config
        assert received_output == output_dir.resolve()
        assert resume == str(old_checkpoint)
        assert old_checkpoint.read_bytes() == b"resume"

    monkeypatch.setattr(train_module, "run_overfit", fake_overfit)

    train_module.main()

    assert old_checkpoint.read_bytes() == b"resume"
    assert "跳过自动清理" in capsys.readouterr().out
