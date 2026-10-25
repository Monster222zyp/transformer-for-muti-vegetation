"""批量消融运行器测试；仅创建临时文件和模拟子进程，不启动任何模型训练。"""

from __future__ import annotations

import copy
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from model import run_ablation as runner


# 人造数据和指标集中声明；这些是运行器 fixture，不是正式实验结果。
TEST_SEED = 42
TEST_FOLDS = 2
TEST_MAX_EPOCHS = 1
TEST_MISSING_MAPE = float("nan")
TEST_R2 = -0.25  # 合法负 R² 必须保留，不能误当作缺失值。
TEST_CASE_COUNT = 8
TEST_FAILURE_EXIT_CODE = 7


def _arguments(tmp_path: Path) -> list[str]:
    """创建轻量配置与假数据，返回该临时批次的 CLI 参数，不加载 HydroDataset。"""
    data = tmp_path / "dataset.jsonl"
    data.write_text("".join(json.dumps({"sample_id": f"sample_{index}", "plants": [{}] * count}) + "\n"
                            for index, count in enumerate((1, 1, 2, 2, 3, 3))), encoding="utf-8")
    config = tmp_path / "base.yaml"
    config.write_text(yaml.safe_dump({
        "data": {"dataset_path": str(data)}, "seed": TEST_SEED,
        "training": {"max_epochs": TEST_MAX_EPOCHS},
        "cross_validation": {"n_splits": TEST_FOLDS},
    }), encoding="utf-8")
    return ["--config", str(config), "--output-dir", str(tmp_path / "batch with spaces"), "--python", sys.executable, "--quiet"]


def _read_comparison(root: Path) -> list[dict[str, str]]:
    """以 Excel 可读的 UTF-8 BOM 编码回读 root 比较表。"""
    with (root / "comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _simulate_training(command: list[str], log_path: Path, quiet: bool) -> int:
    """用 command 指向的实际 YAML 构造完整 fixture 产物，模拟训练成功返回 0。"""
    config_path = Path(command[command.index("--config") + 1])
    artifacts = Path(command[command.index("--artifact-dir") + 1])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifacts.mkdir(parents=True, exist_ok=True)
    relative = config["model"]["use_relative_value"]
    score = 5.0 + int(not config["model"]["use_rope"]) + int(not relative)
    metrics = {
        "split_mode": config["cross_validation"]["split_mode"], "final_retraining_performed": True,
        # 折指标故意与 aggregate 明显不同，验证比较表不去平均或取第一折。
        "aggregate": {"MAPE_D": score if relative else TEST_MISSING_MAPE, "R2": TEST_R2,
                      "RMSE_D": 0.2, "MAPE_coverage": 1.0},
        "folds": [{"fold": fold, "MAPE_D": 99.0, "R2": 0.99} for fold in range(TEST_FOLDS)],
    }
    (artifacts / "cv_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    runner.write_json(artifacts / "resolved_config.json", config)
    (artifacts / "final_model.pt").write_bytes(b"fixture only; not a real checkpoint")
    log_path.write_text("仅模拟训练器产物\n", encoding="utf-8")
    return 0


def test_all_eight_configs_and_metric_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """覆盖八组合、其余模型参数一致、aggregate 指标提取、NaN 空白和负 R²。"""
    arguments = _arguments(tmp_path)
    root = Path(arguments[arguments.index("--output-dir") + 1])
    assert runner.main([*arguments, "--dry-run"]) == 0
    plan = runner.read_json(root / runner.MANIFEST_NAME)
    assert len(plan["cases"]) == TEST_CASE_COUNT
    observed = set()
    reference = None
    for spec in plan["cases"]:
        config = yaml.safe_load((root / spec["case_id"] / runner.CONFIG_NAME).read_text(encoding="utf-8"))
        observed.add((config["cross_validation"]["split_mode"], config["model"]["use_rope"], config["model"]["use_relative_value"]))
        # 去掉仅允许改变的三个变量和产物路径，剩余配置必须逐项相同。
        normalized = copy.deepcopy(config)
        for key in ("use_rope", "use_relative_value"):
            normalized["model"].pop(key)
        normalized["cross_validation"].pop("split_mode")
        normalized["output"].pop("artifact_dir")
        reference = normalized if reference is None else reference
        assert normalized == reference
        assert not (root / spec["case_id"] / runner.ARTIFACTS_NAME).exists()
    assert observed == {(split, rope, relative) for split in runner.SPLIT_MODES for rope in (True, False) for relative in (True, False)}
    assert all(row["status"] == "prepared" and row["MAPE_D"] == "" for row in _read_comparison(root))
    monkeypatch.setattr(runner, "execute_training", _simulate_training)
    assert runner.main(arguments) == 0
    rows = _read_comparison(root)
    assert len(rows) == TEST_CASE_COUNT and all(row["status"] == "completed" for row in rows)
    assert all(float(row["R2"]) == TEST_R2 for row in rows)
    assert sum(row["MAPE_D"] == "" for row in rows) == 4
    assert {float(row["MAPE_D"]) for row in rows if row["MAPE_D"]} == {5.0, 6.0}
    assert sum(row["best_MAPE_D_in_split"] == "True" for row in rows) == 2
    # 再次执行只能跳过八个已完整完成的工况，不重复清理或训练。
    def unexpected(*args: object) -> int:
        pytest.fail("已完成工况不应再次启动训练")
    monkeypatch.setattr(runner, "execute_training", unexpected)
    assert runner.main(arguments) == 0
    assert runner.main([*arguments, "--summarize-only"]) == 0


def test_failure_continues_and_only_failed_case_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """一次非零退出不能阻断其他七工况；续跑只重试失败工况。"""
    arguments = _arguments(tmp_path)
    calls = []
    def first_attempt(command: list[str], log: Path, quiet: bool) -> int:
        calls.append(command)
        return TEST_FAILURE_EXIT_CODE if len(calls) == 1 else _simulate_training(command, log, quiet)
    monkeypatch.setattr(runner, "execute_training", first_attempt)
    assert runner.main(arguments) == 1
    root = Path(arguments[arguments.index("--output-dir") + 1])
    assert len(calls) == TEST_CASE_COUNT
    assert [row["status"] for row in _read_comparison(root)].count("completed") == 7
    calls.clear()
    def retry(command: list[str], log: Path, quiet: bool) -> int:
        calls.append(command)
        return _simulate_training(command, log, quiet)
    monkeypatch.setattr(runner, "execute_training", retry)
    assert runner.main(arguments) == 0
    assert len(calls) == 1


def test_partial_cv_is_not_complete_and_changed_data_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """进程零退出但全量重训未完成时须失败；已有批次不能混入被改动的数据。"""
    arguments = _arguments(tmp_path)
    def partial(command: list[str], log: Path, quiet: bool) -> int:
        _simulate_training(command, log, quiet)
        artifacts = Path(command[command.index("--artifact-dir") + 1])
        path = artifacts / "cv_metrics.json"
        payload = runner.read_json(path)
        payload["final_retraining_performed"] = False
        runner.write_json(path, payload)
        return 0
    monkeypatch.setattr(runner, "execute_training", partial)
    assert runner.main([*arguments, "--fail-fast"]) == 1
    root = Path(arguments[arguments.index("--output-dir") + 1])
    rows = _read_comparison(root)
    assert rows[0]["status"] == "failed" and rows[0]["MAPE_D"] == ""
    assert all(row["status"] == "prepared" for row in rows[1:])
    data = tmp_path / "dataset.jsonl"
    data.write_text(data.read_text(encoding="utf-8") + json.dumps({"sample_id": "new_sample", "plants": [{}]}) + "\n", encoding="utf-8")
    assert runner.main([*arguments, "--dry-run"]) == 1
    # 仅汇总历史结果不要求当前数据仍保持不变，也不启动子进程。
    assert runner.main([*arguments, "--summarize-only"]) == 0


def test_subprocess_uses_argument_list_and_utf8_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """路径含空格时不经过 shell；中文输出同时被正确写入日志。"""
    captured = {}
    class FakeProcess:
        """最小 Popen 替身，提供输出流和进程生命周期方法。"""
        def __init__(self, command: list[str], **kwargs: object):
            captured.update(command=command, **kwargs)
            self.stdout = io.StringIO("中文训练日志\n")
        def wait(self, timeout: object = None) -> int:
            return 0
        def poll(self) -> int:
            return 0
    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    command = [sys.executable, "-u", "-m", "model.train", "--config", str(tmp_path / "path with spaces" / "config.yaml")]
    log = tmp_path / "training.log"
    assert runner.execute_training(command, log, quiet=True) == 0
    assert captured["command"] == command and captured["shell"] is False
    assert captured["cwd"] == runner.REPO_ROOT
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
    assert "中文训练日志" in log.read_text(encoding="utf-8")


def test_interrupt_preserves_finished_cases_and_records_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """用户中断时记录 interrupted 并退出，不继续启动下一个工况。"""
    arguments = _arguments(tmp_path)
    def interrupted(command: list[str], log: Path, quiet: bool) -> int:
        raise KeyboardInterrupt
    monkeypatch.setattr(runner, "execute_training", interrupted)
    assert runner.main(arguments) == 130
    root = Path(arguments[arguments.index("--output-dir") + 1])
    rows = _read_comparison(root)
    assert rows[0]["status"] == "interrupted" and rows[0]["exit_code"] == "130"
    assert all(row["status"] == "prepared" for row in rows[1:])


def test_unowned_output_directory_is_not_modified(tmp_path: Path) -> None:
    """非空且不属于当前运行器的目录必须拒绝，不能清理用户已有文件。"""
    arguments = _arguments(tmp_path)
    root = Path(arguments[arguments.index("--output-dir") + 1])
    root.mkdir()
    important = root / "existing_model.pt"
    important.write_bytes(b"user original content")
    assert runner.main([*arguments, "--dry-run"]) == 1
    assert important.read_bytes() == b"user original content"
    assert list(root.iterdir()) == [important]
