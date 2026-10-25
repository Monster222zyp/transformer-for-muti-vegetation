"""批量比较 RoPE、relative value 的四种组合及两种划分，共八个独立训练工况。

每个工况调用现有 model.train 的 cv 流程。CSV/Markdown 比较表采用合并各折
测试预测后重新计算的 aggregate.MAPE_D、aggregate.R2，不平均各折指标。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml


# 所有 CLI 参数默认值集中在开头；None 表示沿用 base.yaml 的对应设置。
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "model" / "configs" / "base.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT.parent / "ablation_rope_relative_value_20261022"
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_DATA = None
DEFAULT_DEVICE = None
DEFAULT_MAX_EPOCHS = None
DEFAULT_BATCH_SIZE = None
DEFAULT_SEED = None
DEFAULT_N_SPLITS = None
DEFAULT_DRY_RUN = False
DEFAULT_SUMMARIZE_ONLY = False
DEFAULT_FAIL_FAST = False
DEFAULT_QUIET = False
DEFAULT_RERUN_COMPLETED = False

# 固定实验矩阵、持久化文件名与运行参数也集中声明。
SPLIT_MODES = ("sample", "plant_count")
SWITCH_COMBINATIONS = ((True, True), (True, False), (False, True), (False, False))
MANIFEST_NAME = "batch_manifest.json"
CONFIG_NAME = "config.yaml"
STATE_NAME = "run_state.json"
LOG_NAME = "training.log"
ARTIFACTS_NAME = "artifacts"
MANIFEST_KIND = "hydrotransformer_rope_relative_value_ablation_v1"
HASH_BLOCK_SIZE = 1024 * 1024
CHILD_STOP_TIMEOUT_SECONDS = 15
TIMEZONE_HOURS = 8  # 用户时区 Asia/Hong_Kong。
ELAPSED_DECIMALS = 3
CSV_COLUMNS = (
    "case_id", "split_mode", "use_rope", "use_relative_value", "status",
    "MAPE_D", "R2", "MAPE_coverage", "RMSE_D", "best_MAPE_D_in_split",
    "best_R2_in_split", "fold_count", "elapsed_seconds", "exit_code",
    "artifacts_dir", "log_path", "error",
)

# 使用训练入口相同的配置补全逻辑，不另维护一份训练默认参数。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from model.training.config import load_config


def timestamp() -> str:
    """返回香港时区的带偏移时间字符串，用于各工况的开始和结束记录。"""
    return datetime.now(timezone(timedelta(hours=TIMEZONE_HOURS))).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    """读取 path 的 JSON 对象；路径不存在时返回空字典，格式损坏时明确报错。"""
    if not path.exists():
        return {}
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """将 payload 原子写入 path，避免中断时留下半份状态或比较结果。"""
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    """分块计算 path 内容的 SHA-256，检查续跑时数据、配置和训练代码是否改变。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(payload: object) -> str:
    """对配置等 JSON 可表达的 payload 生成稳定摘要，不受字典插入顺序影响。"""
    content = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def case_specs() -> list[dict[str, Any]]:
    """返回固定八工况，顺序先 sample 四组合，再 plant_count 四组合。"""
    return [
        {"case_id": f"{split}__rope_{'on' if rope else 'off'}__relative_value_{'on' if relative else 'off'}",
         "split_mode": split, "use_rope": rope, "use_relative_value": relative}
        for split in SPLIT_MODES for rope, relative in SWITCH_COMBINATIONS
    ]


@contextmanager
def batch_lock(root: Path) -> Iterator[None]:
    """为 root 批次加排他文件锁，避免两个运行器同时清理同一工况；进程退出自动解锁。"""
    with (root / ".batch.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("同一批次已有运行器在执行；请查看已有终端和比较表。") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_output_root(root: Path) -> None:
    """确认 root 是独立的空目录或本脚本已有批次；不接管混有其他文件的目录。"""
    if root == REPO_ROOT or REPO_ROOT.is_relative_to(root):
        raise ValueError("输出必须为独立子文件夹，不能使用仓库或其祖先目录本身。")
    if root.exists():
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"批次根目录必须为普通目录：{root}")
        entries = [path for path in root.iterdir() if path.name != ".batch.lock"]
        if entries and read_json(root / MANIFEST_NAME).get("kind") != MANIFEST_KIND:
            raise ValueError(f"输出目录非空且不是本脚本的批次，请另选 --output-dir：{root}")
    root.mkdir(parents=True, exist_ok=True)


def build_plan(arguments: argparse.Namespace) -> dict[str, Any]:
    """读取基准配置并应用全工况共有的 CLI 参数，返回完整可复现批次计划。"""
    config = load_config(arguments.config)
    config["mode"] = "cv"
    for key in ("dataset_path", "physics_config_path"):
        path = Path(config["data"][key]).expanduser()
        config["data"][key] = str((path if path.is_absolute() else REPO_ROOT / path).resolve())
    if arguments.data is not None:
        config["data"]["dataset_path"] = str(arguments.data.resolve())
    for argument, key in (("device", "device"), ("max_epochs", "max_epochs"), ("batch_size", "batch_size")):
        value = getattr(arguments, argument)
        if value is not None:
            config["training"][key] = value
    if arguments.seed is not None:
        config["seed"] = arguments.seed
    if arguments.n_splits is not None:
        config["cross_validation"]["n_splits"] = arguments.n_splits
    config["output"]["artifact_dir"] = "<每个工况独立指定>"

    # 只做轻量样本统计，避免准备计划时就加载物理求解器或启动训练。
    dataset = Path(config["data"]["dataset_path"])
    plant_counts = []
    sample_ids = set()
    with dataset.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError("JSONL 中存在空行。")
            sample = json.loads(line)
            if sample["sample_id"] in sample_ids or not sample["plants"]:
                raise ValueError("数据集存在重复 sample_id 或空植株列表。")
            sample_ids.add(sample["sample_id"])
            plant_counts.append(len(sample["plants"]))
    folds = config["cross_validation"]["n_splits"]
    if isinstance(folds, bool) or not isinstance(folds, int) or not 2 <= folds <= len(set(plant_counts)):
        raise ValueError(f"n_splits 需为 2～不同植株根数组数；当前根数组数为 {len(set(plant_counts))}。")
    if not 0 < config["cross_validation"]["validation_fraction"] < 1:
        raise ValueError("validation_fraction 必须在 0 和 1 之间。")
    for key in ("max_epochs", "batch_size"):
        if isinstance(config["training"][key], bool) or not isinstance(config["training"][key], int) or config["training"][key] <= 0:
            raise ValueError(f"{key} 必须为正整数。")
    if not arguments.python.is_file():
        raise FileNotFoundError(f"Python 解释器不存在：{arguments.python}")

    # 把训练代码也纳入续跑身份，防止同一比较表混入不同代码或不同数据生成的结果。
    source_paths = [arguments.config.resolve(), dataset, Path(config["data"]["physics_config_path"]), REPO_ROOT / "model" / "train.py"]
    for directory in ("models", "training", "hydro"):
        source_paths.extend((REPO_ROOT / "model" / directory).rglob("*.py"))
    inputs = {str(path.resolve()): sha256(path) for path in sorted(set(source_paths))}
    identity = {"base_config": config, "input_sha256": inputs, "cases": case_specs(), "python": str(arguments.python.resolve())}
    return {"kind": MANIFEST_KIND, "created_at": timestamp(), "fingerprint": fingerprint(identity),
            **identity, "dataset_sample_count": len(plant_counts), "plant_counts": sorted(set(plant_counts)),
            "python": str(arguments.python.resolve()), "repository": str(REPO_ROOT)}


def case_config(plan: dict[str, Any], spec: dict[str, Any], root: Path) -> dict[str, Any]:
    """复制 plan 基准配置，只替换 spec 两开关、划分方式及独立 artifact 路径。"""
    config = copy.deepcopy(plan["base_config"])
    config["model"]["use_rope"] = spec["use_rope"]
    config["model"]["use_relative_value"] = spec["use_relative_value"]
    config["cross_validation"]["split_mode"] = spec["split_mode"]
    config["output"]["artifact_dir"] = str(root / spec["case_id"] / ARTIFACTS_NAME)
    return config


def prepare_batch(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """在 root 保存计划和八份配置；已有批次身份不一致时拒绝复用，不覆盖旧结果。"""
    old = read_json(root / MANIFEST_NAME)
    if old:
        if old.get("fingerprint") != plan["fingerprint"]:
            raise ValueError("已有批次的数据、配置或训练代码已改变；请使用新的 --output-dir。")
        plan = old
    else:
        write_json(root / MANIFEST_NAME, plan)
    for spec in plan["cases"]:
        directory = root / spec["case_id"]
        directory.mkdir(exist_ok=True)
        config = case_config(plan, spec, root)
        path = directory / CONFIG_NAME
        if path.exists():
            if yaml.safe_load(path.read_text(encoding="utf-8")) != config:
                raise ValueError(f"工况配置被改动，拒绝混入当前批次：{path}")
        else:
            # 配置与日志放 artifacts 外面；训练入口可以安全清理自己负责的 artifacts。
            path.write_text("# 自动生成的独立消融工况；其余参数与批次基准完全相同。\n" +
                            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return plan


def completed_metrics(root: Path, plan: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """验证成功状态、配置身份、全部折指标和全量重训产物；不把部分结果视为完成。"""
    directory = root / spec["case_id"]
    state = read_json(directory / STATE_NAME)
    expected = case_config(plan, spec, root)
    if state.get("status") != "completed" or state.get("exit_code") != 0 or state.get("config_fingerprint") != fingerprint(expected):
        return {}
    artifacts = directory / ARTIFACTS_NAME
    try:
        metrics = read_json(artifacts / "cv_metrics.json")
        resolved = read_json(artifacts / "resolved_config.json")
    except (OSError, ValueError):
        return {}
    checkpoint = artifacts / "final_model.pt"
    if not metrics.get("final_retraining_performed") or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return {}
    if metrics.get("split_mode") != spec["split_mode"] or len(metrics.get("folds", [])) != expected["cross_validation"]["n_splits"]:
        return {}
    if resolved.get("model") != expected["model"] or resolved.get("seed") != expected["seed"]:
        return {}
    if resolved.get("cross_validation") != expected["cross_validation"] or resolved.get("data") != expected["data"]:
        return {}
    if resolved.get("training") != expected["training"] or resolved.get("output") != expected["output"] or resolved.get("mode") != "cv":
        return {}
    aggregate = metrics.get("aggregate", {})
    if not isinstance(aggregate, dict) or not all(key in aggregate for key in ("MAPE_D", "R2", "MAPE_coverage", "RMSE_D")):
        return {}
    return metrics


def finite_metric(value: object) -> float | None:
    """将合法指标转为 float；NaN、Inf、缺失值显示为空，保留合法的负 R²。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def write_comparison(root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    """将八工况当前状态与合并测试指标写入 root 的 CSV 和 Markdown，返回八行数据。"""
    rows = []
    for spec in plan["cases"]:
        directory = root / spec["case_id"]
        state = read_json(directory / STATE_NAME)
        metrics = completed_metrics(root, plan, spec)
        status = state.get("status", "prepared")
        if status == "completed" and not metrics:
            status = "incomplete"
        aggregate = metrics.get("aggregate", {})
        rows.append({**spec, "status": status,
                     **{key: finite_metric(aggregate.get(key)) for key in ("MAPE_D", "R2", "MAPE_coverage", "RMSE_D")},
                     "best_MAPE_D_in_split": False, "best_R2_in_split": False,
                     "fold_count": len(metrics["folds"]) if metrics else None,
                     "elapsed_seconds": state.get("elapsed_seconds"), "exit_code": state.get("exit_code"),
                     "artifacts_dir": str(directory / ARTIFACTS_NAME), "log_path": str(directory / LOG_NAME),
                     "error": state.get("error", "")})

    # 仅在同种划分内部标注最优值；两种划分衡量的泛化任务不同，不合并排名。
    for split in SPLIT_MODES:
        for metric, choose in (("MAPE_D", min), ("R2", max)):
            available = [row for row in rows if row["split_mode"] == split and row["status"] == "completed" and row[metric] is not None]
            if available:
                best = choose(row[metric] for row in available)
                for row in available:
                    row[f"best_{metric}_in_split"] = row[metric] == best
    csv_path = root / "comparison.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)

    # MAPE_D 已是百分数，不再次乘 100；缺失或无定义指标显示为“—”。
    lines = ["# 八工况消融实验比较", "", "MAPE_D 单位为 %，越小越好；R² 越大越好。指标取 cv_metrics.json 的 aggregate。",
             "sample 随机划分单条样本；plant_count 将同一植株根数的全部样本作为一组，组间不混入训练与测试。",
             "空白指标表示尚未完整训练成功，或该指标无数学定义；每个工况独立完成交叉验证与全量重训。", "",
             f"数据：{plan['base_config']['data']['dataset_path']}；样本 {plan['dataset_sample_count']}；"
             f"seed={plan['base_config']['seed']}；折数={plan['base_config']['cross_validation']['n_splits']}。", "",
             "| 划分 | RoPE | relative value | 状态 | MAPE_D (%) ↓ | R² ↑ | 产物 |",
             "|---|---|---|---|---:|---:|---|"]
    for row in rows:
        mape = "—" if row["MAPE_D"] is None else f"{row['MAPE_D']:.6f}"
        r2 = "—" if row["R2"] is None else f"{row['R2']:.6f}"
        if row["best_MAPE_D_in_split"]:
            mape = f"**{mape}**"
        if row["best_R2_in_split"]:
            r2 = f"**{r2}**"
        lines.append(f"| {row['split_mode']} | {row['use_rope']} | {row['use_relative_value']} | {row['status']} | "
                     f"{mape} | {r2} | [artifacts]({row['case_id']}/{ARTIFACTS_NAME}/) |")
    markdown = root / "comparison.md"
    temporary = markdown.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(markdown)
    return rows


def training_command(python: Path, root: Path, spec: dict[str, Any]) -> list[str]:
    """返回该工况的参数列表；python 指解释器，root 指批次目录，路径含空格也无需 shell 引号。"""
    directory = root / spec["case_id"]
    return [str(python.resolve()), "-u", "-m", "model.train", "--mode", "cv", "--config", str(directory / CONFIG_NAME),
            "--artifact-dir", str(directory / ARTIFACTS_NAME)]


def execute_training(command: list[str], log_path: Path, quiet: bool) -> int:
    """串行启动 command，追加 UTF-8 日志到 log_path；quiet 仅关闭终端逐行回显。"""
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    with log_path.open("a", encoding="utf-8", newline="") as log:
        log.write(f"\n===== {timestamp()} 新的一次完整训练尝试 =====\n")
        log.write(subprocess.list2cmdline(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=REPO_ROOT, shell=False, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if not quiet:
                    print(line, end="", flush=True)
            return process.wait()
        finally:
            # Ctrl+C 时停止当前子进程，保留已完成的其他工况，避免后台继续占用 GPU。
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=CHILD_STOP_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout is not None:
                process.stdout.close()


def run_cases(root: Path, plan: dict[str, Any], arguments: argparse.Namespace) -> int:
    """按计划逐个训练；默认跳过已完成工况，失败继续，返回 0 或失败退出码 1。"""
    failed = False
    for number, spec in enumerate(plan["cases"], start=1):
        directory = root / spec["case_id"]
        if not arguments.rerun_completed and completed_metrics(root, plan, spec):
            print(f"[{number}/8] 跳过已完成：{spec['case_id']}", flush=True)
            continue
        # 每次训练前复核输入身份。用户中途编辑训练代码或数据时，不混合不同版本。
        for path, expected_hash in plan["input_sha256"].items():
            if sha256(Path(path)) != expected_hash:
                raise ValueError(f"批次执行期间输入发生改变，已停止：{path}")
        previous = read_json(directory / STATE_NAME)
        state = {"case_id": spec["case_id"], "status": "running", "started_at": timestamp(),
                 "attempt": previous.get("attempt", 0) + 1,
                 "config_fingerprint": fingerprint(case_config(plan, spec, root))}
        write_json(directory / STATE_NAME, state)
        write_comparison(root, plan)
        started = time.monotonic()
        print(f"[{number}/8] 开始：{spec['case_id']}；日志：{directory / LOG_NAME}", flush=True)
        try:
            code = execute_training(training_command(arguments.python, root, spec), directory / LOG_NAME, arguments.quiet)
            state.update(status="completed" if code == 0 else "failed", exit_code=code)
            write_json(directory / STATE_NAME, state)
            if code != 0:
                raise RuntimeError(f"训练进程返回 {code}，请查看 training.log。")
            if not completed_metrics(root, plan, spec):
                raise RuntimeError("进程返回 0，但缺少匹配的完整 CV 指标、配置快照或全量重训模型。")
        except KeyboardInterrupt:
            state.update(status="interrupted", exit_code=130, error="用户中断当前训练；已完成工况保留。")
            raise
        except (OSError, ValueError, RuntimeError) as error:
            failed = True
            state.update(status="failed", error=str(error))
        finally:
            state.update(finished_at=timestamp(), elapsed_seconds=round(time.monotonic() - started, ELAPSED_DECIMALS))
            write_json(directory / STATE_NAME, state)
            write_comparison(root, plan)
        print(f"[{number}/8] {state['status']}：{spec['case_id']}", flush=True)
        if state["status"] == "failed" and arguments.fail_fast:
            break
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    """创建 CLI；默认准备/运行全部八组合，参数可统一覆盖基准训练设置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="所有工况共有的基准 YAML")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="八工况与比较表的独立批次目录")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="启动训练所用 Python；默认当前解释器")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="统一覆盖八工况的数据集；省略沿用 YAML")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="统一覆盖 device，例如 cuda/cpu/auto")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="统一覆盖最大 epoch 数")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="统一覆盖 batch size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="八工况共用的随机种子")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS, help="统一覆盖交叉验证折数")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=DEFAULT_DRY_RUN, help="只生成八份配置和待运行比较表，不训练")
    mode.add_argument("--summarize-only", action="store_true", default=DEFAULT_SUMMARIZE_ONLY, help="只从已有批次重建比较表，不训练")
    parser.add_argument("--fail-fast", action="store_true", default=DEFAULT_FAIL_FAST, help="一个工况失败后立即停止；默认继续其他工况")
    parser.add_argument("--quiet", action="store_true", default=DEFAULT_QUIET, help="只显示批次进度，完整训练日志仍写文件")
    parser.add_argument("--rerun-completed", action="store_true", default=DEFAULT_RERUN_COMPLETED,
                        help="完整重训已完成工况；训练入口会清理对应 artifacts，请谨慎使用")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口；argv 为参数列表，None 表示读取命令行。不会自动启动额外后台任务。"""
    arguments = build_parser().parse_args(argv)
    root = arguments.output_dir.expanduser().resolve()
    try:
        if arguments.summarize_only and not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError("批次不存在，不能仅汇总；请先生成计划或开始运行。")
        validate_output_root(root)
        with batch_lock(root):
            if arguments.summarize_only:
                plan = read_json(root / MANIFEST_NAME)
                if plan.get("cases") != case_specs():
                    raise ValueError("批次工况定义与当前八工况矩阵不一致。")
                write_comparison(root, plan)
                print(f"已更新比较表：{root / 'comparison.csv'}")
                return 0
            plan = prepare_batch(root, build_plan(arguments))
            write_comparison(root, plan)
            print(f"批次目录：{root}\n样本：{plan['dataset_sample_count']}；八工况；每工况 {plan['base_config']['cross_validation']['n_splits']} 折及一次全量重训。", flush=True)
            if arguments.dry_run:
                for spec in plan["cases"]:
                    print(subprocess.list2cmdline(training_command(arguments.python, root, spec)))
                print("仅生成计划和配置，未启动训练；comparison.csv 的指标待训练后自动填写。")
                return 0
            code = run_cases(root, plan, arguments)
            print(f"批次结束；比较表：{root / 'comparison.csv'}", flush=True)
            return code
    except KeyboardInterrupt:
        print("已停止当前工况；再次使用相同命令会跳过已完成工况，并重新完整训练未完成工况。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"批量实验失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
