"""训练产物目录的安全初始化与旧结果清理工具。

本模块只服务于新的训练运行，不参与独立评估。清理函数始终针对已经解析为绝对
路径的单次 ``artifact_dir``，并通过受管根目录、ownership marker 和保护路径检查
降低误删源码、输入数据或用户目录的风险。
"""

from __future__ import annotations

import shutil
import stat
from collections.abc import Iterable
from pathlib import Path


ARTIFACT_OWNERSHIP_MARKER = ".hydrotransformer-artifacts"
"""证明自定义非空目录由 HydroTransformer 管理的标记文件名。"""

ARTIFACT_OWNERSHIP_MARKER_CONTENT = (
    "This directory is managed by HydroTransformer training.\n"
)
"""ownership marker 的固定内容；空的同名文件不被视为有效授权。"""

WINDOWS_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
"""Windows symbolic link 与 junction 在 ``st_file_attributes`` 中使用的标志位。"""


def _resolved(path: str | Path) -> Path:
    """把路径展开并解析为规范化绝对路径。

    Args:
        path: 字符串或 :class:`pathlib.Path` 路径。

    Returns:
        不要求目标已经存在的规范化绝对路径。
    """

    return Path(path).expanduser().resolve(strict=False)


def _is_same_or_ancestor(candidate: Path, protected_path: Path) -> bool:
    """判断候选目录是否等于或包含受保护路径。

    Args:
        candidate: 可能被递归清理的目录。
        protected_path: 不允许被删除的目录或文件。

    Returns:
        候选目录等于受保护路径，或是其祖先目录时返回 ``True``。
    """

    return candidate == protected_path or protected_path.is_relative_to(candidate)


def _is_reparse_point(path: Path) -> bool:
    """检查路径自身是否为 symbolic link 或 Windows junction。

    Args:
        path: 要检查的现有路径。

    Returns:
        路径是 symbolic link，或带 Windows reparse-point 属性时返回 ``True``。

    Notes:
        使用 ``lstat`` 读取链接本身，避免为了判断类型而跟随到链接目标。
    """

    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except FileNotFoundError:
        return False
    return bool(attributes & WINDOWS_REPARSE_POINT_ATTRIBUTE)


def _has_valid_ownership_marker(target: Path) -> bool:
    """验证目标目录是否包含内容正确的 ownership marker。

    Args:
        target: 已存在的候选产物目录。

    Returns:
        marker 是普通文件且内容完全匹配时返回 ``True``。
    """

    marker = target / ARTIFACT_OWNERSHIP_MARKER
    if not marker.is_file() or _is_reparse_point(marker):
        return False
    try:
        return marker.read_text(encoding="utf-8") == ARTIFACT_OWNERSHIP_MARKER_CONTENT
    except OSError:
        return False


def _remove_artifact_entry(path: Path) -> None:
    """删除产物目录中的一个直接子项且不跟随链接。

    Args:
        path: 目标产物目录中的直接子文件或子目录。

    Raises:
        OSError: 文件被占用、权限不足或其他文件系统操作失败时原样抛出。
    """

    # POSIX 和 Windows 的普通 symbolic link 一律使用 unlink；``Path.is_dir`` 会
    # 跟随目录链接，若直接据此调用 rmdir，既可能失败也会模糊“不跟随链接”的意图。
    if path.is_symlink():
        path.unlink()
        return
    if _is_reparse_point(path):
        # Windows junction 通常不是 ``is_symlink``，却带 reparse-point 属性。目录
        # junction 用 rmdir 只移除入口本身，不会递归进入其目标。
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def prepare_fresh_artifact_directory(
    target: str | Path,
    *,
    managed_root: str | Path,
    protected_paths: Iterable[str | Path],
) -> Path:
    """校验并清空一次 fresh training 使用的产物目录。

    Args:
        target: 本次运行最终解析出的 ``artifact_dir``。
        managed_root: 项目默认管理的产物根目录；它及其子目录无需旧 marker。
        protected_paths: 配置、数据、源码根目录等绝不能由 ``target`` 包含的路径。

    Returns:
        已创建、已清理并带 ownership marker 的规范化绝对目录。

    Raises:
        ValueError: 目标过宽、包含受保护内容、不是目录、属于链接，或外部非空目录
            没有合法 marker 时抛出。
        RuntimeError: 清理旧文件或写 marker 失败时抛出；训练入口必须停止运行。
    """

    raw_target = Path(target).expanduser()
    # 必须在 resolve 之前检查目标自身，否则 resolve 会跟随链接并隐藏“清理根是链接”
    # 这一危险事实。
    if raw_target.exists() and _is_reparse_point(raw_target):
        raise ValueError(f"artifact_dir 不能是 symbolic link 或 junction：{raw_target}")

    resolved_target = _resolved(raw_target)
    resolved_managed_root = _resolved(managed_root)
    filesystem_root = Path(resolved_target.anchor)
    if resolved_target == filesystem_root:
        raise ValueError(f"拒绝把文件系统根目录作为 artifact_dir：{resolved_target}")

    for protected_path in protected_paths:
        resolved_protected = _resolved(protected_path)
        if _is_same_or_ancestor(resolved_target, resolved_protected):
            raise ValueError(
                "artifact_dir 不能等于或包含受保护路径："
                f"target={resolved_target}, protected={resolved_protected}"
            )

    if resolved_target.exists() and not resolved_target.is_dir():
        raise ValueError(f"artifact_dir 已存在但不是目录：{resolved_target}")

    marker = resolved_target / ARTIFACT_OWNERSHIP_MARKER
    # 受管默认目录也不能信任一个同名链接或目录作为 marker；后续 write_text 可能
    # 跟随恶意链接覆盖目录外文件，所以必须在删除任何旧结果前拒绝。
    if (marker.exists() or marker.is_symlink()) and (
        not marker.is_file() or _is_reparse_point(marker)
    ):
        raise ValueError(f"ownership marker 必须是普通文件，不能是目录或链接：{marker}")

    is_managed_target = (
        resolved_target == resolved_managed_root
        or resolved_target.is_relative_to(resolved_managed_root)
    )
    existing_entries = (
        list(resolved_target.iterdir()) if resolved_target.exists() else []
    )
    if existing_entries and not is_managed_target and not _has_valid_ownership_marker(
        resolved_target
    ):
        raise ValueError(
            "拒绝清理项目默认 artifacts 之外的非空目录；该目录缺少有效的 "
            f"{ARTIFACT_OWNERSHIP_MARKER}：{resolved_target}"
        )

    resolved_target.mkdir(parents=True, exist_ok=True)
    try:
        # marker 最后才会被后续运行识别为清理授权，因此先保留旧 marker，并只删除
        # 其他子项。若中途失败，重试仍能识别该目录，而不会因 marker 先被删除而锁死。
        for child in resolved_target.iterdir():
            if child.name == ARTIFACT_OWNERSHIP_MARKER:
                continue
            _remove_artifact_entry(child)
        marker.write_text(ARTIFACT_OWNERSHIP_MARKER_CONTENT, encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"无法完整清理训练产物目录 {resolved_target}；训练尚未开始：{error}"
        ) from error
    return resolved_target


def prepare_resume_artifact_directory(target: str | Path) -> Path:
    """为 resume run 创建目录，但绝不删除其中已有内容。

    Args:
        target: 本次恢复训练要写入的 ``artifact_dir``。

    Returns:
        已确保存在的规范化绝对目录。

    Raises:
        ValueError: 目标已存在但不是目录时抛出。
        OSError: 目录无法创建时由文件系统原样抛出。
    """

    resolved_target = _resolved(target)
    if resolved_target.exists() and not resolved_target.is_dir():
        raise ValueError(f"artifact_dir 已存在但不是目录：{resolved_target}")
    resolved_target.mkdir(parents=True, exist_ok=True)
    return resolved_target
