"""Sync uncommitted changes from the main working tree to the experiment worktree.

The sync copies tracked modifications and untracked files from the repository's
main working tree into the experiment worktree, excluding the ``.gymrat/``
session directory. When the experiment worktree already has uncommitted changes
that would be overwritten, the sync refuses — no partial application.
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Iterator

from gymrat.errors import GymratError, stderr_text_of
from gymrat.git import run_git
from gymrat.session.paths import SESSION_DIR_NAME, experiment_worktree_dir
from gymrat.session.store import require_open_session

# ``git status -z`` prefixes each entry with two status characters and a space
# (``XY<space>``), then the NUL-delimited path. Rename/copy entries (``R`` or
# ``C`` in X) carry a second NUL-delimited field: ``XY<space>new\0old\0``.
_STATUS_PREFIX_LEN = 3

# Status codes that carry a second path field (the original name).
_RENAME_COPY_CODES = frozenset("RC")


def _raise_file_vs_dir_error(name: str) -> NoReturn:
    """Shared by both sync failure sites so the message and submodule hint stay identical."""
    msg = f"Cannot sync '{name}': expected a file but found a directory"
    raise GymratError(msg, hint="If this is a submodule, commit or remove it before syncing.")


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Outcome of syncing changes from the main tree to the experiment worktree."""

    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DirtyEntry:
    """A single dirty-file entry parsed from ``git status -z``."""

    path: str
    old_path: str | None


def _dirty_entries(directory: str) -> list[_DirtyEntry]:
    """Parse ``git status -z`` into structured entries.

    NUL-delimited output avoids the C-quoting that ``--porcelain`` applies to
    non-ASCII and whitespace-containing paths, so every path is the literal
    filesystem name.
    """
    raw = run_git(["status", "-z", "--untracked-files=all"], directory)
    entries: list[_DirtyEntry] = []
    fields: Iterator[str] = iter(raw.split("\0"))
    for field in fields:
        if len(field) < _STATUS_PREFIX_LEN:
            continue
        status_x = field[0]
        path = field[_STATUS_PREFIX_LEN:]
        old_path: str | None = None
        if status_x in _RENAME_COPY_CODES:
            old_path = next(fields, None)
        entries.append(_DirtyEntry(path=path, old_path=old_path))
    return entries


def _exclude_session_dir(entries: list[_DirtyEntry]) -> list[_DirtyEntry]:
    """Drop any entry rooted under the session directory."""
    prefix = f"{SESSION_DIR_NAME}/"
    return [e for e in entries if not e.path.startswith(prefix) and e.path != SESSION_DIR_NAME]


def _read_dirty_entries(directory: str, error_message: str, hint: str) -> list[_DirtyEntry]:
    """``_dirty_entries(directory)``, wrapping a read failure as a ``GymratError``."""
    try:
        return _dirty_entries(directory)
    except (subprocess.CalledProcessError, OSError) as exc:
        msg = f"{error_message}: {stderr_text_of(exc)}"
        raise GymratError(msg, hint=hint) from exc


def _copy_entry(src: Path, dst: Path) -> None:
    """Copy a single file preserving symlinks and file-mode bits."""
    if src.is_symlink():
        link_target = src.readlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.unlink(missing_ok=True)
        dst.symlink_to(link_target)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        _raise_file_vs_dir_error(src.name)


def sync_to_experiment(root: str) -> SyncResult:
    """Apply uncommitted changes from the main tree onto the experiment worktree.

    Raises:
        GymratError: When no session is open, when the experiment worktree has
            uncommitted changes that overlap with the files to sync, when the
            experiment worktree is missing, or when git itself fails.
    """
    require_open_session(root, "syncing changes")

    experiment = experiment_worktree_dir(root)

    main_entries = _exclude_session_dir(
        _read_dirty_entries(
            root, "Cannot read dirty files", "Check that the repository is not corrupt."
        )
    )
    if not main_entries:
        return SyncResult(files=())

    main_paths = {e.path for e in main_entries}

    experiment_entries = _read_dirty_entries(
        experiment,
        "Cannot read experiment worktree",
        "The experiment worktree may have been deleted. Run 'gymrat start' to begin a new session.",
    )
    experiment_paths = {e.path for e in experiment_entries}
    conflicts = main_paths & experiment_paths
    if conflicts:
        listed = ", ".join(sorted(conflicts))
        message = f"Cannot sync — the experiment worktree has uncommitted changes in: {listed}"
        raise GymratError(message, hint="Settle or revert the experiment worktree first.")

    for entry in main_entries:
        src = Path(root) / entry.path
        dst = Path(experiment) / entry.path
        try:
            if not src.exists() and not src.is_symlink():
                dst.unlink(missing_ok=True)
                continue
            _copy_entry(src, dst)
        except IsADirectoryError:
            _raise_file_vs_dir_error(entry.path)

        if entry.old_path is not None:
            old_dst = Path(experiment) / entry.old_path
            old_dst.unlink(missing_ok=True)

    return SyncResult(files=tuple(sorted(main_paths)))
