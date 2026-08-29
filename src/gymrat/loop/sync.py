"""Sync uncommitted changes from the main working tree to the experiment worktree.

The sync copies tracked modifications and untracked files from the repository's
main working tree into the experiment worktree, excluding the ``.gymrat/``
session directory. When the experiment worktree already has uncommitted changes
that would be overwritten, the sync refuses — no partial application.
"""

from dataclasses import dataclass
from pathlib import Path

from gymrat.errors import GymratError
from gymrat.git import run_git
from gymrat.session.paths import SESSION_DIR_NAME, experiment_worktree_dir
from gymrat.session.store import require_open_session

# porcelain v1: two status chars (positions 0-1) followed by a space, then the
# path. The leading space of an unstaged-only entry is significant, so this
# must be a slice offset rather than a strip.
_STATUS_PREFIX_LEN = 3


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Outcome of syncing changes from the main tree to the experiment worktree."""

    files: tuple[str, ...]


def _dirty_files(directory: str) -> set[str]:
    """Relative paths of every uncommitted file in ``directory``, untracked included."""
    raw = run_git(["status", "--porcelain", "-uall"], directory)  # cspell:disable-line
    paths: set[str] = set()
    for line in raw.split("\n"):
        if not line:
            continue
        path_part = line[_STATUS_PREFIX_LEN:]
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1]
        paths.add(path_part)
    return paths


def _exclude_session_dir(paths: set[str]) -> set[str]:
    """Drop any path rooted under the session directory."""
    prefix = f"{SESSION_DIR_NAME}/"
    return {p for p in paths if not p.startswith(prefix) and p != SESSION_DIR_NAME}


def sync_to_experiment(root: str) -> SyncResult:
    """Apply uncommitted changes from the main tree onto the experiment worktree.

    Raises:
        GymratError: When no session is open, or when the experiment worktree has
            uncommitted changes that overlap with the files to sync.
    """
    require_open_session(root, "syncing changes")

    experiment = experiment_worktree_dir(root)
    main_dirty = _exclude_session_dir(_dirty_files(root))

    if not main_dirty:
        return SyncResult(files=())

    experiment_dirty = _dirty_files(experiment)
    conflicts = main_dirty & experiment_dirty
    if conflicts:
        listed = ", ".join(sorted(conflicts))
        message = f"Cannot sync — the experiment worktree has uncommitted changes in: {listed}"
        raise GymratError(message, hint="Settle or revert the experiment worktree first.")

    for relative in sorted(main_dirty):
        src = Path(root) / relative
        dst = Path(experiment) / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    return SyncResult(files=tuple(sorted(main_dirty)))
