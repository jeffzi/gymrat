"""Temp-directory and permission-edge hardening for the worktree lifecycle.

The worktree base is whatever ``tempfile.gettempdir`` resolves to. These tests
pin what happens at the awkward edges of that directory:

- a read-only base makes worktree creation fail cleanly — the error names the
  directory, the run leaves no half-registered worktree behind, and no raw
  Python traceback reaches the user (exit 2 via the tool-failure path);
- a base reached through a symlink or written with a trailing slash is planned
  under its resolved real path, so a later sweep removes exactly what git
  registered with no leftover and no double-path confusion;
- a directory stranded under the base by a previously killed run is left alone
  by a subsequent normal run — no production sweep deletes it (see
  ``.planning/hardening-decisions.md`` for why a sweep is deliberately absent).

The read-only and symlink tests are POSIX-only: they rely on ``chmod`` mode
bits and real symlinks. ``chmod`` fixtures always restore ``0o700`` in a
``finally`` block so a failed assertion never strands an unreadable directory
for ``pytest-xdist`` workers or ``task clean``. Root ignores the mode bits, so
those tests skip when running as root.
"""

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat_py.errors import GymratError
from gymrat_py.targets import (
    RefTarget,
    cleanup_worktrees,
    materialize_worktree,
    plan_worktree,
)
from tests.hardening._bench_helpers import env as _env
from tests.hardening._bench_helpers import write_committed_bench as _write_committed_bench

# A sha no repository holds, so planning never needs a real commit to build a path.
UNKNOWN_SHA = "0" * 40

_IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0

skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only shell bench and worktree paths"
)
skip_on_windows_or_root = pytest.mark.skipif(
    sys.platform == "win32" or _IS_ROOT,
    reason="Windows lacks EACCES from chmod and root bypasses the mode bits",
)

_ENTRY = [sys.executable, "-m", "gymrat_py.cli.app"]

_FAST_BENCH = "#!/bin/sh\necho 'METRIC x=1'\n"


def _run_git(args: list[str], cwd: str) -> str:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _get_head_sha(repo_dir: str) -> str:
    return _run_git(["rev-parse", "HEAD"], repo_dir).strip()


def _point_temp_base_at(monkeypatch: pytest.MonkeyPatch, real_base: Path, shape: str) -> None:
    """Route ``tempfile.gettempdir`` at ``real_base`` through an awkward spelling.

    ``symlink`` hands back a symlink that resolves to ``real_base``;
    ``trailing-slash`` hands back ``real_base`` with a trailing separator. Both
    must resolve to the same real directory so the planner normalizes them
    identically. The function is patched directly (not the ``TMPDIR`` env var)
    because ``gettempdir`` caches its first result and silently falls back to a
    writable candidate when the env value is not usable.
    """
    if shape == "symlink":
        link = real_base.parent / f"link-{real_base.name}"
        link.symlink_to(real_base)
        configured = str(link)
    else:
        configured = str(real_base) + os.sep
    monkeypatch.setattr(tempfile, "gettempdir", lambda: configured)


# ---------------------------------------------------------------------------
# read-only temp base
# ---------------------------------------------------------------------------


@skip_on_windows_or_root
def test_materialize_worktree_when_temp_dir_read_only_does_fail_naming_dir_without_partial_registration(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    repo = create_scratch_repo()
    sha = _get_head_sha(repo)
    read_only_base = tmp_path / "read-only-base"
    read_only_base.mkdir()
    read_only_base.chmod(0o500)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(read_only_base))
    worktree = plan_worktree(RefTarget(ref=sha, resolved_sha=sha))

    try:
        with pytest.raises(GymratError) as exc_info:
            materialize_worktree(worktree, repo)

        message = str(exc_info.value)
        assert "worktree add failed" in message
        assert os.path.realpath(str(read_only_base)) in message
        assert "Traceback (most recent call last)" not in message
        assert "returned non-zero exit status" not in message
        assert worktree.created is False
        assert list_worktree_dirs(repo, include_main=False) == []
    finally:
        read_only_base.chmod(0o700)


# ---------------------------------------------------------------------------
# symlinked / trailing-slash temp base resolves to the real path
# ---------------------------------------------------------------------------


@skip_on_windows
@pytest.mark.parametrize("shape", ["symlink", "trailing-slash"])
def test_plan_worktree_when_temp_dir_symlink_or_trailing_slash_does_plan_under_resolved_real_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shape: str
):
    real_base = tmp_path / "real-base"
    real_base.mkdir()
    _point_temp_base_at(monkeypatch, real_base, shape)

    worktree = plan_worktree(RefTarget(ref="v1", resolved_sha=UNKNOWN_SHA))

    assert str(Path(worktree.dir).parent) == os.path.realpath(str(real_base))
    assert worktree.dir == os.path.normpath(worktree.dir)


@skip_on_windows
@pytest.mark.parametrize("shape", ["symlink", "trailing-slash"])
def test_cleanup_worktrees_when_temp_dir_symlink_or_trailing_slash_does_sweep_without_leftover(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shape: str,
):
    repo = create_scratch_repo()
    sha = _get_head_sha(repo)
    real_base = tmp_path / "real-base"
    real_base.mkdir()
    _point_temp_base_at(monkeypatch, real_base, shape)
    worktree = plan_worktree(RefTarget(ref=sha, resolved_sha=sha))
    materialize_worktree(worktree, repo)

    result = cleanup_worktrees([worktree], repo)

    assert result.removed == 1
    assert not result.failures
    assert not Path(worktree.dir).exists()
    assert list_worktree_dirs(repo, include_main=False) == []


# ---------------------------------------------------------------------------
# a stranded worktree dir from a killed run survives a subsequent normal run
# ---------------------------------------------------------------------------


@skip_on_windows
def test_compare_when_stranded_worktree_dir_preexists_does_not_sweep_or_corrupt_it(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
    tmp_path: Path,
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _FAST_BENCH)
    _run_git(["switch", "-c", "candidate"], repo)
    _run_git(["switch", "main"], repo)
    controlled_base = tmp_path / "controlled-base"
    controlled_base.mkdir()
    stranded = controlled_base / "gymrat-wt-stranded-from-a-killed-run"
    stranded.mkdir()
    marker = stranded / "leftover.txt"
    marker.write_text("stranded", encoding="utf-8")
    env = _env()
    env["TMPDIR"] = str(controlled_base)

    result = subprocess.run(  # noqa: S603
        [*_ENTRY, "compare", "main", "candidate", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert stranded.is_dir()
    assert marker.read_text(encoding="utf-8") == "stranded"
    assert list_worktree_dirs(repo, include_main=False) == []
