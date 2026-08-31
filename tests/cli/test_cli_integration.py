"""End-to-end CLI tests over real subprocesses, repos, and shell bench scripts.

These exercise paths :class:`typer.testing.CliRunner` cannot reach: running the
installed entry module out of process so the real lock, worktree lifecycle, and
signal-driven cleanup all run. ``python -m gymrat.cli.app`` stands in for the
``gymrat`` console script.
"""

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.session.paths import lockfile_path, repo_root
from tests._cli import ENTRY as _ENTRY
from tests._cli import no_color_env as _env
from tests._git import git as _git
from tests.conftest import hold_lock

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell and signals")

_EMIT_ONE = "#!/bin/sh\necho 'METRIC x=1'\n"
_SLOW_BENCH = "#!/bin/sh\nsleep 5\necho 'METRIC x=1'\n"


# ---------------------------------------------------------------------------
# lock-free run outside a repo
# ---------------------------------------------------------------------------


def test_cli_when_outside_repo_does_measure_lock_free(tmp_path: Path):
    (tmp_path / "bench.sh").write_text(_EMIT_ONE, encoding="utf-8")

    result = subprocess.run(  # noqa: S603
        [*_ENTRY, "measure", "--bench", "sh bench.sh", "--samples", "2"],
        cwd=tmp_path,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "x" in result.stdout


# ---------------------------------------------------------------------------
# rival lock
# ---------------------------------------------------------------------------


def test_cli_when_rival_lock_held_does_exit_two_naming_holder_without_benching(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
):
    repo = create_scratch_repo()
    (Path(repo) / "bench.sh").write_text(_EMIT_ONE, encoding="utf-8")
    _git(repo, "add", "bench.sh")
    _git(repo, "commit", "-m", "add bench")
    lock_path = str(lockfile_path(repo_root(repo)))
    blocker = hold_lock(
        lock_path,
        holder={"pid": os.getpid(), "command": "measure", "at": "2026-01-01T00:00:00.000Z"},
    )

    try:
        result = subprocess.run(  # noqa: S603
            [*_ENTRY, "compare", "main", "main", "--bench", "sh bench.sh", "--samples", "1"],
            cwd=repo,
            env=_env(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        assert result.returncode == 2
        assert f"PID {os.getpid()}" in result.stderr
        assert list_worktree_dirs(repo, include_main=False) == []
    finally:
        blocker.release()


# ---------------------------------------------------------------------------
# usage errors
# ---------------------------------------------------------------------------


def test_cli_when_usage_error_does_exit_two_and_print_once(tmp_path: Path):
    result = subprocess.run(  # noqa: S603
        [*_ENTRY, "compare", "main"],
        cwd=tmp_path,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr.count("Usage:") == 1


# ---------------------------------------------------------------------------
# signal-driven cleanup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("signal_number", "expected_code"),
    [
        pytest.param(signal.SIGINT, 130, id="sigint"),
        pytest.param(signal.SIGTERM, 143, id="sigterm"),
        pytest.param(getattr(signal, "SIGHUP", None), 129, id="sighup"),
    ],
)
def test_cli_when_signalled_mid_run_does_exit_128_plus_signal_number_and_sweep_worktrees(
    signal_number: int,
    expected_code: int,
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
):
    repo = create_scratch_repo()
    (Path(repo) / "bench.sh").write_text(_SLOW_BENCH, encoding="utf-8")
    _git(repo, "add", "bench.sh")
    _git(repo, "commit", "-m", "slow bench")
    _git(repo, "switch", "-c", "candidate")
    _git(repo, "switch", "main")

    proc = subprocess.Popen(  # noqa: S603
        [*_ENTRY, "compare", "main", "candidate", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not list_worktree_dirs(repo, include_main=False):
            if time.monotonic() > deadline:
                proc.kill()
                pytest.fail("worktree never materialized before the signal")
            time.sleep(0.05)
        proc.send_signal(signal_number)
        proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == expected_code
    assert list_worktree_dirs(repo, include_main=False) == []
