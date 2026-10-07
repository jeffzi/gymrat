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
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.session.paths import lockfile_path, repo_root
from tests._cli import ENTRY as _ENTRY
from tests._cli import no_color_env as _env
from tests._cli import run_cli
from tests._git import EMIT_ONE_BENCH, list_worktree_dirs, wait_for_worktrees, write_committed_bench
from tests._git import run_git as _git
from tests._lock import FIXED_HOLDER_AT, hold_lock
from tests._process_helpers import wait_for_pid_file_blocking

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell and signals")

# A bench that records its pid at ``{pid_path}`` before sleeping, so the test can
# register it for reaping.
_SLOW_BENCH = "#!/bin/sh\necho $$ > '{pid_path}'\nsleep 5\necho 'METRIC x=1'\n"


# ---------------------------------------------------------------------------
# lock-free run outside a repo
# ---------------------------------------------------------------------------


def test_cli_when_outside_repo_does_measure_lock_free(tmp_path: Path):
    (tmp_path / "bench.sh").write_text(EMIT_ONE_BENCH, encoding="utf-8")

    result = run_cli(
        ["measure", "--bench", "sh bench.sh", "--samples", "2"], tmp_path, check=False, timeout=60
    )

    assert result.returncode == 0, result.stderr
    assert "x                │ 1 ± 0%" in result.stdout.splitlines()


# ---------------------------------------------------------------------------
# rival lock
# ---------------------------------------------------------------------------


def test_cli_when_rival_lock_held_does_exit_two_naming_holder_without_benching(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    write_committed_bench(repo, EMIT_ONE_BENCH)
    lock_path = lockfile_path(repo_root(repo))
    blocker = hold_lock(
        lock_path,
        holder={"pid": os.getpid(), "command": "measure", "at": FIXED_HOLDER_AT},
    )

    try:
        result = run_cli(
            ["compare", "main", "main", "--bench", "sh bench.sh", "--samples", "1"],
            repo,
            check=False,
            timeout=60,
        )

        assert result.returncode == 2
        assert f"PID {os.getpid()}" in result.stderr
        assert list_worktree_dirs(repo, include_main=False) == []
    finally:
        blocker.release()


# ---------------------------------------------------------------------------
# signal-driven shutdown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("signal_number", "expected_code"),
    [
        pytest.param(signal.SIGINT, 130, id="sigint"),
        pytest.param(signal.SIGTERM, 143, id="sigterm"),
        pytest.param(getattr(signal, "SIGHUP", None), 129, id="sighup"),
    ],
)
def test_cli_when_signalled_mid_run_does_exit_on_the_signal_status_leaving_no_worktree(
    signal_number: int,
    expected_code: int,
    create_scratch_repo: Callable[[], str],
    tmp_path: Path,
    reap_groups: list[int],
):
    repo = create_scratch_repo()
    pid_path = tmp_path / "bench.pid"
    write_committed_bench(repo, _SLOW_BENCH.format(pid_path=pid_path), message="slow bench")
    _git(["switch", "-c", "candidate"], repo)
    _git(["switch", "main"], repo)

    proc = subprocess.Popen(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [*_ENTRY, "compare", "main", "candidate", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_worktrees(repo, 1)
        reap_groups.append(os.getpgid(wait_for_pid_file_blocking(pid_path, timeout_s=30.0)))
        proc.send_signal(signal_number)
        proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == expected_code
    assert list_worktree_dirs(repo, include_main=False) == []


# ---------------------------------------------------------------------------
# a stranded worktree dir from a killed run survives a subsequent normal run
# ---------------------------------------------------------------------------


def test_compare_when_stranded_worktree_dir_preexists_does_not_sweep_or_corrupt_it(
    create_scratch_repo: Callable[[], str],
    tmp_path: Path,
):
    # A sweep by name pattern under a shared temp base cannot tell a stale leftover
    # from a concurrent run's live worktree, so a normal run must leave it alone.
    repo = create_scratch_repo()
    write_committed_bench(repo, EMIT_ONE_BENCH)
    _git(["switch", "-c", "candidate"], repo)
    _git(["switch", "main"], repo)
    controlled_base = tmp_path / "controlled-base"
    controlled_base.mkdir()
    stranded = controlled_base / "gymrat-wt-stranded-from-a-killed-run"
    stranded.mkdir()
    marker = stranded / "leftover.txt"
    marker.write_text("stranded", encoding="utf-8")
    env = _env()
    env["TMPDIR"] = str(controlled_base)

    result = subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
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
