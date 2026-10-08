"""End-to-end CLI tests over real subprocesses, repos, and shell bench scripts.

These exercise paths :class:`typer.testing.CliRunner` cannot reach: running the
installed entry module out of process so the real lock, worktree lifecycle, and
signal-driven cleanup all run. ``python -m gymrat.cli.app`` stands in for the
``gymrat`` console script.
"""

import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.session.paths import lockfile_path, repo_root
from tests._cli import no_color_env as _env
from tests._cli import run_cli
from tests._git import EMIT_ONE_BENCH, list_worktree_dirs, wait_for_worktrees, write_committed_bench
from tests._lock import FIXED_HOLDER_AT, hold_lock
from tests._process_helpers import wait_for_pid_file_blocking
from tests.cli._signalled_cli import pid_recording_script, spawned_gymrat, stop_by_signal

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell and signals")

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


def test_cli_when_signalled_mid_run_does_exit_on_the_signal_status_leaving_no_worktree(
    create_scratch_repo: Callable[[], str],
    tmp_path: Path,
    reap_groups: list[int],
):
    repo = create_scratch_repo()
    pid_path = tmp_path / "bench.pid"
    # The bench records its pid before sleeping, so the test can register it for reaping.
    slow_bench = pid_recording_script(pid_path, "sleep 5\necho 'METRIC x=1'\n")
    write_committed_bench(repo, slow_bench, message="slow bench", branches=("candidate",))

    with spawned_gymrat(
        ["compare", "main", "candidate", "--bench", "sh bench.sh", "--samples", "1"], repo
    ) as proc:
        wait_for_worktrees(repo, 1)
        reap_groups.append(os.getpgid(wait_for_pid_file_blocking(pid_path, timeout_s=30.0)))
        stop_by_signal(proc, signal.SIGHUP)

    assert proc.returncode == 128 + signal.SIGHUP
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
    write_committed_bench(repo, EMIT_ONE_BENCH, branches=("candidate",))
    controlled_base = tmp_path / "controlled-base"
    controlled_base.mkdir()
    stranded = controlled_base / "gymrat-wt-stranded-from-a-killed-run"
    stranded.mkdir()
    marker = stranded / "leftover.txt"
    marker.write_text("stranded", encoding="utf-8")
    env = _env()
    env["TMPDIR"] = str(controlled_base)

    result = run_cli(
        ["compare", "main", "candidate", "--bench", "sh bench.sh", "--samples", "1"],
        repo,
        check=False,
        timeout=60,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert stranded.is_dir()
    assert marker.read_text(encoding="utf-8") == "stranded"
    assert list_worktree_dirs(repo, include_main=False) == []
