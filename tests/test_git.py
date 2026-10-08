"""Behavioral tests for the git subprocess helpers.

Real-subprocess tests are parallel-safe: the ``create_scratch_repo`` factory
(see ``conftest.py``) gives every test its own temp git repository.
"""

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat import signals
from gymrat.git import (
    run_git,
    try_git,
)
from tests._git import install_git_hook, list_worktree_dirs

# ---------------------------------------------------------------------------
# run_git
# ---------------------------------------------------------------------------


def test_run_git_when_rev_parse_head_does_return_forty_hex_sha(repo: str):
    result = run_git(["rev-parse", "HEAD"], repo)

    assert re.fullmatch(r"[0-9a-f]{40}", result.strip())


_GIT_DIR_ARGS = ["rev-parse", "--git-dir"]


@pytest.mark.parametrize(
    ("key", "args", "expected"),
    [
        pytest.param("GIT_DIR", _GIT_DIR_ARGS, ".git\n", id="GIT_DIR"),
        pytest.param("GIT_WORK_TREE", _GIT_DIR_ARGS, ".git\n", id="GIT_WORK_TREE"),
        pytest.param("GIT_COMMON_DIR", _GIT_DIR_ARGS, ".git\n", id="GIT_COMMON_DIR"),
        pytest.param("GIT_OBJECT_DIRECTORY", _GIT_DIR_ARGS, ".git\n", id="GIT_OBJECT_DIRECTORY"),
        pytest.param("GIT_INDEX_FILE", ["ls-files"], "README.md\n", id="GIT_INDEX_FILE"),
    ],
)
def test_run_git_when_repo_env_var_set_does_scrub_it_and_use_cwd(
    repo: str, monkeypatch: pytest.MonkeyPatch, key: str, args: list[str], expected: str
):
    monkeypatch.setenv(key, "/nonexistent/.git")

    result = run_git(args, repo)

    assert result == expected


def test_run_git_when_alternate_object_dirs_set_does_not_read_objects_through_them(
    repo: str, create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    other = create_scratch_repo()
    (Path(other) / "banana.txt").write_text("only in the other repo\n")
    blob = run_git(["hash-object", "-w", "banana.txt"], other).strip()
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(Path(other) / ".git" / "objects"))

    with pytest.raises(subprocess.CalledProcessError):
        run_git(["cat-file", "-e", blob], repo)


def test_run_git_when_extra_env_passed_does_apply_it_to_child_process(repo: str):
    result = run_git(
        ["var", "GIT_AUTHOR_IDENT"],
        repo,
        env={"GIT_AUTHOR_NAME": "Banana", "GIT_AUTHOR_EMAIL": "banana@example.com"},
    )

    assert "Banana" in result


def test_run_git_when_extra_env_overrides_scrubbed_key_does_restore_it(
    repo: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GIT_INDEX_FILE", "/nonexistent/.git/index")
    custom_index = str(tmp_path / "custom-index")
    (Path(repo) / "staged.txt").write_text("banana\n")
    run_git(["read-tree", "--empty"], repo, env={"GIT_INDEX_FILE": custom_index})
    run_git(
        ["update-index", "--add", "--", "staged.txt"],
        repo,
        env={"GIT_INDEX_FILE": custom_index},
    )

    tree_sha = run_git(["write-tree"], repo, env={"GIT_INDEX_FILE": custom_index}).strip()

    listing = run_git(["ls-tree", tree_sha], repo)
    real_index_status = run_git(["diff", "--cached", "--name-only"], repo)

    assert "staged.txt" in listing
    assert real_index_status == ""


# ---------------------------------------------------------------------------
# try_git
# ---------------------------------------------------------------------------


def test_try_git_when_command_succeeds_does_return_none(repo: str):
    result = try_git(["rev-parse", "HEAD"], repo)

    assert result is None


def _git_fails(_monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave ``subprocess.run`` alone; the command itself fails."""


def _git_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_not_found(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        message = "git"
        raise FileNotFoundError(message)

    monkeypatch.setattr(subprocess, "run", raise_not_found)


def _git_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=1)

    monkeypatch.setattr(subprocess, "run", raise_timeout)


@pytest.mark.parametrize(
    ("arrange", "diagnostic"),
    [
        pytest.param(_git_fails, "fatal: Needed a single revision", id="command-fails"),
        pytest.param(_git_missing, "git", id="git-binary-missing"),
        pytest.param(
            _git_times_out,
            "Command '['git']' timed out after 1 seconds",
            id="command-times-out",
        ),
    ],
)
def test_try_git_when_the_call_fails_does_return_the_diagnostic_text(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[pytest.MonkeyPatch], None],
    diagnostic: str,
):
    arrange(monkeypatch)

    result = try_git(["rev-parse", "--verify", "does-not-exist"], repo)

    assert result == diagnostic


# ---------------------------------------------------------------------------
# run_git — termination-signal deferral
# ---------------------------------------------------------------------------

# The whole git call runs inside a post-checkout ``sleep``; a signal is delivered
# to this process partway through that sleep. Masked, the deferred handler cannot
# fire until the mask is restored — well after the sleep ends — so the elapsed
# time measured at the handler is at least this fraction of the sleep.
_WORKTREE_SLEEP_SECONDS = 1
_SIGNAL_DELAY_SECONDS = 0.2
_DEFERRAL_ELAPSED_FRACTION = 0.5
_EXIT_POLL_TIMEOUT_SECONDS = 2.0
_EXIT_POLL_INTERVAL_SECONDS = 0.01


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="Signal masking requires POSIX pthread_sigmask",
)
def test_run_git_when_termination_signal_arrives_mid_call_does_defer_cleanup_until_git_exits(
    repo: str, recorded_exits: list[tuple[int, float]]
):
    # One signal suffices here: the deferred set is pinned in the signals suite.
    term_signal = signal.SIGTERM
    # A hook that only sleeps keeps ``git worktree add`` in flight long enough
    # for a mid-call signal to land.
    install_git_hook(repo, "post-checkout", f"sleep {_WORKTREE_SLEEP_SECONDS}\n")
    worktree_dir = str(Path(repo) / "wt")

    sweep_record: dict[str, bool] = {}

    def sweep_cleanup() -> None:
        sweep_record["worktree_materialized"] = (Path(worktree_dir) / ".git").exists()
        shutil.rmtree(worktree_dir, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],  # noqa: S607 -- git resolved from PATH like a user's shell
            cwd=repo,
            check=False,
            capture_output=True,
        )

    uninstall = signals.install_termination_cleanup(sweep_cleanup)
    timer = threading.Timer(_SIGNAL_DELAY_SECONDS, os.kill, args=(os.getpid(), term_signal))
    try:
        started = time.monotonic()
        timer.start()
        run_git(["worktree", "add", "--detach", worktree_dir, "HEAD"], repo)
        deadline = time.monotonic() + _EXIT_POLL_TIMEOUT_SECONDS
        while not recorded_exits and time.monotonic() < deadline:
            time.sleep(_EXIT_POLL_INTERVAL_SECONDS)
    finally:
        timer.cancel()
        uninstall()

    [(exit_code, exited_at)] = recorded_exits
    assert exit_code == 128 + int(term_signal)
    assert exited_at - started >= _WORKTREE_SLEEP_SECONDS * _DEFERRAL_ELAPSED_FRACTION
    assert sweep_record["worktree_materialized"]
    assert list_worktree_dirs(repo, include_main=False) == []


def test_run_git_when_pthread_sigmask_unavailable_does_run_unmasked_and_return_stdout(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(signals, "pthread_sigmask", None)

    result = run_git(["rev-parse", "HEAD"], repo)

    assert re.fullmatch(r"[0-9a-f]{40}", result.strip())


# ---------------------------------------------------------------------------
# run_git — stdin is closed
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("stdin_holding_text")
def test_run_git_when_a_command_reads_stdin_does_see_it_closed(repo: str):
    # A closed stdin hashes to git's well-known empty-blob id; the inherited one holds text.
    result = run_git(["hash-object", "--stdin"], repo)

    assert result.strip() == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
