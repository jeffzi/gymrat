"""Behavioral tests for the git subprocess helpers and lookup classification.

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

from gymrat import git as git_module
from gymrat import signals
from gymrat.errors import GymratError
from gymrat.git import (
    NotAGitRepositoryError,
    repository_lookup_error,
    run_git,
    try_git,
)


@pytest.fixture
def scratch_repo(create_scratch_repo: Callable[[], str]) -> str:
    """A throwaway git repo on ``main`` with one committed file."""
    return create_scratch_repo()


# ---------------------------------------------------------------------------
# repository_lookup_error
# ---------------------------------------------------------------------------


def test_repository_lookup_error_when_stderr_is_fatal_not_a_repo_does_classify_as_missing():
    cause = subprocess.CalledProcessError(
        128,
        ["git"],
        stderr="fatal: not a git repository (or any of the parent directories): .git\n",
    )

    error = repository_lookup_error("/some/dir", cause)

    assert isinstance(error, NotAGitRepositoryError)


def test_repository_lookup_error_when_phrase_only_inside_path_does_not_classify_as_missing():
    cause = subprocess.CalledProcessError(
        128,
        ["git"],
        stderr="error: cannot open /tmp/not a git repository/config: No such file\n",
    )

    error = repository_lookup_error("/some/dir", cause)

    assert not isinstance(error, NotAGitRepositoryError)
    assert isinstance(error, GymratError)
    assert str(error) == (
        "Cannot determine the git repository at /some/dir: "
        "error: cannot open /tmp/not a git repository/config: No such file"
    )


# ---------------------------------------------------------------------------
# run_git
# ---------------------------------------------------------------------------


def test_run_git_when_rev_parse_head_does_return_forty_hex_sha(scratch_repo: str):
    result = run_git(["rev-parse", "HEAD"], scratch_repo)

    assert re.fullmatch(r"[0-9a-f]{40}", result.strip())


def test_run_git_when_repo_env_vars_set_does_scrub_them_and_use_cwd(
    scratch_repo: str, monkeypatch: pytest.MonkeyPatch
):
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        monkeypatch.setenv(key, "/nonexistent/.git")

    result = run_git(["rev-parse", "--git-dir"], scratch_repo)

    assert result.strip() == ".git"


# ---------------------------------------------------------------------------
# try_git
# ---------------------------------------------------------------------------


def test_try_git_when_command_succeeds_does_return_none(scratch_repo: str):
    assert try_git(["rev-parse", "HEAD"], scratch_repo) is None


def test_try_git_when_command_fails_does_return_stderr_diagnostic(scratch_repo: str):
    result = try_git(["rev-parse", "--verify", "does-not-exist"], scratch_repo)

    assert result
    assert "fatal" in result


def test_try_git_when_git_binary_missing_does_return_diagnostic_string(
    scratch_repo: str, monkeypatch: pytest.MonkeyPatch
):
    def raise_not_found(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        message = "git"
        raise FileNotFoundError(message)

    monkeypatch.setattr(subprocess, "run", raise_not_found)

    result = try_git(["rev-parse", "HEAD"], scratch_repo)

    assert result is not None
    assert isinstance(result, str)


def test_try_git_when_command_times_out_does_return_diagnostic_string(
    scratch_repo: str, monkeypatch: pytest.MonkeyPatch
):
    def raise_timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=1)

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    result = try_git(["rev-parse", "HEAD"], scratch_repo)

    assert result is not None
    assert isinstance(result, str)


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

_TERMINATION_SIGNALS = [
    pytest.param(resolved, id=name)
    for name in ("SIGINT", "SIGTERM", "SIGHUP")
    if (resolved := getattr(signal, name, None)) is not None
]


def _install_sleep_post_checkout_hook(repo_dir: str, seconds: int) -> None:
    """Install a post-checkout hook that only sleeps.

    Keeps ``git worktree add`` in-flight long enough for a mid-call signal to
    land. Borrows the hook mechanism from ``kill_git_during_worktree_add`` but
    inverts its body: instead of killing git, git runs to completion slowly.
    """
    hook_path = Path(repo_dir) / ".git" / "hooks" / "post-checkout"
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(f"#!/bin/sh\nsleep {seconds}\n", encoding="utf-8")
    hook_path.chmod(0o755)


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="Signal masking requires POSIX pthread_sigmask",
)
@pytest.mark.parametrize("term_signal", _TERMINATION_SIGNALS)
def test_run_git_when_termination_signal_arrives_mid_call_does_defer_cleanup_until_git_exits(
    term_signal: signal.Signals,
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    repo = create_scratch_repo()
    _install_sleep_post_checkout_hook(repo, _WORKTREE_SLEEP_SECONDS)
    worktree_dir = str(Path(repo) / "wt")

    exit_record: dict[str, float] = {}

    def record_exit(code: int) -> None:
        exit_record["code"] = code
        exit_record["at"] = time.monotonic()

    monkeypatch.setattr(signals, "_exit_process", record_exit)

    sweep_record: dict[str, bool] = {}

    def sweep_cleanup() -> None:
        sweep_record["worktree_materialized"] = (Path(worktree_dir) / ".git").exists()
        shutil.rmtree(worktree_dir, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],  # noqa: S607
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
        while "code" not in exit_record and time.monotonic() < deadline:
            time.sleep(_EXIT_POLL_INTERVAL_SECONDS)
    finally:
        timer.cancel()
        uninstall()

    assert exit_record["code"] == 128 + int(term_signal)
    assert exit_record["at"] - started >= _WORKTREE_SLEEP_SECONDS * _DEFERRAL_ELAPSED_FRACTION
    assert sweep_record["worktree_materialized"]
    assert list_worktree_dirs(repo, include_main=False) == []


def test_run_git_does_not_define_own_signal_deferral():
    assert not hasattr(git_module, "_deferring_termination_signals")


def test_run_git_when_object_env_vars_set_does_scrub_them_and_use_cwd(
    scratch_repo: str, monkeypatch: pytest.MonkeyPatch
):
    for key in ("GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        monkeypatch.setenv(key, "/nonexistent/objects")

    result = run_git(["rev-parse", "--git-dir"], scratch_repo)

    assert result.strip() == ".git"


def test_run_git_when_pthread_sigmask_unavailable_does_run_unmasked_and_return_stdout(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    repo = create_scratch_repo()
    monkeypatch.setattr(signals, "pthread_sigmask", None)

    result = run_git(["rev-parse", "HEAD"], repo)

    assert re.fullmatch(r"[0-9a-f]{40}", result.strip())


# ---------------------------------------------------------------------------
# run_git — stdin is closed
# ---------------------------------------------------------------------------


def test_run_git_when_invoked_does_pass_stdin_devnull(
    scratch_repo: str, monkeypatch: pytest.MonkeyPatch
):
    captured_kwargs: list[dict[str, object]] = []
    real_run = subprocess.run

    def recording_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_kwargs.append(dict(kwargs))
        return real_run(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "run", recording_run)

    run_git(["rev-parse", "HEAD"], scratch_repo)

    assert captured_kwargs
    assert captured_kwargs[0]["stdin"] is subprocess.DEVNULL
