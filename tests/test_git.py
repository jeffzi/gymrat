"""Behavioral tests for the git subprocess helpers and lookup classification.

Real-subprocess tests are parallel-safe: the ``scratch_repo`` fixture gives
every test its own temp directory (``tempfile.mkdtemp``) resolved through
``os.path.realpath`` so macOS ``/var`` → ``/private/var`` matches what git
reports, and tears it down with ``shutil.rmtree``.
"""

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from gymrat_py.errors import GymratError
from gymrat_py.git import (
    NotAGitRepositoryError,
    repository_lookup_error,
    run_git,
    try_git,
)


def _run(args: list[str], cwd: str) -> None:
    """Run git in ``cwd`` for fixture setup, failing loudly on error."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)  # noqa: S603, S607


@pytest.fixture
def scratch_repo() -> Iterator[str]:
    """A throwaway git repo on ``main`` with one committed file.

    Its own temp directory per test keeps the suite order-independent and safe
    under ``pytest-xdist``.
    """
    directory = os.path.realpath(tempfile.mkdtemp(prefix="gymrat-test-"))
    try:
        _run(["init", "-b", "main"], directory)
        for key, value in (
            ("user.name", "Test User"),
            ("user.email", "test@example.com"),
            ("commit.gpgsign", "false"),
            ("core.autocrlf", "false"),
        ):
            _run(["config", key, value], directory)
        (Path(directory) / "README.md").write_text("# Test Repo\n", encoding="utf-8")
        _run(["add", "README.md"], directory)
        _run(["commit", "-m", "Initial commit"], directory)
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


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
