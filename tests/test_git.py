"""Behavioral tests for the git subprocess helpers.

Real-subprocess tests are parallel-safe: the ``create_scratch_repo`` factory
(see ``conftest.py``) gives every test its own temp git repository.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from gymrat.errors import GymratError
from gymrat.git import (
    run_git,
    run_git_step,
    try_git,
)

# ---------------------------------------------------------------------------
# run_git
# ---------------------------------------------------------------------------


def test_run_git_when_repo_env_var_set_does_resolve_against_cwd(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GIT_DIR", "/nonexistent/.git")

    result = run_git(["rev-parse", "--git-dir"], repo)

    assert result == ".git\n"


def test_run_git_when_extra_env_overrides_scrubbed_key_does_restore_it(
    repo: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GIT_INDEX_FILE", "/nonexistent/.git/index")
    custom_env = {"GIT_INDEX_FILE": str(tmp_path / "custom-index")}
    (Path(repo) / "staged.txt").write_text("banana\n")
    run_git(["read-tree", "--empty"], repo, env=custom_env)

    run_git(["update-index", "--add", "--", "staged.txt"], repo, env=custom_env)

    assert run_git(["ls-files"], repo, env=custom_env) == "staged.txt\n"
    assert run_git(["diff", "--cached", "--name-only"], repo) == ""


# ---------------------------------------------------------------------------
# try_git
# ---------------------------------------------------------------------------


def test_try_git_when_command_succeeds_does_return_none(repo: str):
    result = try_git(["rev-parse", "HEAD"], repo)

    assert result is None


_UNKNOWN_REF_ARGS = ["rev-parse", "--verify", "does-not-exist"]


def _git_fails(_monkeypatch: pytest.MonkeyPatch, repo: str) -> str:
    """Leave ``subprocess.run`` alone and return git's own stderr for the failing command."""
    return subprocess.run(  # noqa: S603 -- fixed git binary plus test-chosen args
        ["git", *_UNKNOWN_REF_ARGS],  # noqa: S607 -- git resolved from PATH like a user's shell
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    ).stderr.strip()


def _git_missing(monkeypatch: pytest.MonkeyPatch, _repo: str) -> str:
    """Make ``subprocess.run`` raise as if git were absent and return the error's message."""
    message = "git"

    def raise_not_found(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(message)

    monkeypatch.setattr(
        subprocess, "run", create_autospec(subprocess.run, side_effect=raise_not_found)
    )
    return message


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_git_fails, id="command-fails"),
        pytest.param(_git_missing, id="git-binary-missing"),
    ],
)
def test_try_git_when_the_call_fails_does_return_the_diagnostic_text(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[pytest.MonkeyPatch, str], str],
):
    expected = arrange(monkeypatch, repo)

    result = try_git(_UNKNOWN_REF_ARGS, repo)

    assert result == expected


# ---------------------------------------------------------------------------
# run_git_step
# ---------------------------------------------------------------------------


def test_run_git_step_when_command_fails_does_raise_with_git_diagnostics(repo: str):
    with pytest.raises(GymratError) as exc_info:
        run_git_step(["rev-parse", "--verify", "does-not-exist"], repo, "lookup failed", "retry")

    assert str(exc_info.value).startswith("lookup failed: ")
    assert isinstance(exc_info.value.__cause__, subprocess.CalledProcessError)
    assert exc_info.value.hint == "retry"


@pytest.mark.parametrize("unusable_git", [pytest.param(False, id="git-missing")], indirect=True)
@pytest.mark.usefixtures("unusable_git")
def test_run_git_step_when_git_cannot_be_started_does_raise_gymrat_error(tmp_path: Path):
    with pytest.raises(GymratError, match=r"^lookup failed: .+"):
        run_git_step(["status"], str(tmp_path), "lookup failed")
