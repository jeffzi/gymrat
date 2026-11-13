"""Behavioral tests for the git subprocess helpers.

Real-subprocess tests are parallel-safe: the ``create_scratch_repo`` factory
(see ``conftest.py``) gives every test its own temp git repository.
"""

import signal
import subprocess
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.git import (
    run_git,
    run_git_step,
    try_git,
)
from gymrat.signals import TERMINATION_SIGNALS
from tests._signal_masking import needs_signal_masking


def _signal_id(signal_number: int) -> str:
    return signal.Signals(signal_number).name


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


def _self_signalling_alias(signal_number: int) -> list[str]:
    """Git args for a shell alias that sends ``signal_number`` to itself, then prints ``survived``."""
    # Git starts the alias in a shell of its own, so the shell is a grandchild
    # of the caller: what it sees is what anything the git child spawns sees.
    return [
        "-c",
        f"alias.self-signal=!kill -{signal_number} $$; echo survived",
        "self-signal",
    ]


@needs_signal_masking
@pytest.mark.parametrize("signal_number", sorted(TERMINATION_SIGNALS), ids=_signal_id)
def test_run_git_when_interrupt_allowed_does_let_a_termination_signal_end_what_the_child_spawns(
    repo: str, signal_number: int
):
    with pytest.raises(subprocess.CalledProcessError):
        run_git(_self_signalling_alias(signal_number), repo, allow_interrupt=True)


@needs_signal_masking
def test_run_git_when_interrupt_not_allowed_does_keep_termination_signals_blocked_in_what_the_child_spawns(
    repo: str,
):
    result = run_git(_self_signalling_alias(signal.SIGTERM), repo)

    assert result == "survived\n"


# ---------------------------------------------------------------------------
# try_git
# ---------------------------------------------------------------------------


def test_try_git_when_command_succeeds_does_return_none(repo: str):
    result = try_git(["rev-parse", "HEAD"], repo)

    assert result is None


_UNKNOWN_REF_ARGS = ["rev-parse", "--verify", "does-not-exist"]


def _unknown_ref_stderr(repo: str) -> str:
    """Git's own stderr for looking up an unknown ref in ``repo``."""
    return subprocess.run(  # noqa: S603 -- fixed git binary plus test-chosen args
        ["git", *_UNKNOWN_REF_ARGS],  # noqa: S607 -- git resolved from PATH like a user's shell
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    ).stderr.strip()


def test_try_git_when_command_fails_does_return_gits_stderr(repo: str):
    expected = _unknown_ref_stderr(repo)

    result = try_git(_UNKNOWN_REF_ARGS, repo)

    assert result == expected


@pytest.mark.parametrize("unusable_git", [pytest.param(False, id="git-missing")], indirect=True)
@pytest.mark.usefixtures("unusable_git")
def test_try_git_when_git_cannot_be_started_does_return_the_launch_error_naming_git(
    tmp_path: Path,
):
    result = try_git(_UNKNOWN_REF_ARGS, str(tmp_path))

    assert result is not None
    assert "'git'" in result


# ---------------------------------------------------------------------------
# run_git_step
# ---------------------------------------------------------------------------


def test_run_git_step_when_command_fails_does_raise_with_git_diagnostics(repo: str):
    expected_stderr = _unknown_ref_stderr(repo)

    with pytest.raises(GymratError) as exc_info:
        run_git_step(_UNKNOWN_REF_ARGS, repo, "lookup failed", "retry")

    assert str(exc_info.value) == f"lookup failed: {expected_stderr}"
    assert isinstance(exc_info.value.__cause__, subprocess.CalledProcessError)
    assert exc_info.value.hint == "retry"


@pytest.mark.parametrize("unusable_git", [pytest.param(False, id="git-missing")], indirect=True)
@pytest.mark.usefixtures("unusable_git")
def test_run_git_step_when_git_cannot_be_started_does_raise_gymrat_error(tmp_path: Path):
    with pytest.raises(GymratError, match=r"^lookup failed: .+") as exc_info:
        run_git_step(["status"], str(tmp_path), "lookup failed")

    assert isinstance(exc_info.value.__cause__, OSError)
