"""Command-level tests for the ``gymrat supervise`` dirty-tree guards.

A dirty working tree, or an experiment worktree holding unmeasured or unsettled
edits, stops the run before the agent starts; ``--allow-dirty`` lets only the
working-tree case through, with a warning. The guards run inside the
pre-flight, so these tests run the real pre-flight with only the baseline bench
replaced.
"""

import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli.supervise.preflight import run_preflight, validate_experiment_worktree
from gymrat.session.paths import experiment_worktree_dir, session_jsonl_path
from gymrat.session.store import append_record
from tests.cli._session import make_discard_repo
from tests.cli.commands.supervise.test_supervise import _err_text, _run, _Seams
from tests.cli.commands.supervise.test_supervise import _install_seams as _install_command_seams
from tests.cli.supervise._fixtures import install_baseline_seam, start_open_session
from tests.session.records._fixtures import committed_keep, finalize_record, iteration_record


def _install_seams(monkeypatch: pytest.MonkeyPatch) -> _Seams:
    """Install the command seams, keeping the real pre-flight with a stand-in baseline bench."""
    seams = _install_command_seams(monkeypatch)
    monkeypatch.setattr("gymrat.cli.commands.supervise.run_preflight", run_preflight)
    install_baseline_seam(monkeypatch)
    return seams


# ---------------------------------------------------------------------------
# dirty-tree guard
# ---------------------------------------------------------------------------


def test_supervise_when_tree_dirty_and_not_allowed_does_exit_two_with_guidance(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    (Path(repo) / "uncommitted.txt").write_text("dirty", encoding="utf-8")

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert re.search(r"dirty|uncommitted|untracked", result.stderr, re.IGNORECASE)
    assert re.search(r"commit|stash", result.stderr, re.IGNORECASE)
    assert "--allow-dirty" in result.stderr


def test_supervise_when_tree_dirty_and_allowed_does_warn_and_proceed(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    (Path(repo) / "uncommitted.txt").write_text("dirty", encoding="utf-8")

    result = _run("optimize it", "--max-minutes", "10", "--allow-dirty")

    assert result.exit_code == 0
    assert re.search(r"dirty|uncommitted|untracked", result.stderr, re.IGNORECASE)


def test_supervise_when_tree_clean_does_not_warn(repo: str, monkeypatch: pytest.MonkeyPatch):
    _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert not re.search(r"dirty|uncommitted|untracked", result.stderr, re.IGNORECASE)


def test_supervise_when_untracked_directory_dirty_does_count_its_files(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    nested = Path(repo) / "new-dir"
    nested.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (nested / name).write_text(name, encoding="utf-8")

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert "3" in result.stderr


# ---------------------------------------------------------------------------
# dirty experiment-worktree guard
# ---------------------------------------------------------------------------


def _dirty_experiment_worktree(repo: str, *names: str) -> None:
    worktree = Path(experiment_worktree_dir(repo))
    for name in names:
        (worktree / name).write_text("dirty\n", encoding="utf-8")


def _setup_finalized_with_dirty_worktree(repo: str) -> None:
    """A finalized session whose experiment worktree still has uncommitted files."""
    start_open_session(repo)
    log = session_jsonl_path(repo)
    append_record(log, iteration_record(seq=1))
    append_record(log, committed_keep(seq=1))
    append_record(log, finalize_record())
    _dirty_experiment_worktree(repo, "stale.txt")


def _setup_open_session_missing_worktree(repo: str) -> None:
    """An open session whose experiment worktree directory no longer exists on disk."""
    start_open_session(repo)
    shutil.rmtree(experiment_worktree_dir(repo))


@pytest.mark.parametrize(
    "extra_args",
    [
        pytest.param((), id="default"),
        pytest.param(("--allow-dirty",), id="allow-dirty"),
    ],
)
def test_supervise_when_experiment_worktree_dirty_with_unsettled_does_exit_two_with_settle_hint(
    repo: str, monkeypatch: pytest.MonkeyPatch, extra_args: tuple[str, ...]
):
    _install_seams(monkeypatch)
    make_discard_repo(repo)
    _dirty_experiment_worktree(repo, "scratch.txt")

    result = _run("optimize it", "--max-minutes", "10", *extra_args)

    assert result.exit_code == 2
    text = _err_text(result)
    assert re.search(r"unsettled", text, re.IGNORECASE)
    assert "gymrat keep" in text
    assert "gymrat discard" in text


def test_supervise_when_experiment_worktree_dirty_without_unsettled_does_exit_two_with_iterate_and_discard_hint(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    start_open_session(repo)
    _dirty_experiment_worktree(repo, "a.txt", "b.txt")

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    text = _err_text(result)
    assert "2 unmeasured edit" in text
    assert "gymrat iterate" in text
    assert "gymrat discard" in text


def test_validate_experiment_worktree_when_session_finalized_does_not_refuse_its_leftover_edits(
    repo: str,
):
    _setup_finalized_with_dirty_worktree(repo)

    validate_experiment_worktree(repo)


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(_setup_open_session_missing_worktree, id="missing-worktree"),
        # An open session whose experiment worktree has no uncommitted changes.
        pytest.param(start_open_session, id="clean-worktree"),
    ],
)
def test_supervise_when_experiment_worktree_guard_finds_no_issue_does_proceed(
    repo: str, monkeypatch: pytest.MonkeyPatch, setup: Callable[[str], None]
):
    _install_seams(monkeypatch)
    setup(repo)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
