"""Behavioral tests for ``run_exit_sequence`` — the settle decisions.

What the sequence keeps, what it discards, and what it refuses to touch because the
tree moved behind the measurement or a hook failed. The frame around those decisions
(the lock wait, the command record, finalize, the wire events, the error boundary) is
pinned in ``test_exit_sequence.py``.

Every test drives the real sequence against a throwaway repository from the shared
``create_scratch_repo`` factory, so the suite is order-independent and safe under
``pytest-xdist`` / ``pytest-randomly``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import create_autospec

import pytest

from gymrat.exec import exec as exec_command
from gymrat.session.paths import experiment_worktree_dir
from gymrat.session.records import DiscardRecord, KeepChecks, KeepRecord
from gymrat.session.workspace import worktree_fingerprint
from tests._exec_fixtures import (
    expected_result,
    install_exec,
)
from tests._git import status_of
from tests.loop._settle import (
    CHECKS,
    KEEP_EXEC,
    UNUSED_EXEC,
    checks_fail,
    checks_pass,
    confirmed_regression,
    edit_experiment,
    settling_record_of,
    start_with,
    unimproved,
)
from tests.session.records._fixtures import (
    append_records,
    blocked_keep,
    command_record,
    hook_record,
    iteration_record,
    last_command_record,
    records_of_type,
)
from tests.supervisor._exit_sequence import (
    fingerprint,
    improved_iteration,
    is_kept_step,
    measured,
    run_sequence,
    session_context,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.exec import ExecOptions, ExecResult
    from gymrat.session.records import SessionLogRecord

#: The token the gate reason carries when an edit moved the tree behind the measured fingerprint.
TREE_CHANGED = "tree changed"


def _edit_again(root: str) -> None:
    """Rewrite a tracked file, leaving the worktree off the measured fingerprint."""
    (Path(experiment_worktree_dir(root)) / "README.md").write_text(
        "# edited again\n", encoding="utf-8"
    )


def _rewriting_checks(monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    """Answer the checks command with a pass that rewrote a tracked file, as a formatter would."""

    async def run(_command: str, _options: ExecOptions) -> ExecResult:
        readme = Path(experiment_worktree_dir(root)) / "README.md"
        await asyncio.to_thread(readme.write_text, "# reformatted\n", encoding="utf-8")
        return expected_result()

    monkeypatch.setattr("gymrat.loop.keep.exec", create_autospec(exec_command, side_effect=run))


def _stale_tree(root: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """An improved iteration whose worktree moved on behind the fingerprint."""
    measured(root, iteration_record(seq=1))
    _edit_again(root)


def _no_fingerprint(root: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """An improved iteration that recorded no fingerprint at all."""
    edit_experiment(root)
    append_records(root, iteration_record(seq=1, measured_tree=None))


def _failed_before_hook_on_a_retry(root: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """An improved iteration whose before hook failed, recorded after an attempt that raised."""
    append_records(
        root,
        hook_record(seq=1, stage="before"),
        command_record(seq=1),
        hook_record(seq=1, stage="before", exit_code=1),
    )
    measured(root, iteration_record(seq=1))


def _failed_after_hook_ahead_of_a_command_record(
    root: str, _monkeypatch: pytest.MonkeyPatch
) -> None:
    """An improved iteration whose after hook failed, with a command record written behind it."""
    measured(
        root,
        iteration_record(seq=1),
        hook_record(seq=1, stage="after", exit_code=1),
        command_record(seq=1),
    )


def _failed_standing_attempt(root: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    # The first attempt's hook passed; the retry that recorded the iteration is the one that failed.
    append_records(
        root,
        hook_record(seq=1, stage="before"),
        hook_record(seq=1, stage="before", exit_code=1),
    )
    measured(root, iteration_record(seq=1))


def _standing_tree_without_fingerprint(root: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An improved iteration whose standing tree cannot be fingerprinted when the gate reads it."""
    improved_iteration(root)
    monkeypatch.setattr(
        "gymrat.supervisor.exit_sequence.worktree_fingerprint",
        create_autospec(worktree_fingerprint, return_value=None),
    )


def _gating_block_iteration(root: str) -> None:
    """A confirmed regression standing under a gating block, ready for the gate to decide it."""
    measured(
        root,
        confirmed_regression(1),
        blocked_keep(1, reason="gating-regression", checks=KeepChecks(configured=True)),
    )


# ---------------------------------------------------------------------------
# keeping an unsettled improved iteration
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_unsettled_improved_and_gate_passes_does_commit_the_iteration(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    measured(repo, iteration_record(seq=3))
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0].kind == "settled"
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.message is not None
    assert "3" in record.message


async def test_run_exit_sequence_when_no_checks_are_configured_does_route_the_missing_checks_warning_to_the_sink(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    start_with(repo)
    improved_iteration(repo)
    install_exec(monkeypatch, KEEP_EXEC, UNUSED_EXEC)

    run = await run_sequence(session_context(repo))

    assert is_kept_step(run.report.steps[0], checks="checks not configured")
    assert run.warnings != []
    assert capsys.readouterr().err == ""


async def test_run_exit_sequence_when_the_checks_rewrite_the_tree_does_note_it_on_the_kept_step(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    _rewriting_checks(monkeypatch, repo)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    kept = run.report.steps[0]
    assert (is_kept_step(kept), TREE_CHANGED in kept.text) == (True, True)


async def test_run_exit_sequence_when_the_kept_tree_has_no_fingerprint_does_add_no_note(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    checks_pass(monkeypatch)
    standing = worktree_fingerprint(experiment_worktree_dir(repo))

    def fingerprint_until_kept(directory: Path) -> str | None:
        # The standing edit fingerprints as usual; once the keep commits it, the
        # clean tree's read fails, so only the kept-tree note loses its input.
        return standing if status_of(str(directory)) else None

    monkeypatch.setattr(
        "gymrat.supervisor.exit_sequence.worktree_fingerprint",
        create_autospec(worktree_fingerprint, side_effect=fingerprint_until_kept),
    )

    run = await run_sequence(session_context(repo, checks=CHECKS))

    kept = run.report.steps[0]
    assert (is_kept_step(kept), TREE_CHANGED in kept.text) == (True, False)


async def test_run_exit_sequence_when_the_checks_fail_does_leave_the_iteration_for_a_person(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    checks_fail(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    (left,) = run.report.steps
    assert (left.kind, "iteration 1" in left.text, "checks failed" in left.text) == (
        "left",
        True,
        True,
    )
    assert status_of(experiment_worktree_dir(repo)) != ""
    command = last_command_record(repo)
    assert (command.exit_code, command.reason, command.seq) == (1, "checks-failed", 1)


async def test_run_exit_sequence_when_the_agent_committed_nothing_does_report_it_settled(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    append_records(repo, iteration_record(seq=1, measured_tree=fingerprint(repo)))
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    settled = run.report.steps[0]
    assert (settled.kind, "iteration 1" in settled.text, "nothing to commit" in settled.text) == (
        "settled",
        True,
        True,
    )
    command = last_command_record(repo)
    assert (command.exit_code, command.reason, command.seq) == (0, None, 1)


# ---------------------------------------------------------------------------
# the gate refusing an improved iteration
# ---------------------------------------------------------------------------


GATE_FAILURES = [
    pytest.param(_stale_tree, TREE_CHANGED, id="tree-changed-after-measuring"),
    pytest.param(_no_fingerprint, "fingerprint unavailable", id="fingerprint-unavailable"),
    pytest.param(
        _failed_before_hook_on_a_retry, "before hook failed", id="before-hook-failed-on-a-retry"
    ),
    pytest.param(
        _failed_after_hook_ahead_of_a_command_record,
        "after hook failed",
        id="after-hook-failed-ahead-of-a-command-record",
    ),
    pytest.param(
        _failed_standing_attempt, "before hook failed", id="only-the-standing-attempt-failed"
    ),
    pytest.param(
        _standing_tree_without_fingerprint,
        "fingerprint unavailable",
        id="standing-tree-without-fingerprint",
    ),
]


@pytest.mark.parametrize(("arrange", "reason"), GATE_FAILURES)
async def test_run_exit_sequence_when_improved_but_the_gate_fails_does_leave_the_iteration_naming_why_and_where(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[str, pytest.MonkeyPatch], None],
    reason: str,
):
    start_with(repo)
    arrange(repo, monkeypatch)
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    left = run.report.steps[0]
    assert (left.kind, "iteration 1" in left.text, reason in left.text) == ("left", True, True)
    assert experiment_worktree_dir(repo) in left.text
    assert records_of_type(repo, KeepRecord | DiscardRecord) == []


# ---------------------------------------------------------------------------
# discarding an unsettled iteration that did not improve
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_unsettled_and_not_improved_does_revert_it_without_checks(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    measured(repo, unimproved(1, "regressed"))
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    (settled,) = run.report.steps
    assert (settled.kind, "iteration 1" in settled.text, "regressed" in settled.text) == (
        "settled",
        True,
        True,
    )
    assert isinstance(settling_record_of(repo), DiscardRecord)


async def test_run_exit_sequence_when_not_improved_and_the_gate_fails_does_revert_nothing(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    measured(repo, unimproved(1, "regressed"))
    _edit_again(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    left = run.report.steps[0]
    assert (left.kind, "regressed" in left.text, TREE_CHANGED in left.text) == (
        "left",
        True,
        True,
    )
    assert records_of_type(repo, KeepRecord | DiscardRecord) == []


# ---------------------------------------------------------------------------
# a standing gating block
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_a_gating_block_stands_and_gate_passes_does_discard_it(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _gating_block_iteration(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    settled = run.report.steps[0]
    assert (settled.kind, "iteration 1" in settled.text, "gating regression" in settled.text) == (
        "settled",
        True,
        True,
    )
    assert status_of(experiment_worktree_dir(repo)) == ""


async def test_run_exit_sequence_when_a_gating_block_stands_and_the_gate_fails_does_leave_the_tree(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _gating_block_iteration(repo)
    _edit_again(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    left = run.report.steps[0]
    assert (left.kind, "gating regression" in left.text, TREE_CHANGED in left.text) == (
        "left",
        True,
        True,
    )
    assert status_of(experiment_worktree_dir(repo)) != ""


# ---------------------------------------------------------------------------
# hook records an earlier attempt at the same iteration left behind
# ---------------------------------------------------------------------------


#: What stands in the log ahead of a clean iteration whose seq a failed attempt already used:
#: the failed attempt's before hook, then what closed it or what the retry ran first.
EARLIER_ATTEMPTS = [
    pytest.param(
        (hook_record(seq=1, stage="before", exit_code=1), hook_record(seq=1, stage="before")),
        id="killed-outright-then-a-passing-hook",
    ),
    pytest.param(
        (hook_record(seq=1, stage="before", exit_code=1), command_record(seq=1)),
        id="closed-by-its-command-record-then-no-hook",
    ),
]


@pytest.mark.parametrize("earlier", EARLIER_ATTEMPTS)
async def test_run_exit_sequence_when_an_earlier_attempt_failed_its_before_hook_does_settle(
    repo: str, monkeypatch: pytest.MonkeyPatch, earlier: tuple[SessionLogRecord, ...]
):
    start_with(repo)
    # The first attempt recorded no iteration, so the retry reuses its seq.
    append_records(repo, *earlier)
    measured(repo, iteration_record(seq=1), hook_record(seq=1, stage="after"))
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert is_kept_step(run.report.steps[0])


# ---------------------------------------------------------------------------
# unmeasured edits nobody asked about
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_only_unmeasured_edits_stand_does_leave_them_counted(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    edit_experiment(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    left = run.report.steps[0]
    assert (left.kind, "2 unmeasured edits" in left.text) == ("left", True)
    assert records_of_type(repo, KeepRecord | DiscardRecord) == []
    assert status_of(experiment_worktree_dir(repo)) != ""
