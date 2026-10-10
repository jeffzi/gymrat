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
from gymrat.supervisor.exit_sequence import ExitStep
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
    KEPT_STEP,
    NOT_FINALIZED_STEP,
    fingerprint,
    improved_iteration,
    measured,
    run_sequence,
    session_context,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.exec import ExecOptions, ExecResult
    from gymrat.session.records import SessionLogRecord
    from gymrat.session.schema import Outcome

#: The gate reason an edit made behind the measured fingerprint reads as.
TREE_CHANGED = (
    "tree changed after measuring — an after hook, a later edit, or an interrupted discard"
)


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


def _failed_before_hook(root: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """An improved iteration whose before hook exited non-zero."""
    append_records(root, hook_record(seq=1, stage="before", exit_code=1))
    measured(root, iteration_record(seq=1))


def _failed_before_hook_on_a_retry(root: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """An improved iteration whose before hook failed, recorded after an attempt that raised."""
    append_records(
        root,
        hook_record(seq=1, stage="before"),
        command_record(seq=1),
        hook_record(seq=1, stage="before", exit_code=1),
    )
    measured(root, iteration_record(seq=1))


def _timed_out_after_hook(root: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """An improved iteration whose after hook ran past its timeout."""
    measured(root, iteration_record(seq=1), hook_record(seq=1, stage="after", timed_out=True))


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
    improved_iteration(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps == (KEPT_STEP, NOT_FINALIZED_STEP)
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert (record.status, record.seq, record.message) == (
        "committed",
        1,
        "supervised: iteration 1",
    )
    assert status_of(experiment_worktree_dir(repo)) == ""


async def test_run_exit_sequence_when_no_checks_are_configured_does_route_the_missing_checks_warning_to_the_sink(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    start_with(repo)
    improved_iteration(repo)
    install_exec(monkeypatch, KEEP_EXEC, UNUSED_EXEC)

    run = await run_sequence(session_context(repo))

    assert run.report.steps[0] == ExitStep(
        kind="settled", text="settled: kept iteration 1 (checks not configured)"
    )
    assert "no checks command is configured" in "\n".join(run.warnings)
    assert capsys.readouterr().err == ""


async def test_run_exit_sequence_when_the_checks_rewrite_the_tree_does_note_it_on_the_kept_step(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    _rewriting_checks(monkeypatch, repo)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="settled",
        text="settled: kept iteration 1 (checks passed); tree changed during checks",
    )


async def test_run_exit_sequence_when_the_kept_tree_has_no_fingerprint_does_add_no_note(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    checks_pass(monkeypatch)
    standing = worktree_fingerprint(Path(experiment_worktree_dir(repo)))

    def fingerprint_until_kept(directory: Path) -> str | None:
        # The standing edit fingerprints as usual; once the keep commits it, the
        # clean tree's read fails, so only the kept-tree note loses its input.
        return standing if status_of(str(directory)) else None

    monkeypatch.setattr(
        "gymrat.supervisor.exit_sequence.worktree_fingerprint",
        create_autospec(worktree_fingerprint, side_effect=fingerprint_until_kept),
    )

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0] == KEPT_STEP


async def test_run_exit_sequence_when_the_checks_fail_does_leave_the_iteration_for_a_person(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    checks_fail(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps == (
        ExitStep(
            kind="left",
            text=(
                "left unsettled: iteration 1 improved but the checks failed "
                "— fix and keep, or discard, by hand"
            ),
        ),
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

    assert run.report.steps[0] == ExitStep(
        kind="settled", text="settled: iteration 1 had nothing to commit"
    )
    command = last_command_record(repo)
    assert (command.exit_code, command.reason, command.seq) == (0, None, 1)


# ---------------------------------------------------------------------------
# the gate refusing an improved iteration
# ---------------------------------------------------------------------------


GATE_FAILURES = [
    pytest.param(_stale_tree, TREE_CHANGED, id="tree-changed-after-measuring"),
    pytest.param(_no_fingerprint, "fingerprint unavailable", id="fingerprint-unavailable"),
    pytest.param(_failed_before_hook, "before hook failed", id="before-hook-failed"),
    pytest.param(_timed_out_after_hook, "after hook failed", id="after-hook-timed-out"),
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
async def test_run_exit_sequence_when_improved_but_the_gate_fails_does_name_the_reason_and_the_diff(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[str, pytest.MonkeyPatch], None],
    reason: str,
):
    start_with(repo)
    arrange(repo, monkeypatch)
    recorder = checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="left",
        text=(
            f"left unsettled: iteration 1 improved but {reason} — keep or discard it by hand "
            f"(git -C {experiment_worktree_dir(repo)} diff first; "
            "keep commits the tree as it stands)"
        ),
    )
    assert records_of_type(repo, KeepRecord | DiscardRecord) == []
    assert recorder.calls == []
    assert status_of(experiment_worktree_dir(repo)) != ""


# ---------------------------------------------------------------------------
# discarding an unsettled iteration that did not improve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["no-signal", "regressed"])
async def test_run_exit_sequence_when_unsettled_and_not_improved_does_revert_it_without_checks(
    repo: str, monkeypatch: pytest.MonkeyPatch, outcome: Outcome
):
    start_with(repo)
    measured(repo, unimproved(1, outcome))
    recorder = checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps == (
        ExitStep(kind="settled", text=f"settled: discarded iteration 1 ({outcome})"),
    )
    assert isinstance(settling_record_of(repo), DiscardRecord)
    assert status_of(experiment_worktree_dir(repo)) == ""
    assert recorder.calls == []


async def test_run_exit_sequence_when_not_improved_and_the_gate_fails_does_revert_nothing(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    measured(repo, unimproved(1, "regressed"))
    _edit_again(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="left",
        text=(
            f"left unsettled: iteration 1 regressed but {TREE_CHANGED} — keep or discard it by hand"
        ),
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

    assert run.report.steps[0] == ExitStep(
        kind="settled", text="settled: discarded iteration 1 (gating regression)"
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

    assert run.report.steps[0] == ExitStep(
        kind="left",
        text=(
            f"left in worktree: iteration 1 blocked by a gating regression but {TREE_CHANGED} "
            "— discard by hand"
        ),
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

    assert run.report.steps[0] == KEPT_STEP


# ---------------------------------------------------------------------------
# unmeasured edits nobody asked about
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "counted"),
    [
        pytest.param(_edit_again, "1 unmeasured edit", id="one-edit"),
        pytest.param(edit_experiment, "2 unmeasured edits", id="two-edits"),
    ],
)
async def test_run_exit_sequence_when_only_unmeasured_edits_stand_does_leave_them_counted(
    repo: str, monkeypatch: pytest.MonkeyPatch, edit: Callable[[str], None], counted: str
):
    start_with(repo)
    edit(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="left", text=f"left in worktree: {counted} — measure or discard by hand"
    )
    assert records_of_type(repo, KeepRecord | DiscardRecord) == []
    assert status_of(experiment_worktree_dir(repo)) != ""
