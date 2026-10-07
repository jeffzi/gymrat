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

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import create_autospec

import pytest

from gymrat.session.paths import experiment_worktree_dir
from gymrat.session.records import DiscardRecord, KeepChecks, KeepRecord
from gymrat.session.workspace import worktree_fingerprint
from gymrat.supervisor.exit_sequence import ExitStep
from tests._exec_fixtures import expected_result
from tests.loop._settle import (
    CHECKS,
    UNUSED_EXEC,
    checks_fail,
    checks_pass,
    confirmed_regression,
    edit_experiment,
    install_exec,
    settling_record_of,
    start_with,
    status_of,
    unimproved,
)
from tests.session.records._fixtures import (
    append_records,
    blocked_keep,
    command_record,
    hook_record,
    iteration_record,
    log_records,
)
from tests.supervisor._exit_sequence import (
    fingerprint,
    improved_iteration,
    measured,
    run_sequence,
    session_context,
    unimproved_iteration,
    write_text,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.exec import ExecOptions, ExecResult
    from gymrat.session.records import IterationRecord, SessionLogRecord
    from gymrat.session.schema import Outcome

#: The gate reason an edit made behind the measured fingerprint reads as.
TREE_CHANGED = (
    "tree changed after measuring — an after hook, a later edit, or an interrupted discard"
)


def _settling_records(root: str) -> list[KeepRecord | DiscardRecord]:
    """Every keep and discard the session log holds."""
    records = log_records(root)
    return [r for r in records if isinstance(r, KeepRecord | DiscardRecord)]


def _edit_again(root: str) -> None:
    """Rewrite a tracked file, leaving the worktree off the measured fingerprint."""
    write_text(str(Path(experiment_worktree_dir(root)) / "README.md"), "# edited again\n")


def _rewriting_checks(monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    """Answer the checks command with a pass that rewrote a tracked file, as a formatter would."""

    async def run(_command: str, _options: ExecOptions) -> ExecResult:
        write_text(str(Path(experiment_worktree_dir(root)) / "README.md"), "# reformatted\n")
        return expected_result()

    monkeypatch.setattr("gymrat.loop.keep.exec", run)


def _stale_tree(root: str) -> None:
    """An improved iteration whose worktree moved on behind the fingerprint."""
    measured(root, iteration_record(seq=1))
    _edit_again(root)


def _no_fingerprint(root: str) -> None:
    """An improved iteration that recorded no fingerprint at all."""
    edit_experiment(root)
    append_records(root, iteration_record(seq=1, measured_tree=None))


def _failed_before_hook(root: str) -> None:
    """An improved iteration whose before hook exited non-zero."""
    append_records(root, hook_record(seq=1, stage="before", exit_code=1))
    measured(root, iteration_record(seq=1))


def _failed_before_hook_on_a_retry(root: str) -> None:
    """An improved iteration whose before hook failed, recorded after an attempt that raised."""
    append_records(
        root,
        hook_record(seq=1, stage="before"),
        command_record(seq=1),
        hook_record(seq=1, stage="before", exit_code=1),
    )
    measured(root, iteration_record(seq=1))


def _timed_out_after_hook(root: str) -> None:
    """An improved iteration whose after hook ran past its timeout."""
    measured(root, iteration_record(seq=1), hook_record(seq=1, stage="after", timed_out=True))


def _failed_after_hook_ahead_of_a_command_record(root: str) -> None:
    """An improved iteration whose after hook failed, with a command record written behind it."""
    measured(
        root,
        iteration_record(seq=1),
        hook_record(seq=1, stage="after", exit_code=1),
        command_record(seq=1),
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

    await run_sequence(session_context(repo, checks=CHECKS))

    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert (record.status, record.seq, record.message) == (
        "committed",
        1,
        "supervised: iteration 1",
    )
    assert status_of(experiment_worktree_dir(repo)) == ""


async def test_run_exit_sequence_when_the_checks_pass_does_report_the_kept_iteration_as_checked(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="settled", text="settled: kept iteration 1 (checks passed)"
    )


async def test_run_exit_sequence_when_no_checks_are_configured_does_say_so_on_the_kept_step(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    install_exec(monkeypatch, UNUSED_EXEC)

    run = await run_sequence(session_context(repo))

    assert run.report.steps[0] == ExitStep(
        kind="settled", text="settled: kept iteration 1 (checks not configured)"
    )


async def test_run_exit_sequence_when_no_checks_are_configured_does_route_the_hint_to_the_warn_sink(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    start_with(repo)
    improved_iteration(repo)
    install_exec(monkeypatch, UNUSED_EXEC)

    run = await run_sequence(session_context(repo))

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
    # The gate reads the standing tree first; only the read after the keep fails.
    monkeypatch.setattr(
        "gymrat.supervisor.exit_sequence.worktree_fingerprint",
        create_autospec(worktree_fingerprint, side_effect=[standing, None]),
    )

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="settled", text="settled: kept iteration 1 (checks passed)"
    )


async def test_run_exit_sequence_when_the_checks_fail_does_leave_the_iteration_for_a_person(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    checks_fail(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="left",
        text=(
            "left unsettled: iteration 1 improved but the checks failed "
            "— fix and keep, or discard, by hand"
        ),
    )
    assert status_of(experiment_worktree_dir(repo)) != ""


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
]


@pytest.mark.parametrize(("arrange", "reason"), GATE_FAILURES)
async def test_run_exit_sequence_when_improved_but_the_gate_fails_does_name_the_reason_and_the_diff(
    repo: str, monkeypatch: pytest.MonkeyPatch, arrange: Callable[[str], None], reason: str
):
    start_with(repo)
    arrange(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="left",
        text=(
            f"left unsettled: iteration 1 improved but {reason} — keep or discard it by hand "
            f"(git -C {experiment_worktree_dir(repo)} diff first; "
            "keep commits the tree as it stands)"
        ),
    )


async def test_run_exit_sequence_when_the_standing_tree_has_no_fingerprint_does_leave_it_unsettled(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    checks_pass(monkeypatch)

    def no_fingerprint(_worktree: Path) -> None:
        return None

    monkeypatch.setattr("gymrat.supervisor.exit_sequence.worktree_fingerprint", no_fingerprint)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    step = run.report.steps[0]
    assert step.kind == "left"
    assert "improved but fingerprint unavailable" in step.text


async def test_run_exit_sequence_when_improved_but_the_gate_fails_does_neither_keep_nor_discard(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _stale_tree(repo)
    recorder = checks_pass(monkeypatch)

    await run_sequence(session_context(repo, checks=CHECKS))

    assert _settling_records(repo) == []
    assert recorder.calls == []
    assert status_of(experiment_worktree_dir(repo)) != ""


# ---------------------------------------------------------------------------
# discarding an unsettled iteration that did not improve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["no-signal", "regressed"])
async def test_run_exit_sequence_when_unsettled_and_not_improved_does_report_the_discard(
    repo: str, monkeypatch: pytest.MonkeyPatch, outcome: Outcome
):
    start_with(repo)
    measured(repo, unimproved(1, outcome))
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="settled", text=f"settled: discarded iteration 1 ({outcome})"
    )


async def test_run_exit_sequence_when_unsettled_and_not_improved_does_revert_without_committing(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    unimproved_iteration(repo)
    recorder = checks_pass(monkeypatch)

    await run_sequence(session_context(repo, checks=CHECKS))

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
    assert _settling_records(repo) == []


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
@pytest.mark.parametrize(
    ("iteration", "settled"),
    [
        pytest.param(iteration_record(seq=1), "kept iteration 1 (checks passed)", id="keep"),
        pytest.param(unimproved(1, "no-signal"), "discarded iteration 1 (no-signal)", id="discard"),
    ],
)
async def test_run_exit_sequence_when_an_earlier_attempt_failed_its_before_hook_does_settle(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    earlier: tuple[SessionLogRecord, ...],
    iteration: IterationRecord,
    settled: str,
):
    start_with(repo)
    # The first attempt recorded no iteration, so the retry reuses its seq.
    append_records(repo, *earlier)
    measured(repo, iteration, hook_record(seq=1, stage="after"))
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(kind="settled", text=f"settled: {settled}")


async def test_run_exit_sequence_when_only_the_standing_attempt_failed_its_before_hook_does_leave_it(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    # The first attempt's hook passed; the retry that recorded the iteration is the one that failed.
    append_records(
        repo,
        hook_record(seq=1, stage="before"),
        hook_record(seq=1, stage="before", exit_code=1),
    )
    measured(repo, iteration_record(seq=1))
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    step = run.report.steps[0]
    assert step.kind == "left"
    assert "iteration 1 improved but before hook failed" in step.text


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
async def test_run_exit_sequence_when_only_unmeasured_edits_stand_does_count_them_and_leave_them(
    repo: str, monkeypatch: pytest.MonkeyPatch, edit: Callable[[str], None], counted: str
):
    start_with(repo)
    edit(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="left", text=f"left in worktree: {counted} — measure or discard by hand"
    )
    assert _settling_records(repo) == []
    assert status_of(experiment_worktree_dir(repo)) != ""
