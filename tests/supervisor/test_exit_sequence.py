"""Behavioral tests for ``run_exit_sequence`` — the frame and the settle decisions.

The lock wait and the skip it ends in, the guarantees the ``with_repo_lock`` seam
makes (a repaired log tail, a ``supervise`` command record, ``settling`` progress),
the already-finalized and nothing-to-settle terminals, the ``FollowUpEvent`` every
step lands on the wire, and the error boundary that keeps the sequence from raising.

Then the settle half: what the sequence keeps, what it discards, and what it refuses
to touch because the tree moved behind the measurement or a hook failed.

Every test drives the real sequence against a throwaway repository from the shared
``create_scratch_repo`` factory, so the suite is order-independent and safe under
``pytest-xdist`` / ``pytest-randomly``. Lock probes are injected as callables where
a test needs the lock to look held without one actually being taken; the tests that
exercise the real probe hold a real OS lock through ``hold_lock``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

from gymrat.config import BenchlessConfig
from gymrat.exec import ExecOptions, ExecResult
from gymrat.git import SHORT_SHA_LENGTH
from gymrat.session import (
    CommandRecord,
    DiscardRecord,
    FinalizeRecord,
    IterationRecord,
    KeepRecord,
    SessionLogRecord,
    SessionRecord,
    append_record,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from gymrat.session.paths import lockfile_path
from gymrat.session.workspace import worktree_fingerprint
from gymrat.supervisor.exit_sequence import (
    ExitPhase,
    ExitReport,
    ExitStep,
    run_exit_sequence,
)
from tests.conftest import hold_lock
from tests.loop.settle._fixtures import (
    CHECKS,
    UNUSED_EXEC,
    checks_fail,
    checks_pass,
    commit_experiment_directly,
    confirmed_regression,
    edit_experiment,
    gating_block,
    git,
    install_exec,
    iteration,
    settling_record_of,
    start_with,
    status_of,
    unimproved,
)
from tests.session.records._fixtures import (
    TORN_PREFIX,
    committed_keep,
    discard_record,
    finalize_record,
    hook_record,
    stop_record,
    tear_final_line,
)
from tests.supervisor._fixtures import collecting_observer, follow_up_events, make_context

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from filelock import FileLock

    from gymrat.session.schema import Outcome
    from gymrat.supervisor.context import SupervisedSession
    from gymrat.supervisor.events import SessionEvent, SessionObserver
    from gymrat.supervisor.supervise import EndedBy
    from tests.loop.settle._fixtures import ExecRecorder

#: The skip wording when the holder record cannot be read.
SKIP_UNKNOWN = "exit sequence skipped: gymrat is still running (PID unknown)"

#: The note the skip line closes on when a cap ended the run.
CAP_NOTE = " — expected after a cap trip: the agent's last command outlives the cap"

#: A complete log line no record schema matches, so ``read_records`` refuses the log.
UNREADABLE_LINE = '{"type": "banana"}\n'

#: The gate reason an edit made behind the measured fingerprint reads as.
TREE_CHANGED = (
    "tree changed after measuring — an after hook, a later edit, or an interrupted discard"
)

#: The step a run closes on when it would have finalized but ``--no-finalize`` was passed.
NOT_FINALIZED_STEP = ExitStep(kind="nothing", text="session left open (--no-finalize)")


class ExitRun(NamedTuple):
    """A finished sequence paired with the progress, events, and warnings it produced."""

    report: ExitReport
    phases: list[ExitPhase]
    events: list[SessionEvent]
    warnings: list[str]


class HeldProbe:
    """A lock probe that always reads the lock as held and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return True


def _benchless_config(timeout_seconds: int, checks: str | None) -> BenchlessConfig:
    """A minimal config whose ``timeout_seconds`` sets the default lock-wait bound."""
    return BenchlessConfig(
        adapter="mitata",
        samples=1,
        timeout_seconds=timeout_seconds,
        unstable_noise_pct=5.0,
        primary="geomean",
        checks=checks,
        runbook=None,
        stop=None,
    )


def _context(
    root: str, *, timeout_seconds: int = 60, checks: str | None = None
) -> SupervisedSession:
    """A supervised session rooted at ``root``, pointed at that repository's lock."""
    return make_context(
        root=root,
        lock_path=lockfile_path(root),
        log_path=str(Path(root) / ".gymrat" / "supervisor.jsonl"),
        config=_benchless_config(timeout_seconds, checks),
    )


async def run_sequence(
    context: SupervisedSession,
    *,
    ended_by: EndedBy = "session",
    finalize: bool = False,
    lock_poll_ms: int = 1,
    lock_wait_ms: int | None = 0,
    is_lock_held: Callable[[], bool] | None = None,
    log: SessionObserver | None = None,
    progress: Callable[[ExitPhase], None] | None = None,
) -> ExitRun:
    """Run the exit sequence over ``context``, collecting its progress, events, and warnings.

    Args:
        context: The supervised session the sequence closes out.
        ended_by: What ended the run, as ``run_exit_sequence`` takes it.
        finalize: Whether the sequence may finalize the session.
        lock_poll_ms: How often the lock probe is retried while waiting.
        lock_wait_ms: How long the lock wait is bound to, or None for the config bound.
        is_lock_held: Lock probe override, or None to use the real one.
        log: Observer override, or None to collect events into the returned run.
        progress: Progress sink override, or None to collect phases into the returned run.

    Returns:
        The report paired with the phases, events, and warnings the run produced.
        A sink override silences the matching list, which stays empty.
    """
    phases: list[ExitPhase] = []
    warnings: list[str] = []
    events, observer = collecting_observer()
    report = await run_exit_sequence(
        context,
        ended_by=ended_by,
        finalize=finalize,
        progress=phases.append if progress is None else progress,
        log=observer if log is None else log,
        warn=warnings.append,
        lock_poll_ms=lock_poll_ms,
        lock_wait_ms=lock_wait_ms,
        is_lock_held=is_lock_held,
    )
    return ExitRun(report, phases, events, warnings)


def _read_bytes(path: str) -> bytes:
    """Filesystem read kept out of the async body so it is not flagged as blocking I/O."""
    return Path(path).read_bytes()


def _read_text(path: str) -> str:
    """Filesystem read kept out of the async body so it is not flagged as blocking I/O."""
    return Path(path).read_text(encoding="utf-8")


def _write_text(path: str, text: str) -> None:
    """Filesystem write kept out of the async body so it is not flagged as blocking I/O."""
    Path(path).write_text(text, encoding="utf-8")


def _make_log_unreadable(root: str) -> str:
    """Append a line no record schema matches, returning the session log path."""
    jsonl_path = session_jsonl_path(root)
    with Path(jsonl_path).open("a", encoding="utf-8") as handle:
        handle.write(UNREADABLE_LINE)
    return jsonl_path


def _command_records(root: str) -> list[CommandRecord]:
    """Every command record the session log holds."""
    return [r for r in read_records(session_jsonl_path(root)) if isinstance(r, CommandRecord)]


def _settling_records(root: str) -> list[KeepRecord | DiscardRecord]:
    """Every keep and discard the session log holds."""
    records = read_records(session_jsonl_path(root))
    return [r for r in records if isinstance(r, KeepRecord | DiscardRecord)]


def _fingerprint(root: str) -> str:
    """The experiment worktree's fingerprint, as an iteration would have measured it."""
    experiment = experiment_worktree_dir(root)
    tree = worktree_fingerprint(Path(experiment))
    if tree is None:
        msg = f"expected a fingerprint for {experiment}"
        raise AssertionError(msg)
    return tree


def _append(root: str, *records: SessionLogRecord) -> None:
    """Append every record to the log of the session already open in ``root``."""
    jsonl_path = session_jsonl_path(root)
    for record in records:
        append_record(jsonl_path, record)


def _measured(root: str, record: IterationRecord, *trailing: SessionLogRecord) -> None:
    """Edit the experiment worktree, then log ``record`` fingerprinted to what it measured."""
    edit_experiment(root)
    _append(root, record.model_copy(update={"measured_tree": _fingerprint(root)}), *trailing)


def _edit_again(root: str) -> None:
    """Rewrite a tracked file, leaving the worktree off the measured fingerprint."""
    _write_text(str(Path(experiment_worktree_dir(root)) / "README.md"), "# edited again\n")


def _rewriting_checks(monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    """Answer the checks command with a pass that rewrote a tracked file, as a formatter would."""

    async def run(_command: str, _options: ExecOptions) -> ExecResult:
        _write_text(str(Path(experiment_worktree_dir(root)) / "README.md"), "# reformatted\n")
        return ExecResult(stdout="", stderr="", exit_code=0, stdout_bytes=0, stderr_bytes=0)

    monkeypatch.setattr("gymrat.loop.settle.checks.exec", run)


def _stale_tree(root: str) -> None:
    """An improved iteration whose worktree moved on behind the fingerprint."""
    _measured(root, iteration(1))
    _edit_again(root)


def _no_fingerprint(root: str) -> None:
    """An improved iteration that recorded no fingerprint at all."""
    edit_experiment(root)
    _append(root, iteration(1, measured_tree=None))


def _failed_before_hook(root: str) -> None:
    """An improved iteration whose before hook exited non-zero."""
    _measured(root, iteration(1), hook_record(seq=1, stage="before", exit_code=1))


def _timed_out_after_hook(root: str) -> None:
    """An improved iteration whose after hook ran past its timeout."""
    _measured(root, iteration(1), hook_record(seq=1, stage="after", timed_out=True))


def _improved_iteration(root: str) -> None:
    """An improved iteration the gate lets through, so the settle step keeps it."""
    _measured(root, iteration(1))


def _unimproved_iteration(root: str) -> None:
    """An iteration that read as no-signal, so the settle step discards it and keeps nothing."""
    _measured(root, unimproved(1, "no-signal"))


def _gating_block_iteration(root: str) -> None:
    """A confirmed regression standing under a gating block, ready for the gate to decide it."""
    _measured(root, confirmed_regression(1), gating_block(1))


def _keep_iteration(root: str, seq: int) -> None:
    """Commit the experiment worktree and log the iteration and the keep that settled it."""
    edit_experiment(root)
    commit = commit_experiment_directly(root)
    _append(root, iteration(seq), committed_keep(seq, commit=commit))


def _session_branch(root: str) -> str:
    """The branch the open session's header names."""
    header = read_records(session_jsonl_path(root))[0]
    assert isinstance(header, SessionRecord), f"expected a header in {session_jsonl_path(root)}"
    return header.branch


def _finalize_record(root: str) -> FinalizeRecord:
    """The record closing the session, failing when the log holds none."""
    for record in reversed(read_records(session_jsonl_path(root))):
        if isinstance(record, FinalizeRecord):
            return record
    msg = f"expected a finalize record in {session_jsonl_path(root)}"
    raise AssertionError(msg)


def _finalized_step(root: str) -> ExitStep:
    """The step the finalize record in ``root``'s log spells out."""
    record = _finalize_record(root)
    return ExitStep(
        kind="finalized",
        text=f"finalized: {record.branch} at {record.commit[:SHORT_SHA_LENGTH]}",
    )


@pytest.fixture
def hold_repo_lock() -> Iterator[Callable[[str], str]]:
    """Hold a repository's lock for the rest of the test, releasing it on teardown."""
    locks: list[FileLock] = []

    def hold(root: str) -> str:
        lock_path = lockfile_path(root)
        locks.append(hold_lock(lock_path))
        return lock_path

    yield hold
    for lock in locks:
        lock.release()


# ---------------------------------------------------------------------------
# lock wait and skip
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_lock_held_past_the_bound_does_skip_naming_the_holder_pid(
    repo: str, hold_repo_lock: Callable[[str], str]
):
    start_with(repo)
    hold_repo_lock(repo)

    run = await run_sequence(_context(repo), lock_wait_ms=5)

    assert run.report == ExitReport(
        steps=(
            ExitStep(
                kind="skipped",
                text=f"exit sequence skipped: gymrat is still running (PID {os.getpid()})",
            ),
        ),
        error=None,
    )
    assert _command_records(repo) == []


async def test_run_exit_sequence_when_lock_held_does_report_waiting_lock_progress_once(
    repo: str, hold_repo_lock: Callable[[str], str]
):
    start_with(repo)
    hold_repo_lock(repo)

    run = await run_sequence(_context(repo), lock_wait_ms=5)

    assert run.phases == [ExitPhase(kind="waiting-lock", pid=os.getpid())]


async def test_run_exit_sequence_when_wait_bound_is_zero_does_probe_once_and_still_report_waiting(
    repo: str,
):
    start_with(repo)
    probe = HeldProbe()

    run = await run_sequence(_context(repo), lock_wait_ms=0, is_lock_held=probe)

    assert probe.calls == 1
    assert run.phases == [ExitPhase(kind="waiting-lock", pid=None)]


async def test_run_exit_sequence_when_lock_stays_held_does_keep_probing_until_the_bound(
    repo: str,
):
    start_with(repo)
    probe = HeldProbe()

    run = await run_sequence(_context(repo), lock_poll_ms=1, lock_wait_ms=20, is_lock_held=probe)

    assert probe.calls > 1
    assert run.report.steps[0].kind == "skipped"


async def test_run_exit_sequence_when_lock_frees_mid_wait_does_proceed_to_settle(
    repo: str,
):
    start_with(repo)
    calls = 0

    def held_then_freed() -> bool:
        nonlocal calls
        calls += 1
        return calls <= 2

    run = await run_sequence(
        _context(repo), lock_poll_ms=1, lock_wait_ms=100, is_lock_held=held_then_freed
    )

    assert calls > 2
    assert ExitPhase(kind="waiting-lock", pid=None) in run.phases
    assert ExitPhase(kind="settling", pid=None) in run.phases
    assert run.report.steps[0].kind != "skipped"


async def test_run_exit_sequence_when_no_wait_bound_given_does_bound_by_the_config_timeout(
    repo: str,
):
    start_with(repo)
    context = _context(repo, timeout_seconds=1)
    started = time.monotonic()

    run = await run_sequence(context, lock_poll_ms=10, lock_wait_ms=None, is_lock_held=HeldProbe())

    assert time.monotonic() - started >= 0.9
    assert run.report.steps[0].kind == "skipped"


@pytest.mark.parametrize(
    ("ended_by", "note"),
    [
        ("wall-clock", CAP_NOTE),
        ("spend-cap", CAP_NOTE),
        ("session", ""),
        ("guard", ""),
    ],
)
async def test_run_exit_sequence_when_skipping_does_note_a_cap_trip_only_for_cap_ends(
    repo: str, ended_by: EndedBy, note: str
):
    start_with(repo)

    run = await run_sequence(_context(repo), ended_by=ended_by, is_lock_held=HeldProbe())

    assert run.report.steps == (ExitStep(kind="skipped", text=SKIP_UNKNOWN + note),)


async def test_run_exit_sequence_when_the_holder_record_is_unreadable_does_report_pid_unknown(
    repo: str, hold_repo_lock: Callable[[str], str]
):
    start_with(repo)
    lock_path = hold_repo_lock(repo)
    _write_text(lock_path, "not a holder record")

    run = await run_sequence(_context(repo))

    assert run.report.steps == (ExitStep(kind="skipped", text=SKIP_UNKNOWN),)


async def test_run_exit_sequence_when_the_lock_is_taken_after_the_probe_does_skip(
    repo: str, hold_repo_lock: Callable[[str], str]
):
    start_with(repo)
    hold_repo_lock(repo)

    run = await run_sequence(_context(repo), is_lock_held=lambda: False)

    assert run.report.steps == (
        ExitStep(
            kind="skipped",
            text=f"exit sequence skipped: gymrat is still running (PID {os.getpid()})",
        ),
    )
    assert run.phases == []


# ---------------------------------------------------------------------------
# under the lock
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_it_runs_does_append_a_supervise_record_for_the_exit_stage(
    repo: str,
):
    start_with(repo)

    await run_sequence(_context(repo))

    command = _command_records(repo)[-1]
    assert command.name == "supervise"
    assert command.args == {"stage": "exit"}
    assert command.exit_code == 0


async def test_run_exit_sequence_when_the_checks_block_the_keep_does_record_exit_one(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _improved_iteration(repo)
    checks_fail(monkeypatch)

    await run_sequence(_context(repo, checks=CHECKS))

    command = _command_records(repo)[-1]
    assert (command.exit_code, command.reason, command.seq) == (1, "checks-failed", 1)


async def test_run_exit_sequence_when_the_log_tail_is_torn_does_repair_it_before_folding(
    repo: str,
):
    start_with(repo)
    jsonl_path = session_jsonl_path(repo)
    tear_final_line(jsonl_path)

    run = await run_sequence(_context(repo))

    assert TORN_PREFIX not in _read_bytes(jsonl_path)
    assert run.report.error is None


async def test_run_exit_sequence_when_it_takes_the_lock_does_report_settling_progress_once(
    repo: str,
):
    start_with(repo)

    run = await run_sequence(_context(repo))

    assert run.phases == [ExitPhase(kind="settling", pid=None)]


# ---------------------------------------------------------------------------
# already finalized
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_session_already_finalized_does_report_nothing_left_to_do(
    repo: str,
):
    start_with(repo, (finalize_record(),))

    run = await run_sequence(_context(repo), finalize=True)

    assert run.report == ExitReport(
        steps=(ExitStep(kind="nothing", text="session already finalized"),), error=None
    )


# ---------------------------------------------------------------------------
# nothing to settle
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_the_session_measured_nothing_does_report_nothing_to_settle(
    repo: str,
):
    start_with(repo)

    run = await run_sequence(_context(repo))

    assert run.report == ExitReport(
        steps=(ExitStep(kind="nothing", text="nothing to settle"),), error=None
    )


async def test_run_exit_sequence_when_no_session_was_opened_does_report_nothing_to_settle(
    repo: str,
):
    run = await run_sequence(_context(repo))

    assert run.report == ExitReport(
        steps=(ExitStep(kind="nothing", text="nothing to settle"),), error=None
    )


async def test_run_exit_sequence_when_the_last_iteration_was_discarded_does_report_nothing_to_settle(
    repo: str,
):
    start_with(repo, (iteration(1), discard_record(1)))

    run = await run_sequence(_context(repo, checks=CHECKS))

    assert run.report == ExitReport(
        steps=(ExitStep(kind="nothing", text="nothing to settle"),), error=None
    )


# ---------------------------------------------------------------------------
# settle: keeping an unsettled improved iteration
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_unsettled_improved_and_gate_passes_does_commit_the_iteration(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _improved_iteration(repo)
    checks_pass(monkeypatch)

    await run_sequence(_context(repo, checks=CHECKS))

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
    _improved_iteration(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="settled", text="settled: kept iteration 1 (checks passed)"
    )


async def test_run_exit_sequence_when_no_checks_are_configured_does_say_so_on_the_kept_step(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _improved_iteration(repo)
    install_exec(monkeypatch, UNUSED_EXEC)

    run = await run_sequence(_context(repo))

    assert run.report.steps[0] == ExitStep(
        kind="settled", text="settled: kept iteration 1 (checks not configured)"
    )


async def test_run_exit_sequence_when_no_checks_are_configured_does_route_the_hint_to_the_warn_sink(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    start_with(repo)
    _improved_iteration(repo)
    install_exec(monkeypatch, UNUSED_EXEC)

    run = await run_sequence(_context(repo))

    assert "no checks command is configured" in "\n".join(run.warnings)
    assert capsys.readouterr().err == ""


async def test_run_exit_sequence_when_the_checks_rewrite_the_tree_does_note_it_on_the_kept_step(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _improved_iteration(repo)
    _rewriting_checks(monkeypatch, repo)

    run = await run_sequence(_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="settled",
        text="settled: kept iteration 1 (checks passed); tree changed during checks",
    )


async def test_run_exit_sequence_when_the_checks_fail_does_leave_the_iteration_for_a_person(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _improved_iteration(repo)
    checks_fail(monkeypatch)

    run = await run_sequence(_context(repo, checks=CHECKS))

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
    _append(repo, iteration(1, measured_tree=_fingerprint(repo)))
    checks_pass(monkeypatch)

    run = await run_sequence(_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="settled", text="settled: iteration 1 had nothing to commit"
    )


async def test_run_exit_sequence_when_the_agent_committed_nothing_does_record_exit_zero(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _append(repo, iteration(1, measured_tree=_fingerprint(repo)))
    checks_pass(monkeypatch)

    await run_sequence(_context(repo, checks=CHECKS))

    command = _command_records(repo)[-1]
    assert (command.exit_code, command.reason, command.seq) == (0, None, 1)


# ---------------------------------------------------------------------------
# settle: the gate refusing an improved iteration
# ---------------------------------------------------------------------------


GATE_FAILURES = [
    pytest.param(_stale_tree, TREE_CHANGED, id="tree-changed-after-measuring"),
    pytest.param(_no_fingerprint, "fingerprint unavailable", id="fingerprint-unavailable"),
    pytest.param(_failed_before_hook, "before hook failed", id="before-hook-failed"),
    pytest.param(_timed_out_after_hook, "after hook failed", id="after-hook-timed-out"),
]


@pytest.mark.parametrize(("arrange", "reason"), GATE_FAILURES)
async def test_run_exit_sequence_when_improved_but_the_gate_fails_does_name_the_reason_and_the_diff(
    repo: str, monkeypatch: pytest.MonkeyPatch, arrange: Callable[[str], None], reason: str
):
    start_with(repo)
    arrange(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="left",
        text=(
            f"left unsettled: iteration 1 improved but {reason} — keep or discard it by hand "
            f"(git -C {experiment_worktree_dir(repo)} diff first; "
            "keep commits the tree as it stands)"
        ),
    )


async def test_run_exit_sequence_when_improved_but_the_gate_fails_does_neither_keep_nor_discard(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _stale_tree(repo)
    recorder = checks_pass(monkeypatch)

    await run_sequence(_context(repo, checks=CHECKS))

    assert _settling_records(repo) == []
    assert recorder.calls == []
    assert status_of(experiment_worktree_dir(repo)) != ""


# ---------------------------------------------------------------------------
# settle: discarding an unsettled iteration that did not improve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["no-signal", "regressed"])
async def test_run_exit_sequence_when_unsettled_and_not_improved_does_report_the_discard(
    repo: str, monkeypatch: pytest.MonkeyPatch, outcome: Outcome
):
    start_with(repo)
    _measured(repo, unimproved(1, outcome))
    checks_pass(monkeypatch)

    run = await run_sequence(_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="settled", text=f"settled: discarded iteration 1 ({outcome})"
    )


async def test_run_exit_sequence_when_unsettled_and_not_improved_does_revert_without_committing(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _unimproved_iteration(repo)
    recorder = checks_pass(monkeypatch)

    await run_sequence(_context(repo, checks=CHECKS))

    assert isinstance(settling_record_of(repo), DiscardRecord)
    assert status_of(experiment_worktree_dir(repo)) == ""
    assert recorder.calls == []


async def test_run_exit_sequence_when_not_improved_and_the_gate_fails_does_revert_nothing(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _measured(repo, unimproved(1, "regressed"))
    _edit_again(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="left",
        text=(
            f"left unsettled: iteration 1 regressed but {TREE_CHANGED} — keep or discard it by hand"
        ),
    )
    assert _settling_records(repo) == []


# ---------------------------------------------------------------------------
# settle: a standing gating block
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_a_gating_block_stands_and_gate_passes_does_discard_it(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _gating_block_iteration(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(_context(repo, checks=CHECKS))

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

    run = await run_sequence(_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="left",
        text=(
            f"left in worktree: iteration 1 blocked by a gating regression but {TREE_CHANGED} "
            "— discard by hand"
        ),
    )
    assert status_of(experiment_worktree_dir(repo)) != ""


# ---------------------------------------------------------------------------
# settle: unmeasured edits nobody asked about
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_only_unmeasured_edits_stand_does_count_them_and_leave_them(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    edit_experiment(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(_context(repo, checks=CHECKS))

    assert run.report.steps[0] == ExitStep(
        kind="left", text="left in worktree: 2 unmeasured edit(s) — measure or discard by hand"
    )
    assert _settling_records(repo) == []
    assert status_of(experiment_worktree_dir(repo)) != ""


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_nothing_left_for_a_person_does_finalize_the_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _improved_iteration(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(_context(repo, checks=CHECKS), finalize=True)

    assert run.report.steps[-1] == _finalized_step(repo)
    assert run.report.error is None


async def test_run_exit_sequence_when_the_log_ends_on_a_stop_record_does_finalize_anyway(
    repo: str,
):
    start_with(repo)
    _keep_iteration(repo, 1)
    _append(repo, stop_record())

    run = await run_sequence(_context(repo), finalize=True)

    assert run.report.steps == (_finalized_step(repo),)


async def test_run_exit_sequence_when_finalize_refuses_does_record_the_refusal_not_an_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _improved_iteration(repo)
    checks_pass(monkeypatch)
    taken = f"{_session_branch(repo)}-final"
    git(["branch", taken], repo)

    run = await run_sequence(_context(repo, checks=CHECKS), finalize=True)

    assert run.report.steps[-1] == ExitStep(
        kind="refused", text=f"Finalize refused: the branch '{taken}' already exists."
    )
    assert run.report.error is None


@pytest.mark.parametrize(
    ("arrange", "install_checks", "trailing"),
    [
        pytest.param(_improved_iteration, checks_pass, (NOT_FINALIZED_STEP,), id="kept"),
        pytest.param(_improved_iteration, checks_fail, (), id="left-for-a-person"),
        pytest.param(_unimproved_iteration, checks_pass, (), id="nothing-kept"),
    ],
)
async def test_run_exit_sequence_when_finalize_is_off_does_note_it_only_when_it_would_have_run(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[str], None],
    install_checks: Callable[[pytest.MonkeyPatch], ExecRecorder],
    trailing: tuple[ExitStep, ...],
):
    start_with(repo)
    arrange(repo)
    install_checks(monkeypatch)

    run = await run_sequence(_context(repo, checks=CHECKS), finalize=False)

    assert run.report.steps[1:] == trailing


async def test_run_exit_sequence_when_finalize_is_off_and_all_settled_does_record_one_step(
    repo: str,
):
    start_with(repo)
    _keep_iteration(repo, 1)

    run = await run_sequence(_context(repo), finalize=False)

    assert run.report.steps == (NOT_FINALIZED_STEP,)


# ---------------------------------------------------------------------------
# on the wire
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_a_step_is_decided_does_emit_it_as_a_follow_up_event(
    repo: str,
):
    start_with(repo)

    run = await run_sequence(_context(repo))

    emitted = follow_up_events(run.events)
    assert [(event.action, event.reason) for event in emitted] == [("ended", "nothing to settle")]
    assert emitted[0].at > 0


async def test_run_exit_sequence_when_the_lock_is_held_does_emit_the_skip_as_a_follow_up_event(
    repo: str,
):
    start_with(repo)

    run = await run_sequence(_context(repo), is_lock_held=HeldProbe())

    assert [(event.action, event.reason) for event in follow_up_events(run.events)] == [
        ("ended", SKIP_UNKNOWN)
    ]


# ---------------------------------------------------------------------------
# the error boundary
# ---------------------------------------------------------------------------

#: What a dead sink raises, so the report it lands in can be checked for it.
BOOM = "the sink is down"

#: The prefix the closing event's reason carries when the sequence failed.
FAILED_PREFIX = "exit sequence failed: "


def _raising_observer(_event: SessionEvent) -> None:
    """An observer that fails on every event it is handed."""
    raise RuntimeError(BOOM)


def _raising_progress(_phase: ExitPhase) -> None:
    """A progress sink that fails on every phase it is handed."""
    raise RuntimeError(BOOM)


class FailsOnNthEvent:
    """An observer that collects every event it is handed except the ``nth``, which it fails on."""

    def __init__(self, nth: int) -> None:
        self.events: list[SessionEvent] = []
        self.calls = 0
        self._nth = nth

    def __call__(self, event: SessionEvent) -> None:
        self.calls += 1
        if self.calls == self._nth:
            raise RuntimeError(BOOM)
        self.events.append(event)


async def test_run_exit_sequence_when_a_step_raises_does_report_the_error_without_raising(
    repo: str,
):
    start_with(repo)
    jsonl_path = _make_log_unreadable(repo)

    run = await run_sequence(_context(repo))

    assert run.report.steps == ()
    assert run.report.error is not None
    assert jsonl_path in run.report.error


async def test_run_exit_sequence_when_a_step_raises_does_record_exit_two_on_the_command_record(
    repo: str,
):
    start_with(repo)
    _make_log_unreadable(repo)

    await run_sequence(_context(repo))

    last_line = _read_text(session_jsonl_path(repo)).splitlines()[-1]
    recorded = json.loads(last_line)
    assert recorded["name"] == "supervise"
    assert recorded["exit_code"] == 2


@pytest.mark.parametrize(
    ("log", "progress"),
    [
        pytest.param(_raising_observer, None, id="skip-event"),
        pytest.param(None, _raising_progress, id="waiting-progress"),
    ],
)
async def test_run_exit_sequence_when_a_sink_raises_during_the_wait_does_report_the_error(
    repo: str,
    log: SessionObserver | None,
    progress: Callable[[ExitPhase], None] | None,
):
    start_with(repo)

    run = await run_sequence(_context(repo), is_lock_held=HeldProbe(), log=log, progress=progress)

    assert run.report.error is not None
    assert BOOM in run.report.error


async def test_run_exit_sequence_when_the_skip_event_raises_after_contention_does_report_the_error(
    repo: str, hold_repo_lock: Callable[[str], str]
):
    start_with(repo)
    hold_repo_lock(repo)

    run = await run_sequence(_context(repo), is_lock_held=lambda: False, log=_raising_observer)

    assert run.report.error is not None
    assert BOOM in run.report.error


async def test_run_exit_sequence_when_a_step_raises_does_emit_one_closing_follow_up_event(
    repo: str,
):
    start_with(repo)
    _make_log_unreadable(repo)

    run = await run_sequence(_context(repo))

    emitted = follow_up_events(run.events)
    assert run.report.error is not None
    assert [(event.action, event.reason) for event in emitted] == [
        ("ended", FAILED_PREFIX + run.report.error)
    ]


async def test_run_exit_sequence_when_a_later_step_raises_does_close_after_the_steps_it_took(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    _improved_iteration(repo)
    checks_pass(monkeypatch)
    kept = ExitStep(kind="settled", text="settled: kept iteration 1 (checks passed)")
    observer = FailsOnNthEvent(2)

    run = await run_sequence(_context(repo, checks=CHECKS), finalize=False, log=observer)

    assert run.report.error is not None
    assert BOOM in run.report.error
    assert run.report.steps == (kept, NOT_FINALIZED_STEP)
    assert [(event.action, event.reason) for event in follow_up_events(observer.events)] == [
        ("ended", kept.text),
        ("ended", FAILED_PREFIX + run.report.error),
    ]


async def test_run_exit_sequence_when_the_closing_event_raises_does_still_report_the_step_error(
    repo: str,
):
    start_with(repo)
    jsonl_path = _make_log_unreadable(repo)

    run = await run_sequence(_context(repo), log=_raising_observer)

    assert run.report.error is not None
    assert jsonl_path in run.report.error
