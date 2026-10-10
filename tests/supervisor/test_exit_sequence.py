"""Behavioral tests for ``run_exit_sequence`` — the frame around the settle decisions.

The lock wait and the skip it ends in, the guarantees the ``with_repo_lock`` seam
makes (a repaired log tail, a ``supervise`` command record, ``settling`` progress),
the already-finalized and nothing-to-settle terminals, finalize, the ``FollowUpEvent``
every step lands on the wire, and the error boundary that keeps the sequence from
raising.

The settle decisions themselves (what the sequence keeps, discards, or refuses to
touch) are pinned in ``test_exit_sequence_settle.py``.

Every test drives the real sequence against a throwaway repository from the shared
``create_scratch_repo`` factory, so the suite is order-independent and safe under
``pytest-xdist`` / ``pytest-randomly``. Lock probes are injected as callables where
a test needs the lock to look held without one actually being taken; the tests that
exercise the real probe hold a real OS lock through ``hold_lock``.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, override
from unittest.mock import create_autospec

import pytest

from gymrat.clock import monotonic_ms, now_ns
from gymrat.session.paths import lockfile_path, session_jsonl_path
from gymrat.session.records import (
    CommandRecord,
    FinalizeRecord,
)
from gymrat.supervisor.events import FollowUpEvent
from gymrat.supervisor.exit_sequence import (
    ExitPhase,
    ExitReport,
    ExitStep,
)
from gymrat.utils import SHORT_SHA_LENGTH
from tests._git import run_git
from tests._imports import loaded_under, modules_imported_by
from tests._lock import hold_lock
from tests.loop._settle import (
    CHECKS,
    checks_pass,
    commit_experiment_directly,
    edit_experiment,
    keep_iteration,
    start_with,
)
from tests.session.records._fixtures import (
    TORN_PREFIX,
    append_records,
    discard_record,
    finalize_record,
    iteration_record,
    log_records,
    records_of_type,
    session_header_of,
    stop_record,
    tear_final_line,
)
from tests.supervisor._exit_sequence import (
    KEPT_STEP,
    NOT_FINALIZED_STEP,
    improved_iteration,
    run_sequence,
    session_context,
)
from tests.supervisor._fixtures import events_of, raising_observer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from filelock import FileLock

    from gymrat.supervisor.events import SessionEvent, SessionObserver
    from gymrat.supervisor.supervise import EndedBy

#: The skip wording when the holder record cannot be read.
SKIP_UNKNOWN = "exit sequence skipped: gymrat is still running (PID unknown)"

#: The note the skip line closes on when a cap ended the run.
CAP_NOTE = " — expected after a cap trip: the agent's last command outlives the cap"

#: A complete log line no record schema matches, so ``read_records`` refuses the log.
UNREADABLE_LINE = '{"type": "banana"}\n'


class HeldProbe:
    """A lock probe that always reads the lock as held and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return True


class SlowHeldProbe(HeldProbe):
    """A held-lock probe whose every call advances a fake monotonic clock by 100 ms.

    Building one patches the exit sequence's clock to read that fake clock.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__()
        self.now_ms = 0.0
        monkeypatch.setattr(
            "gymrat.supervisor.exit_sequence.monotonic_ms",
            create_autospec(monotonic_ms, side_effect=lambda: self.now_ms),
        )

    @override
    def __call__(self) -> bool:
        self.now_ms += 100
        return super().__call__()


def _make_log_unreadable(root: str) -> str:
    """Append a line no record schema matches, returning the session log path."""
    jsonl_path = session_jsonl_path(root)
    with Path(jsonl_path).open("a", encoding="utf-8") as handle:
        handle.write(UNREADABLE_LINE)
    return jsonl_path


def _keep_iteration(root: str, seq: int) -> None:
    """Commit the experiment worktree and log the iteration and the keep that settled it."""
    edit_experiment(root)
    commit = commit_experiment_directly(root)
    keep_iteration(root, seq, commit=commit)


def _finalize_record(root: str) -> FinalizeRecord:
    """The record closing the session, failing when the log holds none."""
    for record in reversed(log_records(root)):
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

    run = await run_sequence(session_context(repo), lock_wait_ms=5)

    assert run.report == ExitReport(
        steps=(
            ExitStep(
                kind="skipped",
                text=f"exit sequence skipped: gymrat is still running (PID {os.getpid()})",
            ),
        ),
        error=None,
    )
    assert run.phases == [ExitPhase(kind="waiting-lock", pid=os.getpid())]
    assert records_of_type(repo, CommandRecord) == []


async def test_run_exit_sequence_when_lock_frees_mid_wait_does_proceed_to_settle(
    repo: str,
):
    start_with(repo)
    calls = 0

    def held_then_freed() -> bool:
        nonlocal calls
        calls += 1
        return calls <= 2

    # The probe frees the lock on its third call, so the bound never elapses; it
    # only has to outlast a loaded runner's delay before the first poll.
    run = await run_sequence(
        session_context(repo), lock_poll_ms=1, lock_wait_ms=60_000, is_lock_held=held_then_freed
    )

    assert calls > 2
    assert ExitPhase(kind="waiting-lock", pid=None) in run.phases
    assert ExitPhase(kind="settling", pid=None) in run.phases
    assert run.report.steps[0].kind != "skipped"


async def test_run_exit_sequence_when_no_wait_bound_given_does_bound_by_the_config_timeout(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    probe = SlowHeldProbe(monkeypatch)

    run = await run_sequence(
        session_context(repo, timeout_seconds=1),
        lock_poll_ms=1,
        lock_wait_ms=None,
        is_lock_held=probe,
    )

    assert (probe.calls, run.report.steps[0].kind) == (10, "skipped")


async def test_run_exit_sequence_when_the_first_probe_outlasts_the_bound_does_not_probe_again(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    probe = SlowHeldProbe(monkeypatch)

    await run_sequence(session_context(repo), lock_wait_ms=50, is_lock_held=probe)

    assert probe.calls == 1


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

    run = await run_sequence(
        session_context(repo), ended_by=ended_by, lock_wait_ms=0, is_lock_held=lambda: True
    )

    assert run.report.steps == (ExitStep(kind="skipped", text=SKIP_UNKNOWN + note),)
    assert run.phases == [ExitPhase(kind="waiting-lock", pid=None)]
    assert [(event.action, event.reason) for event in events_of(run.events, FollowUpEvent)] == [
        ("ended", SKIP_UNKNOWN + note)
    ]


async def test_run_exit_sequence_when_the_holder_record_is_unreadable_does_report_pid_unknown(
    repo: str, hold_repo_lock: Callable[[str], str]
):
    start_with(repo)
    lock_path = hold_repo_lock(repo)
    await asyncio.to_thread(Path(lock_path).write_text, "not a holder record", encoding="utf-8")

    run = await run_sequence(session_context(repo))

    assert run.report.steps == (ExitStep(kind="skipped", text=SKIP_UNKNOWN),)


async def test_run_exit_sequence_when_the_lock_is_taken_after_the_probe_does_skip(
    repo: str, hold_repo_lock: Callable[[str], str]
):
    start_with(repo)
    hold_repo_lock(repo)

    run = await run_sequence(session_context(repo), is_lock_held=lambda: False)

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


async def test_run_exit_sequence_when_the_log_tail_is_torn_does_repair_it_before_folding(
    repo: str,
):
    start_with(repo)
    jsonl_path = session_jsonl_path(repo)
    tear_final_line(jsonl_path)

    run = await run_sequence(session_context(repo))

    assert TORN_PREFIX not in await asyncio.to_thread(Path(jsonl_path).read_bytes)
    assert run.report.error is None


# ---------------------------------------------------------------------------
# already finalized
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_session_already_finalized_does_report_nothing_left_to_do(
    repo: str,
):
    start_with(repo, (finalize_record(),))

    run = await run_sequence(session_context(repo), finalize=True)

    assert run.report == ExitReport(
        steps=(ExitStep(kind="nothing", text="session already finalized"),), error=None
    )


# ---------------------------------------------------------------------------
# nothing to settle
# ---------------------------------------------------------------------------


def _measured_nothing(repo: str) -> None:
    start_with(repo)


def _opened_no_session(repo: str) -> None:
    del repo


def _discarded_the_last_iteration(repo: str) -> None:
    start_with(repo, (iteration_record(seq=1), discard_record(1)))


#: The command record the exit stage leaves in a session's log.
_EXIT_STAGE_COMMAND = ("supervise", {"stage": "exit"}, 0)


@pytest.mark.parametrize(
    ("arrange", "commands"),
    [
        pytest.param(_measured_nothing, [_EXIT_STAGE_COMMAND], id="session-measured-nothing"),
        pytest.param(_opened_no_session, [], id="no-session-opened"),
        pytest.param(
            _discarded_the_last_iteration, [_EXIT_STAGE_COMMAND], id="last-iteration-discarded"
        ),
    ],
)
async def test_run_exit_sequence_when_nothing_is_unsettled_does_report_it_under_the_lock(
    repo: str, arrange: Callable[[str], None], commands: list[tuple[str, dict[str, str], int]]
):
    arrange(repo)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    assert run.report == ExitReport(
        steps=(ExitStep(kind="nothing", text="nothing to settle"),), error=None
    )
    assert run.phases == [ExitPhase(kind="settling", pid=None)]
    emitted = events_of(run.events, FollowUpEvent)
    assert [(event.action, event.reason) for event in emitted] == [("ended", "nothing to settle")]
    assert emitted[0].at > 0
    assert [
        (command.name, command.args, command.exit_code)
        for command in records_of_type(repo, CommandRecord)
    ] == commands


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_nothing_left_for_a_person_does_finalize_the_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    checks_pass(monkeypatch)

    run = await run_sequence(session_context(repo, checks=CHECKS), finalize=True)

    assert run.report.steps[-1] == _finalized_step(repo)
    assert run.report.error is None


async def test_run_exit_sequence_when_the_log_ends_on_a_stop_record_does_finalize_anyway(
    repo: str,
):
    start_with(repo)
    _keep_iteration(repo, 1)
    append_records(repo, stop_record())

    run = await run_sequence(session_context(repo), finalize=True)

    assert run.report.steps == (_finalized_step(repo),)


async def test_run_exit_sequence_when_finalize_refuses_does_record_the_refusal_not_an_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    checks_pass(monkeypatch)
    taken = f"{session_header_of(repo).branch}-final"
    run_git(["branch", taken], repo)

    run = await run_sequence(session_context(repo, checks=CHECKS), finalize=True)

    assert run.report.steps[-1] == ExitStep(
        kind="refused", text=f"Finalize refused: the branch '{taken}' already exists."
    )
    assert run.report.error is None


async def test_run_exit_sequence_when_finalize_is_off_and_all_settled_does_leave_the_session_open(
    repo: str,
):
    start_with(repo)
    _keep_iteration(repo, 1)

    run = await run_sequence(session_context(repo), finalize=False)

    assert run.report.steps == (NOT_FINALIZED_STEP,)


# ---------------------------------------------------------------------------
# the error boundary
# ---------------------------------------------------------------------------

#: What a dead sink raises, so the report it lands in can be checked for it.
BOOM = "the sink is down"

#: The prefix the closing event's reason carries when the sequence failed.
FAILED_PREFIX = "exit sequence failed: "


_raising_observer = raising_observer(BOOM)


def _raising_progress(_phase: ExitPhase) -> None:
    """A progress sink that fails on every phase it is handed."""
    raise RuntimeError(BOOM)


def _raising_probe() -> bool:
    """A lock probe that fails every time it is asked."""
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


async def test_run_exit_sequence_when_a_step_raises_does_close_on_the_error(
    repo: str,
):
    start_with(repo)
    jsonl_path = _make_log_unreadable(repo)

    run = await run_sequence(session_context(repo))

    assert run.report.steps == ()
    assert run.report.error is not None
    assert jsonl_path in run.report.error
    log_text = await asyncio.to_thread(Path(jsonl_path).read_text, encoding="utf-8")
    recorded = json.loads(log_text.splitlines()[-1])
    assert (recorded["name"], recorded["exit_code"]) == ("supervise", 2)
    assert [(event.action, event.reason) for event in events_of(run.events, FollowUpEvent)] == [
        ("ended", FAILED_PREFIX + run.report.error)
    ]


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

    run = await run_sequence(
        session_context(repo), is_lock_held=HeldProbe(), log=log, progress=progress
    )

    assert run.report.error is not None
    assert BOOM in run.report.error


async def test_run_exit_sequence_when_the_lock_probe_raises_does_report_the_error(repo: str):
    start_with(repo)

    run = await run_sequence(session_context(repo), is_lock_held=_raising_probe)

    assert run.report == ExitReport(steps=(), error=BOOM)


async def test_run_exit_sequence_when_finalize_raises_does_keep_the_settled_step_on_the_report(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    checks_pass(monkeypatch)
    monkeypatch.setattr(
        "gymrat.loop.finalize.now_ns",
        create_autospec(now_ns, side_effect=RuntimeError(BOOM)),
    )

    run = await run_sequence(session_context(repo, checks=CHECKS), finalize=True)

    assert run.report == ExitReport(
        steps=(KEPT_STEP,),
        error=BOOM,
    )
    assert [event.reason for event in events_of(run.events, FollowUpEvent)] == [
        KEPT_STEP.text,
        FAILED_PREFIX + BOOM,
    ]


async def test_run_exit_sequence_when_the_skip_event_raises_after_contention_does_report_the_error(
    repo: str, hold_repo_lock: Callable[[str], str]
):
    start_with(repo)
    hold_repo_lock(repo)

    run = await run_sequence(
        session_context(repo), is_lock_held=lambda: False, log=_raising_observer
    )

    assert run.report.error is not None
    assert BOOM in run.report.error


async def test_run_exit_sequence_when_a_later_step_raises_does_close_after_the_steps_it_took(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    checks_pass(monkeypatch)
    observer = FailsOnNthEvent(2)

    run = await run_sequence(session_context(repo, checks=CHECKS), finalize=False, log=observer)

    assert run.report.error is not None
    assert BOOM in run.report.error
    assert run.report.steps == (KEPT_STEP, NOT_FINALIZED_STEP)
    assert [
        (event.action, event.reason) for event in events_of(observer.events, FollowUpEvent)
    ] == [
        ("ended", KEPT_STEP.text),
        ("ended", FAILED_PREFIX + run.report.error),
    ]


async def test_run_exit_sequence_when_the_closing_event_raises_does_still_report_the_step_error(
    repo: str,
):
    start_with(repo)
    jsonl_path = _make_log_unreadable(repo)

    run = await run_sequence(session_context(repo), log=_raising_observer)

    assert run.report.error is not None
    assert jsonl_path in run.report.error


# ---------------------------------------------------------------------------
# layering — the exit sequence stays free of the CLI package
# ---------------------------------------------------------------------------


def test_importing_exit_sequence_when_fresh_interpreter_does_not_load_the_cli_package():
    loaded = modules_imported_by("gymrat.supervisor.exit_sequence")

    assert loaded_under(loaded, "gymrat.cli") == []
