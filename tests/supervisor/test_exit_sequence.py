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

import os
from typing import TYPE_CHECKING, override
from unittest.mock import create_autospec

import pytest

from gymrat.session.paths import lockfile_path, session_jsonl_path
from gymrat.session.records import (
    CommandRecord,
    FinalizeRecord,
)
from gymrat.session.store import append_record
from gymrat.supervisor.exit_sequence import (
    ExitPhase,
    ExitReport,
    ExitStep,
)
from gymrat.utils import SHORT_SHA_LENGTH
from tests._clock import install_monotonic_clock
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
    append_raw,
    discard_record,
    finalize_record,
    iteration_record,
    log_records,
    records_of_type,
    session_header_of,
)
from tests.supervisor._exit_sequence import (
    improved_iteration,
    is_kept_step,
    is_left_open_step,
    run_sequence,
    session_context,
)
from tests.supervisor._fixtures import follow_up_reasons, raising_observer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from filelock import FileLock

    from gymrat.supervisor.events import SessionEvent
    from gymrat.supervisor.supervise import EndedBy

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
        self._clock = install_monotonic_clock(monkeypatch, start_ms=0.0)

    @override
    def __call__(self) -> bool:
        self._clock.tick(100.0)
        return super().__call__()


def _make_log_unreadable(root: str) -> str:
    """Append a line no record schema matches, returning the session log path."""
    jsonl_path = session_jsonl_path(root)
    append_raw(jsonl_path, UNREADABLE_LINE.encode())
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


def _is_finalized_step(step: ExitStep, root: str) -> bool:
    """Whether ``step`` names the branch and short commit of the finalize record in ``root``'s log."""
    record = _finalize_record(root)
    return (
        step.kind == "finalized"
        and record.branch in step.text
        and record.commit[:SHORT_SHA_LENGTH] in step.text
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

    assert [step.kind for step in run.report.steps] == ["skipped"]
    assert str(os.getpid()) in run.report.steps[0].text
    assert run.report.error is None
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
    ("ended_by", "noted"),
    [
        ("wall-clock", True),
        ("session", False),
    ],
)
async def test_run_exit_sequence_when_skipping_does_note_a_cap_trip_only_for_cap_ends(
    repo: str, ended_by: EndedBy, noted: bool
):
    start_with(repo)

    run = await run_sequence(
        session_context(repo), ended_by=ended_by, lock_wait_ms=0, is_lock_held=lambda: True
    )

    (step,) = run.report.steps
    assert (step.kind, "unknown" in step.text, "cap" in step.text) == ("skipped", True, noted)


async def test_run_exit_sequence_when_the_lock_is_taken_after_the_probe_does_skip(
    repo: str, hold_repo_lock: Callable[[str], str]
):
    start_with(repo)
    hold_repo_lock(repo)

    run = await run_sequence(session_context(repo), is_lock_held=lambda: False)

    (step,) = run.report.steps
    assert (step.kind, str(os.getpid()) in step.text) == ("skipped", True)
    assert run.phases == []


# ---------------------------------------------------------------------------
# already finalized
# ---------------------------------------------------------------------------


async def test_run_exit_sequence_when_session_already_finalized_does_report_nothing_left_to_do(
    repo: str,
):
    start_with(repo, (finalize_record(),))

    run = await run_sequence(session_context(repo), finalize=True)

    (step,) = run.report.steps
    assert (step.kind, "finalized" in step.text) == ("nothing", True)
    assert run.report.error is None


# ---------------------------------------------------------------------------
# nothing to settle
# ---------------------------------------------------------------------------


def _measured_nothing(repo: str) -> None:
    start_with(repo)


def _opened_no_session(repo: str) -> None:
    del repo


def _discarded_the_last_iteration(repo: str) -> None:
    start_with(repo, (iteration_record(seq=1), discard_record(1)))


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_measured_nothing, id="session-measured-nothing"),
        pytest.param(_opened_no_session, id="no-session-opened"),
        pytest.param(_discarded_the_last_iteration, id="last-iteration-discarded"),
    ],
)
async def test_run_exit_sequence_when_nothing_is_unsettled_does_report_it_under_the_lock(
    repo: str, arrange: Callable[[str], None]
):
    arrange(repo)

    run = await run_sequence(session_context(repo, checks=CHECKS))

    (step,) = run.report.steps
    assert (step.kind, "settle" in step.text) == ("nothing", True)
    assert run.report.error is None
    assert run.phases == [ExitPhase(kind="settling", pid=None)]
    assert follow_up_reasons(run.events) == [("ended", step.text)]


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

    assert _is_finalized_step(run.report.steps[-1], repo)
    assert run.report.error is None


async def test_run_exit_sequence_when_finalize_refuses_does_record_the_refusal_not_an_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    improved_iteration(repo)
    checks_pass(monkeypatch)
    taken = f"{session_header_of(repo).branch}-final"
    run_git(["branch", taken], repo)

    run = await run_sequence(session_context(repo, checks=CHECKS), finalize=True)

    refused = run.report.steps[-1]
    assert (refused.kind, taken in refused.text) == ("refused", True)
    assert run.report.error is None


async def test_run_exit_sequence_when_finalize_is_off_and_all_settled_does_leave_the_session_open(
    repo: str,
):
    start_with(repo)
    _keep_iteration(repo, 1)

    run = await run_sequence(session_context(repo), finalize=False)

    (step,) = run.report.steps
    assert is_left_open_step(step)


# ---------------------------------------------------------------------------
# the error boundary
# ---------------------------------------------------------------------------

#: What a dead sink raises, so the report it lands in can be checked for it.
BOOM = "the sink is down"


_raising_observer = raising_observer(BOOM)


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
    error = run.report.error
    assert [
        (action, reason is not None and error in reason)
        for action, reason in follow_up_reasons(run.events)
    ] == [("ended", True)]


async def test_run_exit_sequence_when_the_skip_event_raises_during_the_wait_does_report_the_error(
    repo: str,
):
    start_with(repo)

    run = await run_sequence(session_context(repo), is_lock_held=HeldProbe(), log=_raising_observer)

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
        "gymrat.loop.finalize.append_record",
        create_autospec(append_record, side_effect=RuntimeError(BOOM)),
    )

    run = await run_sequence(session_context(repo, checks=CHECKS), finalize=True)

    (kept,) = run.report.steps
    assert is_kept_step(kept)
    assert run.report.error == BOOM


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

    error = run.report.error
    assert error is not None
    assert BOOM in error
    kept, left_open = run.report.steps
    assert (is_kept_step(kept), is_left_open_step(left_open)) == (True, True)
    reasons = follow_up_reasons(observer.events)
    assert [(action, reason is not None and error in reason) for action, reason in reasons] == [
        ("ended", False),
        ("ended", True),
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
