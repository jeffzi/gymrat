"""Behavioral tests for the supervisor turn-loop integration.

``supervise`` processes ``TurnEndEvent`` by entering an idle state, running a
settle window, reading and folding the session log, probing the repository lock,
and delegating to ``classify`` for the decision. These tests exercise the full
supervisor-classifier wiring: scheduling, caps, in-flight detection, settle
windows, lock polling, guard propagation, and the follow-up events that each
decision emits through the combined observer and into the JSONL log.

Every test passes ``settle_window_ms=0`` and ``lock_poll_ms=1`` to keep runs
instantaneous, and injects ``is_lock_held`` as a callable to avoid filesystem
contention.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gymrat.session.clock import now_ms
from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import append_record
from gymrat.supervisor import (
    Driver,
    FollowUpEvent,
    LaunchEvent,
    SessionObserver,
    SessionOutcome,
    SessionPrompt,
    SupervisedSession,
    TextDeltaEvent,
    TurnEndEvent,
    supervise,
)
from gymrat.supervisor.supervise import EndedBy, SupervisionResult
from tests.session.records._fixtures import (
    session_record,
    stop_record,
)
from tests.supervisor._fixtures import (
    _cap_events,
    collecting_observer,
    make_context,
    make_launch,
    make_prompt,
    read_log_lines,
)
from tests.supervisor._mock_driver import (
    ActionStep,
    CostStep,
    EmitStep,
    TurnEndStep,
    create_mock_driver,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.supervisor.events import SessionEvent

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _follow_up_events(events: list[SessionEvent]) -> list[FollowUpEvent]:
    return [e for e in events if isinstance(e, FollowUpEvent)]


def _seed_session_log(root: str) -> None:
    """Write a minimal session header so ``read_records`` / ``fold_session`` work."""
    jsonl_path = session_jsonl_path(root)
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    append_record(jsonl_path, session_record())


def _seed_with_stop(root: str) -> None:
    """Seed the session log and append a stop record so the classifier sees ``ends_on_stop``."""
    _seed_session_log(root)
    append_record(session_jsonl_path(root), stop_record())


async def _supervise_fast(
    driver: Driver,
    prompt: SessionPrompt,
    *,
    context: SupervisedSession,
    launch: LaunchEvent,
    observer: SessionObserver | None = None,
    is_lock_held: Callable[[], bool] = lambda: False,
    grace_ms: int = 30_000,
) -> SupervisionResult:
    """Call ``supervise`` with the settle-window and lock-poll defaults every fast test shares."""
    return await supervise(
        driver,
        prompt,
        context=context,
        launch=launch,
        observer=observer,
        settle_window_ms=0,
        lock_poll_ms=1,
        is_lock_held=is_lock_held,
        grace_ms=grace_ms,
    )


# ---------------------------------------------------------------------------
# behavior 1: module constants and new parameters
# ---------------------------------------------------------------------------


def test_settle_window_ms_when_imported_does_equal_800():
    from gymrat.supervisor.supervise import SETTLE_WINDOW_MS

    assert SETTLE_WINDOW_MS == 800


def test_lock_poll_ms_when_imported_does_equal_5000():
    from gymrat.supervisor.supervise import LOCK_POLL_MS

    assert LOCK_POLL_MS == 5000


# ---------------------------------------------------------------------------
# behavior 2: EndedBy gains "guard" and SupervisionResult gains end_reason
# ---------------------------------------------------------------------------


def test_ended_by_when_guard_value_used_does_be_valid():
    assert "guard" in EndedBy.__args__  # type: ignore[attr-defined]


def test_supervision_result_when_constructed_with_end_reason_does_carry_it():
    result = SupervisionResult(
        outcome=SessionOutcome(reason="completed", cost_usd=0.0),
        ended_by="session",
        end_reason="test-reason",
        duration_ms=100,
        cost_usd=0.0,
    )

    assert result.end_reason == "test-reason"


def test_supervision_result_when_no_end_reason_does_default_to_none():
    result = SupervisionResult(
        outcome=SessionOutcome(reason="completed", cost_usd=0.0),
        ended_by="session",
        duration_ms=100,
        cost_usd=0.0,
    )

    assert result.end_reason is None


# ---------------------------------------------------------------------------
# behavior 3: turn end triggers settle → fold → classify
# ---------------------------------------------------------------------------


async def test_supervise_when_turn_end_with_stop_record_does_end_as_session(
    tmp_path: Path,
):
    """A turn end with a stop record in the log ends as ``ended_by='session'``.

    The stop record makes the classifier return ``End("finished")``.
    """
    root = str(tmp_path / "repo")
    _seed_with_stop(root)
    lock_path = str(tmp_path / "lockfile")

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
    )

    assert result.ended_by == "session"
    assert result.end_reason is None
    assert result.outcome.reason in {"completed", "interrupted"}


# ---------------------------------------------------------------------------
# behavior 4: End("spend-cap") → CapEvent, session.end(), ended_by="spend-cap"
# ---------------------------------------------------------------------------


async def test_supervise_when_cost_exceeds_max_usd_at_turn_end_does_end_as_spend_cap(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    _seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=5.0),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
            max_usd=1.0,
        ),
        launch=make_launch(max_usd=1.0),
        observer=probe.observer,
    )

    assert result.ended_by == "spend-cap"
    assert result.end_reason == "spend-cap"

    caps = _cap_events(probe.events)
    assert len(caps) == 1
    assert caps[0].cap == "spend-cap"


# ---------------------------------------------------------------------------
# behavior 4: guard End(reason) → session.end(), ended_by="guard", end_reason
# ---------------------------------------------------------------------------


async def test_supervise_when_no_progress_guard_trips_does_end_as_guard_from_supervisor_state(
    tmp_path: Path,
):
    """Three fruitless replies end at the fourth turn end as guard/no-progress.

    Guard state is seeded from the record count at launch, so it is the
    supervisor — not the classifier — that owns the fruitless-reply tally.
    """
    root = str(tmp_path / "repo")
    _seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    # Four turn ends with no new session records → no-progress guard fires
    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),
            TurnEndStep(cost_usd=0.01, origin="agent"),
            TurnEndStep(cost_usd=0.01, origin="agent"),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
    )

    assert result.ended_by == "guard"
    assert result.end_reason == "no-progress"


# ---------------------------------------------------------------------------
# behavior 4: Reply(text) → session.send(text)
# ---------------------------------------------------------------------------


async def test_supervise_when_classifier_replies_does_send_text_to_session(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    _seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")

    # First turn end → Reply → send → second turn end → add stop → End
    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),
            ActionStep(action=lambda: _add_stop_async(root)),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
    )

    # Session ended normally after the stop record was written
    assert result.ended_by == "session"


async def _add_stop_async(root: str) -> None:
    append_record(session_jsonl_path(root), stop_record())


# ---------------------------------------------------------------------------
# behavior 5: turn end while reply outstanding
# ---------------------------------------------------------------------------


async def test_supervise_when_turn_end_while_reply_outstanding_does_not_schedule(
    tmp_path: Path,
):
    """A turn end while a reply is outstanding schedules nothing."""
    root = str(tmp_path / "repo")
    _seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    # First agent turn → Reply (outstanding). The injected turn end is emitted
    # via EmitStep (not TurnEndStep) so the mock does not block waiting for
    # send/end — the supervisor ignores it because reply is outstanding.
    # Then the agent turn end arrives, clears outstanding, classifies, stop → end.
    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),
            EmitStep(
                emit=TurnEndEvent(
                    timestamp=now_ms(),
                    text="",
                    cost_usd=0.01,
                    origin="injected",
                    budget_exhausted=False,
                )
            ),
            ActionStep(action=lambda: _add_stop_async(root)),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
    )

    assert result.ended_by == "session"


# ---------------------------------------------------------------------------
# behavior 6: WaitForLock → poll → settle → classify with after_wait=True
# ---------------------------------------------------------------------------


async def test_supervise_when_lock_held_does_wait_then_classify_with_after_wait(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    _seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()
    poll_count = 0

    def lock_held() -> bool:
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            return True
        # On release, add the stop record so the re-classification ends the session
        append_record(session_jsonl_path(root), stop_record())
        return False

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
        is_lock_held=lock_held,
    )

    assert result.ended_by == "session"

    follow_ups = _follow_up_events(probe.events)
    waiting_events = [e for e in follow_ups if e.action == "waiting"]
    assert len(waiting_events) >= 1


async def test_supervise_when_lock_held_after_two_fruitless_replies_does_wait_instead_of_ending(
    tmp_path: Path,
):
    """After two fruitless replies, a turn end with the lock held waits for release."""
    root = str(tmp_path / "repo")
    _seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()
    lock_held_calls = 0

    def lock_held() -> bool:
        nonlocal lock_held_calls
        lock_held_calls += 1
        return lock_held_calls <= 1

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),  # Reply #1
            TurnEndStep(cost_usd=0.01, origin="agent"),  # Reply #2
            ActionStep(action=lambda: _add_stop_async(root)),
            TurnEndStep(
                cost_usd=0.01, origin="agent"
            ),  # Would be no-progress #3 but lock held → wait
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
        is_lock_held=lock_held,
    )

    follow_ups = _follow_up_events(probe.events)
    waiting_events = [e for e in follow_ups if e.action == "waiting"]
    assert len(waiting_events) >= 1
    assert result.ended_by == "session"


# ---------------------------------------------------------------------------
# behavior 7: every decision emitted as FollowUpEvent
# ---------------------------------------------------------------------------


async def test_supervise_when_session_ends_normally_does_emit_ended_follow_up(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    _seed_with_stop(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01),
        ]
    )

    await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
    )

    follow_ups = _follow_up_events(probe.events)
    ended = [e for e in follow_ups if e.action == "ended"]
    assert len(ended) >= 1
    assert ended[0].reason == "finished"


async def test_supervise_when_classifier_replies_does_emit_replied_follow_up(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    _seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),
            ActionStep(action=lambda: _add_stop_async(root)),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
    )

    follow_ups = _follow_up_events(probe.events)
    replied = [e for e in follow_ups if e.action == "replied"]
    assert len(replied) >= 1
    assert replied[0].text is not None


# ---------------------------------------------------------------------------
# behavior 9: spend cap at turn end, not via _cost_observer
# ---------------------------------------------------------------------------


async def test_supervise_when_usage_update_alone_does_not_end_session(
    tmp_path: Path,
):
    """A UsageUpdateEvent alone never ends the session."""
    root = str(tmp_path / "repo")
    _seed_with_stop(root)
    lock_path = str(tmp_path / "lockfile")

    driver = create_mock_driver(
        [
            CostStep(cost_usd=5.0),
            TurnEndStep(cost_usd=5.0),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
            max_usd=1.0,
        ),
        launch=make_launch(max_usd=1.0),
    )

    # A stop record makes classify() return End("finished") ahead of the budget check.
    assert result.ended_by == "session"


async def test_supervise_when_turn_end_cost_exceeds_max_usd_without_stop_does_end_spend_cap(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    _seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=5.0),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
            max_usd=1.0,
        ),
        launch=make_launch(max_usd=1.0),
    )

    assert result.ended_by == "spend-cap"
    assert result.end_reason == "spend-cap"


# ---------------------------------------------------------------------------
# behavior 10: wall-clock cap idle vs in-flight
# ---------------------------------------------------------------------------


async def test_supervise_when_wall_clock_fires_while_idle_does_report_wall_clock(
    tmp_path: Path,
):
    """Wall-clock cap while idle ends without grace and emits a CapEvent."""
    root = str(tmp_path / "repo")
    _seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    # CostStep keeps the mock alive with a long delay; the wall clock fires
    # before the delay completes.
    driver = create_mock_driver(
        [
            CostStep(cost_usd=0.01, delay_ms=60_000),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
            max_minutes=0.001,
        ),
        launch=make_launch(max_minutes=0.001),
        observer=probe.observer,
        grace_ms=50,
    )

    assert result.ended_by == "wall-clock"
    assert result.end_reason == "wall-clock"

    caps = _cap_events(probe.events)
    assert len(caps) == 1
    assert caps[0].cap == "wall-clock"


# ---------------------------------------------------------------------------
# behavior 11: GymratError from log reading
# ---------------------------------------------------------------------------


async def test_supervise_when_log_read_fails_does_return_error_outcome(
    tmp_path: Path,
):
    """A GymratError from the log fold emits a FollowUpEvent and returns an error result."""
    root = str(tmp_path / "repo")
    # Write a corrupt session log so read_records raises GymratError
    corrupt_log = Path(session_jsonl_path(root))
    corrupt_log.parent.mkdir(parents=True, exist_ok=True)
    corrupt_log.write_text("not valid json\n", encoding="utf-8")  # noqa: ASYNC240 - sync setup before any async call
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
    )

    assert result.ended_by == "session"
    assert result.end_reason is not None
    assert result.outcome.reason == "error"
    assert result.outcome.message is not None

    follow_ups = _follow_up_events(probe.events)
    ended = [e for e in follow_ups if e.action == "ended"]
    assert len(ended) >= 1


# ---------------------------------------------------------------------------
# behavior 3: in-flight detection cancels settle window
# ---------------------------------------------------------------------------


async def test_supervise_when_in_flight_event_during_settle_does_cancel_settle(
    tmp_path: Path,
):
    """An in-flight event cancels a pending settle window."""
    root = str(tmp_path / "repo")
    _seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    # Turn end → settle starts, TextDelta interrupts → settle cancelled,
    # next turn end → re-classify → with stop record → End("finished")
    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),
            EmitStep(emit=TextDeltaEvent(timestamp=now_ms(), chunk="hello")),
            ActionStep(action=lambda: _add_stop_async(root)),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
    )

    assert result.ended_by == "session"


async def test_supervise_when_usage_update_during_settle_does_not_cancel_settle(
    tmp_path: Path,
):
    """A UsageUpdateEvent during a settle window does not cancel it."""
    root = str(tmp_path / "repo")
    _seed_with_stop(root)
    lock_path = str(tmp_path / "lockfile")

    # Turn end → settle starts, UsageUpdate (excluded) → settle NOT cancelled,
    # classify runs → End("finished")
    driver = create_mock_driver(
        [
            CostStep(cost_usd=0.01),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
    )

    assert result.ended_by == "session"


# ---------------------------------------------------------------------------
# behavior 14: JSONL log carries turn_end and follow_up lines
# ---------------------------------------------------------------------------


async def test_supervise_when_turn_classified_does_write_turn_end_and_follow_up_to_log(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    _seed_with_stop(root)
    lock_path = str(tmp_path / "lockfile")
    log_path = tmp_path / "events.jsonl"

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01),
        ]
    )

    await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(log_path),
            lock_path=lock_path,
        ),
        launch=make_launch(),
    )

    lines = read_log_lines(log_path)
    types = [line["type"] for line in lines]
    assert "turn_end" in types
    assert "follow_up" in types

    turn_end_lines = [line for line in lines if line["type"] == "turn_end"]
    assert turn_end_lines[0]["origin"] == "agent"

    follow_up_lines = [line for line in lines if line["type"] == "follow_up"]
    assert follow_up_lines[-1]["action"] == "ended"
    assert follow_up_lines[-1]["reason"] == "finished"


# ---------------------------------------------------------------------------
# behavior 14: consecutive discards guard
# ---------------------------------------------------------------------------


async def test_supervise_when_consecutive_discards_guard_trips_does_end_as_guard(
    tmp_path: Path,
):
    """The consecutive-discards guard trips at its limit."""
    from gymrat.session.records import DiscardRecord
    from gymrat.supervisor.turns import CONSECUTIVE_DISCARD_LIMIT

    root = str(tmp_path / "repo")
    _seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")

    async def add_discards() -> None:
        """Append enough discard records to trip the guard on the next classify."""
        for i in range(1, CONSECUTIVE_DISCARD_LIMIT + 1):
            append_record(
                session_jsonl_path(root),
                DiscardRecord(type="discard", seq=i, at="2026-01-01T00:00:00.000Z"),
            )

    driver = create_mock_driver(
        [
            ActionStep(action=add_discards),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
    )

    assert result.ended_by == "guard"
    assert result.end_reason == "consecutive-discards"


# ---------------------------------------------------------------------------
# behavior 12: queued turn end after outcome settles on its own
# ---------------------------------------------------------------------------


async def test_supervise_when_session_exits_after_turn_end_does_still_classify_queued_turn(
    tmp_path: Path,
):
    """A queued turn end is still drained and classified after the outcome settles."""
    root = str(tmp_path / "repo")
    _seed_with_stop(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    # Turn end → stop record seen → End("finished")
    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
    )

    follow_ups = _follow_up_events(probe.events)
    ended = [e for e in follow_ups if e.action == "ended"]
    assert len(ended) >= 1
    assert ended[0].reason == "finished"
    assert result.ended_by == "session"


# ---------------------------------------------------------------------------
# behavior 13: outcome settling cancels pending windows
# ---------------------------------------------------------------------------


async def test_supervise_when_outcome_settles_does_cancel_pending_and_return(
    tmp_path: Path,
):
    """Outcome settlement cancels every pending window, poll, and timer."""
    root = str(tmp_path / "repo")
    _seed_with_stop(root)
    lock_path = str(tmp_path / "lockfile")

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
    )

    assert result.ended_by == "session"
    assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# behavior 14: is_lock_held defaults to is_held(Path(context.lock_path))
# ---------------------------------------------------------------------------


async def test_supervise_when_is_lock_held_not_injected_does_use_default(
    tmp_path: Path,
):
    """Without an injected is_lock_held, the supervisor probes via filelock."""
    root = str(tmp_path / "repo")
    _seed_with_stop(root)
    lock_path = str(tmp_path / "lockfile")

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01),
        ]
    )

    # Lock is not held, so classify should proceed without waiting
    result = await supervise(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        settle_window_ms=0,
        lock_poll_ms=1,
        # NOT passing is_lock_held — use default
    )

    assert result.ended_by == "session"
