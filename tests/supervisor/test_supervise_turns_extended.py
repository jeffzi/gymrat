"""Wall-clock caps, settle-window cancellation, log writing, and consecutive-discard guards."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gymrat.session.clock import now_ns
from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import append_record
from gymrat.supervisor import (
    Driver,
    LaunchEvent,
    SessionObserver,
    SessionPrompt,
    SupervisedSession,
    TextDeltaEvent,
    UsageUpdateEvent,
    supervise,
)
from gymrat.supervisor.turns import CONSECUTIVE_DISCARD_LIMIT

if TYPE_CHECKING:
    from gymrat.supervisor.supervise import SupervisionResult
from tests.session.records._fixtures import discard_record
from tests.supervisor._fixtures import (
    _cap_events,
    add_stop_async,
    collecting_observer,
    emit_turn_end,
    follow_ups_with_action,
    make_context,
    make_launch,
    make_prompt,
    read_log_lines,
    seed_session_log,
    seed_with_stop,
    sent_texts,
    supervise_fast,
)
from tests.supervisor._mock_driver import (
    ActionStep,
    CostStep,
    EmitStep,
    TurnEndStep,
    create_mock_driver,
)

# ---------------------------------------------------------------------------
# behavior 10: wall-clock cap idle vs in-flight
# ---------------------------------------------------------------------------

_WALL_CLOCK_MAX_MINUTES = 0.001
"""A deadline low enough that the wall-clock poll fires on its first check."""


async def _supervise_wall_clock(
    driver: Driver,
    prompt: SessionPrompt,
    *,
    context: SupervisedSession,
    launch: LaunchEvent,
    observer: SessionObserver,
    settle_window_ms: int,
) -> SupervisionResult:
    """Call ``supervise`` with the wall-clock-cap kwargs every wall-clock test shares."""
    return await supervise(
        driver,
        prompt,
        context=context,
        launch=launch,
        observer=observer,
        settle_window_ms=settle_window_ms,
        lock_poll_ms=1,
        is_lock_held=lambda: False,
        grace_ms=50,
        wall_clock_poll_ms=1,
    )


def _calls_of_kind(calls: list[tuple[str, str | None]], kind: str) -> list[tuple[str, str | None]]:
    return [c for c in calls if c[0] == kind]


async def test_supervise_when_wall_clock_fires_during_settle_window_does_call_end_not_interrupt(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver([
        TurnEndStep(cost_usd=0.01),
    ])

    result = await _supervise_wall_clock(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
            max_minutes=_WALL_CLOCK_MAX_MINUTES,
        ),
        launch=make_launch(max_minutes=_WALL_CLOCK_MAX_MINUTES),
        observer=probe.observer,
        settle_window_ms=10_000,
    )

    assert result.ended_by == "wall-clock"
    assert result.end_reason == "wall-clock"
    assert result.outcome.reason == "completed"

    caps = _cap_events(probe.events)
    assert len(caps) == 1
    assert caps[0].cap == "wall-clock"

    session = driver.sessions[0]
    assert len(_calls_of_kind(session.calls, "end")) == 1
    assert len(_calls_of_kind(session.calls, "interrupt")) == 0


async def test_supervise_when_wall_clock_fires_after_reply_sent_does_call_interrupt_not_end(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver([
        TurnEndStep(cost_usd=0.01, origin="agent"),
        CostStep(cost_usd=0.01, delay_ms=2_000),
    ])

    result = await _supervise_wall_clock(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
            max_minutes=_WALL_CLOCK_MAX_MINUTES,
        ),
        launch=make_launch(max_minutes=_WALL_CLOCK_MAX_MINUTES),
        observer=probe.observer,
        settle_window_ms=0,
    )

    assert result.ended_by == "wall-clock"

    caps = _cap_events(probe.events)
    assert len(caps) == 1
    assert caps[0].cap == "wall-clock"

    session = driver.sessions[0]
    assert len(_calls_of_kind(session.calls, "interrupt")) >= 1
    assert len(_calls_of_kind(session.calls, "end")) == 0


# ---------------------------------------------------------------------------
# behavior 11: GymratError from log reading
# ---------------------------------------------------------------------------


async def test_supervise_when_log_read_fails_does_return_error_outcome(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    # Write a corrupt session log so read_records raises GymratError
    corrupt_log = Path(session_jsonl_path(root))
    corrupt_log.parent.mkdir(parents=True, exist_ok=True)
    corrupt_log.write_text("not valid json\n", encoding="utf-8")  # noqa: ASYNC240 - sync setup before any async call
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver([
        TurnEndStep(cost_usd=0.01),
    ])

    result = await supervise_fast(
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

    ended = follow_ups_with_action(probe.events, "ended")
    assert len(ended) >= 1


# ---------------------------------------------------------------------------
# behavior 3: in-flight detection cancels settle window
# ---------------------------------------------------------------------------


async def test_supervise_when_in_flight_event_during_settle_does_cancel_settle(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver([
        emit_turn_end(),
        EmitStep(
            emit=TextDeltaEvent(at=now_ns(), chunk="hello"),
            delay_ms=5,
        ),
        ActionStep(action=lambda: add_stop_async(root)),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])

    result = await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
        settle_window_ms=50,
    )

    assert result.ended_by == "session"

    replied_events = follow_ups_with_action(probe.events, "replied")
    assert len(replied_events) == 0

    session = driver.sessions[0]
    assert len(sent_texts(session)) == 0


async def test_supervise_when_usage_update_during_settle_does_not_cancel_settle(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver([
        emit_turn_end(),
        EmitStep(
            emit=UsageUpdateEvent(at=now_ns(), cost_usd=0.02),
            delay_ms=5,
        ),
        ActionStep(action=lambda: add_stop_async(root), delay_ms=200),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])

    result = await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
        settle_window_ms=50,
    )

    assert result.ended_by == "session"

    replied_events = follow_ups_with_action(probe.events, "replied")
    assert len(replied_events) == 1

    session = driver.sessions[0]
    assert len(sent_texts(session)) == 1


# ---------------------------------------------------------------------------
# behavior 14: JSONL log carries turn_end and follow_up lines
# ---------------------------------------------------------------------------


async def test_supervise_when_turn_classified_does_write_turn_end_and_follow_up_to_log(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_with_stop(root)
    lock_path = str(tmp_path / "lockfile")
    log_path = tmp_path / "events.jsonl"

    driver = create_mock_driver([
        TurnEndStep(cost_usd=0.01),
    ])

    await supervise_fast(
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
# behavior 15: consecutive discards guard
# ---------------------------------------------------------------------------


async def test_supervise_when_consecutive_discards_guard_trips_does_end_as_guard(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")

    async def add_discards() -> None:
        """Append enough discard records to trip the guard on the next classify."""
        for i in range(1, CONSECUTIVE_DISCARD_LIMIT + 1):
            append_record(session_jsonl_path(root), discard_record(seq=i))

    driver = create_mock_driver([
        ActionStep(action=add_discards),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])

    result = await supervise_fast(
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
