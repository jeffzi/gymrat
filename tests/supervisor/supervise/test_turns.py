"""Behavioral tests for the supervisor turn-loop integration.

``supervise`` processes ``TurnEndEvent`` by entering an idle state, running a
settle window, reading and folding the session log, probing the repository lock,
and delegating to ``classify`` for the decision. These tests exercise the full
supervisor-classifier wiring: scheduling, lock polling, guard propagation, and
the follow-up events that each decision emits through the combined observer and
into the JSONL log.

Unless otherwise noted, tests run through ``_supervise``, which passes
``settle_window_ms=0`` and ``lock_poll_ms=1`` to keep runs instantaneous and
injects ``is_lock_held`` as a callable to avoid filesystem contention.

Settle-window and lock-poll cancellation, and the wall-clock cap's quiet tail,
are covered here too.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from gymrat.clock import now_ms, now_ns
from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    TextDeltaEvent,
    ToolStartEvent,
    UsageUpdateEvent,
)
from gymrat.supervisor.turns import CONSECUTIVE_DISCARD_LIMIT
from tests._lock import hold_lock
from tests.session.records._fixtures import (
    append_records,
    command_record,
    discard_record,
)
from tests.supervisor._fixtures import (
    InterruptEmitsEndDriver,
    _supervise,
    add_stop_async,
    collecting_observer,
    emit_turn_end,
    events_of,
    follow_ups_with_action,
    make_launch,
    read_log_lines,
    seed_with_stop,
    sent_texts,
)
from tests.supervisor._mock_driver import (
    ActionStep,
    CostStep,
    EmitStep,
    MockStep,
    TurnEndStep,
    create_mock_driver,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from gymrat.supervisor.events import SessionEvent


# ---------------------------------------------------------------------------
# a turn end settles, folds the log and classifies
# ---------------------------------------------------------------------------


async def test_supervise_when_turn_end_with_stop_record_does_log_a_finished_end(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_with_stop(root)
    probe = collecting_observer()
    driver = create_mock_driver([TurnEndStep(cost_usd=0.01)])

    result = await _supervise(root, driver, observer=probe.observer)

    assert (result.ended_by, result.end_reason, result.outcome.reason) == (
        "session",
        None,
        "completed",
    )
    assert [(e.action, e.reason, e.text) for e in events_of(probe.events, FollowUpEvent)] == [
        ("ended", "finished", None)
    ]
    lines = read_log_lines(tmp_path / "events.jsonl")
    assert [
        {key: value for key, value in line.items() if key != "at"}
        for line in lines
        if line["type"] in {"turn_end", "follow_up"}
    ] == [
        {
            "type": "turn_end",
            "text": "",
            "cost_usd": 0.01,
            "origin": "agent",
            "budget_exhausted": False,
        },
        {"type": "follow_up", "action": "ended", "reason": "finished"},
    ]


# ---------------------------------------------------------------------------
# a turn end while a reply is outstanding
# ---------------------------------------------------------------------------


async def test_supervise_when_turn_end_while_reply_outstanding_does_not_schedule(
    root: str,
):
    # First agent turn → Reply (outstanding). The injected turn end is emitted
    # via EmitStep (not TurnEndStep) so the mock does not block waiting for
    # send/end — the supervisor ignores it because reply is outstanding.
    # Then the agent turn end arrives, clears outstanding, classifies, stop → end.
    driver = create_mock_driver([
        TurnEndStep(cost_usd=0.01, origin="agent"),
        emit_turn_end(origin="injected"),
        ActionStep(action=lambda: add_stop_async(root)),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])
    probe = collecting_observer()

    result = await _supervise(root, driver, observer=probe.observer)

    assert result.ended_by == "session"
    assert [e.action for e in events_of(probe.events, FollowUpEvent)] == ["replied", "ended"]
    assert len(sent_texts(driver.sessions[0])) == 1


# ---------------------------------------------------------------------------
# a held lock: wait, then reply with the after-wait line
# ---------------------------------------------------------------------------


async def test_supervise_when_lock_held_does_wait_then_reply_with_the_after_wait_line(
    root: str,
):
    probe = collecting_observer()
    poll_count = 0

    def lock_held() -> bool:
        nonlocal poll_count
        poll_count += 1
        return poll_count == 1

    driver = create_mock_driver([
        TurnEndStep(cost_usd=0.01, origin="agent"),
        ActionStep(action=lambda: add_stop_async(root)),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])

    result = await _supervise(root, driver, observer=probe.observer, is_lock_held=lock_held)

    assert result.ended_by == "session"
    assert [e.action for e in events_of(probe.events, FollowUpEvent)] == [
        "waiting",
        "replied",
        "ended",
    ]
    replies = sent_texts(driver.sessions[0])
    assert len(replies) == 1
    assert replies[0] is not None
    assert replies[0].endswith(
        "The command you left running has finished; its record, if any, is in the session log."
    )


async def test_supervise_when_real_lock_held_does_wait_then_reply_with_after_wait_line(
    root: str,
    tmp_path: Path,
):
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    lock = hold_lock(lock_path)

    released = False

    async def release_after_waiting() -> None:
        nonlocal released
        while not any(  # noqa: ASYNC110 -- probe.events is a plain list with no hook to wire an Event to
            e.action == "waiting" for e in events_of(probe.events, FollowUpEvent)
        ):
            await asyncio.sleep(0.005)
        lock.release()
        released = True

    _background_task: asyncio.Task[None] | None = None

    async def schedule_release() -> None:
        nonlocal _background_task
        _background_task = asyncio.create_task(release_after_waiting())

    driver = create_mock_driver([
        ActionStep(action=schedule_release),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])

    result = await _supervise(root, driver, observer=probe.observer, is_lock_held=None)

    assert released
    assert result.ended_by == "session"

    follow_ups = events_of(probe.events, FollowUpEvent)
    assert any(e.action == "waiting" for e in follow_ups)
    assert any(e.action == "replied" for e in follow_ups)

    session = driver.sessions[0]
    sent = sent_texts(session)
    assert any(
        "The command you left running has finished" in text for text in sent if text is not None
    )


# ---------------------------------------------------------------------------
# a guard trips: no progress, or a streak of discards
# ---------------------------------------------------------------------------


def _stale_turns_with_commands(root: str) -> list[MockStep]:
    """Four agent turns, each followed by a command record and no outcome record."""
    seqs = iter(range(1, 4))

    async def append_command() -> None:
        append_records(root, command_record(seq=next(seqs)))

    steps: list[MockStep] = []
    for _ in range(3):
        steps += [TurnEndStep(cost_usd=0.01, origin="agent"), ActionStep(action=append_command)]
    return [*steps, TurnEndStep(cost_usd=0.01, origin="agent")]


def _discard_streak(root: str) -> list[MockStep]:
    """Enough discard records to trip the streak guard, then one agent turn."""

    async def add_discards() -> None:
        for seq in range(1, CONSECUTIVE_DISCARD_LIMIT + 1):
            append_records(root, discard_record(seq=seq))

    return [ActionStep(action=add_discards), TurnEndStep(cost_usd=0.01, origin="agent")]


@pytest.mark.parametrize(
    ("steps", "end_reason"),
    [
        pytest.param(_stale_turns_with_commands, "no-progress", id="no-progress-despite-commands"),
        pytest.param(_discard_streak, "consecutive-discards", id="consecutive-discards"),
    ],
)
async def test_supervise_when_a_guard_trips_does_end_as_guard_naming_it(
    root: str, steps: Callable[[str], list[MockStep]], end_reason: str
):
    driver = create_mock_driver(steps(root))

    result = await _supervise(root, driver)

    assert (result.ended_by, result.end_reason) == ("guard", end_reason)


_WALL_CLOCK_MAX_MINUTES = 0.001
"""A deadline low enough that the wall-clock poll fires on its first check."""


# ---------------------------------------------------------------------------
# agent activity cancels the pending settle window or lock poll
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delay_ms", "settle_window_ms", "lock_starts_held", "actions"),
    [
        pytest.param(5, 50, False, ["ended"], id="settle-window"),
        pytest.param(30, 0, True, ["waiting", "ended"], id="lock-poll"),
    ],
)
@pytest.mark.parametrize(
    "agent_event",
    [
        pytest.param(lambda: TextDeltaEvent(at=now_ns(), chunk="typing"), id="text-delta"),
        pytest.param(
            lambda: ToolStartEvent(
                at=now_ns(), tool_use_id="t1", tool_name="Read", input={}, input_summary="/x"
            ),
            id="tool-start",
        ),
    ],
)
async def test_supervise_when_agent_acts_while_a_reply_is_pending_does_cancel_it_quietly(
    root: str,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    *,
    delay_ms: int,
    settle_window_ms: int,
    lock_starts_held: bool,
    actions: list[str],
    agent_event: Callable[[], SessionEvent],
):
    # The lock frees and the stop record lands well after the agent acts, so a
    # pending reply the activity failed to cancel would fire in between and send text.
    held = [lock_starts_held]

    async def release_lock() -> None:
        held[0] = False

    probe = collecting_observer()
    driver = create_mock_driver([
        emit_turn_end(),
        EmitStep(emit=agent_event(), delay_ms=delay_ms),
        ActionStep(action=release_lock),
        ActionStep(action=lambda: add_stop_async(root), delay_ms=200),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])

    result = await _supervise(
        root,
        driver,
        observer=probe.observer,
        settle_window_ms=settle_window_ms,
        is_lock_held=lambda: held[0],
    )

    assert result.ended_by == "session"
    assert [e.action for e in events_of(probe.events, FollowUpEvent)] == actions
    assert sent_texts(driver.sessions[0]) == []
    assert "failed" not in capsys.readouterr().err
    assert [record for record in caplog.records if record.name == "asyncio"] == []


# ---------------------------------------------------------------------------
# events that are not agent activity leave the pending reply alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(UsageUpdateEvent(at=now_ns(), cost_usd=0.02), id="usage-update"),
        pytest.param(CompactionEvent(at=now_ns()), id="compaction"),
    ],
)
async def test_supervise_when_a_passive_event_arrives_during_settle_does_still_reply(
    root: str, event: SessionEvent
):
    probe = collecting_observer()
    driver = create_mock_driver([
        emit_turn_end(),
        EmitStep(emit=event, delay_ms=5),
        ActionStep(action=lambda: add_stop_async(root), delay_ms=200),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])

    result = await _supervise(root, driver, observer=probe.observer, settle_window_ms=50)

    replied = follow_ups_with_action(probe.events, "replied")
    assert result.ended_by == "session"
    assert len(replied) == 1
    assert sent_texts(driver.sessions[0]) == [replied[0].text]


# ---------------------------------------------------------------------------
# cap fired — later turn ends emit no follow-up
# ---------------------------------------------------------------------------


def _tool_starts(events: list[SessionEvent]) -> list[ToolStartEvent]:
    return events_of(events, ToolStartEvent)


def _replied_follow_ups(events: list[SessionEvent]) -> list[FollowUpEvent]:
    return follow_ups_with_action(events, "replied")


@pytest.mark.parametrize(
    ("first_step", "reached"),
    [
        pytest.param(
            EmitStep(
                emit=ToolStartEvent(
                    at=now_ns(),
                    tool_use_id="t1",
                    tool_name="Read",
                    input={},
                    input_summary="/x",
                ),
            ),
            _tool_starts,
            id="in-flight",
        ),
        pytest.param(
            TurnEndStep(cost_usd=0.01, origin="agent"),
            _replied_follow_ups,
            id="reply-outstanding",
        ),
    ],
)
async def test_supervise_when_wall_clock_cap_then_turn_end_does_not_emit_follow_up(
    root: str,
    monkeypatch: pytest.MonkeyPatch,
    first_step: MockStep,
    reached: Callable[[list[SessionEvent]], Sequence[SessionEvent]],
):
    probe = collecting_observer()
    clock = [now_ms()]
    deadline_ms = clock[0] + 60_000
    monkeypatch.setattr("gymrat.supervisor.supervise.now_ms", lambda: clock[0])

    state_reached = asyncio.Event()

    def observe(event: SessionEvent) -> None:
        probe.observer(event)
        if reached(probe.events):
            state_reached.set()

    async def pass_deadline_once_reached() -> None:
        async with asyncio.timeout(5):
            await state_reached.wait()
        clock[0] = deadline_ms

    inner = create_mock_driver([
        first_step,
        ActionStep(action=pass_deadline_once_reached),
        CostStep(cost_usd=0.01, delay_ms=60_000),
    ])
    driver = InterruptEmitsEndDriver(inner)

    result = await _supervise(
        root,
        driver,
        max_minutes=_WALL_CLOCK_MAX_MINUTES,
        deadline_ms=deadline_ms,
        launch=make_launch(max_minutes=_WALL_CLOCK_MAX_MINUTES),
        observer=observe,
        grace_ms=50,
    )

    assert result.ended_by == "wall-clock"

    caps = events_of(probe.events, CapEvent)
    assert len(caps) == 1
    assert caps[0].cap == "wall-clock"

    cap_idx = probe.events.index(caps[0])
    assert probe.events.index(reached(probe.events)[0]) < cap_idx
    events_after_cap = probe.events[cap_idx + 1 :]
    follow_ups_after = events_of(events_after_cap, FollowUpEvent)
    assert follow_ups_after == []
