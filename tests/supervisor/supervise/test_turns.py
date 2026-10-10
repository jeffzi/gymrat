"""Behavioral tests for the supervisor turn-loop integration.

``supervise`` processes ``TurnEndEvent`` by entering an idle state, running a
settle window, reading and folding the session log, probing the repository lock,
and delegating to ``classify`` for the decision. These tests exercise the full
supervisor-classifier wiring: scheduling, lock polling, guard propagation, and
the follow-up events that each decision emits through the combined observer and
into the JSONL log.

Unless otherwise noted, tests run through ``run_supervised``, which passes
``settle_window_ms=0`` and ``lock_poll_ms=1`` to keep runs instantaneous and
injects ``is_lock_held`` as a callable to avoid filesystem contention.

Settle-window and lock-poll cancellation, and the wall-clock cap's quiet tail,
are covered here too.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from gymrat.clock import now_ns
from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    TextDeltaEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
)
from gymrat.supervisor.turns import CONSECUTIVE_DISCARD_LIMIT
from tests._lock import hold_lock
from tests.session.records._fixtures import (
    append_records,
    command_record,
    discard_record,
    stop_record,
)
from tests.supervisor._fixtures import (
    FOLLOW_UP_TIMEOUT_S,
    WAIT_FINISHED_LINE,
    ActionStep,
    EmitStep,
    FollowUpWatch,
    InterruptEmitsEndDriver,
    LockSwitch,
    MockStep,
    SlowEndSession,
    SupervisorClock,
    TurnEndStep,
    WrapDriver,
    append_step,
    blocked_step,
    collecting_observer,
    create_mock_driver,
    driver_calls,
    emit_turn_end,
    event_log_markers,
    events_of,
    follow_ups_with_action,
    lock_file_path,
    run_supervised,
    sent_texts,
    tool_start_event,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gymrat.supervisor.events import SessionEvent


# ---------------------------------------------------------------------------
# a turn end settles, folds the log and classifies
# ---------------------------------------------------------------------------


async def test_supervise_when_turn_end_with_stop_record_does_log_a_finished_end(root: str):
    append_records(root, stop_record())
    probe = collecting_observer()
    driver = create_mock_driver([TurnEndStep(cost_usd=0.01)])

    result = await run_supervised(root, driver, observer=probe.observer)

    assert (result.ended_by, result.end_reason, result.outcome.reason) == (
        "session",
        None,
        "completed",
    )
    assert [(e.action, e.reason, e.text) for e in events_of(probe.events, FollowUpEvent)] == [
        ("ended", "finished", None)
    ]
    assert [
        marker
        for marker in event_log_markers(root)
        if marker == "turn_end" or marker.startswith("follow_up:")
    ] == ["turn_end", "follow_up:ended:finished"]


# ---------------------------------------------------------------------------
# a turn end while a reply is outstanding
# ---------------------------------------------------------------------------


async def test_supervise_when_injected_turn_end_arrives_while_reply_outstanding_does_not_follow_up_on_it(
    root: str,
):
    # First agent turn → Reply (outstanding). The injected turn end is emitted
    # via EmitStep (not TurnEndStep) so the mock does not block waiting for
    # send/end — the supervisor ignores it because reply is outstanding.
    # Then the agent turn end arrives, clears outstanding, classifies, stop → end.
    driver = create_mock_driver([
        TurnEndStep(cost_usd=0.01, origin="agent"),
        emit_turn_end(origin="injected"),
        append_step(root, stop_record()),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])
    probe = collecting_observer()

    result = await run_supervised(root, driver, observer=probe.observer)

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
    lock = LockSwitch(held=True)
    driver = create_mock_driver([
        TurnEndStep(cost_usd=0.01, origin="agent"),
        append_step(root, stop_record()),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])

    result = await run_supervised(
        root, driver, observer=lock.release_on_waiting(probe.observer), is_lock_held=lock.is_held
    )

    replies = sent_texts(driver.sessions[0])
    assert result.ended_by == "session"
    assert [e.action for e in events_of(probe.events, FollowUpEvent)] == [
        "waiting",
        "replied",
        "ended",
    ]
    assert len(replies) == 1
    assert replies[0] is not None
    assert replies[0].endswith(WAIT_FINISHED_LINE)


async def test_supervise_when_lock_file_held_does_wait_until_released(root: str):
    lock = hold_lock(str(lock_file_path(root)))
    watch = FollowUpWatch()

    async def release_after_waiting() -> None:
        await watch.until(lambda seen: "waiting" in seen)
        lock.release()

    driver = create_mock_driver([TurnEndStep(cost_usd=0.01, origin="agent")])

    # The group fails the test when the release never happens, instead of
    # leaving supervision waiting on a lock nobody frees.
    async with asyncio.TaskGroup() as group:
        group.create_task(release_after_waiting())
        result = await run_supervised(root, driver, observer=watch, is_lock_held=None)

    assert result.ended_by == "session"
    assert "waiting" in watch.actions()


# ---------------------------------------------------------------------------
# a held lock: one follow-up per release
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("waits_before_release", "stop_before_release", "actions", "calls"),
    [
        pytest.param(
            1, False, ["waiting", "replied", "ended"], ["send", "end"], id="freed-during-settle"
        ),
        pytest.param(1, True, ["waiting", "ended"], ["end"], id="freed-during-settle-with-stop"),
        pytest.param(
            2,
            False,
            ["waiting", "waiting", "replied", "ended"],
            ["send", "end"],
            id="freed-after-the-newer-turn-waits",
        ),
    ],
)
async def test_supervise_when_turn_end_arrives_while_waiting_on_the_lock_does_follow_up_once(
    root: str,
    *,
    waits_before_release: int,
    stop_before_release: bool,
    actions: list[str],
    calls: list[str],
):
    # The stop record for the reply cases lands well after the release, so a
    # superseded lock poll left pending would follow up a second time in between,
    # or outlive the run as a stray task.
    lock = LockSwitch(held=True)
    watch = FollowUpWatch()

    async def release_lock() -> None:
        await watch.until(lambda seen: seen.count("waiting") == waits_before_release)
        if stop_before_release:
            append_records(root, stop_record())
        await lock.release()
        await watch.until(lambda seen: "replied" in seen or "ended" in seen)

    driver = create_mock_driver([
        emit_turn_end(),
        ActionStep(action=lambda: watch.until(lambda seen: "waiting" in seen)),
        emit_turn_end(),
        ActionStep(action=release_lock),
        append_step(root, stop_record(), delay_ms=200),
        TurnEndStep(cost_usd=0.01),
    ])

    result = await run_supervised(
        root, driver, observer=watch, settle_window_ms=50, is_lock_held=lock.is_held
    )
    # Let the run's cancelled tasks finish unwinding, so only a leaked poll remains.
    await asyncio.sleep(0.05)

    assert result.ended_by == "session"
    assert watch.actions() == actions
    assert driver_calls(driver.sessions[0]) == calls
    assert asyncio.all_tasks() == {asyncio.current_task()}


@pytest.mark.parametrize(
    ("agent_steps", "actions", "calls"),
    [
        pytest.param(
            [], ["waiting", "waiting", "replied", "ended"], ["send", "end"], id="lock-frees"
        ),
        pytest.param(
            [EmitStep(emit=TextDeltaEvent(at=now_ns(), chunk="typing"))],
            ["waiting", "waiting", "ended"],
            ["end"],
            id="agent-acts-then-lock-frees",
        ),
    ],
)
async def test_supervise_when_lock_retaken_during_the_poll_settle_does_wait_again(
    root: str, agent_steps: list[MockStep], actions: list[str], calls: list[str]
):
    # The lock reads free exactly once, on the first probe after the supervisor
    # says it is waiting, and is held again by the time the poll settles; after
    # that it stays held until the script frees it.
    lock = LockSwitch(held=True)
    watch = FollowUpWatch()
    freed_once = False

    def is_held_but_briefly_free_after_waiting() -> bool:
        nonlocal freed_once
        if not freed_once and watch.actions() == ["waiting"]:
            freed_once = True
            return False
        return lock.is_held()

    driver = create_mock_driver([
        emit_turn_end(),
        ActionStep(action=lambda: watch.until(lambda seen: seen.count("waiting") == 2)),
        *agent_steps,
        ActionStep(action=lock.release),
        append_step(root, stop_record(), delay_ms=200),
        TurnEndStep(cost_usd=0.01),
    ])

    async with asyncio.timeout(FOLLOW_UP_TIMEOUT_S):
        result = await run_supervised(
            root, driver, observer=watch, is_lock_held=is_held_but_briefly_free_after_waiting
        )

    assert result.ended_by == "session"
    assert watch.actions() == actions
    assert driver_calls(driver.sessions[0]) == calls


# ---------------------------------------------------------------------------
# a guard trips: no progress, or a streak of discards
# ---------------------------------------------------------------------------


def _stale_turns_with_commands(root: str) -> list[MockStep]:
    """Four agent turns, each followed by a command record and no outcome record."""
    steps: list[MockStep] = []
    for seq in range(1, 4):
        steps += [
            TurnEndStep(cost_usd=0.01, origin="agent"),
            append_step(root, command_record(seq=seq)),
        ]
    return [*steps, TurnEndStep(cost_usd=0.01, origin="agent")]


def _discard_streak(root: str) -> list[MockStep]:
    """Enough discard records to trip the streak guard, then one agent turn."""
    discards = [discard_record(seq=seq) for seq in range(1, CONSECUTIVE_DISCARD_LIMIT + 1)]
    return [append_step(root, *discards), TurnEndStep(cost_usd=0.01, origin="agent")]


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

    result = await run_supervised(root, driver)

    assert (result.ended_by, result.end_reason) == ("guard", end_reason)


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
            lambda: tool_start_event("Read", "t1", input_summary="/x"),
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
    lock = LockSwitch(held=lock_starts_held)
    probe = collecting_observer()
    driver = create_mock_driver([
        emit_turn_end(),
        EmitStep(emit=agent_event(), delay_ms=delay_ms),
        ActionStep(action=lock.release),
        append_step(root, stop_record(), delay_ms=200),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])

    result = await run_supervised(
        root,
        driver,
        observer=probe.observer,
        settle_window_ms=settle_window_ms,
        is_lock_held=lock.is_held,
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
        append_step(root, stop_record(), delay_ms=200),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])

    result = await run_supervised(root, driver, observer=probe.observer, settle_window_ms=50)

    replied = follow_ups_with_action(probe.events, "replied")
    assert result.ended_by == "session"
    assert len(replied) == 1
    assert sent_texts(driver.sessions[0]) == [replied[0].text]


# ---------------------------------------------------------------------------
# cap fired — later turn ends emit no follow-up
# ---------------------------------------------------------------------------


_WALL_CLOCK_MAX_MINUTES = 0.001
"""The wall-clock cap the launch event and the turn classifier see.

The patched clock reaching the explicit deadline, not this value, decides when
the cap fires.
"""


def _tool_starts(events: list[SessionEvent]) -> list[ToolStartEvent]:
    return events_of(events, ToolStartEvent)


def _replied_follow_ups(events: list[SessionEvent]) -> list[FollowUpEvent]:
    return follow_ups_with_action(events, "replied")


@pytest.mark.parametrize(
    ("first_step", "reached"),
    [
        pytest.param(
            EmitStep(emit=tool_start_event("Read", "t1", input_summary="/x")),
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
    supervisor_clock: SupervisorClock,
    first_step: MockStep,
    reached: Callable[[list[SessionEvent]], Sequence[SessionEvent]],
):
    probe = collecting_observer()
    state_reached = asyncio.Event()

    def observe(event: SessionEvent) -> None:
        probe.observer(event)
        if reached(probe.events):
            state_reached.set()

    inner = create_mock_driver([
        first_step,
        supervisor_clock.jump_step(supervisor_clock.deadline_ms, after=state_reached),
        blocked_step(),
    ])
    driver = InterruptEmitsEndDriver(inner)

    await run_supervised(
        root,
        driver,
        max_minutes=_WALL_CLOCK_MAX_MINUTES,
        deadline_ms=supervisor_clock.deadline_ms,
        observer=observe,
        grace_ms=50,
    )

    caps = events_of(probe.events, CapEvent)
    cap_idx = probe.events.index(caps[0])
    follow_ups_after_cap = events_of(probe.events[cap_idx + 1 :], FollowUpEvent)
    assert [cap.cap for cap in caps] == ["wall-clock"]
    assert probe.events.index(reached(probe.events)[0]) < cap_idx
    assert follow_ups_after_cap == []


async def test_supervise_when_cap_ends_session_during_settle_window_does_not_reply(
    root: str, supervisor_clock: SupervisorClock
):
    probe = collecting_observer()
    driver = WrapDriver(
        create_mock_driver([TurnEndStep(cost_usd=0.01)]),
        lambda session, _abort: SlowEndSession(session, 400),
    )

    result = await run_supervised(
        root,
        driver,
        observer=supervisor_clock.jump_on(TurnEndEvent, probe.observer),
        deadline_ms=supervisor_clock.deadline_ms,
        settle_window_ms=150,
    )

    # A cap that lands while the settle is in flight ends the session rather
    # than interrupting a turn, so "ending" shows the cap hit the settle window.
    caps = [(cap.cap, cap.action) for cap in events_of(probe.events, CapEvent)]
    assert (result.ended_by, caps) == ("wall-clock", [("wall-clock", "ending")])
    assert follow_ups_with_action(probe.events, "replied") == []
