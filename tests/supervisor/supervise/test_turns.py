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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

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
    stop_record,
)
from tests.supervisor._fixtures import (
    FOLLOW_UP_TIMEOUT_S,
    WAIT_FINISHED_LINE,
    FollowUpWatch,
    InterruptEmitsEndDriver,
    LockSwitch,
    SupervisorClock,
    _supervise,
    append_step,
    collecting_observer,
    driver_calls,
    emit_turn_end,
    events_log_path,
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
    lines = read_log_lines(events_log_path(root))
    assert [
        (line["type"], line.get("action"), line.get("reason"))
        for line in lines
        if line["type"] in {"turn_end", "follow_up"}
    ] == [("turn_end", None, None), ("follow_up", "ended", "finished")]


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
    lock = LockSwitch(held=True)
    driver = create_mock_driver([
        TurnEndStep(cost_usd=0.01, origin="agent"),
        append_step(root, stop_record()),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])

    result = await _supervise(
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


async def test_supervise_when_lock_file_held_does_wait_until_released(
    root: str,
    tmp_path: Path,
):
    lock = hold_lock(str(tmp_path / "lockfile"))
    watch = FollowUpWatch()

    async def release_after_waiting() -> None:
        await watch.until(lambda seen: "waiting" in seen)
        lock.release()

    driver = create_mock_driver([TurnEndStep(cost_usd=0.01, origin="agent")])

    # The group fails the test when the release never happens, instead of
    # leaving supervision waiting on a lock nobody frees.
    async with asyncio.TaskGroup() as group:
        group.create_task(release_after_waiting())
        result = await _supervise(root, driver, observer=watch, is_lock_held=None)

    assert result.ended_by == "session"
    assert "waiting" in watch.actions()


# ---------------------------------------------------------------------------
# a held lock: one follow-up per release
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ProbedLock(LockSwitch):
    """A lock switch that records every task whose probe found it free."""

    free_probers: set[asyncio.Task[Any] | None] = field(default_factory=set)

    @override
    def is_held(self) -> bool:
        if not self.held:
            self.free_probers.add(asyncio.current_task())
        return self.held


@pytest.mark.parametrize(
    ("waits_before_release", "stop_before_release", "actions", "calls", "free_probers"),
    [
        pytest.param(
            1,
            False,
            ["waiting", "replied", "ended"],
            ["send", "end"],
            2,
            id="freed-during-settle",
        ),
        pytest.param(1, True, ["waiting", "ended"], ["end"], 1, id="freed-during-settle-with-stop"),
        pytest.param(
            2,
            False,
            ["waiting", "waiting", "replied", "ended"],
            ["send", "end"],
            2,
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
    free_probers: int,
):
    # The stop record for the reply cases lands well after the release, so a
    # superseded lock poll left pending would follow up a second time in between.
    # A superseded poll also probes the freed lock from its own task, long before
    # the 50 ms settle ends the session, so counting the tasks that find the lock
    # free catches it even where the end guard hides a second end.
    lock = _ProbedLock(held=True)
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

    result = await _supervise(
        root, driver, observer=watch, settle_window_ms=50, is_lock_held=lock.is_held
    )

    assert result.ended_by == "session"
    assert watch.actions() == actions
    assert driver_calls(driver.sessions[0]) == calls
    assert len(lock.free_probers) == free_probers


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
        result = await _supervise(
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

    result = await _supervise(root, driver)

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
    lock = LockSwitch(held=lock_starts_held)
    probe = collecting_observer()
    driver = create_mock_driver([
        emit_turn_end(),
        EmitStep(emit=agent_event(), delay_ms=delay_ms),
        ActionStep(action=lock.release),
        append_step(root, stop_record(), delay_ms=200),
        TurnEndStep(cost_usd=0.01, origin="agent"),
    ])

    result = await _supervise(
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

    result = await _supervise(root, driver, observer=probe.observer, settle_window_ms=50)

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
    clock = SupervisorClock(monkeypatch, now_ms())
    deadline_ms = clock.now_ms + 60_000
    state_reached = asyncio.Event()

    def observe(event: SessionEvent) -> None:
        probe.observer(event)
        if reached(probe.events):
            state_reached.set()

    inner = create_mock_driver([
        first_step,
        clock.jump_step(deadline_ms, after=state_reached),
        CostStep(cost_usd=0.01, delay_ms=60_000),
    ])
    driver = InterruptEmitsEndDriver(inner)

    await _supervise(
        root,
        driver,
        max_minutes=_WALL_CLOCK_MAX_MINUTES,
        deadline_ms=deadline_ms,
        launch=make_launch(max_minutes=_WALL_CLOCK_MAX_MINUTES),
        observer=observe,
        grace_ms=50,
    )

    caps = events_of(probe.events, CapEvent)
    cap_idx = probe.events.index(caps[0])
    follow_ups_after_cap = events_of(probe.events[cap_idx + 1 :], FollowUpEvent)
    assert [cap.cap for cap in caps] == ["wall-clock"]
    assert probe.events.index(reached(probe.events)[0]) < cap_idx
    assert follow_ups_after_cap == []
