"""Behavioral tests for the mock driver test fixture.

``create_mock_driver`` builds a :class:`Driver` whose ``start`` runs a
caller-supplied script of steps on the running event loop. Tests coordinate
ordering through ``asyncio.Event`` handshakes and small real delays so every
test stays deterministic under ``pytest-randomly`` and ``pytest-xdist``.
Timing is asserted only as loose lower bounds, never exact wall-clock values.
"""

import asyncio
import time

import pytest

from gymrat.supervisor import SessionOutcome, TextDeltaEvent, TurnEndEvent
from gymrat.supervisor.events import SessionEvent
from tests.supervisor._fixtures import collecting_observer, make_prompt, noop_observer
from tests.supervisor._mock_driver import (
    ActionStep,
    CostStep,
    EmitStep,
    TurnEndStep,
    create_mock_driver,
)


async def _noop_action() -> None:
    return None


# ---------------------------------------------------------------------------
# script execution
# ---------------------------------------------------------------------------


async def test_create_mock_driver_when_started_does_defer_first_emit_until_control_yields():
    event = TextDeltaEvent(timestamp=1, chunk="hello")
    probe = collecting_observer()
    driver = create_mock_driver([EmitStep(emit=event)])

    session = driver.start(make_prompt(), probe.observer)

    assert probe.events == []
    await session.outcome  # drain the scheduled script


async def test_create_mock_driver_when_emit_steps_run_does_deliver_events_in_order():
    event0 = TextDeltaEvent(timestamp=1, chunk="hello")
    event1 = TextDeltaEvent(timestamp=2, chunk="world")
    probe = collecting_observer()
    driver = create_mock_driver([EmitStep(emit=event0), EmitStep(emit=event1)])

    session = driver.start(make_prompt(), probe.observer)
    await session.outcome

    assert probe.events == [event0, event1]


async def test_create_mock_driver_when_action_steps_run_does_invoke_callbacks_in_order():
    order: list[int] = []

    async def first() -> None:
        order.append(1)

    async def second() -> None:
        order.append(2)

    driver = create_mock_driver([ActionStep(action=first), ActionStep(action=second)])

    session = driver.start(make_prompt(), noop_observer())
    await session.outcome

    assert order == [1, 2]


async def test_create_mock_driver_when_cost_steps_run_does_emit_usage_updates_in_order():
    probe = collecting_observer()
    driver = create_mock_driver([CostStep(cost_usd=0.05), CostStep(cost_usd=0.1)])

    session = driver.start(make_prompt(), probe.observer)
    await session.outcome

    usage = [e for e in probe.events if e.type == "usage_update"]
    assert [e.cost_usd for e in usage] == [0.05, 0.1]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("steps", "expected_cost"),
    [
        pytest.param([CostStep(cost_usd=0.05), CostStep(cost_usd=0.1)], 0.1, id="last-cost"),
        pytest.param([ActionStep(action=_noop_action)], 0.0, id="no-cost-steps"),
    ],
)
async def test_create_mock_driver_when_script_completes_does_resolve_completed_with_last_cost(
    steps: list[object], expected_cost: float
):
    driver = create_mock_driver(steps)  # type: ignore[arg-type]

    session = driver.start(make_prompt(), noop_observer())
    outcome = await session.outcome

    assert outcome == SessionOutcome(reason="completed", cost_usd=expected_cost)


async def test_create_mock_driver_when_steps_have_delay_ms_does_delay_before_running_them():
    order: list[str] = []

    async def first() -> None:
        order.append("first")

    async def second() -> None:
        order.append("second")

    driver = create_mock_driver(
        [ActionStep(action=first, delay_ms=20), ActionStep(action=second, delay_ms=40)]
    )

    started = time.perf_counter()
    session = driver.start(make_prompt(), noop_observer())
    await session.outcome
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert order == ["first", "second"]
    assert elapsed_ms >= 40


# ---------------------------------------------------------------------------
# interrupt
# ---------------------------------------------------------------------------


async def test_create_mock_driver_when_interrupted_does_stop_successor_steps():
    first_ran = asyncio.Event()
    ran: list[str] = []

    async def first() -> None:
        ran.append("first")
        first_ran.set()

    async def second() -> None:
        ran.append("second")

    driver = create_mock_driver([ActionStep(action=first), ActionStep(action=second, delay_ms=50)])

    session = driver.start(make_prompt(), noop_observer())
    await first_ran.wait()
    await session.interrupt()
    outcome = await session.outcome

    assert ran == ["first"]
    assert outcome.reason == "interrupted"


async def test_create_mock_driver_when_interrupted_does_report_last_known_cost():
    cost_seen = asyncio.Event()

    def observer(event: SessionEvent) -> None:
        if event.type == "usage_update":
            cost_seen.set()

    driver = create_mock_driver(
        [CostStep(cost_usd=0.07), ActionStep(action=_noop_action, delay_ms=1000)]
    )

    session = driver.start(make_prompt(), observer)
    await cost_seen.wait()
    await session.interrupt()
    outcome = await session.outcome

    assert outcome == SessionOutcome(reason="interrupted", cost_usd=0.07)


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


async def test_create_mock_driver_when_action_raises_does_resolve_error_and_stop():
    reached: list[str] = []
    boom_message = "kaboom"

    async def boom() -> None:
        raise RuntimeError(boom_message)

    async def after() -> None:
        reached.append("after")

    driver = create_mock_driver(
        [CostStep(cost_usd=0.03), ActionStep(action=boom), ActionStep(action=after)]
    )

    session = driver.start(make_prompt(), noop_observer())
    outcome = await session.outcome

    assert outcome == SessionOutcome(reason="error", cost_usd=0.03, message="kaboom")
    assert reached == []


# ---------------------------------------------------------------------------
# abort signal
# ---------------------------------------------------------------------------


async def test_create_mock_driver_when_abort_fires_mid_script_does_stop_and_report_interrupted():
    abort = asyncio.Event()
    first_ran = asyncio.Event()
    ran: list[str] = []

    async def first() -> None:
        ran.append("first")
        first_ran.set()

    async def second() -> None:
        ran.append("second")

    driver = create_mock_driver([ActionStep(action=first), ActionStep(action=second, delay_ms=50)])

    session = driver.start(make_prompt(), noop_observer(), abort)
    await first_ran.wait()
    abort.set()
    outcome = await session.outcome

    assert ran == ["first"]
    assert outcome.reason == "interrupted"


async def test_create_mock_driver_when_abort_preset_does_resolve_interrupted_immediately():
    abort = asyncio.Event()
    abort.set()
    ran: list[str] = []

    async def action() -> None:
        ran.append("ran")

    driver = create_mock_driver([ActionStep(action=action)])

    session = driver.start(make_prompt(), noop_observer(), abort)
    outcome = await session.outcome

    assert outcome.reason == "interrupted"
    assert ran == []


# ---------------------------------------------------------------------------
# turn end step
# ---------------------------------------------------------------------------


async def test_create_mock_driver_when_turn_end_step_runs_does_emit_turn_end_event():
    probe = collecting_observer()
    driver = create_mock_driver([CostStep(cost_usd=0.05), TurnEndStep(text="thinking")])

    session = driver.start(make_prompt(), probe.observer)
    await asyncio.sleep(0.05)
    await session.send("continue")
    await session.outcome

    turn_ends = [e for e in probe.events if e.type == "turn_end"]
    assert len(turn_ends) == 1
    te = turn_ends[0]
    assert isinstance(te, TurnEndEvent)
    assert te.text == "thinking"
    assert te.cost_usd == 0.05
    assert te.origin == "agent"
    assert te.budget_exhausted is False


async def test_create_mock_driver_when_turn_end_step_has_cost_usd_does_override_running_cost():
    probe = collecting_observer()
    driver = create_mock_driver([CostStep(cost_usd=0.05), TurnEndStep(text="", cost_usd=0.99)])

    session = driver.start(make_prompt(), probe.observer)
    await asyncio.sleep(0.05)
    await session.send("go")
    outcome = await session.outcome

    turn_ends = [e for e in probe.events if e.type == "turn_end"]
    assert len(turn_ends) == 1
    assert turn_ends[0].cost_usd == 0.99  # type: ignore[attr-defined]
    assert outcome.cost_usd == 0.99


async def test_create_mock_driver_when_turn_end_step_blocks_does_continue_after_send():
    order: list[str] = []

    async def before() -> None:
        order.append("before")

    async def after() -> None:
        order.append("after")

    driver = create_mock_driver(
        [ActionStep(action=before), TurnEndStep(), ActionStep(action=after)]
    )

    session = driver.start(make_prompt(), noop_observer())
    await asyncio.sleep(0.05)
    assert order == ["before"]

    await session.send("go")
    await session.outcome

    assert order == ["before", "after"]


async def test_create_mock_driver_when_end_called_during_turn_end_does_settle_completed():
    driver = create_mock_driver(
        [CostStep(cost_usd=0.1), TurnEndStep(), ActionStep(action=_noop_action)]
    )

    session = driver.start(make_prompt(), noop_observer())
    await asyncio.sleep(0.05)
    await session.end()
    outcome = await session.outcome

    assert outcome == SessionOutcome(reason="completed", cost_usd=0.1)


async def test_create_mock_driver_when_end_called_during_turn_end_does_stop_successor_steps():
    reached: list[str] = []

    async def after() -> None:
        reached.append("after")

    driver = create_mock_driver([TurnEndStep(), ActionStep(action=after)])

    session = driver.start(make_prompt(), noop_observer())
    await asyncio.sleep(0.05)
    await session.end()
    await session.outcome

    assert reached == []


# ---------------------------------------------------------------------------
# session call recording
# ---------------------------------------------------------------------------


async def test_create_mock_driver_when_send_called_does_record_call():
    driver = create_mock_driver([TurnEndStep()])

    session = driver.start(make_prompt(), noop_observer())
    await asyncio.sleep(0.05)
    await session.send("hello")
    await session.outcome

    assert driver.sessions[-1].calls == [("send", "hello")]


async def test_create_mock_driver_when_end_called_does_record_call():
    driver = create_mock_driver([TurnEndStep()])

    session = driver.start(make_prompt(), noop_observer())
    await asyncio.sleep(0.05)
    await session.end()
    await session.outcome

    assert driver.sessions[-1].calls == [("end", None)]


async def test_create_mock_driver_when_multiple_turns_does_record_calls_in_order():
    driver = create_mock_driver([TurnEndStep(), TurnEndStep()])

    session = driver.start(make_prompt(), noop_observer())
    await asyncio.sleep(0.05)
    await session.send("first")
    await asyncio.sleep(0.05)
    await session.send("second")
    await session.outcome

    assert driver.sessions[-1].calls == [("send", "first"), ("send", "second")]


async def test_create_mock_driver_when_interrupt_called_does_record_call_in_order():
    driver = create_mock_driver([TurnEndStep(), TurnEndStep()])

    session = driver.start(make_prompt(), noop_observer())
    await asyncio.sleep(0.05)
    await session.send("first")
    await asyncio.sleep(0.05)
    await session.interrupt()
    await session.outcome

    assert driver.sessions[-1].calls == [("send", "first"), ("interrupt", None)]


# ---------------------------------------------------------------------------
# send/end on settled session
# ---------------------------------------------------------------------------


async def test_create_mock_driver_when_send_on_settled_session_does_noop():
    driver = create_mock_driver([ActionStep(action=_noop_action)])

    session = driver.start(make_prompt(), noop_observer())
    await session.outcome

    await session.send("late")

    assert driver.sessions[-1].calls == []


async def test_create_mock_driver_when_end_on_settled_session_does_noop():
    driver = create_mock_driver([ActionStep(action=_noop_action)])

    session = driver.start(make_prompt(), noop_observer())
    await session.outcome

    await session.end()

    assert driver.sessions[-1].calls == []


async def test_create_mock_driver_when_end_with_no_step_waiting_does_stop_at_next_boundary():
    reached: list[str] = []

    async def first() -> None:
        reached.append("first")

    async def second() -> None:
        reached.append("second")

    driver = create_mock_driver([ActionStep(action=first), ActionStep(action=second, delay_ms=50)])

    session = driver.start(make_prompt(), noop_observer())
    await asyncio.sleep(0.02)
    await session.end()
    outcome = await session.outcome

    assert reached == ["first"]
    assert outcome == SessionOutcome(reason="completed", cost_usd=0.0)


# ---------------------------------------------------------------------------
# sessions tracking
# ---------------------------------------------------------------------------


async def test_create_mock_driver_does_track_sessions():
    driver = create_mock_driver([ActionStep(action=_noop_action)])

    session1 = driver.start(make_prompt(), noop_observer())
    await session1.outcome
    session2 = driver.start(make_prompt(), noop_observer())
    await session2.outcome

    assert len(driver.sessions) == 2


async def test_create_mock_driver_when_interrupted_during_turn_end_does_settle_interrupted():
    driver = create_mock_driver([TurnEndStep()])

    session = driver.start(make_prompt(), noop_observer())
    await asyncio.sleep(0.05)
    await session.interrupt()
    outcome = await session.outcome

    assert outcome.reason == "interrupted"
