"""Behavioral tests for the mock driver test fixture.

``create_mock_driver`` builds a :class:`Driver` whose ``start`` runs a
caller-supplied script of steps on the running event loop. The upstream suite
drove these behaviors with fake timers; the asyncio port replaces them with
small real delays and coordinates ordering through ``asyncio.Event`` handshakes
so every test stays deterministic under ``pytest-randomly`` and
``pytest-xdist``. Timing is asserted only as loose lower bounds, never exact
wall-clock values.
"""

import asyncio
import time

import pytest

from gymrat_py.supervisor import SessionOutcome, TextDeltaEvent
from gymrat_py.supervisor.events import SessionEvent
from tests.supervisor._fixtures import collecting_observer, make_prompt, noop_observer
from tests.supervisor._mock_driver import ActionStep, CostStep, EmitStep, create_mock_driver


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
