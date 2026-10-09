"""Behavioral tests for the supervisor orchestrator.

``supervise`` runs a driver session under a wall-clock cap, an optional spend
cap, and a grace fallback that arms the driver's abort event when a cap fires
but the session keeps running. Tests use tiny ``max_minutes``/``grace_ms`` values
and ``asyncio.Event`` handshakes so the session stays open only as long as a
test needs it. Timing is asserted as loose lower bounds, never exact values, to
stay deterministic under ``pytest-randomly`` and ``pytest-xdist``.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Coroutine
from typing import override

import pytest

from gymrat.supervisor.driver import DriverSession, SessionOutcome, SessionPrompt
from gymrat.supervisor.events import (
    CapEvent,
    SessionEvent,
    SessionObserver,
    TextDeltaEvent,
    UsageUpdateEvent,
)
from tests._clock import install_monotonic_clock
from tests.supervisor._fixtures import (
    DelegatingSession,
    SlowEndSession,
    SupervisorClock,
    _supervise,
    _WrapDriver,
    blocked_step,
    collecting_observer,
    event_log_markers,
    events_of,
    make_launch,
    raising_observer,
)
from tests.supervisor._mock_driver import (
    ActionStep,
    CostStep,
    EmitStep,
    MockStep,
    TurnEndStep,
    create_mock_driver,
)

# ---------------------------------------------------------------------------
# test doubles
# ---------------------------------------------------------------------------


class _ThrowingInterruptSession(DelegatingSession):
    """Raises from ``interrupt`` before any coroutine exists, to exercise the grace fallback."""

    @override
    def interrupt(self) -> Coroutine[object, object, None]:
        message = "interrupt exploded"
        raise RuntimeError(message)


class _RejectingInterruptSession(DelegatingSession):
    """Hands back an ``interrupt`` coroutine that fails once the supervisor awaits it."""

    @override
    async def interrupt(self) -> None:
        message = "interrupt exploded"
        raise RuntimeError(message)


class _FutureSession:
    """A session whose ``outcome`` is a caller-supplied future."""

    def __init__(self, outcome: Awaitable[SessionOutcome]) -> None:
        self._outcome = outcome

    @property
    def outcome(self) -> Awaitable[SessionOutcome]:
        return self._outcome

    async def interrupt(self) -> None:
        return None

    async def send(self, text: str) -> None:
        return None

    async def end(self) -> None:
        return None


class _RejectingDriver:
    """Exercise the supervisor's handling of a driver that errors before any turn runs."""

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None = None,
    ) -> DriverSession:
        loop = asyncio.get_running_loop()
        settled: asyncio.Future[SessionOutcome] = loop.create_future()
        settled.set_exception(RuntimeError("session crashed"))
        return _FutureSession(settled)


# ---------------------------------------------------------------------------
# normal completion
# ---------------------------------------------------------------------------


async def test_supervise_when_session_completes_does_report_outcome_with_events_in_order(
    root: str,
):
    probe = collecting_observer()
    launch = make_launch()
    text = TextDeltaEvent(at=2_000_000_000_000, chunk="hello")
    driver = create_mock_driver([
        EmitStep(emit=text),
        CostStep(cost_usd=0.05),
        CostStep(cost_usd=0.12),
    ])

    result = await _supervise(
        root, driver, launch=launch, observer=probe.observer, is_lock_held=None
    )

    assert (result.ended_by, result.outcome) == (
        "session",
        SessionOutcome(reason="completed", cost_usd=0.12),
    )
    assert event_log_markers(root) == [
        "launch",
        "text_delta",
        "usage_update",
        "usage_update",
    ]
    assert probe.events[:2] == [launch, text]
    assert [event.type for event in probe.events[2:]] == ["usage_update", "usage_update"]


async def test_supervise_when_session_spans_time_does_report_duration_from_monotonic_clock(
    root: str, monkeypatch: pytest.MonkeyPatch
):
    clock = install_monotonic_clock(monkeypatch)

    async def spend_250_ms() -> None:
        clock.tick(250.0)

    driver = create_mock_driver([ActionStep(action=spend_250_ms), CostStep(cost_usd=0.05)])

    result = await _supervise(root, driver, is_lock_held=None)

    assert result.duration_ms == 250


# ---------------------------------------------------------------------------
# wall-clock cap
# ---------------------------------------------------------------------------


async def test_supervise_when_wall_clock_caps_a_long_session_does_interrupt_with_one_wall_clock_cap(
    root: str,
):
    probe = collecting_observer()
    # A single step delayed far past the cap, so the wall-clock cap always wins.
    driver = create_mock_driver([blocked_step()])

    result = await _supervise(
        root, driver, max_minutes=0.001, observer=probe.observer, grace_ms=50, is_lock_held=None
    )

    assert (result.ended_by, result.outcome.reason) == ("wall-clock", "interrupted")
    assert [(cap.cap, cap.action) for cap in events_of(probe.events, CapEvent)] == [
        ("wall-clock", "interrupting")
    ]


# ---------------------------------------------------------------------------
# grace fallback
# ---------------------------------------------------------------------------


def _abort_bound_driver() -> tuple[_WrapDriver, list[float]]:
    """A driver whose one step ignores ``interrupt`` and waits for the abort event.

    Returns:
        The driver and the list it appends the ``perf_counter`` reading to once
        the abort fires.
    """
    aborted_at: list[float] = []

    async def wait_for_abort() -> None:
        await wrapper.abort.wait()
        aborted_at.append(time.perf_counter())

    wrapper = _WrapDriver(create_mock_driver([ActionStep(action=wait_for_abort)]))
    return wrapper, aborted_at


async def test_supervise_when_grace_elapses_does_arm_abort_only_after_grace(root: str):
    grace_ms = 100
    wrapper, aborted_at = _abort_bound_driver()
    at_cap: list[tuple[float, bool]] = []

    def observer(event: SessionEvent) -> None:
        if isinstance(event, CapEvent):
            at_cap.append((time.perf_counter(), wrapper.abort.is_set()))

    result = await asyncio.wait_for(
        _supervise(
            root,
            wrapper,
            max_minutes=0.001,
            observer=observer,
            grace_ms=grace_ms,
            is_lock_held=None,
        ),
        timeout=2.0,
    )

    [(cap_at, armed_at_cap)] = at_cap
    assert armed_at_cap is False
    assert (aborted_at[0] - cap_at) * 1000 >= grace_ms * 0.5
    assert result.ended_by == "wall-clock"


# ---------------------------------------------------------------------------
# spend cap
# ---------------------------------------------------------------------------


async def test_supervise_when_spend_cap_trips_at_turn_end_does_report_spend_cap(
    root: str,
):
    probe = collecting_observer()
    driver = create_mock_driver([
        CostStep(cost_usd=5.0),
        TurnEndStep(cost_usd=5.0),
    ])

    result = await _supervise(
        root,
        driver,
        max_usd=1.0,
        launch=make_launch(max_usd=None),
        observer=probe.observer,
    )

    caps = events_of(probe.events, CapEvent)
    assert result.ended_by == "spend-cap"
    assert len(caps) == 1
    assert caps[0].cap == "spend-cap"
    assert caps[0].action == "ending"
    assert result.outcome.cost_usd == 5.0
    types = event_log_markers(root)
    assert types.index("usage_update") < types.index("cap")


async def test_supervise_when_wall_clock_passes_after_spend_cap_fired_does_emit_one_cap(
    root: str, supervisor_clock: SupervisorClock
):
    probe = collecting_observer()
    driver = _WrapDriver(
        create_mock_driver([TurnEndStep(cost_usd=5.0)]),
        lambda session, _abort: SlowEndSession(session, 300),
    )

    result = await _supervise(
        root,
        driver,
        observer=supervisor_clock.jump_on(CapEvent, probe.observer),
        max_usd=1.0,
        deadline_ms=supervisor_clock.deadline_ms,
    )

    caps = [(cap.cap, cap.action) for cap in events_of(probe.events, CapEvent)]
    assert (result.ended_by, caps) == ("spend-cap", [("spend-cap", "ending")])


# ---------------------------------------------------------------------------
# error outcome
# ---------------------------------------------------------------------------


def _error_steps() -> list[MockStep]:
    boom_message = "kaboom"

    async def boom() -> None:
        raise RuntimeError(boom_message)

    return [
        EmitStep(emit=TextDeltaEvent(at=2_000_000_000_000, chunk="partial output")),
        CostStep(cost_usd=0.02),
        ActionStep(action=boom),
    ]


async def test_supervise_when_driver_errors_does_report_error_with_events_logged_up_to_failure(
    root: str,
):
    driver = create_mock_driver(_error_steps())

    result = await _supervise(root, driver, is_lock_held=None)

    assert result.outcome.reason == "error"
    assert result.outcome.message == "kaboom"
    assert result.ended_by == "session"
    types = event_log_markers(root)
    assert types[0] == "launch"
    assert "text_delta" in types
    assert "usage_update" in types


# ---------------------------------------------------------------------------
# cap robustness: a failing interrupt or session outcome
# ---------------------------------------------------------------------------


#: Far under the blocked step's delay, so only the grace abort can end the run in time.
_GRACE_RECOVERY_TIMEOUT_S = 5


@pytest.mark.parametrize(
    "make_session",
    [
        pytest.param(_ThrowingInterruptSession, id="raises-synchronously"),
        pytest.param(_RejectingInterruptSession, id="coroutine-fails"),
    ],
)
async def test_supervise_when_interrupt_fails_does_recover_via_grace_with_a_warning(
    root: str,
    capsys: pytest.CaptureFixture[str],
    make_session: Callable[[DriverSession], DriverSession],
):
    inner = create_mock_driver([blocked_step()])
    driver = _WrapDriver(inner, lambda session, _abort: make_session(session))

    async with asyncio.timeout(_GRACE_RECOVERY_TIMEOUT_S):
        result = await _supervise(
            root,
            driver,
            max_minutes=0.001,
            grace_ms=50,
            is_lock_held=None,
        )
    await asyncio.sleep(0)

    assert result.ended_by == "wall-clock"
    assert result.outcome.reason == "interrupted"
    assert "session interrupt failed: interrupt exploded" in capsys.readouterr().err.splitlines()


async def test_supervise_when_outcome_rejects_does_propagate_rejection(root: str):
    probe = collecting_observer()

    with pytest.raises(RuntimeError, match="session crashed"):
        await _supervise(
            root,
            _RejectingDriver(),
            max_minutes=5,
            observer=probe.observer,
            is_lock_held=None,
        )

    assert events_of(probe.events, CapEvent) == []


# ---------------------------------------------------------------------------
# observer resilience: the cap fires even when an observer raises
# ---------------------------------------------------------------------------


async def test_supervise_when_observer_raises_does_still_fire_spend_cap(root: str):
    observer_message = "observer boom"
    throwing = raising_observer(observer_message, on=UsageUpdateEvent)
    driver = create_mock_driver([CostStep(cost_usd=0.5), TurnEndStep(cost_usd=0.5)])

    with pytest.warns(RuntimeWarning, match=observer_message):
        result = await _supervise(root, driver, max_usd=0.1, observer=throwing)

    assert result.ended_by == "spend-cap"


async def test_supervise_when_observer_raises_on_cap_event_does_still_arm_grace(
    root: str,
):
    wrapper, aborted_at = _abort_bound_driver()
    failing_observer = raising_observer("observer explodes on cap", on=CapEvent)

    with pytest.warns(RuntimeWarning, match="observer explodes on cap"):
        result = await asyncio.wait_for(
            _supervise(
                root,
                wrapper,
                max_minutes=0.001,
                observer=failing_observer,
                grace_ms=100,
                is_lock_held=None,
            ),
            timeout=2.0,
        )

    assert len(aborted_at) == 1
    assert result.ended_by == "wall-clock"


# ---------------------------------------------------------------------------
# interrupt task teardown
# ---------------------------------------------------------------------------


class _SlowInterruptSession(DelegatingSession):
    """An interrupt that blocks until cancelled, to detect leaked tasks."""

    def __init__(self, inner: DriverSession) -> None:
        super().__init__(inner)
        self.interrupt_cancelled = asyncio.Event()

    @override
    async def interrupt(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.interrupt_cancelled.set()
            raise


async def test_supervise_when_session_ends_does_cancel_interrupt_task(root: str):
    slow_session: _SlowInterruptSession | None = None

    def wrap(inner: DriverSession, _abort: asyncio.Event) -> DriverSession:
        nonlocal slow_session
        slow_session = _SlowInterruptSession(inner)
        return slow_session

    inner = create_mock_driver([blocked_step()])
    driver = _WrapDriver(inner, wrap)

    result = await _supervise(
        root,
        driver,
        max_minutes=0.001,
        grace_ms=50,
        is_lock_held=None,
    )
    # Give the event loop a tick so the CancelledError propagates.
    await asyncio.sleep(0)

    assert result.ended_by == "wall-clock"
    assert slow_session is not None
    assert slow_session.interrupt_cancelled.is_set()


# ---------------------------------------------------------------------------
# wall-clock poll
# ---------------------------------------------------------------------------


async def test_supervise_when_wall_clock_fires_via_poll_does_end_at_deadline(
    root: str, supervisor_clock: SupervisorClock
):
    # Only a poll that rereads the clock sees the jump; a single sleep to the
    # deadline would leave the run waiting a real minute.
    driver = create_mock_driver([
        supervisor_clock.jump_step(supervisor_clock.deadline_ms),
        blocked_step(),
    ])

    result = await _supervise(
        root,
        driver,
        deadline_ms=supervisor_clock.deadline_ms,
        wall_clock_poll_ms=10,
        is_lock_held=None,
    )

    assert result.ended_by == "wall-clock"


# ---------------------------------------------------------------------------
# _spawn exception reporting
# ---------------------------------------------------------------------------


class _EndThenRaiseSession(DelegatingSession):
    """Delegates ``end()`` to the inner session, then raises.

    The inner call releases the mock's turn gate so the session settles
    normally; the post-call raise is the exception that ``_spawn``'s
    done-callback should surface via ``warn_to_stderr``.
    """

    @override
    async def end(self) -> None:
        await self._inner.end()
        msg = "end exploded"
        raise RuntimeError(msg)


async def test_supervise_when_spawned_end_raises_does_warn_to_stderr(
    root: str,
    capsys: pytest.CaptureFixture[str],
):
    inner = create_mock_driver([TurnEndStep(cost_usd=0.5)])
    driver = _WrapDriver(inner, lambda session, _abort: _EndThenRaiseSession(session))

    result = await _supervise(root, driver, max_usd=0.1)
    # Yield once so the background end's done-callback reports its failure.
    await asyncio.sleep(0)

    assert result.ended_by == "spend-cap"
    assert "background task failed: end exploded" in capsys.readouterr().err.splitlines()
