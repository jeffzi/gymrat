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
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import cast, override

import pytest

from gymrat.supervisor.driver import Driver, DriverSession
from gymrat.supervisor.events import (
    CapEvent,
    SessionEvent,
    TurnEndEvent,
)
from gymrat.supervisor.supervise import SupervisionResult
from tests._clock import install_monotonic_clock
from tests.supervisor._fixtures import (
    ActionStep,
    CostStep,
    DelegatingSession,
    MockStep,
    SlowEndSession,
    SupervisorClock,
    TurnEndStep,
    WrapDriver,
    blocked_step,
    cap_actions,
    collecting_observer,
    create_mock_driver,
    run_supervised,
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


async def test_supervise_when_session_spans_time_does_report_duration_from_monotonic_clock(
    root: str, monkeypatch: pytest.MonkeyPatch
):
    clock = install_monotonic_clock(monkeypatch)

    async def spend_250_ms() -> None:
        clock.tick(250.0)

    driver = create_mock_driver([ActionStep(action=spend_250_ms), CostStep(cost_usd=0.05)])

    result = await run_supervised(root, driver)

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

    result = await run_supervised(
        root, driver, max_minutes=0.001, observer=probe.observer, grace_ms=50
    )

    assert (result.ended_by, result.outcome.reason) == ("wall-clock", "interrupted")
    assert cap_actions(probe.events) == [("wall-clock", "interrupting")]


# ---------------------------------------------------------------------------
# grace fallback
# ---------------------------------------------------------------------------


def _abort_bound_driver() -> tuple[WrapDriver, list[float]]:
    """A driver whose one step ignores ``interrupt`` and waits for the abort event.

    Returns:
        The driver and the list it appends the ``perf_counter`` reading to once
        the abort fires.
    """
    aborted_at: list[float] = []

    async def wait_for_abort() -> None:
        await wrapper.abort.wait()
        aborted_at.append(time.perf_counter())

    wrapper = WrapDriver(create_mock_driver([ActionStep(action=wait_for_abort)]))
    return wrapper, aborted_at


async def test_supervise_when_grace_elapses_does_arm_abort_only_after_grace(root: str):
    grace_ms = 100
    wrapper, aborted_at = _abort_bound_driver()
    at_cap: list[tuple[float, bool]] = []

    def observer(event: SessionEvent) -> None:
        if isinstance(event, CapEvent):
            at_cap.append((time.perf_counter(), wrapper.abort.is_set()))

    result = await asyncio.wait_for(
        run_supervised(
            root,
            wrapper,
            max_minutes=0.001,
            observer=observer,
            grace_ms=grace_ms,
        ),
        timeout=2.0,
    )

    [(cap_at, armed_at_cap)] = at_cap
    assert armed_at_cap is False
    assert (aborted_at[0] - cap_at) * 1000 >= grace_ms * 0.5
    assert result.ended_by == "wall-clock"


class _AbortSettledSession(DelegatingSession):
    """A session that ignores ``end`` and settles only once the abort event is set.

    Models a backend that stays up after a polite end. ``settling`` is the task
    that interrupts the inner session on abort, for the test to await.

    Args:
        inner: The session every call but ``end`` is delegated to.
        abort: The event whose setting interrupts ``inner`` and settles the session.
    """

    def __init__(self, inner: DriverSession, abort: asyncio.Event) -> None:
        super().__init__(inner)
        self.settling = asyncio.ensure_future(self._settle_on_abort(abort))

    async def _settle_on_abort(self, abort: asyncio.Event) -> None:
        await abort.wait()
        await self._inner.interrupt()

    @override
    async def end(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _BoundaryEndCase:
    """A script that stops at a turn end, the knobs that end the session there, and the ending."""

    script: Callable[[str], list[MockStep]]
    ended_by: str
    max_usd: float | None = None
    deadline_after_turn_end: bool = False
    settle_window_ms: int = 0


async def _supervise_boundary_case(
    root: str, driver: Driver, case: _BoundaryEndCase, clock: SupervisorClock
) -> SupervisionResult:
    """Supervise ``driver`` under ``case``'s caps, its deadline passing only after a turn end.

    Args:
        root: The repository the session runs in.
        driver: The agent driver to supervise.
        case: The script's caps and settle window.
        clock: The supervisor's wall clock, moved to its deadline on a turn end.

    Returns:
        How the supervised session ended.
    """
    return await run_supervised(
        root,
        driver,
        grace_ms=50,
        max_usd=case.max_usd,
        deadline_ms=clock.deadline_ms,
        observer=clock.jump_on(TurnEndEvent) if case.deadline_after_turn_end else None,
        settle_window_ms=case.settle_window_ms,
    )


_SPEND_CAP_AT_SETTLE = _BoundaryEndCase(
    script=lambda _root: [TurnEndStep(cost_usd=5.0)],
    ended_by="spend-cap",
    max_usd=1.0,
)

_BOUNDARY_END_CASES = [
    pytest.param(
        _BoundaryEndCase(
            script=lambda _root: [TurnEndStep(cost_usd=0.01)],
            ended_by="wall-clock",
            deadline_after_turn_end=True,
            settle_window_ms=10_000,
        ),
        id="wall-clock-while-idle",
    ),
    pytest.param(_SPEND_CAP_AT_SETTLE, id="spend-cap-at-settle"),
]


@pytest.mark.parametrize("case", _BOUNDARY_END_CASES)
async def test_supervise_when_driver_ignores_end_at_turn_boundary_does_arm_abort_after_grace(
    root: str, case: _BoundaryEndCase, supervisor_clock: SupervisorClock
):
    driver = WrapDriver(create_mock_driver(case.script(root)), _AbortSettledSession)

    result = await asyncio.wait_for(
        _supervise_boundary_case(root, driver, case, supervisor_clock), timeout=2.0
    )
    await cast("_AbortSettledSession", driver.session).settling

    assert driver.abort.is_set()
    assert result.ended_by == case.ended_by


async def test_supervise_when_driver_settles_on_end_at_turn_boundary_does_not_abort(
    root: str, supervisor_clock: SupervisorClock
):
    case = _SPEND_CAP_AT_SETTLE
    driver = WrapDriver(create_mock_driver(case.script(root)))

    result = await _supervise_boundary_case(root, driver, case, supervisor_clock)
    # Outlast the grace period, so a grace timer left armed would set the abort.
    await asyncio.sleep(0.15)

    assert not driver.abort.is_set()
    assert result.ended_by == case.ended_by
    assert asyncio.all_tasks() == {asyncio.current_task()}


# ---------------------------------------------------------------------------
# spend cap
# ---------------------------------------------------------------------------


async def test_supervise_when_wall_clock_passes_after_spend_cap_fired_does_emit_one_cap(
    root: str, supervisor_clock: SupervisorClock
):
    probe = collecting_observer()
    driver = WrapDriver(
        create_mock_driver([TurnEndStep(cost_usd=5.0)]),
        lambda session, _abort: SlowEndSession(session, 300),
    )

    result = await run_supervised(
        root,
        driver,
        observer=supervisor_clock.jump_on(CapEvent, probe.observer),
        max_usd=1.0,
        deadline_ms=supervisor_clock.deadline_ms,
    )

    assert (result.ended_by, cap_actions(probe.events)) == ("spend-cap", [("spend-cap", "ending")])


# ---------------------------------------------------------------------------
# cap robustness: a failing interrupt
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
    driver = WrapDriver(inner, lambda session, _abort: make_session(session))

    async with asyncio.timeout(_GRACE_RECOVERY_TIMEOUT_S):
        result = await run_supervised(
            root,
            driver,
            max_minutes=0.001,
            grace_ms=50,
        )
    await asyncio.sleep(0)

    assert result.ended_by == "wall-clock"
    assert result.outcome.reason == "interrupted"
    assert "interrupt exploded" in capsys.readouterr().err


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
    driver = WrapDriver(inner, wrap)

    result = await run_supervised(
        root,
        driver,
        max_minutes=0.001,
        grace_ms=50,
    )
    # Give the event loop a tick so the CancelledError propagates.
    await asyncio.sleep(0)

    assert result.ended_by == "wall-clock"
    assert slow_session is not None
    assert slow_session.interrupt_cancelled.is_set()


# ---------------------------------------------------------------------------
# an exception from a spawned session call surfaces on stderr
# ---------------------------------------------------------------------------


class _EndThenRaiseSession(DelegatingSession):
    """Delegates ``end()`` to the inner session, then raises.

    The inner call releases the mock's turn gate so the session settles
    normally; the post-call raise fails a session call the supervisor spawned
    in the background, which it must surface through ``warn_to_stderr``.
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
    driver = WrapDriver(inner, lambda session, _abort: _EndThenRaiseSession(session))

    result = await run_supervised(root, driver, max_usd=0.1)
    # Yield once so the background end's done-callback reports its failure.
    await asyncio.sleep(0)

    assert result.ended_by == "spend-cap"
    assert "end exploded" in capsys.readouterr().err
