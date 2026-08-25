"""Behavioral tests for the supervisor orchestrator.

``supervise`` runs a driver session under a wall-clock cap, an optional spend
cap, and a grace fallback that arms the driver's abort event when a cap fires
but the session keeps running. The upstream suite drove every timing behavior
with fake timers; the asyncio port uses tiny ``max_minutes``/``grace_ms`` values
and ``asyncio.Event`` handshakes so the session stays open only as long as a
test needs it. Timing is asserted as loose lower bounds, never exact values, to
stay deterministic under ``pytest-randomly`` and ``pytest-xdist``.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest

from gymrat_py.supervisor import (
    SessionOutcome,
    TextDeltaEvent,
    UsageUpdateEvent,
    supervise,
)
from gymrat_py.supervisor.driver import Driver, DriverSession, SessionPrompt
from gymrat_py.supervisor.events import SessionEvent, SessionObserver
from tests.supervisor._fixtures import (
    collecting_observer,
    make_launch,
    make_prompt,
    read_log_lines,
)
from tests.supervisor._mock_driver import ActionStep, CostStep, EmitStep, create_mock_driver


async def _noop_action() -> None:
    return None


_GUARD_MESSAGE = "this step must not run after a cap fires"


async def _guard_action() -> None:
    raise AssertionError(_GUARD_MESSAGE)


@dataclass
class _Box:
    """A mutable integer holder for counting side effects across closures."""

    value: int = 0


# ---------------------------------------------------------------------------
# test doubles
# ---------------------------------------------------------------------------


class _DelegatingSession:
    """Wraps a ``DriverSession``, delegating ``outcome`` and ``interrupt``."""

    def __init__(self, inner: DriverSession) -> None:
        self._inner = inner

    @property
    def outcome(self) -> Awaitable[SessionOutcome]:
        return self._inner.outcome

    async def interrupt(self) -> None:
        await self._inner.interrupt()


class _CountingSession(_DelegatingSession):
    """Counts ``interrupt`` calls before delegating them."""

    def __init__(self, inner: DriverSession, counter: _Box) -> None:
        super().__init__(inner)
        self._counter = counter

    @override
    async def interrupt(self) -> None:
        self._counter.value += 1
        await self._inner.interrupt()


class _ThrowingInterruptSession(_DelegatingSession):
    """Raises synchronously from ``interrupt`` to exercise the grace fallback."""

    @override
    async def interrupt(self) -> None:
        message = "interrupt exploded"
        raise RuntimeError(message)


class _WrapDriver:
    """Record the abort event and wrap the session an inner driver returns."""

    def __init__(
        self,
        inner: Driver,
        make_session: Callable[[DriverSession], DriverSession] = lambda s: s,
    ) -> None:
        self._inner = inner
        self._make_session = make_session
        self.captured_abort: asyncio.Event | None = None

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None = None,
    ) -> DriverSession:
        self.captured_abort = abort
        return self._make_session(self._inner.start(prompt, observer, abort))


class _FutureSession:
    """A session whose ``outcome`` is a caller-supplied future."""

    def __init__(self, outcome: Awaitable[SessionOutcome]) -> None:
        self._outcome = outcome

    @property
    def outcome(self) -> Awaitable[SessionOutcome]:
        return self._outcome

    async def interrupt(self) -> None:
        return None


class _SyncSpendDriver:
    """Emits a spend-cap-tripping usage update synchronously inside ``start``."""

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None = None,
    ) -> DriverSession:
        observer(UsageUpdateEvent(timestamp=1, cost_usd=5.0))
        loop = asyncio.get_running_loop()
        settled: asyncio.Future[SessionOutcome] = loop.create_future()
        settled.set_result(SessionOutcome(reason="interrupted", cost_usd=5.0))
        return _FutureSession(settled)


class _RejectingDriver:
    """Returns a session whose ``outcome`` rejects immediately."""

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


def _cap_events(events: list[SessionEvent]) -> list[SessionEvent]:
    return [event for event in events if event.type == "cap"]


# ---------------------------------------------------------------------------
# normal completion
# ---------------------------------------------------------------------------


async def test_supervise_when_session_completes_does_report_session_outcome_and_cost(
    tmp_path: Path,
):
    driver = create_mock_driver([CostStep(cost_usd=0.05), CostStep(cost_usd=0.12)])

    result = await supervise(
        driver,
        make_prompt(),
        max_minutes=10,
        log_path=tmp_path / "events.jsonl",
        launch=make_launch(),
    )

    assert result.ended_by == "session"
    assert result.outcome == SessionOutcome(reason="completed", cost_usd=0.12)
    assert result.cost_usd == 0.12
    assert result.duration_ms >= 0


async def test_supervise_when_session_runs_does_log_launch_first_then_events_in_order(
    tmp_path: Path,
):
    steps = [EmitStep(emit=TextDeltaEvent(timestamp=2000, chunk="hello")), CostStep(cost_usd=0.01)]
    driver = create_mock_driver(steps)
    log_path = tmp_path / "events.jsonl"

    await supervise(driver, make_prompt(), max_minutes=10, log_path=log_path, launch=make_launch())

    lines = read_log_lines(log_path)
    assert [line["type"] for line in lines[:3]] == ["launch", "text_delta", "usage_update"]


async def test_supervise_when_observer_given_does_forward_launch_first_and_session_events(
    tmp_path: Path,
):
    probe = collecting_observer()
    launch = make_launch()
    driver = create_mock_driver([CostStep(cost_usd=0.03)])

    await supervise(
        driver,
        make_prompt(),
        max_minutes=10,
        log_path=tmp_path / "events.jsonl",
        launch=launch,
        observer=probe.observer,
    )

    assert probe.events[0] == launch
    assert any(event.type == "usage_update" for event in probe.events)


# ---------------------------------------------------------------------------
# wall-clock cap
# ---------------------------------------------------------------------------


async def test_supervise_when_wall_clock_elapses_does_interrupt_and_report_wall_clock(
    tmp_path: Path,
):
    driver = create_mock_driver([CostStep(cost_usd=0.05, delay_ms=60_000)])

    result = await supervise(
        driver,
        make_prompt(),
        max_minutes=0.001,
        log_path=tmp_path / "events.jsonl",
        launch=make_launch(max_minutes=0.001),
    )

    assert result.ended_by == "wall-clock"
    assert result.outcome.reason == "interrupted"


async def test_supervise_when_wall_clock_elapses_does_emit_single_wall_clock_cap_event(
    tmp_path: Path,
):
    probe = collecting_observer()
    driver = create_mock_driver([CostStep(cost_usd=0.05, delay_ms=60_000)])

    await supervise(
        driver,
        make_prompt(),
        max_minutes=0.001,
        log_path=tmp_path / "events.jsonl",
        launch=make_launch(max_minutes=0.001),
        observer=probe.observer,
    )

    caps = _cap_events(probe.events)
    assert len(caps) == 1
    assert caps[0].cap == "wall-clock"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# grace fallback
# ---------------------------------------------------------------------------


async def test_supervise_when_grace_elapses_does_arm_abort_only_after_grace(tmp_path: Path):
    release = asyncio.Event()
    cap_seen = asyncio.Event()

    async def block() -> None:
        await release.wait()

    def observer(event: SessionEvent) -> None:
        if event.type == "cap":
            cap_seen.set()

    wrapper = _WrapDriver(create_mock_driver([ActionStep(action=block)]))
    grace_ms = 100

    async def run() -> object:
        return await supervise(
            wrapper,
            make_prompt(),
            max_minutes=0.001,
            log_path=tmp_path / "events.jsonl",
            launch=make_launch(max_minutes=0.001),
            observer=observer,
            grace_ms=grace_ms,
        )

    task = asyncio.create_task(run())
    await cap_seen.wait()
    captured = wrapper.captured_abort
    assert captured is not None

    cap_at = time.perf_counter()
    assert not captured.is_set()
    await asyncio.wait_for(captured.wait(), timeout=2.0)
    grace_elapsed_ms = (time.perf_counter() - cap_at) * 1000
    assert grace_elapsed_ms >= grace_ms * 0.5

    release.set()
    result = await asyncio.wait_for(task, timeout=2.0)
    assert result.ended_by == "wall-clock"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# spend cap
# ---------------------------------------------------------------------------


async def test_supervise_when_cost_reaches_max_usd_does_interrupt_and_report_spend_cap(
    tmp_path: Path,
):
    steps = [CostStep(cost_usd=0.05), CostStep(cost_usd=0.12), ActionStep(action=_guard_action)]
    driver = create_mock_driver(steps)

    result = await supervise(
        driver,
        make_prompt(),
        max_minutes=10,
        max_usd=0.1,
        log_path=tmp_path / "events.jsonl",
        launch=make_launch(max_usd=0.1),
    )

    assert result.ended_by == "spend-cap"
    assert result.outcome.reason == "interrupted"


async def test_supervise_when_cost_reaches_max_usd_does_emit_single_spend_cap_event(
    tmp_path: Path,
):
    probe = collecting_observer()
    driver = create_mock_driver([CostStep(cost_usd=0.05), CostStep(cost_usd=0.12)])

    await supervise(
        driver,
        make_prompt(),
        max_minutes=10,
        max_usd=0.1,
        log_path=tmp_path / "events.jsonl",
        launch=make_launch(max_usd=0.1),
        observer=probe.observer,
    )

    caps = _cap_events(probe.events)
    assert len(caps) == 1
    assert caps[0].cap == "spend-cap"  # type: ignore[attr-defined]


async def test_supervise_when_max_usd_none_does_not_enforce_cost(tmp_path: Path):
    driver = create_mock_driver([CostStep(cost_usd=5.0), CostStep(cost_usd=10.0)])

    result = await supervise(
        driver,
        make_prompt(),
        max_minutes=10,
        log_path=tmp_path / "events.jsonl",
        launch=make_launch(),
    )

    assert result.ended_by == "session"
    assert result.outcome.reason == "completed"
    assert result.cost_usd == 10.0


# ---------------------------------------------------------------------------
# cap racing
# ---------------------------------------------------------------------------


async def test_supervise_when_both_caps_could_fire_does_report_first_and_interrupt_once(
    tmp_path: Path,
):
    probe = collecting_observer()
    counter = _Box()
    inner = create_mock_driver(
        [CostStep(cost_usd=0.15), ActionStep(action=_noop_action, delay_ms=60_000)]
    )
    driver = _WrapDriver(inner, lambda session: _CountingSession(session, counter))

    result = await supervise(
        driver,
        make_prompt(),
        max_minutes=0.001,
        max_usd=0.1,
        log_path=tmp_path / "events.jsonl",
        launch=make_launch(max_minutes=0.001, max_usd=0.1),
        observer=probe.observer,
        grace_ms=50,
    )

    assert result.ended_by == "spend-cap"
    assert counter.value == 1
    assert len(_cap_events(probe.events)) == 1


async def test_supervise_when_spend_cap_trips_inside_start_does_defer_cap_and_report_spend_cap(
    tmp_path: Path,
):
    probe = collecting_observer()

    result = await supervise(
        _SyncSpendDriver(),
        make_prompt(),
        max_minutes=0.001,
        max_usd=1.0,
        log_path=tmp_path / "events.jsonl",
        launch=make_launch(max_minutes=0.001, max_usd=1.0),
        observer=probe.observer,
    )

    caps = _cap_events(probe.events)
    assert result.ended_by == "spend-cap"
    assert len(caps) == 1
    assert caps[0].cap == "spend-cap"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# error outcome
# ---------------------------------------------------------------------------


def _error_steps() -> list[object]:
    boom_message = "kaboom"

    async def boom() -> None:
        raise RuntimeError(boom_message)

    return [
        EmitStep(emit=TextDeltaEvent(timestamp=2000, chunk="partial output")),
        CostStep(cost_usd=0.02),
        ActionStep(action=boom),
    ]


async def test_supervise_when_driver_errors_does_surface_message_and_end_by_session(
    tmp_path: Path,
):
    driver = create_mock_driver(_error_steps())  # type: ignore[arg-type]

    result = await supervise(
        driver,
        make_prompt(),
        max_minutes=10,
        log_path=tmp_path / "events.jsonl",
        launch=make_launch(),
    )

    assert result.outcome.reason == "error"
    assert result.outcome.message == "kaboom"
    assert result.ended_by == "session"


async def test_supervise_when_driver_errors_does_log_launch_and_events_up_to_failure(
    tmp_path: Path,
):
    driver = create_mock_driver(_error_steps())  # type: ignore[arg-type]
    log_path = tmp_path / "events.jsonl"

    await supervise(driver, make_prompt(), max_minutes=10, log_path=log_path, launch=make_launch())

    types = [line["type"] for line in read_log_lines(log_path)]
    assert types[0] == "launch"
    assert "text_delta" in types
    assert "usage_update" in types


# ---------------------------------------------------------------------------
# cap robustness
# ---------------------------------------------------------------------------


async def test_supervise_when_observer_raises_does_still_fire_spend_cap(tmp_path: Path):
    observer_message = "observer boom"

    def throwing(event: SessionEvent) -> None:
        if event.type == "usage_update":
            raise RuntimeError(observer_message)

    driver = create_mock_driver([CostStep(cost_usd=0.5)])

    result = await supervise(
        driver,
        make_prompt(),
        max_minutes=10,
        max_usd=0.1,
        log_path=tmp_path / "events.jsonl",
        launch=make_launch(max_usd=0.1),
        observer=throwing,
    )

    assert result.ended_by == "spend-cap"


async def test_supervise_when_interrupt_throws_does_recover_via_grace(tmp_path: Path):
    inner = create_mock_driver([CostStep(cost_usd=0.01, delay_ms=60_000)])
    driver = _WrapDriver(inner, _ThrowingInterruptSession)

    result = await supervise(
        driver,
        make_prompt(),
        max_minutes=0.001,
        log_path=tmp_path / "events.jsonl",
        launch=make_launch(max_minutes=0.001),
        grace_ms=50,
    )

    assert result.ended_by == "wall-clock"


async def test_supervise_when_outcome_rejects_does_propagate_and_not_cap(tmp_path: Path):
    probe = collecting_observer()

    with pytest.raises(RuntimeError, match="session crashed"):
        await supervise(
            _RejectingDriver(),
            make_prompt(),
            max_minutes=5,
            log_path=tmp_path / "events.jsonl",
            launch=make_launch(),
            observer=probe.observer,
        )

    assert _cap_events(probe.events) == []
