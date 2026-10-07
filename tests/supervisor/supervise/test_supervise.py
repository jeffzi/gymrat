"""Behavioral tests for the supervisor orchestrator.

``supervise`` runs a driver session under a wall-clock cap, an optional spend
cap, and a grace fallback that arms the driver's abort event when a cap fires
but the session keeps running. Tests use tiny ``max_minutes``/``grace_ms`` values
and ``asyncio.Event`` handshakes so the session stays open only as long as a
test needs it. Timing is asserted as loose lower bounds, never exact values, to
stay deterministic under ``pytest-randomly`` and ``pytest-xdist``.
"""

import asyncio
import itertools
import sys
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest

from gymrat.session.paths import lockfile_path
from gymrat.supervisor.driver import DriverSession, SessionOutcome, SessionPrompt
from gymrat.supervisor.events import CapEvent, SessionEvent, SessionObserver, TextDeltaEvent
from gymrat.supervisor.supervise import SupervisionResult, supervise
from tests.supervisor._fixtures import (
    DelegatingSession,
    _supervise,
    _WrapDriver,
    collecting_observer,
    events_of,
    make_context,
    make_launch,
    make_prompt,
    read_log_lines,
    wait_for_event_or_task,
)
from tests.supervisor._mock_driver import (
    ActionStep,
    CostStep,
    EmitStep,
    MockStep,
    TurnEndStep,
    create_mock_driver,
)


@dataclass
class _Box:
    """A mutable integer holder for counting side effects across closures."""

    value: int = 0


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


async def test_supervise_when_session_completes_does_report_outcome(
    tmp_path: Path,
):
    driver = create_mock_driver([CostStep(cost_usd=0.05), CostStep(cost_usd=0.12)])

    result = await _supervise(str(tmp_path / "repo"), driver, is_lock_held=None)

    assert result.ended_by == "session"
    assert result.outcome == SessionOutcome(reason="completed", cost_usd=0.12)
    assert result.outcome.cost_usd == 0.12


async def test_supervise_when_clock_faked_does_report_duration_from_monotonic_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ticks = itertools.count(start=1000.0, step=250.0)
    monkeypatch.setattr("gymrat.clock.monotonic_ms", lambda: next(ticks))
    driver = create_mock_driver([CostStep(cost_usd=0.05)])

    result = await _supervise(str(tmp_path / "repo"), driver, is_lock_held=None)

    assert result.duration_ms == 250


async def test_supervise_when_session_runs_does_log_launch_first_then_events_in_order(
    tmp_path: Path,
):
    steps = [
        EmitStep(emit=TextDeltaEvent(at=2_000_000_000_000, chunk="hello")),
        CostStep(cost_usd=0.01),
    ]
    driver = create_mock_driver(steps)
    log_path = tmp_path / "events.jsonl"

    await _supervise(str(tmp_path / "repo"), driver, is_lock_held=None)

    lines = read_log_lines(log_path)
    assert [line["type"] for line in lines[:3]] == ["launch", "text_delta", "usage_update"]


async def test_supervise_when_observer_given_does_forward_events_in_order(
    tmp_path: Path,
):
    probe = collecting_observer()
    launch = make_launch()
    driver = create_mock_driver([CostStep(cost_usd=0.03)])

    await _supervise(
        str(tmp_path / "repo"), driver, launch=launch, observer=probe.observer, is_lock_held=None
    )

    assert probe.events[0] == launch
    assert any(event.type == "usage_update" for event in probe.events)


# ---------------------------------------------------------------------------
# wall-clock cap
# ---------------------------------------------------------------------------


async def test_supervise_when_wall_clock_caps_a_long_session_does_interrupt_with_one_wall_clock_cap(
    tmp_path: Path,
):
    root = str(tmp_path)
    probe = collecting_observer()
    # A single step delayed far past the cap, so the wall-clock cap always wins.
    driver = create_mock_driver([CostStep(cost_usd=0.01, delay_ms=60_000)])
    log_path = tmp_path / "supervisor-events.jsonl"

    result = await supervise(
        driver=driver,
        prompt=make_prompt(cwd=root),
        context=make_context(
            root=root, lock_path=lockfile_path(root), max_minutes=0.001, log_path=str(log_path)
        ),
        launch=make_launch(max_minutes=0.001),
        observer=probe.observer,
        grace_ms=50,
    )

    assert (result.ended_by, result.outcome.reason) == ("wall-clock", "interrupted")
    assert [(cap.cap, cap.action) for cap in events_of(probe.events, CapEvent)] == [
        ("wall-clock", "interrupting")
    ]
    cap_lines = [line for line in read_log_lines(log_path) if line["type"] == "cap"]
    assert [line["cap"] for line in cap_lines] == ["wall-clock"]


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

    async def run() -> SupervisionResult:
        return await _supervise(
            str(tmp_path / "repo"),
            wrapper,
            max_minutes=0.001,
            launch=make_launch(max_minutes=0.001),
            observer=observer,
            grace_ms=grace_ms,
            is_lock_held=None,
        )

    task = asyncio.create_task(run())
    await wait_for_event_or_task(cap_seen, task)
    captured = wrapper.captured_abort
    if captured is None:
        pytest.fail("the driver never started, so no abort event was captured")

    cap_at = time.perf_counter()
    armed_at_cap = captured.is_set()
    await asyncio.wait_for(captured.wait(), timeout=2.0)
    grace_elapsed_ms = (time.perf_counter() - cap_at) * 1000
    release.set()
    result = await asyncio.wait_for(task, timeout=2.0)

    assert armed_at_cap is False
    assert grace_elapsed_ms >= grace_ms * 0.5
    assert result.ended_by == "wall-clock"


async def test_supervise_when_session_settles_within_grace_does_cancel_grace_timer(
    tmp_path: Path,
):
    wrapper = _WrapDriver(create_mock_driver([CostStep(cost_usd=0.01, delay_ms=60_000)]))
    grace_ms = 50

    result = await _supervise(
        str(tmp_path / "repo"),
        wrapper,
        max_minutes=0.001,
        launch=make_launch(max_minutes=0.001),
        grace_ms=grace_ms,
        is_lock_held=None,
    )

    await asyncio.sleep(grace_ms * 3 / 1000)

    assert result.ended_by == "wall-clock"
    assert wrapper.captured_abort is not None
    assert not wrapper.captured_abort.is_set()
    assert asyncio.all_tasks() == {asyncio.current_task()}


# ---------------------------------------------------------------------------
# spend cap
# ---------------------------------------------------------------------------


async def test_supervise_when_spend_cap_trips_at_turn_end_does_report_spend_cap(
    tmp_path: Path,
):
    probe = collecting_observer()
    driver = create_mock_driver([
        CostStep(cost_usd=5.0),
        TurnEndStep(cost_usd=5.0),
    ])

    result = await _supervise(
        str(tmp_path / "repo"),
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
    types = [line["type"] for line in read_log_lines(tmp_path / "events.jsonl")]
    assert types.index("usage_update") < types.index("cap")


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
    tmp_path: Path,
):
    driver = create_mock_driver(_error_steps())

    result = await _supervise(str(tmp_path / "repo"), driver, is_lock_held=None)

    assert result.outcome.reason == "error"
    assert result.outcome.message == "kaboom"
    assert result.ended_by == "session"
    types = [line["type"] for line in read_log_lines(tmp_path / "events.jsonl")]
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

    driver = create_mock_driver([TurnEndStep(cost_usd=0.5)])

    result = await _supervise(str(tmp_path / "repo"), driver, max_usd=0.1, observer=throwing)

    assert result.ended_by == "spend-cap"


#: Far under the mock step's 60 s delay, so only the grace abort can end the run in time.
_GRACE_RECOVERY_TIMEOUT_S = 5


@pytest.mark.parametrize(
    "make_session",
    [
        pytest.param(_ThrowingInterruptSession, id="raises-synchronously"),
        pytest.param(_RejectingInterruptSession, id="coroutine-fails"),
    ],
)
async def test_supervise_when_interrupt_fails_does_recover_via_grace_with_a_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    make_session: Callable[[DriverSession], DriverSession],
):
    inner = create_mock_driver([CostStep(cost_usd=0.01, delay_ms=60_000)])
    driver = _WrapDriver(inner, make_session)

    async with asyncio.timeout(_GRACE_RECOVERY_TIMEOUT_S):
        result = await _supervise(
            str(tmp_path / "repo"),
            driver,
            max_minutes=0.001,
            launch=make_launch(max_minutes=0.001),
            grace_ms=50,
            is_lock_held=None,
        )
    await asyncio.sleep(0)

    assert result.ended_by == "wall-clock"
    assert result.outcome.reason == "interrupted"
    assert "session interrupt failed: interrupt exploded" in capsys.readouterr().err.splitlines()


async def test_supervise_when_outcome_rejects_does_propagate_rejection(tmp_path: Path):
    probe = collecting_observer()

    with pytest.raises(RuntimeError, match="session crashed"):
        await _supervise(
            str(tmp_path / "repo"),
            _RejectingDriver(),
            max_minutes=5,
            observer=probe.observer,
            is_lock_held=None,
        )

    assert events_of(probe.events, CapEvent) == []


# ---------------------------------------------------------------------------
# observer resilience: the cap fires even when an observer raises
# ---------------------------------------------------------------------------


async def test_supervise_when_observer_raises_on_cap_event_does_still_arm_grace(
    tmp_path: Path,
):
    cap_seen = asyncio.Event()
    release = asyncio.Event()

    def failing_observer(event: SessionEvent) -> None:
        if event.type == "cap":
            cap_seen.set()
            msg = "observer explodes on cap"
            raise RuntimeError(msg)

    async def block() -> None:
        await release.wait()

    wrapper = _WrapDriver(create_mock_driver([ActionStep(action=block)]))
    grace_ms = 100

    async def run() -> SupervisionResult:
        return await _supervise(
            str(tmp_path / "repo"),
            wrapper,
            max_minutes=0.001,
            launch=make_launch(max_minutes=0.001),
            observer=failing_observer,
            grace_ms=grace_ms,
            is_lock_held=None,
        )

    task = asyncio.create_task(run())

    with pytest.warns(RuntimeWarning, match="observer explodes on cap"):
        await asyncio.wait_for(cap_seen.wait(), timeout=2.0)
    captured = wrapper.captured_abort
    if captured is None:
        pytest.fail("the driver never started, so no abort event was captured")
    await asyncio.wait_for(captured.wait(), timeout=2.0)
    release.set()
    result = await asyncio.wait_for(task, timeout=2.0)

    assert captured.is_set()
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


async def test_supervise_when_session_ends_does_cancel_interrupt_task(tmp_path: Path):
    slow_session: _SlowInterruptSession | None = None

    def wrap(inner: DriverSession) -> DriverSession:
        nonlocal slow_session
        slow_session = _SlowInterruptSession(inner)
        return slow_session

    inner = create_mock_driver([CostStep(cost_usd=0.01, delay_ms=60_000)])
    driver = _WrapDriver(inner, wrap)

    result = await _supervise(
        str(tmp_path / "repo"),
        driver,
        max_minutes=0.001,
        launch=make_launch(max_minutes=0.001),
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    start_ms = 1_000_000
    deadline_ms = start_ms + 10 * 60 * 1000
    clock = [start_ms]
    monkeypatch.setattr("gymrat.supervisor.supervise.now_ms", lambda: clock[0])

    async def pass_deadline() -> None:
        clock[0] = deadline_ms

    # Only a poll that rereads the clock sees the jump; a single sleep to the
    # deadline would leave the run waiting ten real minutes.
    driver = create_mock_driver([
        ActionStep(action=pass_deadline),
        CostStep(cost_usd=0.01, delay_ms=60_000),
    ])

    result = await _supervise(
        str(tmp_path / "repo"),
        driver,
        deadline_ms=deadline_ms,
        launch=make_launch(max_minutes=10),
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
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    inner = create_mock_driver([TurnEndStep(cost_usd=0.5)])
    driver = _WrapDriver(inner, _EndThenRaiseSession)

    result = await _supervise(str(tmp_path / "repo"), driver, max_usd=0.1)

    await asyncio.sleep(0)

    assert result.ended_by == "spend-cap"
    assert "background task failed: end exploded" in capsys.readouterr().err.splitlines()


# ---------------------------------------------------------------------------
# _end_session re-entry guard
# ---------------------------------------------------------------------------


class _CountingEndSession(DelegatingSession):
    """Counts ``end`` calls to detect duplicate ``_end_session`` invocations."""

    def __init__(self, inner: DriverSession, counter: _Box) -> None:
        super().__init__(inner)
        self._counter = counter

    @override
    async def end(self) -> None:
        self._counter.value += 1
        await self._inner.end()


async def test_supervise_when_end_session_called_twice_does_fire_session_end_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    end_count = _Box()

    def wrap(inner: DriverSession) -> DriverSession:
        return _CountingEndSession(inner, end_count)

    inner = create_mock_driver([TurnEndStep(cost_usd=0.5)])
    driver = _WrapDriver(inner, wrap)

    supervise_mod = sys.modules["gymrat.supervisor.supervise"]
    cls = supervise_mod._Supervision  # type: ignore[attr-defined] -- module looked up dynamically
    original_end_session = cls._end_session
    entry_count = _Box()

    def double_end(self_inner: object, *args: object, **kwargs: object) -> None:
        entry_count.value += 1
        original_end_session(self_inner, *args, **kwargs)
        if entry_count.value == 1:
            original_end_session(self_inner, *args, **kwargs)

    monkeypatch.setattr(cls, "_end_session", double_end)

    result = await _supervise(str(tmp_path / "repo"), driver, max_usd=0.1)

    assert result.ended_by == "spend-cap"
    assert end_count.value == 1
