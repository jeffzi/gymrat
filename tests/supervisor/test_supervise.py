"""Behavioral tests for the supervisor orchestrator.

``supervise`` runs a driver session under a wall-clock cap, an optional spend
cap, and a grace fallback that arms the driver's abort event when a cap fires
but the session keeps running. Tests use tiny ``max_minutes``/``grace_ms`` values
and ``asyncio.Event`` handshakes so the session stays open only as long as a
test needs it. Timing is asserted as loose lower bounds, never exact values, to
stay deterministic under ``pytest-randomly`` and ``pytest-xdist``.
"""

import asyncio
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest

from gymrat.supervisor import (
    LaunchEvent,
    SessionOutcome,
    SupervisedSession,
    SupervisionResult,
    TextDeltaEvent,
    supervise,
)
from gymrat.supervisor.driver import Driver, DriverSession, SessionPrompt
from gymrat.supervisor.events import SessionEvent, SessionObserver
from tests.supervisor._fixtures import (
    _cap_events,
    collecting_observer,
    make_context,
    make_launch,
    make_prompt,
    read_log_lines,
)
from tests.supervisor._mock_driver import (
    ActionStep,
    CostStep,
    EmitStep,
    TurnEndStep,
    create_mock_driver,
)


async def _noop_action() -> None:
    return None


_GUARD_MESSAGE = "this step must not run after a cap fires"


async def _guard_action() -> None:
    raise AssertionError(_GUARD_MESSAGE)


async def _supervise_fast(
    driver: Driver,
    prompt: SessionPrompt,
    *,
    context: SupervisedSession,
    launch: LaunchEvent,
    observer: SessionObserver | None = None,
    is_lock_held: Callable[[], bool] = lambda: False,
    grace_ms: int = 30_000,
) -> SupervisionResult:
    """Call ``supervise`` with the settle-window and lock-poll defaults every fast test shares."""
    return await supervise(
        driver,
        prompt,
        context=context,
        launch=launch,
        observer=observer,
        settle_window_ms=0,
        lock_poll_ms=1,
        is_lock_held=is_lock_held,
        grace_ms=grace_ms,
    )


@dataclass
class _Box:
    """A mutable integer holder for counting side effects across closures."""

    value: int = 0


# ---------------------------------------------------------------------------
# test doubles
# ---------------------------------------------------------------------------


class _DelegatingSession:
    """Wraps a ``DriverSession``.

    Delegates ``outcome``, ``interrupt``, ``send``, and ``end``.
    """

    def __init__(self, inner: DriverSession) -> None:
        self._inner = inner

    @property
    def outcome(self) -> Awaitable[SessionOutcome]:
        return self._inner.outcome

    async def interrupt(self) -> None:
        await self._inner.interrupt()

    async def send(self, text: str) -> None:
        await self._inner.send(text)

    async def end(self) -> None:
        await self._inner.end()


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

    async def send(self, text: str) -> None:
        return None

    async def end(self) -> None:
        return None


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


# ---------------------------------------------------------------------------
# normal completion
# ---------------------------------------------------------------------------


async def test_supervise_when_session_completes_does_report_outcome(
    tmp_path: Path,
):
    driver = create_mock_driver([CostStep(cost_usd=0.05), CostStep(cost_usd=0.12)])

    result = await supervise(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, log_path=str(tmp_path / "events.jsonl")),
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

    await supervise(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, log_path=str(log_path)),
        launch=make_launch(),
    )

    lines = read_log_lines(log_path)
    assert [line["type"] for line in lines[:3]] == ["launch", "text_delta", "usage_update"]


async def test_supervise_when_observer_given_does_forward_events_in_order(
    tmp_path: Path,
):
    probe = collecting_observer()
    launch = make_launch()
    driver = create_mock_driver([CostStep(cost_usd=0.03)])

    await supervise(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, log_path=str(tmp_path / "events.jsonl")),
        launch=launch,
        observer=probe.observer,
    )

    assert probe.events[0] == launch
    assert any(event.type == "usage_update" for event in probe.events)


# ---------------------------------------------------------------------------
# wall-clock cap
# ---------------------------------------------------------------------------


async def test_supervise_when_wall_clock_elapses_does_report_wall_clock(
    tmp_path: Path,
):
    driver = create_mock_driver([CostStep(cost_usd=0.05, delay_ms=60_000)])

    result = await supervise(
        driver,
        make_prompt(),
        context=make_context(max_minutes=0.001, log_path=str(tmp_path / "events.jsonl")),
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
        context=make_context(max_minutes=0.001, log_path=str(tmp_path / "events.jsonl")),
        launch=make_launch(max_minutes=0.001),
        observer=probe.observer,
    )

    caps = _cap_events(probe.events)
    assert len(caps) == 1
    assert caps[0].cap == "wall-clock"


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
            context=make_context(max_minutes=0.001, log_path=str(tmp_path / "events.jsonl")),
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


async def test_supervise_when_cost_reaches_max_usd_does_report_spend_cap(
    tmp_path: Path,
):
    driver = create_mock_driver([TurnEndStep(cost_usd=0.12)])

    result = await _supervise_fast(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, max_usd=0.1, log_path=str(tmp_path / "events.jsonl")),
        launch=make_launch(max_usd=0.1),
    )

    assert result.ended_by == "spend-cap"


async def test_supervise_when_cost_reaches_max_usd_does_emit_single_spend_cap_event(
    tmp_path: Path,
):
    probe = collecting_observer()
    driver = create_mock_driver([TurnEndStep(cost_usd=0.12)])

    await _supervise_fast(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, max_usd=0.1, log_path=str(tmp_path / "events.jsonl")),
        launch=make_launch(max_usd=0.1),
        observer=probe.observer,
    )

    caps = _cap_events(probe.events)
    assert len(caps) == 1
    assert caps[0].cap == "spend-cap"


async def test_supervise_when_max_usd_none_does_not_enforce_cost(tmp_path: Path):
    driver = create_mock_driver([CostStep(cost_usd=5.0), CostStep(cost_usd=10.0)])

    result = await supervise(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, log_path=str(tmp_path / "events.jsonl")),
        launch=make_launch(),
    )

    assert result.ended_by == "session"
    assert result.outcome.reason == "completed"
    assert result.cost_usd == 10.0


async def test_supervise_when_spend_cap_trips_does_log_usage_update_before_cap(
    tmp_path: Path,
):
    driver = create_mock_driver([CostStep(cost_usd=0.12), TurnEndStep(cost_usd=0.12)])
    log_path = tmp_path / "events.jsonl"

    await _supervise_fast(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, max_usd=0.1, log_path=str(log_path)),
        launch=make_launch(max_usd=0.1),
    )

    lines = read_log_lines(log_path)
    types = [line["type"] for line in lines]
    assert "cap" in types
    cap_idx = types.index("cap")
    preceding = types[:cap_idx]
    assert "usage_update" in preceding


# ---------------------------------------------------------------------------
# cap racing
# ---------------------------------------------------------------------------


async def test_supervise_when_both_caps_could_fire_does_report_first_cap_only(
    tmp_path: Path,
):
    probe = collecting_observer()
    # The turn end carries cost above max_usd. The settle → classify fires
    # the spend-cap before the wall clock reaches its deadline, proving the
    # spend cap wins the race when both could fire.
    driver = create_mock_driver([TurnEndStep(cost_usd=0.15)])

    result = await _supervise_fast(
        driver,
        make_prompt(),
        context=make_context(
            max_minutes=0.001, max_usd=0.1, log_path=str(tmp_path / "events.jsonl")
        ),
        launch=make_launch(max_minutes=0.001, max_usd=0.1),
        observer=probe.observer,
        grace_ms=50,
    )

    assert result.ended_by == "spend-cap"
    assert len(_cap_events(probe.events)) == 1


async def test_supervise_when_spend_cap_trips_at_turn_end_does_report_spend_cap(
    tmp_path: Path,
):
    probe = collecting_observer()
    driver = create_mock_driver(
        [
            CostStep(cost_usd=5.0),
            TurnEndStep(cost_usd=5.0),
        ]
    )

    result = await _supervise_fast(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, max_usd=1.0, log_path=str(tmp_path / "events.jsonl")),
        launch=make_launch(max_usd=1.0),
        observer=probe.observer,
    )

    caps = _cap_events(probe.events)
    assert result.ended_by == "spend-cap"
    assert len(caps) == 1
    assert caps[0].cap == "spend-cap"


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


async def test_supervise_when_driver_errors_does_report_error_outcome(
    tmp_path: Path,
):
    driver = create_mock_driver(_error_steps())  # type: ignore[arg-type]

    result = await supervise(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, log_path=str(tmp_path / "events.jsonl")),
        launch=make_launch(),
    )

    assert result.outcome.reason == "error"
    assert result.outcome.message == "kaboom"
    assert result.ended_by == "session"


async def test_supervise_when_driver_errors_does_log_events_up_to_failure(
    tmp_path: Path,
):
    driver = create_mock_driver(_error_steps())  # type: ignore[arg-type]
    log_path = tmp_path / "events.jsonl"

    await supervise(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, log_path=str(log_path)),
        launch=make_launch(),
    )

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

    driver = create_mock_driver([TurnEndStep(cost_usd=0.5)])

    result = await _supervise_fast(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, max_usd=0.1, log_path=str(tmp_path / "events.jsonl")),
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
        context=make_context(max_minutes=0.001, log_path=str(tmp_path / "events.jsonl")),
        launch=make_launch(max_minutes=0.001),
        grace_ms=50,
    )

    assert result.ended_by == "wall-clock"


async def test_supervise_when_outcome_rejects_does_propagate_rejection(tmp_path: Path):
    probe = collecting_observer()

    with pytest.raises(RuntimeError, match="session crashed"):
        await supervise(
            _RejectingDriver(),
            make_prompt(),
            context=make_context(max_minutes=5, log_path=str(tmp_path / "events.jsonl")),
            launch=make_launch(),
            observer=probe.observer,
        )

    assert _cap_events(probe.events) == []


# ---------------------------------------------------------------------------
# B31 — observer resilience: cap fires even when an observer raises
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("default::RuntimeWarning")
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

    async def run() -> object:
        return await supervise(
            wrapper,
            make_prompt(),
            context=make_context(max_minutes=0.001, log_path=str(tmp_path / "events.jsonl")),
            launch=make_launch(max_minutes=0.001),
            observer=failing_observer,
            grace_ms=grace_ms,
        )

    task = asyncio.create_task(run())
    await asyncio.wait_for(cap_seen.wait(), timeout=2.0)

    captured = wrapper.captured_abort
    assert captured is not None
    await asyncio.wait_for(captured.wait(), timeout=2.0)

    release.set()
    result = await asyncio.wait_for(task, timeout=2.0)
    assert result.ended_by == "wall-clock"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# B35 — interrupt task teardown
# ---------------------------------------------------------------------------


class _SlowInterruptSession(_DelegatingSession):
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

    result = await supervise(
        driver,
        make_prompt(),
        context=make_context(max_minutes=0.001, log_path=str(tmp_path / "events.jsonl")),
        launch=make_launch(max_minutes=0.001),
        grace_ms=50,
    )

    assert result.ended_by == "wall-clock"
    assert slow_session is not None

    # Give the event loop a tick so the CancelledError propagates.
    await asyncio.sleep(0)
    assert slow_session.interrupt_cancelled.is_set()


# ---------------------------------------------------------------------------
# wall-clock poll
# ---------------------------------------------------------------------------


async def test_supervise_when_wall_clock_fires_via_poll_does_end_at_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The poll loop compares ``now_ms()`` against a wall-clock deadline.

    Advancing the fake clock past the deadline on the first poll makes the cap
    fire immediately, proving the loop checks ``now_ms()`` rather than relying
    on a single ``asyncio.sleep`` for the full duration.
    """
    supervise_mod = sys.modules["gymrat.supervisor.supervise"]

    call_count = 0
    start_time = 1_000_000

    def fake_now_ms() -> float:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return start_time
        return start_time + 10 * 60 * 1000 + 1

    monkeypatch.setattr(supervise_mod, "now_ms", fake_now_ms)
    driver = create_mock_driver([CostStep(cost_usd=0.01, delay_ms=60_000)])

    deadline_ms = start_time + 10 * 60 * 1000
    result = await supervise(
        driver,
        make_prompt(),
        context=make_context(
            max_minutes=10,
            log_path=str(tmp_path / "events.jsonl"),
            deadline_ms=deadline_ms,
        ),
        launch=make_launch(max_minutes=10),
        wall_clock_poll_ms=10,
    )

    assert result.ended_by == "wall-clock"
    assert call_count >= 3


# ---------------------------------------------------------------------------
# _spawn exception reporting
# ---------------------------------------------------------------------------


class _EndThenRaiseSession(_DelegatingSession):
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

    result = await _supervise_fast(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, max_usd=0.1, log_path=str(tmp_path / "events.jsonl")),
        launch=make_launch(max_usd=0.1),
    )

    await asyncio.sleep(0)

    assert result.ended_by == "spend-cap"
    captured = capsys.readouterr()
    assert "end exploded" in captured.err


# ---------------------------------------------------------------------------
# _end_session re-entry guard
# ---------------------------------------------------------------------------


class _CountingEndSession(_DelegatingSession):
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
    cls = supervise_mod._Supervision  # type: ignore[attr-defined]
    original_end_session = cls._end_session
    entry_count = _Box()

    def double_end(self_inner: object, *args: object, **kwargs: object) -> None:
        entry_count.value += 1
        original_end_session(self_inner, *args, **kwargs)
        if entry_count.value == 1:
            original_end_session(self_inner, *args, **kwargs)

    monkeypatch.setattr(cls, "_end_session", double_end)

    result = await _supervise_fast(
        driver,
        make_prompt(),
        context=make_context(max_minutes=10, max_usd=0.1, log_path=str(tmp_path / "events.jsonl")),
        launch=make_launch(max_usd=0.1),
    )

    assert result.ended_by == "spend-cap"
    assert end_count.value == 1
