"""Behavioral tests for the supervise progress reporter.

The reporter turns a stream of :class:`~gymrat_py.supervisor.events.SessionEvent`
values into a ``budget · loop · liveness`` status line. Every test injects both
the clock (``now``) and the session reader (``read_session``) so nothing depends
on real time or disk, and patches ``create_status_line`` as imported in the
reporter module with a fake that records each ``.write`` and captures the
``on_tick`` callback. Driving that captured callback (``line.tick()``) after
advancing the injected clock exercises the ~1s TTY refresh deterministically.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Any, NamedTuple

import pytest

from gymrat_py.cli.status_line import RenderMode
from gymrat_py.cli.supervise_progress import (
    CapType,
    ReadSessionResult,
    SuperviseReporter,
    create_supervise_reporter,
)
from gymrat_py.session import IterationPrimary, IterationRecord
from gymrat_py.session.store import SessionState
from gymrat_py.supervisor.events import (
    CapEvent,
    LaunchEvent,
    SessionObserver,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
    UsageUpdateEvent,
)
from tests.session._records import finalize_record, iteration_record

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class Clock:
    """A mutable millisecond clock a test advances by assigning ``now``."""

    def __init__(self, start: int):
        self.now = start

    def __call__(self) -> int:
        return self.now


class FakeStatusLine:
    """Records every ``.write`` and captures ``on_tick`` for manual firing."""

    def __init__(self, mode: RenderMode, on_tick: Callable[[], str] | None = None):
        self.mode = mode
        self.on_tick = on_tick
        self.writes: list[str] = []
        self.warnings: list[str] = []
        self.stopped = False

    def write(self, text: str) -> None:
        self.writes.append(text)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def stop(self) -> None:
        self.stopped = True

    def tick(self) -> None:
        """Mimic the real primitive's timer: write whatever ``on_tick`` renders."""
        if self.on_tick is not None:
            self.writes.append(self.on_tick())


class StatusLineFactory:
    """A ``create_status_line`` stand-in that remembers the lines it built."""

    def __init__(self):
        self.instances: list[FakeStatusLine] = []

    def __call__(
        self, mode: RenderMode, on_tick: Callable[[], str] | None = None
    ) -> FakeStatusLine:
        line = FakeStatusLine(mode, on_tick)
        self.instances.append(line)
        return line

    @property
    def line(self) -> FakeStatusLine:
        return self.instances[-1]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def empty_session_state() -> SessionState:
    """A session that has opened but measured nothing yet."""
    return SessionState(
        session=None,
        iteration_count=0,
        last_iteration=None,
        unsettled=False,
        keep_count=0,
        discard_count=0,
        target_reached_and_kept=False,
        last_seq=0,
        last_kept_commit=None,
        ends_on_gating_block=False,
        finalized=None,
    )


def session_state(**changes: Any) -> SessionState:
    """The empty session state with the named fields overridden."""
    return replace(empty_session_state(), **changes)


def make_iteration(delta_pct: float | None, outcome: str, seq: int = 1) -> IterationRecord:
    """An iteration whose only reporter-visible fields are its delta and outcome."""
    return iteration_record(
        seq=seq,
        primary=IterationPrimary(kind="geomean", delta_pct=delta_pct),
        outcome=outcome,
    )


def make_read_session(state: SessionState, *, has_baseline: bool):
    """A ``read_session`` that always returns ``state`` and ``has_baseline``."""
    result = ReadSessionResult(state=state, has_baseline=has_baseline)
    return lambda: result


# ---------------------------------------------------------------------------
# Event firers
# ---------------------------------------------------------------------------


def fire_launch(observer: SessionObserver, timestamp: int = 1000) -> None:
    observer(
        LaunchEvent(
            timestamp=timestamp,
            head_sha="abc123",
            dirty=False,
            max_minutes=60,
            max_usd=None,
            model=None,
            runbook_path="/path/to/runbook.md",
            kickoff_summary="test kickoff",
        )
    )


def fire_tool_start(
    observer: SessionObserver, tool_name: str, tool_use_id: str, timestamp: int = 2000
) -> None:
    observer(
        ToolStartEvent(
            timestamp=timestamp,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            input={},
            input_summary="...",
        )
    )


def fire_tool_end(
    observer: SessionObserver, tool_name: str, tool_use_id: str, timestamp: int = 3000
) -> None:
    observer(
        ToolEndEvent(
            timestamp=timestamp,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            duration_ms=timestamp - 2000,
            result="ok",
            result_summary="ok",
        )
    )


def fire_usage_update(observer: SessionObserver, cost_usd: float, timestamp: int = 4000) -> None:
    observer(UsageUpdateEvent(timestamp=timestamp, cost_usd=cost_usd))


def fire_cap(observer: SessionObserver, cap: CapType, timestamp: int = 5000) -> None:
    observer(CapEvent(timestamp=timestamp, cap=cap))


# ---------------------------------------------------------------------------
# Reporter setup
# ---------------------------------------------------------------------------


class ReporterHarness(NamedTuple):
    reporter: SuperviseReporter
    line: FakeStatusLine


def set_up_reporter(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> ReporterHarness:
    """Build a reporter with plain-mode defaults, patching the status line factory."""
    factory = StatusLineFactory()
    monkeypatch.setattr("gymrat_py.cli.supervise_progress.create_status_line", factory)
    params: dict[str, Any] = {
        "root": "/tmp/repo",
        "max_minutes": 60,
        "mode": "plain",
        "now": Clock(1000),
        "read_session": make_read_session(empty_session_state(), has_baseline=False),
    }
    params.update(overrides)
    reporter = create_supervise_reporter(**params)
    return ReporterHarness(reporter, factory.line)


def _contains(writes: list[str], needle: str) -> bool:
    return any(needle in chunk for chunk in writes)


def _find(writes: list[str], needle: str) -> str:
    return next(chunk for chunk in writes if needle in chunk)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def test_create_reporter_when_built_does_expose_observer_and_stop(monkeypatch: pytest.MonkeyPatch):
    harness = set_up_reporter(monkeypatch)

    assert callable(harness.reporter.observer)
    assert callable(harness.reporter.stop)
    harness.reporter.stop()


# ---------------------------------------------------------------------------
# budget segment
# ---------------------------------------------------------------------------


def test_budget_when_launched_does_render_elapsed_over_max_minutes(monkeypatch: pytest.MonkeyPatch):
    harness = set_up_reporter(monkeypatch, mode="overwrite", now=Clock(66_000))

    fire_launch(harness.reporter.observer, 1000)

    assert _contains(harness.line.writes, "1m 5s / 60m")


def test_budget_when_max_usd_set_and_no_cost_yet_does_show_cost_placeholder(
    monkeypatch: pytest.MonkeyPatch,
):
    harness = set_up_reporter(monkeypatch, mode="overwrite", max_usd=5.0, now=Clock(1000))

    fire_launch(harness.reporter.observer, 1000)

    assert _contains(harness.line.writes, "$— / $5.00")


def test_budget_when_usage_update_and_max_usd_set_does_show_cost_against_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = Clock(1000)
    harness = set_up_reporter(monkeypatch, mode="overwrite", max_usd=5.0, now=clock)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    clock.now = 2000
    fire_usage_update(observer, 1.42, 2000)

    assert _contains(harness.line.writes, "$1.42 / $5.00")


def test_budget_when_usage_update_and_no_max_usd_does_show_bare_cost(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = Clock(1000)
    harness = set_up_reporter(monkeypatch, mode="overwrite", now=clock)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    clock.now = 2000
    fire_usage_update(observer, 2.5, 2000)

    cost_line = _find(harness.line.writes, "$2.50")
    assert "/ $" not in cost_line


def test_budget_when_now_omitted_does_use_the_wall_clock(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("time.time", lambda: 66.0)
    factory = StatusLineFactory()
    monkeypatch.setattr("gymrat_py.cli.supervise_progress.create_status_line", factory)
    reporter = create_supervise_reporter(
        root="/tmp/repo",
        max_minutes=60,
        mode="overwrite",
        read_session=make_read_session(empty_session_state(), has_baseline=False),
    )

    fire_launch(reporter.observer, 1000)

    assert _contains(factory.line.writes, "1m 5s / 60m")


@pytest.mark.parametrize(
    ("max_minutes", "expected"),
    [
        pytest.param(5.5, "1m 5s / 5.5m", id="fractional-keeps-decimal"),
        pytest.param(10.0, "1m 5s / 10m", id="whole-float-drops-decimal"),
        pytest.param(10, "1m 5s / 10m", id="integer-has-no-decimal"),
    ],
)
def test_budget_when_max_minutes_given_does_render_the_actual_cap_value(
    monkeypatch: pytest.MonkeyPatch, max_minutes: float, expected: str
):
    harness = set_up_reporter(
        monkeypatch, mode="overwrite", max_minutes=max_minutes, now=Clock(66_000)
    )

    fire_launch(harness.reporter.observer, 1000)

    assert _contains(harness.line.writes, expected)


# ---------------------------------------------------------------------------
# loop segment
# ---------------------------------------------------------------------------


def _throwing_read() -> ReadSessionResult:
    message = "no session file"
    raise RuntimeError(message)


def test_loop_when_read_session_throws_does_report_no_session_yet(monkeypatch: pytest.MonkeyPatch):
    harness = set_up_reporter(
        monkeypatch, mode="overwrite", read_session=_throwing_read, now=Clock(1000)
    )

    fire_launch(harness.reporter.observer, 1000)

    assert _contains(harness.line.writes, "no session yet")


def test_loop_when_baseline_but_no_iterations_does_report_baseline_recorded(
    monkeypatch: pytest.MonkeyPatch,
):
    harness = set_up_reporter(
        monkeypatch,
        mode="overwrite",
        max_iterations=20,
        read_session=make_read_session(empty_session_state(), has_baseline=True),
        now=Clock(1000),
    )
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    fire_tool_end(observer, "Bash", "bash-1", 3000)

    assert _contains(harness.line.writes, "baseline recorded")


def test_loop_when_iterations_present_does_show_counts_kept_and_discarded(
    monkeypatch: pytest.MonkeyPatch,
):
    state = session_state(
        iteration_count=3,
        keep_count=2,
        discard_count=1,
        last_iteration=make_iteration(-3.2, "improved"),
    )
    harness = set_up_reporter(
        monkeypatch,
        mode="overwrite",
        max_iterations=20,
        read_session=make_read_session(state, has_baseline=True),
        now=Clock(1000),
    )
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    fire_tool_end(observer, "Bash", "bash-1", 3000)

    loop_line = _find(harness.line.writes, "iter 3/20")
    assert "2 kept" in loop_line
    assert "1 discarded" in loop_line


def test_loop_when_max_iterations_absent_does_omit_the_denominator(monkeypatch: pytest.MonkeyPatch):
    state = session_state(
        iteration_count=2,
        keep_count=1,
        discard_count=1,
        last_iteration=make_iteration(1.5, "regressed"),
    )
    harness = set_up_reporter(
        monkeypatch,
        mode="overwrite",
        read_session=make_read_session(state, has_baseline=True),
        now=Clock(1000),
    )
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    fire_tool_end(observer, "Bash", "bash-1", 3000)

    loop_line = _find(harness.line.writes, "iter 2")
    assert "iter 2/" not in loop_line


def test_loop_when_last_iteration_present_does_show_delta_and_outcome(
    monkeypatch: pytest.MonkeyPatch,
):
    state = session_state(
        iteration_count=2,
        keep_count=1,
        last_iteration=make_iteration(-3.2, "improved"),
    )
    harness = set_up_reporter(
        monkeypatch,
        mode="overwrite",
        read_session=make_read_session(state, has_baseline=True),
        now=Clock(1000),
    )
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    fire_tool_end(observer, "Bash", "bash-1", 3000)

    loop_line = _find(harness.line.writes, "improved")
    assert "-3.2%" in loop_line


def test_loop_when_last_delta_is_none_does_render_em_dash(monkeypatch: pytest.MonkeyPatch):
    state = session_state(
        iteration_count=1,
        last_iteration=make_iteration(None, "no-signal"),
    )
    harness = set_up_reporter(
        monkeypatch,
        mode="overwrite",
        read_session=make_read_session(state, has_baseline=True),
        now=Clock(1000),
    )
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    fire_tool_end(observer, "Bash", "bash-1", 3000)

    loop_line = _find(harness.line.writes, "no-signal")
    assert "—" in loop_line


def test_loop_when_unsettled_does_append_unsettled(monkeypatch: pytest.MonkeyPatch):
    state = session_state(
        iteration_count=1,
        unsettled=True,
        last_iteration=make_iteration(-2.0, "improved"),
    )
    harness = set_up_reporter(
        monkeypatch,
        mode="overwrite",
        read_session=make_read_session(state, has_baseline=True),
        now=Clock(1000),
    )
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    fire_tool_end(observer, "Bash", "bash-1", 3000)

    assert _contains(harness.line.writes, "unsettled")


def test_loop_when_finalized_does_report_finalized(monkeypatch: pytest.MonkeyPatch):
    state = session_state(
        iteration_count=3,
        keep_count=2,
        discard_count=1,
        last_iteration=make_iteration(-5.0, "improved"),
        finalized=finalize_record(),
    )
    harness = set_up_reporter(
        monkeypatch,
        mode="overwrite",
        read_session=make_read_session(state, has_baseline=True),
        now=Clock(1000),
    )
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    fire_tool_end(observer, "Bash", "bash-1", 3000)

    assert _contains(harness.line.writes, "finalized")


def test_loop_when_last_iteration_absent_does_omit_last(monkeypatch: pytest.MonkeyPatch):
    state = session_state(iteration_count=2, keep_count=1, discard_count=1)
    harness = set_up_reporter(
        monkeypatch,
        mode="overwrite",
        read_session=make_read_session(state, has_baseline=True),
        now=Clock(1000),
    )
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    fire_tool_end(observer, "Bash", "bash-1", 3000)

    iter_lines = [chunk for chunk in harness.line.writes if "iter 2" in chunk]
    assert iter_lines
    assert all("last" not in line for line in iter_lines)


# ---------------------------------------------------------------------------
# session re-read
# ---------------------------------------------------------------------------


class CountingRead:
    """A ``read_session`` that tallies how many times it was called."""

    def __init__(self):
        self.count = 0

    def __call__(self) -> ReadSessionResult:
        self.count += 1
        return ReadSessionResult(state=empty_session_state(), has_baseline=False)


def test_reread_when_non_bash_tool_ends_does_not_reread_but_bash_does(
    monkeypatch: pytest.MonkeyPatch,
):
    counting = CountingRead()
    harness = set_up_reporter(monkeypatch, read_session=counting)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    after_launch = counting.count
    fire_tool_start(observer, "Read", "read-1", 2000)
    fire_tool_end(observer, "Read", "read-1", 3000)
    after_read = counting.count
    fire_tool_start(observer, "Bash", "bash-1", 4000)
    fire_tool_end(observer, "Bash", "bash-1", 5000)
    after_bash = counting.count

    assert after_read == after_launch
    assert after_bash > after_launch


def test_reread_when_tool_end_has_unknown_id_does_reread(monkeypatch: pytest.MonkeyPatch):
    counting = CountingRead()
    harness = set_up_reporter(monkeypatch, read_session=counting)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    after_launch = counting.count
    fire_tool_end(observer, "Bash", "unknown-id", 3000)

    assert counting.count > after_launch


# ---------------------------------------------------------------------------
# liveness segment
# ---------------------------------------------------------------------------


def test_liveness_before_any_tool_does_show_starting(monkeypatch: pytest.MonkeyPatch):
    harness = set_up_reporter(monkeypatch, mode="overwrite", now=Clock(1000))

    fire_launch(harness.reporter.observer, 1000)

    assert _contains(harness.line.writes, "starting")


def test_liveness_when_tool_in_flight_does_show_tool_and_elapsed(monkeypatch: pytest.MonkeyPatch):
    clock = Clock(1000)
    harness = set_up_reporter(monkeypatch, mode="overwrite", max_usd=5.0, now=clock)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    clock.now = 2000
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    clock.now = 7000
    harness.line.tick()

    assert _contains(harness.line.writes, "Bash 5s")


def test_liveness_when_last_tool_ends_does_show_ago(monkeypatch: pytest.MonkeyPatch):
    clock = Clock(1000)
    harness = set_up_reporter(monkeypatch, mode="overwrite", max_usd=5.0, now=clock)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    clock.now = 2000
    fire_tool_start(observer, "Read", "read-1", 2000)
    clock.now = 3000
    fire_tool_end(observer, "Read", "read-1", 3000)
    clock.now = 6000
    harness.line.tick()

    assert _contains(harness.line.writes, "Read 3s ago")


def test_liveness_when_idle_past_threshold_does_show_idle(monkeypatch: pytest.MonkeyPatch):
    clock = Clock(1000)
    harness = set_up_reporter(monkeypatch, mode="overwrite", max_usd=5.0, now=clock)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    clock.now = 2000
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    clock.now = 3000
    fire_tool_end(observer, "Bash", "bash-1", 3000)
    clock.now = 3000 + 180_001
    harness.line.tick()

    assert _contains(harness.line.writes, "idle")


def test_liveness_when_tool_ends_below_idle_threshold_does_show_ago_in_place(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = Clock(1000)
    harness = set_up_reporter(monkeypatch, mode="overwrite", now=clock)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    clock.now = 2000
    fire_tool_start(observer, "Read", "read-1", 2000)
    clock.now = 3000
    fire_tool_end(observer, "Read", "read-1", 3000)

    assert _contains(harness.line.writes, "Read 0s ago")


def test_liveness_when_tool_ends_past_idle_threshold_does_show_idle_in_place(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = Clock(1000)
    harness = set_up_reporter(monkeypatch, mode="overwrite", now=clock)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    clock.now = 2000
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    clock.now = 200_000
    fire_tool_end(observer, "Bash", "bash-1", 3000)

    assert _contains(harness.line.writes, "idle")


def test_liveness_when_tool_starts_does_show_tool_name(monkeypatch: pytest.MonkeyPatch):
    clock = Clock(1000)
    harness = set_up_reporter(monkeypatch, mode="overwrite", now=clock)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    clock.now = 2000
    fire_tool_start(observer, "Read", "read-1", 2000)

    assert _contains(harness.line.writes, "Read 0s")


def test_liveness_when_one_of_several_tools_ends_does_fall_back_to_remaining(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = Clock(1000)
    harness = set_up_reporter(monkeypatch, mode="overwrite", now=clock)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    clock.now = 2000
    fire_tool_start(observer, "Read", "read-1", 2000)
    clock.now = 2500
    fire_tool_start(observer, "Bash", "bash-1", 2500)
    clock.now = 3000
    fire_tool_end(observer, "Read", "read-1", 3000)

    assert "Bash" in harness.line.writes[-1]


def test_liveness_when_untracked_tool_ends_does_not_change(monkeypatch: pytest.MonkeyPatch):
    harness = set_up_reporter(monkeypatch, mode="overwrite", now=Clock(1000))
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_tool_end(observer, "Bash", "never-started", 3000)

    assert _contains(harness.line.writes, "starting")


def test_liveness_when_text_delta_or_thinking_update_does_not_render(
    monkeypatch: pytest.MonkeyPatch,
):
    harness = set_up_reporter(monkeypatch)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    after_launch = len(harness.line.writes)
    observer(TextDeltaEvent(timestamp=2500, chunk="hello"))
    observer(ThinkingUpdateEvent(timestamp=2500, estimated_tokens=100, delta=10))

    assert len(harness.line.writes) == after_launch


def test_liveness_when_tick_fires_before_launch_does_use_zero_elapsed(
    monkeypatch: pytest.MonkeyPatch,
):
    harness = set_up_reporter(monkeypatch, mode="overwrite", now=Clock(1000))

    harness.line.tick()

    assert _contains(harness.line.writes, "0s")


# ---------------------------------------------------------------------------
# cap event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", ["wall-clock", "spend-cap"])
def test_cap_when_fired_in_tty_does_show_interrupting_with_cap_type(
    monkeypatch: pytest.MonkeyPatch, cap: CapType
):
    harness = set_up_reporter(monkeypatch, mode="overwrite", now=Clock(1000))
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_cap(observer, cap)

    assert _contains(harness.line.writes, f"interrupting ({cap})")


def test_cap_when_fired_does_freeze_liveness_against_later_tool_events(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = Clock(1000)
    harness = set_up_reporter(monkeypatch, mode="overwrite", now=clock)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_cap(observer, "wall-clock")
    clock.now = 6000
    fire_tool_start(observer, "Bash", "bash-1", 6000)
    fire_tool_end(observer, "Bash", "bash-1", 7000)

    assert "interrupting" in harness.line.writes[-1]
    assert "Bash" not in harness.line.writes[-1]


def test_cap_when_tool_progress_after_launch_does_rerender(monkeypatch: pytest.MonkeyPatch):
    harness = set_up_reporter(monkeypatch, mode="overwrite", now=Clock(1000))
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    observer(ToolProgressEvent(timestamp=2000, tool_use_id="tp-1", elapsed_ms=500))

    assert len(harness.line.writes) >= 2


# ---------------------------------------------------------------------------
# plain mode output
# ---------------------------------------------------------------------------


def test_plain_when_launched_with_spend_cap_does_print_caps_with_dollars(
    monkeypatch: pytest.MonkeyPatch,
):
    harness = set_up_reporter(monkeypatch, max_usd=5.0)

    fire_launch(harness.reporter.observer, 1000)

    assert "caps 60m, $5.00" in harness.line.writes


def test_plain_when_launched_without_spend_cap_does_print_bare_caps(
    monkeypatch: pytest.MonkeyPatch,
):
    harness = set_up_reporter(monkeypatch, max_minutes=30)

    fire_launch(harness.reporter.observer, 1000)

    assert "caps 30m" in harness.line.writes


@pytest.mark.parametrize(
    ("max_minutes", "expected"),
    [
        pytest.param(5.5, "caps 5.5m", id="fractional-keeps-decimal"),
        pytest.param(10.0, "caps 10m", id="whole-float-drops-decimal"),
    ],
)
def test_plain_caps_when_max_minutes_given_does_render_the_actual_cap_value(
    monkeypatch: pytest.MonkeyPatch, max_minutes: float, expected: str
):
    harness = set_up_reporter(monkeypatch, max_minutes=max_minutes)

    fire_launch(harness.reporter.observer, 1000)

    assert expected in harness.line.writes


def test_plain_when_usage_update_does_print_cost(monkeypatch: pytest.MonkeyPatch):
    harness = set_up_reporter(monkeypatch, max_usd=5.0)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_usage_update(observer, 1.42, 2000)

    assert "cost $1.42" in harness.line.writes


def test_plain_when_loop_changes_does_print_loop_segment(monkeypatch: pytest.MonkeyPatch):
    state = session_state(
        iteration_count=2,
        keep_count=1,
        discard_count=1,
        last_iteration=make_iteration(3.2, "regressed"),
    )
    harness = set_up_reporter(
        monkeypatch,
        max_iterations=20,
        read_session=make_read_session(state, has_baseline=True),
    )
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    fire_tool_end(observer, "Bash", "bash-1", 3000)

    loop_line = _find(harness.line.writes, "iter 2/20")
    assert "+3.2%" in loop_line
    assert "regressed" in loop_line


def test_plain_when_no_session_yet_does_not_print_loop_segment(monkeypatch: pytest.MonkeyPatch):
    harness = set_up_reporter(monkeypatch, read_session=_throwing_read)

    fire_launch(harness.reporter.observer, 1000)

    assert "caps 60m" in harness.line.writes
    assert not _contains(harness.line.writes, "no session yet")


def test_plain_when_capped_does_print_cap_interrupting(monkeypatch: pytest.MonkeyPatch):
    harness = set_up_reporter(monkeypatch)
    observer = harness.reporter.observer

    fire_launch(observer, 1000)
    fire_cap(observer, "wall-clock")

    assert _contains(harness.line.writes, "cap wall-clock")
    assert _contains(harness.line.writes, "interrupting")


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


def test_stop_when_called_does_not_raise(monkeypatch: pytest.MonkeyPatch):
    harness = set_up_reporter(monkeypatch)

    fire_launch(harness.reporter.observer, 1000)
    harness.reporter.stop()
