"""Tests for the iterate progress renderer (live tree + plain modes).

Tests inject a deterministic ``Clock`` from ``tests._rich`` and capture
output through ``sealed_console``.  Frame content is pinned with syrupy
golden snapshots; plain-mode milestones use exact-line equality; live wiring
assertions check ``Live`` attributes directly.
"""

from __future__ import annotations

import sys
from io import StringIO
from typing import TYPE_CHECKING

import pytest

from gymrat_py.cli.iterate_progress import (
    IterateRenderer,
    create_fan_out,
    create_iterate_renderer,
)
from gymrat_py.progress_events import (
    ConfirmFinished,
    ConfirmStarted,
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressEvent,
)
from tests._rich import Clock, frame_text, screen_lines, sealed_console

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from rich.console import Console
    from syrupy.assertion import SnapshotAssertion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output(console: Console) -> str:
    f = console.file
    assert isinstance(f, StringIO)
    return f.getvalue()


def _last_line(console: Console) -> str:
    lines = [ln for ln in _output(console).splitlines() if ln.strip()]
    return lines[-1]


def _ms(clock: Clock) -> int:
    return int(clock.now * 1000)


def _pass_started(
    round_num: int,
    total_rounds: int,
    *,
    at_ms: int,
    target_index: int = 0,
    target_count: int = 1,
    label: str = "bench",
    phase: Literal["measure", "confirm"] = "measure",
) -> PassStarted:
    return PassStarted(
        round=round_num,
        total_rounds=total_rounds,
        target_index=target_index,
        target_count=target_count,
        label=label,
        at_ms=at_ms,
        phase=phase,
    )


def _pass_finished(
    round_num: int,
    total_rounds: int,
    *,
    at_ms: int,
    target_index: int = 0,
    target_count: int = 1,
    label: str = "bench",
    phase: Literal["measure", "confirm"] = "measure",
) -> PassFinished:
    return PassFinished(
        round=round_num,
        total_rounds=total_rounds,
        target_index=target_index,
        target_count=target_count,
        label=label,
        at_ms=at_ms,
        phase=phase,
    )


def _renderer(
    mode: Literal["live", "plain"],
    *,
    width: int = 80,
    height: int = 24,
    seq: int = 1,
    session_id: str = "test-session",
    sample_count: int = 5,
    metric_count: int = 3,
    primary_metric: str = "geomean",
    verbose: bool = False,
) -> tuple[Console, Clock, IterateRenderer]:
    clock = Clock()
    console = sealed_console(width=width, height=height)
    renderer = create_iterate_renderer(
        mode=mode,
        console=console,
        seq=seq,
        session_id=session_id,
        sample_count=sample_count,
        metric_count=metric_count,
        primary_metric=primary_metric,
        verbose=verbose,
        clock=clock,
    )
    return console, clock, renderer


def _live(
    *,
    width: int = 80,
    height: int = 24,
    seq: int = 1,
    session_id: str = "test-session",
    sample_count: int = 5,
    metric_count: int = 3,
    primary_metric: str = "geomean",
    verbose: bool = False,
) -> tuple[Console, Clock, IterateRenderer]:
    return _renderer(
        "live",
        width=width,
        height=height,
        seq=seq,
        session_id=session_id,
        sample_count=sample_count,
        metric_count=metric_count,
        primary_metric=primary_metric,
        verbose=verbose,
    )


def _plain(
    *,
    width: int = 80,
    seq: int = 1,
    session_id: str = "test-session",
    sample_count: int = 5,
    metric_count: int = 3,
    primary_metric: str = "geomean",
    verbose: bool = False,
) -> tuple[Console, Clock, IterateRenderer]:
    return _renderer(
        "plain",
        width=width,
        seq=seq,
        session_id=session_id,
        sample_count=sample_count,
        metric_count=metric_count,
        primary_metric=primary_metric,
        verbose=verbose,
    )


def _fake_install(
    registered: list[object],
) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """A fake ``install_termination_cleanup`` that records registrations."""

    def install(cb: Callable[[], None]) -> Callable[[], None]:
        registered.append(cb)
        return lambda: None

    return install


# ---------------------------------------------------------------------------
# Frame golden snapshots via frame_text(renderer.frame())
# ---------------------------------------------------------------------------


def test_frame_when_initial_does_show_all_nodes_pending(
    snapshot: SnapshotAssertion,
):
    """All six tree nodes pending, header shows seq and session id."""
    _console, _clock, renderer = _live(
        seq=3,
        session_id="abc-123",
        metric_count=4,
        primary_metric="geomean",
    )

    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_before_hook_running_does_show_spinner(
    snapshot: SnapshotAssertion,
):
    """You should see before hook with a running spinner, all others pending."""
    _console, _clock, renderer = _live()

    renderer.report(HookStarted(stage="before", at_ms=0))
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_both_worktrees_prepared_does_show_elapsed(
    snapshot: SnapshotAssertion,
):
    """Prepare node done with elapsed, sub-items for each prepared label."""
    _console, clock, renderer = _live()

    renderer.report(HookFinished(stage="before", at_ms=0))
    renderer.report(PrepareStarted(label="baseline", at_ms=0))
    clock.tick(3)
    renderer.report(PrepareFinished(label="baseline", at_ms=_ms(clock)))
    renderer.report(PrepareStarted(label="candidate", at_ms=_ms(clock)))
    clock.tick(2)
    renderer.report(PrepareFinished(label="candidate", at_ms=_ms(clock)))
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_passes_mid_run_does_show_bar_eta_and_detail(
    snapshot: SnapshotAssertion,
):
    """Passes node running with bar, ETA, and detail line naming round and label."""
    _console, clock, renderer = _live(sample_count=5)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    renderer.report(
        _pass_started(
            1,
            5,
            target_index=0,
            target_count=2,
            label="baseline",
            at_ms=_ms(clock),
        )
    )
    clock.tick(10)
    renderer.report(
        _pass_finished(
            1,
            5,
            target_index=0,
            target_count=2,
            label="baseline",
            at_ms=_ms(clock),
        )
    )
    clock.tick(1)
    renderer.report(
        _pass_started(
            1,
            5,
            target_index=1,
            target_count=2,
            label="candidate",
            at_ms=_ms(clock),
        )
    )
    clock.tick(10)
    renderer.report(
        _pass_finished(
            1,
            5,
            target_index=1,
            target_count=2,
            label="candidate",
            at_ms=_ms(clock),
        )
    )
    clock.tick(1)
    renderer.report(
        _pass_started(
            2,
            5,
            target_index=0,
            target_count=2,
            label="baseline",
            at_ms=_ms(clock),
        )
    )
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_judge_finished_does_show_delta_and_regressed(
    snapshot: SnapshotAssertion,
):
    """Judge node done with delta percentage and regressed metric names."""
    _console, clock, renderer = _live(sample_count=1)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    renderer.report(
        _pass_finished(1, 1, label="bench", at_ms=5000),
    )
    clock.tick(6)
    renderer.report(
        JudgeFinished(
            primary_delta_pct=-3.2,
            regressed=("latency", "throughput"),
            at_ms=_ms(clock),
        )
    )
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_judge_alerting_and_confirm_running_does_show_bar(
    snapshot: SnapshotAssertion,
):
    """Judge shows ! glyph, confirm shows a bar for the filtered metrics."""
    _console, clock, renderer = _live(sample_count=5)

    renderer.report(JudgeFinished(primary_delta_pct=2.5, regressed=("latency",), at_ms=5000))
    renderer.report(ConfirmStarted(filtered_metrics=("latency",), at_ms=5100))
    clock.tick(6)
    renderer.report(
        _pass_started(
            1,
            5,
            target_index=0,
            target_count=2,
            label="baseline",
            at_ms=_ms(clock),
            phase="confirm",
        )
    )
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_recorded_improved_does_show_seq_and_outcome(
    snapshot: SnapshotAssertion,
):
    """You should see record node done with 'seq 3 . improved'."""
    _console, _clock, renderer = _live(seq=3)

    renderer.report(IterationRecorded(seq=3, outcome="improved", at_ms=15000))
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_recorded_unsettled_does_not_duplicate_outcome(
    snapshot: SnapshotAssertion,
):
    """Record shows 'seq 2 . unsettled' without duplicating the outcome word."""
    _console, _clock, renderer = _live(seq=2)

    renderer.report(IterationRecorded(seq=2, outcome="unsettled", at_ms=15000))
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_compact_layout_does_show_single_row(
    snapshot: SnapshotAssertion,
):
    """You should see compact single-row progress bar."""
    _console, clock, renderer = _live(height=10, sample_count=5)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    renderer.report(
        _pass_started(
            1,
            5,
            target_index=0,
            target_count=2,
            label="A",
            at_ms=_ms(clock),
        )
    )
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


# ---------------------------------------------------------------------------
# Single-node deltas (not full-frame golden snapshots)
# ---------------------------------------------------------------------------


def test_frame_when_confirm_reproduced_does_show_reproduced():
    _console, _clock, renderer = _live(sample_count=1)
    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("x",), at_ms=5000))
    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5100))
    renderer.report(ConfirmFinished(reproduced=True, at_ms=10000))

    result = frame_text(renderer.frame())

    assert "reproduced" in result
    assert "not reproduced" not in result
    renderer.stop()


def test_frame_when_confirm_not_reproduced_does_show_not_reproduced():
    _console, _clock, renderer = _live(sample_count=1)
    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("x",), at_ms=5000))
    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5100))
    renderer.report(ConfirmFinished(reproduced=False, at_ms=10000))

    result = frame_text(renderer.frame())

    assert "not reproduced" in result
    renderer.stop()


def test_frame_when_confirm_unfiltered_does_show_full_suite_label():
    _console, _clock, renderer = _live(sample_count=5)
    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("x",), at_ms=5000))
    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5100))

    result = frame_text(renderer.frame())

    assert "full suite" in result.lower()
    renderer.stop()


# ---------------------------------------------------------------------------
# Plain mode -- exact timestamped milestone lines
# ---------------------------------------------------------------------------


def test_plain_when_prepare_done_does_print_timestamped_line(
    snapshot: SnapshotAssertion,
):
    console, clock, renderer = _plain()

    renderer.report(PrepareStarted(label="baseline", at_ms=0))
    clock.tick(5)
    renderer.report(PrepareFinished(label="baseline", at_ms=_ms(clock)))

    assert _last_line(console) == snapshot
    renderer.stop()


def test_plain_when_passes_done_does_print_timestamped_line(
    snapshot: SnapshotAssertion,
):
    console, clock, renderer = _plain(sample_count=1)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    renderer.report(_pass_started(1, 1, at_ms=_ms(clock)))
    clock.tick(10)
    renderer.report(_pass_finished(1, 1, at_ms=_ms(clock)))

    assert _last_line(console) == snapshot
    renderer.stop()


def test_plain_when_judge_finished_does_print_timestamped_line(
    snapshot: SnapshotAssertion,
):
    console, clock, renderer = _plain(sample_count=1)

    clock.tick(6)
    renderer.report(JudgeFinished(primary_delta_pct=-2.0, regressed=(), at_ms=_ms(clock)))

    assert _last_line(console) == snapshot
    renderer.stop()


def test_plain_when_confirm_finished_does_print_timestamped_line(
    snapshot: SnapshotAssertion,
):
    console, clock, renderer = _plain(sample_count=1)

    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5000))
    clock.tick(10)
    renderer.report(ConfirmFinished(reproduced=True, at_ms=_ms(clock)))

    assert _last_line(console) == snapshot
    renderer.stop()


def test_plain_when_recorded_does_print_timestamped_line(
    snapshot: SnapshotAssertion,
):
    console, clock, renderer = _plain(sample_count=1)

    clock.tick(15)
    renderer.report(IterationRecorded(seq=2, outcome="improved", at_ms=_ms(clock)))

    assert _last_line(console) == snapshot
    renderer.stop()


def test_plain_when_any_event_does_not_emit_ansi_codes():
    console, clock, renderer = _plain()

    renderer.report(PrepareStarted(label="bench", at_ms=0))
    clock.tick(1)
    renderer.report(PrepareFinished(label="bench", at_ms=_ms(clock)))
    renderer.report(JudgeFinished(primary_delta_pct=-1.0, regressed=(), at_ms=_ms(clock)))
    renderer.report(IterationRecorded(seq=1, outcome="improved", at_ms=_ms(clock)))
    renderer.stop()

    output = _output(console)

    assert "\x1b[" not in output


# ---------------------------------------------------------------------------
# Live wiring -- Live attributes and refresh path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verbose", "expected_transient"),
    [
        pytest.param(False, True, id="verbose-off"),
        pytest.param(True, False, id="verbose-on"),
    ],
)
def test_live_wiring_when_created_does_set_transient_from_verbose(
    verbose: bool,
    expected_transient: bool,
):
    _console, _clock, renderer = _live(verbose=verbose)

    assert renderer._live is not None
    assert renderer._live.transient is expected_transient
    renderer.stop()


def test_refresh_live_when_called_does_render_from_frame(
    monkeypatch: pytest.MonkeyPatch,
):
    _console, _clock, renderer = _live()
    calls: list[object] = []
    original_frame = renderer.frame

    def spy_frame() -> object:
        result = original_frame()
        calls.append(result)
        return result

    monkeypatch.setattr(renderer, "frame", spy_frame)

    renderer._refresh_live()

    assert len(calls) == 1
    renderer.stop()


def test_live_mode_when_created_does_register_termination_cleanup_once(
    monkeypatch: pytest.MonkeyPatch,
):
    registered: list[object] = []
    monkeypatch.setattr(
        "gymrat_py.cli.iterate_progress.install_termination_cleanup",
        _fake_install(registered),
    )

    _console, _clock, renderer = _live()
    renderer.stop()

    assert len(registered) == 1


def test_plain_mode_when_created_does_not_register_termination_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    registered: list[object] = []
    monkeypatch.setattr(
        "gymrat_py.cli.iterate_progress.install_termination_cleanup",
        _fake_install(registered),
    )

    _console, _clock, renderer = _plain()
    renderer.stop()

    assert len(registered) == 0


def test_stop_when_called_twice_does_not_raise():
    _console, _clock, renderer = _live()
    renderer.report(PrepareStarted(label="bench", at_ms=0))

    renderer.stop()
    renderer.stop()

    assert renderer._live is None


# ---------------------------------------------------------------------------
# Signal cleanup
# ---------------------------------------------------------------------------


def test_clear_on_signal_when_live_up_does_leave_screen_blank(
    monkeypatch: pytest.MonkeyPatch,
):
    _console, _clock, renderer = _live()
    renderer.report(PrepareStarted(label="bench", at_ms=0))
    buf = StringIO()
    monkeypatch.setattr(sys, "stderr", buf)

    renderer._clear_on_signal()

    assert screen_lines(buf.getvalue()) == []


def test_clear_on_signal_when_after_stop_does_write_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    _console, _clock, renderer = _live()
    renderer.report(PrepareStarted(label="bench", at_ms=0))
    renderer.stop()
    buf = StringIO()
    monkeypatch.setattr(sys, "stderr", buf)

    renderer._clear_on_signal()

    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Fan-out on_progress callback
# ---------------------------------------------------------------------------


def test_create_fan_out_when_called_does_dispatch_event_to_all_subscribers():
    received_a: list[ProgressEvent] = []
    received_b: list[ProgressEvent] = []
    fan_out = create_fan_out([received_a.append, received_b.append])

    event = PrepareStarted(label="test", at_ms=0)
    fan_out(event)

    assert received_a == [event]
    assert received_b == [event]


def test_create_fan_out_when_subscriber_raises_does_still_call_remaining():
    received: list[ProgressEvent] = []

    def failing_subscriber(event: ProgressEvent) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    fan_out = create_fan_out([failing_subscriber, received.append])

    event = PrepareStarted(label="test", at_ms=0)
    fan_out(event)

    assert received == [event]


def test_create_fan_out_when_no_subscribers_does_not_raise():
    fan_out = create_fan_out([])

    event = PrepareStarted(label="test", at_ms=0)
    fan_out(event)
