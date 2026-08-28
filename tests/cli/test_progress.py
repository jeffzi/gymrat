"""Tests for the rich-based progress renderer (live + plain modes).

Tests inject a deterministic ``Clock`` from ``tests._rich`` and capture
output through ``sealed_console``.  Frame content is pinned with syrupy
golden snapshots; plain-mode milestones use exact-line equality; live wiring
assertions check ``Live`` attributes directly.
"""

from __future__ import annotations

import sys
from io import StringIO
from typing import TYPE_CHECKING, Literal

from gymrat_py.cli.progress import ProgressReporter, create_progress_reporter
from gymrat_py.progress_events import (
    HookStarted,
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
)
from tests._rich import Clock, frame_text, screen_lines, sealed_console

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest
    from rich.console import Console
    from syrupy.assertion import SnapshotAssertion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reporter(
    mode: Literal["live", "plain"],
    *,
    width: int = 80,
    height: int = 24,
    target_count: int = 1,
    sample_count: int = 3,
) -> tuple[Console, Clock, ProgressReporter]:
    """Wire a progress reporter to a sealed console and a hand-advanced clock."""
    clock = Clock()
    console = sealed_console(width=width, height=height)
    reporter = create_progress_reporter(
        mode=mode,
        console=console,
        target_count=target_count,
        sample_count=sample_count,
        clock=clock,
    )
    return console, clock, reporter


def _output(console: Console) -> str:
    """Return all text written to the console's StringIO."""
    f = console.file
    assert isinstance(f, StringIO)
    return f.getvalue()


def _ms(clock: Clock) -> int:
    """Return the clock's current time in milliseconds, for ``at_ms`` fields."""
    return int(clock.now * 1000)


def _pass_started(
    round_num: int,
    total_rounds: int,
    *,
    at_ms: int,
    target_index: int = 0,
    target_count: int = 1,
    label: str = "bench",
) -> PassStarted:
    return PassStarted(
        round=round_num,
        total_rounds=total_rounds,
        target_index=target_index,
        target_count=target_count,
        label=label,
        at_ms=at_ms,
    )


def _pass_finished(
    round_num: int,
    total_rounds: int,
    *,
    at_ms: int,
    target_index: int = 0,
    target_count: int = 1,
    label: str = "bench",
) -> PassFinished:
    return PassFinished(
        round=round_num,
        total_rounds=total_rounds,
        target_index=target_index,
        target_count=target_count,
        label=label,
        at_ms=at_ms,
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
# Frame golden snapshots via frame_text(reporter.frame())
# ---------------------------------------------------------------------------


def test_frame_when_prepare_running_does_show_spinner_and_label(
    snapshot: SnapshotAssertion,
):
    """You should see a spinner and the 'bench' prepare label."""
    _console, _clock, reporter = _reporter("live")
    reporter.report(PrepareStarted(label="bench", at_ms=0))

    result = frame_text(reporter.frame())

    assert result == snapshot
    reporter.stop()


def test_frame_when_prepare_done_and_first_pass_running_does_show_pending_eta(
    snapshot: SnapshotAssertion,
):
    """You should see prepare with a done marker, a progress bar, and 'estimating time left'."""
    _console, clock, reporter = _reporter("live")
    reporter.report(PrepareStarted(label="bench", at_ms=0))
    clock.tick(2)
    reporter.report(PrepareFinished(label="bench", at_ms=_ms(clock)))
    clock.tick(1)
    reporter.report(_pass_started(1, 3, at_ms=_ms(clock)))

    result = frame_text(reporter.frame())

    assert result == snapshot
    reporter.stop()


def test_frame_when_mid_run_with_computed_eta_does_show_detail_line(
    snapshot: SnapshotAssertion,
):
    """You should see a computed ETA (not 'estimating'), a detail line with round and label."""
    _console, clock, reporter = _reporter("live")
    reporter.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    reporter.report(_pass_started(1, 3, at_ms=_ms(clock)))
    clock.tick(10)
    reporter.report(_pass_finished(1, 3, at_ms=_ms(clock)))
    clock.tick(1)
    reporter.report(_pass_started(2, 3, at_ms=_ms(clock)))

    result = frame_text(reporter.frame())

    assert result == snapshot
    reporter.stop()


def test_frame_when_multi_target_compare_does_name_running_target(
    snapshot: SnapshotAssertion,
):
    """You should see bar total of samples x targets, running target named in detail."""
    _console, clock, reporter = _reporter("live", target_count=2, sample_count=5)
    reporter.report(PrepareFinished(label="main", at_ms=0))
    clock.tick(1)
    reporter.report(
        _pass_started(1, 5, target_index=1, target_count=2, label="candidate", at_ms=_ms(clock))
    )

    result = frame_text(reporter.frame())

    assert result == snapshot
    reporter.stop()


def test_frame_when_compact_layout_on_short_console_does_show_sample_count(
    snapshot: SnapshotAssertion,
):
    """You should see compact single-row progress with 'sample N/M'."""
    _console, clock, reporter = _reporter("live", height=10)
    reporter.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    reporter.report(_pass_started(1, 3, at_ms=_ms(clock)))

    result = frame_text(reporter.frame())

    assert result == snapshot
    reporter.stop()


# ---------------------------------------------------------------------------
# Plain mode -- exact timestamped milestone lines
# ---------------------------------------------------------------------------


def test_plain_renderer_when_prepare_finished_does_print_exact_timestamped_line():
    console, clock, reporter = _reporter("plain")
    reporter.report(PrepareStarted(label="bench", at_ms=0))
    clock.tick(5)
    reporter.report(PrepareFinished(label="bench", at_ms=_ms(clock)))

    output = _output(console)
    lines = [ln for ln in output.splitlines() if ln.strip()]

    assert lines[-1] == "[00:00:00] prepared bench (5s)"
    reporter.stop()


def test_plain_renderer_when_pass_finished_does_print_exact_timestamped_line(
    snapshot: SnapshotAssertion,
):
    console, clock, reporter = _reporter("plain", sample_count=3)
    reporter.report(PrepareFinished(label="bench", at_ms=0))
    reporter.report(_pass_started(1, 3, at_ms=0))
    clock.tick(20)
    reporter.report(_pass_finished(1, 3, at_ms=_ms(clock)))

    output = _output(console)
    lines = [ln for ln in output.splitlines() if ln.strip()]

    assert lines[-1] == snapshot
    reporter.stop()


def test_plain_renderer_when_any_event_does_not_emit_ansi_codes():
    console, clock, reporter = _reporter("plain")
    reporter.report(PrepareStarted(label="bench", at_ms=0))
    clock.tick(1)
    reporter.report(PrepareFinished(label="bench", at_ms=_ms(clock)))
    clock.tick(1)
    reporter.report(_pass_started(1, 3, at_ms=_ms(clock)))
    clock.tick(10)
    reporter.report(_pass_finished(1, 3, at_ms=_ms(clock)))
    reporter.stop()

    output = _output(console)

    assert "\x1b[" not in output


# ---------------------------------------------------------------------------
# Live wiring -- Live attributes and refresh path
# ---------------------------------------------------------------------------


def test_live_wiring_when_created_does_set_transient_and_no_redirect_stderr():
    _console, _clock, reporter = _reporter("live")

    assert reporter._live is not None
    assert reporter._live.transient is True
    assert reporter._live._redirect_stderr is False
    reporter.stop()


def test_refresh_live_when_called_does_render_from_frame(
    monkeypatch: pytest.MonkeyPatch,
):
    _console, _clock, reporter = _reporter("live")
    calls: list[object] = []
    original_frame = reporter.frame

    def spy_frame() -> object:
        result = original_frame()
        calls.append(result)
        return result

    monkeypatch.setattr(reporter, "frame", spy_frame)

    reporter._refresh_live()

    assert len(calls) == 1
    reporter.stop()


def test_warn_when_live_mode_does_route_through_console_print():
    console, _clock, reporter = _reporter("live")
    reporter.report(PrepareStarted(label="bench", at_ms=0))

    reporter.warn("heads up: slow disk")

    output = _output(console)
    assert "heads up: slow disk" in output
    reporter.stop()


def test_live_mode_when_created_does_register_termination_cleanup_once(
    monkeypatch: pytest.MonkeyPatch,
):
    registered: list[object] = []
    monkeypatch.setattr(
        "gymrat_py.cli.progress.install_termination_cleanup",
        _fake_install(registered),
    )

    _console, _clock, reporter = _reporter("live")
    reporter.stop()

    assert len(registered) == 1


def test_plain_mode_when_created_does_not_register_termination_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    registered: list[object] = []
    monkeypatch.setattr(
        "gymrat_py.cli.progress.install_termination_cleanup",
        _fake_install(registered),
    )

    _console, _clock, reporter = _reporter("plain")
    reporter.stop()

    assert len(registered) == 0


def test_stop_when_called_twice_does_not_raise_and_clears_live():
    _console, _clock, reporter = _reporter("live")
    reporter.report(PrepareStarted(label="bench", at_ms=0))

    reporter.stop()
    reporter.stop()

    assert reporter._live is None


# ---------------------------------------------------------------------------
# Summary line -- exact via injected clock
# ---------------------------------------------------------------------------


def test_stop_when_measure_done_does_print_summary(snapshot: SnapshotAssertion):
    console, clock, reporter = _reporter("live", sample_count=2)
    reporter.report(PrepareStarted(label="bench", at_ms=0))
    clock.tick(1)
    reporter.report(PrepareFinished(label="bench", at_ms=_ms(clock)))
    clock.tick(1)
    reporter.report(_pass_started(1, 2, at_ms=_ms(clock)))
    clock.tick(10)
    reporter.report(_pass_finished(1, 2, at_ms=_ms(clock)))
    clock.tick(1)
    reporter.report(_pass_started(2, 2, at_ms=_ms(clock)))
    clock.tick(10)
    reporter.report(_pass_finished(2, 2, at_ms=_ms(clock)))

    reporter.stop()

    visible = screen_lines(_output(console))
    summary = visible[-1] if visible else ""
    assert summary == snapshot


def test_stop_when_compare_done_does_print_summary(snapshot: SnapshotAssertion):
    console, clock, reporter = _reporter("live", target_count=2, sample_count=2)
    reporter.report(PrepareStarted(label="main", at_ms=0))
    clock.tick(1)
    reporter.report(PrepareFinished(label="main", at_ms=_ms(clock)))
    for i in range(4):
        target_idx = i % 2
        label = "main" if target_idx == 0 else "candidate"
        rnd = i // 2 + 1
        clock.tick(1)
        reporter.report(
            _pass_started(
                rnd,
                2,
                target_index=target_idx,
                target_count=2,
                label=label,
                at_ms=_ms(clock),
            )
        )
        clock.tick(10)
        reporter.report(
            _pass_finished(
                rnd,
                2,
                target_index=target_idx,
                target_count=2,
                label=label,
                at_ms=_ms(clock),
            )
        )

    reporter.stop()

    visible = screen_lines(_output(console))
    summary = visible[-1] if visible else ""
    assert summary == snapshot


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_live_renderer_when_console_width_zero_does_render_as_plain():
    console, _clock, reporter = _reporter("live", width=0)

    reporter.report(PrepareStarted(label="bench", at_ms=0))
    reporter.report(PrepareFinished(label="bench", at_ms=1000))
    reporter.stop()

    assert reporter._live is None
    output = _output(console)
    assert "\x1b[" not in output


def test_reporter_when_non_relevant_event_does_silently_ignore():
    console, _clock, reporter = _reporter("plain")

    reporter.report(HookStarted(stage="before", at_ms=0))

    output = _output(console)
    assert output == ""
    reporter.stop()


def test_clear_on_signal_when_live_up_does_leave_screen_blank(
    monkeypatch: pytest.MonkeyPatch,
):
    _console, _clock, reporter = _reporter("live")
    reporter.report(PrepareStarted(label="bench", at_ms=0))
    buf = StringIO()
    monkeypatch.setattr(sys, "stderr", buf)

    reporter._clear_on_signal()

    assert screen_lines(buf.getvalue()) == []


def test_clear_on_signal_when_after_stop_does_write_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    _console, _clock, reporter = _reporter("live")
    reporter.report(PrepareStarted(label="bench", at_ms=0))
    reporter.stop()
    buf = StringIO()
    monkeypatch.setattr(sys, "stderr", buf)

    reporter._clear_on_signal()

    assert buf.getvalue() == ""
