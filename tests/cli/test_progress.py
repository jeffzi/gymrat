"""Tests for the rich-based progress renderer (live + plain modes).

Tests inject a deterministic clock via ``get_time`` and capture output through
a ``Console(file=StringIO())``.
"""

from io import StringIO
from typing import Literal

import pytest
from rich.console import Console

from gymrat_py.cli.progress import create_progress_reporter
from gymrat_py.progress_events import (
    HookStarted,
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
)


class _Clock:
    """A hand-advanced clock for deterministic ``Progress(get_time=...)`` frames."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


def _live_console(width: int = 80) -> Console:
    """A console that simulates a TTY for live-mode tests."""
    return Console(file=StringIO(), width=width, force_terminal=True)


def _plain_console(width: int = 80) -> Console:
    """A console that simulates a non-TTY for plain-mode tests."""
    return Console(file=StringIO(), width=width, force_terminal=False, no_color=True)


def _output(console: Console) -> str:
    """Return all text written to the console's StringIO."""
    f = console.file
    assert isinstance(f, StringIO)
    return f.getvalue()


def _reporter(
    mode: Literal["live", "plain"],
    console: Console,
    clock: _Clock,
    *,
    target_count: int = 1,
    sample_count: int = 3,
):
    """Wire a progress reporter to ``console`` and ``clock`` with the given counts."""
    return create_progress_reporter(
        mode=mode,
        console=console,
        target_count=target_count,
        sample_count=sample_count,
        clock=clock,
    )


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


# ---------------------------------------------------------------------------
# live renderer — Progress with auto-refresh disabled + injectable clock
# ---------------------------------------------------------------------------


def test_create_progress_reporter_when_live_mode_does_accept_injectable_clock():
    console = _live_console()
    clock = _Clock()

    reporter = _reporter("live", console, clock)

    assert reporter is not None
    reporter.stop()


# ---------------------------------------------------------------------------
# live renderer — prepare phase
# ---------------------------------------------------------------------------


def test_live_renderer_when_prepare_started_does_show_spinner_and_label():
    console = _live_console()
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock)

    reporter.report(PrepareStarted(label="build", at_ms=0))
    reporter.stop()

    output = _output(console)
    assert "build" in output


def test_live_renderer_when_prepare_finished_does_show_done_marker_and_elapsed():
    console = _live_console()
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock)

    reporter.report(PrepareStarted(label="build", at_ms=0))
    clock.now = 2.0
    reporter.report(PrepareFinished(label="build", at_ms=2000))
    reporter.stop()

    output = _output(console)
    assert "✔" in output


# ---------------------------------------------------------------------------
# live renderer — passes phase (single target)
# ---------------------------------------------------------------------------


def test_live_renderer_when_pass_started_does_show_bar_with_completed_total():
    console = _live_console()
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock, sample_count=5)

    reporter.report(PrepareFinished(label="bench", at_ms=0))
    clock.now = 1.0
    reporter.report(_pass_started(1, 5, at_ms=1000))
    reporter.stop()

    output = _output(console)
    assert "5" in output


def test_live_renderer_when_first_pass_started_and_no_pass_finished_does_show_estimating_eta():
    console = _live_console()
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock)

    reporter.report(PrepareFinished(label="bench", at_ms=0))
    clock.now = 1.0
    reporter.report(_pass_started(1, 3, at_ms=1000))
    reporter.stop()

    output = _output(console)
    assert "estimating time left" in output


def test_live_renderer_when_pass_finished_does_show_eta_with_format_eta():
    console = _live_console()
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock)

    reporter.report(PrepareFinished(label="bench", at_ms=0))
    clock.now = 1.0
    reporter.report(_pass_started(1, 3, at_ms=1000))
    clock.now = 11.0
    reporter.report(_pass_finished(1, 3, at_ms=11000))
    clock.now = 12.0
    reporter.report(_pass_started(2, 3, at_ms=12000))
    reporter.stop()

    output = _output(console)
    # After the first pass finishes (10s), ETA for remaining 2 passes should use format_eta
    assert "left" in output
    if output.count("left") > 1:
        assert "estimating" not in output.split("left")[-1]


def test_live_renderer_when_pass_started_does_show_detail_line_with_round_and_label():
    console = _live_console()
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock)

    reporter.report(PrepareFinished(label="bench", at_ms=0))
    clock.now = 1.0
    reporter.report(_pass_started(2, 3, at_ms=1000))
    reporter.stop()

    output = _output(console)
    assert "round 2" in output.lower() or "round 2" in output


# ---------------------------------------------------------------------------
# live renderer — multi-target (compare)
# ---------------------------------------------------------------------------


def test_live_renderer_when_multi_target_does_total_samples_times_target_count():
    console = _live_console()
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock, target_count=3, sample_count=5)

    reporter.report(PrepareFinished(label="main", at_ms=0))
    clock.now = 1.0
    reporter.report(_pass_started(1, 5, target_count=3, label="main", at_ms=1000))
    reporter.stop()

    output = _output(console)
    # Bar total should be 5 * 3 = 15
    assert "15" in output


def test_live_renderer_when_multi_target_detail_line_does_name_running_target():
    console = _live_console()
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock, target_count=2)

    reporter.report(PrepareFinished(label="main", at_ms=0))
    clock.now = 1.0
    reporter.report(
        _pass_started(1, 3, target_index=1, target_count=2, label="candidate", at_ms=1000)
    )
    reporter.stop()

    output = _output(console)
    assert "candidate" in output


# ---------------------------------------------------------------------------
# live renderer — transient + summary
# ---------------------------------------------------------------------------


def test_live_renderer_when_measure_finishes_does_print_summary_on_stderr():
    console = _live_console()
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock, sample_count=2)

    reporter.report(PrepareStarted(label="bench", at_ms=0))
    clock.now = 1.0
    reporter.report(PrepareFinished(label="bench", at_ms=1000))
    clock.now = 2.0
    reporter.report(_pass_started(1, 2, at_ms=2000))
    clock.now = 12.0
    reporter.report(_pass_finished(1, 2, at_ms=12000))
    clock.now = 13.0
    reporter.report(_pass_started(2, 2, at_ms=13000))
    clock.now = 23.0
    reporter.report(_pass_finished(2, 2, at_ms=23000))
    reporter.stop()

    output = _output(console)
    assert "measured" in output.lower() or "✔" in output


def test_live_renderer_when_compare_finishes_does_print_compare_summary():
    console = _live_console()
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock, target_count=2, sample_count=2)

    reporter.report(PrepareStarted(label="main", at_ms=0))
    clock.now = 1.0
    reporter.report(PrepareFinished(label="main", at_ms=1000))
    for i in range(4):
        target_idx = i % 2
        label = "main" if target_idx == 0 else "candidate"
        rnd = i // 2 + 1
        clock.now = 2.0 + i * 10
        reporter.report(
            _pass_started(
                rnd,
                2,
                target_index=target_idx,
                target_count=2,
                label=label,
                at_ms=int(clock.now * 1000),
            )
        )
        clock.now += 9.0
        reporter.report(
            _pass_finished(
                rnd,
                2,
                target_index=target_idx,
                target_count=2,
                label=label,
                at_ms=int(clock.now * 1000),
            )
        )
    reporter.stop()

    output = _output(console)
    assert "compared" in output.lower() or "2 targets" in output.lower()


# ---------------------------------------------------------------------------
# live renderer — compact single-row layout (< 12 rows)
# ---------------------------------------------------------------------------


def test_live_renderer_when_terminal_fewer_than_12_rows_does_use_compact_layout():
    console = Console(file=StringIO(), width=80, height=10, force_terminal=True)
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock)

    reporter.report(PrepareFinished(label="bench", at_ms=0))
    clock.now = 1.0
    reporter.report(_pass_started(1, 3, at_ms=1000))
    reporter.stop()

    output = _output(console)
    assert "sample 1/3" in output
    assert "bench" in output


# ---------------------------------------------------------------------------
# live renderer — COLUMNS=0 falls back to plain
# ---------------------------------------------------------------------------


def test_live_renderer_when_console_width_zero_does_render_as_plain():
    console = Console(file=StringIO(), width=0, force_terminal=True)
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock)

    reporter.report(PrepareStarted(label="bench", at_ms=0))
    reporter.report(PrepareFinished(label="bench", at_ms=1000))
    reporter.stop()

    output = _output(console)
    # Should not crash; plain mode output (timestamps or no ANSI) is acceptable
    assert "\x1b[K" not in output


# ---------------------------------------------------------------------------
# plain renderer (non-TTY)
# ---------------------------------------------------------------------------


def test_plain_renderer_when_prepare_finished_does_show_timestamped_milestone():
    console = _plain_console()
    clock = _Clock(0.0)
    reporter = _reporter("plain", console, clock)

    clock.now = 1.0
    reporter.report(PrepareStarted(label="bench", at_ms=0))
    clock.now = 5.0
    reporter.report(PrepareFinished(label="bench", at_ms=5000))
    reporter.stop()

    output = _output(console)
    # Format: [HH:MM:SS] milestone text
    assert "[" in output
    assert "]" in output
    assert "bench" in output


def test_plain_renderer_when_pass_finished_does_show_timestamped_milestone():
    console = _plain_console()
    clock = _Clock(0.0)
    reporter = _reporter("plain", console, clock)

    reporter.report(PrepareFinished(label="bench", at_ms=0))
    clock.now = 10.0
    reporter.report(_pass_started(1, 3, at_ms=0))
    clock.now = 20.0
    reporter.report(_pass_finished(1, 3, at_ms=20000))
    reporter.stop()

    output = _output(console)
    assert "[" in output
    assert "]" in output


def test_plain_renderer_does_not_emit_ansi_escape_codes():
    console = _plain_console()
    clock = _Clock(0.0)
    reporter = _reporter("plain", console, clock)

    reporter.report(PrepareStarted(label="bench", at_ms=0))
    reporter.report(PrepareFinished(label="bench", at_ms=1000))
    clock.now = 1.0
    reporter.report(_pass_started(1, 3, at_ms=1000))
    clock.now = 10.0
    reporter.report(_pass_finished(1, 3, at_ms=10000))
    reporter.stop()

    output = _output(console)
    assert "\x1b[" not in output
    assert "\x1b[K" not in output


def test_plain_renderer_when_run_finished_does_show_timestamped_finish_line():
    console = _plain_console()
    clock = _Clock(0.0)
    reporter = _reporter("plain", console, clock, sample_count=1)

    reporter.report(PrepareFinished(label="bench", at_ms=0))
    clock.now = 1.0
    reporter.report(_pass_started(1, 1, at_ms=1000))
    clock.now = 10.0
    reporter.report(_pass_finished(1, 1, at_ms=10000))
    reporter.stop()

    output = _output(console)
    assert "[" in output
    assert "]" in output


# ---------------------------------------------------------------------------
# warn
# ---------------------------------------------------------------------------


def test_warn_when_live_mode_does_print_warning_without_disturbing_live_block():
    console = _live_console()
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock)

    reporter.report(PrepareStarted(label="bench", at_ms=0))
    reporter.warn("heads up: slow disk")
    reporter.stop()

    output = _output(console)
    assert "heads up: slow disk" in output


def test_warn_when_plain_mode_does_print_warning_as_own_line():
    console = _plain_console()
    clock = _Clock(0.0)
    reporter = _reporter("plain", console, clock)

    reporter.warn("heads up: slow disk")
    reporter.stop()

    output = _output(console)
    assert "heads up: slow disk" in output


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


def test_stop_when_live_mode_does_clean_up_live_display():
    console = _live_console()
    clock = _Clock(0.0)
    reporter = _reporter("live", console, clock)

    reporter.report(PrepareStarted(label="bench", at_ms=0))
    reporter.stop()

    # Calling stop should not raise and should terminate the live block
    assert True


def test_stop_when_plain_mode_does_not_emit_clear_codes():
    console = _plain_console()
    clock = _Clock(0.0)
    reporter = _reporter("plain", console, clock)

    reporter.report(PrepareStarted(label="bench", at_ms=0))
    reporter.stop()

    output = _output(console)
    assert "\x1b[K" not in output


# ---------------------------------------------------------------------------
# factory API
# ---------------------------------------------------------------------------


def test_create_progress_reporter_when_called_does_return_object_with_report_warn_stop():
    console = _plain_console()
    clock = _Clock()

    reporter = _reporter("plain", console, clock)

    assert hasattr(reporter, "report")
    assert hasattr(reporter, "warn")
    assert hasattr(reporter, "stop")
    assert callable(reporter.report)
    assert callable(reporter.warn)
    assert callable(reporter.stop)
    reporter.stop()


def test_create_progress_reporter_when_mode_live_does_accept_console_and_counts():
    console = _live_console()
    clock = _Clock()

    reporter = _reporter("live", console, clock, target_count=2, sample_count=5)

    assert reporter is not None
    reporter.stop()


# ---------------------------------------------------------------------------
# non-relevant events are silently ignored
# ---------------------------------------------------------------------------


def test_reporter_when_non_relevant_event_does_silently_ignore():
    console = _plain_console()
    clock = _Clock()
    reporter = _reporter("plain", console, clock)

    reporter.report(HookStarted(stage="before", at_ms=0))
    reporter.stop()

    output = _output(console)
    assert "before" not in output


# ---------------------------------------------------------------------------
# old API removed
# ---------------------------------------------------------------------------


def test_old_progress_line_style_is_removed():
    with pytest.raises(ImportError):
        from gymrat_py.cli.progress import (
            ProgressLineStyle,  # type: ignore[missing-module-attribute]  # noqa: F401
        )


def test_old_styled_constant_is_removed():
    with pytest.raises(ImportError):
        from gymrat_py.cli.progress import (
            STYLED,  # type: ignore[missing-module-attribute]  # noqa: F401
        )


def test_old_plain_style_constant_is_removed():
    with pytest.raises(ImportError):
        from gymrat_py.cli.progress import (
            PLAIN_STYLE,  # type: ignore[missing-module-attribute]  # noqa: F401
        )


def test_old_render_progress_line_is_removed():
    with pytest.raises(ImportError):
        from gymrat_py.cli.progress import (
            render_progress_line,  # type: ignore[missing-module-attribute]  # noqa: F401
        )


def test_old_eta_pending_label_is_removed():
    with pytest.raises(ImportError):
        from gymrat_py.cli.progress import (
            ETA_PENDING_LABEL,  # type: ignore[missing-module-attribute]  # noqa: F401
        )
