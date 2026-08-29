"""Tests for the iterate progress renderer (live checklist + plain modes).

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

from gymrat.cli.iterate_progress import (
    IterateRenderer,
    create_fan_out,
    create_iterate_renderer,
)
from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmStarted,
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    JudgeStarted,
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


def _report_full_pass(
    renderer: IterateRenderer,
    clock: Clock,
    round_num: int,
    total_rounds: int,
    *,
    target_index: int = 0,
    target_count: int = 1,
    label: str = "bench",
    phase: Literal["measure", "confirm"] = "measure",
    duration_s: float,
) -> None:
    renderer.report(
        _pass_started(
            round_num,
            total_rounds,
            target_index=target_index,
            target_count=target_count,
            label=label,
            phase=phase,
            at_ms=_ms(clock),
        )
    )
    clock.tick(duration_s)
    renderer.report(
        _pass_finished(
            round_num,
            total_rounds,
            target_index=target_index,
            target_count=target_count,
            label=label,
            phase=phase,
            at_ms=_ms(clock),
        )
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
    checks_cmd: str | None = None,
    has_before_hook: bool = False,
    has_after_hook: bool = False,
) -> tuple[Console, Clock, IterateRenderer]:
    clock = Clock()
    console = sealed_console(width=width, height=height, get_time=clock)
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
        checks_cmd=checks_cmd,
        has_before_hook=has_before_hook,
        has_after_hook=has_after_hook,
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
    checks_cmd: str | None = None,
    has_before_hook: bool = False,
    has_after_hook: bool = False,
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
        checks_cmd=checks_cmd,
        has_before_hook=has_before_hook,
        has_after_hook=has_after_hook,
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
    checks_cmd: str | None = None,
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
        checks_cmd=checks_cmd,
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
    """All checklist rows pending with their hints; header shows seq and session id."""
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
    """Before-hook row spins while running; the record hint names the after hook."""
    _console, _clock, renderer = _live(has_before_hook=True, has_after_hook=True)

    renderer.report(HookStarted(stage="before", at_ms=0))
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_both_worktrees_prepared_does_show_elapsed(
    snapshot: SnapshotAssertion,
):
    """Prepare row done with the elapsed accumulated across both worktrees."""
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


def test_frame_when_passes_mid_run_does_show_bar_count_and_clock(
    snapshot: SnapshotAssertion,
):
    """Sampling bar mid-run: count, target label, and a clock ticking past the last event."""
    _console, clock, renderer = _live(sample_count=5)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    _report_full_pass(
        renderer, clock, 1, 5, target_index=0, target_count=2, label="baseline", duration_s=10
    )
    clock.tick(1)
    _report_full_pass(
        renderer, clock, 1, 5, target_index=1, target_count=2, label="candidate", duration_s=10
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
    clock.tick(41)
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_judge_finished_does_show_delta_and_regressed(
    snapshot: SnapshotAssertion,
):
    """Judge done line carries the delta plus the regressed count and names."""
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


def test_frame_when_recorded_does_show_outcome_suggested(
    snapshot: SnapshotAssertion,
):
    """Record row reads 'recorded <outcome> suggested' with no seq text."""
    _console, _clock, renderer = _live(seq=3)

    renderer.report(IterationRecorded(seq=3, outcome="improved", at_ms=15000))
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


def test_frame_when_header_before_first_pass_completes_does_show_elapsed_without_eta(
    snapshot: SnapshotAssertion,
):
    """You should see elapsed in the header but no ETA before any pass finishes."""
    _console, clock, renderer = _live()

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(7)
    renderer.report(
        _pass_started(1, 5, label="baseline", at_ms=_ms(clock)),
    )
    clock.tick(3)

    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_judge_started_does_show_running_with_elapsed(
    snapshot: SnapshotAssertion,
):
    """You should see running judge spinner with elapsed ticking from JudgeStarted."""
    _console, clock, renderer = _live(sample_count=1)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    renderer.report(_pass_finished(1, 1, label="bench", at_ms=5000))
    clock.tick(6)
    renderer.report(JudgeStarted(at_ms=_ms(clock)))
    clock.tick(3)

    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_judge_finished_after_started_does_show_elapsed(
    snapshot: SnapshotAssertion,
):
    """You should see judge done node with elapsed computed from JudgeStarted to JudgeFinished."""
    _console, clock, renderer = _live(sample_count=1)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    renderer.report(_pass_finished(1, 1, label="bench", at_ms=5000))
    clock.tick(6)
    renderer.report(JudgeStarted(at_ms=_ms(clock)))
    clock.tick(4)
    renderer.report(
        JudgeFinished(primary_delta_pct=-3.2, regressed=("latency",), at_ms=_ms(clock)),
    )

    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_recorded_with_checks_cmd_does_show_gymrat_keep(
    snapshot: SnapshotAssertion,
):
    """You should see record node with 'checks (cmd) run at gymrat keep'."""
    _console, _clock, renderer = _live(
        seq=3,
        checks_cmd="npm run check && npm test",
    )

    renderer.report(IterationRecorded(seq=3, outcome="unsettled", at_ms=15000))

    result = frame_text(renderer.frame(), width=100)

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
    _report_full_pass(
        renderer, clock, 1, 1, target_index=0, target_count=2, label="baseline", duration_s=10
    )
    _report_full_pass(
        renderer, clock, 1, 1, target_index=1, target_count=2, label="experiment", duration_s=10
    )

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


def test_refresh_live_when_called_does_call_refresh_on_live(
    monkeypatch: pytest.MonkeyPatch,
):
    _console, _clock, renderer = _live()
    refreshed = False

    def spy_refresh() -> None:
        nonlocal refreshed
        refreshed = True

    monkeypatch.setattr(renderer._live, "refresh", spy_refresh)

    renderer._refresh_live()

    assert refreshed
    renderer.stop()


def test_live_mode_when_created_does_register_termination_cleanup_once(
    monkeypatch: pytest.MonkeyPatch,
):
    registered: list[object] = []
    monkeypatch.setattr(
        "gymrat.cli.iterate_progress.install_termination_cleanup",
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
        "gymrat.cli.iterate_progress.install_termination_cleanup",
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
# Ticking display (#14) — live auto-refresh and clock-driven frame updates
# ---------------------------------------------------------------------------


def test_live_wiring_when_created_does_set_auto_refresh_true():
    """Live is configured with auto_refresh so time-derived text ticks between events."""
    _console, _clock, renderer = _live()

    assert renderer._live is not None
    assert renderer._live.auto_refresh is True
    renderer.stop()


# ---------------------------------------------------------------------------
# Judge verdicts
# ---------------------------------------------------------------------------


def test_frame_when_judge_finished_no_regressions_does_drop_confirm_and_show_verdict(
    snapshot: SnapshotAssertion,
):
    """No regression: the confirm row leaves the checklist and the judge line carries the verdict."""
    _console, clock, renderer = _live(sample_count=1, metric_count=4)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    renderer.report(_pass_finished(1, 1, label="bench", at_ms=5000))
    clock.tick(6)
    renderer.report(
        JudgeFinished(primary_delta_pct=-2.0, regressed=(), at_ms=_ms(clock)),
    )

    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_plain_when_judge_finished_with_regressions_does_print_count_and_names():
    """Plain mode judge line carries delta, improve/noise count, and the regressed list."""
    console, clock, renderer = _plain(metric_count=5)
    clock.tick(6)
    renderer.report(
        JudgeFinished(
            primary_delta_pct=-6.8,
            regressed=("latency",),
            at_ms=_ms(clock),
        ),
    )

    assert _last_line(console) == "[00:00:00] judge -6.8% · 4 improve/noise · 1 regressed: latency"
    renderer.stop()


def test_plain_when_more_regressions_than_cap_does_trail_off_after_three_names():
    """The regressed list caps at three names and trails off; the count stays exact."""
    console, clock, renderer = _plain(width=120, metric_count=5)
    clock.tick(6)
    renderer.report(
        JudgeFinished(
            primary_delta_pct=-6.8,
            regressed=("latency", "alloc", "throughput", "parse"),
            at_ms=_ms(clock),
        ),
    )

    assert _last_line(console) == (
        "[00:00:00] judge -6.8% · 1 improve/noise · 4 regressed: latency, alloc, throughput, …"
    )
    renderer.stop()


# ---------------------------------------------------------------------------
# Confirm done summary (#17)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reproduced", "expected_fragment"),
    [
        pytest.param(True, "regressions reproduced", id="reproduced"),
        pytest.param(False, "regressions not reproduced", id="not-reproduced"),
    ],
)
def test_frame_when_confirm_finished_does_show_summary_on_node_line(
    reproduced: bool,
    expected_fragment: str,
):
    """Confirm done carries pass count and reproduced status; no stale sub-line text."""
    _console, clock, renderer = _live(sample_count=2)
    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("x",), at_ms=5000))
    renderer.report(ConfirmStarted(filtered_metrics=("x",), at_ms=5100))
    at = 5100
    for rnd in range(1, 3):
        for t_idx in range(2):
            lbl = "baseline" if t_idx == 0 else "experiment"
            at += 500
            renderer.report(
                _pass_started(
                    rnd,
                    2,
                    target_index=t_idx,
                    target_count=2,
                    label=lbl,
                    at_ms=at,
                    phase="confirm",
                ),
            )
            at += 500
            renderer.report(
                _pass_finished(
                    rnd,
                    2,
                    target_index=t_idx,
                    target_count=2,
                    label=lbl,
                    at_ms=at,
                    phase="confirm",
                ),
            )
    clock.tick(20)
    renderer.report(ConfirmFinished(reproduced=reproduced, at_ms=_ms(clock)))

    result = frame_text(renderer.frame())

    assert f"4/4 · {expected_fragment}" in result
    assert "estimating time left" not in result
    renderer.stop()


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
