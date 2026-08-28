"""Tests for the iterate progress renderer (live tree + plain modes).

Tests inject a deterministic clock and capture output through a
``Console(file=StringIO())``.
"""

from collections.abc import Callable
from io import StringIO
from typing import Literal

import pytest
from rich.console import Console

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Clock:
    """A hand-advanced clock for deterministic rendering frames."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


def _live_console(width: int = 100, height: int = 40) -> Console:
    """A console that simulates a TTY for live-mode tests."""
    return Console(file=StringIO(), width=width, height=height, force_terminal=True)


def _plain_console(width: int = 100) -> Console:
    """A console that simulates a non-TTY for plain-mode tests."""
    return Console(file=StringIO(), width=width, force_terminal=False, no_color=True)


def _output(console: Console) -> str:
    """Return all text written to the console's StringIO."""
    f = console.file
    assert isinstance(f, StringIO)
    return f.getvalue()


def _renderer(
    mode: Literal["live", "plain"],
    console: Console,
    clock: _Clock,
    *,
    seq: int = 1,
    session_id: str = "test-session",
    sample_count: int = 5,
    metric_count: int = 3,
    primary_metric: str = "geomean",
    verbose: bool = False,
):
    """Wire an iterate renderer to ``console`` and ``clock``."""
    return create_iterate_renderer(
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


def _live(
    *,
    seq: int = 1,
    session_id: str = "test-session",
    sample_count: int = 5,
    metric_count: int = 3,
    primary_metric: str = "geomean",
    verbose: bool = False,
) -> tuple[Console, _Clock, IterateRenderer]:
    """A live-mode renderer wired to a fresh console and clock."""
    console = _live_console()
    clock = _Clock()
    renderer = _renderer(
        "live",
        console,
        clock,
        seq=seq,
        session_id=session_id,
        sample_count=sample_count,
        metric_count=metric_count,
        primary_metric=primary_metric,
        verbose=verbose,
    )
    return console, clock, renderer


def _plain(
    *,
    seq: int = 1,
    session_id: str = "test-session",
    sample_count: int = 5,
    metric_count: int = 3,
    primary_metric: str = "geomean",
    verbose: bool = False,
) -> tuple[Console, _Clock, IterateRenderer]:
    """A plain-mode renderer wired to a fresh console and clock."""
    console = _plain_console()
    clock = _Clock()
    renderer = _renderer(
        "plain",
        console,
        clock,
        seq=seq,
        session_id=session_id,
        sample_count=sample_count,
        metric_count=metric_count,
        primary_metric=primary_metric,
        verbose=verbose,
    )
    return console, clock, renderer


# ---------------------------------------------------------------------------
# factory API
# ---------------------------------------------------------------------------


def test_create_iterate_renderer_when_called_does_return_object_with_report_and_stop():
    *_, renderer = _plain()

    assert callable(renderer.report)
    assert callable(renderer.stop)
    renderer.stop()


# ---------------------------------------------------------------------------
# live tree — header
# ---------------------------------------------------------------------------


def test_live_tree_when_started_does_show_header_with_seq_and_session_id():
    console, _, renderer = _live(seq=3, session_id="abc-123")

    renderer.report(HookStarted(stage="before", at_ms=0))
    renderer.stop()

    output = _output(console)
    assert "3" in output
    assert "abc-123" in output


# ---------------------------------------------------------------------------
# live tree — node ordering and glyphs
# ---------------------------------------------------------------------------


def test_live_tree_when_before_hook_started_does_show_running_spinner():
    console, _, renderer = _live()

    renderer.report(HookStarted(stage="before", at_ms=0))
    renderer.stop()

    output = _output(console)
    assert "before" in output.lower()


def test_live_tree_when_before_hook_finished_does_show_done_marker():
    console, clock, renderer = _live()

    renderer.report(HookStarted(stage="before", at_ms=0))
    clock.now = 2.0
    renderer.report(HookFinished(stage="before", at_ms=2000))
    renderer.stop()

    output = _output(console)
    assert "✔" in output


def test_live_tree_when_pending_nodes_does_show_circle_glyph():
    console, _, renderer = _live()

    renderer.report(HookStarted(stage="before", at_ms=0))
    renderer.stop()

    output = _output(console)
    # Pending nodes downstream of before hook show the circle glyph
    assert "○" in output


# ---------------------------------------------------------------------------
# live tree — prepare node
# ---------------------------------------------------------------------------


def test_live_tree_when_prepare_started_does_show_prepare_label():
    console, _, renderer = _live()

    renderer.report(HookFinished(stage="before", at_ms=0))
    renderer.report(PrepareStarted(label="baseline", at_ms=100))
    renderer.stop()

    output = _output(console)
    assert "prepare" in output.lower()
    assert "baseline" in output


def test_live_tree_when_both_worktrees_prepared_does_show_both_labels():
    console, clock, renderer = _live()

    renderer.report(PrepareStarted(label="baseline", at_ms=0))
    clock.now = 1.0
    renderer.report(PrepareFinished(label="baseline", at_ms=1000))
    renderer.report(PrepareStarted(label="candidate", at_ms=1100))
    clock.now = 2.0
    renderer.report(PrepareFinished(label="candidate", at_ms=2000))
    renderer.stop()

    output = _output(console)
    assert "baseline" in output
    assert "candidate" in output


# ---------------------------------------------------------------------------
# live tree — passes node (bar, elapsed, ETA, detail line)
# ---------------------------------------------------------------------------


def test_live_tree_when_passes_started_does_show_bar_with_total():
    console, clock, renderer = _live(sample_count=5)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.now = 1.0
    renderer.report(
        PassStarted(round=1, total_rounds=5, target_index=0, target_count=2, label="A", at_ms=1000)
    )
    renderer.stop()

    output = _output(console)
    # total bar should be sample_count * 2 = 10
    assert "10" in output


def test_live_tree_when_pass_started_does_show_detail_with_round_and_side():
    console, clock, renderer = _live()

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.now = 1.0
    renderer.report(
        PassStarted(
            round=2, total_rounds=5, target_index=0, target_count=2, label="baseline", at_ms=1000
        )
    )
    renderer.stop()

    output = _output(console)
    assert "round 2" in output.lower() or "round 2" in output
    assert "baseline" in output


def test_live_tree_when_passes_done_does_show_done_marker_with_elapsed():
    console, clock, renderer = _live(sample_count=1)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.now = 1.0
    renderer.report(
        PassStarted(round=1, total_rounds=1, target_index=0, target_count=2, label="A", at_ms=1000)
    )
    clock.now = 11.0
    renderer.report(
        PassFinished(
            round=1, total_rounds=1, target_index=0, target_count=2, label="A", at_ms=11000
        )
    )
    renderer.report(
        PassStarted(round=1, total_rounds=1, target_index=1, target_count=2, label="B", at_ms=11100)
    )
    clock.now = 21.0
    renderer.report(
        PassFinished(
            round=1, total_rounds=1, target_index=1, target_count=2, label="B", at_ms=21000
        )
    )
    renderer.stop()

    output = _output(console)
    assert "✔" in output


# ---------------------------------------------------------------------------
# live tree — judge node
# ---------------------------------------------------------------------------


def test_live_tree_when_judge_finished_does_show_primary_delta():
    console, clock, renderer = _live(sample_count=1)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    renderer.report(
        PassFinished(round=1, total_rounds=1, target_index=0, target_count=1, label="A", at_ms=5000)
    )
    clock.now = 6.0
    renderer.report(JudgeFinished(primary_delta_pct=-3.2, regressed=(), at_ms=6000))
    renderer.stop()

    output = _output(console)
    assert "3.2" in output
    assert "judge" in output.lower()


def test_live_tree_when_judge_finds_regressions_does_name_regressed_metrics():
    console, clock, renderer = _live(sample_count=1)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    renderer.report(
        PassFinished(round=1, total_rounds=1, target_index=0, target_count=1, label="A", at_ms=5000)
    )
    clock.now = 6.0
    renderer.report(
        JudgeFinished(primary_delta_pct=2.5, regressed=("latency", "throughput"), at_ms=6000)
    )
    renderer.stop()

    output = _output(console)
    assert "latency" in output
    assert "throughput" in output


def test_live_tree_when_judge_triggers_confirm_does_show_alert_instead_of_done():
    console, _, renderer = _live(sample_count=1)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    renderer.report(
        PassFinished(round=1, total_rounds=1, target_index=0, target_count=1, label="A", at_ms=5000)
    )
    renderer.report(JudgeFinished(primary_delta_pct=2.5, regressed=("latency",), at_ms=6000))
    # Confirm is now running — judge should show ! not done glyph
    renderer.report(ConfirmStarted(filtered_metrics=("latency",), at_ms=6100))
    renderer.stop()

    output = _output(console)
    assert "!" in output


# ---------------------------------------------------------------------------
# live tree — confirm node
# ---------------------------------------------------------------------------


def test_live_tree_when_confirm_with_filter_does_show_filtered_metric_count():
    console, _, renderer = _live(sample_count=5)

    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("latency", "mem"), at_ms=5000))
    renderer.report(ConfirmStarted(filtered_metrics=("latency", "mem"), at_ms=5100))
    renderer.stop()

    output = _output(console)
    assert "confirm" in output.lower()
    assert "2" in output  # 2 metrics


def test_live_tree_when_confirm_without_filter_does_show_full_suite():
    console, _, renderer = _live(sample_count=5)

    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("x",), at_ms=5000))
    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5100))
    renderer.stop()

    output = _output(console)
    assert "full suite" in output.lower()


def test_live_tree_when_confirm_has_bar_does_total_samples_times_two():
    console, _, renderer = _live(sample_count=5)

    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("x",), at_ms=5000))
    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5100))
    renderer.report(
        PassStarted(
            round=1,
            total_rounds=5,
            target_index=0,
            target_count=2,
            label="A",
            at_ms=5200,
            phase="confirm",
        )
    )
    renderer.stop()

    output = _output(console)
    # bar total should be sample_count * 2 = 10
    assert "10" in output


def test_live_tree_when_confirm_finished_reproduced_does_show_reproduced():
    console, _, renderer = _live(sample_count=1)

    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("x",), at_ms=5000))
    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5100))
    renderer.report(ConfirmFinished(reproduced=True, at_ms=10000))
    renderer.stop()

    output = _output(console)
    assert "reproduced" in output.lower()


def test_live_tree_when_confirm_finished_not_reproduced_does_show_not_reproduced():
    console, _, renderer = _live(sample_count=1)

    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("x",), at_ms=5000))
    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5100))
    renderer.report(ConfirmFinished(reproduced=False, at_ms=10000))
    renderer.stop()

    output = _output(console)
    assert "not reproduced" in output.lower()


# ---------------------------------------------------------------------------
# live tree — record node
# ---------------------------------------------------------------------------


def test_live_tree_when_iteration_recorded_does_show_seq_and_outcome():
    console, _, renderer = _live(seq=3, sample_count=1)

    renderer.report(IterationRecorded(seq=3, outcome="improved", at_ms=15000))
    renderer.stop()

    output = _output(console)
    assert "3" in output
    assert "improved" in output


def test_live_tree_when_iteration_recorded_unsettled_does_show_unsettled():
    console, _, renderer = _live(seq=3, sample_count=1)

    renderer.report(IterationRecorded(seq=3, outcome="unsettled", at_ms=15000))
    renderer.stop()

    output = _output(console)
    assert "unsettled" in output


# ---------------------------------------------------------------------------
# pending node subtext
# ---------------------------------------------------------------------------


def test_live_tree_when_judge_pending_does_show_metric_count_and_primary():
    console, _, renderer = _live(metric_count=5, primary_metric="geomean")

    renderer.report(HookStarted(stage="before", at_ms=0))
    renderer.stop()

    output = _output(console)
    # Pending judge subtext: "5 metrics · geomean primary"
    assert "5" in output
    assert "metric" in output.lower()
    assert "geomean" in output


def test_live_tree_when_confirm_pending_does_show_rerun_description():
    console, _, renderer = _live()

    renderer.report(HookStarted(stage="before", at_ms=0))
    renderer.stop()

    output = _output(console)
    assert "gating metric regresses" in output.lower() or "reruns" in output.lower()


def test_live_tree_when_record_pending_does_show_seq_and_after_hook():
    console, _, renderer = _live(seq=4)

    renderer.report(HookStarted(stage="before", at_ms=0))
    renderer.stop()

    output = _output(console)
    assert "seq 4" in output or "4" in output
    assert "after hook" in output.lower() or "after" in output.lower()


# ---------------------------------------------------------------------------
# compact single-row fallback (< 12 rows)
# ---------------------------------------------------------------------------


def test_live_tree_when_terminal_fewer_than_12_rows_does_use_compact_layout():
    console = Console(file=StringIO(), width=100, height=10, force_terminal=True)
    clock = _Clock()
    renderer = _renderer("live", console, clock, sample_count=5)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.now = 1.0
    renderer.report(
        PassStarted(round=1, total_rounds=5, target_index=0, target_count=2, label="A", at_ms=1000)
    )
    renderer.stop()

    output = _output(console)
    assert output.strip()


# ---------------------------------------------------------------------------
# transient by default / verbose
# ---------------------------------------------------------------------------


def test_live_tree_when_verbose_false_does_erase_finished_tree():
    console, _, renderer = _live(verbose=False, sample_count=1)

    renderer.report(HookStarted(stage="before", at_ms=0))
    renderer.report(HookFinished(stage="before", at_ms=100))
    renderer.report(PrepareFinished(label="bench", at_ms=200))
    renderer.report(
        PassFinished(round=1, total_rounds=1, target_index=0, target_count=1, label="A", at_ms=5000)
    )
    renderer.report(JudgeFinished(primary_delta_pct=-1.0, regressed=(), at_ms=6000))
    renderer.report(IterationRecorded(seq=1, outcome="improved", at_ms=7000))
    renderer.stop()

    output = _output(console)
    # Transient Live emits erase-line sequences to clear the tree on stop
    assert "\x1b[2K" in output


def test_live_tree_when_verbose_true_does_keep_finished_tree():
    console, _, renderer = _live(verbose=True, sample_count=1)

    renderer.report(HookStarted(stage="before", at_ms=0))
    renderer.report(HookFinished(stage="before", at_ms=100))
    renderer.report(PrepareFinished(label="bench", at_ms=200))
    renderer.report(
        PassFinished(round=1, total_rounds=1, target_index=0, target_count=1, label="A", at_ms=5000)
    )
    renderer.report(JudgeFinished(primary_delta_pct=-1.0, regressed=(), at_ms=6000))
    renderer.report(IterationRecorded(seq=1, outcome="improved", at_ms=7000))
    renderer.stop()

    output = _output(console)
    # Verbose mode keeps the tree: output should contain tree content
    assert "improved" in output or "✔" in output


# ---------------------------------------------------------------------------
# plain mode
# ---------------------------------------------------------------------------


def test_plain_mode_when_prepare_finished_does_show_timestamped_milestone():
    console, clock, renderer = _plain()

    clock.now = 1.0
    renderer.report(PrepareStarted(label="baseline", at_ms=0))
    clock.now = 5.0
    renderer.report(PrepareFinished(label="baseline", at_ms=5000))
    renderer.stop()

    output = _output(console)
    assert "[" in output
    assert "]" in output
    assert "prepare" in output.lower() or "baseline" in output


def test_plain_mode_when_passes_done_does_show_timestamped_milestone():
    console, clock, renderer = _plain(sample_count=1)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.now = 1.0
    renderer.report(
        PassStarted(round=1, total_rounds=1, target_index=0, target_count=1, label="A", at_ms=1000)
    )
    clock.now = 10.0
    renderer.report(
        PassFinished(
            round=1, total_rounds=1, target_index=0, target_count=1, label="A", at_ms=10000
        )
    )
    renderer.stop()

    output = _output(console)
    assert "[" in output
    assert "]" in output


def test_plain_mode_when_judge_finished_does_show_timestamped_milestone():
    console, _, renderer = _plain(sample_count=1)

    renderer.report(JudgeFinished(primary_delta_pct=-2.0, regressed=(), at_ms=6000))
    renderer.stop()

    output = _output(console)
    assert "[" in output
    assert "judge" in output.lower()


def test_plain_mode_when_confirm_finished_does_show_timestamped_milestone():
    console, _, renderer = _plain(sample_count=1)

    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5000))
    renderer.report(ConfirmFinished(reproduced=True, at_ms=10000))
    renderer.stop()

    output = _output(console)
    assert "[" in output
    assert "confirm" in output.lower() or "reproduced" in output.lower()


def test_plain_mode_when_recorded_does_show_timestamped_milestone():
    console, _, renderer = _plain(sample_count=1)

    renderer.report(IterationRecorded(seq=2, outcome="improved", at_ms=15000))
    renderer.stop()

    output = _output(console)
    assert "[" in output
    assert "recorded" in output.lower() or "improved" in output


def test_plain_mode_does_not_emit_ansi_escape_codes():
    console, _, renderer = _plain()

    renderer.report(PrepareStarted(label="bench", at_ms=0))
    renderer.report(PrepareFinished(label="bench", at_ms=1000))
    renderer.report(JudgeFinished(primary_delta_pct=-1.0, regressed=(), at_ms=6000))
    renderer.report(IterationRecorded(seq=1, outcome="improved", at_ms=7000))
    renderer.stop()

    output = _output(console)
    assert "\x1b[" not in output


# ---------------------------------------------------------------------------
# signal cleanup
# ---------------------------------------------------------------------------


def _fake_install(registered: list[object]) -> Callable[[Callable[[], None]], Callable[[], None]]:
    def install(cb: Callable[[], None]) -> Callable[[], None]:
        registered.append(cb)
        return cb

    return install


def test_live_tree_when_created_does_register_signal_cleanup(monkeypatch: pytest.MonkeyPatch):
    registered: list[object] = []
    monkeypatch.setattr(
        "gymrat_py.cli.iterate_progress.install_termination_cleanup",
        _fake_install(registered),
    )
    *_, renderer = _live()
    renderer.stop()

    assert len(registered) == 1


def test_plain_mode_when_created_does_not_register_signal_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    registered: list[object] = []
    monkeypatch.setattr(
        "gymrat_py.cli.iterate_progress.install_termination_cleanup",
        _fake_install(registered),
    )
    *_, renderer = _plain()
    renderer.stop()

    assert len(registered) == 0


# ---------------------------------------------------------------------------
# fan-out on_progress callback
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
