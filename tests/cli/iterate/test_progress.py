"""Tests for the iterate progress renderer (live checklist + plain modes).

Tests inject a deterministic ``Clock`` from ``tests._rich`` and capture
output through ``sealed_console``.  Frame content is pinned with syrupy
golden snapshots; plain-mode milestones use exact-line equality; live wiring
assertions check ``Live`` attributes directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from gymrat.cli.iterate.progress import IterateRenderer
from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmStarted,
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    JudgeStarted,
    PrepareFinished,
    PrepareStarted,
)
from tests._rich import (
    Clock,
    console_output,
    frame_text,
    screen_lines,
    sealed_console,
)
from tests.cli._progress_helpers import (
    ms_from_clock as _ms,
)
from tests.cli._progress_helpers import (
    pass_finished as _pass_finished,
)
from tests.cli._progress_helpers import (
    pass_started as _pass_started,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Literal

    from rich.console import Console, RenderableType
    from syrupy.assertion import SnapshotAssertion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_line(console: Console) -> str:
    lines = [ln for ln in console_output(console).splitlines() if ln.strip()]
    return lines[-1]


def _report_full_pass(
    renderer: IterateRenderer,
    clock: Clock,
    round_num: int,
    total_rounds: int,
    *,
    target_count: int = 1,
    label: str = "bench",
    phase: Literal["measure", "confirm"] = "measure",
    duration_s: float,
) -> None:
    renderer.report(
        _pass_started(
            round_num,
            total_rounds,
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
            target_count=target_count,
            label=label,
            phase=phase,
            at_ms=_ms(clock),
        )
    )


_live_renderers: list[IterateRenderer] = []


@pytest.fixture(autouse=True)
def _stop_renderers() -> Iterator[None]:
    """Stop every renderer a test built, so a failing test leaks no live display."""
    yield
    while _live_renderers:
        _live_renderers.pop().stop()


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
    renderer = IterateRenderer(
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
    _live_renderers.append(renderer)
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


# ---------------------------------------------------------------------------
# Frame golden snapshots via frame_text(renderer.frame())
# ---------------------------------------------------------------------------


def test_frame_when_initial_does_show_all_nodes_pending(
    snapshot: SnapshotAssertion,
):
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
    _console, _clock, renderer = _live(has_before_hook=True, has_after_hook=True)

    renderer.report(HookStarted(stage="before", at_ms=0))
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_worktree_preparing_does_name_its_target(
    snapshot: SnapshotAssertion,
):
    _console, _clock, renderer = _live()

    renderer.report(PrepareStarted(label="baseline", at_ms=0))
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_both_worktrees_prepared_does_show_elapsed(
    snapshot: SnapshotAssertion,
):
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
    _console, clock, renderer = _live(sample_count=5)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    _report_full_pass(renderer, clock, 1, 5, target_count=2, label="baseline", duration_s=10)
    clock.tick(1)
    _report_full_pass(renderer, clock, 1, 5, target_count=2, label="candidate", duration_s=10)
    clock.tick(1)
    renderer.report(
        _pass_started(
            2,
            5,
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
            metric_count=3,
            at_ms=_ms(clock),
        )
    )
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_judge_alerting_and_confirm_running_does_show_bar(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(sample_count=5)

    renderer.report(
        JudgeFinished(primary_delta_pct=2.5, regressed=("latency",), metric_count=3, at_ms=5000)
    )
    renderer.report(ConfirmStarted(filtered_metrics=("latency",), at_ms=5100))
    clock.tick(6)
    renderer.report(
        _pass_started(
            1,
            5,
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
    _console, _clock, renderer = _live(seq=3)

    renderer.report(IterationRecorded(seq=3, outcome="improved", at_ms=15000))
    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_compact_layout_does_show_single_row(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(height=10, sample_count=5)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    renderer.report(
        _pass_started(
            1,
            5,
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
    _console, clock, renderer = _live(sample_count=1)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    renderer.report(_pass_finished(1, 1, label="bench", at_ms=5000))
    clock.tick(6)
    renderer.report(JudgeStarted(at_ms=_ms(clock)))
    clock.tick(4)
    renderer.report(
        JudgeFinished(
            primary_delta_pct=-3.2, regressed=("latency",), metric_count=3, at_ms=_ms(clock)
        ),
    )

    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_recorded_with_checks_cmd_does_show_gymrat_keep(
    snapshot: SnapshotAssertion,
):
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


@pytest.mark.parametrize(
    "reproduced",
    [
        pytest.param(True, id="reproduced"),
        pytest.param(False, id="not-reproduced"),
    ],
)
def test_frame_when_confirm_finished_does_show_outcome(
    snapshot: SnapshotAssertion, reproduced: bool
):
    _console, _clock, renderer = _live(sample_count=1)
    renderer.report(
        JudgeFinished(primary_delta_pct=2.0, regressed=("x",), metric_count=3, at_ms=5000)
    )
    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5100))
    renderer.report(ConfirmFinished(reproduced=reproduced, at_ms=10000))

    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


def test_frame_when_confirm_unfiltered_does_show_full_suite_label(snapshot: SnapshotAssertion):
    _console, _clock, renderer = _live(sample_count=5)
    renderer.report(
        JudgeFinished(primary_delta_pct=2.0, regressed=("x",), metric_count=3, at_ms=5000)
    )
    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5100))

    result = frame_text(renderer.frame())

    assert result == snapshot
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
    _report_full_pass(renderer, clock, 1, 1, target_count=2, label="baseline", duration_s=10)
    _report_full_pass(renderer, clock, 1, 1, target_count=2, label="experiment", duration_s=10)

    assert _last_line(console) == snapshot
    renderer.stop()


def test_plain_when_judge_finished_does_print_timestamped_line(
    snapshot: SnapshotAssertion,
):
    console, clock, renderer = _plain(sample_count=1)

    clock.tick(6)
    renderer.report(
        JudgeFinished(primary_delta_pct=-2.0, regressed=(), metric_count=3, at_ms=_ms(clock))
    )

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
    renderer.report(
        JudgeFinished(primary_delta_pct=-1.0, regressed=(), metric_count=3, at_ms=_ms(clock))
    )
    renderer.report(IterationRecorded(seq=1, outcome="improved", at_ms=_ms(clock)))
    renderer.stop()

    output = console_output(console)

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

    assert renderer.live is not None
    assert renderer.live.transient is expected_transient
    renderer.stop()


# ---------------------------------------------------------------------------
# Ticking display (#14) — live auto-refresh and clock-driven frame updates
# ---------------------------------------------------------------------------


def test_live_wiring_when_created_does_set_auto_refresh_true():
    _console, _clock, renderer = _live()

    assert renderer.live is not None
    assert renderer.live.auto_refresh is True
    renderer.stop()


# ---------------------------------------------------------------------------
# warn -- above the live frame, on its own line in plain mode
# ---------------------------------------------------------------------------


def test_warn_when_live_mode_does_print_the_message_above_the_intact_frame():
    console, clock, renderer = _live()
    renderer.report(PrepareStarted(label="baseline", at_ms=0))

    renderer.warn("warning: disk full")

    frame = frame_text(renderer.frame(), get_time=clock)
    assert screen_lines(console_output(console)) == ["warning: disk full", *frame.splitlines()]
    renderer.stop()


def test_warn_when_plain_mode_does_print_the_message_verbatim_on_its_own_line():
    console, _clock, renderer = _plain()
    before = console_output(console)

    renderer.warn("warning: cannot write [/tmp/progress.json]")

    assert console_output(console) == before + "warning: cannot write [/tmp/progress.json]\n"
    renderer.stop()


# ---------------------------------------------------------------------------
# Judge verdicts
# ---------------------------------------------------------------------------


def test_frame_when_judge_finished_no_regressions_does_drop_confirm_and_show_verdict(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(sample_count=1, metric_count=4)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    renderer.report(_pass_finished(1, 1, label="bench", at_ms=5000))
    clock.tick(6)
    renderer.report(
        JudgeFinished(primary_delta_pct=-2.0, regressed=(), metric_count=4, at_ms=_ms(clock)),
    )

    result = frame_text(renderer.frame())

    assert result == snapshot
    renderer.stop()


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        pytest.param(2.2, "judged +2.2% on geomean", id="positive"),
        pytest.param(-1.3, "judged -1.3% on geomean", id="negative"),
        pytest.param(0.0, "judged 0.0% on geomean", id="zero"),
        pytest.param(0.04, "judged 0.0% on geomean", id="positive-rounds-to-zero"),
        pytest.param(-0.04, "judged 0.0% on geomean", id="negative-rounds-to-zero"),
        pytest.param(None, "judged —", id="missing"),
        pytest.param(math.nan, "judged —", id="nan"),
        pytest.param(math.inf, "judged —", id="positive-infinity"),
        pytest.param(-math.inf, "judged —", id="negative-infinity"),
    ],
)
def test_frame_when_judge_finished_does_print_delta_like_the_report(
    delta: float | None, expected: str
):
    _console, _clock, renderer = _live(sample_count=1)
    renderer.report(
        JudgeFinished(primary_delta_pct=delta, regressed=(), metric_count=3, at_ms=6000),
    )

    result = next(
        line.strip() for line in frame_text(renderer.frame()).splitlines() if "judged" in line
    )

    assert result == f"✓ {expected} · no gating regression"


def test_plain_when_judge_finished_with_regressions_does_print_count_and_names():
    console, clock, renderer = _plain(metric_count=5)
    clock.tick(6)
    renderer.report(
        JudgeFinished(
            primary_delta_pct=-6.8,
            regressed=("latency",),
            metric_count=5,
            at_ms=_ms(clock),
        ),
    )

    assert _last_line(console) == "[00:00:00] judge -6.8% · 4 improve/noise · 1 regressed: latency"
    renderer.stop()


def test_plain_when_more_regressions_than_cap_does_trail_off_after_three_names():
    console, clock, renderer = _plain(width=120, metric_count=5)
    clock.tick(6)
    renderer.report(
        JudgeFinished(
            primary_delta_pct=-6.8,
            regressed=("latency", "alloc", "throughput", "parse"),
            metric_count=5,
            at_ms=_ms(clock),
        ),
    )

    assert _last_line(console) == (
        "[00:00:00] judge -6.8% · 1 improve/noise · 4 regressed: latency, alloc, throughput, …"
    )
    renderer.stop()


def test_plain_when_judge_finished_does_use_event_metric_count_not_renderer_metric_count():
    console, clock, renderer = _plain(metric_count=0)

    clock.tick(6)
    renderer.report(
        JudgeFinished(
            primary_delta_pct=-2.5,
            regressed=("latency",),
            metric_count=5,
            at_ms=_ms(clock),
        ),
    )

    assert _last_line(console) == "[00:00:00] judge -2.5% · 4 improve/noise · 1 regressed: latency"
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
    _console, clock, renderer = _live(sample_count=2)
    renderer.report(
        JudgeFinished(primary_delta_pct=2.0, regressed=("x",), metric_count=3, at_ms=5000)
    )
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
# Zero-width console
# ---------------------------------------------------------------------------


def test_live_when_console_width_zero_does_render_as_plain():
    console, clock, renderer = _live(width=0)

    renderer.report(PrepareFinished(label="bench", at_ms=_ms(clock)))
    renderer.stop()

    assert renderer.live is None
    assert "\x1b[" not in console_output(console)


# ---------------------------------------------------------------------------
# Compact mode -- sampling passes
# ---------------------------------------------------------------------------


def test_frame_when_compact_pass_finished_does_advance_the_bar_and_show_an_eta():
    _console, clock, renderer = _live(height=10, sample_count=5)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    _report_full_pass(renderer, clock, 1, 5, label="A", duration_s=2)

    renderer.report(_pass_started(2, 5, target_count=1, label="B", at_ms=_ms(clock)))

    result = frame_text(renderer.frame())
    assert result.endswith(" 10% · B · 00:02/00:20")


def test_frame_when_compact_confirm_started_does_restart_the_count_and_the_eta():
    _console, clock, renderer = _live(height=10, sample_count=1, metric_count=3)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    _report_full_pass(renderer, clock, 1, 1, duration_s=5)
    renderer.report(
        JudgeFinished(
            primary_delta_pct=-2.5, regressed=("latency",), metric_count=3, at_ms=_ms(clock)
        )
    )

    renderer.report(ConfirmStarted(filtered_metrics=("latency",), at_ms=_ms(clock)))

    result = frame_text(renderer.frame())
    assert result.split()[1:] == ["confirming", "0%", "00:00/00:00"]


# ---------------------------------------------------------------------------
# Compact mode -- confirm phase
# ---------------------------------------------------------------------------


def test_frame_when_compact_confirm_started_does_reset_bar_for_rerun():
    _console, clock, renderer = _live(height=10, sample_count=1, metric_count=3)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    _report_full_pass(renderer, clock, 1, 1, target_count=1, duration_s=5)

    clock.tick(1)
    renderer.report(
        JudgeFinished(
            primary_delta_pct=-2.5,
            regressed=("latency",),
            metric_count=3,
            at_ms=_ms(clock),
        )
    )
    renderer.report(ConfirmStarted(filtered_metrics=("latency",), at_ms=_ms(clock)))
    clock.tick(1)
    renderer.report(
        _pass_started(
            1,
            2,
            target_count=1,
            label="baseline",
            at_ms=_ms(clock),
            phase="confirm",
        )
    )

    result = frame_text(renderer.frame())

    assert "confirm" in result.lower()
    assert "100%" not in result
    renderer.stop()


# ---------------------------------------------------------------------------
# Inline name formatting in judge row
# ---------------------------------------------------------------------------


def _judge_row_style_runs(renderable: RenderableType) -> list[tuple[str, str]]:
    """Render *renderable* in color and return the judge row as ``(text, style)`` runs.

    Adjacent segments sharing a style merge into one run, so the result pins
    what the user sees regardless of how the row's spans are split.
    """
    styled = sealed_console(width=120, no_color=False)
    lines = styled.render_lines(renderable, pad=False)
    judge_row = next(line for line in lines if "judged" in "".join(seg.text for seg in line))
    runs: list[tuple[str, str]] = []
    for seg in judge_row:
        style = str(seg.style) if seg.style else ""
        if runs:
            prev_text, prev_style = runs[-1]
            if prev_style == style:
                runs[-1] = (prev_text + seg.text, style)
            else:
                runs.append((seg.text, style))
        else:
            runs.append((seg.text, style))
    return runs


@pytest.mark.parametrize(
    "regressed",
    [
        pytest.param((), id="zero"),
        pytest.param(("latency",), id="one"),
        pytest.param(
            ("node/access#time", "parse[json]", "throughput", "alloc"),
            id="several-capped-with-bracket",
        ),
    ],
)
def test_frame_when_judge_finished_does_style_regressed_names_in_judge_row(
    snapshot: SnapshotAssertion,
    regressed: tuple[str, ...],
):
    _console, clock, renderer = _live(sample_count=1, metric_count=5)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    renderer.report(_pass_finished(1, 1, label="bench", at_ms=5000))
    clock.tick(6)
    renderer.report(
        JudgeFinished(
            primary_delta_pct=-3.2,
            regressed=regressed,
            metric_count=5,
            at_ms=_ms(clock),
        )
    )

    result = _judge_row_style_runs(renderer.frame())

    assert result == snapshot
