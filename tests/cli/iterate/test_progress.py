"""Tests for the iterate progress renderer (live checklist + plain modes).

Tests inject a deterministic ``Clock`` from ``tests._rich`` and capture
output through ``sealed_console``.  Frame content is pinned with syrupy
golden snapshots; plain-mode milestones use exact-line equality; live wiring
assertions check ``Live`` attributes directly.
"""

from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING

import pytest

from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmSkipped,
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
from tests.cli._progress_helpers import iterate_renderer
from tests.cli._progress_helpers import ms_from_clock as _ms
from tests.cli._progress_helpers import pass_finished as _pass_finished
from tests.cli._progress_helpers import pass_started as _pass_started

if TYPE_CHECKING:
    from rich.console import Console, RenderableType
    from rich.segment import Segment
    from syrupy.assertion import SnapshotAssertion

    from gymrat.cli.iterate.progress import IterateRenderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_line(console: Console) -> str:
    lines = [ln for ln in console_output(console).splitlines() if ln.strip()]
    return lines[-1]


def _report_full_pass(
    renderer: IterateRenderer,
    clock: Clock[float],
    round_num: int,
    total_rounds: int,
    *,
    target_count: int = 1,
    label: str = "bench",
    duration_s: float,
) -> None:
    renderer.report(
        _pass_started(
            round_num,
            total_rounds,
            target_count=target_count,
            label=label,
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
            at_ms=_ms(clock),
        )
    )


_live = functools.partial(iterate_renderer, "live")
_plain = functools.partial(iterate_renderer, "plain")


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


def test_frame_when_before_hook_running_does_show_spinner(
    snapshot: SnapshotAssertion,
):
    _console, _clock, renderer = _live(has_before_hook=True, has_after_hook=True)

    renderer.report(HookStarted(stage="before", at_ms=0))
    result = frame_text(renderer.frame())

    assert result == snapshot


def test_frame_when_worktree_preparing_does_name_its_target(
    snapshot: SnapshotAssertion,
):
    _console, _clock, renderer = _live()

    renderer.report(PrepareStarted(label="baseline", at_ms=0))
    result = frame_text(renderer.frame())

    assert result == snapshot


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


def test_frame_when_judge_alerting_and_confirm_running_does_show_bar(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(sample_count=5)

    renderer.report(JudgeFinished(primary_delta_pct=2.5, regressed=("latency",), at_ms=5000))
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


def test_frame_when_compact_layout_does_show_single_row():
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

    assert result == "⠋ sampling                                            0% · A · 00:00/--:--"


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


def test_frame_when_judge_finished_after_started_does_show_elapsed(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(sample_count=1)

    renderer.report(PrepareFinished(label="bench", at_ms=0))
    renderer.report(_pass_finished(1, 1, label="bench", at_ms=5000))
    clock.tick(6)
    renderer.report(JudgeStarted(at_ms=_ms(clock)))
    clock.tick(4)
    renderer.report(JudgeFinished(primary_delta_pct=-3.2, regressed=("latency",), at_ms=_ms(clock)))

    result = frame_text(renderer.frame())

    assert result == snapshot


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
    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("x",), at_ms=5000))
    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5100))
    renderer.report(ConfirmFinished(reproduced=reproduced, at_ms=10000))

    result = frame_text(renderer.frame())

    assert result == snapshot


def test_frame_when_confirm_skipped_after_a_regression_does_keep_a_skipped_confirm_row(
    snapshot: SnapshotAssertion,
):
    _console, _clock, renderer = _live(sample_count=1)
    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("latency",), at_ms=5000))
    renderer.report(ConfirmSkipped(at_ms=5000))
    renderer.report(IterationRecorded(seq=1, outcome="regressed", at_ms=6000))

    result = frame_text(renderer.frame())

    assert result == snapshot


def test_frame_when_confirm_unfiltered_does_show_full_suite_label(snapshot: SnapshotAssertion):
    _console, _clock, renderer = _live(sample_count=5)
    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("x",), at_ms=5000))
    renderer.report(ConfirmStarted(filtered_metrics=None, at_ms=5100))

    result = frame_text(renderer.frame())

    assert result == snapshot


# ---------------------------------------------------------------------------
# Plain mode -- exact timestamped milestone lines
# ---------------------------------------------------------------------------


def test_report_when_plain_prepare_done_does_print_timestamped_line():
    console, clock, renderer = _plain()

    renderer.report(PrepareStarted(label="baseline", at_ms=0))
    clock.tick(5)
    renderer.report(PrepareFinished(label="baseline", at_ms=_ms(clock)))

    assert _last_line(console) == "[00:00:05] prepare baseline done (5s)"


def test_report_when_plain_mode_does_not_emit_ansi_codes():
    console, clock, renderer = _plain()

    renderer.report(PrepareStarted(label="bench", at_ms=0))
    clock.tick(1)
    renderer.report(PrepareFinished(label="bench", at_ms=_ms(clock)))
    renderer.report(JudgeFinished(primary_delta_pct=-1.0, regressed=(), at_ms=_ms(clock)))
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
def test_iterate_renderer_when_created_does_set_live_transient_from_verbose(
    verbose: bool,
    expected_transient: bool,
):
    _console, _clock, renderer = _live(verbose=verbose)

    assert renderer.live is not None
    assert renderer.live.transient is expected_transient


# ---------------------------------------------------------------------------
# warn -- above the live frame, on its own line in plain mode
# ---------------------------------------------------------------------------


def test_warn_when_live_mode_does_print_the_message_above_the_intact_frame():
    console, clock, renderer = _live()
    renderer.report(PrepareStarted(label="baseline", at_ms=0))

    renderer.warn("warning: disk full")

    frame = frame_text(renderer.frame(), get_time=clock)
    assert screen_lines(console_output(console)) == ["warning: disk full", *frame.splitlines()]


def test_warn_when_plain_mode_does_print_the_message_verbatim_on_its_own_line():
    console, _clock, renderer = _plain()
    before = console_output(console)

    renderer.warn("warning: cannot write [/tmp/progress.json]")

    assert console_output(console) == before + "warning: cannot write [/tmp/progress.json]\n"


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
    renderer.report(JudgeFinished(primary_delta_pct=-2.0, regressed=(), at_ms=_ms(clock)))

    result = frame_text(renderer.frame())

    assert result == snapshot


@pytest.mark.parametrize(
    ("primary_metric", "delta", "regressed", "expected"),
    [
        pytest.param(
            "geomean",
            -2.0,
            (),
            "[00:00:00] judge -2.0% on geomean · no gating regression",
            id="no-regression",
        ),
        pytest.param(
            "geomean",
            -6.8,
            ("latency",),
            "[00:00:00] judge -6.8% on geomean · 1 regressed: latency",
            id="one-regressed",
        ),
        pytest.param(
            "geomean",
            -6.8,
            ("latency", "alloc", "throughput", "parse"),
            "[00:00:00] judge -6.8% on geomean · 4 regressed: latency, alloc, throughput, …",
            id="more-regressed-than-cap",
        ),
        pytest.param(
            "geomean",
            None,
            ("latency",),
            "[00:00:00] judge — · 1 regressed: latency",
            id="missing-delta",
        ),
        pytest.param(
            "cpu:fire:total",
            2.0,
            ("lat:100:p99#time",),
            "[00:00:00] judge +2.0% on cpu:fire:total · 1 regressed: lat:100:p99#time",
            id="name-with-emoji-code",
        ),
    ],
)
def test_report_when_plain_judge_finished_does_print_exact_line(
    primary_metric: str, delta: float | None, regressed: tuple[str, ...], expected: str
):
    console, _clock, renderer = _plain(width=120, metric_count=5, primary_metric=primary_metric)

    renderer.report(JudgeFinished(primary_delta_pct=delta, regressed=regressed, at_ms=0))

    assert _last_line(console) == expected


@pytest.mark.parametrize(
    "events",
    [
        pytest.param((), id="pending"),
        pytest.param((JudgeStarted(at_ms=0),), id="running"),
        pytest.param(
            (JudgeFinished(primary_delta_pct=2.0, regressed=("lat:100:p99#time",), at_ms=0),),
            id="finished",
        ),
    ],
)
def test_frame_when_metric_name_has_emoji_code_does_print_the_name_literally(
    snapshot: SnapshotAssertion, events: tuple[JudgeStarted | JudgeFinished, ...]
):
    _console, _clock, renderer = _live(width=120, primary_metric="cpu:fire:total")
    for event in events:
        renderer.report(event)

    result = frame_text(renderer.frame(), width=120)

    assert result == snapshot


# ---------------------------------------------------------------------------
# Confirm done summary (#17)
# ---------------------------------------------------------------------------


def test_frame_when_confirm_finished_does_show_summary_on_node_line(snapshot: SnapshotAssertion):
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
    renderer.report(ConfirmFinished(reproduced=True, at_ms=_ms(clock)))

    result = frame_text(renderer.frame())

    assert result == snapshot


# ---------------------------------------------------------------------------
# Zero-width console
# ---------------------------------------------------------------------------


def test_iterate_renderer_when_console_width_zero_does_not_mount_live():
    _console, _clock, renderer = _live(width=0)

    assert renderer.live is None


def test_report_when_console_width_zero_does_emit_no_ansi():
    console, clock, renderer = _live(width=0)

    renderer.report(PrepareStarted(label="bench", at_ms=0))
    clock.tick(1)
    renderer.report(PrepareFinished(label="bench", at_ms=_ms(clock)))
    renderer.stop()

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

    assert result == "⠴ sampling ━━━━                                      10% · B · 00:02/00:20"


def test_frame_when_compact_confirm_started_does_restart_the_count_and_the_eta():
    _console, clock, renderer = _live(height=10, sample_count=1, metric_count=3)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    _report_full_pass(renderer, clock, 1, 1, duration_s=5)
    renderer.report(JudgeFinished(primary_delta_pct=-2.5, regressed=("latency",), at_ms=_ms(clock)))

    renderer.report(ConfirmStarted(filtered_metrics=("latency",), at_ms=_ms(clock)))

    result = frame_text(renderer.frame())

    assert result == "⠹ confirming                                            0%  00:00/00:00"


# ---------------------------------------------------------------------------
# Inline name formatting in judge row
# ---------------------------------------------------------------------------


def _style_name(segment: Segment) -> str:
    return str(segment.style) if segment.style else ""


def _judge_row_style_runs(renderable: RenderableType) -> list[tuple[str, str]]:
    """Render *renderable* in color; return its judge row as style-merged ``(text, style)`` runs."""
    styled = sealed_console(width=120, no_color=False, color_system="truecolor")
    lines = styled.render_lines(renderable, pad=False)
    judge_row = next(line for line in lines if "judged" in "".join(seg.text for seg in line))
    return [
        ("".join(seg.text for seg in segments), style)
        for style, segments in itertools.groupby(judge_row, key=_style_name)
    ]


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
    renderer.report(JudgeFinished(primary_delta_pct=-3.2, regressed=regressed, at_ms=_ms(clock)))

    result = _judge_row_style_runs(renderer.frame())

    assert result == snapshot
