"""Tests for the Rich renderables of the supervise dashboard frame.

The dashboard-styling tests verify that the TUI renders with appropriate colors
and styles instead of plain white text: they render ``reporter.frame()`` through
a color-enabled console and check the style of the one segment that holds the
content under test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
from rich.box import ROUNDED
from rich.style import Style

from gymrat.cli.style import STYLE_LABEL, STYLE_META
from gymrat.cli.supervise.progress import IDLE_WARN_MS
from gymrat.session.store import BestIteration
from gymrat.supervisor.exit_sequence import ExitPhase
from tests.cli.supervise._fixtures import (
    BASH_CYCLE_END_MS,
    ReporterKit,
    cap_event,
    color_console,
    fire_launch_and_bash_cycle,
    fire_launch_and_edit_cycle,
    fire_launch_and_iterate_start,
    follow_up_event,
    launch_event,
    launched,
    make_reporter,
    model_phase_event,
    reporter_showing_session,
    reporter_with_nested_read,
    session_state_three_iterations,
    turn_end_event,
)
from tests.session.records._fixtures import (
    make_iteration,
    session_state,
)

if TYPE_CHECKING:
    from rich.segment import Segment

    from gymrat.cli.supervise.progress import SuperviseReporter

# Frames are rendered straight from ``reporter.frame()``, so no test here mounts a
# real dashboard: a live ``ErasableLive`` would start a refresh thread painting on
# the process's stderr while the test moves the clock.
pytestmark = pytest.mark.usefixtures("mock_live_cls")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# The panel's border rows open on these corners; every other row is content.
_TOP_LEFT = ROUNDED.top_left
_BOTTOM_LEFT = ROUNDED.bottom_left

# Rich lays the panel's border style under its title, so the label keeps its own
# style on top of the border's.
_RENDERED_LABEL_STYLE = str(Style.parse(STYLE_META) + Style.parse(STYLE_LABEL))


def _is_border(line: list[Segment], corner: str) -> bool:
    """Whether a rendered *line* is the panel border row that opens on *corner*."""
    return "".join(segment.text for segment in line).startswith(corner)


def _styles_of(lines: list[list[Segment]], token: str) -> list[str]:
    """The styles of the segments of *lines* holding *token*, ``"none"`` when unstyled."""
    return [
        str(segment.style or Style.null())
        for line in lines
        for segment in line
        if token in segment.text
    ]


def _frame_lines(reporter: SuperviseReporter) -> list[list[Segment]]:
    """The reporter's current frame as rendered lines, through a color-enabled console."""
    console = color_console()
    return console.render_lines(reporter.frame(), console.options)


def _segment_style(reporter: SuperviseReporter, token: str) -> str:
    """Return the style of the one rendered content segment whose text holds *token*.

    The panel's top and bottom border rows are left out, so the title and
    border styling stay out of the result.

    Args:
        reporter: The reporter whose current frame is rendered.
        token: Text that exactly one rendered segment must contain.

    Returns:
        The segment's style as rich spells it, ``"none"`` when unstyled.

    Raises:
        AssertionError: When no segment, or more than one, holds *token*.
    """
    content = [
        line
        for line in _frame_lines(reporter)
        if not _is_border(line, _TOP_LEFT) and not _is_border(line, _BOTTOM_LEFT)
    ]
    styles = _styles_of(content, token)
    if len(styles) != 1:
        msg = f"expected one segment holding {token!r}, found {len(styles)}"
        raise AssertionError(msg)
    return styles[0]


def _title_styles(reporter: SuperviseReporter, token: str) -> set[str]:
    """Return the styles of the rendered panel-title segments whose text holds *token*.

    Args:
        reporter: The reporter whose current frame title is rendered.
        token: Text the segments must contain.

    Returns:
        Each distinct segment style as rich spells it, ``"none"`` when unstyled.
    """
    title_rows = [line for line in _frame_lines(reporter) if _is_border(line, _TOP_LEFT)]
    return set(_styles_of(title_rows, token))


# ---------------------------------------------------------------------------
# panel title styling
# ---------------------------------------------------------------------------


def test_panel_title_when_rendered_does_set_the_label_apart_from_its_dim_connectors():
    kit = launched(make_reporter())

    styles = {token: _title_styles(kit.reporter, token) for token in ("supervise", "·", "session")}

    assert styles == {
        "supervise": {_RENDERED_LABEL_STYLE},
        "·": {STYLE_META},
        "session": {STYLE_META},
    }


# ---------------------------------------------------------------------------
# summary row labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["cost", "loop", "best", "turns"])
def test_summary_row_when_rendered_with_color_does_open_on_a_dim_label(label: str):
    state = session_state_three_iterations(-3.2, "improved", seq=3)
    best = BestIteration(delta_pct=-3.2, seq=3, label="geomean")
    kit = reporter_showing_session(state, best=best)
    kit.reporter.observer(turn_end_event(4000))
    kit.reporter.observer(follow_up_event(5000, action="replied"))

    style = _segment_style(kit.reporter, label)

    assert style == "dim"


# ---------------------------------------------------------------------------
# loop row styling
# ---------------------------------------------------------------------------


def test_loop_iter_count_when_rendered_with_color_does_emit_bold_styling():
    state = session_state_three_iterations(-3.2, "improved")
    kit = reporter_showing_session(state, max_iterations=20)

    style = _segment_style(kit.reporter, "iterations")

    assert style == "bold dim"


@pytest.mark.parametrize(
    ("outcome", "delta_pct", "expected_style"),
    [
        pytest.param("regressed", 3.2, "dim red", id="regressed-red"),
        pytest.param("improved", -3.2, "dim green", id="improved-green"),
    ],
)
def test_loop_outcome_when_rendered_with_color_does_emit_expected_styling(
    outcome: str, delta_pct: float, expected_style: str
) -> None:
    state = session_state(
        iteration_count=1,
        last_iteration=make_iteration(delta_pct, outcome),
    )
    kit = reporter_showing_session(state)

    style = _segment_style(kit.reporter, outcome)

    assert style == expected_style


# ---------------------------------------------------------------------------
# best row styling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta_pct", "direction", "expected_style"),
    [
        pytest.param(-6.8, "lower", "dim green", id="lower-better-negative-delta-green"),
        pytest.param(3.5, "lower", "dim red", id="lower-better-positive-delta-red"),
        pytest.param(0.0, "lower", "dim red", id="lower-better-zero-delta-red"),
        pytest.param(12.0, "higher", "dim green", id="higher-better-gain-green"),
        pytest.param(-3.0, "higher", "dim red", id="higher-better-loss-red"),
    ],
)
def test_best_delta_when_rendered_with_color_does_style_an_improvement_green_per_direction(
    delta_pct: float, direction: Literal["lower", "higher"], expected_style: str
) -> None:
    kit = reporter_showing_session(
        session_state(iteration_count=1, keep_count=1),
        best=BestIteration(delta_pct=delta_pct, seq=1, label="primary", direction=direction),
    )

    style = _segment_style(kit.reporter, "%")

    assert style == expected_style


# ---------------------------------------------------------------------------
# liveness styling
# ---------------------------------------------------------------------------


def _fire_liveness_scenario(kit: ReporterKit, scenario: str) -> None:
    """Drive the reporter into the named liveness state."""
    if scenario in {"waiting", "no-output"}:
        fire_launch_and_bash_cycle(kit.reporter.observer, clock=kit.clock)
        if scenario == "no-output":
            kit.clock.now = BASH_CYCLE_END_MS + IDLE_WARN_MS + 1
        return
    if scenario == "in-flight":
        fire_launch_and_iterate_start(kit)
        kit.clock.now = 7000
        return
    kit.reporter.observer(launch_event(1000))
    if scenario == "composing":
        kit.reporter.observer(model_phase_event(2000, "tool_input", tool_name="Edit"))
    elif scenario == "responding":
        kit.reporter.observer(model_phase_event(2000, "responding"))
    elif scenario == "capped":
        kit.reporter.observer(cap_event("wall-clock"))
    elif scenario == "exiting-lock":
        kit.reporter.exit_phase(ExitPhase(kind="waiting-lock", pid=4242))
    elif scenario == "exiting-settling":
        kit.reporter.exit_phase(ExitPhase(kind="settling", pid=None))


@pytest.mark.parametrize(
    ("scenario", "needle", "expected_style"),
    [
        pytest.param("starting", "starting", "dim", id="starting-dim"),
        pytest.param("in-flight", "Bash", "none", id="in-flight-unstyled"),
        pytest.param("responding", "responding", "dim", id="responding-dim"),
        pytest.param("composing", "preparing", "dim", id="composing-dim"),
        pytest.param("waiting", "waiting", "dim", id="waiting-below-threshold-dim"),
        pytest.param("no-output", "no output", "yellow", id="waiting-above-threshold-yellow"),
        pytest.param("capped", "interrupting", "yellow", id="capped-yellow"),
        pytest.param("exiting-lock", "waiting for gymrat", "dim", id="exiting-waiting-lock-dim"),
        pytest.param("exiting-settling", "settling", "dim", id="exiting-settling-dim"),
    ],
)
def test_liveness_line_when_rendered_with_color_does_carry_its_state_styling(
    scenario: str, needle: str, expected_style: str
):
    kit = make_reporter()
    _fire_liveness_scenario(kit, scenario)

    style = _segment_style(kit.reporter, needle)

    assert style == expected_style


# ---------------------------------------------------------------------------
# finished tool lines styling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected_style"),
    [
        pytest.param("ok", "dim", id="ok-dim"),
        pytest.param("error", "dim red", id="failed-dim-red"),
    ],
)
def test_finished_tool_line_when_rendered_with_color_does_style_it_by_result(
    result: Literal["ok", "error"], expected_style: str
):
    kit = make_reporter()
    fire_launch_and_edit_cycle(kit, result=result)

    style = _segment_style(kit.reporter, "archetype")

    assert style == expected_style


# ---------------------------------------------------------------------------
# nested subagent line styling
# ---------------------------------------------------------------------------


def test_nested_tool_line_when_rendered_with_color_does_emit_dim_styling():
    kit = reporter_with_nested_read()

    style = _segment_style(kit.reporter, "config.ts")

    assert style == "dim"
