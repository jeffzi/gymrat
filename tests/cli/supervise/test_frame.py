"""Tests for the Rich renderables of the supervise dashboard frame.

The dashboard-styling tests verify that the TUI renders with appropriate colors
and styles instead of plain white text: they render ``reporter.frame()`` through
a color-enabled console and check the style of the one segment that holds the
content under test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from gymrat.cli.style import STYLE_LABEL, STYLE_META
from gymrat.cli.supervise.progress import IDLE_WARN_MS
from gymrat.cli.supervise.types import BestIteration
from gymrat.supervisor.exit_sequence import ExitPhase
from tests._rich import sealed_console
from tests.cli.supervise._fixtures import (
    BASH_CYCLE_END_MS,
    FRAME_WIDTH,
    ReporterKit,
    cap_event,
    fire_launch_and_bash_cycle,
    fire_launch_and_bash_start,
    follow_up_event,
    launch_event,
    line_after,
    lines_containing,
    make_read_session,
    make_reporter,
    model_phase_event,
    render_frame,
    session_state_three_iterations,
    tool_end_event,
    tool_start_event,
    turn_end_event,
)
from tests.session.records._fixtures import (
    make_iteration,
    session_state,
)

if TYPE_CHECKING:
    from gymrat.cli.supervise.progress import SuperviseReporter
    from gymrat.config import Effort
    from gymrat.supervisor.events import ModelPhase


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _segment_style(reporter: SuperviseReporter, token: str) -> str:
    """Return the style of the one rendered content segment whose text holds *token*.

    Renders the panel's inner content, so the border styling stays out of the
    result, through a color-enabled console at the frame width.

    Args:
        reporter: The reporter whose current frame is rendered.
        token: Text that exactly one rendered segment must contain.

    Returns:
        The segment's style as rich spells it, ``"none"`` when unstyled.

    Raises:
        TypeError: When the frame is not a panel.
        AssertionError: When no segment, or more than one, holds *token*.
    """
    panel = reporter.frame()
    if not isinstance(panel, Panel):
        msg = f"frame is a {type(panel).__name__}, not a Panel"
        raise TypeError(msg)
    console = sealed_console(width=FRAME_WIDTH, no_color=False, color_system="standard")
    styles = [
        str(segment.style or Style.null())
        for line in console.render_lines(panel.renderable, console.options)
        for segment in line
        if token in segment.text
    ]
    if len(styles) != 1:
        msg = f"expected one segment holding {token!r}, found {len(styles)}"
        raise AssertionError(msg)
    return styles[0]


def _styles_at(text: Text, offset: int) -> set[str]:
    """The span styles that cover character *offset* of *text*."""
    return {str(span.style) for span in text.spans if span.start <= offset < span.end}


def _panel_title_text(frame: str) -> str:
    """The panel's title text, with the top border's box-drawing chars stripped."""
    return frame.splitlines()[0].strip("╭╮─ ")


def _content(line: str) -> str:
    """Strip the panel's side borders and padding from one frame line."""
    return line.strip("│").strip()


# ---------------------------------------------------------------------------
# panel title styling
# ---------------------------------------------------------------------------


def test_panel_title_when_rendered_does_set_the_label_apart_from_its_dim_connectors():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))

    panel = kit.reporter.frame()

    assert isinstance(panel, Panel)
    title = panel.title
    assert isinstance(title, Text)
    separator = title.plain.index(" · ")
    connector = title.plain.index("session")
    assert _styles_at(title, 0) == {STYLE_LABEL}
    assert (_styles_at(title, separator + 1), _styles_at(title, connector)) == (
        {STYLE_META},
        {STYLE_META},
    )


# ---------------------------------------------------------------------------
# summary row labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["cost", "loop", "best", "turns"])
def test_summary_row_when_rendered_with_color_does_open_on_a_dim_label(label: str):
    state = session_state_three_iterations(-3.2, "improved", seq=3)
    best = BestIteration(delta_pct=-3.2, seq=3, label="geomean")
    kit = make_reporter(read_session=make_read_session(state, has_baseline=True, best=best))
    fire_launch_and_bash_cycle(kit.reporter.observer)
    kit.reporter.observer(turn_end_event(4000))
    kit.reporter.observer(follow_up_event(5000, action="replied"))

    style = _segment_style(kit.reporter, label)

    assert style == "dim"


# ---------------------------------------------------------------------------
# loop row styling
# ---------------------------------------------------------------------------


def test_loop_iter_count_when_rendered_with_color_does_emit_bold_styling():
    state = session_state_three_iterations(-3.2, "improved")
    kit = make_reporter(
        max_iterations=20,
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

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
    kit = make_reporter(
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

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
    kit = make_reporter(
        read_session=make_read_session(
            session_state(iteration_count=1, keep_count=1),
            has_baseline=True,
            best=BestIteration(delta_pct=delta_pct, seq=1, label="primary", direction=direction),
        ),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

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
    kit.reporter.observer(launch_event(1000))
    if scenario == "composing":
        kit.reporter.observer(model_phase_event(2000, "tool_input", tool_name="Edit"))
    elif scenario == "responding":
        kit.reporter.observer(model_phase_event(2000, "responding"))
    elif scenario == "in-flight":
        kit.clock.now = 2000
        kit.reporter.observer(
            tool_start_event("Bash", "bash-1", 2000, input_summary="gymrat iterate")
        )
        kit.clock.now = 7000
    elif scenario == "capped":
        kit.reporter.observer(cap_event("wall-clock"))


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
    ],
)
def test_liveness_line_when_rendered_with_color_does_carry_its_state_styling(
    scenario: str, needle: str, expected_style: str
):
    kit = make_reporter()
    _fire_liveness_scenario(kit, scenario)

    style = _segment_style(kit.reporter, needle)

    assert style == expected_style


@pytest.mark.parametrize(
    ("phase", "needle"),
    [
        pytest.param(
            ExitPhase(kind="waiting-lock", pid=4242), "waiting for gymrat", id="waiting-lock"
        ),
        pytest.param(ExitPhase(kind="settling", pid=None), "settling", id="settling"),
    ],
)
def test_liveness_exiting_when_rendered_with_color_does_emit_dim_styling(
    phase: ExitPhase, needle: str
):
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.reporter.exit_phase(phase)

    style = _segment_style(kit.reporter, needle)

    assert style == "dim"


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
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(
        tool_start_event("Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    )
    kit.clock.now = 3000
    kit.reporter.observer(tool_end_event("Edit", "edit-1", 3000, result=result))

    style = _segment_style(kit.reporter, "archetype")

    assert style == expected_style


# ---------------------------------------------------------------------------
# nested subagent line
# ---------------------------------------------------------------------------


def _reporter_with_nested_read() -> ReporterKit:
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    observer(
        tool_start_event(
            "Read",
            "nested-read-1",
            2000,
            parent_tool_use_id="bash-1",
            input_summary="src/config.ts",
        )
    )
    kit.clock.now = 5000
    return kit


def test_nested_tool_when_in_flight_does_render_an_arrow_line_under_parent():
    kit = _reporter_with_nested_read()

    nested_line = line_after(render_frame(kit.reporter), "Bash")

    assert _content(nested_line) == "↳ Read src/config.ts  3s"


def test_nested_tool_line_when_rendered_with_color_does_emit_dim_styling():
    kit = _reporter_with_nested_read()

    style = _segment_style(kit.reporter, "config.ts")

    assert style == "dim"


@pytest.mark.parametrize(
    ("phase", "tool_name", "now", "expected"),
    [
        pytest.param("thinking", None, 4000, "↳ thinking  2s", id="thinking"),
        pytest.param("responding", None, 3000, "↳ responding  1s", id="responding"),
        pytest.param("tool_input", "Edit", 3000, "↳ preparing Edit  1s", id="composing"),
    ],
)
def test_nested_phase_when_reported_does_render_an_arrow_line_naming_it(
    phase: ModelPhase, tool_name: str | None, now: int, expected: str
):
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    observer(model_phase_event(2000, phase, tool_name=tool_name, parent_tool_use_id="bash-1"))
    kit.clock.now = now

    nested_line = line_after(render_frame(kit.reporter), "Bash")

    assert _content(nested_line) == expected


def test_nested_when_no_activity_does_end_the_panel_on_the_bash_row():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    kit.clock.now = 2000
    observer(tool_start_event("Bash", "bash-1", 2000, input_summary="gymrat iterate"))
    kit.clock.now = 5000

    row_after_bash = line_after(render_frame(kit.reporter), "Bash")

    assert row_after_bash == "╰" + "─" * (FRAME_WIDTH - 2) + "╯"


# ---------------------------------------------------------------------------
# tool-name column width — nested tools excluded
# ---------------------------------------------------------------------------


def test_tool_name_column_width_when_nested_tool_present_does_ignore_nested_width():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))

    kit.clock.now = 2000
    observer(tool_start_event("Bash", "bash-1", 2000, input_summary="run tests"))
    kit.clock.now = 2500
    observer(
        tool_start_event(
            "LongNestedToolName",
            "nested-1",
            2500,
            parent_tool_use_id="bash-1",
            input_summary="something",
        )
    )
    kit.clock.now = 3000

    frame = render_frame(kit.reporter)
    bash_lines = lines_containing(frame, "Bash")

    # "Bash" padded to the 5-column floor; counting the nested name would widen it.
    assert [_content(line) for line in bash_lines] == ["00:00:02  Bash   run tests  1s"]


# ---------------------------------------------------------------------------
# dashboard title — model and effort display
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "effort", "expected_title"),
    [
        pytest.param("opus", None, "supervise · model opus", id="model-only"),
        pytest.param(None, "high", "supervise · effort high", id="effort-only"),
        pytest.param("opus", "max", "supervise · model opus · effort max", id="model-and-effort"),
        pytest.param(None, None, "supervise", id="neither"),
    ],
)
def test_panel_title_when_model_or_effort_in_force_does_show_labelled_value(
    model: str | None, effort: Effort | None, expected_title: str
) -> None:
    kit = make_reporter(session_id="", branch="", model=model, effort=effort)
    kit.reporter.observer(launch_event(1000))

    frame = render_frame(kit.reporter)

    assert _panel_title_text(frame) == expected_title
