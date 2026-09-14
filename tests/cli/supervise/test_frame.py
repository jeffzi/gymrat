"""Tests for the Rich renderables of the supervise dashboard frame.

The dashboard-styling tests verify that the TUI renders with appropriate colors
and styles instead of plain white text: they render ``reporter.frame()`` through
a color-enabled console and check for ANSI escape codes on specific content
lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.panel import Panel

from gymrat.cli.supervise.state import ReadSessionResult
from gymrat.session.progress_file import ProgressSnapshot
from gymrat.supervisor.exit_sequence import ExitPhase
from tests._ansi import (
    SGR_BLUE,
    SGR_BOLD,
    SGR_CYAN,
    SGR_DIM,
    SGR_GREEN,
    SGR_RED,
    SGR_YELLOW,
    assert_has_sgr,
    has_sgr,
    strip_sgr,
)
from tests.cli.supervise._fixtures import (
    FRAME_WIDTH,
    IDLE_WARN_MS,
    ReporterKit,
    fire_cap,
    fire_launch,
    fire_launch_and_bash_cycle,
    fire_launch_and_bash_start,
    fire_model_phase,
    fire_thinking_update,
    fire_tool_end,
    fire_tool_start,
    fire_usage_update,
    make_iteration,
    make_read_session,
    make_reporter,
    render_colored,
    render_frame,
    session_state,
    session_state_three_iterations,
)

if TYPE_CHECKING:
    from gymrat.cli.supervise.progress import SuperviseReporter
    from gymrat.config import Effort


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _render_content_colored(reporter: SuperviseReporter, *, width: int = FRAME_WIDTH) -> str:
    """Render the panel's inner content with standard color.

    Extracts ``panel.renderable`` so Panel border styling does not leak
    into content-styling assertions.
    """
    panel = reporter.frame()
    assert isinstance(panel, Panel)
    return render_colored(panel.renderable, width=width)


def _lines_containing(output: str, needle: str) -> list[str]:
    """Return raw (styled) lines whose plain-text content contains *needle*."""
    return [line for line in output.splitlines() if needle in strip_sgr(line)]


def _line_after(frame: str, needle: str) -> str:
    """Return the line immediately following the first line containing *needle*."""
    lines = frame.splitlines()
    idx = next(i for i, line in enumerate(lines) if needle in line)
    return lines[idx + 1]


def _panel_title_text(frame: str) -> str:
    """The panel's title text, with the top border's box-drawing chars stripped."""
    return frame.splitlines()[0].strip("╭╮─ ")


def _assert_is_nested_line(line: str) -> None:
    """A nested subagent line is marked with an arrow (``↳`` or its ASCII fallback)."""
    assert "↳" in line or "->" in line


def _fire_waiting_bash_cycle(kit: ReporterKit, *, above_threshold: bool = False) -> None:
    """Launch, then run a Bash start/end cycle, optionally idling past the warn threshold."""
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000)
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Bash", "bash-1", 3000)
    if above_threshold:
        kit.clock.now = 3000 + IDLE_WARN_MS + 1


# ---------------------------------------------------------------------------
# panel border
# ---------------------------------------------------------------------------


def test_build_frame_panel_when_launched_does_have_nondefault_border_style():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    panel = kit.reporter.frame()

    assert isinstance(panel, Panel)
    assert panel.border_style != "none"


# ---------------------------------------------------------------------------
# panel title styling
# ---------------------------------------------------------------------------


def test_panel_title_when_label_present_does_style_supervise_and_label_with_label_style():
    kit = make_reporter(label="ecstatic-ts", session_id="", branch="")
    fire_launch(kit.reporter.observer, 1000)

    panel = kit.reporter.frame()
    assert isinstance(panel, Panel)
    colored = render_colored(panel)
    title_line = colored.splitlines()[0]

    assert_has_sgr([title_line], SGR_BOLD)
    assert_has_sgr([title_line], SGR_BLUE)


def test_panel_title_when_connector_present_does_dim_the_connector_word():
    kit = make_reporter(
        label="",
        session_id="20260813-125044-34ec",
        branch="gymrat/20260813-125044-34ec",
    )
    fire_launch(kit.reporter.observer, 1000)

    panel = kit.reporter.frame()
    assert isinstance(panel, Panel)
    colored = render_colored(panel)
    title_line = colored.splitlines()[0]

    assert_has_sgr([title_line], SGR_DIM)


# ---------------------------------------------------------------------------
# cost row styling
# ---------------------------------------------------------------------------


def test_cost_when_rendered_with_color_does_emit_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_usage_update(kit.reporter.observer, 4.12, 2000)

    colored = _render_content_colored(kit.reporter)
    cost_lines = _lines_containing(colored, "cost")

    assert cost_lines
    assert any("\x1b[" in line for line in cost_lines)


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

    colored = _render_content_colored(kit.reporter)
    loop_lines = _lines_containing(colored, "iter")

    assert_has_sgr(loop_lines, SGR_BOLD)


@pytest.mark.parametrize(
    ("outcome", "delta_pct", "expected_sgr"),
    [
        pytest.param("regressed", 3.2, SGR_RED, id="regressed-red"),
        pytest.param("improved", -3.2, SGR_GREEN, id="improved-green"),
    ],
)
def test_loop_outcome_when_rendered_with_color_does_emit_expected_styling(
    outcome: str, delta_pct: float, expected_sgr: int
) -> None:
    state = session_state(
        iteration_count=1,
        last_iteration=make_iteration(delta_pct, outcome),
    )
    kit = make_reporter(
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    colored = _render_content_colored(kit.reporter)
    outcome_lines = _lines_containing(colored, outcome)

    assert_has_sgr(outcome_lines, expected_sgr)


# ---------------------------------------------------------------------------
# best row styling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta_pct", "expected_sgr"),
    [
        pytest.param(-6.8, SGR_GREEN, id="negative-delta-green"),
        pytest.param(3.5, SGR_RED, id="positive-delta-red"),
    ],
)
def test_best_delta_when_rendered_with_color_does_emit_sign_dependent_styling(
    delta_pct: float, expected_sgr: int
) -> None:
    outcome = "improved" if delta_pct < 0 else "regressed"
    state = session_state(
        iteration_count=3,
        keep_count=1,
        discard_count=2,
        last_iteration=make_iteration(delta_pct, outcome, seq=3),
    )
    kit = make_reporter(
        read_session=make_read_session(
            state,
            has_baseline=True,
            best_delta_pct=delta_pct,
            best_seq=3,
        ),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    colored = _render_content_colored(kit.reporter)
    best_lines = _lines_containing(colored, "best")

    assert_has_sgr(best_lines, expected_sgr)


# ---------------------------------------------------------------------------
# liveness styling
# ---------------------------------------------------------------------------


def test_liveness_starting_when_rendered_with_color_does_emit_dim_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    colored = _render_content_colored(kit.reporter)
    starting_lines = _lines_containing(colored, "starting")

    assert_has_sgr(starting_lines, SGR_DIM)


def test_liveness_inflight_when_rendered_with_color_does_not_emit_special_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000, input_summary="gymrat iterate")
    kit.clock.now = 7000

    colored = _render_content_colored(kit.reporter)
    bash_lines = _lines_containing(colored, "Bash")

    assert bash_lines
    assert not any(has_sgr(line, SGR_BOLD) for line in bash_lines)
    assert not any(has_sgr(line, SGR_DIM) for line in bash_lines)
    assert not any(has_sgr(line, SGR_CYAN) for line in bash_lines)


def test_liveness_inflight_when_rendered_does_match_finished_tool_column_layout():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000, input_summary="gymrat iterate")
    kit.clock.now = 7000

    plain = render_frame(kit.reporter)
    bash_lines = _lines_containing(plain, "Bash")

    assert bash_lines, "expected at least one line containing 'Bash'"
    inflight_line = bash_lines[0]
    expected_wall = "00:00:02"
    assert expected_wall in inflight_line, f"wall-clock {expected_wall!r} not in {inflight_line!r}"
    assert "Bash" in inflight_line
    assert "gymrat iterate" in inflight_line
    assert "5s" in inflight_line


def test_liveness_responding_when_rendered_with_color_does_emit_dim_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    fire_model_phase(kit.reporter.observer, 2000, "responding")

    colored = _render_content_colored(kit.reporter)
    responding_lines = _lines_containing(colored, "responding")

    assert_has_sgr(responding_lines, SGR_DIM)


def test_liveness_composing_when_rendered_with_color_does_emit_dim_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    fire_model_phase(kit.reporter.observer, 2000, "tool_input", tool_name="Edit")

    colored = _render_content_colored(kit.reporter)
    preparing_lines = _lines_containing(colored, "preparing")

    assert_has_sgr(preparing_lines, SGR_DIM)


def test_liveness_waiting_when_below_threshold_rendered_with_color_does_emit_dim_styling():
    kit = make_reporter()
    _fire_waiting_bash_cycle(kit)

    colored = _render_content_colored(kit.reporter)
    waiting_lines = _lines_containing(colored, "waiting")

    assert_has_sgr(waiting_lines, SGR_DIM)


def test_liveness_waiting_when_above_threshold_rendered_with_color_does_emit_yellow_styling():
    kit = make_reporter()
    _fire_waiting_bash_cycle(kit, above_threshold=True)

    colored = _render_content_colored(kit.reporter)
    no_output_lines = _lines_containing(colored, "no output")

    assert_has_sgr(no_output_lines, SGR_YELLOW)


@pytest.mark.parametrize(
    ("phase", "needle"),
    [
        pytest.param("responding", "responding", id="responding"),
        pytest.param("tool_input", "preparing", id="composing-as-preparing"),
    ],
)
def test_liveness_phase_when_color_off_does_not_emit_sgr(phase: str, needle: str):
    kit = make_reporter(color=False)
    fire_launch(kit.reporter.observer, 1000)
    tool_name = "Edit" if phase == "tool_input" else None
    fire_model_phase(kit.reporter.observer, 2000, phase, tool_name=tool_name)

    colored = _render_content_colored(kit.reporter)
    phase_lines = _lines_containing(colored, needle)

    assert phase_lines
    assert not any("\x1b[" in line for line in phase_lines)


def test_liveness_waiting_when_below_threshold_color_off_does_not_emit_sgr():
    kit = make_reporter(color=False)
    _fire_waiting_bash_cycle(kit)

    colored = _render_content_colored(kit.reporter)
    waiting_lines = _lines_containing(colored, "waiting")

    assert waiting_lines
    assert not any("\x1b[" in line for line in waiting_lines)


def test_liveness_waiting_when_above_threshold_color_off_does_not_emit_sgr():
    kit = make_reporter(color=False)
    _fire_waiting_bash_cycle(kit, above_threshold=True)

    colored = _render_content_colored(kit.reporter)
    no_output_lines = _lines_containing(colored, "no output")

    assert no_output_lines
    assert not any("\x1b[" in line for line in no_output_lines)


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
    fire_launch(kit.reporter.observer, 1000)
    kit.reporter.exit_phase(phase)

    colored = _render_content_colored(kit.reporter)
    exiting_lines = _lines_containing(colored, needle)

    assert_has_sgr(exiting_lines, SGR_DIM)


def test_liveness_capped_when_rendered_with_color_does_emit_yellow_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    fire_cap(kit.reporter.observer, "wall-clock")

    colored = _render_content_colored(kit.reporter)
    cap_lines = _lines_containing(colored, "interrupting")

    assert_has_sgr(cap_lines, SGR_YELLOW)


# ---------------------------------------------------------------------------
# finished tool lines ordering
# ---------------------------------------------------------------------------


def test_finished_tools_when_three_completed_does_render_newest_first():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Read", "read-1", 2000, input_summary="oldest.ts")
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Read", "read-1", 3000)

    kit.clock.now = 4000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 4000, input_summary="middle.ts")
    kit.clock.now = 5000
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 5000)

    kit.clock.now = 6000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 6000, input_summary="newest.ts")
    kit.clock.now = 7000
    fire_tool_end(kit.reporter.observer, "Bash", "bash-1", 7000)

    plain = render_frame(kit.reporter)
    tool_lines = [
        line
        for line in plain.splitlines()
        if any(name in line for name in ("oldest.ts", "middle.ts", "newest.ts"))
    ]

    assert len(tool_lines) == 3, (
        f"expected 3 tool history lines, got {len(tool_lines)}: {tool_lines}"
    )
    assert "newest.ts" in tool_lines[0], f"first line should be newest: {tool_lines[0]}"
    assert "oldest.ts" in tool_lines[-1], f"last line should be oldest: {tool_lines[-1]}"


# ---------------------------------------------------------------------------
# finished tool lines styling
# ---------------------------------------------------------------------------


def test_finished_tool_line_does_emit_dim_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 3000)

    colored = _render_content_colored(kit.reporter)
    finished_lines = _lines_containing(colored, "archetype")

    assert_has_sgr(finished_lines, SGR_DIM)


def test_finished_tool_line_when_failed_does_emit_dim_red_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 3000, result="error")

    colored = _render_content_colored(kit.reporter)
    edit_lines = _lines_containing(colored, "Edit")

    assert_has_sgr(edit_lines, SGR_RED)
    assert_has_sgr(edit_lines, SGR_DIM)


# ---------------------------------------------------------------------------
# nested subagent line
# ---------------------------------------------------------------------------


def test_nested_tool_when_in_flight_does_render_arrow_line_under_parent():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    fire_tool_start(
        observer,
        "Read",
        "nested-read-1",
        2000,
        parent_tool_use_id="bash-1",
        input_summary="src/config.ts",
    )
    kit.clock.now = 5000

    nested_line = _line_after(render_frame(kit.reporter), "Bash")

    _assert_is_nested_line(nested_line)
    assert "Read" in nested_line
    assert "src/config.ts" in nested_line
    assert "3s" in nested_line


def test_nested_phase_when_thinking_does_render_arrow_line_with_thinking():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    fire_model_phase(observer, 2000, "thinking", parent_tool_use_id="bash-1")
    kit.clock.now = 4000

    nested_line = _line_after(render_frame(kit.reporter), "Bash")

    _assert_is_nested_line(nested_line)
    assert "thinking" in nested_line
    assert "2s" in nested_line


def test_nested_phase_when_responding_does_render_arrow_line_with_responding():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    fire_model_phase(observer, 2000, "responding", parent_tool_use_id="bash-1")
    kit.clock.now = 3000

    nested_line = _line_after(render_frame(kit.reporter), "Bash")

    _assert_is_nested_line(nested_line)
    assert "responding" in nested_line


def test_nested_phase_when_composing_does_render_arrow_line_with_preparing():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    fire_model_phase(observer, 2000, "tool_input", tool_name="Edit", parent_tool_use_id="bash-1")
    kit.clock.now = 3000

    nested_line = _line_after(render_frame(kit.reporter), "Bash")

    _assert_is_nested_line(nested_line)
    assert "preparing" in nested_line
    assert "Edit" in nested_line


def test_nested_when_no_activity_does_not_render_arrow_line():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(observer, "Bash", "bash-1", 2000, input_summary="gymrat iterate")
    kit.clock.now = 5000

    frame = render_frame(kit.reporter)

    assert "↳" not in frame
    assert "->" not in frame or "gymrat" in frame


def test_nested_tool_when_rendered_with_color_does_emit_dim_styling():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    fire_tool_start(
        observer,
        "Read",
        "nested-read-1",
        2000,
        parent_tool_use_id="bash-1",
        input_summary="src/config.ts",
    )
    kit.clock.now = 5000

    colored = _render_content_colored(kit.reporter)
    nested_lines = _lines_containing(colored, "config.ts")

    assert_has_sgr(nested_lines, SGR_DIM)


# ---------------------------------------------------------------------------
# tool-name column width — nested tools excluded
# ---------------------------------------------------------------------------


def test_tool_name_column_width_when_nested_tool_present_does_ignore_nested_width():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)

    kit.clock.now = 2000
    fire_tool_start(observer, "Bash", "bash-1", 2000, input_summary="run tests")
    kit.clock.now = 2500
    fire_tool_start(
        observer,
        "LongNestedToolName",
        "nested-1",
        2500,
        parent_tool_use_id="bash-1",
        input_summary="something",
    )
    kit.clock.now = 3000

    frame = render_frame(kit.reporter)
    bash_lines = _lines_containing(frame, "Bash")

    assert bash_lines
    assert "LongNestedToolName" not in bash_lines[0]


# ---------------------------------------------------------------------------
# no token count on non-Thinking states
# ---------------------------------------------------------------------------


def test_liveness_responding_when_rendered_does_not_show_token_count():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    fire_thinking_update(observer, 1500, estimated_tokens=500)
    fire_model_phase(observer, 2000, "responding")

    frame = render_frame(kit.reporter)

    assert "responding" in frame
    assert "500" not in frame


def test_liveness_composing_when_rendered_does_not_show_token_count():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    fire_thinking_update(observer, 1500, estimated_tokens=500)
    fire_model_phase(observer, 2000, "tool_input", tool_name="Edit")

    frame = render_frame(kit.reporter)

    assert "preparing" in frame
    assert "500" not in frame


# ---------------------------------------------------------------------------
# no "idle" anywhere in frame
# ---------------------------------------------------------------------------


def test_frame_when_any_state_does_never_contain_idle():
    kit = make_reporter()
    _fire_waiting_bash_cycle(kit, above_threshold=True)

    frame = render_frame(kit.reporter)

    assert "idle" not in frame


# ---------------------------------------------------------------------------
# dashboard title — model and effort display
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "effort", "expected_title"),
    [
        pytest.param("opus", None, "supervise ecstatic-ts · model opus", id="model-only"),
        pytest.param(None, "high", "supervise ecstatic-ts · effort high", id="effort-only"),
        pytest.param(
            "opus",
            "max",
            "supervise ecstatic-ts · model opus · effort max",
            id="model-and-effort",
        ),
        pytest.param(None, None, "supervise ecstatic-ts", id="neither"),
    ],
)
def test_panel_title_when_model_or_effort_in_force_does_show_labelled_value(
    model: str | None, effort: Effort | None, expected_title: str
) -> None:
    kit = make_reporter(label="ecstatic-ts", session_id="", branch="", model=model, effort=effort)
    fire_launch(kit.reporter.observer, 1000)

    frame = render_frame(kit.reporter)

    assert _panel_title_text(frame) == expected_title


# ---------------------------------------------------------------------------
# MCP iterate tool detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_name", "tool_use_id", "passes_completed", "passes_total", "last_pass_duration_ms"),
    [
        pytest.param("mcp__gymrat__iterate", "mcp-1", 4, 8, 120_000.0, id="mcp-tool-name"),
        pytest.param("Bash", "bash-1", 3, 6, 100_000.0, id="bash-tool-name"),
    ],
)
def test_liveness_when_iterate_tool_has_sidecar_does_show_passes_nest(
    tool_name: str,
    tool_use_id: str,
    passes_completed: int,
    passes_total: int,
    last_pass_duration_ms: float,
):
    sidecar = ProgressSnapshot(
        passes_completed=passes_completed,
        passes_total=passes_total,
        last_pass_duration_ms=last_pass_duration_ms,
    )

    def fake_read_progress(_root: str) -> ProgressSnapshot | None:
        return sidecar

    kit = make_reporter(read_progress=fake_read_progress)
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(
        kit.reporter.observer,
        tool_name,
        tool_use_id,
        2000,
        input_summary="gymrat iterate",
    )
    kit.clock.now = 2000 + 10 * 60 * 1000

    frame = render_frame(kit.reporter)

    assert f"{passes_completed}/{passes_total}" in frame
    passes_line = _line_after(frame, tool_name)
    assert "passes" in passes_line


def test_liveness_when_mcp_probe_tool_in_flight_does_show_summary_not_nest():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(
        kit.reporter.observer,
        "mcp__gymrat__probe",
        "mcp-2",
        2000,
        input_summary="gymrat probe a b",
    )
    kit.clock.now = 5000

    frame = render_frame(kit.reporter)

    assert "mcp__gymrat__probe" in frame
    assert "gymrat probe a b" in frame
    assert "↳" not in frame


def test_liveness_when_non_iterate_mcp_tool_in_flight_does_not_show_sidecar():
    sidecar = ProgressSnapshot(
        passes_completed=4,
        passes_total=8,
        last_pass_duration_ms=120_000.0,
    )

    def fake_read_progress(_root: str) -> ProgressSnapshot | None:
        return sidecar

    kit = make_reporter(read_progress=fake_read_progress)
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(
        kit.reporter.observer,
        "mcp__gymrat__probe",
        "mcp-2",
        2000,
        input_summary="gymrat probe a b",
    )
    kit.clock.now = 5000

    frame = render_frame(kit.reporter)

    assert "4/8" not in frame


# ---------------------------------------------------------------------------
# nested tool end triggers session refresh through reporter
# ---------------------------------------------------------------------------


def test_nested_tool_end_when_session_changes_does_reflect_new_state_in_frame():
    initial_state = session_state(iteration_count=0)
    updated_state = session_state(
        iteration_count=2,
        keep_count=1,
        discard_count=1,
        last_iteration=make_iteration(-3.0, "improved"),
    )

    call_count = 0

    def switching_read_session() -> ReadSessionResult:
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            return ReadSessionResult(state=initial_state, has_baseline=True)
        return ReadSessionResult(state=updated_state, has_baseline=True)

    kit = make_reporter(
        max_iterations=20,
        read_session=switching_read_session,
    )
    observer = kit.reporter.observer

    fire_launch(observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(observer, "Bash", "agent-1", 2000, input_summary="run agent")
    kit.clock.now = 3000
    fire_tool_start(
        observer,
        "Read",
        "nested-read-1",
        3000,
        parent_tool_use_id="agent-1",
        input_summary="src/config.ts",
    )
    kit.clock.now = 4000
    fire_tool_end(
        observer,
        "Read",
        "nested-read-1",
        4000,
        parent_tool_use_id="agent-1",
    )

    frame = render_frame(kit.reporter)

    assert "2/20" in frame
    assert "improved" in frame
