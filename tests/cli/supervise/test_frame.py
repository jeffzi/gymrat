"""Tests for Rich styling in the supervise dashboard frame.

Verify that the TUI renders with appropriate colors and styles instead of
plain white text.  Tests render ``reporter.frame()`` through a color-enabled
console and check for ANSI escape codes on specific content lines.
"""

from __future__ import annotations

import re
from io import StringIO
from typing import TYPE_CHECKING

import pytest
from rich.console import Console
from rich.panel import Panel

from gymrat.cli.style import CLI_THEME
from tests.cli.supervise._fixtures import (
    FRAME_WIDTH,
    IDLE_WARN_MS,
    fire_cap,
    fire_launch,
    fire_launch_and_bash_cycle,
    fire_tool_end,
    fire_tool_start,
    fire_usage_update,
    make_iteration,
    make_read_session,
    make_reporter,
    session_state,
)

if TYPE_CHECKING:
    from gymrat.cli.supervise.progress import SuperviseReporter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_ANSI_ESCAPE = re.compile(r"\x1b\[([0-9;]*)m")

# Standard ANSI SGR parameter codes
_SGR_BOLD = 1
_SGR_DIM = 2
_SGR_RED = 31
_SGR_GREEN = 32
_SGR_YELLOW = 33


def _render_content_colored(reporter: SuperviseReporter, *, width: int = FRAME_WIDTH) -> str:
    """Render the panel's inner content with standard color.

    Extracts ``panel.renderable`` so Panel border styling does not leak
    into content-styling assertions.
    """
    panel = reporter.frame()
    assert isinstance(panel, Panel)
    buf = StringIO()
    console = Console(
        file=buf,
        width=width,
        force_terminal=True,
        no_color=False,
        color_system="standard",
        legacy_windows=False,
        _environ={},
        theme=CLI_THEME,
    )
    console.print(panel.renderable)
    return buf.getvalue()


def _lines_containing(output: str, needle: str) -> list[str]:
    """Return raw (styled) lines whose plain-text content contains *needle*."""
    return [line for line in output.splitlines() if needle in _ANSI_ESCAPE.sub("", line)]


def _has_sgr(text: str, code: int) -> bool:
    """True when *text* contains an ANSI SGR sequence with parameter *code*."""
    target = str(code)
    return any(target in match.group(1).split(";") for match in _ANSI_ESCAPE.finditer(text))


def _assert_has_sgr(lines: list[str], code: int) -> None:
    """Assert *lines* is non-empty and at least one line carries SGR *code*."""
    assert lines
    assert any(_has_sgr(line, code) for line in lines)


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
    state = session_state(
        iteration_count=3,
        keep_count=2,
        discard_count=1,
        last_iteration=make_iteration(-3.2, "improved"),
    )
    kit = make_reporter(
        max_iterations=20,
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    colored = _render_content_colored(kit.reporter)
    loop_lines = _lines_containing(colored, "iter")

    _assert_has_sgr(loop_lines, _SGR_BOLD)


@pytest.mark.parametrize(
    ("outcome", "delta_pct", "expected_sgr"),
    [
        pytest.param("regressed", 3.2, _SGR_RED, id="regressed-red"),
        pytest.param("improved", -3.2, _SGR_GREEN, id="improved-green"),
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

    _assert_has_sgr(outcome_lines, expected_sgr)


# ---------------------------------------------------------------------------
# best row styling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta_pct", "expected_sgr"),
    [
        pytest.param(-6.8, _SGR_GREEN, id="negative-delta-green"),
        pytest.param(3.5, _SGR_RED, id="positive-delta-red"),
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

    _assert_has_sgr(best_lines, expected_sgr)


# ---------------------------------------------------------------------------
# liveness styling
# ---------------------------------------------------------------------------


def test_liveness_starting_when_rendered_with_color_does_emit_dim_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    colored = _render_content_colored(kit.reporter)
    starting_lines = _lines_containing(colored, "starting")

    _assert_has_sgr(starting_lines, _SGR_DIM)


def test_liveness_inflight_when_rendered_with_color_does_emit_yellow_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000, input_summary="gymrat iterate")
    kit.clock.now = 7000

    colored = _render_content_colored(kit.reporter)
    bash_lines = _lines_containing(colored, "Bash")

    _assert_has_sgr(bash_lines, _SGR_YELLOW)


def test_liveness_idle_when_rendered_with_color_does_emit_yellow_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000)
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Bash", "bash-1", 3000)
    kit.clock.now = 3000 + IDLE_WARN_MS + 1

    colored = _render_content_colored(kit.reporter)
    idle_lines = _lines_containing(colored, "idle")

    _assert_has_sgr(idle_lines, _SGR_YELLOW)


def test_liveness_capped_when_rendered_with_color_does_emit_yellow_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    fire_cap(kit.reporter.observer, "wall-clock")

    colored = _render_content_colored(kit.reporter)
    cap_lines = _lines_containing(colored, "interrupting")

    _assert_has_sgr(cap_lines, _SGR_YELLOW)


# ---------------------------------------------------------------------------
# finished tool lines styling
# ---------------------------------------------------------------------------


def test_finished_tool_line_when_rendered_with_color_does_emit_dim_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 3000)

    colored = _render_content_colored(kit.reporter)
    edit_lines = _lines_containing(colored, "Edit")

    _assert_has_sgr(edit_lines, _SGR_DIM)
