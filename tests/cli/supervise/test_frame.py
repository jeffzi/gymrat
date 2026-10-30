"""Tests for the Rich renderables of the supervise dashboard frame.

The dashboard-styling tests verify that the TUI renders with appropriate colors
and styles instead of plain white text: they render ``reporter.frame()`` through
a color-enabled console and check the style of the one segment that holds the
content under test. The ETA text the frame shows is pinned through
``format_eta`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
from rich.box import ROUNDED
from rich.style import Style

from gymrat.cli.style import STYLE_DONE, STYLE_META, STYLE_REGRESSED
from gymrat.cli.supervise.frame import format_eta
from gymrat.session.store import BestIteration
from tests.cli.supervise._fixtures import (
    color_console,
    fire_launch_and_edit_cycle,
    make_reporter,
    reporter_showing_session,
)
from tests.session.records._fixtures import (
    make_iteration,
    session_state,
)

if TYPE_CHECKING:
    from rich.segment import Segment

    from gymrat.cli.supervise.progress import SuperviseReporter
    from gymrat.session.schema import Outcome

pytestmark = pytest.mark.usefixtures("mock_live_cls")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# The panel's border rows open on these corners; every other row is content.
_TOP_LEFT = ROUNDED.top_left
_BOTTOM_LEFT = ROUNDED.bottom_left


# A meta row's dim base combines with the content style the project picks.
_META_DONE = str(Style.parse(STYLE_META) + Style.parse(STYLE_DONE))
_META_REGRESSED = str(Style.parse(STYLE_META) + Style.parse(STYLE_REGRESSED))
_FAILED_TOOL = str(Style.parse(f"dim {STYLE_REGRESSED}"))


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


# ---------------------------------------------------------------------------
# loop row styling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "delta_pct", "expected_style"),
    [
        pytest.param("regressed", 3.2, _META_REGRESSED, id="regressed-red"),
        pytest.param("improved", -3.2, _META_DONE, id="improved-green"),
    ],
)
def test_loop_outcome_when_rendered_with_color_does_style_regressed_red_and_improved_green(
    outcome: Outcome, delta_pct: float, expected_style: str
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
        pytest.param(-6.8, "lower", _META_DONE, id="lower-better-negative-delta-green"),
        pytest.param(3.5, "lower", _META_REGRESSED, id="lower-better-positive-delta-red"),
    ],
)
def test_best_delta_when_rendered_with_color_does_style_an_improvement_green_and_a_regression_red(
    delta_pct: float, direction: Literal["lower", "higher"], expected_style: str
) -> None:
    kit = reporter_showing_session(
        session_state(iteration_count=1, keep_count=1),
        best=BestIteration(delta_pct=delta_pct, seq=1, label="primary", direction=direction),
    )

    style = _segment_style(kit.reporter, "%")

    assert style == expected_style


# ---------------------------------------------------------------------------
# finished tool lines styling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected_style"),
    [
        pytest.param("ok", STYLE_META, id="ok-dim"),
        pytest.param("error", _FAILED_TOOL, id="failed-dim-red"),
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
# format_eta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        (0, "~1s left"),
        (48200, "~48s left"),
        (59999, "~1m left"),
        (130000, "~2m 10s left"),
        (3599999, "~1h left"),
        (3900000, "~1h 05m left"),
    ],
)
def test_format_eta_when_given_milliseconds_does_round_to_the_two_largest_tiers_clamped_to_one_second(
    ms: float, expected: str
) -> None:
    assert format_eta(ms) == expected
