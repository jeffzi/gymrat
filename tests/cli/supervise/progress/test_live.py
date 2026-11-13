"""Live-wiring, repaint, session-refresh and render-failure tests.

Tests for the refresh rate the live display is built with, the single repaint an
event or exit phase that changes state triggers, the session refresh that
re-reads the session and repaints only when the re-read succeeds, and a frame
that fails to render: the last good frame holds with one warning, a fresh frame
returns once the failures stop, and a setup frame that fails warns through the
dashboard.

The zero-width fallback, what a termination signal does to a display, and the
``stop()`` guards every renderer shares are tested in
``tests/cli/test_live_display.py`` and ``tests/cli/test_live_display_mixin.py``.
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING, override
from unittest.mock import DEFAULT, MagicMock, Mock, patch

import pytest

from gymrat.supervisor.events import TextDeltaEvent
from gymrat.supervisor.exit_sequence import ExitPhase
from tests._rich import (
    Clock,
    frame_text,
    sealed_console,
)
from tests.cli.supervise._fixtures import (
    EMPTY_READ,
    FRAME_WIDTH,
    KEPT_READ,
    ReporterKit,
    fire_launch_and_iterate_start,
    launched,
    make_reporter,
    render_frame,
)
from tests.supervisor._fixtures import (
    tool_start_event,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from gymrat.session.progress_file import ProgressSnapshot
    from gymrat.session.store import ReadSessionResult
    from gymrat.supervisor.events import SessionEvent

# The terminal the dashboard paints on: wide enough for the golden frame width.
_SCREEN_WIDTH = FRAME_WIDTH


def _repaints_from_now(live_cls: MagicMock) -> Callable[[], int]:
    """A counter of the dashboard repaints made from this call on."""
    live = live_cls.return_value
    painted = live.refresh.call_count
    return lambda: live.refresh.call_count - painted


# ---------------------------------------------------------------------------
# Live construction — refresh rate and renderable
# ---------------------------------------------------------------------------


def test_create_reporter_when_live_mode_does_derive_the_refresh_rate_from_refresh_ms(
    mock_live_cls: MagicMock,
):
    make_reporter(mode="live", refresh_ms=500)

    assert mock_live_cls.call_args.kwargs.get("refresh_per_second") == 2


# ---------------------------------------------------------------------------
# render calls — one refresh per state-changing event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event", "expected_repaints"),
    [
        pytest.param(
            tool_start_event("Bash", "bash-1", 2000, input_summary="npm test"),
            1,
            id="changes-state",
        ),
        pytest.param(
            TextDeltaEvent(at=2_000_000_000, chunk="hello"), 0, id="leaves-state-unchanged"
        ),
    ],
)
def test_observer_when_live_mode_does_repaint_once_per_state_change(
    mock_live_cls: MagicMock, event: SessionEvent, expected_repaints: int
):
    kit = launched(make_reporter())
    repaints = _repaints_from_now(mock_live_cls)

    kit.reporter.observer(event)

    assert repaints() == expected_repaints


def test_exit_phase_when_live_mode_and_phase_changes_does_repaint_once(mock_live_cls: MagicMock):
    kit = launched(make_reporter())
    kit.reporter.exit_phase(ExitPhase(kind="waiting-lock", pid=4242))
    repaints = _repaints_from_now(mock_live_cls)

    kit.reporter.exit_phase(ExitPhase(kind="settling", pid=None))

    assert repaints() == 1


@pytest.mark.parametrize(
    ("reread", "expected_session", "expected_repaints"),
    [
        pytest.param(KEPT_READ, KEPT_READ, 1, id="reread"),
        pytest.param(
            RuntimeError("session file unreadable"), EMPTY_READ, 0, id="reread-fails-keeps-previous"
        ),
    ],
)
def test_refresh_session_when_live_does_show_only_a_successful_reread(
    mock_live_cls: MagicMock,
    reread: ReadSessionResult | Exception,
    expected_session: ReadSessionResult,
    expected_repaints: int,
):
    kit = launched(make_reporter(read_session=Mock(side_effect=[EMPTY_READ, reread])))
    repaints = _repaints_from_now(mock_live_cls)

    kit.reporter.refresh_session()

    assert (kit.reporter.session_result(), repaints()) == (expected_session, expected_repaints)


# ---------------------------------------------------------------------------
# Terminal — the stderr the dashboard paints on
# ---------------------------------------------------------------------------


@pytest.fixture
def terminal() -> Iterator[StringIO]:
    """The terminal the live dashboard's console writes to."""
    term = StringIO()
    console = sealed_console(width=_SCREEN_WIDTH, color_system=None)
    console.file = term
    with patch("gymrat.cli.supervise.progress.stderr_console", autospec=True, return_value=console):
        yield term


# ---------------------------------------------------------------------------
# Render failure — the dashboard survives a frame that fails to render
# ---------------------------------------------------------------------------

# Message of the error the flaky sidecar raises, which the warning names.
_RENDER_FAILURE = "sidecar unreadable"


def _assert_one_warning_naming_the_render_error(terminal: StringIO) -> None:
    """Assert the terminal holds exactly one warning, naming the render error."""
    lines = terminal.getvalue().splitlines()
    assert len(lines) == 1
    assert lines[0].endswith(f"RuntimeError: {_RENDER_FAILURE}")


class _FlakySidecar:
    """A sidecar reader a test arms to fail its next reads.

    ``fail_next(count)`` makes the next *count* reads raise. The dashboard reads
    the sidecar on every frame while an iterate call is in flight, so each failed
    read is a failed render.
    """

    def __init__(self) -> None:
        self._failures_left = 0

    def fail_next(self, count: int) -> None:
        self._failures_left = count

    def __call__(self, _root: str) -> ProgressSnapshot | None:
        if self._failures_left > 0:
            self._failures_left -= 1
            raise RuntimeError(_RENDER_FAILURE)
        return None


def _dashboard_reading(sidecar: _FlakySidecar) -> ReporterKit:
    """Mount a live dashboard with an iterate call reading *sidecar*."""
    kit = make_reporter(mode="live", read_progress=sidecar)
    fire_launch_and_iterate_start(kit)
    return kit


def test_live_frame_when_frames_keep_failing_does_hold_the_last_good_frame_with_one_warning(
    terminal: StringIO, mock_live_cls: MagicMock
):
    sidecar = _FlakySidecar()
    kit = _dashboard_reading(sidecar)
    get_renderable = mock_live_cls.call_args.kwargs["get_renderable"]
    last_good = frame_text(get_renderable(), width=FRAME_WIDTH)
    sidecar.fail_next(2)
    kit.clock.now = 9000

    failed = [frame_text(get_renderable(), width=FRAME_WIDTH) for _ in range(2)]

    assert failed == [last_good, last_good]
    _assert_one_warning_naming_the_render_error(terminal)


@pytest.mark.usefixtures("terminal")
def test_live_frame_when_failures_stop_does_render_a_fresh_frame(mock_live_cls: MagicMock):
    sidecar = _FlakySidecar()
    kit = _dashboard_reading(sidecar)
    get_renderable = mock_live_cls.call_args.kwargs["get_renderable"]
    get_renderable()
    sidecar.fail_next(1)
    kit.clock.now = 9000
    get_renderable()

    recovered = frame_text(get_renderable(), width=FRAME_WIDTH)

    assert recovered == render_frame(kit.reporter)


class _FailingClock(Clock[int]):
    """A clock that raises on every read, so any frame built from it fails."""

    def __init__(self) -> None:
        super().__init__(1000)

    @override
    def __call__(self) -> int:
        raise RuntimeError(_RENDER_FAILURE)


def _build_a_frame_on_construction(
    *_args: object, get_renderable: Callable[[], object], **_kwargs: object
) -> object:
    # rich builds a first frame while a Live is being constructed.
    get_renderable()
    return DEFAULT


def test_create_reporter_when_setup_frame_fails_to_render_does_warn_through_the_dashboard(
    terminal: StringIO, mock_live_cls: MagicMock
):
    plain_lines: list[str] = []
    mock_live_cls.side_effect = _build_a_frame_on_construction

    make_reporter(mode="live", clock=_FailingClock(), plain_write=plain_lines.append)

    assert plain_lines == []
    _assert_one_warning_naming_the_render_error(terminal)
