"""Live-wiring, render-refresh, warning, and render-failure tests.

Tests for the ``Live`` construction contract (rich's refresh timer at one frame
per second rendering through ``get_renderable``, ``transient=True``, rich's
stderr redirect left on), the single ``refresh()`` an event that changes state
triggers, the skipped repaint for events that leave state unchanged, the repaint
a session refresh triggers only when the re-read succeeds, and ``_stop_live``
suppression scope (``OSError`` and a closed-stream ``ValueError`` only).

The ``terminal`` fixture makes a sealed terminal console the dashboard's
console and the process's stderr, with one line kept above the dashboard. The
warning tests patch out the ``Live`` class so no frame is painted, and read
what the dashboard printed. What a termination signal does to the dashboard is
tested alongside every other CLI progress renderer, in
``tests/cli/test_progress.py``.
"""

from __future__ import annotations

import sys
from io import StringIO
from typing import TYPE_CHECKING, override
from unittest.mock import DEFAULT, MagicMock, Mock, patch

import pytest

from gymrat.supervisor.events import TextDeltaEvent
from gymrat.supervisor.exit_sequence import ExitPhase
from tests._logging import unhandled_logging
from tests._rich import (
    KEPT_LINE,
    Clock,
    frame_text,
    screen_lines,
    sealed_console,
    stop_tracked,
)
from tests.cli.supervise._fixtures import (
    EMPTY_READ,
    FRAME_WIDTH,
    KEPT_READ,
    LIVE_CLASS_PATH,
    ReporterKit,
    _throwing_read,
    fire_launch_and_iterate_start,
    launch_event,
    make_reporter,
    render_frame,
    tool_start_event,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rich.console import Console

    from gymrat.cli.supervise.types import ReadSessionResult
    from gymrat.session.progress_file import ProgressSnapshot
    from gymrat.supervisor.events import SessionEvent

# The terminal the dashboard paints on: wide enough for the golden frame width
# and tall enough that the frame is never cropped.
_SCREEN_WIDTH = FRAME_WIDTH
_SCREEN_HEIGHT = 40


@pytest.fixture
def mock_live_cls() -> Iterator[MagicMock]:
    """The ``ErasableLive`` class the live-mode reporter builds, patched with an autospec."""
    with patch(LIVE_CLASS_PATH, autospec=True) as live_cls:
        yield live_cls


def _launched_live(read_session: Callable[[], ReadSessionResult] | None = None) -> ReporterKit:
    """A live-mode reporter reading the session through *read_session*, past its launch."""
    kit = make_reporter(mode="live", read_session=read_session)
    kit.reporter.observer(launch_event(1000))
    return kit


def _repaints_from_now(live_cls: MagicMock) -> Callable[[], int]:
    """A counter of the dashboard repaints made from this call on."""
    live = live_cls.return_value
    painted = live.refresh.call_count
    return lambda: live.refresh.call_count - painted


# ---------------------------------------------------------------------------
# Live color override
# ---------------------------------------------------------------------------


def test_create_reporter_when_color_false_does_build_colorless_console(mock_live_cls: MagicMock):
    make_reporter(mode="live", color=False)

    console = mock_live_cls.call_args.kwargs.get("console")
    assert console is not None
    assert console.color_system is None


# ---------------------------------------------------------------------------
# Live construction — refresh timer, transient, renderable, stderr redirect
# ---------------------------------------------------------------------------


def test_create_reporter_when_live_mode_does_configure_the_live_display(mock_live_cls: MagicMock):
    kit = make_reporter(mode="live")

    call_kwargs = mock_live_cls.call_args.kwargs
    assert call_kwargs.get("auto_refresh") is True
    assert call_kwargs.get("refresh_per_second") == 1
    assert frame_text(call_kwargs["get_renderable"](), width=FRAME_WIDTH) == render_frame(
        kit.reporter
    )
    assert call_kwargs.get("transient") is True
    # rich's stderr redirect is what lands a stray stderr write above the frame
    assert call_kwargs.get("redirect_stderr", True) is True


def test_create_reporter_when_plain_mode_does_not_create_live(mock_live_cls: MagicMock):
    make_reporter(mode="plain", plain_write=lambda _: None)

    mock_live_cls.assert_not_called()


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
    kit = _launched_live()
    repaints = _repaints_from_now(mock_live_cls)

    kit.reporter.observer(event)

    assert repaints() == expected_repaints


def test_exit_phase_when_live_mode_and_phase_changes_does_repaint_once(mock_live_cls: MagicMock):
    kit = _launched_live()
    kit.reporter.exit_phase(ExitPhase(kind="waiting-lock", pid=4242))
    repaints = _repaints_from_now(mock_live_cls)

    kit.reporter.exit_phase(ExitPhase(kind="settling", pid=None))

    assert repaints() == 1


@pytest.mark.parametrize(
    ("reread", "expected_repaints"),
    [
        pytest.param(KEPT_READ, 1, id="reread"),
        pytest.param(RuntimeError("session file unreadable"), 0, id="reread-fails"),
    ],
)
def test_refresh_session_when_live_does_repaint_only_after_a_successful_reread(
    mock_live_cls: MagicMock, reread: ReadSessionResult | Exception, expected_repaints: int
):
    kit = _launched_live(read_session=Mock(side_effect=[EMPTY_READ, reread]))
    repaints = _repaints_from_now(mock_live_cls)

    kit.reporter.refresh_session()

    assert repaints() == expected_repaints


# ---------------------------------------------------------------------------
# warn — messages printed verbatim
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mock_live_cls")
def test_warn_when_live_message_contains_brackets_does_print_it_verbatim(terminal: StringIO):
    kit = make_reporter(mode="live")

    kit.reporter.warn("missing [banana] key")

    assert terminal.getvalue() == f"{KEPT_LINE}\nmissing [banana] key\n"


# ---------------------------------------------------------------------------
# refresh_session — a failing session read reported without a traceback
# ---------------------------------------------------------------------------

_READ_FAILED = "session read failed: no session file"


@pytest.mark.usefixtures("mock_live_cls")
def test_refresh_session_when_live_session_read_fails_does_report_it_without_a_traceback(
    terminal: StringIO,
):
    kit = make_reporter(mode="live", read_session=_throwing_read)

    with unhandled_logging():
        kit.reporter.refresh_session()

    assert _screen(terminal.getvalue()) == [KEPT_LINE, _READ_FAILED]


# ---------------------------------------------------------------------------
# _stop_live — suppresses OSError and a closed-stream ValueError only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(OSError("stderr closed"), id="os-error"),
        pytest.param(ValueError("I/O operation on closed file"), id="closed-stream-value-error"),
    ],
)
def test_stop_when_live_stop_raises_a_closed_stream_error_does_suppress(
    mock_live_cls: MagicMock, error: Exception
):
    mock_live_cls.return_value.stop.side_effect = error
    kit = make_reporter(mode="live")

    kit.reporter.stop()


def test_stop_when_live_stop_raises_unrelated_value_error_does_propagate(mock_live_cls: MagicMock):
    mock_live_cls.return_value.stop.side_effect = [ValueError("unexpected"), None]
    kit = make_reporter(mode="live")

    with pytest.raises(ValueError, match="unexpected"):
        kit.reporter.stop()


# ---------------------------------------------------------------------------
# Terminal — the stderr the dashboard paints on
# ---------------------------------------------------------------------------


def _paint_dashboards_on(console: Console, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hand every dashboard built from here on *console* as its stderr console."""

    def dashboard_console(**_kwargs: object) -> Console:
        return console

    monkeypatch.setattr("gymrat.cli.supervise.progress.stderr_console", dashboard_console)


def _mount_terminal(term: StringIO, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make *term* the terminal a dashboard paints on and the process's stderr.

    One line is already printed on it, above where the dashboard will go.

    Args:
        term: The buffer standing in for the terminal.
        monkeypatch: Patches the dashboard console factory and ``sys.stderr``.
    """
    console = sealed_console(width=_SCREEN_WIDTH, height=_SCREEN_HEIGHT, color_system=None)
    console.file = term
    console.print(KEPT_LINE)
    _paint_dashboards_on(console, monkeypatch)
    monkeypatch.setattr(sys, "stderr", term)


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[StringIO]:
    """The stderr terminal the live dashboard paints on, with one line kept above it."""
    term = StringIO()
    _mount_terminal(term, monkeypatch)
    yield term
    # Stop the dashboards while stderr is still this terminal: a Live stopped
    # after monkeypatch's undo would re-point sys.stderr at this dead buffer.
    stop_tracked()


def _screen(raw: str) -> list[str]:
    return screen_lines(raw, width=_SCREEN_WIDTH, height=_SCREEN_HEIGHT)


# ---------------------------------------------------------------------------
# Render failure — the dashboard survives a frame that fails to render
# ---------------------------------------------------------------------------

# Message of the error the flaky sidecar raises, which the warning names.
_RENDER_FAILURE = "sidecar unreadable"

# The one warning line a failed render leaves on the terminal.
_RENDER_FAILURE_WARNING = (
    f"warning: dashboard frame failed to render: RuntimeError: {_RENDER_FAILURE}"
)


class _FlakySidecar:
    """A sidecar reader a test arms to fail its next reads.

    ``fail_next(count)`` makes the next *count* reads raise; ``fail_next(None)``
    makes every read from then on raise. The dashboard reads the sidecar on
    every frame while an iterate call is in flight, so each failed read is a
    failed render.
    """

    def __init__(self) -> None:
        self._failures_left: int | None = 0

    def fail_next(self, count: int | None) -> None:
        self._failures_left = count

    def __call__(self, _root: str) -> ProgressSnapshot | None:
        if self._failures_left is None or self._failures_left > 0:
            if self._failures_left is not None:
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

    assert (failed, terminal.getvalue()) == (
        [last_good, last_good],
        f"{KEPT_LINE}\n{_RENDER_FAILURE_WARNING}\n",
    )


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


def test_stop_when_final_frame_fails_to_render_does_return_normally(terminal: StringIO):
    sidecar = _FlakySidecar()
    kit = _dashboard_reading(sidecar)
    sidecar.fail_next(None)

    kit.reporter.stop()


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

    assert (plain_lines, terminal.getvalue()) == ([], f"{KEPT_LINE}\n{_RENDER_FAILURE_WARNING}\n")
