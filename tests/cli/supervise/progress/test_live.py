"""Live-wiring, render-refresh, and signal-erase tests.

Tests for the ``Live`` construction contract (rich's refresh timer at one frame
per second rendering through ``get_renderable``, ``transient=True``, mounted via
``start()``), the single ``refresh()`` an event that changes state triggers, the
skipped repaint for events that leave state unchanged, and ``_stop_live``
suppression scope (``OSError`` and a closed-stream ``ValueError`` only).

The signal tests paint the dashboard on a sealed terminal console that is also
the process's stderr from before the reporter is built, as a real terminal is,
so the dashboard's stderr redirect is in effect when a signal lands. They run
the installed termination handler with the process exit stubbed out, then
replay what reached the terminal through a ``pyte`` screen: the screen a user
is left with after ``os._exit``. The dashboard's erase is handed to the
handler, which writes it together with any warnings a later cleanup raises, so
the tests read the screen once the handler has run.
"""

from __future__ import annotations

import sys
from io import StringIO
from typing import TYPE_CHECKING, Literal, override
from unittest.mock import patch

import pytest

from gymrat.signals import install_termination_cleanup
from gymrat.supervisor.events import TextDeltaEvent
from gymrat.supervisor.exit_sequence import ExitPhase
from tests._logging import unhandled_logging
from tests._rich import (
    KEPT_LINE,
    TERMINATION_SIGNAL,
    WARNING_LINE,
    Clock,
    frame_text,
    screen_lines,
    sealed_console,
    stop_tracked,
)
from tests.cli.supervise._fixtures import (
    FRAME_WIDTH,
    LIVE_CLASS_PATH,
    ReporterKit,
    _throwing_read,
    fire_launch_and_bash_cycle,
    launch_event,
    make_reporter,
    render_frame,
    tool_start_event,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rich.console import Console

    from gymrat.session.progress_file import ProgressSnapshot
    from gymrat.supervisor.events import SessionEvent

# The terminal the signal tests paint the dashboard on: wide enough for the
# golden frame width and tall enough that the frame is never cropped.
_SCREEN_WIDTH = FRAME_WIDTH
_SCREEN_HEIGHT = 40


# ---------------------------------------------------------------------------
# Live color override
# ---------------------------------------------------------------------------


def test_create_reporter_when_color_false_does_build_colorless_console():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        make_reporter(mode="live", color=False)

        call_kwargs = mock_live_cls.call_args.kwargs
        console = call_kwargs.get("console")
        assert console is not None
        assert console.color_system is None


# ---------------------------------------------------------------------------
# Live construction — refresh timer, transient, mounted, initial paint
# ---------------------------------------------------------------------------


def test_create_reporter_when_live_mode_does_mount_a_configured_live():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        kit = make_reporter(mode="live")

        call_kwargs = mock_live_cls.call_args.kwargs
        assert call_kwargs.get("auto_refresh") is True
        assert call_kwargs.get("refresh_per_second") == 1
        assert frame_text(call_kwargs["get_renderable"](), width=FRAME_WIDTH) == render_frame(
            kit.reporter
        )
        assert call_kwargs.get("transient") is True
        mock_live.start.assert_called_once()
        mock_live.refresh.assert_called_once()


def test_create_reporter_when_plain_mode_does_not_create_live():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
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
    event: SessionEvent, expected_repaints: int
):
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        live = mock_live_cls.return_value
        kit = make_reporter(mode="live")
        kit.reporter.observer(launch_event(1000))
        painted = live.refresh.call_count

        kit.reporter.observer(event)

        assert live.refresh.call_count - painted == expected_repaints


def test_exit_phase_when_live_mode_and_phase_changes_does_repaint_once():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        live = mock_live_cls.return_value
        kit = make_reporter(mode="live")
        kit.reporter.observer(launch_event(1000))
        kit.reporter.exit_phase(ExitPhase(kind="waiting-lock", pid=4242))
        painted = live.refresh.call_count

        kit.reporter.exit_phase(ExitPhase(kind="settling", pid=None))

        assert live.refresh.call_count - painted == 1


# ---------------------------------------------------------------------------
# warn — messages printed verbatim
# ---------------------------------------------------------------------------


def test_warn_when_live_message_contains_brackets_does_print_it_verbatim(terminal: StringIO):
    kit = make_reporter(mode="live")

    kit.reporter.warn("missing [banana] key")
    kit.reporter.stop()

    assert _screen(terminal.getvalue()) == [KEPT_LINE, "missing [banana] key"]


# ---------------------------------------------------------------------------
# refresh_session — a failing session read reported once
# ---------------------------------------------------------------------------

_READ_FAILED = "session read failed: no session file"


@pytest.mark.parametrize(
    ("mode", "screen", "plain_lines"),
    [
        pytest.param("live", [KEPT_LINE, _READ_FAILED], [], id="live-prints-above-the-frame"),
        pytest.param("plain", [KEPT_LINE], [_READ_FAILED], id="plain-writes-a-milestone-line"),
    ],
)
def test_refresh_session_when_session_read_keeps_failing_does_report_it_once_without_a_traceback(
    mode: Literal["live", "plain"],
    screen: list[str],
    plain_lines: list[str],
    terminal: StringIO,
):
    writes: list[str] = []
    kit = make_reporter(mode=mode, read_session=_throwing_read, plain_write=writes.append)

    with unhandled_logging():
        fire_launch_and_bash_cycle(kit.reporter.observer)
        kit.reporter.refresh_session()
    kit.reporter.stop()

    assert (_screen(terminal.getvalue()), [line for line in writes if "failed" in line]) == (
        screen,
        plain_lines,
    )


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
def test_stop_when_live_stop_raises_a_closed_stream_error_does_suppress(error: Exception):
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        mock_live.stop.side_effect = error
        kit = make_reporter(mode="live")

        kit.reporter.stop()


def test_stop_when_live_stop_raises_unrelated_value_error_does_propagate():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        mock_live.stop.side_effect = [ValueError("unexpected"), None]
        kit = make_reporter(mode="live")

        with pytest.raises(ValueError, match="unexpected"):
            kit.reporter.stop()


# ---------------------------------------------------------------------------
# Termination signal — erase the dashboard without waiting on its lock
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
    console = sealed_console(width=_SCREEN_WIDTH, height=_SCREEN_HEIGHT)
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


def test_stderr_write_when_live_dashboard_up_does_land_above_the_frame(terminal: StringIO):
    kit = make_reporter(mode="live")
    kit.reporter.observer(launch_event(1000))

    sys.stderr.write(f"{WARNING_LINE}\n")
    sys.stderr.flush()

    frame_rows = _screen(render_frame(kit.reporter, width=_SCREEN_WIDTH))
    assert _screen(terminal.getvalue()) == [KEPT_LINE, WARNING_LINE, *frame_rows]


def test_signal_when_dashboard_already_stopped_does_leave_the_screen_untouched(
    terminal: StringIO, raise_signal: Callable[[int], int]
):
    kit = make_reporter(mode="live")
    kit.reporter.observer(launch_event(1000))
    kit.reporter.stop()
    install_termination_cleanup(lambda: None)
    before = terminal.getvalue()

    raise_signal(TERMINATION_SIGNAL)

    assert terminal.getvalue() == before


def test_signal_when_plain_mode_does_write_nothing(
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
):
    make_reporter(mode="plain", plain_write=lambda _: None)
    install_termination_cleanup(lambda: None)
    buffer = StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)

    raise_signal(TERMINATION_SIGNAL)

    assert buffer.getvalue() == ""


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
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(tool_start_event("Bash", "bash-1", 2000, input_summary="gymrat iterate"))
    return kit


@pytest.mark.usefixtures("terminal")
def test_live_frame_when_a_frame_fails_to_render_does_show_the_last_good_frame():
    sidecar = _FlakySidecar()
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        kit = _dashboard_reading(sidecar)
    get_renderable = mock_live_cls.call_args.kwargs["get_renderable"]
    last_good = frame_text(get_renderable(), width=FRAME_WIDTH)
    sidecar.fail_next(1)
    kit.clock.now = 9000

    failed = frame_text(get_renderable(), width=FRAME_WIDTH)

    assert failed == last_good


@pytest.mark.usefixtures("terminal")
def test_live_frame_when_a_render_fails_once_does_render_the_next_frame_normally():
    sidecar = _FlakySidecar()
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        kit = _dashboard_reading(sidecar)
    get_renderable = mock_live_cls.call_args.kwargs["get_renderable"]
    sidecar.fail_next(1)
    kit.clock.now = 9000
    get_renderable()

    recovered = frame_text(get_renderable(), width=FRAME_WIDTH)

    assert recovered == render_frame(kit.reporter)


def test_live_frame_when_frames_fail_to_render_does_warn_once_naming_the_error(
    terminal: StringIO,
):
    sidecar = _FlakySidecar()
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        _dashboard_reading(sidecar)
    get_renderable = mock_live_cls.call_args.kwargs["get_renderable"]
    sidecar.fail_next(2)

    get_renderable()
    get_renderable()

    assert _screen(terminal.getvalue()) == [KEPT_LINE, _RENDER_FAILURE_WARNING]


def test_stop_when_plain_mode_frame_would_fail_does_not_warn():
    sidecar = _FlakySidecar()
    sidecar.fail_next(None)
    plain_lines: list[str] = []
    kit = make_reporter(mode="plain", read_progress=sidecar, plain_write=plain_lines.append)
    kit.reporter.observer(launch_event(1000))
    kit.reporter.observer(tool_start_event("Bash", "bash-1", 2000, input_summary="gymrat iterate"))

    kit.reporter.stop()

    assert plain_lines == ["caps 480m"]


def test_stop_when_final_frame_fails_to_render_does_return_normally(terminal: StringIO):
    sidecar = _FlakySidecar()
    kit = _dashboard_reading(sidecar)
    sidecar.fail_next(None)

    kit.reporter.stop()


class _SetupFailingClock(Clock[int]):
    """A clock that raises while ``failing`` is set, as a frame built during setup would."""

    def __init__(self) -> None:
        super().__init__(1000)
        self.failing = True

    @override
    def __call__(self) -> int:
        if self.failing:
            raise RuntimeError(_RENDER_FAILURE)
        return super().__call__()


def test_create_reporter_when_setup_frame_fails_to_render_does_warn_through_the_dashboard(
    terminal: StringIO,
):
    clock = _SetupFailingClock()
    plain_lines: list[str] = []
    kit = make_reporter(mode="live", clock=clock, plain_write=plain_lines.append)
    clock.failing = False

    kit.reporter.stop()

    assert (plain_lines, _screen(terminal.getvalue())) == (
        [],
        [KEPT_LINE, _RENDER_FAILURE_WARNING],
    )
