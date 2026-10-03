"""Live-wiring, render-refresh, and signal-erase tests.

Tests for the ``Live`` construction contract (rich's refresh timer at one frame
per second rendering through ``get_renderable``, ``transient=True``, mounted via
``start()``), the single ``refresh()`` an event that changes state triggers, the
skipped repaint for events that leave state unchanged, and ``_stop_live``
suppression scope.

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
import threading
import time
from io import StringIO
from typing import TYPE_CHECKING, override
from unittest.mock import patch

import pytest

from gymrat.signals import install_termination_cleanup
from gymrat.supervisor.events import TextDeltaEvent
from gymrat.supervisor.exit_sequence import ExitPhase
from tests._rich import (
    HIDE_CURSOR,
    KEPT_LINE,
    TERMINATION_SIGNAL,
    WARNING_LINE,
    CleanupRegistry,
    InterruptedTerminal,
    ProcessExit,
    cursor_hidden,
    frame_text,
    screen_lines,
    sealed_console,
    track_mounted_cleanups,
)
from tests.cli.supervise._fixtures import (
    FRAME_WIDTH,
    LIVE_CLASS_PATH,
    Clock,
    ReporterKit,
    launch_event,
    make_reporter,
    render_frame,
    tool_start_event,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console

    from gymrat.session.progress_file import ProgressSnapshot

# Failure message the mount test raises and then matches.
_MOUNT_FAILURE = "mount failed"

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


def test_create_reporter_when_live_mode_does_configure_and_mount_live():
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


@pytest.mark.parametrize("failing_step", ["start", "refresh"])
def test_create_reporter_when_mounting_live_raises_does_leave_nothing_running(
    dashboard_cleanups: CleanupRegistry, failing_step: str
):
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        getattr(mock_live, failing_step).side_effect = RuntimeError(_MOUNT_FAILURE)

        with pytest.raises(RuntimeError, match=_MOUNT_FAILURE):
            make_reporter(mode="live")

    mock_live.stop.assert_called_once()
    assert dashboard_cleanups.live() == []


# ---------------------------------------------------------------------------
# render calls — one refresh per state-changing event
# ---------------------------------------------------------------------------


def test_render_when_event_changes_state_in_live_mode_does_refresh_live_once():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        kit = make_reporter(mode="live")
        mock_live.reset_mock()

        kit.reporter.observer(launch_event(1000))

        mock_live.refresh.assert_called_once()


def test_render_when_event_leaves_state_unchanged_does_not_repaint_live():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        live = mock_live_cls.return_value
        kit = make_reporter(mode="live")
        kit.reporter.observer(launch_event(1000))
        painted = live.refresh.call_count

        kit.reporter.observer(TextDeltaEvent(at=2_000_000_000, chunk="hello"))

        assert live.refresh.call_count == painted


def test_exit_phase_when_live_mode_does_repaint_live():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        live = mock_live_cls.return_value
        kit = make_reporter(mode="live")
        kit.reporter.observer(launch_event(1000))
        painted = live.refresh.call_count

        kit.reporter.exit_phase(ExitPhase(kind="settling", pid=None))

        assert live.refresh.call_count > painted


# ---------------------------------------------------------------------------
# warn — live mode prints messages verbatim
# ---------------------------------------------------------------------------


def test_warn_when_live_message_contains_brackets_does_print_it_verbatim(terminal: StringIO):
    kit = make_reporter(mode="live")

    kit.reporter.warn("missing [banana] key")

    kit.reporter.stop()
    assert _screen(terminal.getvalue()) == [KEPT_LINE, "missing [banana] key"]


def test_render_when_plain_mode_does_not_create_live():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        make_reporter(mode="plain", plain_write=lambda _: None)

        mock_live_cls.assert_not_called()


# ---------------------------------------------------------------------------
# _stop_live — suppresses only OSError
# ---------------------------------------------------------------------------


def test_stop_when_live_stop_raises_os_error_does_suppress():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        mock_live.stop.side_effect = OSError("stderr closed")
        kit = make_reporter(mode="live")

        kit.reporter.stop()


def test_stop_when_live_stop_raises_non_os_error_does_propagate():
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
    """
    console = sealed_console(width=_SCREEN_WIDTH, height=_SCREEN_HEIGHT)
    console.file = term
    console.print(KEPT_LINE)
    _paint_dashboards_on(console, monkeypatch)
    monkeypatch.setattr(sys, "stderr", term)


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    """The stderr terminal the live dashboard paints on, with one line kept above it."""
    term = StringIO()
    _mount_terminal(term, monkeypatch)
    return term


@pytest.fixture
def dashboard_cleanups(monkeypatch: pytest.MonkeyPatch) -> CleanupRegistry:
    """Record the termination cleanups the dashboard installs, in place of the real handler."""
    return track_mounted_cleanups(monkeypatch)


def _screen(raw: str) -> list[str]:
    return screen_lines(raw, width=_SCREEN_WIDTH, height=_SCREEN_HEIGHT)


def _cursor_hidden(raw: str) -> bool:
    return cursor_hidden(raw, width=_SCREEN_WIDTH, height=_SCREEN_HEIGHT)


def test_stderr_write_when_live_dashboard_up_does_land_above_the_frame(terminal: StringIO):
    kit = make_reporter(mode="live")
    kit.reporter.observer(launch_event(1000))

    sys.stderr.write(f"{WARNING_LINE}\n")
    sys.stderr.flush()

    frame_rows = _screen(render_frame(kit.reporter, width=_SCREEN_WIDTH))
    assert _screen(terminal.getvalue()) == [KEPT_LINE, WARNING_LINE, *frame_rows]


def test_signal_when_dashboard_just_hid_the_cursor_does_restore_the_screen(
    raise_signal: Callable[[int], int],
    monkeypatch: pytest.MonkeyPatch,
):
    term = InterruptedTerminal()
    _mount_terminal(term, monkeypatch)
    install_termination_cleanup(lambda: None)
    term.interrupt_write(lambda: raise_signal(TERMINATION_SIGNAL), marker=HIDE_CURSOR, lands=True)

    with pytest.raises(ProcessExit):
        make_reporter(mode="live")

    assert (_screen(term.at_exit), _cursor_hidden(term.at_exit)) == ([KEPT_LINE], False)


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


# ---------------------------------------------------------------------------
# Render failure — the dashboard survives a frame that fails to render
# ---------------------------------------------------------------------------

# Message of the error the flaky sidecar raises, which the warning names.
_RENDER_FAILURE = "sidecar unreadable"

# The one warning line a failed render leaves on the terminal.
_RENDER_FAILURE_WARNING = (
    f"warning: dashboard frame failed to render: RuntimeError: {_RENDER_FAILURE}"
)

# Refresh interval for the render-failure tests: fast enough that the refresh
# thread renders many frames while a test waits on it.
_FAST_REFRESH_MS = 10

# Upper bound on how long a test waits for the refresh thread to render again.
_RECOVERY_TIMEOUT_S = 5.0


class _FlakySidecar:
    """A sidecar reader a test arms to fail its next reads.

    ``fail_next(count)`` makes the next *count* reads raise; the first read
    that succeeds after them sets ``recovered``. ``fail_next(None)`` makes every
    read from then on raise. The dashboard reads the sidecar on every frame
    while an iterate call is in flight, so each failed read is a failed render.
    """

    def __init__(self) -> None:
        self.recovered = threading.Event()
        self._lock = threading.Lock()
        self._failures_left: int | None = 0
        self._armed = False

    def fail_next(self, count: int | None) -> None:
        with self._lock:
            self._failures_left = count
            self._armed = True

    def __call__(self, _root: str) -> ProgressSnapshot | None:
        with self._lock:
            if self._failures_left is None or self._failures_left > 0:
                if self._failures_left is not None:
                    self._failures_left -= 1
                raise RuntimeError(_RENDER_FAILURE)
            if self._armed:
                self.recovered.set()
        return None


def _dashboard_reading(sidecar: _FlakySidecar) -> ReporterKit:
    """Mount a fast-refreshing live dashboard with an iterate call reading *sidecar*."""
    kit = make_reporter(mode="live", read_progress=sidecar, refresh_ms=_FAST_REFRESH_MS)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(tool_start_event("Bash", "bash-1", 2000, input_summary="gymrat iterate"))
    return kit


def _painted_after(terminal: StringIO, length: int) -> bool:
    """Wait for the refresh thread to write past *length* characters of *terminal*."""
    deadline = time.monotonic() + _RECOVERY_TIMEOUT_S
    while time.monotonic() < deadline:
        if len(terminal.getvalue()) > length:
            return True
        time.sleep(_FAST_REFRESH_MS / 1000)
    return False


def test_refresh_when_a_frame_fails_to_render_does_keep_painting_later_frames(
    terminal: StringIO,
):
    sidecar = _FlakySidecar()
    _dashboard_reading(sidecar)
    sidecar.fail_next(1)
    recovered = sidecar.recovered.wait(_RECOVERY_TIMEOUT_S)

    painted = _painted_after(terminal, len(terminal.getvalue()))

    assert (recovered, painted) == (True, True)


def test_refresh_when_frames_fail_to_render_in_live_mode_does_warn_once_naming_the_error(
    terminal: StringIO,
):
    sidecar = _FlakySidecar()
    kit = _dashboard_reading(sidecar)
    sidecar.fail_next(2)
    recovered = sidecar.recovered.wait(_RECOVERY_TIMEOUT_S)

    kit.reporter.stop()

    assert (recovered, _screen(terminal.getvalue())) == (
        True,
        [KEPT_LINE, _RENDER_FAILURE_WARNING],
    )


def test_create_reporter_when_plain_mode_frame_would_fail_does_not_warn():
    sidecar = _FlakySidecar()
    sidecar.fail_next(None)
    plain_lines: list[str] = []
    kit = make_reporter(mode="plain", read_progress=sidecar, plain_write=plain_lines.append)
    kit.reporter.observer(launch_event(1000))
    kit.reporter.observer(tool_start_event("Bash", "bash-1", 2000, input_summary="gymrat iterate"))

    kit.reporter.stop()

    assert not any("failed to render" in line for line in plain_lines)


def test_stop_when_final_frame_fails_to_render_does_return_normally(terminal: StringIO):
    sidecar = _FlakySidecar()
    kit = _dashboard_reading(sidecar)
    sidecar.fail_next(None)

    kit.reporter.stop()


class _SetupFailingClock(Clock):
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
