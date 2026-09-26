"""Tests for CLI style vocabulary constants, theme wiring, and signal clearing.

Running-state elements (spinners, in-flight timers) render cyan.  Alert
surfaces (idle warnings, caps) render yellow.  The theme entries that Rich
progress columns hard-code (``progress.spinner``, ``progress.elapsed``) follow
the style constants so a colour change in one place propagates everywhere.

``LiveDisplayMixin.clear_on_signal`` writes raw escapes past rich, so its tests
replay the console's paint history plus the handler's write through a ``pyte``
screen: the screen a user is left with after ``os._exit``.

A signal can land while a frame is being painted, either by the auto-refresh
thread or by the main thread itself.  The race tests freeze a paint at a known
point -- a frame whose render blocks until released, or a console file that
runs the handler in place of its next write -- so the outcome never depends on
scheduling.
"""

from __future__ import annotations

import sys
import threading
import time
from io import StringIO
from typing import TYPE_CHECKING, override

import pyte
import pyte.modes
import pytest
from rich.style import Style
from rich.text import Text

from gymrat.cli.style import (
    CLI_THEME,
    STYLE_ALERT,
    STYLE_RUNNING,
    STYLE_TIMER_RUNNING,
    ErasableLive,
    LiveDisplayMixin,
)
from tests._rich import console_output, screen_lines, sealed_console

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rich.console import Console, ConsoleOptions, RenderableType, RenderResult

# ---------------------------------------------------------------------------
# running vs alert colour split
# ---------------------------------------------------------------------------


def test_style_running_when_referenced_does_equal_cyan():
    assert STYLE_RUNNING == "cyan"


def test_style_timer_running_when_referenced_does_equal_cyan():
    assert STYLE_TIMER_RUNNING == "cyan"


def test_style_alert_when_referenced_does_equal_yellow():
    assert STYLE_ALERT == "yellow"


# ---------------------------------------------------------------------------
# CLI_THEME wiring
# ---------------------------------------------------------------------------


def test_cli_theme_when_spinner_resolved_does_match_running_style():
    assert CLI_THEME.styles["progress.spinner"] == Style.parse(STYLE_RUNNING)


def test_cli_theme_when_elapsed_resolved_does_match_timer_running_style():
    assert CLI_THEME.styles["progress.elapsed"] == Style.parse(STYLE_TIMER_RUNNING)


# ---------------------------------------------------------------------------
# LiveDisplayMixin.clear_on_signal
# ---------------------------------------------------------------------------


class _Renderer(LiveDisplayMixin):
    """A minimal mixin client that paints whatever frame it is handed."""

    def __init__(self, live: ErasableLive | None) -> None:
        self._live = live
        self._stopped = False

    def paint(self, frame: Text) -> None:
        assert self._live is not None
        self._live.update(frame)
        self._refresh_live()

    def queue(self, frame: RenderableType) -> None:
        assert self._live is not None
        self._live.update(frame)


def _rows(count: int) -> Text:
    return Text("\n".join(f"row {index}" for index in range(1, count + 1)))


# Generous bound on waits that only time out when the code under test is broken.
_WAIT_SECONDS = 5.0
# How long the handler must wait for a frozen refresh-thread paint to resume.
_RELEASE_DELAY_SECONDS = 0.05
# Many refresh intervals at 100 refreshes per second: a live thread paints here.
_QUIET_SECONDS = 0.2


class _GatedFrame:
    """A frame whose first render blocks until the test releases it."""

    def __init__(self, rows: int, *, gate_seconds: float = _WAIT_SECONDS) -> None:
        self._text = _rows(rows)
        self._gate_seconds = gate_seconds
        self.entered = threading.Event()
        self.release = threading.Event()

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if not self.entered.is_set():
            self.entered.set()
            self.release.wait(timeout=self._gate_seconds)
        yield self._text


class _WriteLog(StringIO):
    """A console file that flags every write made off the main thread."""

    def __init__(self) -> None:
        super().__init__()
        self.painted_off_main = threading.Event()

    @override
    def write(self, text: str) -> int:
        written = super().write(text)
        if threading.current_thread() is not threading.main_thread():
            self.painted_off_main.set()
        return written


class _StalledWrites(StringIO):
    """A console file whose off-main-thread writes, once armed, block until released."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = threading.Event()
        self.stalled = threading.Event()
        self.release = threading.Event()

    @override
    def write(self, text: str) -> int:
        if self.armed.is_set() and threading.current_thread() is not threading.main_thread():
            self.stalled.set()
            self.release.wait(timeout=2 * _WAIT_SECONDS)
        return super().write(text)


class _ProcessExit(BaseException):
    """Stands in for the ``os._exit`` that follows the handler in production."""


class _InterruptedFile(StringIO):
    """A console file that runs a signal handler in place of its next write."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: Callable[[], None] | None = None

    def interrupt_next_write(self, handler: Callable[[], None]) -> None:
        self._handler = handler

    @override
    def write(self, text: str) -> int:
        handler, self._handler = self._handler, None
        if handler is None:
            return super().write(text)
        handler()
        raise _ProcessExit


def _replay(raw: str) -> pyte.Screen:
    screen = pyte.Screen(80, 24)
    screen.set_mode(pyte.modes.LNM)
    pyte.Stream(screen).feed(raw)
    return screen


@pytest.fixture
def live_renderer() -> Iterator[Callable[..., _Renderer]]:
    """Build live renderers on a console and stop their displays at teardown."""
    started: list[ErasableLive] = []

    def build(console: Console, *, auto_refresh: bool = False) -> _Renderer:
        live = ErasableLive(
            console=console,
            auto_refresh=auto_refresh,
            refresh_per_second=100,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        live.start()
        started.append(live)
        return _Renderer(live)

    yield build
    for live in started:
        live.stop()


def test_clear_on_signal_when_live_up_does_show_cursor(
    monkeypatch: pytest.MonkeyPatch,
    live_renderer: Callable[..., _Renderer],
):
    console = sealed_console()
    renderer = live_renderer(console)
    renderer.paint(_rows(3))
    monkeypatch.setattr(sys, "stderr", console.file)
    hidden_before = _replay(console_output(console)).cursor.hidden

    renderer.clear_on_signal()

    assert (hidden_before, _replay(console_output(console)).cursor.hidden) == (True, False)


@pytest.mark.parametrize("row_count", [1, 4])
def test_clear_on_signal_when_live_up_does_blank_every_frame_row(
    monkeypatch: pytest.MonkeyPatch,
    live_renderer: Callable[..., _Renderer],
    row_count: int,
):
    console = sealed_console()
    console.print("kept above")
    renderer = live_renderer(console)
    renderer.paint(_rows(row_count))
    monkeypatch.setattr(sys, "stderr", console.file)

    renderer.clear_on_signal()

    assert screen_lines(console_output(console)) == ["kept above"]


def test_clear_on_signal_when_live_never_painted_does_only_show_cursor(
    monkeypatch: pytest.MonkeyPatch,
    live_renderer: Callable[..., _Renderer],
):
    renderer = live_renderer(sealed_console())
    buffer = StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)

    renderer.clear_on_signal()

    assert buffer.getvalue() == "\x1b[?25h"


def test_clear_on_signal_when_plain_mode_does_clear_only_current_line(
    monkeypatch: pytest.MonkeyPatch,
):
    renderer = _Renderer(None)
    buffer = StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)

    renderer.clear_on_signal()

    assert buffer.getvalue() == "\r\x1b[K"


def test_clear_on_signal_when_called_twice_does_write_nothing_second_time(
    monkeypatch: pytest.MonkeyPatch,
    live_renderer: Callable[..., _Renderer],
):
    console = sealed_console()
    renderer = live_renderer(console)
    renderer.paint(_rows(3))
    monkeypatch.setattr(sys, "stderr", StringIO())
    renderer.clear_on_signal()
    buffer = StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)

    renderer.clear_on_signal()

    assert buffer.getvalue() == ""


@pytest.mark.parametrize(
    ("rows_before", "rows_after"),
    [pytest.param(2, 4, id="grows"), pytest.param(4, 2, id="shrinks")],
)
def test_clear_on_signal_when_refresh_thread_mid_paint_does_erase_frame_it_paints(
    monkeypatch: pytest.MonkeyPatch,
    live_renderer: Callable[..., _Renderer],
    rows_before: int,
    rows_after: int,
):
    console = sealed_console()
    log = _WriteLog()
    console.file = log
    console.print("kept above")
    renderer = live_renderer(console, auto_refresh=True)
    renderer.paint(_rows(rows_before))
    frame = _GatedFrame(rows_after)
    renderer.queue(frame)
    frame.entered.wait(timeout=_WAIT_SECONDS)
    log.painted_off_main.clear()
    monkeypatch.setattr(sys, "stderr", log)
    threading.Timer(_RELEASE_DELAY_SECONDS, frame.release.set).start()

    renderer.clear_on_signal()
    log.painted_off_main.wait(timeout=_WAIT_SECONDS)

    assert screen_lines(log.getvalue()) == ["kept above"]


def test_clear_on_signal_when_refresh_thread_paint_never_finishes_does_return(
    monkeypatch: pytest.MonkeyPatch,
    live_renderer: Callable[..., _Renderer],
):
    console = sealed_console()
    console.print("kept above")
    renderer = live_renderer(console, auto_refresh=True)
    renderer.paint(_rows(2))
    # The gate outlasts the assertion bound, so an unbounded wait fails decisively.
    frame = _GatedFrame(3, gate_seconds=2 * _WAIT_SECONDS)
    renderer.queue(frame)
    frame.entered.wait(timeout=_WAIT_SECONDS)
    buffer = StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)

    started = time.monotonic()
    try:
        renderer.clear_on_signal()
    finally:
        returned_after = time.monotonic() - started
        on_screen = console_output(console) + buffer.getvalue()
        frame.release.set()

    assert returned_after < _WAIT_SECONDS
    assert screen_lines(on_screen) == ["kept above"]


@pytest.mark.parametrize(
    ("rows_before", "rows_after"),
    [pytest.param(2, 4, id="grows"), pytest.param(4, 2, id="shrinks")],
)
def test_clear_on_signal_when_refresh_thread_write_stalls_does_erase_frame_on_screen(
    monkeypatch: pytest.MonkeyPatch,
    live_renderer: Callable[..., _Renderer],
    rows_before: int,
    rows_after: int,
):
    console = sealed_console()
    terminal = _StalledWrites()
    console.file = terminal
    console.print("kept above")
    renderer = live_renderer(console, auto_refresh=True)
    renderer.paint(_rows(rows_before))
    terminal.armed.set()
    renderer.queue(_rows(rows_after))
    terminal.stalled.wait(timeout=_WAIT_SECONDS)
    buffer = StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)

    try:
        renderer.clear_on_signal()
    finally:
        on_screen = terminal.getvalue() + buffer.getvalue()
        terminal.release.set()

    assert screen_lines(on_screen) == ["kept above"]


def test_clear_on_signal_when_refresh_thread_running_does_stop_its_paints(
    monkeypatch: pytest.MonkeyPatch,
    live_renderer: Callable[..., _Renderer],
):
    console = sealed_console()
    log = _WriteLog()
    console.file = log
    renderer = live_renderer(console, auto_refresh=True)
    renderer.paint(_rows(2))
    log.painted_off_main.wait(timeout=_WAIT_SECONDS)
    monkeypatch.setattr(sys, "stderr", StringIO())

    renderer.clear_on_signal()
    log.painted_off_main.clear()

    assert not log.painted_off_main.wait(timeout=_QUIET_SECONDS)


@pytest.mark.parametrize(
    ("rows_before", "rows_after"),
    [pytest.param(2, 4, id="grows"), pytest.param(4, 2, id="shrinks")],
)
def test_clear_on_signal_when_main_thread_mid_paint_does_erase_frame_on_screen(
    monkeypatch: pytest.MonkeyPatch,
    live_renderer: Callable[..., _Renderer],
    rows_before: int,
    rows_after: int,
):
    console = sealed_console()
    terminal = _InterruptedFile()
    console.file = terminal
    console.print("kept above")
    renderer = live_renderer(console)
    renderer.paint(_rows(rows_before))
    monkeypatch.setattr(sys, "stderr", terminal)
    terminal.interrupt_next_write(renderer.clear_on_signal)

    with pytest.raises(_ProcessExit):
        renderer.paint(_rows(rows_after))

    assert screen_lines(terminal.getvalue()) == ["kept above"]
