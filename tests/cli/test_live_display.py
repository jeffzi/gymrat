"""Tests for the live display's erase on a termination signal.

``mount_live`` installs a termination cleanup that hands raw escapes to the
handler, which writes them past rich.  Its tests mount a live display, run the
installed handler with the process exit stubbed out, and replay the console's
paint history plus the handler's write through a ``pyte`` screen: the screen a
user is left with after ``os._exit``.

A signal can land while a frame is being painted, either by the auto-refresh
thread or by the main thread itself.  The race tests freeze a paint at a known
point -- a frame whose render blocks until released, a console file whose write
stalls, or one that runs the handler at a chosen write -- so the outcome never
depends on scheduling.  While a write is in flight the erase cannot know whether
its frame reached the terminal, so it covers the shorter of the old and new
frames: rows of the taller one may remain, but no line above the frame goes.
"""

from __future__ import annotations

import sys
import threading
import time
import warnings
from io import StringIO
from typing import TYPE_CHECKING, NamedTuple, override

import pytest
from rich.text import Text

from gymrat.cli.live_display import ErasableLive, mount_live
from gymrat.signals import install_termination_cleanup, write_on_exit
from tests._process_helpers import InterruptedTerminal, ProcessExit, track_cleanups
from tests._rich import (
    HIDE_CURSOR,
    KEPT_LINE,
    TERMINATION_SIGNAL,
    WARNING_LINE,
    console_output,
    cursor_hidden,
    screen_lines,
    sealed_console,
)
from tests._streams import (
    RaisingStream,
    RecordingStream,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rich.console import Console, ConsoleOptions, RenderableType, RenderResult

# ---------------------------------------------------------------------------
# mount_live -- erase on a termination signal
# ---------------------------------------------------------------------------


# The error a frame raises when its first paint fails.
_PAINT_FAILURE = "first paint failed"
# The error a frame raises when a later paint, such as Live.stop()'s, fails.
_REPAINT_FAILURE = "final paint failed"
# The error a terminal that can no longer be written raises.
_TERMINAL_GONE = "terminal gone"
# Generous bound on waits that only time out when the code under test is broken.
_WAIT_SECONDS = 5.0
# How often a helper thread checks whether the display has been erased yet.
_POLL_SECONDS = 0.005


class _InFlight(NamedTuple):
    """A frame of old rows repainted as new rows, with a signal landing mid-write.

    ``lands`` says whether the new frame's write reached the terminal before the
    signal; ``screen`` is what the terminal shows once the erase has run.
    """

    rows_before: int
    rows_after: int
    lands: bool
    screen: list[str]


# The signal lands before the new frame's write, so the old frame is still on screen.
_BEFORE_WRITE_CASES = [
    pytest.param(_InFlight(2, 4, lands=False, screen=[KEPT_LINE]), id="grows-before-write"),
    pytest.param(
        _InFlight(4, 2, lands=False, screen=[KEPT_LINE, "old 1", "old 2"]),
        id="shrinks-before-write",
    ),
]

# The erase covers the shorter frame, so the taller frame's extra rows remain.
_IN_FLIGHT_CASES = [
    *_BEFORE_WRITE_CASES,
    pytest.param(
        _InFlight(2, 4, lands=True, screen=[KEPT_LINE, "new 1", "new 2"]), id="grows-after-write"
    ),
    pytest.param(_InFlight(4, 2, lands=True, screen=[KEPT_LINE]), id="shrinks-after-write"),
]


def _rows(count: int, label: str = "row") -> Text:
    return Text("\n".join(f"{label} {index}" for index in range(1, count + 1)))


# The thread the termination handler writes its exit output from.
_EXIT_WRITER_THREAD = "gymrat-exit-output"


def _on_refresh_thread() -> bool:
    # Besides the main thread, only rich's refresh thread paints; the handler's
    # exit-output writer is the one other thread that writes to the terminal.
    thread = threading.current_thread()
    return thread is not threading.main_thread() and thread.name != _EXIT_WRITER_THREAD


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


class _ResizableFrame:
    """A frame of "old" rows that the test resizes in place into "new" rows.

    rich's print path repaints the renderable the last ``refresh()`` handed it,
    so a print picks up a height change only from a frame that changes itself.
    """

    def __init__(self, count: int) -> None:
        self._text = _rows(count, "old")

    def resize(self, count: int) -> None:
        self._text = _rows(count, "new")

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self._text


class _Abort(BaseException):
    """Stands in for a ``KeyboardInterrupt``, which would end the test run itself."""


class _FailingFrame:
    """A frame whose first render raises, and every later render *repaint_error*, if given."""

    def __init__(self, *, repaint_error: BaseException | None = None) -> None:
        self._repaint_error = repaint_error
        self._rendered = False

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if not self._rendered:
            self._rendered = True
            raise RuntimeError(_PAINT_FAILURE)
        if self._repaint_error is not None:
            raise self._repaint_error
        yield from ()


def _release_once_erased(live: ErasableLive, release: threading.Event) -> None:
    """Set *release* once the erase has begun, so a gated paint resumes during its wait."""
    deadline = time.monotonic() + _WAIT_SECONDS
    while not live.erased and time.monotonic() < deadline:
        time.sleep(_POLL_SECONDS)
    release.set()


def _spawned_threads_ended(before: set[threading.Thread]) -> bool:
    """Whether every thread started since *before* was snapshotted has ended."""
    spawned = [thread for thread in threading.enumerate() if thread not in before]
    for thread in spawned:
        thread.join(timeout=_WAIT_SECONDS)
    return not any(thread.is_alive() for thread in spawned)


class _WriteLog(StringIO):
    """A console file that flags every write rich's refresh thread makes."""

    def __init__(self) -> None:
        super().__init__()
        self.painted_by_refresh = threading.Event()

    @override
    def write(self, text: str) -> int:
        written = super().write(text)
        if _on_refresh_thread():
            self.painted_by_refresh.set()
        return written


class _StalledWrites(StringIO):
    """A console file that stalls rich's refresh thread on its first write of a marked frame.

    The refresh-thread write carrying *marker* sets ``stalled`` and blocks,
    holding rich's display lock, until ``release`` is set. Its text lands before
    the stall when *lands* is true and after it otherwise. Every other write,
    such as the signal handler's, passes straight through.
    """

    def __init__(self, marker: str, *, lands: bool) -> None:
        super().__init__()
        self._marker = marker
        self._lands = lands
        self.stalled = threading.Event()
        self.release = threading.Event()

    @override
    def write(self, text: str) -> int:
        if self.stalled.is_set() or self._marker not in text or not _on_refresh_thread():
            return super().write(text)
        if self._lands:
            super().write(text)
        self.stalled.set()
        self.release.wait(timeout=2 * _WAIT_SECONDS)
        return len(text) if self._lands else super().write(text)


def _warn() -> None:
    warnings.warn(WARNING_LINE, RuntimeWarning, stacklevel=1)


@pytest.fixture
def mounted_live() -> Iterator[Callable[..., ErasableLive]]:
    """Mount live displays on a console and take them down at teardown."""
    mounted: list[tuple[ErasableLive, Callable[[], None]]] = []

    def mount(
        console: Console,
        frame: RenderableType | None,
        *,
        auto_refresh: bool = False,
        redirect_stdout: bool = False,
        redirect_stderr: bool = False,
        get_renderable: Callable[[], RenderableType] | None = None,
    ) -> ErasableLive:
        live = ErasableLive(
            frame,
            console=console,
            auto_refresh=auto_refresh,
            refresh_per_second=100,
            redirect_stdout=redirect_stdout,
            redirect_stderr=redirect_stderr,
            get_renderable=get_renderable,
        )
        mounted.append((live, mount_live(live)))
        return live

    yield mount
    for live, uninstall in mounted:
        uninstall()
        live.stop()


class _MidPaint(NamedTuple):
    """The console, write log and gated frame of a display erased mid-paint, plus the threads alive before it mounted."""

    console: Console
    log: _WriteLog
    frame: _GatedFrame
    threads_before: set[threading.Thread]


def _signal_mid_paint(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
) -> _MidPaint:
    """Erase on a signal while a refresh-thread paint outlasts the erase's wait."""
    threads_before = set(threading.enumerate())
    console = sealed_console()
    log = _WriteLog()
    console.file = log
    console.print(KEPT_LINE)
    live = mounted_live(console, _rows(2), auto_refresh=True)
    # The gate outlasts the erase's wait, so the paint is still in flight after it.
    frame = _GatedFrame(3, gate_seconds=2 * _WAIT_SECONDS)
    live.update(frame)
    frame.entered.wait(timeout=_WAIT_SECONDS)
    monkeypatch.setattr(sys, "stderr", log)
    raise_signal(TERMINATION_SIGNAL)
    return _MidPaint(console, log, frame, threads_before)


@pytest.mark.parametrize(
    ("row_count", "redirect_stderr"),
    [
        pytest.param(1, False, id="one-row"),
        pytest.param(4, False, id="four-rows"),
        pytest.param(3, True, id="live-redirects-stderr"),
    ],
)
def test_mount_live_when_signal_arrives_does_blank_every_frame_row_with_the_cursor_shown(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
    row_count: int,
    redirect_stderr: bool,
):
    console = sealed_console()
    console.print(KEPT_LINE)
    monkeypatch.setattr(sys, "stderr", console.file)
    mounted_live(console, _rows(row_count), redirect_stderr=redirect_stderr)

    raise_signal(TERMINATION_SIGNAL)

    out = console_output(console)
    assert (screen_lines(out), cursor_hidden(out)) == ([KEPT_LINE], False)


def test_mount_live_when_signal_lands_before_first_paint_does_restore_the_screen(
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
):
    console = sealed_console()
    terminal = InterruptedTerminal()
    console.file = terminal
    console.print(KEPT_LINE)
    monkeypatch.setattr(sys, "stderr", terminal)
    terminal.interrupt_write(
        lambda: raise_signal(TERMINATION_SIGNAL), marker=HIDE_CURSOR, lands=True
    )
    live = ErasableLive(_rows(3), console=console, auto_refresh=False)

    with pytest.raises(ProcessExit):
        mount_live(live)

    assert (screen_lines(terminal.at_exit), cursor_hidden(terminal.at_exit)) == ([KEPT_LINE], False)


def test_mount_live_when_mounted_does_start_the_display(
    mounted_live: Callable[..., ErasableLive],
):
    console = sealed_console()

    live = mounted_live(console, _rows(2))

    assert live.is_started is True


def _live_whose_first_paint_raises() -> ErasableLive:
    """An auto-refreshing live display whose first frame render raises."""
    return ErasableLive(
        _FailingFrame(), console=sealed_console(), auto_refresh=True, refresh_per_second=4
    )


def _live_on_a_gone_terminal() -> ErasableLive:
    """An auto-refreshing live display whose console's every write raises."""
    console = sealed_console()
    console.file = RaisingStream(OSError(_TERMINAL_GONE))
    return ErasableLive(_rows(2), console=console, auto_refresh=True, refresh_per_second=4)


@pytest.mark.parametrize(
    ("build_live", "error", "message"),
    [
        pytest.param(
            _live_whose_first_paint_raises, RuntimeError, _PAINT_FAILURE, id="first-paint-raises"
        ),
        pytest.param(_live_on_a_gone_terminal, OSError, _TERMINAL_GONE, id="start-raises"),
    ],
)
def test_mount_live_when_mounting_raises_does_roll_back_the_mount(
    build_live: Callable[[], ErasableLive],
    error: type[Exception],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
):
    registry = track_cleanups(monkeypatch, "gymrat.cli.live_display")
    live = build_live()
    threads_before = set(threading.enumerate())

    with pytest.raises(error, match=message):
        mount_live(live)

    assert (live.is_started, _spawned_threads_ended(threads_before), registry.live()) == (
        False,
        True,
        [],
    )


@pytest.mark.parametrize(
    "stop_error",
    [
        pytest.param(RuntimeError(_REPAINT_FAILURE), id="exception"),
        pytest.param(_Abort(_REPAINT_FAILURE), id="base-exception"),
    ],
)
def test_mount_live_when_rollback_stop_raises_does_raise_the_mount_failure_with_a_note(
    stop_error: BaseException,
):
    live = ErasableLive(
        _FailingFrame(repaint_error=stop_error), console=sealed_console(), auto_refresh=False
    )

    with pytest.raises(RuntimeError, match=_PAINT_FAILURE) as raised:
        mount_live(live)

    assert raised.value.__notes__ == [f"stopping the display also failed: {stop_error!r}"]


def test_erasable_live_refresh_when_erased_does_not_build_or_paint_a_frame(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
):
    builds: list[None] = []

    def build_frame() -> Text:
        builds.append(None)
        return _rows(2)

    console = sealed_console()
    console.print(KEPT_LINE)
    live = mounted_live(console, None, get_renderable=build_frame)
    monkeypatch.setattr(sys, "stderr", console.file)
    raise_signal(TERMINATION_SIGNAL)
    built, erased = len(builds), console_output(console)

    live.refresh()

    assert (len(builds), console_output(console)) == (built, erased)


def test_mount_live_when_later_cleanup_warns_does_write_erase_and_warning_in_one_write(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
):
    console = sealed_console()
    console.print(KEPT_LINE)
    mounted_live(console, _rows(3))
    stderr = RecordingStream()
    monkeypatch.setattr(sys, "stderr", stderr)
    install_termination_cleanup(_warn)

    raise_signal(TERMINATION_SIGNAL)

    on_screen = console_output(console) + "".join(stderr.writes)
    assert (len(stderr.writes), screen_lines(on_screen)) == (1, [KEPT_LINE, WARNING_LINE])


def test_mount_live_when_live_redirects_stdio_does_hand_later_cleanups_the_real_streams(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
):
    stdout, stderr = StringIO(), StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    mounted_live(sealed_console(), _rows(2), redirect_stdout=True, redirect_stderr=True)
    seen: list[tuple[object, object]] = []
    install_termination_cleanup(lambda: seen.append((sys.stdout, sys.stderr)))

    raise_signal(TERMINATION_SIGNAL)

    assert seen == [(stdout, stderr)]


def test_erase_for_exit_when_display_already_erased_does_leave_the_screen_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
):
    console = sealed_console()
    console.print(KEPT_LINE)
    live = mounted_live(console, _rows(3))
    monkeypatch.setattr(sys, "stderr", console.file)
    install_termination_cleanup(lambda: write_on_exit(live.erase_for_exit()))

    raise_signal(TERMINATION_SIGNAL)

    assert screen_lines(console_output(console)) == [KEPT_LINE]


@pytest.mark.parametrize(
    ("rows_before", "rows_after"),
    [pytest.param(2, 4, id="grows"), pytest.param(4, 2, id="shrinks")],
)
def test_mount_live_when_refresh_thread_mid_paint_does_erase_frame_it_paints(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
    rows_before: int,
    rows_after: int,
):
    threads_before = set(threading.enumerate())
    console = sealed_console()
    log = _WriteLog()
    console.file = log
    console.print(KEPT_LINE)
    live = mounted_live(console, _rows(rows_before), auto_refresh=True)
    frame = _GatedFrame(rows_after)
    live.update(frame)
    frame.entered.wait(timeout=_WAIT_SECONDS)
    monkeypatch.setattr(sys, "stderr", log)
    threading.Thread(target=_release_once_erased, args=(live, frame.release), daemon=True).start()

    raise_signal(TERMINATION_SIGNAL)

    assert (_spawned_threads_ended(threads_before), screen_lines(log.getvalue())) == (
        True,
        [KEPT_LINE],
    )


def test_mount_live_when_refresh_thread_paint_never_finishes_does_return_after_one_wait(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
):
    console = sealed_console()
    console.print(KEPT_LINE)
    live = mounted_live(console, _rows(2), auto_refresh=True)
    # The gate outlasts the assertion bound, so an unbounded wait fails decisively.
    frame = _GatedFrame(3, gate_seconds=2 * _WAIT_SECONDS)
    live.update(frame)
    frame.entered.wait(timeout=_WAIT_SECONDS)
    buffer = StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)

    started = time.monotonic()
    try:
        raise_signal(TERMINATION_SIGNAL)
    finally:
        returned_after = time.monotonic() - started
        on_screen = console_output(console) + buffer.getvalue()
        frame.release.set()

    assert returned_after < _WAIT_SECONDS
    assert screen_lines(on_screen) == [KEPT_LINE]


def test_mount_live_when_refresh_thread_paint_outlasts_erase_does_not_land_its_frame(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
):
    _console, log, frame, threads_before = _signal_mid_paint(
        monkeypatch, mounted_live, raise_signal
    )

    frame.release.set()

    assert (_spawned_threads_ended(threads_before), screen_lines(log.getvalue())) == (
        True,
        [KEPT_LINE],
    )


def test_console_print_when_signal_erased_display_mid_paint_does_print_without_frame(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
):
    console, log, frame, _threads_before = _signal_mid_paint(
        monkeypatch, mounted_live, raise_signal
    )
    # rich holds the console lock while rendering a frame, so the print waits
    # for the in-flight paint either way; releasing it first keeps the test fast.
    frame.release.set()

    console.print("banana")

    assert screen_lines(log.getvalue()) == [KEPT_LINE, "banana"]


def test_console_print_when_signal_erased_display_lock_held_does_return_within_bound(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
):
    armed, held, release = threading.Event(), threading.Event(), threading.Event()

    def build_frame() -> Text:
        # rich builds the frame while holding the display lock, so a stalled
        # build keeps that lock without holding the console's own.
        if armed.is_set() and not held.is_set():
            held.set()
            release.wait(timeout=2 * _WAIT_SECONDS)
        return _rows(2)

    console = sealed_console()
    live = mounted_live(console, None, get_renderable=build_frame)
    monkeypatch.setattr(sys, "stderr", console.file)
    armed.set()
    threading.Thread(target=live.refresh, daemon=True).start()
    held.wait(timeout=_WAIT_SECONDS)
    raise_signal(TERMINATION_SIGNAL)

    started = time.monotonic()
    try:
        console.print("banana")
    finally:
        returned_after = time.monotonic() - started
        release.set()

    assert returned_after < _WAIT_SECONDS


@pytest.mark.parametrize("case", _IN_FLIGHT_CASES)
def test_mount_live_when_refresh_thread_write_stalls_does_erase_the_shorter_frame(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
    case: _InFlight,
):
    console = sealed_console()
    terminal = _StalledWrites("new", lands=case.lands)
    console.file = terminal
    console.print(KEPT_LINE)
    live = mounted_live(console, _rows(case.rows_before, "old"), auto_refresh=True)
    live.update(_rows(case.rows_after, "new"))
    terminal.stalled.wait(timeout=_WAIT_SECONDS)
    buffer = StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)

    try:
        raise_signal(TERMINATION_SIGNAL)
    finally:
        on_screen = terminal.getvalue() + buffer.getvalue()
        terminal.release.set()

    assert screen_lines(on_screen) == case.screen


def test_mount_live_when_refresh_thread_running_does_stop_its_paints(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
):
    threads_before = set(threading.enumerate())
    console = sealed_console()
    log = _WriteLog()
    console.file = log
    mounted_live(console, _rows(2), auto_refresh=True)
    log.painted_by_refresh.wait(timeout=_WAIT_SECONDS)
    monkeypatch.setattr(sys, "stderr", StringIO())

    raise_signal(TERMINATION_SIGNAL)

    assert _spawned_threads_ended(threads_before)


@pytest.mark.parametrize("case", _IN_FLIGHT_CASES)
def test_mount_live_when_main_thread_paint_interrupted_does_erase_the_shorter_frame(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
    case: _InFlight,
):
    console = sealed_console()
    terminal = InterruptedTerminal()
    console.file = terminal
    console.print(KEPT_LINE)
    live = mounted_live(console, _rows(case.rows_before, "old"))
    monkeypatch.setattr(sys, "stderr", terminal)
    terminal.interrupt_write(lambda: raise_signal(TERMINATION_SIGNAL), lands=case.lands)

    with pytest.raises(ProcessExit):
        live.update(_rows(case.rows_after, "new"), refresh=True)

    assert screen_lines(terminal.at_exit) == case.screen


# The thread whose render of the frame the next test holds at a gate.
_GATED_REFRESH_THREAD = "gated-refresh"


class _FrameGatedOnThread(_ResizableFrame):
    """A resizable frame whose render blocks, on the thread named ``_GATED_REFRESH_THREAD`` only, until released."""

    def __init__(self, count: int) -> None:
        super().__init__(count)
        self.entered = threading.Event()
        self.release = threading.Event()

    @override
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if threading.current_thread().name == _GATED_REFRESH_THREAD:
            self.entered.set()
            self.release.wait(timeout=_WAIT_SECONDS)
        yield from super().__rich_console__(console, options)


def test_erase_for_exit_when_paint_dropped_while_earlier_paint_in_flight_does_erase_the_shorter_frame(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
):
    console = sealed_console()
    console.print(KEPT_LINE)
    frame = _FrameGatedOnThread(2)
    live = mounted_live(console, frame)
    # The erase waits on the display lock until the refresh has dropped its
    # frame and let go, however late the release fires.
    live.paint_wait_seconds = _WAIT_SECONDS
    monkeypatch.setattr(sys, "stderr", console.file)
    frame.resize(4)
    refresh = threading.Thread(target=live.refresh, name=_GATED_REFRESH_THREAD, daemon=True)

    def signal_once_landing() -> None:
        # The print has rendered its four-row frame and will land it, but has
        # not written yet: the two old rows are still on screen. A refresh then
        # records the four-row frame as the one it replaces, holds the display
        # lock through the erase's wait, and drops its frame once erased.
        if threading.current_thread() is threading.main_thread():
            refresh.start()
            frame.entered.wait(timeout=_WAIT_SECONDS)
            # The refresh must reach its landing decision after the erase has
            # begun, so it drops its frame rather than landing it.
            threading.Thread(
                target=_release_once_erased, args=(live, frame.release), daemon=True
            ).start()
            raise_signal(TERMINATION_SIGNAL)
            raise ProcessExit

    live.on_frame_landing = signal_once_landing

    try:
        with pytest.raises(ProcessExit):
            console.print("banana")
    finally:
        frame.release.set()

    assert screen_lines(console_output(console)) == [KEPT_LINE]


@pytest.mark.parametrize(
    "case",
    [
        *_BEFORE_WRITE_CASES,
        pytest.param(
            _InFlight(2, 4, lands=True, screen=[KEPT_LINE, "banana", "new 1", "new 2"]),
            id="grows-after-write",
        ),
        pytest.param(
            _InFlight(4, 2, lands=True, screen=[KEPT_LINE, "banana"]), id="shrinks-after-write"
        ),
    ],
)
def test_mount_live_when_print_interrupted_does_erase_the_shorter_frame(
    monkeypatch: pytest.MonkeyPatch,
    mounted_live: Callable[..., ErasableLive],
    raise_signal: Callable[[int], int],
    case: _InFlight,
):
    console = sealed_console()
    terminal = InterruptedTerminal()
    console.file = terminal
    console.print(KEPT_LINE)
    frame = _ResizableFrame(case.rows_before)
    mounted_live(console, frame)
    frame.resize(case.rows_after)
    monkeypatch.setattr(sys, "stderr", terminal)
    terminal.interrupt_write(lambda: raise_signal(TERMINATION_SIGNAL), lands=case.lands)

    with pytest.raises(ProcessExit):
        console.print("banana")

    assert screen_lines(terminal.at_exit) == case.screen
