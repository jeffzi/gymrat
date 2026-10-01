"""Shared Rich / pyte test helpers for progress renderer tests.

Provides a sealed console, a one-shot plain-text renderer, a pyte screen
replay helper, a hand-advanced clock, and the terminal and cleanup doubles
the signal-erase tests share.  Every helper is deterministic and isolated
from the developer's environment.
"""

from __future__ import annotations

import re
import signal
from io import StringIO
from typing import TYPE_CHECKING, override

import pyte
import pyte.modes
from rich.console import Console, RenderableType

from gymrat.cli.style import CLI_THEME

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

#: The escape a live display writes as it starts, hiding the cursor until it stops.
HIDE_CURSOR = "\x1b[?25l"

#: The signal the signal-erase tests deliver; any termination signal takes the same path.
TERMINATION_SIGNAL = signal.SIGTERM

#: The line printed above a live display, which the erase must leave in place.
KEPT_LINE = "kept above"

#: A warning a cleanup installed after the erase raises.
WARNING_LINE = "warning: banana"


def _fixed_time() -> float:
    """The default pinned clock for one-shot renders."""
    return 0.0


class Clock:
    """A hand-advanced clock for deterministic ``Progress(get_time=...)`` frames.

    ``Clock(start=0.0)`` starts at the given time.  Call ``tick(seconds)`` to
    advance, or read ``.now`` directly.  Callable -- returns ``self.now`` -- so
    it plugs into any ``get_time`` parameter.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


def sealed_console(
    *,
    width: int = 80,
    height: int = 24,
    no_color: bool = True,
    color_system: str | None = None,
    get_time: Callable[[], float] | None = None,
) -> Console:
    """A ``Console`` sealed from the developer's environment.

    Fixed dimensions, ``force_terminal=True``, ``legacy_windows=False``,
    ``_environ={}``.  When *no_color* is ``True`` (default), sets
    ``no_color=True``; when ``False``, sets ``color_system`` to the given
    value (default ``"truecolor"``).  # cspell:disable-line

    *get_time* pins the console clock. Without it, a ``Live`` refreshing on
    this console anchors spinner animations to the wall clock, making frames
    nondeterministic; pass the test's ``Clock`` so every paint reads it.

    Returns a ``Console`` writing to a ``StringIO``.
    """
    resolved_color_system = (
        "auto" if no_color else (color_system or "truecolor")  # cspell:disable-line
    )
    return Console(
        file=StringIO(),
        width=width,
        height=height,
        force_terminal=True,
        legacy_windows=False,
        _environ={},
        no_color=no_color or None,
        color_system=resolved_color_system,  # type: ignore[arg-type]
        theme=CLI_THEME,
        get_time=get_time,
    )


def frame_text(
    renderable: RenderableType,
    *,
    width: int = 80,
    get_time: Callable[[], float] | None = None,
) -> str:
    """Render *renderable* through a throwaway non-terminal console, return plain text.

    The console is NOT a terminal -- it is a one-shot renderer with
    ``force_terminal=False`` and ``no_color=True``.

    *get_time* pins the console clock so that a ``Spinner`` picks a deterministic
    frame rather than whatever the wall clock says. Defaults to ``_fixed_time``.
    """
    buf = StringIO()
    console = Console(
        file=buf,
        width=width,
        force_terminal=False,
        no_color=True,
        legacy_windows=False,
        _environ={},
        get_time=get_time or _fixed_time,
    )
    console.print(renderable)
    return "\n".join(line.rstrip() for line in buf.getvalue().splitlines())


def screen_lines(raw: str, *, width: int = 80, height: int = 24) -> list[str]:
    """Replay *raw* through a ``pyte.Screen`` and return visible rows.

    Sets LNM (``pyte.modes.LNM``) so that LF translates to CR+LF the way a
    real terminal does.  Trailing whitespace is stripped per line; trailing
    empty lines are stripped from the result.
    """
    lines = [line.rstrip() for line in _replay(raw, width, height).display]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def cursor_hidden(raw: str, *, width: int = 80, height: int = 24) -> bool:
    """Replay *raw* through a ``pyte.Screen`` and report whether the cursor ends hidden."""
    return _replay(raw, width, height).cursor.hidden


def _replay(raw: str, width: int, height: int) -> pyte.Screen:
    """Feed *raw* to a fresh ``pyte.Screen`` with LF translated to CR+LF, as a terminal does."""
    screen = pyte.Screen(width, height)
    screen.set_mode(pyte.modes.LNM)
    pyte.Stream(screen).feed(raw)
    return screen


def console_output(console: Console) -> str:
    """Return all text written to the console's StringIO."""
    f = console.file
    assert isinstance(f, StringIO)
    return f.getvalue()


def unwrap_panel(text: str) -> str:
    """Flatten a rich panel's borders and line wraps into single-spaced text."""
    return re.sub(r"[│╭╮╰╯─\s]+", " ", text)


def fake_install(
    registered: list[object],
) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """A fake ``install_termination_cleanup`` that records registrations."""

    def install(cb: Callable[[], None]) -> Callable[[], None]:
        registered.append(cb)
        return lambda: None

    return install


class CleanupRegistry:
    """The termination cleanups installed through a patched seam and not yet uninstalled.

    ``install_termination_cleanup`` hands back an uninstall callable, so recording
    both halves gives the live set at any moment: read it from inside a seam to see
    what is armed, and read it afterwards to see what was left behind.
    """

    def __init__(self) -> None:
        self._live: list[Callable[[], None]] = []

    def install(self, cleanup: Callable[[], None]) -> Callable[[], None]:
        """Record *cleanup* as armed.

        Args:
            cleanup: The termination cleanup being installed.

        Returns:
            A callable that removes *cleanup* from the armed set.
        """
        self._live.append(cleanup)

        def uninstall() -> None:
            self._live = [live for live in self._live if live is not cleanup]

        return uninstall

    def live(self) -> list[Callable[[], None]]:
        """Return the cleanups installed and not yet uninstalled, in install order."""
        return list(self._live)


def track_mounted_cleanups(monkeypatch: pytest.MonkeyPatch) -> CleanupRegistry:
    """Swap the cleanup installer ``mount_live`` uses for a registry a test can read.

    Args:
        monkeypatch: The fixture that installs the swap.

    Returns:
        The registry recording every erase cleanup a display mounts.
    """
    registry = CleanupRegistry()
    monkeypatch.setattr("gymrat.cli.style.install_termination_cleanup", registry.install)
    return registry


class ProcessExit(BaseException):
    """Stands in for the ``os._exit`` that ends the process once a signal is handled."""


class InterruptedTerminal(StringIO):
    """A terminal file that runs a signal handler at one chosen write, then exits.

    ``interrupt_write`` arms it for the next write whose text contains *marker*.
    The handler runs in place of that write, or right after the text lands when
    *lands* is true. ``at_exit`` then keeps what reached the terminal, and the
    write raises ``ProcessExit`` where the real process would end, so nothing
    written while the exception unwinds counts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.at_exit = ""
        self._handler: Callable[[], object] | None = None
        self._marker = ""
        self._lands = False

    def interrupt_write(
        self, handler: Callable[[], object], *, marker: str = "", lands: bool = False
    ) -> None:
        """Arm the terminal to run *handler* at the next matching write.

        Args:
            handler: Runs in place of the matching write, or right after its text
                lands when *lands* is true.
            marker: Text the write must contain to trigger the handler; the empty
                string matches every write.
            lands: Whether the write's text reaches the terminal before the
                handler runs.
        """
        self._handler, self._marker, self._lands = handler, marker, lands

    @override
    def write(self, text: str) -> int:
        handler = self._handler
        if handler is None or self._marker not in text:
            return super().write(text)
        self._handler = None
        if self._lands:
            super().write(text)
        handler()
        self.at_exit = self.getvalue()
        raise ProcessExit
