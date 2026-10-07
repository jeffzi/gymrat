"""Shared Rich / pyte test helpers for progress renderer tests.

Provides a sealed console, a one-shot plain-text renderer, a pyte screen
replay helper, a hand-advanced clock, and the constants the signal-erase
tests share.  Every helper is deterministic and isolated from the
developer's environment.
"""

from __future__ import annotations

import re
import signal
from io import StringIO
from typing import TYPE_CHECKING, Literal

import pyte
import pyte.modes
from rich.console import Console, RenderableType

from gymrat.cli.style import CLI_THEME

if TYPE_CHECKING:
    from collections.abc import Callable

#: The escape a live display writes as it starts, hiding the cursor until it stops.
HIDE_CURSOR = "\x1b[?25l"

#: The signal the signal-erase tests deliver; any termination signal takes the same path.
TERMINATION_SIGNAL = signal.SIGTERM

#: The line printed above a live display, which the erase must leave in place.
KEPT_LINE = "kept above"

#: A warning a cleanup installed after the erase raises.
WARNING_LINE = "warning: banana"


class Clock[T: (int, float)]:
    """A hand-advanced clock, in whatever unit and number type the test starts it with.

    A float clock in seconds plugs into ``Progress(get_time=...)`` for
    deterministic frames; an int clock in milliseconds drives a reporter's
    ``now``.  Call ``tick(amount)`` to advance, or read and assign ``.now``
    directly.  Callable -- returns ``self.now``.
    """

    def __init__(self, start: T) -> None:
        self.now = start

    def __call__(self) -> T:
        return self.now

    def tick(self, amount: T) -> None:
        # pyrefly: ignore[unsupported-operation] -- T is int or float, and each adds to itself
        self.now += amount


def sealed_console(
    *,
    width: int = 80,
    height: int = 24,
    no_color: bool = True,
    color_system: Literal["auto", "standard", "truecolor"] | None = "auto",
    get_time: Callable[[], float] | None = None,
) -> Console:
    """A ``Console`` sealed from the developer's environment.

    Fixed dimensions, ``force_terminal=True``, ``legacy_windows=False``,
    ``_environ={}``, writing to a ``StringIO``.

    Args:
        width: Console width in columns.
        height: Console height in rows.
        no_color: Whether color is off. Attributes such as bold and dim still
            reach the output unless ``color_system`` is None.
        color_system: The color system the console writes. None with
            ``no_color`` set emits no SGR at all, as ``--no-color`` does.
        get_time: Pins the console clock. Without it, a ``Live`` refreshing on
            this console anchors spinner animations to the wall clock, making
            frames nondeterministic; pass the test's ``Clock`` so every paint
            reads it.

    Returns:
        The sealed console.
    """
    return Console(
        file=StringIO(),
        width=width,
        height=height,
        force_terminal=True,
        legacy_windows=False,
        _environ={},
        no_color=no_color or None,
        color_system=color_system,
        theme=CLI_THEME,
        get_time=get_time,
    )


def frame_text(
    renderable: RenderableType,
    *,
    width: int = 80,
    get_time: Callable[[], float] | None = None,
) -> str:
    """Render ``renderable`` through a throwaway non-terminal console, as plain text.

    The console is NOT a terminal -- it is a one-shot renderer with
    ``force_terminal=False`` and ``no_color=True``.

    Args:
        renderable: What to render.
        width: Console width in columns.
        get_time: Pins the console clock so that a ``Spinner`` picks a
            deterministic frame rather than whatever the wall clock says.
            None uses a clock stopped at zero.

    Returns:
        The rendered text, each line stripped of trailing whitespace.
    """
    buf = StringIO()
    console = Console(
        file=buf,
        width=width,
        force_terminal=False,
        no_color=True,
        legacy_windows=False,
        _environ={},
        get_time=get_time or Clock(0.0),
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
