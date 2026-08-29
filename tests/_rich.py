"""Shared Rich / pyte test helpers for progress renderer tests.

Provides a sealed console, a one-shot plain-text renderer, a pyte screen
replay helper, and a hand-advanced clock.  Every helper is deterministic
and isolated from the developer's environment.
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

import pyte
from rich.console import Console, RenderableType

from gymrat.cli.style import CLI_THEME

if TYPE_CHECKING:
    from collections.abc import Callable


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
    screen = pyte.Screen(width, height)
    screen.set_mode(pyte.modes.LNM)
    pyte.Stream(screen).feed(raw)
    lines = [line.rstrip() for line in screen.display]
    while lines and not lines[-1]:
        lines.pop()
    return lines
