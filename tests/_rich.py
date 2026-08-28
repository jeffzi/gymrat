"""Shared Rich / pyte test helpers for progress renderer tests.

Provides a sealed console, a one-shot plain-text renderer, a pyte screen
replay helper, and a hand-advanced clock.  Every helper is deterministic
and isolated from the developer's environment.
"""

from __future__ import annotations

from io import StringIO

import pyte
from rich.console import Console, RenderableType


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
        """Advance the clock by *seconds*."""
        self.now += seconds


def sealed_console(
    *,
    width: int = 80,
    height: int = 24,
    no_color: bool = True,
    color_system: str | None = None,
) -> Console:
    """A ``Console`` sealed from the developer's environment.

    Fixed dimensions, ``force_terminal=True``, ``legacy_windows=False``,
    ``_environ={}``.  When *no_color* is ``True`` (default), sets
    ``no_color=True``; when ``False``, sets ``color_system`` to the given
    value (default ``"truecolor"``).  # cspell:disable-line

    Returns a ``Console`` writing to a ``StringIO``.
    """
    kwargs: dict[str, object] = {
        "file": StringIO(),
        "width": width,
        "height": height,
        "force_terminal": True,
        "legacy_windows": False,
        "_environ": {},
    }
    if no_color:
        kwargs["no_color"] = True
    else:
        kwargs["color_system"] = color_system or "truecolor"  # cspell:disable-line
    return Console(**kwargs)  # type: ignore[arg-type]


def frame_text(renderable: RenderableType, *, width: int = 80) -> str:
    """Render *renderable* through a throwaway non-terminal console, return plain text.

    The console is NOT a terminal -- it is a one-shot renderer with
    ``force_terminal=False`` and ``no_color=True``.
    """
    buf = StringIO()
    console = Console(
        file=buf,
        width=width,
        force_terminal=False,
        no_color=True,
        _environ={},
    )
    console.print(renderable)
    return buf.getvalue()


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
