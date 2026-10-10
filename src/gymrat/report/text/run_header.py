"""The run header every report opens on: the command, the target, the samples, the adapter.

A comparison, a measurement and a probe each name a different target and count
their samples differently, but lay the line out the same way, so the layout is
stated once here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markup import escape

from gymrat.report.style import join_header_parts, markup

if TYPE_CHECKING:
    from collections.abc import Sequence


def run_header(
    command: str, target: str, samples: str, adapter: str, extra: Sequence[str] = ()
) -> str:
    """A run header as markup: the command, the target, the samples, and the adapter.

    Args:
        command: The subcommand the report belongs to, such as ``"measure"``.
        target: The markup naming what the run measured.
        samples: The plain-text sample count, escaped here.
        adapter: The adapter that parsed the bench output.
        extra: Further markup parts appended after the adapter.

    Returns:
        The header parts joined into one markup line.
    """
    return join_header_parts([
        markup(f"gymrat {command}", "bold"),
        target,
        escape(samples),
        f"adapter: {escape(adapter)}",
        *extra,
    ])
