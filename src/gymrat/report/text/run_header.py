"""The run header every report opens on: the command, the target, the samples, the adapter.

A comparison names its baseline and candidates and counts paired samples; a
measurement and a probe both name one run target and count its samples. All
three lay the line out the same way, so the layout is stated once here, and the
single-target shape measure and probe share is stated once too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markup import escape

from gymrat.report.style import VARIANT_NAME_STYLE, join_header_parts, markup
from gymrat.utils import pluralize

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


def target_run_header(
    command: str, label: str, samples: int, adapter: str, extra: Sequence[str] = ()
) -> str:
    """A single-target run header as markup, the shape measure and probe share.

    Args:
        command: The subcommand the report belongs to, such as ``"measure"``.
        label: The plain-text name of the run target, styled here as a variant
            name.
        samples: How many samples the run took, counted here as ``N samples``.
        adapter: The adapter that parsed the bench output.
        extra: Further markup parts appended after the adapter.

    Returns:
        The header parts joined into one markup line.
    """
    return run_header(
        command,
        markup(label, VARIANT_NAME_STYLE),
        pluralize(samples, "sample"),
        adapter,
        extra,
    )
