"""Verdict classification: mapping stored verdicts to display classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

from gymrat.model import PERMUTATION_MIN_N

if TYPE_CHECKING:
    from gymrat.model import MetricVerdict

type DisplayClass = Literal[
    "improved",
    "regressed",
    "unstable",
    "identical",
    "within-noise",
    "inconclusive",
]
"""How a verdict presents itself in the report.

``no-signal`` splits in three here: a metric resting on too few pairs for the
band method reads ``inconclusive``, one whose two sides measured close enough
to identical to starve the permutation test reads ``identical``, and every
other no-signal verdict reads ``within-noise``. The split is presentation only
— the stored verdict stays ``no-signal``.
"""


def display_class(verdict: MetricVerdict) -> DisplayClass:
    """Which display class a verdict reads as.

    Any non-exact verdict whose pair count sits below
    :data:`~gymrat.model.PERMUTATION_MIN_N` reads ``inconclusive``: the sample is too
    small for a statistical verdict — band or permutation — to be trusted.
    Exact verdicts are decided on paired medians, so no statistical minimum
    applies to them.

    A band verdict with enough pairs for the permutation test reads
    ``identical`` only when every one of them tied (``usable_n == 0``);
    anywhere ``usable_n`` sits between zero and that floor
    some pairs did differ, so that reads ``within-noise``. An exact
    no-signal always reads ``within-noise``.

    Args:
        verdict: The metric verdict to classify.

    Returns:
        The display class the verdict presents as in the report.
    """
    if verdict.method != "exact" and verdict.n < PERMUTATION_MIN_N:
        return "inconclusive"
    if verdict.verdict != "no-signal":
        return verdict.verdict
    return _no_signal_class(verdict)


def _no_signal_class(verdict: MetricVerdict) -> DisplayClass:
    match verdict.method:
        case "band":
            return "identical" if verdict.usable_n == 0 else "within-noise"
        case "permutation" | "exact":
            return "within-noise"
        case _ as unreachable:  # pragma: no cover — exhaustive match over MetricVerdict.method
            assert_never(unreachable)


GLYPHS: dict[DisplayClass, str] = {
    "improved": "✓",
    "regressed": "✗",
    "unstable": "≈",
    "identical": "=",
    "within-noise": "~",
    "inconclusive": "?",
}
"""The glyph each display class is drawn with in the report's rows and legend."""


VERDICT_GLOSSES: dict[DisplayClass, str] = {cls: cls.replace("-", " ") for cls in GLYPHS}
"""The word for each display class: its name with hyphens as spaces.

Used in the verdict tally and as the delta cell of an unstable verdict.
"""

QUIET_VERDICTS: frozenset[DisplayClass] = frozenset({
    "within-noise",
    "identical",
    "inconclusive",
    "unstable",
})
"""The display classes that report no real change.

A metric row draws a quiet verdict's delta in the verdict color, and a geomean whose metrics are
all quiet stays uncolored whatever its value.
"""
