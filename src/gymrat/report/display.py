"""Verdict classification: mapping stored verdicts to display classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

from gymrat.model import PERMUTATION_DESCRIPTOR

if TYPE_CHECKING:
    from gymrat.model import MetricVerdict

MIN_PERMUTATION_N = PERMUTATION_DESCRIPTOR.min_n

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
    :data:`MIN_PERMUTATION_N` reads ``inconclusive``: the sample is too
    small for a statistical verdict — band or permutation — to be trusted.
    Exact verdicts are decided on paired medians, so no statistical minimum
    applies to them.

    A band verdict with enough pairs for the permutation test reads
    ``identical`` only when every one of them tied (``usable_n == 0``);
    anywhere ``usable_n`` sits between zero and :data:`MIN_PERMUTATION_N`
    some pairs did differ, so that reads ``within-noise``. An exact
    no-signal always reads ``within-noise``.
    """
    if verdict.method != "exact" and verdict.n < MIN_PERMUTATION_N:
        return "inconclusive"
    if verdict.verdict != "no-signal":
        return verdict.verdict
    return _no_signal_class(verdict)


def _no_signal_class(verdict: MetricVerdict) -> DisplayClass:
    """The display class a no-signal verdict reads as, by the method that produced it.

    The method union is discriminated exhaustively so a new method fails to
    type-check here until it decides what its no-signal reads as. A band
    verdict measured identical when every pair tied (``usable_n == 0`` — the
    caller has already ruled out a pair count below :data:`MIN_PERMUTATION_N`);
    every other no-signal — band or otherwise — reads within noise.
    """
    match verdict.method:
        case "band":
            return "identical" if verdict.usable_n == 0 else "within-noise"
        case "permutation" | "exact":
            return "within-noise"
        case _ as unreachable:  # pragma: no cover — exhaustive match over Method
            assert_never(unreachable)


_GLYPHS: dict[DisplayClass, str] = {
    "improved": "✓",
    "regressed": "✗",
    "unstable": "≈",
    "identical": "=",
    "within-noise": "~",
    "inconclusive": "?",
}


def get_glyph(shown: DisplayClass) -> str:
    """The glyph a display class is drawn with in the report's rows and legend."""
    return _GLYPHS[shown]


def shown_class(verdict: MetricVerdict | None) -> DisplayClass | None:
    """:func:`display_class`, or ``None`` when there is no verdict to show one for."""
    return None if verdict is None else display_class(verdict)


VERDICT_GLOSSES: dict[DisplayClass, str] = {
    "improved": "improved",
    "regressed": "regressed",
    "unstable": "unstable",
    "identical": "identical",
    "within-noise": "within noise",
    "inconclusive": "inconclusive",
}

QUIET_VERDICTS: frozenset[DisplayClass] = frozenset(
    {"within-noise", "identical", "inconclusive", "unstable"}
)
