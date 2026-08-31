"""Footer generation: method lines and samples hint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

from gymrat.report.display import MIN_PERMUTATION_N
from gymrat.report.format import (
    format_pair_count,
)
from gymrat.report.style import markup

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gymrat.model import MetricVerdict
    from gymrat.report.types import MetricComparisons


def _samples_hint(command: str) -> str:
    """The hint asking for more samples, naming the command that would carry them.

    The command is stated whole and backtick-marked so
    :func:`gymrat.report.style.format_hint` sets it apart from the prose: a
    reader copies the line rather than assembling the invocation themselves.
    """
    return (
        f"re-run with `gymrat {command} --samples {MIN_PERMUTATION_N}` "
        f"or more for statistical verdicts"
    )


_DROPPED_ROUNDS_HINT = (
    "some rounds were dropped — not all samples produced paired measurements for every metric"
)

_BAND_METHOD = "noise band ±(half-range × K)"


@dataclass(slots=True)
class _FooterData:
    """The pair counts the footer sorts by the cause that forced each fallback.

    ``permutation`` carries the pair counts of every permutation verdict.
    ``shortage`` and ``ties`` split the band-method verdicts by cause: too few
    total pairs, or too many of them tied away.
    """

    permutation: list[int]
    shortage: list[int]
    ties: list[int]


def _classify_verdict(verdict: MetricVerdict, data: _FooterData) -> None:
    """Sort one verdict's pair count into the footer cause it belongs to.

    The method union is discriminated exhaustively: exact verdicts contribute
    nothing to the footer by decision, an explicit arm rather than a fall-through
    a new method could slip past unnoticed.
    """
    match verdict.method:
        case "permutation":
            data.permutation.append(verdict.n)
        case "band":
            if verdict.n < MIN_PERMUTATION_N:
                data.shortage.append(verdict.n)
            else:
                data.ties.append(verdict.usable_n)
        case "exact":
            return
        case _ as unreachable:  # pragma: no cover — exhaustive match over VerdictMethod
            assert_never(unreachable)


def _collect_footer_data(metrics: MetricComparisons) -> _FooterData:
    """Sort every verdict's pair count into the cause it belongs to, in one pass."""
    data = _FooterData(permutation=[], shortage=[], ties=[])
    for metric in metrics.values():
        for candidate in metric.candidates:
            if candidate.verdict is not None:
                _classify_verdict(candidate.verdict, data)
    return data


def _method_lines(data: _FooterData) -> list[str]:
    """The verbose method lines naming how each verdict was decided, each dimmed.

    A band fallback gets one line per cause: the highest total pair count for a
    shortage — even the best-off metric fell this far short — and the lowest
    usable pair count for ties, so each line stays true of every metric behind
    it.
    """
    lines: list[str] = []
    if data.permutation:
        desc = (
            f"verdicts: sign-flip permutation test on pairs "
            f"({format_pair_count(min(data.permutation))} ≥ {MIN_PERMUTATION_N}) "
            f"· ~ = no signal at α=0.05"
        )
        lines.append(markup(desc, "dim"))
    if data.shortage:
        desc = (
            f"{_BAND_METHOD} — {format_pair_count(max(data.shortage))} "
            f"below permutation floor ({MIN_PERMUTATION_N} pairs)"
        )
        lines.append(markup(desc, "dim"))
    if data.ties:
        desc = (
            f"{_BAND_METHOD} — ties left {format_pair_count(min(data.ties))} "
            f"usable pairs ({MIN_PERMUTATION_N} needed)"
        )
        lines.append(markup(desc, "dim"))
    return lines


def _shortage_hint(shortage: Sequence[int], samples: int | None, command: str) -> str | None:
    """The hint for metrics that fell to the band because their paired count was short.

    When the run's own sample count is below the floor, more samples are the
    fix. When it had enough samples but rounds were dropped during pairing,
    suggesting more samples is misleading.
    """
    if not shortage:
        return None
    if samples is not None and samples >= MIN_PERMUTATION_N:
        return _DROPPED_ROUNDS_HINT
    return _samples_hint(command)


def footer_lines(
    metrics: MetricComparisons,
    *,
    verbose: bool,
    format_hint: Callable[[str], str],
    command: str,
    samples: int | None = None,
) -> list[str]:
    """The footer: how each verdict was decided when verbose, and the samples hint.

    When ``samples`` is provided the hint distinguishes insufficient samples from
    dropped rounds. Renderers differ only in how they format the hint line, which
    ``format_hint`` owns.

    Args:
        metrics: Every metric of the run, keyed by name.
        verbose: Whether to include the method lines naming each verdict's basis.
        format_hint: Turns a bare hint into the renderer's own hint line.
        command: The subcommand the report was produced by, so a hint suggesting
            a re-run names the whole invocation.
        samples: The run's sample count, to distinguish shortage from dropped
            rounds. Left ``None``, a shortage always suggests more samples.

    Returns:
        The footer lines, method lines (when verbose) first, then the hint.
    """
    data = _collect_footer_data(metrics)
    hint = _shortage_hint(data.shortage, samples, command)
    lines = _method_lines(data) if verbose else []
    if hint is not None:
        lines.append(format_hint(hint))
    return lines
