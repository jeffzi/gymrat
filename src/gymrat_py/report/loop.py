"""How the loop states an iteration: its run header and the verdict it closes on.

The table between an iteration's header and its verdict is the comparison report
:mod:`gymrat_py.report.text` already renders: the loop only replaces the header
above it and appends the verdict below it, so a reader who knows ``gymrat
compare`` reads an iteration without relearning anything. The lines follow the
same conventions — the verdict block's colors, the dimmed ``·`` — so an
iteration reads as the comparison it is built on.

These fragments return rich-markup strings rather than raw ANSI, so a renderer
decides color once through :func:`gymrat_py.report.style.render_lines`. Dynamic
text inside a styled span is escaped so a metric named ``[i]`` renders literally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rich.markup import escape

from gymrat_py.model import Effect
from gymrat_py.report.format import format_delta, is_improvement
from gymrat_py.report.style import format_hint_label
from gymrat_py.report.table import markup
from gymrat_py.report.text import paired_samples

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat_py.report.types import MetricComparison, MetricComparisons

# ---------------------------------------------------------------------------
# Outcome and primary-figure types
# ---------------------------------------------------------------------------

#: What an iteration amounted to.
#:
#: Narrower than a metric verdict: an iteration has no ``unstable`` of its own,
#: because a primary figure too noisy to read is one that reported nothing, which
#: is what ``no-signal`` already says.
LoopOutcome = Literal["improved", "regressed", "no-signal"]


@dataclass(frozen=True, slots=True)
class GeomeanPrimary:
    """The run's geomean read as the one figure an iteration is judged on.

    A geomean carries no name: it is the direction-normalized aggregate over the
    run's gating metrics, so a negative value improves whichever way its metrics
    point.

    Attributes:
        delta_pct: How far the geomean moved, or ``None`` where the ratio had no
            value — a baseline median of zero leaves nothing to normalize
            against, a figure with no direction to read rather than one that
            stood still, so nothing may coerce it to ``0``.
    """

    delta_pct: float | None
    kind: Literal["geomean"] = "geomean"


@dataclass(frozen=True, slots=True)
class MetricPrimary:
    """One named metric read as the figure an iteration is judged on.

    A named metric keeps its name, because the direction it improves in is its
    own metadata's to say.

    Attributes:
        name: The metric's key in the run's comparisons.
        delta_pct: How far the metric moved, or ``None`` where the ratio had no
            value (see :class:`GeomeanPrimary`).
    """

    name: str
    delta_pct: float | None
    kind: Literal["metric"] = "metric"


#: The one figure an iteration is read on, and how far it moved.
LoopPrimary = GeomeanPrimary | MetricPrimary

#: What a confirmation rerun had to say about one metric it re-measured.
#:
#: ``absent`` is not a weaker ``disagreed``: a rerun that never reported the
#: metric disproved nothing, so the regression the first run called still stands.
#: Only ``disagreed`` — the rerun measured the metric and did not call it
#: regressed — takes a regression back.
RerunAnswer = Literal["confirmed", "disagreed", "absent"]


@dataclass(frozen=True, slots=True)
class RerunConfirmation:
    """One metric a confirmation rerun was asked about, and what it answered.

    Attributes:
        metric: The metric the rerun re-measured.
        answer: What the rerun settled about it.
    """

    metric: str
    answer: RerunAnswer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The candidate an iteration measures: the experiment, judged against the baseline.
EXPERIMENT_INDEX = 0

#: How many leading characters of a commit SHA a report abbreviates it to.
SHORT_SHA_LENGTH = 7

#: What the loop's header says it compared, fixed for every iteration.
_COMPARED = "experiment vs baseline"

#: What an iteration that met the configured target says, and what it asks for.
_TARGET_REACHED = "target reached — keep it"

#: The word each outcome is announced with.
_OUTCOME_WORDS: dict[LoopOutcome, str] = {
    "improved": "IMPROVED",
    "regressed": "REGRESSED",
    "no-signal": "NO-SIGNAL",
}

#: How each outcome's word is painted: emboldened whatever it says, and colored
#: only where there is a direction to report. A no-signal iteration is neither
#: good nor bad, so it wears no color rather than a hedged one.
_OUTCOME_STYLES: dict[LoopOutcome, str] = {
    "improved": "bold green",
    "regressed": "bold red",
    "no-signal": "bold",
}

#: What each rerun answer reads as, and the style it is painted with. An absent
#: answer wears the yellow the table paints an unstable metric with, because it
#: is the same kind of news: a reading nobody could take.
_RERUN_PHRASES: dict[RerunAnswer, tuple[str, str]] = {
    "confirmed": ("regression confirmed on rerun", "red"),
    "disagreed": ("regression not confirmed on rerun", "dim"),
    "absent": ("not measured on rerun", "yellow"),
}

#: The ``·`` the loop's lines separate their parts with, dimmed in color.
_SEPARATOR = "·"


def _separator() -> str:
    """The dimmed ``·`` separator, spaces left outside the dim so each dot stands alone."""
    return f" {markup(_SEPARATOR, 'dim')} "


def _format_primary_delta(delta_pct: float | None) -> str:
    """A primary figure's move as a leading-space signed percentage, or nothing.

    Blank is what the table already shows for a ratio with no value, so the two
    agree: a reader is shown no percentage rather than one they could read a
    direction into.
    """
    if delta_pct is None:
        return ""
    return f" {format_delta(Effect(value=delta_pct, unit='percent'))}"


# ---------------------------------------------------------------------------
# Header and verdict block
# ---------------------------------------------------------------------------


def format_loop_header(seq: int, samples: int) -> str:
    """The run header of one iteration: which iteration, what it compared, and how many rounds.

    Passed to the report renderer as its header override, so the table below it
    opens on the loop's own terms rather than on ``gymrat compare``'s. The
    adapter goes unnamed: a session fixes it once, so repeating it every
    iteration spends a header on news the reader already had.

    Args:
        seq: The iteration's number.
        samples: How many paired samples stand behind the comparison.

    Returns:
        The header as a single rich-markup line.
    """
    parts = [markup(f"iteration {seq}", "bold"), _COMPARED, paired_samples(samples)]
    return _separator().join(parts)


def _format_rerun_line(rerun: RerunConfirmation) -> str:
    """What the rerun settled about one metric, painted the way the table paints that answer."""
    text, style = _RERUN_PHRASES[rerun.answer]
    return f"{escape(rerun.metric)}: {markup(text, style)}"


def format_verdict_block(
    *,
    outcome: LoopOutcome,
    primary: LoopPrimary,
    next_step: str,
    reruns: Sequence[RerunConfirmation] = (),
    target_reached: bool = False,
) -> list[str]:
    """The lines closing an iteration: the rerun, the primary figure, the verdict, the next step.

    The rerun lines open the block rather than close it because they qualify the
    table above — a metric the table shows at rest that the first run had called
    a regression is only readable once the rerun is named.

    A reached target is stated last, directly above the next step, because it is
    an instruction rather than a reading: the loop only stops once the iteration
    that reached the target is kept.

    Args:
        outcome: What the iteration amounted to.
        primary: The one figure the iteration is read on.
        next_step: The step that follows this iteration.
        reruns: What a confirmation rerun settled about each metric it re-measured.
        target_reached: Whether this iteration met the configured target.

    Returns:
        The block as rich-markup lines, so the caller appends them to the report
        it already holds as lines.
    """
    verdict = markup(_OUTCOME_WORDS[outcome], _OUTCOME_STYLES[outcome])
    # The delta renders blank when the ratio had no value, so the parts are
    # joined rather than interpolated: a blank between a space and the separator
    # would read as a gap.
    verdict_line = _separator().join(
        [f"primary:{_format_primary_delta(primary.delta_pct)}", f"verdict: {verdict}"]
    )
    lines = [_format_rerun_line(rerun) for rerun in reruns]
    lines.append(verdict_line)
    if target_reached:
        lines.append(markup(_TARGET_REACHED, "green"))
    lines.append(f"{format_hint_label()} {escape(next_step)}")
    return lines


# ---------------------------------------------------------------------------
# Outcome derivation
# ---------------------------------------------------------------------------


def _experiment_regressed(metric: MetricComparison) -> bool:
    """Whether the experiment's side of ``metric`` came back regressed."""
    if len(metric.candidates) <= EXPERIMENT_INDEX:
        return False
    verdict = metric.candidates[EXPERIMENT_INDEX].verdict
    return verdict is not None and verdict.verdict == "regressed"


def _has_gating_regression(metrics: MetricComparisons) -> bool:
    """Whether any metric the run is gated on came back regressed for the experiment."""
    return any(metric.meta.gating and _experiment_regressed(metric) for metric in metrics.values())


def _primary_improved(metrics: MetricComparisons, primary: LoopPrimary) -> bool:
    """Whether the primary figure moved the way its direction calls an improvement.

    A figure whose ratio had no value moved in no direction at all, so it
    improves nothing. The geomean case routes through :func:`is_improvement`
    wrapping a percent effect; a named metric combines that verdict with its own
    direction — a ``higher`` metric improves on the opposite sign, and never on a
    delta of exactly zero.
    """
    if primary.delta_pct is None:
        return False
    effect = Effect(value=primary.delta_pct, unit="percent")
    if isinstance(primary, GeomeanPrimary):
        return is_improvement(effect)
    metric = metrics.get(primary.name)
    if metric is None:
        return False
    if metric.meta.direction == "higher":
        return not is_improvement(effect) and primary.delta_pct != 0
    return is_improvement(effect)


def derive_outcome(metrics: MetricComparisons, primary: LoopPrimary) -> LoopOutcome:
    """What an iteration amounted to, read off its metrics and its primary figure.

    A gating regression settles it whatever the primary did: the run is judged on
    every metric it gates, so a headline that improved while a gate broke is still
    an iteration to fix rather than one to keep.

    Everything that is neither a gating regression nor an improvement in the
    primary's own direction reads ``no-signal`` — including a primary the run
    never measured, which reports nothing rather than reporting zero.

    Args:
        metrics: The run's per-metric comparisons.
        primary: The one figure the iteration is read on.

    Returns:
        The iteration's outcome.
    """
    if _has_gating_regression(metrics):
        return "regressed"
    return "improved" if _primary_improved(metrics, primary) else "no-signal"
