"""How the loop states an iteration: its run header and the verdict it closes on.

The table between an iteration's header and its verdict is the comparison report
:mod:`gymrat.report.text` already renders: the loop only replaces the header
above it and appends the verdict below it, so a reader who knows ``gymrat
compare`` reads an iteration without relearning anything. The lines follow the
same conventions — the verdict block's colors, the dimmed ``·`` — so an
iteration reads as the comparison it is built on.

The same fragments assemble ``gymrat status``: a header naming the session, one
line per record in file order (a baseline, an iteration and its settle state, or
a refused keep standing alone), the totals, and — when the session was
finalized — the line it closes on.

These fragments return rich-markup strings rather than raw ANSI, so a renderer
decides color once through :func:`gymrat.report.style.render_lines`. Dynamic
text inside a styled span is escaped so a metric named ``[i]`` renders literally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never

from rich.markup import escape

from gymrat.git import SHORT_SHA_LENGTH
from gymrat.metric_name import format_inline, parse
from gymrat.model import Effect
from gymrat.plural import pluralize
from gymrat.report.display import get_glyph
from gymrat.report.format import format_delta, format_value, is_improvement
from gymrat.report.style import VARIANT_NAME_STYLE, format_hint, markup
from gymrat.report.text import paired_samples
from gymrat.stats.descriptive import compute_median

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.config import StopConfig
    from gymrat.report.display import DisplayClass
    from gymrat.report.types import MetricComparison, MetricComparisons
    from gymrat.session import (
        BaselineRecord,
        BaselineRef,
        FinalizeRecord,
        SessionRecord,
    )
    from gymrat.session.records import SampleRound
    from gymrat.session.schema import KeepReason

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

#: What the loop's header says it compared, fixed for every iteration. The two
#: targets wear the style the table heads its columns with, so the header names
#: them the way the columns below it do.
_COMPARED = (
    f"{markup('experiment', VARIANT_NAME_STYLE)} vs {markup('baseline', VARIANT_NAME_STYLE)}"
)

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
    parts = [markup(f"iteration {seq}", "bold"), _COMPARED, escape(paired_samples(samples))]
    return _separator().join(parts)


def _format_rerun_line(rerun: RerunConfirmation) -> str:
    """What the rerun settled about one metric, painted the way the table paints that answer."""
    text, style = _RERUN_PHRASES[rerun.answer]
    name = format_inline(parse(rerun.metric))
    return f"{name}: {markup(text, style)}"


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
    if target_reached and outcome != "regressed":
        lines.append(markup(_TARGET_REACHED, "green"))
    lines.append(format_hint(next_step))
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


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SettleKept:
    """A kept iteration, carrying the commit it landed as once one is known.

    Attributes:
        commit: The commit the keep made, or ``None`` while it is not yet known —
            a keep can be recorded before its squash commit exists.
    """

    commit: str | None = None
    kind: Literal["kept"] = "kept"


@dataclass(frozen=True, slots=True)
class SettleDiscarded:
    """An iteration thrown away."""

    kind: Literal["discarded"] = "discarded"


@dataclass(frozen=True, slots=True)
class SettleUnsettled:
    """An iteration nobody has settled yet."""

    kind: Literal["unsettled"] = "unsettled"


@dataclass(frozen=True, slots=True)
class SettleKeepBlocked:
    """An iteration whose keep was refused, carrying why when the log knows.

    A blocked keep is its own state rather than a variant of unsettled: the
    iteration is still waiting to be settled, but the log knows why the last
    attempt did not land, and that reason is what the agent acts on.

    Attributes:
        reason: Why the keep was refused, or ``None`` when the log names none.
    """

    reason: KeepReason | None = None
    kind: Literal["keep-blocked"] = "keep-blocked"


#: What a single settling record says became of the iteration it settles.
SettleState = SettleKept | SettleDiscarded | SettleUnsettled | SettleKeepBlocked


@dataclass(frozen=True, slots=True)
class StatusIteration:
    """One iteration as a status line states it.

    Attributes:
        seq: The iteration's number.
        delta_pct: How far the iteration's primary figure moved, or ``None``
            where the ratio had no value.
        outcome: What the iteration amounted to.
        settle: What became of it.
    """

    seq: int
    delta_pct: float | None
    outcome: LoopOutcome
    settle: SettleState


@dataclass(frozen=True, slots=True)
class StatusSummary:
    """What a session adds up to: how its iterations settled, and how near its stop it is.

    Attributes:
        iteration_count: How many iterations the session measured.
        keep_count: How many of them it kept.
        discard_count: How many of them it threw away.
        target_reached: Whether a kept iteration reached the configured target.
        stop: The configured stop conditions, or ``None`` when the session runs
            until the agent stops it.
    """

    iteration_count: int
    keep_count: int
    discard_count: int
    target_reached: bool
    stop: StopConfig | None = None


#: The glyph each outcome wears, borrowed from the comparison table's vocabulary.
#:
#: A no-signal iteration takes the table's within-noise glyph: both say the same
#: thing — the figure moved by nothing the run can stand behind.
OUTCOME_GLYPHS: dict[LoopOutcome, DisplayClass] = {
    "improved": "improved",
    "regressed": "regressed",
    "no-signal": "within-noise",
}


def format_baseline_ref(baseline: BaselineRef) -> str:
    """A session's baseline in the ``ref@sha`` form (sha shortened) every summary states it with."""
    return f"{baseline.ref}@{baseline.sha[:SHORT_SHA_LENGTH]}"


def format_status_header(session: SessionRecord) -> list[str]:
    """The session a status report opens on: what it is, what it forked from, where it works.

    Both worktree paths are named for the reason ``gymrat start`` names them: the
    agent edits in one of them and must never touch the other.

    Args:
        session: The session header the log opens on.

    Returns:
        The header as rich-markup lines.
    """
    baseline = format_baseline_ref(session.baseline)
    return [
        _separator().join(
            [
                markup(f"session {session.session_id}", "bold"),
                f"baseline {escape(baseline)}",
                f"adapter {escape(session.config.adapter)}",
            ]
        ),
        f"branch {escape(session.branch)}",
        f"experiment worktree {escape(session.worktrees.experiment)}",
        f"baseline worktree {escape(session.worktrees.baseline)}",
    ]


def format_status_settle(settle: SettleState) -> str:
    """How an iteration was settled, in the words ``status`` reports it with.

    A settling record that settled no iteration — a keep refused for want of a
    measurement — stands on a line of its own, and this is all that line says.
    """
    match settle:
        case SettleKept():
            return "kept" if settle.commit is None else f"kept {settle.commit[:SHORT_SHA_LENGTH]}"
        case SettleDiscarded():
            return "discarded"
        case SettleUnsettled():
            return "unsettled"
        case SettleKeepBlocked():
            return "keep-blocked" if settle.reason is None else f"keep-blocked ({settle.reason})"
        case _ as unreachable:
            assert_never(unreachable)


def format_status_iteration(iteration: StatusIteration) -> str:
    """One iteration of the session's history: which one, what it did, what became of it.

    The glyph carries the outcome's own color, so a session's course is legible
    down the left of the report before a word of it is read.
    """
    glyph = markup(get_glyph(OUTCOME_GLYPHS[iteration.outcome]), _OUTCOME_STYLES[iteration.outcome])
    return _separator().join(
        [
            f"iteration {iteration.seq}",
            f"{glyph}{_format_primary_delta(iteration.delta_pct)}",
            format_status_settle(iteration.settle),
        ]
    )


def _metric_medians(samples: Sequence[SampleRound]) -> list[tuple[str, float]]:
    """The median each metric measured across ``samples``, in the order rounds first named them.

    A metric is averaged only over the rounds that reported it, so a metric a
    late round dropped keeps the median of the rounds that carried it.
    """
    readings: dict[str, list[float]] = {}
    for round_ in samples:
        for name, value in round_.items():
            readings.setdefault(name, []).append(value)
    return [(name, compute_median(values)) for name, values in readings.items()]


def format_status_baseline(record: BaselineRecord) -> str:
    """A recorded baseline measurement: what was measured, and the median each metric came to.

    The log stores every round the measurement took, so the medians are computed
    here rather than stored — a later statistics change re-reads the same records
    instead of invalidating them.
    """
    parts = [f"baseline {escape(record.label)}"]
    parts.extend(
        f"{escape(name)} {format_value(median)}" for name, median in _metric_medians(record.samples)
    )
    return _separator().join(parts)


def _format_stop_state(summary: StatusSummary) -> str | None:
    """Where the session stands against its configured stop, or nothing when none is configured."""
    stop = summary.stop
    if stop is None:
        return None
    parts: list[str] = []
    if stop.max_iterations is not None:
        parts.append(f"{summary.iteration_count} of {stop.max_iterations} iterations")
    if stop.target_value is not None:
        parts.append("target reached" if summary.target_reached else "target pending")
    if not parts:
        return None
    return f"{markup('stop:', 'dim')} {_separator().join(parts)}"


def format_status_footer(summary: StatusSummary) -> list[str]:
    """The lines closing a status report: how the iterations settled, and how near the stop is.

    The stop line is left out when nothing is configured rather than reported as
    unlimited: a loop the agent stops when it likes has no state to state.
    """
    totals = _separator().join(
        [
            pluralize(summary.iteration_count, "iteration"),
            f"{summary.keep_count} kept",
            f"{summary.discard_count} discarded",
        ]
    )
    stop = _format_stop_state(summary)
    return [totals] if stop is None else [totals, stop]


def format_status_finalized(finalized: FinalizeRecord) -> str:
    """The line a finalized session's report ends on: where its work ended up.

    It sits under the totals rather than in the header because closing the
    session is the last thing that happened to it, and the branch and commit it
    names are what the reader goes to next — everything above them is history.
    """
    return _separator().join(
        [
            markup("finalized", "bold"),
            f"branch {escape(finalized.branch)}",
            f"commit {finalized.commit[:SHORT_SHA_LENGTH]}",
        ]
    )
