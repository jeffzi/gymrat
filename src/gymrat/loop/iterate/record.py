"""Build the iteration record from a judged run's outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat.loop.iterate.bench import Judged, recorded_delta
from gymrat.model import ExactVerdict, MetricVerdict, PermutationVerdict, ResolvedMetricMeta
from gymrat.report.loop import LoopOutcome, LoopPrimary, MetricPrimary
from gymrat.session import (
    Confirm,
    IterationPrimary,
    IterationRecord,
)
from gymrat.session import MetricVerdict as RecordMetricVerdict
from gymrat.session.clock import now_ns

if TYPE_CHECKING:
    from gymrat.loop.iterate.confirm import Confirmation


@dataclass(frozen=True, slots=True)
class IterationJudgment:
    """The outcome, primary figure, and optional confirmation rerun to report.

    Attributes:
        outcome: The loop outcome derived from the comparison.
        primary: The primary figure that drives the verdict display.
        confirmation: What the confirmation rerun found, or ``None`` when no
            rerun was needed.
        reached_target: Whether the run met its declared target.
    """

    outcome: LoopOutcome
    primary: LoopPrimary
    confirmation: Confirmation | None
    reached_target: bool


def recorded_verdicts(
    verdicts: dict[str, MetricVerdict],
    metric_meta: dict[str, ResolvedMetricMeta],
    confirmation: Confirmation | None,
) -> dict[str, RecordMetricVerdict]:
    """The per-metric verdicts as the log keeps them, flattened out of the method shapes.

    Args:
        verdicts: The per-metric verdicts to flatten, by name.
        metric_meta: The resolved metadata for each measured metric, by name.
        confirmation: What the confirmation rerun found, or ``None`` when no
            rerun was needed.

    Returns:
        The per-metric verdicts keyed by metric name, each carrying only the
        fields the log serializes: ``p`` is omitted for non-permutation
        verdicts, ``noise_pct`` for exact ones, so a record handed to a
        caller matches the one read back off the log.
    """
    recorded: dict[str, RecordMetricVerdict] = {}
    for name, verdict in verdicts.items():
        meta = metric_meta.get(name)
        recorded[name] = RecordMetricVerdict(
            delta_pct=recorded_delta(verdict.delta.value),
            verdict=verdict.verdict,
            method=verdict.method,
            gating=meta.gating if meta is not None else True,
            confirmed=name in confirmation.confirmed if confirmation is not None else False,
            p=verdict.p if isinstance(verdict, PermutationVerdict) else None,
            noise_pct=None if isinstance(verdict, ExactVerdict) else verdict.noise_pct,
        )
    return recorded


def build_iteration_record(
    judged: Judged,
    seq: int,
    judgment: IterationJudgment,
    *,
    duration_ms: int | None = None,
    measured_tree: str | None = None,
) -> IterationRecord:
    """Assemble the session-log record for one measured iteration.

    An empty ``absent`` tuple collapses to ``None`` so the log never
    carries ``[]``.

    Args:
        judged: The bench run outputs and comparison result.
        seq: The 1-based iteration sequence number.
        judgment: The outcome, primary, and optional confirmation.
        duration_ms: Wall-clock milliseconds the iteration took, or ``None``
            when timing is unavailable.
        measured_tree: The experiment worktree fingerprint at measurement time,
            or ``None`` when fingerprinting failed.

    Returns:
        The iteration record ready to append to the session log.
    """
    confirmation = judgment.confirmation
    confirm: Confirm | None = None
    if confirmation is not None:
        absent = tuple(name for name in confirmation.filtered if name in confirmation.absent)
        confirm = Confirm(
            ran=True,
            filtered=confirmation.filtered,
            samples=confirmation.samples,
            absent=absent or None,
        )
    primary = judgment.primary
    return IterationRecord(
        type="iteration",
        seq=seq,
        at=now_ns(),
        samples=judged.samples,
        metrics=recorded_verdicts(judged.run.verdicts, judged.run.metric_meta, confirmation),
        primary=IterationPrimary(
            kind=primary.kind,
            delta_pct=primary.delta_pct,
            name=primary.name if isinstance(primary, MetricPrimary) else None,
        ),
        outcome=judgment.outcome,
        target_reached=judgment.reached_target,
        confirm=confirm,
        duration_ms=duration_ms,
        measured_tree=measured_tree,
    )
