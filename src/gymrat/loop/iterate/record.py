"""Build the iteration record from a judged run's outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat.loop.iterate.bench import _Judged, recorded_delta
from gymrat.model import ExactVerdict, MetricVerdict, PermutationVerdict, ResolvedMetricMeta

if TYPE_CHECKING:
    from gymrat.loop.iterate.confirm import Confirmation
from gymrat.report.loop import LoopOutcome, LoopPrimary, MetricPrimary
from gymrat.session import (
    Confirm,
    IterationPrimary,
    IterationRecord,
)
from gymrat.session import MetricVerdict as RecordMetricVerdict
from gymrat.session.clock import now_iso


@dataclass(frozen=True, slots=True)
class IterationJudgment:
    """The iteration's judgment: what happened, what drove it, and whether a target was met."""

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

    A key whose value is absent is left out — ``p`` off a non-permutation verdict,
    ``noise_pct`` off an exact one — so a record handed to a caller matches the one
    read back off the log.
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
    judged: _Judged,
    seq: int,
    judgment: IterationJudgment,
) -> IterationRecord:
    """Assemble the iteration record from the judged run."""
    confirmation = judgment.confirmation
    confirm: Confirm | None = None
    if confirmation is not None:
        absent = tuple(name for name in confirmation.filtered if name in confirmation.absent)
        confirm = Confirm(
            ran=True,
            filtered=tuple(confirmation.filtered),
            samples=confirmation.samples,
            absent=absent or None,
        )
    primary = judgment.primary
    return IterationRecord(
        type="iteration",
        seq=seq,
        at=now_iso(),
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
    )
