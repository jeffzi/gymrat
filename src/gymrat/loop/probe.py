"""Bench the session's experiment worktree against the recorded baseline.

A probe is the loop's read-only measurement: it runs the bench once in the
experiment worktree and pairs every measured median with the reference median
from the newest baseline record. Nothing is appended to the session log, no hook
runs, and the progress sidecar is left alone, so a probe never changes what the
session's record says happened.

The sample count is the probe's own — short by default, since a probe answers
"did that edit help?" rather than standing behind a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymrat.errors import GymratError
from gymrat.loop.baseline import measure_baseline
from gymrat.loop.iterate.confirm import scoped_bench
from gymrat.report.loop import baseline_medians
from gymrat.sampling import RunOptions, TargetSpec
from gymrat.session.store import latest_baseline, require_open_session

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.config import ResolvedConfig
    from gymrat.model import ResolvedMetricMeta
    from gymrat.progress_events import ProgressCallback
    from gymrat.warn import WarnSink

#: Rounds a probe runs when the caller names no count of its own.
PROBE_DEFAULT_SAMPLES = 6

#: The label a probe's target carries, naming the worktree it benched.
_EXPERIMENT_LABEL = "experiment"


@dataclass(frozen=True, slots=True)
class ProbeOptions:
    """What one probe measures, and where its progress and warnings go.

    Attributes:
        names: Metric names to narrow the bench to, in the order the caller gave
            them. Empty runs the whole bench.
        samples: Rounds to run; ``None`` falls back to
            :data:`PROBE_DEFAULT_SAMPLES`. The configured ``samples`` is never
            used — it sizes an iteration's verdict, which a probe does not reach.
        on_progress: Sink for the run's progress events, or ``None`` to drop them.
        warn: Sink for warnings the adapter raises, or ``None`` to drop them.
    """

    names: Sequence[str] = ()
    samples: int | None = None
    on_progress: ProgressCallback | None = None
    warn: WarnSink | None = None


@dataclass(frozen=True, slots=True)
class ProbeMetric:
    """One metric the probe measured, beside the baseline it is read against.

    Attributes:
        name: The metric's name as the bench reported it.
        median: The median the probe measured, or ``None`` when no round
            reported the metric.
        spread: Half the measured range as a percentage of the median, or
            ``None`` when there was no run-to-run jitter to report.
        reference_median: The median the newest baseline record came to for this
            metric, or ``None`` when that baseline never reported it.
        delta_pct: The signed percentage change from ``reference_median`` to
            ``median``, or ``None`` when either is missing or the reference is
            zero. Positive means the probe measured a larger number, whatever
            ``meta.direction`` makes of that.
        meta: The metric's resolved metadata, carrying the direction and unit a
            renderer needs to style the delta.
    """

    name: str
    median: float | None
    spread: float | None
    reference_median: float | None
    delta_pct: float | None
    meta: ResolvedMetricMeta


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Everything one probe produced, ready for a renderer.

    Attributes:
        label: The display label of the worktree that was benched.
        samples: The rounds the run actually took, after the fallback to
            :data:`PROBE_DEFAULT_SAMPLES`.
        adapter: The bench-output adapter the run parsed with.
        metrics: One entry per metric the run reported, in report order.
        scoped: Whether the bench was narrowed to ``names``.
        names: The metric names the bench was narrowed to, in the order given.
    """

    label: str
    samples: int
    adapter: str
    metrics: tuple[ProbeMetric, ...]
    scoped: bool
    names: tuple[str, ...]


def _delta_pct(median: float | None, reference: float | None) -> float | None:
    """The signed percentage change from ``reference`` to ``median``, when there is one."""
    if median is None or reference is None or reference == 0:
        return None
    return (median - reference) / reference * 100


def _probe_bench(config: ResolvedConfig, names: tuple[str, ...]) -> str:
    """The bench command a probe of ``names`` runs.

    Args:
        config: The run configuration, carrying the bench command and the
            optional ``filter`` template.
        names: The metric names to narrow the bench to; empty narrows nothing.

    Returns:
        ``config.bench`` when there is nothing to narrow, otherwise the
        ``filter`` template with the shell-quoted names substituted in.

    Raises:
        GymratError: When ``names`` is non-empty and no ``filter`` is configured
            — silently benching everything would answer a different question
            than the one asked.
    """
    if not names:
        return config.bench
    if config.filter is None:
        message = "filter is not configured — set filter in gymrat.toml to scope a probe"
        raise GymratError(message, reason="no-filter")
    return scoped_bench(config, names)


async def probe_session(
    root: str,
    config: ResolvedConfig,
    options: ProbeOptions,
) -> ProbeResult:
    """Bench the open session's experiment worktree against its recorded baseline.

    Args:
        root: Repository root whose session log is read.
        config: The resolved run configuration: bench, filter, and every run
            setting except the sample count.
        options: The metric scope, sample count, and progress/warning sinks.

    Returns:
        The measured metrics in report order, each paired with the newest
        baseline's median for that metric.

    Raises:
        GymratError: When no session is open or the session was finalized, when
            metric names were given without a configured ``filter``, or when the
            session has no baseline record to compare against. The refusals are
            raised before any bench runs.
    """
    required = require_open_session(root, "probing")
    names = tuple(options.names)
    bench = _probe_bench(config, names)

    baseline = latest_baseline(required.records)
    if baseline is None:
        message = f"Session {required.session.session_id} has no recorded baseline"
        raise GymratError(
            message,
            hint="Run gymrat measure --record to record one before probing.",
            reason="no-baseline",
        )

    samples = PROBE_DEFAULT_SAMPLES if options.samples is None else options.samples
    run_options = RunOptions(
        bench=bench,
        prepare=config.prepare,
        adapter=config.adapter,
        samples=samples,
        timeout_seconds=config.timeout_seconds,
        config_metrics=config.metrics,
        config_kinds=config.kinds,
        on_progress=options.on_progress,
        warn=options.warn,
    )
    target = TargetSpec(label=_EXPERIMENT_LABEL, target=required.session.worktrees.experiment)
    result, _ = await measure_baseline(target, run_options)

    references = baseline_medians(baseline)
    metrics = []
    for name, metric in result.metrics.items():
        reference = references.get(name)
        metrics.append(
            ProbeMetric(
                name=name,
                median=metric.median,
                spread=metric.spread,
                reference_median=reference,
                delta_pct=_delta_pct(metric.median, reference),
                meta=metric.meta,
            )
        )
    return ProbeResult(
        label=_EXPERIMENT_LABEL,
        samples=samples,
        adapter=config.adapter,
        metrics=tuple(metrics),
        scoped=bool(names),
        names=names,
    )
