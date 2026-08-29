"""Shared metric-input builders for verdict aggregation tests.

A compact ``MetricSpec`` describes one metric's contribution to a run, and
:func:`build_inputs` turns a list of specs into the ``(verdicts, metric_meta)``
pair the aggregation layer consumes, preserving spec order.

This is test-support code, not a test module: ``test_geomean`` and (later)
``test_aggregate`` import it. It carries no test functions of its own.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from gymrat.model import (
    BandVerdict,
    Direction,
    Effect,
    ExactVerdict,
    MetricVerdict,
    ResolvedMetricMeta,
    Verdict,
)


def exact_verdict(delta: float) -> ExactVerdict:
    """An exact verdict no exclusion rule drops.

    ``rho`` becomes ``1 + delta / 100`` when lower is better, so the sign of
    ``delta`` alone decides the verdict: negative improves, positive regresses,
    zero is no signal.
    """
    verdict: Verdict
    if delta < 0:
        verdict = "improved"
    elif delta > 0:
        verdict = "regressed"
    else:
        verdict = "no-signal"
    return ExactVerdict(
        method="exact",
        verdict=verdict,
        delta=Effect(value=delta, unit="percent"),
        n=1,
    )


@dataclass(frozen=True)
class MetricSpec:
    """One metric's contribution to a run: what it is, and how it moved."""

    name: str
    """Full metric name — the key the verdict and exclusion lists report it under."""

    short_name: str | None = None
    """Defaults to ``name``, matching an adapter that reports no grouping."""

    direction: Direction = "lower"
    """Defaults to ``"lower"``, so a directionless spec list is lower-is-better."""

    gating: bool = True
    """Defaults to ``True``, matching every case that isn't explicitly non-gating."""

    kind: str = "time"
    """Defaults to ``"time"``, so a kindless spec list describes a single-kind run."""

    delta: float | None = None
    """Percentage delta behind an exact verdict. Ignored when ``verdict`` is given."""

    verdict: MetricVerdict | None = None
    """A full verdict object, for band and unstable cases :func:`exact_verdict` cannot express."""

    no_verdict: bool = False
    """True to add ``name`` to ``metric_meta`` without a matching verdict — the no-verdict case."""


def _resolve_verdict(spec: MetricSpec) -> MetricVerdict | None:
    """The verdict a spec contributes, or ``None`` for the no-verdict case."""
    if spec.no_verdict:
        return None
    return spec.verdict if spec.verdict is not None else exact_verdict(spec.delta or 0.0)


def build_inputs(
    specs: Sequence[MetricSpec],
) -> tuple[dict[str, MetricVerdict], dict[str, ResolvedMetricMeta]]:
    """Verdicts and metadata keyed by metric name, in the order the specs are listed."""
    verdicts: dict[str, MetricVerdict] = {}
    metric_meta: dict[str, ResolvedMetricMeta] = {}

    for spec in specs:
        short_name = spec.short_name if spec.short_name is not None else spec.name
        verdict = _resolve_verdict(spec)
        if verdict is not None:
            verdicts[spec.name] = verdict

        metric_meta[spec.name] = ResolvedMetricMeta(
            direction=spec.direction,
            gating=spec.gating,
            exact=verdict is not None and verdict.method == "exact",
            unit=None,
            kind=spec.kind,
            short_name=short_name,
        )

    return verdicts, metric_meta


def _noop_warn(_message: str) -> None:
    """Swallow divergence warnings so these cases stay silent on stderr."""


def create_samples(n: int, value: float) -> list[dict[str, float]]:
    return [{"metric": value} for _ in range(n)]


def unstable_band_verdict() -> BandVerdict:
    """A band verdict too noisy to judge, regardless of its ratio."""
    return BandVerdict(
        method="band",
        verdict="unstable",
        usable_n=4,
        noise_pct=250.0,
        noise_abs=25.0,
        delta=Effect(value=-50.0, unit="percent"),
        n=4,
    )


# Re-exported for callers building explicit band/permutation verdicts.
__all__ = [
    "MetricSpec",
    "_noop_warn",
    "build_inputs",
    "create_samples",
    "exact_verdict",
    "unstable_band_verdict",
]
