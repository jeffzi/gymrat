"""Shared metric-input builders for verdict aggregation tests.

A compact ``MetricSpec`` describes one metric's contribution to a run, and
:func:`build_inputs` turns a list of specs into the ``(verdicts, metric_meta)``
pair the aggregation layer consumes, preserving spec order.

This is test-support code, not a test module: the verdict tests import it. It
carries no test functions of its own.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from gymrat.model import (
    BandVerdict,
    Direction,
    MetricVerdict,
    ResolvedMetricMeta,
)
from tests.report._verdicts import band_verdict, exact_verdict, metric_meta


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One metric's contribution to a run: what it is, and how it moved."""

    name: str
    """Full metric name — the key the verdict and exclusion lists report it under."""

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
    if spec.verdict is not None:
        return spec.verdict
    return exact_verdict(delta=spec.delta or 0.0, n=1)


def build_inputs(
    specs: Sequence[MetricSpec],
) -> tuple[dict[str, MetricVerdict], dict[str, ResolvedMetricMeta]]:
    """Verdicts and metadata keyed by metric name, in the order the specs are listed."""
    verdicts: dict[str, MetricVerdict] = {}
    meta_by_name: dict[str, ResolvedMetricMeta] = {}

    for spec in specs:
        verdict = _resolve_verdict(spec)
        if verdict is not None:
            verdicts[spec.name] = verdict

        meta_by_name[spec.name] = metric_meta(
            spec.name,
            direction=spec.direction,
            gating=spec.gating,
            exact=verdict is not None and verdict.method == "exact",
            kind=spec.kind,
        )

    return verdicts, meta_by_name


def unstable_band_verdict() -> BandVerdict:
    """A band verdict too noisy to judge, regardless of its ratio."""
    return band_verdict(
        verdict="unstable", usable_n=4, noise_pct=250.0, noise_abs=25.0, delta=-50.0, n=4
    )
