"""Geometric-mean aggregation over per-metric verdicts.

Averages exactly the metrics named in ``metric_meta`` into a single geometric
mean, in log space over each metric's normalized ratio, and propagates their
noise bands in quadrature. Metrics that cannot contribute a usable ratio — never
judged, judged unstable, or yielding a degenerate ratio — are reported as
exclusions instead, keeping the aggregate's scope accountable for every metric
the caller asked it to cover.
"""

from collections.abc import Mapping

from gymrat_py.model import Exclusion, GeomeanResult, MetricMeta, MetricVerdict
from gymrat_py.stats import combine_geomean, normalize_ratio


def compute_geomean(
    verdicts: Mapping[str, MetricVerdict],
    metric_meta: Mapping[str, MetricMeta],
) -> GeomeanResult:
    """Aggregate the metrics named in ``metric_meta`` into a geometric mean.

    Which metrics belong in the geomean is the caller's decision: every metric
    ``metric_meta`` names is in scope, gating or not, and ``verdicts`` may carry
    others that are ignored. Each in-scope metric is either included as a
    ``(rho, noise_pct)`` pair or reported as an exclusion, so ``n`` plus the
    number of exclusions always equals the number of metrics in scope.

    A metric is excluded, in this order, when it has no verdict
    (``"no-verdict"``), when its verdict is unstable (``"unstable"``, decided
    before the ratio so an unstable verdict with a NaN delta is still reported
    unstable), or when its ratio is degenerate (``"undefined-ratio"`` for a NaN
    delta, ``"infinite-rho"`` for a non-positive or non-finite ratio).

    Args:
        verdicts: Per-metric verdicts keyed by metric name.
        metric_meta: The metrics to average, keyed by name, in the order they
            should be considered.

    Returns:
        A :class:`GeomeanResult` carrying the combined value, the count of
        included metrics, the propagated noise band, and every exclusion.
    """
    entries: list[tuple[float, float]] = []
    exclusions: list[Exclusion] = []

    for name, meta in metric_meta.items():
        verdict = verdicts.get(name)
        if verdict is None:
            exclusions.append(Exclusion(metric=name, reason="no-verdict"))
            continue
        if verdict.verdict == "unstable":
            exclusions.append(Exclusion(metric=name, reason="unstable"))
            continue

        outcome = normalize_ratio(verdict.delta.value, meta.direction)
        if outcome.reason is not None:
            exclusions.append(Exclusion(metric=name, reason=outcome.reason))
            continue

        rho = outcome.rho
        assert rho is not None  # noqa: S101 -- reason is None, so normalize_ratio guarantees a rho
        noise_pct = 0.0 if verdict.method == "exact" else verdict.noise_pct
        entries.append((rho, noise_pct))

    combination = combine_geomean(entries)
    return GeomeanResult(
        value=combination.value,
        n=combination.n,
        band=combination.band,
        excluded=tuple(exclusions),
    )
