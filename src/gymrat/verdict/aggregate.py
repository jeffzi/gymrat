"""Hierarchical aggregation: one geomean per kind, per group, and per gating subset."""

from collections.abc import Mapping
from dataclasses import dataclass

from gymrat.metric_name import parse as parse_metric_name
from gymrat.model import GeomeanResult, MetricVerdict, ResolvedMetricMeta
from gymrat.verdict.geomean import compute_geomean

type MetricEntry = tuple[str, ResolvedMetricMeta]
"""A metric name paired with the metadata resolved for it."""


@dataclass(frozen=True, slots=True)
class GroupAggregate:
    """The geomean over one group of a kind's metrics.

    Attributes:
        group: Path prefix the group's metrics share — the name minus its last
            segment.
        geomean: Geomean over the group's metrics, gating and non-gating alike.
    """

    group: str
    geomean: GeomeanResult


@dataclass(frozen=True, slots=True)
class KindAggregate:
    """One kind's aggregation for a single candidate.

    ``geomean`` covers every metric of the kind and ``gated_geomean`` only the
    gating ones, so a report can show what the whole section did next to what the
    run is judged on. The two coincide when every metric of the kind gates.

    Attributes:
        kind: The metric kind these aggregates summarize.
        geomean: Over every metric of the kind, gating and non-gating alike.
        groups: One entry per group the kind's metric paths name, empty when
            every path has a single segment.
        gated_geomean: Over the kind's gating metrics alone, ``None`` when the
            kind has none.
    """

    kind: str
    geomean: GeomeanResult
    groups: tuple[GroupAggregate, ...]
    gated_geomean: GeomeanResult | None = None


@dataclass(slots=True)
class _KindBucket:
    """A kind's metrics, and the groups their short names sort them into."""

    metrics: list[MetricEntry]
    groups: dict[str, list[MetricEntry]]


def infer_group(name: str) -> str | None:
    """The group a metric belongs to, derived from its name's path segments.

    Parses ``name`` through :func:`gymrat.metric_name.parse` and returns the
    path prefix (all segments but the last, joined with ``/``).  Single-segment
    paths have no group.

    Exposed so a renderer laying out group blocks sorts its rows by the same rule
    the aggregates were computed under — a second rule would put a metric in one
    group and its geomean in another.
    """
    return parse_metric_name(name).group


def _bucket_by_kind(
    metric_meta: Mapping[str, ResolvedMetricMeta],
) -> dict[str, _KindBucket]:
    """Sort every metric into its kind, and within that kind into its group.

    Both levels are dicts keyed by name, so iteration yields kinds and groups in
    the order their first metric introduced them — Python dicts preserve
    insertion order.
    """
    buckets: dict[str, _KindBucket] = {}

    for name, meta in metric_meta.items():
        bucket = buckets.setdefault(meta.kind, _KindBucket(metrics=[], groups={}))

        entry: MetricEntry = (name, meta)
        bucket.metrics.append(entry)

        group = infer_group(name)
        if group is None:
            continue

        bucket.groups.setdefault(group, []).append(entry)

    return buckets


def compute_kind_aggregates(
    verdicts: Mapping[str, MetricVerdict],
    metric_meta: Mapping[str, ResolvedMetricMeta],
) -> list[KindAggregate]:
    """Aggregate one candidate's verdicts into a geomean per kind, group, and gating subset.

    Kinds, groups, and the metrics inside them keep the order ``metric_meta``
    lists them in, which is the order the run first reported each metric — so a
    report drawn from these aggregates reads in the same order as the metric
    table.

    Grouping is decided per kind: a group exists only where a name's path has
    more than one segment, and a kind of single-segment names has no groups at
    all rather than one group per metric. Inside a kind that does have groups, a
    single-segment name joins none of them, yet still counts toward the kind.

    Every geomean here is a plain ``compute_geomean`` call over a chosen subset,
    so the unstable, undefined-ratio and infinite-rho exclusions apply throughout,
    each reported against the subset it was excluded from.

    Args:
        verdicts: The candidate's verdicts, keyed by metric name.
        metric_meta: Metadata for every metric of the run, in first-appearance
            order.

    Returns:
        One :class:`KindAggregate` per kind, in first-appearance order.
    """
    aggregates: list[KindAggregate] = []

    for kind, bucket in _bucket_by_kind(metric_meta).items():
        gating = [entry for entry in bucket.metrics if entry[1].gating]
        aggregates.append(
            KindAggregate(
                kind=kind,
                geomean=compute_geomean(verdicts, dict(bucket.metrics)),
                groups=tuple(
                    GroupAggregate(group=group, geomean=compute_geomean(verdicts, dict(members)))
                    for group, members in bucket.groups.items()
                ),
                gated_geomean=compute_geomean(verdicts, dict(gating)) if gating else None,
            ),
        )

    return aggregates
