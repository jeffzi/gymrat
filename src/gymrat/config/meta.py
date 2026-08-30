"""Metric metadata resolution from adapter defaults and config overrides."""

from collections.abc import Sequence

from gymrat.adapters.defaults import DEFAULT_GATING, DEFAULT_METRIC_KIND
from gymrat.adapters.types import Adapter
from gymrat.config.types import KindEntry, MetricEntry
from gymrat.model import ResolvedMetricMeta


def _resolve_one_metric(
    name: str,
    entry: MetricEntry | None,
    adapter: Adapter,
    config_kinds: dict[str, KindEntry] | None,
) -> ResolvedMetricMeta:
    defaults = adapter.defaults(name)
    kind = defaults.kind if defaults.kind is not None else DEFAULT_METRIC_KIND
    direction = (
        entry.direction if entry is not None and entry.direction is not None else defaults.direction
    )

    gating = DEFAULT_GATING
    if entry is not None and entry.gating is not None:
        gating = entry.gating
    elif config_kinds is not None:
        kind_entry = config_kinds.get(kind)
        if kind_entry is not None and kind_entry.gating is not None:
            gating = kind_entry.gating

    exact = entry.exact if entry is not None and entry.exact is not None else False
    short_name = defaults.short_name if defaults.short_name is not None else name

    return ResolvedMetricMeta(
        direction=direction,
        gating=gating,
        exact=exact,
        unit=defaults.unit,
        kind=kind,
        short_name=short_name,
    )


def resolve_metric_meta(
    metric_names: Sequence[str],
    config_metrics: dict[str, MetricEntry] | None,
    adapter: Adapter,
    config_kinds: dict[str, KindEntry] | None = None,
) -> dict[str, ResolvedMetricMeta]:
    """Resolve each metric's display metadata from adapter defaults and config overrides.

    For every name in ``metric_names`` (preserving input order), the adapter's
    per-metric defaults are the base; a matching ``config_metrics`` entry overrides
    direction, gating, and exact, and a ``config_kinds`` entry for the resolved kind
    supplies gating when the metric entry does not. A per-metric gating override wins
    over its kind's gating.
    """
    return {
        name: _resolve_one_metric(
            name,
            config_metrics.get(name) if config_metrics is not None else None,
            adapter,
            config_kinds,
        )
        for name in metric_names
    }
