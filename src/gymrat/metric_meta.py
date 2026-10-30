"""Metric-metadata resolution for collected samples.

For each metric name the samples carry, the adapter's defaults give the unit,
direction and resolution, and the kind and short name when the adapter reports
them (otherwise ``other`` and the metric name itself); gating defaults to on and
exact to off. A per-kind config entry may set gating and resolution, and a
per-metric entry overrides direction, gating, exact and resolution.

Resolving metadata from samples also warns about a time metric with no
resolution whose readings repeat the way a quantized timer's do.
"""

import itertools
import statistics
from collections.abc import Sequence
from typing import Final

from gymrat.adapters.contract import Adapter
from gymrat.adapters.registry import get_adapter
from gymrat.config import GEOMEAN_PRIMARY, KindEntry, MetricEntry, ResolvedConfig
from gymrat.errors import GymratError
from gymrat.model import GEOMEAN_DIRECTION, Direction, ResolvedMetricMeta
from gymrat.report.format import format_value
from gymrat.utils import WarnSink, own_values, warn_to_stderr

DEFAULT_METRIC_KIND: Final[str] = "other"
"""The kind a metric falls under when its adapter reports none."""

_DEFAULT_GATING = True
"""Whether a metric gates when neither a ``metrics`` entry nor a ``kinds`` entry names it."""


def _first_set[T](default: T, *overrides: T | None) -> T:
    return next((value for value in overrides if value is not None), default)


def resolve_metric_meta(
    name: str,
    entry: MetricEntry | None,
    adapter: Adapter,
    config_kinds: dict[str, KindEntry] | None,
) -> ResolvedMetricMeta:
    """Resolve one metric's metadata from its config entry, its kind and the adapter.

    The adapter's defaults for the metric are the base. The ``metrics`` entry
    overrides direction, gating, exact and resolution wherever it sets them.
    Gating and resolution the entry leaves unset come from the ``kinds`` entry
    of the metric's kind; past that, gating is on and resolution is the
    adapter's. A ``kinds`` entry for another kind is ignored.

    Every reader of a metric's direction resolves it here, so a verdict and a
    display of the same metric cannot disagree about which way is better.

    Args:
        name: The metric's full name, as the adapter reports it.
        entry: The metric's ``metrics`` entry from config, or ``None`` when
            config does not name it.
        adapter: The adapter whose defaults seed the metadata.
        config_kinds: Per-kind overrides from config, keyed by kind name, or
            ``None``.

    Returns:
        The metric's resolved metadata.
    """
    defaults = adapter.defaults(name)
    kind = defaults.kind if defaults.kind is not None else DEFAULT_METRIC_KIND
    direction = (
        entry.direction if entry is not None and entry.direction is not None else defaults.direction
    )

    metric_entry = entry if entry is not None else MetricEntry()
    kind_entry = (config_kinds or {}).get(kind, KindEntry())

    gating = _first_set(_DEFAULT_GATING, metric_entry.gating, kind_entry.gating)
    resolution = _first_set(defaults.resolution, metric_entry.resolution, kind_entry.resolution)

    exact = entry.exact if entry is not None and entry.exact is not None else False
    short_name = defaults.short_name if defaults.short_name is not None else name

    return ResolvedMetricMeta(
        direction=direction,
        gating=gating,
        exact=exact,
        unit=defaults.unit,
        resolution=resolution,
        kind=kind,
        short_name=short_name,
    )


def primary_direction(config: ResolvedConfig) -> Direction:
    """Whether a lower or a higher value of the configured primary is the better outcome.

    Args:
        config: The resolved config, naming the primary, its metric entries,
            kinds and adapter.

    Returns:
        :data:`~gymrat.model.GEOMEAN_DIRECTION` for the geomean primary,
        otherwise the direction the named metric resolves to.
    """
    if config.primary == GEOMEAN_PRIMARY:
        return GEOMEAN_DIRECTION
    entry = config.metrics.get(config.primary) if config.metrics is not None else None
    return resolve_metric_meta(
        config.primary, entry, get_adapter(config.adapter), config.kinds
    ).direction


def resolve_metric_meta_from_samples(
    sample_sets: Sequence[list[dict[str, float]]],
    config_metrics: dict[str, MetricEntry] | None,
    adapter: Adapter,
    config_kinds: dict[str, KindEntry] | None,
    *,
    warn: WarnSink = warn_to_stderr,
) -> dict[str, ResolvedMetricMeta]:
    """Collect every metric name across the sample sets and resolve its metadata.

    The union of names is taken in first-appearance order across the flattened
    samples, so the resolved metadata — and every report drawn from it — reads in
    the order the run first reported each metric. Each metric resolves as
    :func:`resolve_metric_meta` describes.

    A statistical ``ns`` metric with no resolution gets one warning when its
    readings look quantized: some target reads one value at least twice, and
    either every reading is that value or the smallest gap between distinct
    values, pooled across targets, is above 0.5% of the median. Readings within
    a millionth of the median count as one value. The warning never changes the
    resolved metadata.

    Args:
        sample_sets: One list of per-round metric records per target.
        config_metrics: Per-metric overrides from config, keyed by metric name,
            or ``None``.
        adapter: The adapter whose defaults seed each metric's metadata.
        config_kinds: Per-kind overrides from config, keyed by kind name, or
            ``None``.
        warn: Where the warning about quantized readings goes.

    Returns:
        The resolved metadata for each metric, keyed by metric name.

    Raises:
        GymratError: No sample set reported any metric. Adapters reject empty
            output themselves, so this guards the otherwise-unreachable case.
    """
    names = dict.fromkeys(name for samples in sample_sets for sample in samples for name in sample)
    if not names:
        message = "No metrics found in benchmark output"
        raise GymratError(message)

    resolved = {
        name: resolve_metric_meta(
            name,
            config_metrics.get(name) if config_metrics is not None else None,
            adapter,
            config_kinds,
        )
        for name in names
    }
    for name, meta in resolved.items():
        if meta.unit != "ns" or meta.exact or meta.resolution is not None:
            continue
        message = _quantized_readings_warning(
            name, [own_values(samples, name) for samples in sample_sets]
        )
        if message is not None:
            warn(message)
    return resolved


_SAME_VALUE_FRACTION = 1e-6
"""Readings closer than this fraction of the median count as one value."""

_QUANTIZED_GAP_FRACTION = 0.005
"""A smallest gap above this fraction of the median, with a repeat, looks like a timer's step."""


def _quantized_readings_warning(name: str, per_target: list[list[float]]) -> str | None:
    pooled = sorted(value for values in per_target for value in values)
    median = statistics.median(pooled)
    tolerance = abs(median) * _SAME_VALUE_FRACTION
    repeats = any(
        later - earlier <= tolerance
        for values in per_target
        for earlier, later in itertools.pairwise(sorted(values))
    )
    if not repeats:
        return None

    distinct: list[float] = []
    for value in pooled:
        if not distinct or value - distinct[-1] > tolerance:
            distinct.append(value)
    gap = min((later - earlier for earlier, later in itertools.pairwise(distinct)), default=None)
    if gap is not None and gap <= abs(median) * _QUANTIZED_GAP_FRACTION:
        return None

    step = "" if gap is None else f"; its step is at most {format_value(gap, 'ns')}"
    return (
        f"{name}: readings repeat as a quantized timer's would{step}. "
        "Set `resolution` to the timer's tick divided by the calls each reading averages, "
        "or time more iterations per reading"
    )
