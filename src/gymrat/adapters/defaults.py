"""Seam 7: name-based metric defaults plus the fallback kind and gating default.

Holds the suffix-to-metadata table and the fallback kind and gating default that
v0.6 config resolution reads here instead of re-declaring, keeping the
name-based metric defaults and the resolver's fallbacks from drifting apart.
"""

from typing import Final

from gymrat.adapters.types import MetricDefaults
from gymrat.model.metrics import MetricUnit

_METRIC_SUFFIXES: Final[tuple[tuple[str, MetricUnit, str], ...]] = (
    ("#time", "ns", "time"),
    ("#heap", "bytes", "memory"),
)
"""Suffix→(unit, kind) table walked by :func:`defaults_from_suffixes`; first match wins."""

DEFAULT_METRIC_KIND: Final[str] = "other"
"""The kind a metric falls under when its adapter reports none.

Consumed by v0.6 config resolution as the fallback kind for a metric whose
adapter defaults carry no kind.
"""

DEFAULT_GATING: Final[bool] = True
"""Whether a metric gates when nothing names it.

Consumed by v0.6 config resolution: the gating value applied when neither a
``metrics`` entry nor a ``kinds`` entry names the metric.
"""


def defaults_from_suffixes(metric_name: str) -> MetricDefaults:
    """Derive :class:`MetricDefaults` from a metric name by matching suffixes.

    Walks :data:`_METRIC_SUFFIXES` in order and returns the defaults for the
    first suffix ``metric_name`` ends with. When the prefix before the suffix is
    empty (the name equals the suffix, e.g. ``#time``), ``short_name`` is the
    full metric name so the report renders a visible label. A name matching no
    suffix yields defaults carrying only ``direction``.
    """
    for suffix, unit, kind in _METRIC_SUFFIXES:
        if metric_name.endswith(suffix):
            prefix = metric_name[: -len(suffix)]
            return MetricDefaults(
                direction="lower",
                unit=unit,
                kind=kind,
                short_name=metric_name if prefix == "" else prefix,
            )
    return MetricDefaults(direction="lower")
