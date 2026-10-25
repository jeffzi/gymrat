"""The contract every bench-harness adapter satisfies.

An adapter turns a bench harness's stdout into gymrat's metric map. Every adapter
satisfies the :class:`Adapter` contract. :class:`MetricDefaults` is what an
adapter knows about a metric from its name alone; every built-in adapter derives
it from one suffix table through :func:`defaults_from_suffixes`.

An adapter sends a complaint about each part of the output it cannot read cleanly
to a :data:`~gymrat.utils.WarnSink`; most such parts (a line, a run, a benchmark,
a task, a duplicate name, a metric name with an empty path segment or an empty
kind) are skipped, so every name an adapter emits follows the grammar of
:mod:`gymrat.metric_name`. It raises :class:`AdapterError`, aborting the whole
parse, when the output yields no usable metric or a metric name with more than one
``#``.
"""

from dataclasses import dataclass
from typing import Final, Protocol

from gymrat.errors import GymratError
from gymrat.model import Direction, MetricUnit
from gymrat.utils import WarnSink, warn_to_stderr


class AdapterError(GymratError):
    """A bench script produced output the adapter could not read.

    The class itself is the signal: the CLI error formatter matches on it to
    prefix the message with the class name, which is what tells the user the
    fault is in their bench script's output rather than in gymrat's git or
    config handling. Adapters raise this and nothing else for unparseable
    output.
    """


@dataclass(frozen=True, slots=True)
class MetricDefaults:
    """What an adapter knows about a metric from its name alone.

    Attributes:
        direction: Whether a lower or higher raw value is the better outcome.
        unit: The metric's physical unit, or ``None`` when the adapter cannot
            tell, in which case the report prints the raw value rather than
            scaling it.
        kind: The label grouping metrics an adapter emits for the same benchmark
            (a harness may emit both a ``time`` and a ``memory`` metric per
            benchmark), or ``None`` when the adapter cannot tell.
        short_name: The benchmark's name with the kind suffix stripped, or
            ``None`` when the adapter cannot tell, leaving the full metric name
            as the only thing the report can show.
        resolution: The smallest step by which the metric's raw value can
            change (``1.0`` for a byte count), or ``None`` when the adapter
            cannot tell.
    """

    direction: Direction
    unit: MetricUnit | None = None
    kind: str | None = None
    short_name: str | None = None
    resolution: float | None = None


class Adapter(Protocol):
    """Turns a benchmark harness's stdout into gymrat's metric map.

    Conformance is structural: any object exposing ``name``, ``parse``, and
    ``defaults`` satisfies the protocol without inheriting from it.

    ``defaults`` is consulted once per metric name during config resolution, and
    only for fields the user's config does not override.
    """

    @property
    def name(self) -> str:
        """The adapter's identifier, as named by the ``adapter`` config key."""
        ...

    def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
        """Parse ``stdout`` into a metric map, routing complaints to ``warn``.

        Args:
            stdout: The bench script's full standard output.
            warn: Where to send complaints about individual unreadable lines.

        Returns:
            One value per metric name.

        Raises:
            AdapterError: When ``stdout`` yields no usable metric, or yields a
                metric name the adapter cannot accept (such as one carrying
                more than one ``#``); either aborts the whole parse.
        """
        ...

    def defaults(self, metric_name: str) -> MetricDefaults:
        """Return name-derived defaults for ``metric_name``."""
        ...


TIME_SUFFIX: Final = "#time"
"""Suffix marking a metric name as a time reading, in nanoseconds.

Adapters append it to a metric prefix to form the full metric name, and
:func:`defaults_from_suffixes` matches on it.
"""

MEMORY_SUFFIX: Final = "#memory"
"""Suffix marking a metric name as a memory reading, in bytes.

Adapters append it to a metric prefix to form the full metric name, and
:func:`defaults_from_suffixes` matches on it.
"""

_METRIC_SUFFIXES: Final[tuple[tuple[str, MetricUnit, str, float | None], ...]] = (
    (TIME_SUFFIX, "ns", "time", None),
    (MEMORY_SUFFIX, "bytes", "memory", 1.0),
)
"""Metric name suffixes and the defaults each one implies.

Each entry is ``(suffix, unit, kind, resolution)``, with ``unit``, ``kind`` and
``resolution`` as described on :class:`MetricDefaults`. A memory metric counts
whole bytes, so its resolution is one byte; a time metric has none, because a
timer's step depends on the machine. :func:`defaults_from_suffixes` walks the
entries in order, and the first matching suffix wins.
"""


def defaults_from_suffixes(metric_name: str) -> MetricDefaults:
    """Derive :class:`MetricDefaults` from a metric name by matching suffixes.

    Walks :data:`_METRIC_SUFFIXES` in order, so the first matching suffix wins.

    Args:
        metric_name: The metric name to match against the suffix table. It
            follows the metric name grammar, so a path precedes the suffix.

    Returns:
        Defaults carrying the unit, kind and resolution of the first matching
        suffix and the name with that suffix stripped, or direction-only
        defaults when no suffix matches.
    """
    for suffix, unit, kind, resolution in _METRIC_SUFFIXES:
        if metric_name.endswith(suffix):
            return MetricDefaults(
                direction="lower",
                unit=unit,
                kind=kind,
                short_name=metric_name.removesuffix(suffix),
                resolution=resolution,
            )
    return MetricDefaults(direction="lower")
