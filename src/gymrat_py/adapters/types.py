"""Adapter contract for bench-harness parsers.

Defines the interface a bench harness parser implements, the value type
describing what an adapter knows about a metric, and the warning sink and error
it uses when output cannot be read.
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from gymrat_py.errors import GymratError
from gymrat_py.model.metrics import Direction, MetricUnit


class AdapterError(GymratError):
    """A bench script produced output the adapter could not read.

    The class itself is the signal: the CLI error formatter matches on it to
    prefix the message with the class name, which is what tells the user the
    fault is in their bench script's output rather than in gymrat's git or
    config handling. Adapters raise this and nothing else for unparseable
    output.
    """


type WarnSink = Callable[[str], None]
"""Where an adapter sends a complaint about output it could not read.

The caller owns the destination so a warning can be interleaved with whatever
else is on the terminal — the CLI's progress line, for one — instead of landing
on stderr wherever the cursor happens to be.
"""


def warn_to_stderr(message: str) -> None:
    """Default :data:`WarnSink` for adapters called without an explicit one.

    Writes ``message`` followed by a newline to stderr.
    """
    sys.stderr.write(f"{message}\n")


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
    """

    direction: Direction
    unit: MetricUnit | None = None
    kind: str | None = None
    short_name: str | None = None


@runtime_checkable
class Adapter(Protocol):
    """Turns a benchmark harness's stdout into gymrat's metric map.

    Conformance is structural: any object exposing ``name``, ``parse``, and
    ``defaults`` satisfies the protocol without inheriting from it.

    ``parse`` receives a bench script's full stdout and raises
    :class:`AdapterError` when it yields no usable metric — returning an empty
    map instead would let a silently broken bench script read as a run with
    nothing to compare. Complaints about individual unreadable lines go to
    ``warn``, which defaults to stderr so a direct caller need not supply one.

    ``defaults`` is consulted once per metric name during config resolution, and
    only for fields the user's config does not override.
    """

    name: str

    def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
        """Parse ``stdout`` into a metric map, routing complaints to ``warn``."""
        ...

    def defaults(self, metric_name: str) -> MetricDefaults:
        """Return name-derived defaults for ``metric_name``."""
        ...
