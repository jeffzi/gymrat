"""Dataclasses and constants for the sampling subsystem."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from gymrat.adapters.types import WarnSink
from gymrat.config.types import KindEntry, MetricEntry
from gymrat.progress_events import ProgressCallback, default_clock
from gymrat.targets import Target


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """One target a comparison or measurement names, before resolution.

    Attributes:
        label: An explicit display label, or ``None`` to derive one from the
            resolved target (a ref's name or a directory's basename).
        target: A git ref (resolved to a throwaway worktree) or a filesystem
            directory path (benched in place).
    """

    label: str | None
    target: str


@dataclass(frozen=True, slots=True)
class TargetContext:
    """A target paired with where and how it is run.

    Attributes:
        target: The thing being benchmarked.
        dir: The directory the command runs in.
        label: The target's display label.
        position: Which side of a comparison the target occupies, or ``None``
            when the run is not a two-sided comparison.
    """

    target: Target
    dir: str
    label: str
    position: Literal["old", "new"] | None = None


@dataclass(frozen=True, slots=True)
class TargetSamples:
    """Every metric record collected for one target, with its context.

    Attributes:
        ctx: The context the samples were collected under.
        samples: One metric record per successful bench run, in round order.
    """

    ctx: TargetContext
    samples: list[dict[str, float]]


@dataclass(frozen=True, slots=True)
class SamplingOptions:
    """Inputs governing a sampling run.

    Attributes:
        bench: The command run once per target per round.
        prepare: A command run once per target before sampling, or ``None``.
        samples: The number of rounds.
        timeout_seconds: Per-command wall-clock budget, in seconds.
        on_progress: Invoked with each progress event, or ``None`` for silence.
        warn: Where an adapter sends complaints about output it could not read,
            or ``None`` to use the adapter's own default.
        clock: A source of monotonic millisecond timestamps for event stamping.
            Defaults to :func:`~gymrat.progress_events.default_clock`.
    """

    bench: str
    prepare: str | None
    samples: int
    timeout_seconds: float
    on_progress: ProgressCallback | None = None
    warn: WarnSink | None = None
    clock: Callable[[], float] = default_clock


@dataclass(frozen=True, slots=True, kw_only=True)
class RunOptions:
    """The run settings a comparison and a measurement both take.

    Beyond the sampling fields :class:`SamplingOptions` reads, this adds the
    three inputs a caller needs to turn raw samples into a report: which adapter
    parses the bench output, and the per-metric and per-kind config overrides
    that settle each metric's metadata.

    Attributes:
        bench: The command run once per target per round.
        prepare: A command run once per target before sampling, or ``None``.
        adapter: Which output format ``bench`` writes, by adapter name.
        samples: The number of rounds.
        timeout_seconds: Per-command wall-clock budget, in seconds.
        config_metrics: Per-metric overrides from config, or ``None``.
        config_kinds: Per-kind overrides from config, or ``None``.
        on_progress: Invoked with each progress event, or ``None`` for silence.
        warn: Where an adapter sends complaints about unreadable output, or
            ``None`` to use the adapter's own default.
    """

    bench: str
    prepare: str | None
    adapter: str
    samples: int
    timeout_seconds: float
    config_metrics: dict[str, MetricEntry] | None
    config_kinds: dict[str, KindEntry] | None
    on_progress: ProgressCallback | None = None
    warn: WarnSink | None = None


@dataclass(frozen=True, slots=True)
class MetricStats:
    """A metric's central value and relative spread.

    Attributes:
        median: The metric's median, or ``None`` when there were no values.
        spread: The half-range as a percentage of the median's magnitude, or
            ``None`` when it is undefined (fewer than two values, a zero median,
            or a non-finite ratio).
    """

    median: float | None
    spread: float | None


_REF_HINT = (
    "the worktree only contains files tracked at this ref; "
    "untracked, gitignored, or not-yet-committed files are absent"
)
_LABEL_WIDTH = 11
_MIN_SPREAD_SAMPLES = 2
