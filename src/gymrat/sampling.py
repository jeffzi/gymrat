"""Sequential sample collection for a benchmark comparison.

Runs a bench command a fixed number of times against each target, parsing every
run's output into a metric record. Runs are strictly sequential and interleaved
by round: every target is sampled once before any target is sampled again, so a
transient slowdown on the machine spreads across both sides rather than skewing
one. An optional prepare command runs once per target before sampling begins.

A single policy site turns a failed command (non-zero exit or timeout) into a
:class:`~gymrat.errors.CommandError`, so a failure anywhere stops the schedule
with the same formatted diagnosis.

The module also resolves each target to the directory and display label it runs
under, and wraps a phase that may claim git worktrees so they are swept on every
exit path.

Besides collection, it holds the sampling dataclasses, the per-metric summary
statistics, and metric-meta resolution from collected samples. For each metric
name the samples carry, the adapter's defaults give the unit and direction, and
the kind and short name when the adapter reports them (otherwise ``other`` and
the metric name itself); gating defaults to on and exact to off. A per-kind
config entry may set gating, and a per-metric entry overrides direction, gating
and exact.
"""

import asyncio
import math
import statistics
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from gymrat.adapters import DEFAULT_METRIC_KIND, Adapter
from gymrat.clock import monotonic_ms
from gymrat.config.types import KindEntry, MetricEntry, ResolvedConfig
from gymrat.errors import CommandError, GymratError, hint_of
from gymrat.eta import MS_PER_SECOND
from gymrat.exec import (
    ExecOptions,
    ExecResult,
    ExecTimeoutError,
    exec,  # noqa: A004 -- names the subprocess executor `exec`
    kill_live_process_groups,
)
from gymrat.model import ResolvedMetricMeta
from gymrat.progress_events import (
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressCallback,
    emit_progress,
)
from gymrat.report.text.render import format_cleanup_failures
from gymrat.signals import install_termination_cleanup
from gymrat.stats import compute_half_range
from gymrat.targets import (
    CleanupResult,
    RefTarget,
    Target,
    WorktreeInfo,
    cleanup_worktrees,
    materialize_worktree,
    plan_worktree,
)
from gymrat.warn import WarnSink, warn_to_stderr

# ---------------------------------------------------------------------------
# sampling types
# ---------------------------------------------------------------------------


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

    @property
    def display_label(self) -> str:
        """The label to show before the target is resolved: the explicit one, else the target."""
        return self.label or self.target


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
        warn: Where an adapter sends complaints about output it could not read.
        clock: A source of monotonic millisecond timestamps for event stamping.
            Defaults to :func:`~gymrat.clock.monotonic_ms`.
    """

    bench: str
    prepare: str | None
    samples: int
    timeout_seconds: float
    on_progress: ProgressCallback | None = None
    warn: WarnSink = warn_to_stderr
    clock: Callable[[], float] = monotonic_ms


@dataclass(frozen=True, slots=True, kw_only=True)
class RunOptions:
    """The run settings a comparison and a measurement both take.

    Beyond the settings the collector reads, this adds the three inputs a caller
    needs to turn raw samples into a report: which adapter parses the bench
    output, and the per-metric and per-kind config overrides that settle each
    metric's metadata.

    Attributes:
        sampling: The bench and prepare commands, round count, timeout, and
            sinks the collector runs under.
        adapter: Which output format ``bench`` writes, by adapter name.
        config_metrics: Per-metric overrides from config, or ``None``.
        config_kinds: Per-kind overrides from config, or ``None``.
    """

    sampling: SamplingOptions
    adapter: str
    config_metrics: dict[str, MetricEntry] | None
    config_kinds: dict[str, KindEntry] | None

    @classmethod
    def from_config(
        cls,
        config: ResolvedConfig,
        *,
        samples: int | None = None,
        bench: str | None = None,
        on_progress: ProgressCallback | None = None,
        warn: WarnSink = warn_to_stderr,
    ) -> Self:
        """Copy the run settings out of a resolved configuration.

        Args:
            config: The resolved configuration to read.
            samples: The number of rounds, or ``None`` for the configured count.
            bench: The command to run, or ``None`` for the configured one.
            on_progress: Invoked with each progress event, or ``None`` for
                silence.
            warn: Where an adapter sends complaints about unreadable output.

        Returns:
            The run settings, with the collector's clock left at its default.
        """
        return cls(
            sampling=SamplingOptions(
                bench=config.bench if bench is None else bench,
                prepare=config.prepare,
                samples=config.samples if samples is None else samples,
                timeout_seconds=config.timeout_seconds,
                on_progress=on_progress,
                warn=warn,
            ),
            adapter=config.adapter,
            config_metrics=config.metrics,
            config_kinds=config.kinds,
        )


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


# ---------------------------------------------------------------------------
# metric metadata
# ---------------------------------------------------------------------------

_DEFAULT_GATING = True
"""Whether a metric gates when neither a ``metrics`` entry nor a ``kinds`` entry names it."""


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

    gating = _DEFAULT_GATING
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


# ---------------------------------------------------------------------------
# sample summaries
# ---------------------------------------------------------------------------


_MIN_SPREAD_SAMPLES = 2


def compute_metric_stats(values: Sequence[float]) -> MetricStats:
    """Summarize a metric's samples as a median and relative spread.

    Args:
        values: The metric's sampled values.

    Returns:
        The median and its half-range as a percentage of ``abs(median)``. The
        spread is absent when there are fewer than two values, the median is
        zero, or the ratio is non-finite.
    """
    if not values:
        return MetricStats(median=None, spread=None)

    median = statistics.median(values)
    if len(values) < _MIN_SPREAD_SAMPLES or median == 0:
        return MetricStats(median=median, spread=None)

    ratio = compute_half_range(values) / abs(median) * 100
    if not math.isfinite(ratio):
        return MetricStats(median=median, spread=None)
    return MetricStats(median=median, spread=ratio)


def own_values(samples: Sequence[dict[str, float]], name: str) -> list[float]:
    """Collect the values a side reported for ``name``, skipping rounds without it.

    Args:
        samples: One metric record per round.
        name: The metric to extract.

    Returns:
        The reported values for ``name``, in round order.
    """
    return [record[name] for record in samples if name in record]


def resolve_metric_meta_from_samples(
    sample_sets: Sequence[list[dict[str, float]]],
    config_metrics: dict[str, MetricEntry] | None,
    adapter: Adapter,
    config_kinds: dict[str, KindEntry] | None = None,
) -> dict[str, ResolvedMetricMeta]:
    """Collect every metric name across the sample sets and resolve its metadata.

    The union of names is taken in first-appearance order across the flattened
    samples, so the resolved metadata — and every report drawn from it — reads in
    the order the run first reported each metric.

    The adapter's per-metric defaults are the base; a matching ``config_metrics``
    entry overrides direction, gating, and exact, and a ``config_kinds`` entry
    for the resolved kind supplies gating when the metric entry does not. A
    per-metric gating override wins over its kind's gating.

    Args:
        sample_sets: One list of per-round metric records per target.
        config_metrics: Per-metric overrides from config, keyed by metric name,
            or ``None``.
        adapter: The adapter whose defaults seed each metric's metadata.
        config_kinds: Per-kind gating overrides from config, keyed by kind name,
            or ``None``.

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

    return {
        name: _resolve_one_metric(
            name,
            config_metrics.get(name) if config_metrics is not None else None,
            adapter,
            config_kinds,
        )
        for name in names
    }


# ---------------------------------------------------------------------------
# sample collection
# ---------------------------------------------------------------------------


_REF_HINT = (
    "the worktree only contains files tracked at this ref; "
    "untracked, gitignored, or not-yet-committed files are absent"
)
_LABEL_WIDTH = 11


async def collect_samples(
    adapter: Adapter,
    targets: Sequence[TargetContext],
    options: SamplingOptions,
    abort: asyncio.Event,
) -> list[TargetSamples]:
    """Run the prepare and bench schedule and collect each target's samples.

    Prepare (when set) runs once per target in order before any bench run. Then
    for each round, bench runs once per target in order, so round ``n+1`` never
    starts before round ``n`` has run every target. Each successful bench run
    contributes one parsed metric record to its target.

    Args:
        adapter: Parses a bench run's stdout into a metric record.
        targets: The targets to sample, in the order they are run and returned.
        options: The bench and prepare commands, round count, timeout, and hooks.
        abort: An event whose being set kills the in-flight command; passed
            through to the command runner.

    Returns:
        One :class:`TargetSamples` per target, in target order.

    Raises:
        CommandError: A prepare or bench command timed out or exited non-zero.
    """
    timeout_ms = int(options.timeout_seconds * MS_PER_SECOND)
    collected: list[list[dict[str, float]]] = [[] for _ in targets]

    if options.prepare is not None:
        for ctx in targets:
            emit_progress(
                options.on_progress, PrepareStarted(label=ctx.label, at_ms=options.clock())
            )
            await _run_command("prepare", None, options.prepare, ctx, timeout_ms, abort)
            emit_progress(
                options.on_progress, PrepareFinished(label=ctx.label, at_ms=options.clock())
            )

    for round_number in range(1, options.samples + 1):
        for ctx, records in zip(targets, collected, strict=True):
            emit_progress(
                options.on_progress,
                PassStarted(
                    round=round_number,
                    total_rounds=options.samples,
                    target_count=len(targets),
                    label=ctx.label,
                    at_ms=options.clock(),
                ),
            )
            stdout = await _run_command(
                "bench", round_number, options.bench, ctx, timeout_ms, abort
            )
            emit_progress(
                options.on_progress,
                PassFinished(
                    round=round_number,
                    total_rounds=options.samples,
                    target_count=len(targets),
                    label=ctx.label,
                    at_ms=options.clock(),
                ),
            )
            records.append(adapter.parse(stdout, options.warn))

    return [
        TargetSamples(ctx=ctx, samples=samples)
        for ctx, samples in zip(targets, collected, strict=True)
    ]


async def _run_command(  # noqa: PLR0913, PLR0917 -- one parameter per command-execution axis
    phase: str,
    sample_index: int | None,
    command: str,
    ctx: TargetContext,
    timeout_ms: int,
    abort: asyncio.Event,
) -> str:
    """Run one command and return its stdout, or raise on failure."""
    result = await exec(command, ExecOptions(cwd=ctx.dir, timeout_ms=timeout_ms, abort=abort))
    if isinstance(result, ExecTimeoutError) or result.exit_code != 0:
        raise to_command_error(phase, sample_index, command, ctx, result)
    return result.stdout


# ---------------------------------------------------------------------------
# Command failure diagnosis
# ---------------------------------------------------------------------------


def to_command_error(
    phase: str,
    sample_index: int | None,
    command: str,
    ctx: TargetContext,
    result: ExecResult | ExecTimeoutError,
) -> CommandError:
    """Map a command failure to a target-specific :class:`CommandError`.

    A ref target contributes ``ref`` and ``worktree`` location lines plus the
    hint that the worktree only holds tracked files; a plain directory
    contributes a single ``dir`` line and no hint.

    Args:
        phase: Name of the phase the command ran in, e.g. ``"prepare"``.
        sample_index: The sample number to mention in the header, or ``None``
            when the failure is not tied to a specific sample.
        command: The command line that failed.
        ctx: The target context supplying the header label, position, and
            location lines.
        result: The execution outcome, either a failed ``ExecResult`` or a
            timeout.

    Returns:
        The formatted command error with target-specific location context.
    """
    target = ctx.target
    if isinstance(target, RefTarget):
        location = [_field("ref", target.ref), _field("worktree", ctx.dir)]
        hint = _REF_HINT
    else:
        location = [_field("dir", ctx.dir)]
        hint = None

    position = f"{ctx.position}, " if ctx.position is not None else ""
    sample = f", sample {sample_index}" if sample_index is not None else ""
    if isinstance(result, ExecTimeoutError):
        outcome_label = "timed out"
        outcome = _field("timeout", f"{result.timeout_ms}ms")
    else:
        outcome_label = "failed"
        outcome = _field("exit code", result.exit_code)
    header = f'{phase} command {outcome_label} ({position}"{ctx.label}"{sample})'

    lines = [header, *location, _field("command", command), outcome]
    lines.extend(
        _captured_output(result.stdout, result.stdout_bytes, result.stderr, result.stderr_bytes)
    )

    return CommandError("\n".join(lines), hint=hint)


def _field(label: str, value: object) -> str:
    """Format an indented, column-aligned ``label: value`` detail line."""
    return f"  {(label + ':').ljust(_LABEL_WIDTH)}{value}"


def _captured_output(stdout: str, stdout_bytes: int, stderr: str, stderr_bytes: int) -> list[str]:
    """Render the captured output of a failed command.

    A lone non-empty stream is emitted bare unless its captured text was
    truncated, in which case it — like every stream when both are present —
    becomes a labeled entry annotated with the true byte total.

    Args:
        stdout: The captured stdout text.
        stdout_bytes: The true byte total of stdout before truncation.
        stderr: The captured stderr text.
        stderr_bytes: The true byte total of stderr before truncation.

    Returns:
        Lines of rendered output, ready for joining into the error message.
    """
    streams = [
        ("stderr", stderr, stderr_bytes),
        ("stdout", stdout, stdout_bytes),
    ]
    present = [(label, text, total) for label, text, total in streams if text]

    if len(present) == 1 and not _is_truncated(*present[0][1:]):
        return [present[0][1]]
    return [line for entry in present for line in _labeled(*entry)]


def _is_truncated(text: str, total_bytes: int) -> bool:
    """Whether ``total_bytes`` exceeds what survived capture in ``text``."""
    return total_bytes > len(text.encode())


def _labeled(label: str, text: str, total: int) -> list[str]:
    """Build a labeled stream entry, flagging truncation against the byte total."""
    suffix = f" (truncated, {total} bytes total)" if _is_truncated(text, total) else ""
    return [f"--- {label}{suffix} ---", text]


# ---------------------------------------------------------------------------
# Target resolution and worktrees
# ---------------------------------------------------------------------------


def resolve_dir(target: Target, repo_dir: str, worktrees: list[WorktreeInfo]) -> str:
    """The directory a target runs in, materializing a worktree for a ref.

    A ref is benchmarked from its own worktree. The planned worktree is appended
    to ``worktrees`` before ``git worktree add`` runs, so a caller sweeping the
    registry on termination can remove a directory a killed add left behind.

    Args:
        target: The target to locate.
        repo_dir: The repository the worktree is added from.
        worktrees: The live registry of claimed worktrees, appended to in place.

    Returns:
        The directory the benchmark runs in.

    Raises:
        GymratError: When the system temp directory cannot be resolved, or
            ``git worktree add`` fails for the ref.
    """
    if isinstance(target, RefTarget):
        worktree = plan_worktree(target)
        worktrees.append(worktree)
        materialize_worktree(worktree, repo_dir)
        return worktree.dir
    return target.dir


def to_context(
    spec: TargetSpec,
    target: Target,
    repo_dir: str,
    worktrees: list[WorktreeInfo],
    position: Literal["old", "new"] | None = None,
) -> TargetContext:
    """Pair a resolved target with the directory it runs in and its display label.

    Args:
        spec: The target as the caller named it, carrying any explicit label.
        target: What ``spec`` resolved to.
        repo_dir: The repository a ref's worktree is added from.
        worktrees: The live registry of claimed worktrees, appended to in place.
        position: Which side of a comparison the target occupies, or ``None``
            when the run is not a two-sided comparison.

    Returns:
        The context the target is sampled under.

    Raises:
        GymratError: When the system temp directory cannot be resolved, or
            ``git worktree add`` fails for a ref.
    """
    return TargetContext(
        target=target,
        dir=resolve_dir(target, repo_dir, worktrees),
        label=resolve_label(spec.label, target),
        position=position,
    )


def resolve_label(explicit: str | None, target: Target) -> str:
    """The display label for a target.

    Args:
        explicit: A caller-supplied label, or ``None`` to derive one.
        target: The target a label is derived from when ``explicit`` is ``None``.

    Returns:
        ``explicit`` when given, else a ref's name, else an in-place target's
        directory basename.
    """
    if explicit is not None:
        return explicit
    if isinstance(target, RefTarget):
        return target.ref
    return Path(target.dir).name


async def run_with_worktrees[M, R](
    phase: Callable[[str, list[WorktreeInfo], asyncio.Event], Awaitable[M]],
    build_result: Callable[[M, CleanupResult], R],
) -> R:
    """Run a phase that may claim worktrees, sweeping them on every exit path.

    A termination cleanup is installed before any worktree exists, so a signal
    arriving mid-run still sweeps whatever was claimed; that cleanup aborts the
    run and sweeps. The normal path sweeps exactly once whether the phase returns
    or raises. When the sweep leaves worktrees behind, the phase's error is
    re-raised wrapped with the cleanup diagnostics.

    Args:
        phase: The work to run. It receives the repository directory, the
            registry it appends claimed worktrees to, and an abort event a
            termination signal sets.
        build_result: Combines the phase's measurement with the cleanup outcome
            into the return value.

    Returns:
        The value ``build_result`` produced from the measurement and cleanup.

    Raises:
        Exception: Whatever ``phase`` raises, propagated as-is when the worktree
            sweep succeeds, or — when the sweep also failed — a same-typed
            replacement whose message appends the cleanup diagnostics.
    """
    repo_dir = str(Path.cwd())
    worktrees: list[WorktreeInfo] = []
    abort = asyncio.Event()

    def terminate() -> None:
        abort.set()
        # Kill any live bench group synchronously: on the signal path the loop
        # may not resume to process the abort before the sweep runs, so the
        # child must be dead before cleanup_worktrees touches the worktrees.
        kill_live_process_groups()
        cleanup_worktrees(worktrees, repo_dir)

    uninstall = install_termination_cleanup(terminate)
    try:
        try:
            measurement = await phase(repo_dir, worktrees, abort)
        except Exception as error:
            cleanup = cleanup_worktrees(worktrees, repo_dir)
            wrapped = _with_cleanup_failures(error, cleanup)
            if wrapped is error:
                raise
            raise wrapped from error
        cleanup = cleanup_worktrees(worktrees, repo_dir)
        return build_result(measurement, cleanup)
    finally:
        uninstall()


def _with_cleanup_failures(error: Exception, cleanup: CleanupResult) -> Exception:
    """Fold cleanup diagnostics into ``error``, preserving its subclass and hint.

    Returns ``error`` unchanged when the sweep was clean. Otherwise returns a new
    exception of the same type carrying the original message plus the cleanup
    diagnostics, with ``error`` chained as its cause.

    ``type(error)(...)`` reconstructs the exact subclass — a
    :class:`~gymrat.errors.CommandError` stays a ``CommandError`` and an
    ``AdapterError`` stays an ``AdapterError`` — because every
    :class:`~gymrat.errors.GymratError` shares the ``(message, *, hint)``
    signature. A non-gymrat error has no hint and becomes a plain ``Exception``.

    Args:
        error: The error the phase raised.
        cleanup: The outcome of the worktree sweep.

    Returns:
        ``error`` when the sweep left nothing behind, else a same-typed
        replacement whose message appends the cleanup diagnostics.
    """
    details = format_cleanup_failures(cleanup.failures, cleanup.prune_error)
    if not details:
        return error

    combined = "\n".join([str(error), "", "cleanup did not finish:", *details])
    if isinstance(error, GymratError):
        return type(error)(combined, hint=hint_of(error))
    return Exception(combined)
