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
from typing import Literal

from gymrat.adapters import DEFAULT_METRIC_KIND, Adapter
from gymrat.clock import monotonic_ms
from gymrat.config.types import KindEntry, MetricEntry
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
from gymrat.warn import WarnSink

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
            Defaults to :func:`~gymrat.clock.monotonic_ms`.
    """

    bench: str
    prepare: str | None
    samples: int
    timeout_seconds: float
    on_progress: ProgressCallback | None = None
    warn: WarnSink | None = None
    clock: Callable[[], float] = monotonic_ms


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

    def sampling(self) -> SamplingOptions:
        """The sampling settings this run hands the collector.

        The collector's clock is left at its default.

        Returns:
            The bench and prepare commands, round count, timeout, and hooks of
            this run.
        """
        return SamplingOptions(
            bench=self.bench,
            prepare=self.prepare,
            samples=self.samples,
            timeout_seconds=self.timeout_seconds,
            on_progress=self.on_progress,
            warn=self.warn,
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

    Args:
        metric_names: Metric names to resolve, in the order they should appear
            in the result.
        config_metrics: Per-metric overrides from the config file, keyed by
            metric name, or ``None`` if none are configured.
        adapter: Adapter supplying each metric's defaults.
        config_kinds: Per-kind gating overrides from the config file, keyed by
            kind name, or ``None`` if none are configured.

    Returns:
        An ordered mapping from metric name to its resolved
        :class:`ResolvedMetricMeta`.
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


def paired_or_own_values(
    paired: Sequence[float],
    samples: Sequence[dict[str, float]],
    name: str,
) -> list[float]:
    """Prefer already-paired values, falling back to a side's own values.

    Args:
        paired: Values paired across sides; used as-is when non-empty.
        samples: One metric record per round, used only for the fallback.
        name: The metric to extract when falling back.

    Returns:
        ``paired`` when it holds any values, otherwise ``own_values(samples, name)``.
    """
    return list(paired) or own_values(samples, name)


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

    Args:
        sample_sets: One list of per-round metric records per target.
        config_metrics: Per-metric overrides from config, or ``None``.
        adapter: The adapter whose defaults seed each metric's metadata.
        config_kinds: Per-kind overrides from config, or ``None``.

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

    return resolve_metric_meta(list(names), config_metrics, adapter, config_kinds)


# ---------------------------------------------------------------------------
# sample collection
# ---------------------------------------------------------------------------


_REF_HINT = (
    "the worktree only contains files tracked at this ref; "
    "untracked, gitignored, or not-yet-committed files are absent"
)
_LABEL_WIDTH = 11

# ---------------------------------------------------------------------------
# Sampling schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Schedule:
    """The fixed inputs shared by the prepare and bench loops."""

    targets: Sequence[TargetContext]
    options: SamplingOptions
    timeout_ms: int
    abort: asyncio.Event


async def _run_prepare(schedule: _Schedule) -> None:
    """Run the prepare command once per target before any bench run."""
    options = schedule.options
    prepare = options.prepare
    if prepare is None:
        return
    for ctx in schedule.targets:
        emit_progress(options.on_progress, PrepareStarted(label=ctx.label, at_ms=options.clock()))
        await _run_command("prepare", None, prepare, ctx, schedule.timeout_ms, schedule.abort)
        emit_progress(options.on_progress, PrepareFinished(label=ctx.label, at_ms=options.clock()))


async def _run_one_pass(
    round_index: int,
    schedule: _Schedule,
    collected: list[list[dict[str, float]]],
    adapter: Adapter,
) -> None:
    """Run one bench round across all targets, appending parsed records."""
    round_number = round_index + 1
    options = schedule.options
    target_count = len(schedule.targets)
    for target_index, ctx in enumerate(schedule.targets):
        emit_progress(
            options.on_progress,
            PassStarted(
                round=round_number,
                total_rounds=options.samples,
                target_count=target_count,
                label=ctx.label,
                at_ms=options.clock(),
            ),
        )
        stdout = await _run_command(
            "bench", round_number, options.bench, ctx, schedule.timeout_ms, schedule.abort
        )
        emit_progress(
            options.on_progress,
            PassFinished(
                round=round_number,
                total_rounds=options.samples,
                target_count=target_count,
                label=ctx.label,
                at_ms=options.clock(),
            ),
        )
        collected[target_index].append(_parse(adapter, stdout, options.warn))


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
    schedule = _Schedule(
        targets=targets,
        options=options,
        timeout_ms=int(options.timeout_seconds * MS_PER_SECOND),
        abort=abort,
    )
    collected: list[list[dict[str, float]]] = [[] for _ in targets]

    await _run_prepare(schedule)

    for round_index in range(options.samples):
        await _run_one_pass(round_index, schedule, collected, adapter)

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
        raise to_command_error(phase, sample_index, command, ctx, result, timeout_ms)
    return result.stdout


def _parse(adapter: Adapter, stdout: str, warn: WarnSink | None) -> dict[str, float]:
    if warn is None:
        return adapter.parse(stdout)
    return adapter.parse(stdout, warn)


# ---------------------------------------------------------------------------
# Command failure diagnosis
# ---------------------------------------------------------------------------


def to_command_error(  # noqa: PLR0913, PLR0917 -- one field per failure axis
    phase: str,
    sample_index: int | None,
    command: str,
    ctx: TargetContext,
    result: ExecResult | ExecTimeoutError,
    request_timeout_ms: int,
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
        request_timeout_ms: The configured timeout, used when ``result`` did
            not time out.

    Returns:
        The formatted command error with target-specific location context.
    """
    timed_out = isinstance(result, ExecTimeoutError)
    if timed_out:
        timeout_ms = result.timeout_ms
        exit_code: int | None = None
    else:
        timeout_ms = request_timeout_ms
        exit_code = result.exit_code

    target = ctx.target
    if isinstance(target, RefTarget):
        location = [_field("ref", target.ref), _field("worktree", ctx.dir)]
        hint = _REF_HINT
    else:
        location = [_field("dir", ctx.dir)]
        hint = None

    position = f"{ctx.position}, " if ctx.position is not None else ""
    sample = f", sample {sample_index}" if sample_index is not None else ""
    outcome_label = "timed out" if timed_out else "failed"
    header = f'{phase} command {outcome_label} ({position}"{ctx.label}"{sample})'

    lines = [header, *location, _field("command", command)]
    if timed_out:
        lines.append(_field("timeout", f"{timeout_ms}ms"))
    else:
        lines.append(_field("exit code", exit_code))
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

    if len(present) == 1:
        label, text, total = present[0]
        if not _is_truncated(text, total):
            return [text]
        return _labeled(label, text, total)

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
        wrapped: Exception = type(error)(combined, hint=hint_of(error))
    else:
        wrapped = Exception(combined)
    return wrapped
