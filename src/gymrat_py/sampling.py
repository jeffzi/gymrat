"""Sequential sample collection for a benchmark comparison.

Runs a bench command a fixed number of times against each target, parsing every
run's output into a metric record. Runs are strictly sequential and interleaved
by round: every target is sampled once before any target is sampled again, so a
transient slowdown on the machine spreads across both sides rather than skewing
one. An optional prepare command runs once per target before sampling begins.

A single policy site turns a failed command (non-zero exit or timeout) into a
:class:`~gymrat_py.errors.CommandError`, so a failure anywhere stops the schedule
with the same formatted diagnosis.
"""

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gymrat_py.adapters.types import Adapter, WarnSink
from gymrat_py.config import KindEntry, MetricEntry, resolve_metric_meta
from gymrat_py.errors import CommandError, GymratError, hint_of, message_of
from gymrat_py.exec import ExecOptions, ExecResult, ExecTimeoutError, kill_live_process_groups
from gymrat_py.exec import exec as exec  # noqa: A004, PLC0414
from gymrat_py.model import ResolvedMetricMeta
from gymrat_py.report.text import format_cleanup_failures
from gymrat_py.signals import install_termination_cleanup
from gymrat_py.stats.descriptive import compute_half_range, compute_median
from gymrat_py.targets import (
    CleanupResult,
    RefTarget,
    Target,
    WorktreeInfo,
    cleanup_worktrees,
    materialize_worktree,
    plan_worktree,
)


@dataclass(frozen=True, slots=True)
class PrepareProgressStep:
    """A prepare command is about to run for a target.

    Attributes:
        label: The target's display label.
    """

    label: str


@dataclass(frozen=True, slots=True)
class SampleProgressStep:
    """A bench command is about to run for one round against a target.

    Attributes:
        index: The 1-based round number.
        total: The total number of rounds.
        label: The target's display label.
    """

    index: int
    total: int
    label: str


type ProgressStep = PrepareProgressStep | SampleProgressStep
"""A step the sampler is starting, reported to a progress callback."""


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
        on_progress: Called at the start of each step, or ``None`` for silence.
        warn: Where an adapter sends complaints about output it could not read,
            or ``None`` to use the adapter's own default.
    """

    bench: str
    prepare: str | None
    samples: int
    timeout_seconds: float
    on_progress: Callable[[ProgressStep], None] | None = None
    warn: WarnSink | None = None


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
        on_progress: Called at the start of each step, or ``None`` for silence.
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
    on_progress: Callable[[ProgressStep], None] | None = None
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


@dataclass(frozen=True, slots=True)
class _Step:
    """Which command in the schedule is running: its phase and round, if any."""

    phase: str
    sample_index: int | None


@dataclass(frozen=True, slots=True)
class _CommandSuccess:
    stdout: str


@dataclass(frozen=True, slots=True)
class _CommandFailure:
    phase: str
    ctx: TargetContext
    command: str
    sample_index: int | None
    timed_out: bool
    timeout_ms: int
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_bytes: int
    stderr_bytes: int


type _CommandOutcome = _CommandSuccess | _CommandFailure

_REF_HINT = (
    "the worktree only contains files tracked at this ref; "
    "untracked, gitignored, or not-yet-committed files are absent"
)
_LABEL_WIDTH = 11
_MIN_SPREAD_SAMPLES = 2


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
    timeout_ms = int(options.timeout_seconds * 1000)
    collected: list[list[dict[str, float]]] = [[] for _ in targets]

    if options.prepare is not None:
        for ctx in targets:
            _report(options, PrepareProgressStep(label=ctx.label))
            outcome = await _run_command(
                _Step("prepare", None), options.prepare, ctx, timeout_ms, abort
            )
            _enforce(outcome)

    for round_index in range(options.samples):
        for target_index, ctx in enumerate(targets):
            _report(
                options,
                SampleProgressStep(index=round_index + 1, total=options.samples, label=ctx.label),
            )
            outcome = await _run_command(
                _Step("bench", round_index + 1), options.bench, ctx, timeout_ms, abort
            )
            stdout = _enforce(outcome)
            collected[target_index].append(_parse(adapter, stdout, options.warn))

    return [
        TargetSamples(ctx=ctx, samples=samples)
        for ctx, samples in zip(targets, collected, strict=True)
    ]


async def _run_command(
    step: _Step,
    command: str,
    ctx: TargetContext,
    timeout_ms: int,
    abort: asyncio.Event,
) -> _CommandOutcome:
    """Run one command and classify its result as success or failure."""
    result = await exec(command, ExecOptions(cwd=ctx.dir, timeout_ms=timeout_ms, abort=abort))
    if isinstance(result, ExecTimeoutError) or result.exit_code != 0:
        return _to_failure(step, command, ctx, result, timeout_ms)
    return _CommandSuccess(stdout=result.stdout)


def _to_failure(
    step: _Step,
    command: str,
    ctx: TargetContext,
    result: ExecResult | ExecTimeoutError,
    request_timeout_ms: int,
) -> _CommandFailure:
    """Build a `_CommandFailure` from a run's outcome and its captured streams.

    A timeout carries its own budget as ``timeout_ms``; a non-zero exit carries
    the exit code and the run's requested budget (unused when rendering, but kept
    for a uniform record shape).
    """
    if isinstance(result, ExecTimeoutError):
        timed_out, timeout_ms, exit_code = True, result.timeout_ms, None
    else:
        timed_out, timeout_ms, exit_code = False, request_timeout_ms, result.exit_code
    return _CommandFailure(
        phase=step.phase,
        ctx=ctx,
        command=command,
        sample_index=step.sample_index,
        timed_out=timed_out,
        timeout_ms=timeout_ms,
        exit_code=exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        stdout_bytes=result.stdout_bytes,
        stderr_bytes=result.stderr_bytes,
    )


def _enforce(outcome: _CommandOutcome) -> str:
    """Return the run's stdout, or raise the failure as a formatted error.

    This is the single site that converts a failed command into a
    :class:`CommandError`; every command runs through it.
    """
    if isinstance(outcome, _CommandFailure):
        raise _to_command_error(outcome)
    return outcome.stdout


def _to_command_error(failure: _CommandFailure) -> CommandError:
    """Attach the target's location lines and ref-only hint to the error.

    A ref target contributes ``ref`` and ``worktree`` location lines plus the
    hint that the worktree only holds tracked files; a plain directory
    contributes a single ``dir`` line and no hint.
    """
    target = failure.ctx.target
    if isinstance(target, RefTarget):
        location = [_field("ref", target.ref), _field("worktree", failure.ctx.dir)]
        hint = _REF_HINT
    else:
        location = [_field("dir", failure.ctx.dir)]
        hint = None

    lines = [_header(failure), *location, _field("command", failure.command)]
    if failure.timed_out:
        lines.append(_field("timeout", f"{failure.timeout_ms}ms"))
    else:
        lines.append(_field("exit code", failure.exit_code))
    lines.extend(_captured_output(failure))

    return CommandError("\n".join(lines), hint=hint)


def _header(failure: _CommandFailure) -> str:
    """Build the one-line summary naming phase, outcome, target, and round."""
    position = f"{failure.ctx.position}, " if failure.ctx.position is not None else ""
    sample = f", sample {failure.sample_index}" if failure.sample_index is not None else ""
    outcome = "timed out" if failure.timed_out else "failed"
    return f'{failure.phase} command {outcome} ({position}"{failure.ctx.label}"{sample})'


def _field(label: str, value: object) -> str:
    """Format an indented, column-aligned ``label: value`` detail line."""
    return f"  {(label + ':').ljust(_LABEL_WIDTH)}{value}"


def _captured_output(failure: _CommandFailure) -> list[str]:
    """Render whatever the failed command wrote to its output streams.

    A lone non-empty stream is emitted bare unless its captured text was
    truncated, in which case it — like every stream when both are present —
    becomes a labeled entry annotated with the true byte total.
    """
    streams = [
        ("stderr", failure.stderr, failure.stderr_bytes),
        ("stdout", failure.stdout, failure.stdout_bytes),
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


def _report(options: SamplingOptions, step: ProgressStep) -> None:
    """Fire the progress callback when one is registered."""
    if options.on_progress is not None:
        options.on_progress(step)


def _parse(adapter: Adapter, stdout: str, warn: WarnSink | None) -> dict[str, float]:
    """Parse a bench run's stdout, routing warnings through ``warn`` when given."""
    if warn is None:
        return adapter.parse(stdout)
    return adapter.parse(stdout, warn)


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

    median = compute_median(values)
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
    if paired:
        return list(paired)
    return own_values(samples, name)


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
    :class:`~gymrat_py.errors.CommandError` stays a ``CommandError`` and an
    ``AdapterError`` stays an ``AdapterError`` — because every
    :class:`~gymrat_py.errors.GymratError` shares the ``(message, *, hint)``
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

    combined = "\n".join([message_of(error), "", "cleanup did not finish:", *details])
    if isinstance(error, GymratError):
        wrapped: Exception = type(error)(combined, hint=hint_of(error))
    else:
        wrapped = Exception(combined)
    wrapped.__cause__ = error
    return wrapped
