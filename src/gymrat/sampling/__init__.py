"""Sequential sample collection for a benchmark comparison.

Runs a bench command a fixed number of times against each target, parsing every
run's output into a metric record. Runs are strictly sequential and interleaved
by round: every target is sampled once before any target is sampled again, so a
transient slowdown on the machine spreads across both sides rather than skewing
one. An optional prepare command runs once per target before sampling begins.

A single policy site turns a failed command (non-zero exit or timeout) into a
:class:`~gymrat.errors.CommandError`, so a failure anywhere stops the schedule
with the same formatted diagnosis.

The package also resolves each target to the directory and display label it runs
under, and wraps a phase that may claim git worktrees so they are swept on every
exit path.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from gymrat.adapters.types import Adapter
from gymrat.errors import CommandError, GymratError, hint_of
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError, kill_live_process_groups
from gymrat.exec import (
    exec as exec,  # noqa: A004 -- names the subprocess executor `exec`
)
from gymrat.progress_events import (
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    emit_progress,
)
from gymrat.report.text import format_cleanup_failures
from gymrat.sampling.summary import (
    compute_metric_stats,
    own_values,
    paired_or_own_values,
    resolve_metric_meta_from_samples,
)
from gymrat.sampling.types import (
    MetricStats,
    RunOptions,
    SamplingOptions,
    TargetContext,
    TargetSamples,
    TargetSpec,
)
from gymrat.signals import install_termination_cleanup
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

__all__ = [
    "MetricStats",
    "RunOptions",
    "SamplingOptions",
    "TargetContext",
    "TargetSamples",
    "TargetSpec",
    "collect_samples",
    "compute_metric_stats",
    "own_values",
    "paired_or_own_values",
    "resolve_dir",
    "resolve_label",
    "resolve_metric_meta_from_samples",
    "run_with_worktrees",
]

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
        timeout_ms=int(options.timeout_seconds * 1000),
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
