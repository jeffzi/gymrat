"""Sequential sample collection for a benchmark comparison.

Runs a bench command a fixed number of times against each target, parsing every
run's output into a metric record. Runs are strictly sequential and interleaved
by round: every target is sampled once before any target is sampled again, so a
transient slowdown on the machine spreads across both sides rather than skewing
one. The rounds follow a Williams order, so no target always runs first and no
target always runs after the same one: over one cycle (one round for a single
target, two for two, ``N`` for an even count ``N`` of three or more, ``2N`` for
an odd one) each target holds each run position equally often, and runs directly
after each other target equally often. Two targets alternate, baseline first. A
sample count that is not a multiple of the cycle is refused before any command
runs. An optional prepare command runs once per target, in target order, before
sampling begins.

A single policy site turns a failed command (non-zero exit or timeout) into a
:class:`~gymrat.errors.CommandError`, so a failure anywhere stops the schedule
with the same formatted diagnosis. A bench run whose stdout overran exec's
capture cap stops the schedule the same way, before an adapter reads the cut
output.

The module also resolves each target to the directory and display label it runs
under, checking a ref out into a worktree of its own; sweeping those worktrees
lives in :mod:`gymrat.worktree_sweep`.

Besides collection, it holds the sampling dataclasses.
"""

import asyncio
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from gymrat import utils
from gymrat.adapters.contract import Adapter
from gymrat.config import KindEntry, MetricEntry, ResolvedConfig
from gymrat.errors import CommandError, GymratError
from gymrat.exec import (
    OUTPUT_CAP,
    ExecOptions,
    ExecResult,
    ExecTimeoutError,
    exec,  # noqa: A004 -- names the subprocess executor `exec`
)
from gymrat.git import run_git_step
from gymrat.progress_events import (
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressCallback,
    emit_progress,
)
from gymrat.sample_balance import require_balanced_samples
from gymrat.targets import RefTarget, Target
from gymrat.utils import (
    MS_PER_SECOND,
    WarnSink,
    balancing_cycle,
    stderr_text_of,
    warn_to_stderr,
)

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
            Defaults to :func:`~gymrat.utils.monotonic_ms`.
    """

    bench: str
    prepare: str | None
    samples: int
    timeout_seconds: float
    on_progress: ProgressCallback | None = None
    warn: WarnSink = warn_to_stderr
    clock: Callable[[], float] = utils.monotonic_ms


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


# ---------------------------------------------------------------------------
# sample collection
# ---------------------------------------------------------------------------


_REF_HINT = (
    "the worktree only contains files tracked at this ref; "
    "untracked, gitignored, or not-yet-committed files are absent"
)
_CAP_MIB = OUTPUT_CAP // (1024 * 1024)
_STDOUT_CAP_HINT = (
    "make the bench print less to stdout, for instance by keeping task functions "
    f"from logging, so it stays under the {_CAP_MIB} MiB capture cap"
)
_LABEL_WIDTH = 11


async def collect_samples(
    adapter: Adapter,
    targets: Sequence[TargetContext],
    options: SamplingOptions,
    abort: asyncio.Event,
) -> list[TargetSamples]:
    """Run the prepare and bench schedule and collect each target's samples.

    The sample count is checked first, so a count that cannot balance the
    targets is refused before any prepare or bench command runs. Prepare (when
    set) then runs once per target before any bench run. Then for each round,
    bench runs once per target, so round ``n+1`` never starts before round
    ``n`` has run every target. The rounds follow the Williams order of
    :func:`_williams_order`, cycling through it, so each target holds each run
    position and runs directly after each other target equally often. Each
    successful bench run contributes one parsed metric record to its target, so
    a target's samples stay in round order whatever the run order.

    Args:
        adapter: Parses a bench run's stdout into a metric record.
        targets: The targets to sample, in the order they are prepared and returned.
        options: The bench and prepare commands, round count, timeout, and hooks.
        abort: An event whose being set kills the in-flight command; passed
            through to the command runner.

    Returns:
        One :class:`TargetSamples` per target, in target order.

    Raises:
        GymratError: ``options.samples`` is not a multiple of the balancing
            cycle for the target count; the error carries the
            ``unbalanced-samples`` reason.
        CommandError: A prepare or bench command timed out or exited non-zero,
            or a bench run's stdout overran exec's capture cap.
        AdapterError: A bench run's output yielded no usable metric or an
            unacceptable metric name.
    """
    require_balanced_samples(options.samples, len(targets))
    timeout_ms = int(options.timeout_seconds * MS_PER_SECOND)
    collected = [TargetSamples(ctx=ctx, samples=[]) for ctx in targets]

    def emit_pass(
        event: type[PassStarted] | type[PassFinished], round_number: int, ctx: TargetContext
    ) -> None:
        emit_progress(
            options.on_progress,
            event(
                round=round_number,
                total_rounds=options.samples,
                target_count=len(targets),
                label=ctx.label,
                at_ms=options.clock(),
            ),
        )

    if options.prepare is not None:
        for ctx in targets:
            emit_progress(
                options.on_progress, PrepareStarted(label=ctx.label, at_ms=options.clock())
            )
            await _run_command("prepare", None, options.prepare, ctx, timeout_ms, abort)
            emit_progress(
                options.on_progress, PrepareFinished(label=ctx.label, at_ms=options.clock())
            )

    run_orders = _williams_order(len(targets))
    for round_number in range(1, options.samples + 1):
        for index in run_orders[(round_number - 1) % len(run_orders)]:
            target = collected[index]
            emit_pass(PassStarted, round_number, target.ctx)
            result = await _run_command(
                "bench", round_number, options.bench, target.ctx, timeout_ms, abort
            )
            stdout = _require_whole_stdout(round_number, options.bench, target.ctx, result)
            emit_pass(PassFinished, round_number, target.ctx)
            target.samples.append(adapter.parse(stdout, options.warn))

    return collected


def _williams_order(count: int) -> list[list[int]]:
    """Build one balancing cycle of run orders over ``count`` target indexes.

    The first row interleaves the low and high ends (0, 1, n-1, 2, n-2, ...)
    and every later row shifts it by one, so each index sits in each position
    once and follows each other index once. An odd count cannot balance
    followers with one square, so the reversed rows are appended.

    Args:
        count: How many targets each round benches; at least 1.

    Returns:
        One run order per round of the cycle, :func:`~gymrat.utils.balancing_cycle`
        rows in all, each listing every target index once.
    """
    first = [
        (column + 1) // 2 if column % 2 else (count - column // 2) % count
        for column in range(count)
    ]
    rows = [[(index + shift) % count for index in first] for shift in range(count)]
    if balancing_cycle(count) > count:
        rows += [row[::-1] for row in rows]
    return rows


async def _run_command(  # noqa: PLR0913, PLR0917 -- one parameter per command-execution axis
    phase: str,
    sample_index: int | None,
    command: str,
    ctx: TargetContext,
    timeout_ms: int,
    abort: asyncio.Event,
) -> ExecResult:
    """Run one command and return its zero-exit result, or raise on failure."""
    result = await exec(command, ExecOptions(cwd=ctx.dir, timeout_ms=timeout_ms, abort=abort))
    if isinstance(result, ExecTimeoutError) or result.exit_code != 0:
        raise _to_command_error(phase, sample_index, command, ctx, result)
    return result


# ---------------------------------------------------------------------------
# Command failure diagnosis
# ---------------------------------------------------------------------------


def _to_command_error(
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
    if isinstance(result, ExecTimeoutError):
        outcome_label = "timed out"
        outcome = _field("timeout", f"{result.timeout_ms}ms")
    else:
        outcome_label = "failed"
        outcome = _field("exit code", result.exit_code)
    hint = _REF_HINT if isinstance(ctx.target, RefTarget) else None
    return _command_error(
        _header(phase, outcome_label, sample_index, ctx),
        ctx,
        command,
        [outcome, *_captured_output(result)],
        hint,
    )


def _require_whole_stdout(
    sample_index: int, command: str, ctx: TargetContext, result: ExecResult
) -> str:
    """Return a bench run's stdout, refusing one that exec cut at its capture cap.

    The error states the bytes written against the cap rather than a kept
    size: exec may keep up to one read past the cap.

    Args:
        sample_index: The sample number to mention in the header.
        command: The bench command line.
        ctx: The target context supplying the header label, position, and
            location lines.
        result: The bench run's zero-exit outcome.

    Returns:
        The captured stdout, whole.

    Raises:
        CommandError: The stdout overran the capture cap; the hint asks for a
            bench that prints less.
    """
    if not result.stdout_truncated:
        return result.stdout
    written = f"{result.stdout_bytes} bytes written, exceeded the {_CAP_MIB} MiB capture cap"
    overflow = _command_error(
        _header("bench", "wrote too much to stdout", sample_index, ctx),
        ctx,
        command,
        [_field("stdout", written)],
        _STDOUT_CAP_HINT,
    )
    raise overflow


def _command_error(
    header: str, ctx: TargetContext, command: str, details: list[str], hint: str | None
) -> CommandError:
    """Build the failed-command and stdout-overflow errors here so both share one layout."""
    target = ctx.target
    if isinstance(target, RefTarget):
        location = [_field("ref", target.ref), _field("worktree", ctx.dir)]
    else:
        location = [_field("dir", ctx.dir)]
    lines = [header, *location, _field("command", command), *details]
    return CommandError("\n".join(lines), hint=hint)


def _header(phase: str, outcome: str, sample_index: int | None, ctx: TargetContext) -> str:
    """Build a command error's first line naming the phase, outcome, target, and sample."""
    position = f"{ctx.position}, " if ctx.position is not None else ""
    sample = f", sample {sample_index}" if sample_index is not None else ""
    return f'{phase} command {outcome} ({position}"{ctx.label}"{sample})'


def _field(label: str, value: object) -> str:
    """Format an indented, column-aligned ``label: value`` detail line."""
    return f"  {(label + ':').ljust(_LABEL_WIDTH)}{value}"


def _captured_output(result: ExecResult | ExecTimeoutError) -> list[str]:
    """Render the captured output of a failed command.

    A lone non-empty stream is emitted bare unless exec recorded it as
    truncated, in which case it — like every stream when both are present —
    becomes a labeled entry, annotated with the true byte total when truncated.

    Args:
        result: The failed command's outcome, carrying each stream's captured
            text, its true byte total, and whether exec truncated it.

    Returns:
        Lines of rendered output, ready for joining into the error message.
    """
    streams = [
        ("stderr", result.stderr, result.stderr_bytes, result.stderr_truncated),
        ("stdout", result.stdout, result.stdout_bytes, result.stdout_truncated),
    ]
    present = [(label, text, total, truncated) for label, text, total, truncated in streams if text]

    match present:
        case [(_, text, _, truncated)] if not truncated:
            return [text]
    return [
        line
        for label, text, total, truncated in present
        for line in _labeled(label, text, total, truncated=truncated)
    ]


def _labeled(label: str, text: str, total: int, *, truncated: bool) -> list[str]:
    """Build a labeled stream entry, flagging truncation with the byte total."""
    suffix = f" (truncated, {total} bytes total)" if truncated else ""
    return [f"--- {label}{suffix} ---", text]


# ---------------------------------------------------------------------------
# Target resolution and worktrees
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorktreeInfo:
    """A worktree directory this process claimed for a ref, pinned to a SHA.

    The directory need not exist: :func:`plan_worktree` reserves the path before
    any git runs, and :func:`materialize_worktree` can fail after creating it.
    Cleanup treats an absent directory as nothing to do rather than as an error.

    Attributes:
        dir: The path the worktree will live at (may not yet exist on disk).
        sha: The commit the worktree is checked out at.
        created: Whether ``git worktree add`` ever put this directory on disk.
            :func:`plan_worktree` starts it ``False`` and
            :func:`materialize_worktree` raises it once the add leaves something
            behind, which is what lets cleanup tell a worktree that was never
            created from one that was created and has since vanished — only the
            latter can leave a registry entry behind to clear. It is cleared
            again on a successful removal.
    """

    dir: str
    sha: str
    created: bool


def plan_worktree(ref: RefTarget) -> WorktreeInfo:
    """Choose where a ref target's worktree will live, without touching disk.

    Deciding the path up front is what lets a caller register the directory for
    cleanup before git can create it: ``git worktree add`` can be killed once the
    worktree is on disk but before it returns, and cleanup only sweeps paths
    something already names.

    Args:
        ref: The ref target whose commit the worktree will be pinned to.

    Returns:
        A planned :class:`WorktreeInfo` whose directory does not yet exist.

    Raises:
        GymratError: When the system temp directory cannot be resolved.
    """
    tmp_base = tempfile.gettempdir()
    try:
        resolved_base = str(Path(tmp_base).resolve(strict=True))
    except OSError as error:
        message = f"Cannot resolve temp directory '{tmp_base}': {stderr_text_of(error)}"
        raise GymratError(message) from error

    return WorktreeInfo(
        dir=str(Path(resolved_base) / f"gymrat-wt-{uuid.uuid4()}"),
        sha=ref.resolved_sha,
        created=False,
    )


def materialize_worktree(worktree: WorktreeInfo, repo_dir: str) -> None:
    """Check a planned worktree out into its directory, detached at its SHA.

    Cleanup must be able to tell a worktree that was never created from one
    that was.

    Args:
        worktree: The planned worktree to check out; mutated in place so
            ``created`` reflects what landed on disk.
        repo_dir: The repository the worktree is added to.

    Raises:
        GymratError: When ``git worktree add`` fails or git cannot be started —
            the planned directory may exist anyway, so callers must still hand
            it to :func:`~gymrat.worktree_sweep.cleanup_worktrees`.
    """
    try:
        run_git_step(
            ["worktree", "add", "--detach", worktree.dir, worktree.sha],
            repo_dir,
            f"git worktree add failed for {worktree.sha}",
            allow_interrupt=True,
        )
    finally:
        # git registers the worktree before the command returns, and the add can
        # be killed in between, so what landed on disk — not whether git exited
        # zero — is what says a registry entry may exist.
        worktree.created = Path(worktree.dir).exists()


def to_context(
    spec: TargetSpec,
    target: Target,
    repo_dir: str,
    worktrees: list[WorktreeInfo],
    position: Literal["old", "new"] | None = None,
) -> TargetContext:
    """Pair a resolved target with the directory it runs in and its display label.

    A ref is benchmarked from its own worktree. The planned worktree is appended
    to ``worktrees`` before ``git worktree add`` runs, so a caller sweeping the
    registry on termination can remove a directory a killed add left behind.

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
    if isinstance(target, RefTarget):
        worktree = plan_worktree(target)
        worktrees.append(worktree)
        materialize_worktree(worktree, repo_dir)
        directory = worktree.dir
    else:
        directory = target.dir
    return TargetContext(
        target=target,
        dir=directory,
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
