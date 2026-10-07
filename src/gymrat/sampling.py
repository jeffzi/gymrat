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
import subprocess
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, Self

from gymrat.adapters import Adapter
from gymrat.clock import monotonic_ms
from gymrat.config import KindEntry, MetricEntry, ResolvedConfig
from gymrat.errors import CommandError, GymratError
from gymrat.exec import (
    ExecOptions,
    ExecResult,
    ExecTimeoutError,
    exec,  # noqa: A004 -- names the subprocess executor `exec`
    kill_live_process_groups,
)
from gymrat.git import run_git, try_git
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
from gymrat.signals import install_termination_cleanup, write_on_exit
from gymrat.stats import compute_half_range
from gymrat.targets import RefTarget, Target, WorktreeRemovalFailure
from gymrat.utils import MS_PER_SECOND, WarnSink, stderr_text_of, warn_to_stderr

DEFAULT_METRIC_KIND: Final[str] = "other"
"""The kind a metric falls under when its adapter reports none."""

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


def resolve_metric_meta(
    name: str,
    entry: MetricEntry | None,
    adapter: Adapter,
    config_kinds: dict[str, KindEntry] | None,
) -> ResolvedMetricMeta:
    """Resolve one metric's metadata from its config entry, its kind and the adapter.

    The adapter's defaults for the metric are the base. The ``metrics`` entry
    overrides direction, gating and exact wherever it sets them. Gating the
    entry leaves unset comes from the ``kinds`` entry of the metric's kind, and
    gates by default when that is unset too.

    Every reader of a metric's direction resolves it here, so a verdict and a
    display of the same metric cannot disagree about which way is better.

    Args:
        name: The metric's full name, as the adapter reports it.
        entry: The metric's ``metrics`` entry from config, or ``None`` when
            config does not name it.
        adapter: The adapter whose defaults seed the metadata.
        config_kinds: Per-kind gating overrides from config, keyed by kind name,
            or ``None``.

    Returns:
        The metric's resolved metadata.
    """
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
    config_kinds: dict[str, KindEntry] | None,
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
        name: resolve_metric_meta(
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

    for round_number in range(1, options.samples + 1):
        for target in collected:
            emit_pass(PassStarted, round_number, target.ctx)
            stdout = await _run_command(
                "bench", round_number, options.bench, target.ctx, timeout_ms, abort
            )
            emit_pass(PassFinished, round_number, target.ctx)
            target.samples.append(adapter.parse(stdout, options.warn))

    return collected


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
        raise _to_command_error(phase, sample_index, command, ctx, result)
    return result.stdout


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

    lines = [header, *location, _field("command", command), outcome, *_captured_output(result)]

    return CommandError("\n".join(lines), hint=hint)


def _field(label: str, value: object) -> str:
    """Format an indented, column-aligned ``label: value`` detail line."""
    return f"  {(label + ':').ljust(_LABEL_WIDTH)}{value}"


def _captured_output(result: ExecResult | ExecTimeoutError) -> list[str]:
    """Render the captured output of a failed command.

    A lone non-empty stream is emitted bare unless its captured text was
    truncated, in which case it — like every stream when both are present —
    becomes a labeled entry annotated with the true byte total.

    Args:
        result: The failed command's outcome, carrying each stream's captured
            text and its true byte total before truncation.

    Returns:
        Lines of rendered output, ready for joining into the error message.
    """
    streams = [
        ("stderr", result.stderr, result.stderr_bytes),
        ("stdout", result.stdout, result.stdout_bytes),
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


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """Outcome of a worktree cleanup sweep.

    Attributes:
        removed: Worktrees this call took off disk.
        failures: Worktrees left on disk, one entry each.
        prune_error: Why the repo-wide ``git worktree prune`` sweep failed, or
            ``None`` if it succeeded or never ran. Prune runs once per call, not
            once per worktree, so it gets its own slot rather than a synthetic
            entry in ``failures``.
    """

    removed: int
    failures: tuple[WorktreeRemovalFailure, ...]
    prune_error: str | None


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

    Records on ``worktree`` whether anything reached disk, so cleanup can tell a
    worktree that was never created from one that was.

    Args:
        worktree: The planned worktree to check out; mutated in place so
            ``created`` reflects what landed on disk.
        repo_dir: The repository the worktree is added to.

    Raises:
        GymratError: When ``git worktree add`` fails or git cannot be started —
            the planned directory may exist anyway, so callers must still hand
            it to :func:`cleanup_worktrees`.
    """
    try:
        run_git(["worktree", "add", "--detach", worktree.dir, worktree.sha], repo_dir)
    except (subprocess.SubprocessError, OSError) as error:
        # OSError is a git binary that is missing or cannot be executed.
        message = f"git worktree add failed for {worktree.sha}: {stderr_text_of(error)}"
        raise GymratError(message) from error
    finally:
        # git registers the worktree before the command returns, and the add can
        # be killed in between, so what landed on disk — not whether git exited
        # zero — is what says a registry entry may exist.
        worktree.created = Path(worktree.dir).exists()


# What handing one worktree to git accomplished. ``removed`` took a directory off
# disk; ``stale`` is a vanished directory whose entry git would not clear, which a
# prune must collect. ``None`` leaves nothing to count or sweep: a worktree git
# never put on disk, or a vanished one whose entry git cleared. A
# :class:`WorktreeRemovalFailure` is a directory git refused to remove.
type _RemovalStatus = Literal["removed", "stale"] | None
type _RemovalOutcome = _RemovalStatus | WorktreeRemovalFailure


def _remove_worktree(worktree: WorktreeInfo, repo_dir: str) -> _RemovalOutcome:
    """Take one worktree off disk, or clear the entry left behind if it is gone.

    The removal names the worktree's own path instead of sweeping the
    repository, because git clears the entry of a directory that vanished behind
    its back only when asked for that path — which leaves a worktree of the
    user's own that is merely temporarily absent registered.

    Args:
        worktree: The worktree to remove.
        repo_dir: The repository the worktree belongs to.

    Returns:
        The removal status or a :class:`WorktreeRemovalFailure` when git
        refused.
    """
    on_disk = Path(worktree.dir).exists()
    if not on_disk and not worktree.created:
        return None

    error = try_git(["worktree", "remove", "--force", worktree.dir], repo_dir)
    if error is not None:
        # Nothing stands for the user to clear by hand when the directory is
        # already gone, so a refusal there is a reason to sweep, not to report.
        if on_disk:
            return WorktreeRemovalFailure(dir=worktree.dir, error=error)
        return "stale"

    # The entry is gone — clear the flag so a later sweep leaves it alone rather
    # than reclassifying it as stale.
    worktree.created = False
    return "removed" if on_disk else None


@dataclass(slots=True)
class _SweepLedger:
    """The worktrees no sweep has taken yet, and what the sweeps so far have learned.

    A signal can land while one sweep is inside a git call, and the sweep the
    signal path then runs is the last thing the process does. Both sweeps write
    to one ledger, so the second reports the worktrees the first could not
    remove and collects the stale entry the first had met.

    Attributes:
        worktrees: The registry of claimed worktrees, emptied as sweeps take them.
        removed: Worktrees the sweeps took off disk.
        failures: Worktrees the sweeps left on disk, one entry each.
        prune_owed: Whether a targeted removal came back stale and no sweep has
            started the prune that collects it.
    """

    worktrees: list[WorktreeInfo]
    removed: int = 0
    failures: list[WorktreeRemovalFailure] = field(default_factory=list)
    prune_owed: bool = False

    def __iter__(self) -> Iterator[WorktreeInfo]:
        # Each worktree leaves the registry before it is yielded, so a sweep that
        # starts while another is inside a git call finds only the worktrees no
        # sweep has taken: git is asked to remove each one once, and a worktree git
        # has just removed is never re-read as a vanished one that needs a prune.
        while self.worktrees:
            yield self.worktrees.pop(0)


def cleanup_worktrees(worktrees: Iterable[WorktreeInfo], repo_dir: str) -> CleanupResult:
    """Remove each of ``worktrees`` git has anything for.

    Never raises: callers invoke this while already handling a failed run, so a
    throw here would replace the error the user actually needs to see.
    Everything that went wrong lands in the returned result instead.

    Args:
        worktrees: The worktrees to sweep, each taken just before git is asked
            to remove it. A sweep that takes over from an interrupted one
            carries on from what that sweep had already recorded.
        repo_dir: The repository the removals and prune run against.

    Returns:
        A :class:`CleanupResult` counting removals, listing failures, and
        carrying any prune-sweep error.
    """
    ledger = worktrees if isinstance(worktrees, _SweepLedger) else _SweepLedger([])

    for worktree in worktrees:
        outcome = _remove_worktree(worktree, repo_dir)
        if isinstance(outcome, WorktreeRemovalFailure):
            ledger.failures.append(outcome)
        elif outcome == "removed":
            ledger.removed += 1
        elif outcome == "stale":
            ledger.prune_owed = True

    # Prune only when a targeted removal failed and may have left an entry with
    # no directory behind it. Naming each worktree already clears its own entry,
    # so an all-success sweep — or one with nothing to ask git for, in a repo_dir
    # that may not even be a git repo — has nothing to collect. Pruning anyway
    # would deregister worktrees of the user's own that are only temporarily
    # absent: an unmounted volume, a directory moved aside.
    prune_error = None
    if ledger.prune_owed:
        # Settled before git is asked: a sweep that starts while this prune is
        # running must not run a second one.
        ledger.prune_owed = False
        prune_error = try_git(["worktree", "prune"], repo_dir)
    return CleanupResult(
        removed=ledger.removed, failures=tuple(ledger.failures), prune_error=prune_error
    )


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


_CLEANUP_UNFINISHED = "cleanup did not finish:"


async def run_with_worktrees[M, R](
    phase: Callable[[str, list[WorktreeInfo], asyncio.Event], Awaitable[M]],
    build_result: Callable[[M, CleanupResult], R],
) -> R:
    """Run a phase that may claim worktrees, sweeping them on every exit path.

    A termination cleanup is installed before any worktree exists, so a signal
    arriving mid-run still sweeps whatever was claimed; that cleanup aborts the
    run, sweeps, and hands what the sweep left unfinished to the exit output.
    The normal path sweeps exactly once whether the phase returns or raises. A
    signal that lands during the normal path's sweep takes over only the
    worktrees that sweep has not reached, and inherits its accounting: the exit
    output names the worktrees that sweep could not remove, and a stale entry
    it had met is pruned once.

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
    ledger = _SweepLedger(worktrees)
    abort = asyncio.Event()

    def terminate() -> None:
        abort.set()
        # Kill any live bench group synchronously: on the signal path the loop
        # may not resume to process the abort before the sweep runs, so the
        # child must be dead before cleanup_worktrees touches the worktrees.
        kill_live_process_groups()
        cleanup = cleanup_worktrees(ledger, repo_dir)
        details = format_cleanup_failures(cleanup.failures, cleanup.prune_error)
        if details:
            # The process exits from the signal handler, so no caller is left
            # to report a worktree this sweep could not remove.
            write_on_exit("\n".join([_CLEANUP_UNFINISHED, *details]) + "\n")

    uninstall = install_termination_cleanup(terminate)
    try:
        measurement = await phase(repo_dir, worktrees, abort)
    except Exception as error:
        cleanup = cleanup_worktrees(ledger, repo_dir)
        wrapped = _with_cleanup_failures(error, cleanup)
        if wrapped is error:
            raise
        raise wrapped from error
    else:
        cleanup = cleanup_worktrees(ledger, repo_dir)
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

    combined = "\n".join([str(error), "", _CLEANUP_UNFINISHED, *details])
    if isinstance(error, GymratError):
        return type(error)(combined, hint=error.hint)
    return Exception(combined)
