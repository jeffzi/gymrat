"""Shared sampling-pipeline stubs and run options for the orchestrator tests.

The sampling pipeline (target resolution, worktree lifecycle, exec) is a system
boundary; these stubs replace it in memory so the compare and measure
orchestrator tests pin result assembly without spawning processes or creating
worktrees. :func:`run_options` builds the run settings those tests hand the
orchestrators. :func:`quantized_warning` builds the text sampling warns with
when time readings look quantized, and :func:`one_step_up` builds readings that
trigger it.
"""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType

import pytest

from gymrat import sampling
from gymrat.adapters import Adapter
from gymrat.config import KindEntry, MetricEntry
from gymrat.progress_events import ProgressEvent
from gymrat.sampling import (
    CleanupResult,
    RunOptions,
    SamplingOptions,
    TargetContext,
    TargetSamples,
    WorktreeInfo,
    WorktreeRemovalFailure,
)
from gymrat.targets import InPlaceTarget
from gymrat.utils import WarnSink, warn_to_stderr
from tests._process_helpers import fake_install

CLEAN_RESULT = CleanupResult(removed=0, failures=(), prune_error=None)

DIRTY_RESULT = CleanupResult(
    removed=1,
    failures=(WorktreeRemovalFailure(dir="/tmp/gymrat-wt", error="contains modified files"),),
    prune_error="could not prune",
)
"""A sweep that removed one worktree, left another behind, and could not prune."""

QUANTIZED_STEP_NS = 42.0
"""The step, in nanoseconds, that :func:`one_step_up` readings climb and the warning names."""


def quantized_warning(name: str, gap: str | None) -> str:
    """The warning sampling sends when ``name``'s time readings look quantized.

    Args:
        name: The time metric the warning is about.
        gap: The step the warning names, such as ``"42ns"``, or ``None`` for the
            wording that names no step.

    Returns:
        The warning text.
    """
    step = "" if gap is None else f"; its step is at most {gap}"
    return (
        f"{name}: readings repeat as a quantized timer's would{step}. "
        "Set `resolution` to the timer's tick, or time more iterations per reading"
    )


QUANTIZED_TIME_WARNING = quantized_warning("bench#time", f"{QUANTIZED_STEP_NS:g}ns")
"""The warning for ``bench#time`` readings that repeat and differ by one 42ns step."""


def one_step_up(start: float, *names: str) -> list[dict[str, float]]:
    """Six rounds of quantized readings: five at ``start``, then one a step above.

    Args:
        start: The value the first five rounds report for every metric.
        *names: The metrics each round reports.

    Returns:
        The rounds, the last of which reads :data:`QUANTIZED_STEP_NS` above ``start``.
    """
    return [dict.fromkeys(names, start)] * 5 + [dict.fromkeys(names, start + QUANTIZED_STEP_NS)]


def run_options(
    *,
    samples: int,
    bench: str = "run",
    prepare: str | None = "prep",
    timeout_seconds: float = 1.0,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    warn: WarnSink = warn_to_stderr,
    config_metrics: dict[str, MetricEntry] | None = None,
    config_kinds: dict[str, KindEntry] | None = None,
) -> RunOptions:
    """Build ``metric-lines`` run options; the defaults suit a stubbed pipeline.

    Args:
        samples: The number of rounds.
        bench: The command run once per target per round.
        prepare: The command run once per target before sampling, or ``None``.
        timeout_seconds: The per-command wall-clock budget.
        on_progress: Receives each progress event, or ``None`` for silence.
        warn: Where adapter complaints go.
        config_metrics: Per-metric overrides, or ``None``.
        config_kinds: Per-kind gating overrides, or ``None``.

    Returns:
        Run options for the ``metric-lines`` adapter.
    """
    return RunOptions(
        sampling=SamplingOptions(
            bench=bench,
            prepare=prepare,
            samples=samples,
            timeout_seconds=timeout_seconds,
            on_progress=on_progress,
            warn=warn,
        ),
        adapter="metric-lines",
        config_metrics=config_metrics,
        config_kinds=config_kinds,
    )


@dataclass(slots=True)
class CapturedCall:
    """The contexts the stubbed collector was handed."""

    contexts: list[TargetContext] | None = None


def install_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator: ModuleType,
    sample_sets: list[list[dict[str, float]]],
    cleanup: CleanupResult = CLEAN_RESULT,
    *,
    progress_event: ProgressEvent | None = None,
    warning: str | None = None,
) -> CapturedCall:
    """Replace resolution, collection, and worktree cleanup with in-memory stubs.

    ``resolve_target`` yields an in-place target rooted at the spec's raw target,
    so the derived label is that string's basename.

    Args:
        monkeypatch: The fixture the stubs are installed through.
        orchestrator: The module whose ``resolve_target`` / ``collect_samples``
            seams are patched.
        sample_sets: The samples ``collect_samples`` pairs with each built
            context, index 0 for the baseline and the rest for the candidates
            in order.
        cleanup: What the worktree cleanup seam returns.
        progress_event: An event the collector sends through the sampling
            options' ``on_progress`` callback, or ``None`` to send nothing.
        warning: A message the collector sends through the sampling options'
            ``warn`` sink, or ``None`` to send nothing.

    Returns:
        The contexts the orchestrator handed the collector.
    """
    captured = CapturedCall()

    def fake_resolve_target(target_input: str, repo_dir: str) -> InPlaceTarget:
        return InPlaceTarget(dir=f"/repo/{target_input}")

    async def fake_collect(
        adapter: Adapter,
        contexts: Sequence[TargetContext],
        options: SamplingOptions,
        abort: asyncio.Event,
    ) -> list[TargetSamples]:
        captured.contexts = list(contexts)
        if progress_event is not None and options.on_progress is not None:
            options.on_progress(progress_event)
        if warning is not None:
            options.warn(warning)
        return [
            TargetSamples(ctx=ctx, samples=sample_sets[index]) for index, ctx in enumerate(contexts)
        ]

    monkeypatch.setattr(orchestrator, "resolve_target", fake_resolve_target)
    monkeypatch.setattr(orchestrator, "collect_samples", fake_collect)
    monkeypatch.setattr(sampling, "install_termination_cleanup", fake_install([]))
    install_cleanup(monkeypatch, cleanup)
    return captured


def install_cleanup(
    monkeypatch: pytest.MonkeyPatch, result: CleanupResult
) -> list[tuple[list[WorktreeInfo], str]]:
    """Replace the sampling worktree cleanup with one answering ``result``, recording each sweep.

    Args:
        monkeypatch: Patches ``sampling.cleanup_worktrees`` for the duration.
        result: What every sweep returns.

    Returns:
        The ``(worktrees, repo_dir)`` each sweep was handed, in call order.
    """
    sweeps: list[tuple[list[WorktreeInfo], str]] = []

    def fake_cleanup_worktrees(worktrees: Sequence[WorktreeInfo], repo_dir: str) -> CleanupResult:
        sweeps.append((list(worktrees), repo_dir))
        return result

    monkeypatch.setattr(sampling, "cleanup_worktrees", fake_cleanup_worktrees)
    return sweeps
