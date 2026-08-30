"""Shared sampling-pipeline stubs for the compare and measure orchestrator tests.

The sampling pipeline (target resolution, worktree lifecycle, exec) is a system
boundary; these stubs replace it in memory so orchestrator tests pin result
assembly without spawning processes or creating worktrees.
"""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType

import pytest

from gymrat import sampling
from gymrat.adapters.types import Adapter
from gymrat.sampling import SamplingOptions, TargetContext, TargetSamples
from gymrat.targets import CleanupResult, InPlaceTarget, WorktreeInfo

CLEAN_RESULT = CleanupResult(removed=0, failures=(), prune_error=None)


@dataclass
class CapturedCall:
    """The ``SamplingOptions`` and contexts the stubbed collector was handed."""

    options: SamplingOptions | None = None
    contexts: list[TargetContext] | None = None


def install_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator: ModuleType,
    sample_sets: list[list[dict[str, float]]],
    cleanup: CleanupResult = CLEAN_RESULT,
) -> CapturedCall:
    """Replace resolution, collection, and worktree cleanup with in-memory stubs.

    ``resolve_target`` yields an in-place target rooted at the spec's raw target
    (so the derived label is that string's basename), ``collect_samples`` echoes
    each built context back paired with the matching entry of ``sample_sets``
    (index 0 is the baseline, the rest candidates in order), and the cleanup seam
    returns ``cleanup`` unchanged. ``orchestrator`` is the module whose
    ``resolve_target`` / ``collect_samples`` seams are patched. Returns the
    ``SamplingOptions`` and contexts the orchestrator handed the collector.
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
        captured.options = options
        captured.contexts = list(contexts)
        return [
            TargetSamples(ctx=ctx, samples=sample_sets[index]) for index, ctx in enumerate(contexts)
        ]

    def fake_install(cleanup_cb: Callable[[], None]) -> Callable[[], None]:
        return lambda: None

    def fake_cleanup_worktrees(worktrees: Sequence[WorktreeInfo], repo_dir: str) -> CleanupResult:
        return cleanup

    monkeypatch.setattr(orchestrator, "resolve_target", fake_resolve_target)
    monkeypatch.setattr(orchestrator, "collect_samples", fake_collect)
    monkeypatch.setattr(sampling, "install_termination_cleanup", fake_install)
    monkeypatch.setattr(sampling, "cleanup_worktrees", fake_cleanup_worktrees)
    return captured
