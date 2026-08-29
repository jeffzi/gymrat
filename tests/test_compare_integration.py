"""End-to-end ``compare`` tests over real scratch repos and shell bench scripts.

These drive the full sampling pipeline — target resolution for every ref before
any worktree is materialized, the worktree lifecycle, and ``sh`` subprocesses
whose stdout the ``metric-lines`` adapter parses — so the star comparison and
worktree-sweep behavior is exercised for real.
"""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.compare import CompareOptions, compare
from gymrat.errors import GymratError
from gymrat.sampling import TargetSpec
from tests._git import git as _git

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell")


def _commit_bench(repo: str, value: int) -> None:
    (Path(repo) / "bench.sh").write_text(f"#!/bin/sh\necho 'METRIC x={value}'\n", encoding="utf-8")
    _git(repo, "add", "bench.sh")
    _git(repo, "commit", "-m", f"bench emits {value}")


def _options(baseline: str, candidate: str) -> CompareOptions:
    return CompareOptions(
        bench="sh bench.sh",
        prepare=None,
        adapter="metric-lines",
        samples=3,
        timeout_seconds=30.0,
        config_metrics=None,
        config_kinds=None,
        baseline=TargetSpec(label=None, target=baseline),
        candidates=[TargetSpec(label=None, target=candidate)],
        unstable_noise_pct=None,
    )


async def test_compare_when_two_refs_does_produce_comparison_and_sweep(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    repo = create_scratch_repo()
    _commit_bench(repo, 1)
    _git(repo, "switch", "-c", "candidate")
    _commit_bench(repo, 2)
    _git(repo, "switch", "main")
    monkeypatch.chdir(repo)

    result = await compare(_options("main", "candidate"))

    assert result.baseline_label == "main"
    assert result.candidates[0].label == "candidate"
    assert result.metrics["x"].baseline_median == 1.0
    assert result.metrics["x"].candidates[0].median == 2.0
    assert result.worktrees_removed >= 2
    assert list_worktree_dirs(repo, include_main=False) == []


async def test_compare_when_candidate_unresolvable_does_fail_with_nothing_on_disk(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    repo = create_scratch_repo()
    _commit_bench(repo, 1)
    monkeypatch.chdir(repo)

    with pytest.raises(GymratError, match="no-such-ref"):
        await compare(_options("main", "no-such-ref"))

    assert list_worktree_dirs(repo, include_main=False) == []
