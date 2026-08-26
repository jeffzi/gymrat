"""End-to-end ``measure`` tests over real scratch repos and shell bench scripts.

These drive the full sampling pipeline — target resolution, worktree lifecycle,
and ``sh`` subprocesses whose stdout the ``metric-lines`` adapter parses — so the
worktree-sweep and error-surfacing behavior is exercised for real. Signal and
subprocess-kill assertions are out of scope here (they belong to the CLI suite).
"""

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from gymrat_py import sampling
from gymrat_py.errors import CommandError
from gymrat_py.measure import MeasureOptions, measure
from gymrat_py.sampling import TargetSpec
from gymrat_py.targets import CleanupResult, WorktreeInfo, WorktreeRemovalFailure
from tests._git import git as _git

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell")

_EMIT_ONE = "#!/bin/sh\necho 'METRIC x=1'\n"
_FAIL = "#!/bin/sh\nexit 1\n"


def _commit_bench(repo: str, script: str) -> None:
    (Path(repo) / "bench.sh").write_text(script, encoding="utf-8")
    _git(repo, "add", "bench.sh")
    _git(repo, "commit", "-m", "add bench")


def _options(target: str) -> MeasureOptions:
    return MeasureOptions(
        bench="sh bench.sh",
        prepare=None,
        adapter="metric-lines",
        samples=2,
        timeout_seconds=30.0,
        config_metrics=None,
        config_kinds=None,
        target=TargetSpec(label=None, target=target),
    )


async def test_measure_when_in_place_target_does_bench_without_worktree(
    create_scratch_repo: Callable[[], str],
    create_in_place_target_dir: Callable[[str, str, str], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    repo = create_scratch_repo()
    target_dir = create_in_place_target_dir(repo, "bench", _EMIT_ONE)
    monkeypatch.chdir(repo)

    result = await measure(_options(target_dir))

    assert result.metrics["x"].median == 1.0
    assert result.worktrees_removed == 0
    assert list_worktree_dirs(repo, include_main=False) == []


async def test_measure_when_ref_target_does_bench_in_worktree_and_sweep(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    repo = create_scratch_repo()
    _commit_bench(repo, _EMIT_ONE)
    monkeypatch.chdir(repo)

    result = await measure(_options("HEAD"))

    assert result.metrics["x"].median == 1.0
    assert result.worktrees_removed >= 1
    assert list_worktree_dirs(repo, include_main=False) == []


async def test_measure_when_bench_fails_does_reject_and_remove_worktrees(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    repo = create_scratch_repo()
    _commit_bench(repo, _FAIL)
    monkeypatch.chdir(repo)

    with pytest.raises(CommandError):
        await measure(_options("HEAD"))

    assert list_worktree_dirs(repo, include_main=False) == []


async def test_measure_when_bench_fails_and_worktree_unremovable_does_name_stranded_dir(
    create_scratch_repo: Callable[[], str],
    monkeypatch: pytest.MonkeyPatch,
):
    repo = create_scratch_repo()
    _commit_bench(repo, _FAIL)
    monkeypatch.chdir(repo)
    dirty = CleanupResult(
        removed=0,
        failures=(WorktreeRemovalFailure(dir="/tmp/stranded-wt", error="in use"),),
        prune_error=None,
    )

    def fake_cleanup_worktrees(worktrees: Sequence[WorktreeInfo], repo_dir: str) -> CleanupResult:
        return dirty

    monkeypatch.setattr(sampling, "cleanup_worktrees", fake_cleanup_worktrees)

    with pytest.raises(CommandError) as caught:
        await measure(_options("HEAD"))

    message = str(caught.value)
    assert "/tmp/stranded-wt" in message
    assert "bench command failed" in message
