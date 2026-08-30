"""Tests for the ``measure`` orchestrator.

Unit tests stub the sampling pipeline to pin how ``measure`` assembles a
:class:`MeasurementResult` from canned per-round samples. End-to-end tests
drive real scratch repos and shell bench scripts through the full pipeline —
target resolution, worktree lifecycle, and ``sh`` subprocesses whose stdout the
``metric-lines`` adapter parses.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat import measure as measure_mod
from gymrat import sampling
from gymrat.errors import CommandError, GymratError
from gymrat.measure import MeasureOptions, measure
from gymrat.sampling import TargetSpec
from gymrat.targets import CleanupResult, WorktreeInfo, WorktreeRemovalFailure
from tests._git import git as _git
from tests._pipeline import install_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gymrat.adapters.types import WarnSink
    from gymrat.progress_events import ProgressEvent


def _options(
    *,
    target: str = "main",
    spec: TargetSpec | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    warn: WarnSink | None = None,
) -> MeasureOptions:
    resolved_spec = spec if spec is not None else TargetSpec(label=None, target=target)
    return MeasureOptions(
        bench="run",
        prepare=None,
        adapter="metric-lines",
        samples=3,
        timeout_seconds=1.0,
        config_metrics=None,
        config_kinds=None,
        target=resolved_spec,
        on_progress=on_progress,
        warn=warn,
    )


async def test_measure_when_target_benched_does_report_metric_median_and_spread(
    monkeypatch: pytest.MonkeyPatch,
):
    install_pipeline(monkeypatch, measure_mod, [[{"x": 10.0}, {"x": 20.0}, {"x": 30.0}]])

    result = await measure(_options(target="main"))

    assert result.metrics["x"].median == 20.0
    assert result.metrics["x"].spread == 50.0
    assert result.rounds == ({"x": 10.0}, {"x": 20.0}, {"x": 30.0})
    assert result.samples == 3
    assert result.adapter == "metric-lines"
    assert result.label == "main"


async def test_measure_when_explicit_label_given_does_use_it(monkeypatch: pytest.MonkeyPatch):
    install_pipeline(monkeypatch, measure_mod, [[{"x": 1.0}]])

    result = await measure(_options(spec=TargetSpec(label="custom", target="whatever")))

    assert result.label == "custom"


async def test_measure_when_rounds_report_different_metrics_does_median_over_present_rounds(
    monkeypatch: pytest.MonkeyPatch,
):
    install_pipeline(monkeypatch, measure_mod, [[{"x": 1.0}, {"y": 2.0}, {"x": 3.0}]])

    result = await measure(_options())

    assert result.metrics["x"].median == 2.0
    assert result.metrics["y"].median == 2.0
    assert result.rounds == ({"x": 1.0}, {"y": 2.0}, {"x": 3.0})


async def test_measure_when_no_metrics_does_raise_gymrat_error(monkeypatch: pytest.MonkeyPatch):
    install_pipeline(monkeypatch, measure_mod, [[{}, {}]])

    with pytest.raises(GymratError, match="No metrics found in benchmark output"):
        await measure(_options())


async def test_measure_when_metric_median_zero_does_report_no_spread(
    monkeypatch: pytest.MonkeyPatch,
):
    install_pipeline(monkeypatch, measure_mod, [[{"x": -1.0}, {"x": 0.0}, {"x": 1.0}]])

    result = await measure(_options())

    assert result.metrics["x"].median == 0.0
    assert result.metrics["x"].spread is None


async def test_measure_when_cleanup_reports_removals_does_map_worktree_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    dirty = CleanupResult(
        removed=1,
        failures=(WorktreeRemovalFailure(dir="/tmp/wt", error="busy"),),
        prune_error="could not prune",
    )
    install_pipeline(monkeypatch, measure_mod, [[{"x": 1.0}]], cleanup=dirty)

    result = await measure(_options())

    assert result.worktrees_removed == 1
    assert result.worktrees_left_behind == (WorktreeRemovalFailure(dir="/tmp/wt", error="busy"),)
    assert result.worktree_prune_error == "could not prune"


async def test_measure_when_progress_and_warn_given_does_forward_to_sampling(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = install_pipeline(monkeypatch, measure_mod, [[{"x": 1.0}]])
    steps: list[object] = []
    warnings: list[str] = []
    options = _options(on_progress=steps.append, warn=warnings.append)

    await measure(options)

    forwarded = captured.options
    assert forwarded is not None
    assert forwarded.on_progress is options.on_progress
    assert forwarded.warn is options.warn
    assert forwarded.bench == "run"
    assert forwarded.prepare is None
    assert forwarded.samples == 3
    assert forwarded.timeout_seconds == 1.0


# --- End-to-end tests (real subprocesses, POSIX only) ---

_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell")

_EMIT_ONE = "#!/bin/sh\necho 'METRIC x=1'\n"
_FAIL = "#!/bin/sh\nexit 1\n"


def _commit_bench(repo: str, script: str) -> None:
    (Path(repo) / "bench.sh").write_text(script, encoding="utf-8")
    _git(repo, "add", "bench.sh")
    _git(repo, "commit", "-m", "add bench")


def _e2e_options(target: str) -> MeasureOptions:
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


@_posix_only
async def test_measure_when_in_place_target_does_bench_without_worktree(
    create_scratch_repo: Callable[[], str],
    create_in_place_target_dir: Callable[[str, str, str], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    repo = create_scratch_repo()
    target_dir = create_in_place_target_dir(repo, "bench", _EMIT_ONE)
    monkeypatch.chdir(repo)

    result = await measure(_e2e_options(target_dir))

    assert result.metrics["x"].median == 1.0
    assert result.worktrees_removed == 0
    assert list_worktree_dirs(repo, include_main=False) == []


@_posix_only
async def test_measure_when_ref_target_does_bench_in_worktree_and_sweep(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    repo = create_scratch_repo()
    _commit_bench(repo, _EMIT_ONE)
    monkeypatch.chdir(repo)

    result = await measure(_e2e_options("HEAD"))

    assert result.metrics["x"].median == 1.0
    assert result.worktrees_removed >= 1
    assert list_worktree_dirs(repo, include_main=False) == []


@_posix_only
async def test_measure_when_bench_fails_does_reject_and_remove_worktrees(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    repo = create_scratch_repo()
    _commit_bench(repo, _FAIL)
    monkeypatch.chdir(repo)

    with pytest.raises(CommandError):
        await measure(_e2e_options("HEAD"))

    assert list_worktree_dirs(repo, include_main=False) == []


@_posix_only
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
        await measure(_e2e_options("HEAD"))

    message = str(caught.value)
    assert "/tmp/stranded-wt" in message
    assert "bench command failed" in message
