"""Tests for the ``compare`` orchestrator.

Unit tests stub the sampling pipeline to pin how ``compare`` assembles a
:class:`ComparisonResult`: the metric union across targets, the star topology
that judges every candidate against the shared baseline, and the
candidate-paired restriction on the displayed baseline median. End-to-end tests
drive real scratch repos and shell bench scripts through the full pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat import compare as compare_mod
from gymrat.adapters import get_adapter
from gymrat.compare import CompareOptions, compare
from gymrat.errors import GymratError
from gymrat.model import Observations
from gymrat.sampling import (
    TargetSpec,
    resolve_metric_meta_from_samples,
)
from gymrat.targets import CleanupResult, WorktreeRemovalFailure
from gymrat.verdict import compute_verdicts
from tests._git import git as _git
from tests._pipeline import install_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.adapters.types import WarnSink
    from gymrat.progress_events import ProgressEvent


def _options(
    *,
    baseline: TargetSpec | None = None,
    candidate_targets: tuple[str, ...] = ("cand",),
    candidates: list[TargetSpec] | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    warn: WarnSink | None = None,
) -> CompareOptions:
    resolved_baseline = baseline if baseline is not None else TargetSpec(label=None, target="base")
    resolved_candidates = (
        candidates
        if candidates is not None
        else [TargetSpec(label=None, target=name) for name in candidate_targets]
    )
    return CompareOptions(
        bench="run",
        prepare=None,
        adapter="metric-lines",
        samples=4,
        timeout_seconds=1.0,
        config_metrics=None,
        config_kinds=None,
        baseline=resolved_baseline,
        candidates=resolved_candidates,
        unstable_noise_pct=None,
        on_progress=on_progress,
        warn=warn,
    )


async def test_compare_when_candidates_judged_does_use_shared_baseline(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"x": 10.0}, {"x": 11.0}, {"x": 10.5}, {"x": 10.2}, {"x": 10.8}, {"x": 10.1}]
    cand_a = [{"x": 20.0}, {"x": 21.0}, {"x": 20.5}, {"x": 20.2}, {"x": 20.8}, {"x": 20.1}]
    cand_b = [{"x": 5.0}, {"x": 5.1}, {"x": 4.9}, {"x": 5.2}, {"x": 4.8}, {"x": 5.05}]
    install_pipeline(monkeypatch, compare_mod, [baseline, cand_a, cand_b])

    result = await compare(_options(candidate_targets=("a", "b")))

    meta = resolve_metric_meta_from_samples(
        [baseline, cand_a, cand_b], None, get_adapter("metric-lines"), None
    )
    expected_a = compute_verdicts(
        Observations.from_rounds(baseline), Observations.from_rounds(cand_a), meta
    )["x"]
    expected_b = compute_verdicts(
        Observations.from_rounds(baseline), Observations.from_rounds(cand_b), meta
    )["x"]
    assert result.metrics["x"].candidates[0].verdict == expected_a
    assert result.metrics["x"].candidates[1].verdict == expected_b


async def test_compare_when_metric_on_one_side_only_does_include_union_in_order(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"a": 1.0}, {"a": 2.0}]
    candidate = [{"b": 3.0}, {"b": 4.0}]
    install_pipeline(monkeypatch, compare_mod, [baseline, candidate])

    result = await compare(_options())

    assert list(result.metrics.keys()) == ["a", "b"]


async def test_compare_when_metric_named_like_dict_method_does_treat_as_ordinary_key(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"items": 1.0}, {"items": 2.0}]
    candidate = [{"items": 3.0}, {"items": 4.0}]
    install_pipeline(monkeypatch, compare_mod, [baseline, candidate])

    result = await compare(_options())

    assert result.metrics["items"].baseline_median == 1.5


async def test_compare_when_no_metrics_anywhere_does_raise_gymrat_error(
    monkeypatch: pytest.MonkeyPatch,
):
    install_pipeline(monkeypatch, compare_mod, [[{}, {}], [{}, {}]])

    with pytest.raises(GymratError, match="No metrics found in benchmark output"):
        await compare(_options())


async def test_compare_when_baseline_round_unpaired_does_exclude_from_baseline_median(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"x": 1.0}, {"x": 2.0}, {"x": 100.0}]
    candidate = [{"x": 10.0}, {"x": 20.0}]
    install_pipeline(monkeypatch, compare_mod, [baseline, candidate])

    result = await compare(_options())

    # Round 2 (value 100) has no candidate at the same index, so it is dropped
    # from the displayed baseline median; over all three rounds the median is 2.0.
    assert result.metrics["x"].baseline_median == 1.5


async def test_compare_when_candidate_fully_paired_does_report_candidate_median(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"x": 1.0}, {"x": 2.0}]
    candidate = [{"x": 10.0}, {"x": 20.0}]
    install_pipeline(monkeypatch, compare_mod, [baseline, candidate])

    result = await compare(_options())

    assert result.metrics["x"].candidates[0].median == 15.0


async def test_compare_when_baseline_median_zero_does_omit_spread(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"x": -1.0}, {"x": 0.0}, {"x": 1.0}]
    candidate = [{"x": -1.0}, {"x": 0.0}, {"x": 1.0}]
    install_pipeline(monkeypatch, compare_mod, [baseline, candidate])

    result = await compare(_options())

    assert result.metrics["x"].baseline_median == 0.0
    assert result.metrics["x"].baseline_spread is None


async def test_compare_when_explicit_labels_given_does_flow_to_result(
    monkeypatch: pytest.MonkeyPatch,
):
    install_pipeline(monkeypatch, compare_mod, [[{"x": 1.0}, {"x": 2.0}], [{"x": 3.0}, {"x": 4.0}]])

    result = await compare(
        _options(
            baseline=TargetSpec(label="base-label", target="b"),
            candidates=[TargetSpec(label="cand-label", target="c")],
        )
    )

    assert result.baseline_label == "base-label"
    assert result.candidates[0].label == "cand-label"


async def test_compare_when_cleanup_reports_removals_does_map_worktree_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    dirty = CleanupResult(
        removed=2,
        failures=(WorktreeRemovalFailure(dir="/tmp/wt", error="busy"),),
        prune_error="could not prune",
    )
    install_pipeline(
        monkeypatch, compare_mod, [[{"x": 1.0}, {"x": 2.0}], [{"x": 3.0}, {"x": 4.0}]], dirty
    )

    result = await compare(_options())

    assert result.worktrees_removed == 2
    assert result.worktrees_left_behind == (WorktreeRemovalFailure(dir="/tmp/wt", error="busy"),)
    assert result.worktree_prune_error == "could not prune"
    assert result.samples == 4
    assert result.adapter == "metric-lines"


async def test_compare_when_progress_and_warn_given_does_forward_to_sampling(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = install_pipeline(
        monkeypatch, compare_mod, [[{"x": 1.0}, {"x": 2.0}], [{"x": 3.0}, {"x": 4.0}]]
    )
    steps: list[object] = []
    warnings: list[str] = []
    options = _options(on_progress=steps.append, warn=warnings.append)

    await compare(options)

    forwarded = captured.options
    assert forwarded is not None
    assert forwarded.on_progress is options.on_progress
    assert forwarded.warn is options.warn
    assert forwarded.bench == "run"
    assert forwarded.samples == 4
    assert forwarded.timeout_seconds == 1.0


# ---------------------------------------------------------------------------
# End-to-end tests (real subprocesses, POSIX only)
# ---------------------------------------------------------------------------

_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell")


def _commit_bench(repo: str, value: int) -> None:
    (Path(repo) / "bench.sh").write_text(f"#!/bin/sh\necho 'METRIC x={value}'\n", encoding="utf-8")
    _git(repo, "add", "bench.sh")
    _git(repo, "commit", "-m", f"bench emits {value}")


def _e2e_options(baseline: str, candidate: str) -> CompareOptions:
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


@_posix_only
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

    result = await compare(_e2e_options("main", "candidate"))

    assert result.baseline_label == "main"
    assert result.candidates[0].label == "candidate"
    assert result.metrics["x"].baseline_median == 1.0
    assert result.metrics["x"].candidates[0].median == 2.0
    assert result.worktrees_removed >= 2
    assert list_worktree_dirs(repo, include_main=False) == []


@_posix_only
async def test_compare_when_candidate_unresolvable_does_fail_with_nothing_on_disk(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    repo = create_scratch_repo()
    _commit_bench(repo, 1)
    monkeypatch.chdir(repo)

    with pytest.raises(GymratError, match="no-such-ref"):
        await compare(_e2e_options("main", "no-such-ref"))

    assert list_worktree_dirs(repo, include_main=False) == []
