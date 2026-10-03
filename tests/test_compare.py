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
from gymrat.config.types import KindEntry, MetricEntry
from gymrat.errors import GymratError
from gymrat.model import DEFAULT_UNSTABLE_NOISE_PCT
from gymrat.sampling import (
    RunOptions,
    SamplingOptions,
    TargetSpec,
    resolve_metric_meta_from_samples,
)
from gymrat.targets import CleanupResult, WorktreeRemovalFailure
from gymrat.verdict import compute_verdicts
from gymrat.warn import warn_to_stderr
from tests._git import run_git as _git
from tests._pipeline import install_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.progress_events import ProgressEvent
    from gymrat.warn import WarnSink


def _options(
    *,
    baseline: TargetSpec | None = None,
    candidate_targets: tuple[str, ...] = ("cand",),
    candidates: list[TargetSpec] | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    warn: WarnSink = warn_to_stderr,
    config_metrics: dict[str, MetricEntry] | None = None,
    config_kinds: dict[str, KindEntry] | None = None,
) -> CompareOptions:
    resolved_baseline = baseline if baseline is not None else TargetSpec(label=None, target="base")
    resolved_candidates = (
        candidates
        if candidates is not None
        else [TargetSpec(label=None, target=name) for name in candidate_targets]
    )
    return CompareOptions(
        run=RunOptions(
            sampling=SamplingOptions(
                bench="run",
                prepare="prep",
                samples=4,
                timeout_seconds=1.0,
                on_progress=on_progress,
                warn=warn,
            ),
            adapter="metric-lines",
            config_metrics=config_metrics,
            config_kinds=config_kinds,
        ),
        baseline=resolved_baseline,
        candidates=resolved_candidates,
        unstable_noise_pct=DEFAULT_UNSTABLE_NOISE_PCT,
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
    expected_a = compute_verdicts(baseline, cand_a, meta)["x"]
    expected_b = compute_verdicts(baseline, cand_b, meta)["x"]
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


async def test_compare_when_metric_on_one_side_only_does_report_that_sides_own_median(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"a": 1.0}, {"a": 2.0}]
    candidate = [{"b": 3.0}, {"b": 4.0}]
    install_pipeline(monkeypatch, compare_mod, [baseline, candidate])

    result = await compare(_options())

    assert result.metrics["a"].baseline_median == 1.5
    assert result.metrics["b"].candidates[0].median == 3.5


async def test_compare_when_targets_sampled_does_place_baseline_old_and_candidates_new(
    monkeypatch: pytest.MonkeyPatch,
):
    samples = [{"x": 1.0}, {"x": 2.0}]
    captured = install_pipeline(monkeypatch, compare_mod, [samples, samples, samples])

    await compare(_options(candidate_targets=("one", "two")))

    assert captured.contexts is not None
    assert [ctx.position for ctx in captured.contexts] == ["old", "new", "new"]


async def test_compare_when_a_round_is_one_sided_does_send_the_dropped_window_warning_to_the_sink(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"x": 1.0}, {"x": 2.0}, {}]
    candidate = [{"x": 3.0}, {"x": 4.0}, {"x": 5.0}]
    install_pipeline(monkeypatch, compare_mod, [baseline, candidate])
    warnings: list[str] = []

    await compare(_options(warn=warnings.append))

    assert warnings == [
        "x: dropped 1 paired window(s) where the metric was measured on only one side"
    ]


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
    sampling = options.run.sampling
    assert forwarded.on_progress is sampling.on_progress
    assert forwarded.warn is sampling.warn
    assert forwarded.bench == "run"
    assert forwarded.prepare == "prep"
    assert forwarded.samples == 4
    assert forwarded.timeout_seconds == 1.0


async def test_compare_when_config_overrides_given_does_apply_them_to_the_result(
    monkeypatch: pytest.MonkeyPatch,
):
    install_pipeline(monkeypatch, compare_mod, [[{"x": 1.0}, {"x": 2.0}], [{"x": 3.0}, {"x": 4.0}]])
    kinds = {"other": KindEntry(gating=False)}

    result = await compare(
        _options(config_metrics={"x": MetricEntry(direction="higher")}, config_kinds=kinds)
    )

    assert result.metrics["x"].meta.direction == "higher"
    assert result.metrics["x"].meta.gating is False
    assert result.config_kinds == kinds


# ---------------------------------------------------------------------------
# End-to-end tests (real subprocesses, POSIX only)
# ---------------------------------------------------------------------------

_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell")


def _commit_bench(repo: str, value: int) -> None:
    (Path(repo) / "bench.sh").write_text(f"#!/bin/sh\necho 'METRIC x={value}'\n", encoding="utf-8")
    _git(["add", "bench.sh"], repo)
    _git(["commit", "-m", f"bench emits {value}"], repo)


def _e2e_options(baseline: str, candidate: str) -> CompareOptions:
    return CompareOptions(
        run=RunOptions(
            sampling=SamplingOptions(
                bench="sh bench.sh", prepare=None, samples=3, timeout_seconds=30.0
            ),
            adapter="metric-lines",
            config_metrics=None,
            config_kinds=None,
        ),
        baseline=TargetSpec(label=None, target=baseline),
        candidates=[TargetSpec(label=None, target=candidate)],
        unstable_noise_pct=DEFAULT_UNSTABLE_NOISE_PCT,
    )


@_posix_only
async def test_compare_when_two_refs_does_produce_comparison_and_sweep(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
):
    repo = create_scratch_repo()
    _commit_bench(repo, 1)
    _git(["switch", "-c", "candidate"], repo)
    _commit_bench(repo, 2)
    _git(["switch", "main"], repo)
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
