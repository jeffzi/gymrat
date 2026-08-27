"""Unit tests for the ``compare`` orchestrator with the sampling seam replayed.

The sampling pipeline (target resolution, worktree lifecycle, exec) is a system
boundary; these tests stub it so the assertions pin how ``compare`` assembles a
:class:`ComparisonResult`: the metric union across targets, the star topology
that judges every candidate against the shared baseline, and the
candidate-paired restriction on the displayed baseline median.
"""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest

from gymrat_py import compare as compare_mod
from gymrat_py import sampling
from gymrat_py.adapters import get_adapter
from gymrat_py.adapters.types import Adapter, WarnSink
from gymrat_py.compare import CompareOptions, compare
from gymrat_py.errors import GymratError
from gymrat_py.model import Observations
from gymrat_py.progress_events import ProgressEvent
from gymrat_py.sampling import (
    SamplingOptions,
    TargetContext,
    TargetSamples,
    TargetSpec,
    resolve_metric_meta_from_samples,
)
from gymrat_py.targets import CleanupResult, InPlaceTarget, WorktreeInfo, WorktreeRemovalFailure
from gymrat_py.verdict import compute_verdicts

_CLEAN = CleanupResult(removed=0, failures=(), prune_error=None)


@dataclass
class _CapturedCall:
    """The ``SamplingOptions`` and contexts the stubbed collector was handed."""

    options: SamplingOptions | None = None
    contexts: list[TargetContext] | None = None


def _install_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    sample_sets: list[list[dict[str, float]]],
    cleanup: CleanupResult = _CLEAN,
) -> _CapturedCall:
    """Replace resolution, collection, and worktree cleanup with in-memory stubs.

    ``resolve_target`` yields an in-place target rooted at the spec's raw target
    (so the derived label is that string's basename), ``collect_samples`` echoes
    each built context back paired with the matching entry of ``sample_sets``
    (index 0 is the baseline, the rest candidates in order), and the cleanup seam
    returns ``cleanup`` unchanged. Returns the ``SamplingOptions`` and contexts
    the orchestrator handed the collector.
    """
    captured = _CapturedCall()

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

    monkeypatch.setattr(compare_mod, "resolve_target", fake_resolve_target)
    monkeypatch.setattr(compare_mod, "collect_samples", fake_collect)
    monkeypatch.setattr(sampling, "install_termination_cleanup", fake_install)
    monkeypatch.setattr(sampling, "cleanup_worktrees", fake_cleanup_worktrees)
    return captured


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
    _install_pipeline(monkeypatch, [baseline, cand_a, cand_b])

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
    _install_pipeline(monkeypatch, [baseline, candidate])

    result = await compare(_options())

    assert list(result.metrics.keys()) == ["a", "b"]


async def test_compare_when_metric_named_like_dict_method_does_treat_as_ordinary_key(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"items": 1.0}, {"items": 2.0}]
    candidate = [{"items": 3.0}, {"items": 4.0}]
    _install_pipeline(monkeypatch, [baseline, candidate])

    result = await compare(_options())

    assert result.metrics["items"].baseline_median == 1.5


async def test_compare_when_no_metrics_anywhere_does_raise_gymrat_error(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_pipeline(monkeypatch, [[{}, {}], [{}, {}]])

    with pytest.raises(GymratError, match="No metrics found in benchmark output"):
        await compare(_options())


async def test_compare_when_baseline_round_unpaired_does_exclude_from_baseline_median(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"x": 1.0}, {"x": 2.0}, {"x": 100.0}]
    candidate = [{"x": 10.0}, {"x": 20.0}]
    _install_pipeline(monkeypatch, [baseline, candidate])

    result = await compare(_options())

    # Round 2 (value 100) has no candidate at the same index, so it is dropped
    # from the displayed baseline median; over all three rounds the median is 2.0.
    assert result.metrics["x"].baseline_median == 1.5


async def test_compare_when_candidate_fully_paired_does_report_candidate_median(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"x": 1.0}, {"x": 2.0}]
    candidate = [{"x": 10.0}, {"x": 20.0}]
    _install_pipeline(monkeypatch, [baseline, candidate])

    result = await compare(_options())

    assert result.metrics["x"].candidates[0].median == 15.0


async def test_compare_when_baseline_median_zero_does_omit_spread(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"x": -1.0}, {"x": 0.0}, {"x": 1.0}]
    candidate = [{"x": -1.0}, {"x": 0.0}, {"x": 1.0}]
    _install_pipeline(monkeypatch, [baseline, candidate])

    result = await compare(_options())

    assert result.metrics["x"].baseline_median == 0.0
    assert result.metrics["x"].baseline_spread is None


async def test_compare_when_explicit_labels_given_does_flow_to_result(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_pipeline(monkeypatch, [[{"x": 1.0}, {"x": 2.0}], [{"x": 3.0}, {"x": 4.0}]])

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
    _install_pipeline(monkeypatch, [[{"x": 1.0}, {"x": 2.0}], [{"x": 3.0}, {"x": 4.0}]], dirty)

    result = await compare(_options())

    assert result.worktrees_removed == 2
    assert result.worktrees_left_behind == (WorktreeRemovalFailure(dir="/tmp/wt", error="busy"),)
    assert result.worktree_prune_error == "could not prune"
    assert result.samples == 4
    assert result.adapter == "metric-lines"


async def test_compare_when_progress_and_warn_given_does_forward_to_sampling(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = _install_pipeline(monkeypatch, [[{"x": 1.0}, {"x": 2.0}], [{"x": 3.0}, {"x": 4.0}]])
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
