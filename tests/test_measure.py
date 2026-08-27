"""Unit tests for the ``measure`` orchestrator with the sampling seam replayed.

The sampling pipeline (target resolution, worktree lifecycle, exec) is a system
boundary; these tests stub it so the assertions pin how ``measure`` assembles a
:class:`MeasurementResult` from canned per-round samples.
"""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest

from gymrat_py import measure as measure_mod
from gymrat_py import sampling
from gymrat_py.adapters.types import Adapter, WarnSink
from gymrat_py.errors import GymratError
from gymrat_py.measure import MeasureOptions, measure
from gymrat_py.progress_events import ProgressEvent
from gymrat_py.sampling import (
    SamplingOptions,
    TargetContext,
    TargetSamples,
    TargetSpec,
)
from gymrat_py.targets import CleanupResult, InPlaceTarget, WorktreeInfo, WorktreeRemovalFailure

_CLEAN = CleanupResult(removed=0, failures=(), prune_error=None)


@dataclass
class _CapturedCall:
    """The ``SamplingOptions`` and contexts the stubbed collector was handed."""

    options: SamplingOptions | None = None
    contexts: list[TargetContext] | None = None


def _install_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    samples: list[dict[str, float]],
    cleanup: CleanupResult = _CLEAN,
) -> _CapturedCall:
    """Replace resolution, collection, and worktree cleanup with in-memory stubs.

    ``resolve_target`` yields an in-place target rooted at the spec's raw target
    (so the derived label is that string's basename), ``collect_samples`` echoes
    the built context back paired with ``samples``, and the cleanup seam returns
    ``cleanup`` unchanged. Returns the ``SamplingOptions`` and contexts the
    orchestrator handed the collector.
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
        return [TargetSamples(ctx=contexts[0], samples=samples)]

    def fake_install(cleanup_cb: Callable[[], None]) -> Callable[[], None]:
        return lambda: None

    def fake_cleanup_worktrees(worktrees: Sequence[WorktreeInfo], repo_dir: str) -> CleanupResult:
        return cleanup

    monkeypatch.setattr(measure_mod, "resolve_target", fake_resolve_target)
    monkeypatch.setattr(measure_mod, "collect_samples", fake_collect)
    monkeypatch.setattr(sampling, "install_termination_cleanup", fake_install)
    monkeypatch.setattr(sampling, "cleanup_worktrees", fake_cleanup_worktrees)
    return captured


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
    _install_pipeline(monkeypatch, [{"x": 10.0}, {"x": 20.0}, {"x": 30.0}])

    result = await measure(_options(target="main"))

    assert result.metrics["x"].median == 20.0
    assert result.metrics["x"].spread == 50.0
    assert result.rounds == ({"x": 10.0}, {"x": 20.0}, {"x": 30.0})
    assert result.samples == 3
    assert result.adapter == "metric-lines"
    assert result.label == "main"


async def test_measure_when_explicit_label_given_does_use_it(monkeypatch: pytest.MonkeyPatch):
    _install_pipeline(monkeypatch, [{"x": 1.0}])

    result = await measure(_options(spec=TargetSpec(label="custom", target="whatever")))

    assert result.label == "custom"


async def test_measure_when_rounds_report_different_metrics_does_median_over_present_rounds(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_pipeline(monkeypatch, [{"x": 1.0}, {"y": 2.0}, {"x": 3.0}])

    result = await measure(_options())

    assert result.metrics["x"].median == 2.0
    assert result.metrics["y"].median == 2.0
    assert result.rounds == ({"x": 1.0}, {"y": 2.0}, {"x": 3.0})


async def test_measure_when_no_metrics_does_raise_gymrat_error(monkeypatch: pytest.MonkeyPatch):
    _install_pipeline(monkeypatch, [{}, {}])

    with pytest.raises(GymratError, match="No metrics found in benchmark output"):
        await measure(_options())


async def test_measure_when_metric_median_zero_does_report_no_spread(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_pipeline(monkeypatch, [{"x": -1.0}, {"x": 0.0}, {"x": 1.0}])

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
    _install_pipeline(monkeypatch, [{"x": 1.0}], cleanup=dirty)

    result = await measure(_options())

    assert result.worktrees_removed == 1
    assert result.worktrees_left_behind == (WorktreeRemovalFailure(dir="/tmp/wt", error="busy"),)
    assert result.worktree_prune_error == "could not prune"


async def test_measure_when_progress_and_warn_given_does_forward_to_sampling(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = _install_pipeline(monkeypatch, [{"x": 1.0}])
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
