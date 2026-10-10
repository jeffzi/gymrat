"""Tests for the ``measure`` orchestrator.

Unit tests stub the sampling pipeline to pin how ``measure`` assembles a
:class:`MeasurementResult` from canned per-round samples. They also pin, once
for both orchestrators, the run wiring ``measure`` shares with ``compare``:
cleanup metadata, sampling callbacks, and config overrides. End-to-end tests
drive real scratch repos and shell bench scripts through the full pipeline —
target resolution, worktree lifecycle, and ``sh`` subprocesses whose stdout the
``metric-lines`` adapter parses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat import compare as compare_mod
from gymrat import measure as measure_mod
from gymrat.compare import CompareOptions, compare
from gymrat.config import KindEntry, MetricEntry
from gymrat.errors import CommandError
from gymrat.measure import MeasureOptions, measure
from gymrat.model import DEFAULT_UNSTABLE_NOISE_PCT
from gymrat.progress_events import PrepareStarted
from gymrat.sampling import TargetSpec
from tests._git import (
    EMIT_ONE_BENCH,
    FAILING_BENCH,
    create_in_place_target_dir,
    list_worktree_dirs,
    write_committed_bench,
)
from tests._pipeline import DIRTY_RESULT, install_pipeline, run_options
from tests._platform import needs_posix_shell

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import ModuleType

    from gymrat.progress_events import ProgressEvent
    from gymrat.report.types import ComparisonResult, MeasurementResult
    from gymrat.sampling import RunOptions

    type Orchestrate = Callable[[RunOptions], Awaitable[MeasurementResult | ComparisonResult]]


def _measure_main(run: RunOptions) -> Awaitable[MeasurementResult]:
    return measure(MeasureOptions(run=run, target=TargetSpec(label=None, target="main")))


def _compare_base_to_cand(run: RunOptions) -> Awaitable[ComparisonResult]:
    return compare(
        CompareOptions(
            run=run,
            baseline=TargetSpec(label=None, target="base"),
            candidates=[TargetSpec(label=None, target="cand")],
            unstable_noise_pct=DEFAULT_UNSTABLE_NOISE_PCT,
        )
    )


_ORCHESTRATORS = [
    pytest.param(measure_mod, _measure_main, 1, id="measure"),
    pytest.param(compare_mod, _compare_base_to_cand, 2, id="compare"),
]
"""Each orchestrator, how to run it on a stubbed pipeline, and how many targets it samples."""


async def test_measure_when_target_benched_does_assemble_the_result(
    monkeypatch: pytest.MonkeyPatch,
):
    install_pipeline(
        monkeypatch,
        measure_mod,
        [[{"x": 10.0, "y": 4.0}, {"x": 20.0}, {"x": 30.0, "y": 6.0}]],
    )

    result = await _measure_main(run_options(samples=3))

    assert set(result.metrics) == {"x", "y"}
    assert result.metrics["x"].median == 20.0
    assert result.metrics["x"].spread == 50.0
    assert result.metrics["y"].median == 5.0
    assert result.rounds == ({"x": 10.0, "y": 4.0}, {"x": 20.0}, {"x": 30.0, "y": 6.0})
    assert result.label == "main"


# ---------------------------------------------------------------------------
# Run wiring shared with compare
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("orchestrator", "orchestrate", "targets"), _ORCHESTRATORS)
async def test_orchestrator_when_pipeline_completes_does_assemble_result_metadata(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator: ModuleType,
    orchestrate: Orchestrate,
    targets: int,
):
    install_pipeline(monkeypatch, orchestrator, [[{"x": 1.0}, {"x": 2.0}]] * targets, DIRTY_RESULT)

    result = await orchestrate(run_options(samples=3))

    assert result.cleanup == DIRTY_RESULT
    assert result.samples == 3
    assert result.adapter == "metric-lines"


@pytest.mark.parametrize(("orchestrator", "orchestrate", "targets"), _ORCHESTRATORS)
async def test_orchestrator_when_sampling_callbacks_given_does_deliver_what_the_pipeline_emits(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator: ModuleType,
    orchestrate: Orchestrate,
    targets: int,
):
    event = PrepareStarted(label="main", at_ms=0.0)
    install_pipeline(
        monkeypatch,
        orchestrator,
        [[{"x": 1.0}, {"x": 2.0}]] * targets,
        progress_event=event,
        warning="banana",
    )
    steps: list[ProgressEvent] = []
    warnings: list[str] = []

    await orchestrate(run_options(samples=2, on_progress=steps.append, warn=warnings.append))

    assert (steps, warnings) == ([event], ["banana"])


@pytest.mark.parametrize(("orchestrator", "orchestrate", "targets"), _ORCHESTRATORS)
async def test_orchestrator_when_config_overrides_given_does_apply_them_to_the_result(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator: ModuleType,
    orchestrate: Orchestrate,
    targets: int,
):
    install_pipeline(monkeypatch, orchestrator, [[{"x": 1.0}, {"x": 2.0}]] * targets)
    kinds = {"other": KindEntry(gating=False)}

    result = await orchestrate(
        run_options(
            samples=2, config_metrics={"x": MetricEntry(direction="higher")}, config_kinds=kinds
        )
    )

    assert result.metrics["x"].meta.direction == "higher"
    assert result.metrics["x"].meta.gating is False
    assert result.config_kinds == kinds


# ---------------------------------------------------------------------------
# End-to-end tests (real subprocesses, POSIX only)
# ---------------------------------------------------------------------------


def _e2e_options(target: str) -> MeasureOptions:
    return MeasureOptions(
        run=run_options(samples=2, bench="sh bench.sh", prepare=None, timeout_seconds=30.0),
        target=TargetSpec(label=None, target=target),
    )


@needs_posix_shell
async def test_measure_when_in_place_target_does_bench_without_worktree(
    repo: str,
):
    target_dir = create_in_place_target_dir(repo, "bench", EMIT_ONE_BENCH)

    result = await measure(_e2e_options(target_dir))

    assert result.metrics["x"].median == 1.0
    assert result.cleanup.removed == 0
    assert list_worktree_dirs(repo, include_main=False) == []


@needs_posix_shell
async def test_measure_when_ref_target_does_bench_in_a_disposable_worktree(
    repo: str,
):
    write_committed_bench(repo, EMIT_ONE_BENCH)

    result = await measure(_e2e_options("HEAD"))

    assert result.metrics["x"].median == 1.0
    assert result.cleanup.removed >= 1
    assert list_worktree_dirs(repo, include_main=False) == []


@needs_posix_shell
async def test_measure_when_bench_fails_does_fail_with_nothing_on_disk(
    repo: str,
):
    write_committed_bench(repo, FAILING_BENCH)

    with pytest.raises(CommandError, match=r'^bench command failed \("'):
        await measure(_e2e_options("HEAD"))

    assert list_worktree_dirs(repo, include_main=False) == []
