"""Tests for the ``measure`` orchestrator.

Unit tests stub the sampling pipeline to pin how ``measure`` assembles a
:class:`MeasurementResult` from canned per-round samples. End-to-end tests
drive real scratch repos and shell bench scripts through the full pipeline —
target resolution, worktree lifecycle, and ``sh`` subprocesses whose stdout the
``metric-lines`` adapter parses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat import measure as measure_mod
from gymrat.config import KindEntry, MetricEntry
from gymrat.errors import CommandError
from gymrat.measure import MeasureOptions, measure
from gymrat.progress_events import PrepareStarted
from gymrat.sampling import TargetSpec
from gymrat.utils import warn_to_stderr
from tests._git import (
    EMIT_ONE_BENCH,
    create_in_place_target_dir,
    list_worktree_dirs,
    write_committed_bench,
)
from tests._pipeline import DIRTY_RESULT, install_pipeline, run_options
from tests._platform import needs_posix_shell

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.progress_events import ProgressEvent
    from gymrat.utils import WarnSink


def _options(
    *,
    target: str = "main",
    on_progress: Callable[[ProgressEvent], None] | None = None,
    warn: WarnSink = warn_to_stderr,
    config_metrics: dict[str, MetricEntry] | None = None,
    config_kinds: dict[str, KindEntry] | None = None,
) -> MeasureOptions:
    return MeasureOptions(
        run=run_options(
            samples=3,
            on_progress=on_progress,
            warn=warn,
            config_metrics=config_metrics,
            config_kinds=config_kinds,
        ),
        target=TargetSpec(label=None, target=target),
    )


async def test_measure_when_target_benched_does_assemble_the_result(
    monkeypatch: pytest.MonkeyPatch,
):
    install_pipeline(
        monkeypatch,
        measure_mod,
        [[{"x": 10.0, "y": 4.0}, {"x": 20.0}, {"x": 30.0, "y": 6.0}]],
        cleanup=DIRTY_RESULT,
    )

    result = await measure(_options(target="main"))

    assert set(result.metrics) == {"x", "y"}
    assert result.metrics["x"].median == 20.0
    assert result.metrics["x"].spread == 50.0
    assert result.metrics["y"].median == 5.0
    assert result.rounds == ({"x": 10.0, "y": 4.0}, {"x": 20.0}, {"x": 30.0, "y": 6.0})
    assert result.samples == 3
    assert result.adapter == "metric-lines"
    assert result.label == "main"
    assert result.worktrees_removed == DIRTY_RESULT.removed
    assert result.worktrees_left_behind == DIRTY_RESULT.failures
    assert result.worktree_prune_error == DIRTY_RESULT.prune_error


async def test_measure_when_sampling_callbacks_given_does_deliver_what_the_pipeline_emits(
    monkeypatch: pytest.MonkeyPatch,
):
    event = PrepareStarted(label="main", at_ms=0.0)
    install_pipeline(
        monkeypatch, measure_mod, [[{"x": 1.0}]], progress_event=event, warning="banana"
    )
    steps: list[ProgressEvent] = []
    warnings: list[str] = []

    await measure(_options(on_progress=steps.append, warn=warnings.append))

    assert (steps, warnings) == ([event], ["banana"])


async def test_measure_when_config_overrides_given_does_apply_them_to_the_result(
    monkeypatch: pytest.MonkeyPatch,
):
    install_pipeline(monkeypatch, measure_mod, [[{"x": 1.0}, {"x": 2.0}]])
    kinds = {"other": KindEntry(gating=False)}

    result = await measure(
        _options(config_metrics={"x": MetricEntry(direction="higher")}, config_kinds=kinds)
    )

    assert result.metrics["x"].meta.direction == "higher"
    assert result.metrics["x"].meta.gating is False
    assert result.config_kinds == kinds


# ---------------------------------------------------------------------------
# End-to-end tests (real subprocesses, POSIX only)
# ---------------------------------------------------------------------------


_FAIL = "#!/bin/sh\nexit 1\n"


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
    assert result.worktrees_removed == 0
    assert list_worktree_dirs(repo, include_main=False) == []


@needs_posix_shell
async def test_measure_when_ref_target_does_bench_in_a_disposable_worktree(
    repo: str,
):
    write_committed_bench(repo, EMIT_ONE_BENCH)

    result = await measure(_e2e_options("HEAD"))

    assert result.metrics["x"].median == 1.0
    assert result.worktrees_removed >= 1
    assert list_worktree_dirs(repo, include_main=False) == []


@needs_posix_shell
async def test_measure_when_bench_fails_does_fail_with_nothing_on_disk(
    repo: str,
):
    write_committed_bench(repo, _FAIL)

    with pytest.raises(CommandError, match=r'^bench command failed \("'):
        await measure(_e2e_options("HEAD"))

    assert list_worktree_dirs(repo, include_main=False) == []
