"""Tests for the ``measure`` orchestrator.

Unit tests fake the bench process to pin how ``measure`` assembles a
:class:`MeasurementResult` from canned per-round samples. End-to-end tests
drive real scratch repos and shell bench scripts through the full pipeline —
target resolution, worktree lifecycle, and ``sh`` subprocesses whose stdout the
``metric-lines`` adapter parses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymrat.measure import MeasureOptions, measure
from gymrat.sampling import TargetSpec
from tests._git import (
    EMIT_ONE_BENCH,
    list_worktree_dirs,
    write_committed_bench,
)
from tests._pipeline import (
    install_pipeline,
    run_options,
)
from tests._platform import needs_posix_shell

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from pathlib import Path

    import pytest

    from gymrat.report.types import MeasurementResult
    from gymrat.sampling import RunOptions


def _measure_main(run: RunOptions) -> Awaitable[MeasurementResult]:
    return measure(MeasureOptions(run=run, target=TargetSpec(label=None, target="main")))


async def test_measure_when_target_benched_does_assemble_the_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    install_pipeline(
        monkeypatch,
        tmp_path,
        {"main": [{"x": 10.0, "y": 4.0}, {"x": 20.0}, {"x": 30.0, "y": 6.0}]},
    )

    result = await _measure_main(run_options(samples=3))

    assert result.metrics["x"].median == 20.0
    assert result.metrics["x"].spread == 50.0
    assert result.metrics["y"].median == 5.0


# ---------------------------------------------------------------------------
# End-to-end tests (real subprocesses, POSIX only)
# ---------------------------------------------------------------------------


def _e2e_options(target: str) -> MeasureOptions:
    return MeasureOptions(
        run=run_options(samples=2, bench="sh bench.sh", prepare=None, timeout_seconds=30.0),
        target=TargetSpec(label=None, target=target),
    )


@needs_posix_shell
async def test_measure_when_ref_target_does_bench_in_a_disposable_worktree(
    repo: str,
):
    write_committed_bench(repo, EMIT_ONE_BENCH)

    result = await measure(_e2e_options("HEAD"))

    assert result.metrics["x"].median == 1.0
    assert result.cleanup.removed >= 1
    assert list_worktree_dirs(repo, include_main=False) == []
