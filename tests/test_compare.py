"""Tests for the ``compare`` orchestrator.

Unit tests fake the bench processes to pin how ``compare`` assembles a
:class:`ComparisonResult`: the metric union across targets, the star topology
that judges every candidate against the shared baseline, and the
candidate-paired restriction on the displayed baseline median. Another test pins
that the sample-balance refusal counts the baseline among the targets. End-to-end
tests drive real scratch repos and shell bench scripts through the full pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.compare import CompareOptions, compare
from gymrat.errors import GymratError
from gymrat.model import DEFAULT_UNSTABLE_NOISE_PCT
from gymrat.sample_balance import require_balanced_samples
from gymrat.sampling import TargetSpec
from tests._git import (
    emit_bench,
    list_worktree_dirs,
    write_committed_bench,
)
from tests._git import run_git as _git
from tests._pipeline import install_pipeline, run_options
from tests._platform import needs_posix_shell

if TYPE_CHECKING:
    from pathlib import Path


def _options(
    *,
    samples: int,
    baseline: TargetSpec | None = None,
    candidate_targets: tuple[str, ...] = ("cand",),
    candidates: list[TargetSpec] | None = None,
) -> CompareOptions:
    resolved_baseline = baseline if baseline is not None else TargetSpec(label=None, target="base")
    resolved_candidates = (
        candidates
        if candidates is not None
        else [TargetSpec(label=None, target=name) for name in candidate_targets]
    )
    return CompareOptions(
        run=run_options(samples=samples),
        baseline=resolved_baseline,
        candidates=resolved_candidates,
        unstable_noise_pct=DEFAULT_UNSTABLE_NOISE_PCT,
    )


async def test_compare_when_candidates_judged_does_use_shared_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    baseline = [{"x": 10.0}, {"x": 11.0}, {"x": 10.5}, {"x": 10.2}, {"x": 10.8}, {"x": 10.1}]
    cand_a = [{"x": 20.0}, {"x": 21.0}, {"x": 20.5}, {"x": 20.2}, {"x": 20.8}, {"x": 20.1}]
    cand_b = [{"x": 5.0}, {"x": 5.1}, {"x": 4.9}, {"x": 5.2}, {"x": 4.8}, {"x": 5.05}]
    install_pipeline(monkeypatch, tmp_path, {"base": baseline, "a": cand_a, "b": cand_b})

    result = await compare(_options(samples=6, candidate_targets=("a", "b")))

    # Against the shared baseline (median 10.35) a moves about +97% and b about
    # -51%; judging b against a instead would put b near -75%.
    outcomes = [
        None if candidate.verdict is None else (candidate.verdict.verdict, candidate.verdict.delta)
        for candidate in result.metrics["x"].candidates
    ]
    assert outcomes == [
        ("regressed", pytest.approx(96.6, abs=0.1)),
        ("improved", pytest.approx(-51.4, abs=0.1)),
    ]
    assert [candidate.kinds[0].geomean.value for candidate in result.candidates] == pytest.approx(
        [96.6, -51.4], abs=0.1
    )


async def test_compare_when_metric_on_one_side_only_does_include_it_with_that_sides_median(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    baseline = [{"items": 1.0}, {"items": 2.0}]
    candidate = [{"y": 3.0}, {"y": 4.0}]
    install_pipeline(monkeypatch, tmp_path, {"base": baseline, "cand": candidate})

    result = await compare(_options(samples=2))

    assert result.metrics["items"].baseline_median == 1.5
    assert result.metrics["y"].candidates[0].median == 3.5


async def test_compare_when_baseline_round_unpaired_does_median_each_side_over_paired_rounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    baseline = [{"x": 1.0}, {"x": 2.0}, {"x": 100.0}, {"x": 200.0}]
    candidate = [{"x": 10.0}, {"x": 20.0}, {"y": 1.0}, {"y": 1.0}]
    install_pipeline(monkeypatch, tmp_path, {"base": baseline, "cand": candidate})

    result = await compare(_options(samples=4))

    # Rounds 2 and 3 report no candidate ``x``, so they are dropped from the
    # displayed baseline median; over all four rounds it would be 51.0.
    assert result.metrics["x"].baseline_median == 1.5
    assert result.metrics["x"].candidates[0].median == 15.0


def _balance_refusal(samples: int, targets: int) -> str:
    try:
        require_balanced_samples(samples, targets)
    except GymratError as error:
        return str(error)
    message = f"{samples} samples balance {targets} targets"
    raise AssertionError(message)


async def test_compare_when_samples_cannot_balance_the_baseline_and_candidates_does_refuse():
    expected = _balance_refusal(4, 3)
    options = CompareOptions(
        run=run_options(samples=4),
        baseline=TargetSpec(label=None, target="base"),
        candidates=[TargetSpec(label=None, target="a"), TargetSpec(label=None, target="b")],
        unstable_noise_pct=DEFAULT_UNSTABLE_NOISE_PCT,
    )

    with pytest.raises(GymratError) as caught:
        await compare(options)

    assert str(caught.value) == expected


# ---------------------------------------------------------------------------
# End-to-end tests (real subprocesses, POSIX only)
# ---------------------------------------------------------------------------


def _commit_bench(repo: str, value: int) -> None:
    write_committed_bench(repo, emit_bench(value), message=f"bench emits {value}")


def _e2e_options(baseline: str, candidate: str) -> CompareOptions:
    return CompareOptions(
        run=run_options(samples=4, bench="sh bench.sh", prepare=None, timeout_seconds=30.0),
        baseline=TargetSpec(label=None, target=baseline),
        candidates=[TargetSpec(label=None, target=candidate)],
        unstable_noise_pct=DEFAULT_UNSTABLE_NOISE_PCT,
    )


@needs_posix_shell
async def test_compare_when_two_refs_given_does_compare_them_in_disposable_worktrees(
    repo: str,
):
    _commit_bench(repo, 1)
    _git(["switch", "-c", "candidate"], repo)
    _commit_bench(repo, 2)
    _git(["switch", "main"], repo)

    result = await compare(_e2e_options("main", "candidate"))

    assert result.metrics["x"].baseline_median == 1.0
    assert result.metrics["x"].candidates[0].median == 2.0
    assert result.cleanup.removed >= 2
    assert list_worktree_dirs(repo, include_main=False) == []
