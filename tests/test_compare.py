"""Tests for the ``compare`` orchestrator.

Unit tests stub the sampling pipeline to pin how ``compare`` assembles a
:class:`ComparisonResult`: the metric union across targets, the star topology
that judges every candidate against the shared baseline, and the
candidate-paired restriction on the displayed baseline median. The run wiring
``compare`` shares with ``measure`` (cleanup metadata, sampling callbacks, config
overrides) is pinned once for both in :mod:`tests.test_measure`. End-to-end tests
drive real scratch repos and shell bench scripts through the full pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat import compare as compare_mod
from gymrat.compare import CompareOptions, compare
from gymrat.errors import CommandError, GymratError
from gymrat.model import DEFAULT_UNSTABLE_NOISE_PCT
from gymrat.sampling import TargetSpec
from gymrat.utils import warn_to_stderr
from tests._git import (
    EMIT_ONE_BENCH,
    FAILING_BENCH,
    emit_bench,
    list_worktree_dirs,
    write_committed_bench,
)
from tests._git import run_git as _git
from tests._pipeline import install_pipeline, run_options
from tests._platform import needs_posix_shell

if TYPE_CHECKING:
    from gymrat.utils import WarnSink


def _options(
    *,
    baseline: TargetSpec | None = None,
    candidate_targets: tuple[str, ...] = ("cand",),
    candidates: list[TargetSpec] | None = None,
    warn: WarnSink = warn_to_stderr,
) -> CompareOptions:
    resolved_baseline = baseline if baseline is not None else TargetSpec(label=None, target="base")
    resolved_candidates = (
        candidates
        if candidates is not None
        else [TargetSpec(label=None, target=name) for name in candidate_targets]
    )
    return CompareOptions(
        run=run_options(samples=4, warn=warn),
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
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"items": 1.0}, {"items": 2.0}]
    candidate = [{"y": 3.0}, {"y": 4.0}]
    install_pipeline(monkeypatch, compare_mod, [baseline, candidate])

    result = await compare(_options())

    assert list(result.metrics.keys()) == ["items", "y"]
    assert result.metrics["items"].baseline_median == 1.5
    assert result.metrics["y"].candidates[0].median == 3.5


async def test_compare_when_a_round_is_one_sided_does_send_the_dropped_window_warning_to_the_sink(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"x": 1.0}, {"x": 2.0}, {}]
    candidate = [{"x": 3.0}, {"x": 4.0}, {"x": 5.0}]
    install_pipeline(monkeypatch, compare_mod, [baseline, candidate])
    warnings: list[str] = []

    await compare(_options(warn=warnings.append))

    assert [warning.split(": ", 1)[0] for warning in warnings] == ["x"]


async def test_compare_when_baseline_round_unpaired_does_median_each_side_over_paired_rounds(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = [{"x": 1.0}, {"x": 2.0}, {"x": 100.0}]
    candidate = [{"x": 10.0}, {"x": 20.0}]
    install_pipeline(monkeypatch, compare_mod, [baseline, candidate])

    result = await compare(_options())

    # Round 2 (value 100) has no candidate at the same index, so it is dropped
    # from the displayed baseline median; over all three rounds the median is 2.0.
    assert result.metrics["x"].baseline_median == 1.5
    assert result.metrics["x"].candidates[0].median == 15.0


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


# ---------------------------------------------------------------------------
# End-to-end tests (real subprocesses, POSIX only)
# ---------------------------------------------------------------------------


def _commit_bench(repo: str, value: int) -> None:
    write_committed_bench(repo, emit_bench(value), message=f"bench emits {value}")


def _e2e_options(baseline: str, candidate: str) -> CompareOptions:
    return CompareOptions(
        run=run_options(samples=3, bench="sh bench.sh", prepare=None, timeout_seconds=30.0),
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

    assert result.baseline_label == "main"
    assert result.candidates[0].label == "candidate"
    assert result.metrics["x"].baseline_median == 1.0
    assert result.metrics["x"].candidates[0].median == 2.0
    assert result.worktrees_removed >= 2
    assert list_worktree_dirs(repo, include_main=False) == []


@pytest.mark.parametrize(
    ("baseline_bench", "candidate_bench", "header"),
    [
        pytest.param(
            FAILING_BENCH, emit_bench(2), 'bench command failed (old, "main"', id="baseline-fails"
        ),
        pytest.param(
            EMIT_ONE_BENCH,
            FAILING_BENCH,
            'bench command failed (new, "candidate"',
            id="candidate-fails",
        ),
    ],
)
@needs_posix_shell
async def test_compare_when_a_bench_fails_does_name_the_target_by_its_comparison_side(
    repo: str,
    baseline_bench: str,
    candidate_bench: str,
    header: str,
):
    write_committed_bench(repo, baseline_bench, message="baseline bench")
    _git(["switch", "-c", "candidate"], repo)
    write_committed_bench(repo, candidate_bench, message="candidate bench")
    _git(["switch", "main"], repo)

    with pytest.raises(CommandError) as caught:
        await compare(_e2e_options("main", "candidate"))

    assert str(caught.value).startswith(header)


@needs_posix_shell
async def test_compare_when_candidate_unresolvable_does_fail_with_nothing_on_disk(
    repo: str,
):
    _commit_bench(repo, 1)

    with pytest.raises(GymratError, match="no-such-ref"):
        await compare(_e2e_options("main", "no-such-ref"))

    assert list_worktree_dirs(repo, include_main=False) == []
