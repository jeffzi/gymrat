"""Behavioral tests for ``iterate_session``: measuring one edit of an open session.

The one boundary these tests mock is sampling, which shells out to the
consumer's bench script; everything downstream of it — verdicts, aggregation,
the record, the report — runs for real. Sessions are laid down on disk with the
real record builders against a throwaway repository, so the suite is
order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from gymrat_py.config import MetricEntry, StopConfig
from gymrat_py.errors import GymratError, hint_of
from gymrat_py.loop.iterate import LoopStopError, iterate_session
from gymrat_py.sampling import TargetContext
from gymrat_py.session import (
    PairedSamples,
    read_records,
    session_jsonl_path,
)
from gymrat_py.targets import InPlaceTarget
from tests.loop._iterate import (
    as_logged,
    baseline_rounds,
    improved_rounds,
    install_collect_samples,
    iteration,
    last_iteration_of,
    resolved_config,
    sampling_call,
    session_record,
    stub_runs,
    stub_samples,
    trimmed_report_lines,
)
from tests.session._records import (
    committed_keep,
    discard_record,
    finalize_record,
    iteration_record,
    write_session_log,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.loop._iterate import CollectSamplesRecorder

ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def _on_target_iteration(seq: int):
    """The iteration numbered ``seq``, measured at or past the configured target."""
    return iteration_record(seq=seq, target_reached=True)


def _plain(report: str) -> str:
    """The report stripped of color, as a terminal's visible text would read."""
    return "\n".join(trimmed_report_lines(report))


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str]) -> str:
    return create_scratch_repo()


@pytest.fixture
def samples_mock(monkeypatch: pytest.MonkeyPatch):
    return install_collect_samples(monkeypatch)


@pytest.fixture
def settled(repo: str, samples_mock: CollectSamplesRecorder):
    """A settled session on disk — one kept iteration — with sampling stubbed improved."""
    write_session_log(repo, session_record(repo), (iteration(1), committed_keep(1)))
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())
    return repo


# ---------------------------------------------------------------------------
# refusing to measure
# ---------------------------------------------------------------------------


async def test_iterate_session_when_no_session_does_refuse_pointing_at_start(
    repo: str, samples_mock: CollectSamplesRecorder
):
    with pytest.raises(GymratError) as exc:
        await iterate_session(repo, resolved_config())

    assert "gymrat start" in (hint_of(exc.value) or "")
    assert samples_mock.call_count == 0


async def test_iterate_session_when_session_finalized_does_refuse_pointing_at_start(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(
        repo,
        session_record(repo),
        (iteration(1), committed_keep(1), finalize_record()),
    )

    with pytest.raises(GymratError) as exc:
        await iterate_session(repo, resolved_config())

    assert "gymrat start" in (hint_of(exc.value) or "")
    assert samples_mock.call_count == 0


async def test_iterate_session_when_last_iteration_unsettled_does_refuse_naming_both_paths(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(repo, session_record(repo), (iteration(1),))

    with pytest.raises(GymratError) as exc:
        await iterate_session(repo, resolved_config())

    hint = hint_of(exc.value) or ""
    assert "gymrat keep" in hint
    assert "gymrat discard" in hint
    assert samples_mock.call_count == 0


# ---------------------------------------------------------------------------
# a configured stop condition already met
# ---------------------------------------------------------------------------


async def test_iterate_session_when_max_iterations_reached_does_refuse_without_measuring(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(
        repo,
        session_record(repo),
        (iteration(1), committed_keep(1), iteration(2), committed_keep(2)),
    )
    stub_runs(samples_mock, repo, [])

    with pytest.raises(LoopStopError) as exc:
        await iterate_session(repo, resolved_config(stop=StopConfig(max_iterations=2)))

    assert "max iterations" in str(exc.value)
    assert "2" in str(exc.value)
    assert samples_mock.call_count == 0
    assert len(read_records(session_jsonl_path(repo))) == 5


async def test_iterate_session_when_target_kept_does_refuse_without_measuring(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(repo, session_record(repo), (_on_target_iteration(1), committed_keep(1)))
    stub_runs(samples_mock, repo, [])

    with pytest.raises(LoopStopError) as exc:
        await iterate_session(
            repo, resolved_config(primary="total_ms", stop=StopConfig(target_value=95))
        )

    assert "target reached" in str(exc.value)
    assert samples_mock.call_count == 0
    assert len(read_records(session_jsonl_path(repo))) == 3


async def test_iterate_session_when_target_iteration_discarded_does_measure_again(
    repo: str, samples_mock: CollectSamplesRecorder
):
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())
    write_session_log(repo, session_record(repo), (_on_target_iteration(1), discard_record(1)))

    result = await iterate_session(
        repo, resolved_config(primary="total_ms", stop=StopConfig(target_value=95))
    )

    assert result.record.seq == 2


async def test_iterate_session_when_no_stop_configured_does_measure_past_a_kept_target(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(
        repo,
        session_record(repo),
        (_on_target_iteration(1), committed_keep(1), iteration(2), committed_keep(2)),
    )
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())

    result = await iterate_session(repo, resolved_config())

    assert result.record.seq == 3


# ---------------------------------------------------------------------------
# measuring a settled session
# ---------------------------------------------------------------------------


async def test_iterate_session_when_measuring_does_bench_both_worktrees_baseline_first(
    settled: str, samples_mock: CollectSamplesRecorder
):
    await iterate_session(settled, resolved_config())

    worktrees = session_record(settled).worktrees
    assert sampling_call(samples_mock, 0).targets == [
        TargetContext(
            target=InPlaceTarget(dir=worktrees.baseline),
            dir=worktrees.baseline,
            label="baseline",
            position="old",
        ),
        TargetContext(
            target=InPlaceTarget(dir=worktrees.experiment),
            dir=worktrees.experiment,
            label="experiment",
            position="new",
        ),
    ]


async def test_iterate_session_when_measuring_does_append_iteration_after_last_settled(
    settled: str, samples_mock: CollectSamplesRecorder
):
    await iterate_session(settled, resolved_config())

    record = last_iteration_of(settled)
    assert record.seq == 2
    assert ISO_PATTERN.match(record.at)
    assert record.samples == PairedSamples(
        experiment=tuple(improved_rounds()), baseline=tuple(baseline_rounds())
    )
    total = record.metrics["total_ms"]
    assert total.delta_pct == pytest.approx(-10, abs=1e-6)
    assert total.verdict == "improved"
    assert total.method == "permutation"
    assert total.p is not None
    assert total.noise_pct is not None
    assert total.gating is True
    assert total.confirmed is False
    alloc = record.metrics["alloc_bytes"]
    assert alloc.delta_pct == pytest.approx(-20, abs=1e-6)
    assert alloc.verdict == "improved"
    assert record.primary.kind == "geomean"
    assert record.primary.name is None
    assert record.primary.delta_pct == pytest.approx(-15.1472, abs=1e-3)
    assert record.outcome == "improved"
    assert record.target_reached is False


async def test_iterate_session_when_measuring_does_hand_back_the_record_it_appended(
    settled: str, samples_mock: CollectSamplesRecorder
):
    result = await iterate_session(settled, resolved_config())

    assert as_logged(result.record) == as_logged(last_iteration_of(settled))


async def test_iterate_session_when_primary_is_named_metric_does_read_it_alone(
    settled: str, samples_mock: CollectSamplesRecorder
):
    result = await iterate_session(settled, resolved_config(primary="total_ms"))

    assert result.record.primary.kind == "metric"
    assert result.record.primary.name == "total_ms"
    assert result.record.primary.delta_pct == pytest.approx(-10, abs=1e-6)


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        pytest.param(None, False, id="no-stop"),
        pytest.param(StopConfig(target_value=85), False, id="target-ahead"),
        pytest.param(StopConfig(target_value=95), True, id="target-met"),
    ],
)
async def test_iterate_session_when_target_configured_does_record_target_reached(
    settled: str, samples_mock: CollectSamplesRecorder, stop: StopConfig | None, expected: bool
):
    result = await iterate_session(settled, resolved_config(primary="total_ms", stop=stop))

    assert result.record.target_reached is expected


async def test_iterate_session_when_target_is_higher_is_better_does_read_the_other_side(
    settled: str, samples_mock: CollectSamplesRecorder
):
    resolved = resolved_config(
        primary="total_ms",
        stop=StopConfig(target_value=85),
        metrics={"total_ms": MetricEntry(direction="higher")},
    )

    result = await iterate_session(settled, resolved)

    assert result.record.target_reached is True


async def test_iterate_session_when_target_met_does_state_it_above_the_next_step(
    settled: str, samples_mock: CollectSamplesRecorder
):
    result = await iterate_session(
        settled, resolved_config(primary="total_ms", stop=StopConfig(target_value=95))
    )

    assert trimmed_report_lines(result.report)[-2] == "target reached — keep it"


@pytest.mark.parametrize(
    "stop",
    [
        pytest.param(StopConfig(target_value=85), id="target-ahead"),
        pytest.param(None, id="no-stop"),
    ],
)
async def test_iterate_session_when_target_not_met_does_leave_it_out_of_the_report(
    settled: str, samples_mock: CollectSamplesRecorder, stop: StopConfig | None
):
    result = await iterate_session(settled, resolved_config(primary="total_ms", stop=stop))

    assert "target reached" not in _plain(result.report)


async def test_iterate_session_when_measuring_does_open_report_on_the_loop_header(
    settled: str, samples_mock: CollectSamplesRecorder
):
    result = await iterate_session(settled, resolved_config())

    plain = _plain(result.report)
    assert plain.split("\n")[0] == "iteration 2 · experiment vs baseline · 10 paired samples"
    assert "total_ms" in plain
