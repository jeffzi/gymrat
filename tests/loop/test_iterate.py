"""Behavioral tests for ``iterate_session``: measuring one edit of an open session.

The one boundary these tests mock is sampling, which shells out to the
consumer's bench script; everything downstream of it — verdicts, aggregation,
the record, the report — runs for real. Sessions are laid down on disk with the
real record builders against a throwaway repository, so the suite is
order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat.config import HooksConfig, MetricEntry, StopConfig
from gymrat.errors import GymratError, hint_of
from gymrat.loop.iterate import IterateOptions, LoopStopError, iterate_session
from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmStarted,
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    JudgeStarted,
    PassFinished,
    PassStarted,
    ProgressEvent,
)
from gymrat.sampling import SamplingOptions, TargetContext, TargetSamples
from gymrat.session import (
    PairedSamples,
    read_records,
    session_jsonl_path,
)
from gymrat.targets import InPlaceTarget
from tests.loop._hooks import HookScripts
from tests.loop._iterate import (
    BASELINE_BYTES,
    BASELINE_MS,
    PairedRun,
    as_logged,
    baseline_rounds,
    improved_rounds,
    install_collect_samples,
    iteration,
    last_iteration_of,
    resolved_config,
    rounds,
    sampling_call,
    scaled,
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
    from collections.abc import Callable, Sequence

    from tests.loop._iterate import CollectSamplesRecorder

ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

#: The confirm-rerun template a consumer configures when their bench can be narrowed.
FILTER = "npm run bench -- --filter {names}"


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


async def test_iterate_session_when_color_false_does_suppress_ansi_in_report(
    settled: str, samples_mock: CollectSamplesRecorder
):
    result = await iterate_session(settled, resolved_config(), color=False)

    assert "\x1b[" not in result.report


async def test_iterate_session_when_color_true_does_emit_ansi_in_report(
    settled: str, samples_mock: CollectSamplesRecorder
):
    result = await iterate_session(settled, resolved_config(), color=True)

    assert "\x1b[" in result.report


async def test_iterate_session_when_measuring_does_open_report_on_the_loop_header(
    settled: str, samples_mock: CollectSamplesRecorder
):
    result = await iterate_session(settled, resolved_config())

    plain = _plain(result.report)
    assert plain.split("\n")[0] == "iteration 2 · experiment vs baseline · 10 paired samples"
    assert "total_ms" in plain


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# progress events emitted by iterate_session
# ---------------------------------------------------------------------------


async def test_iterate_session_when_hooks_configured_does_emit_hook_events(
    repo: str, samples_mock: CollectSamplesRecorder
):
    experiment_dir = session_record(repo).worktrees.experiment
    _ensure_dir(experiment_dir)
    hooks = HookScripts(repo, experiment_dir)
    write_session_log(repo, session_record(repo), (iteration(1), committed_keep(1)))
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())
    events: list[ProgressEvent] = []
    config = resolved_config(
        hooks=HooksConfig(before=hooks.printing("hi"), after=hooks.printing("bye"))
    )

    await iterate_session(repo, config, options=IterateOptions(on_progress=events.append))

    hook_events = [e for e in events if isinstance(e, (HookStarted, HookFinished))]
    assert len(hook_events) == 4
    assert isinstance(hook_events[0], HookStarted)
    assert hook_events[0].stage == "before"
    assert isinstance(hook_events[1], HookFinished)
    assert hook_events[1].stage == "before"
    assert isinstance(hook_events[2], HookStarted)
    assert hook_events[2].stage == "after"
    assert isinstance(hook_events[3], HookFinished)
    assert hook_events[3].stage == "after"


async def test_iterate_session_when_no_hooks_configured_does_emit_no_hook_events(
    settled: str, samples_mock: CollectSamplesRecorder
):
    events: list[ProgressEvent] = []

    await iterate_session(
        settled, resolved_config(), options=IterateOptions(on_progress=events.append)
    )

    hook_events = [e for e in events if isinstance(e, (HookStarted, HookFinished))]
    assert hook_events == []


async def test_iterate_session_when_measuring_does_emit_judge_started_after_the_bench_passes(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    """The judge phase opens when the verdict is computed, not while the bench still runs."""
    write_session_log(repo, session_record(repo), (iteration(1), committed_keep(1)))
    worktrees = session_record(repo).worktrees
    by_dir = {worktrees.experiment: improved_rounds(), worktrees.baseline: baseline_rounds()}

    async def sample_reporting_passes(
        adapter: object,
        targets: Sequence[TargetContext],
        options: SamplingOptions,
        abort: object,
    ) -> list[TargetSamples]:
        """Stand in for the bench, reporting one pass per target the way sampling does."""
        forward = options.on_progress
        assert forward is not None, "iterate_session should forward sampling progress"
        contexts = list(targets)
        collected: list[TargetSamples] = []
        for index, ctx in enumerate(contexts):
            for event_type in (PassStarted, PassFinished):
                forward(
                    event_type(
                        round=1,
                        total_rounds=1,
                        target_index=index,
                        target_count=len(contexts),
                        label=ctx.label,
                        at_ms=0,
                    )
                )
            collected.append(TargetSamples(ctx=ctx, samples=by_dir[ctx.dir]))
        return collected

    monkeypatch.setattr("gymrat.loop.iterate.collect_samples", sample_reporting_passes)
    events: list[ProgressEvent] = []

    await iterate_session(
        repo, resolved_config(), options=IterateOptions(on_progress=events.append)
    )

    judge_started = [e for e in events if isinstance(e, JudgeStarted)]
    assert len(judge_started) == 1
    event_types = [type(e).__name__ for e in events]
    started_idx = event_types.index("JudgeStarted")
    finished_idx = event_types.index("JudgeFinished")
    last_pass_idx = max(i for i, name in enumerate(event_types) if name == "PassFinished")
    assert last_pass_idx < started_idx
    assert started_idx < finished_idx


async def test_iterate_session_when_measuring_does_emit_judge_finished(
    settled: str, samples_mock: CollectSamplesRecorder
):
    events: list[ProgressEvent] = []

    await iterate_session(
        settled, resolved_config(), options=IterateOptions(on_progress=events.append)
    )

    judge_events = [e for e in events if isinstance(e, JudgeFinished)]
    assert len(judge_events) == 1
    judge = judge_events[0]
    assert judge.primary_delta_pct == pytest.approx(-15.1472, abs=1e-3)
    assert judge.regressed == ()


async def test_iterate_session_when_gating_regression_does_emit_judge_with_regressed_names(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(repo, session_record(repo))
    regressed_rounds = rounds(scaled(BASELINE_MS, 1.1), scaled(BASELINE_BYTES, 1.1))
    stub_samples(samples_mock, repo, regressed_rounds, baseline_rounds())
    events: list[ProgressEvent] = []

    await iterate_session(
        repo, resolved_config(), options=IterateOptions(on_progress=events.append)
    )

    judge_events = [e for e in events if isinstance(e, JudgeFinished)]
    assert len(judge_events) == 1
    assert set(judge_events[0].regressed) == {"total_ms", "alloc_bytes"}


async def test_iterate_session_when_confirmation_triggers_does_emit_confirm_events(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(repo, session_record(repo))
    regressed_rounds = rounds(scaled(BASELINE_MS, 1.1), scaled(BASELINE_BYTES, 1.1))
    stub_runs(
        samples_mock,
        repo,
        [
            PairedRun(regressed_rounds, baseline_rounds()),
            PairedRun(regressed_rounds, baseline_rounds()),
        ],
    )
    events: list[ProgressEvent] = []

    await iterate_session(
        repo, resolved_config(filter=FILTER), options=IterateOptions(on_progress=events.append)
    )

    confirm_started = [e for e in events if isinstance(e, ConfirmStarted)]
    confirm_finished = [e for e in events if isinstance(e, ConfirmFinished)]
    assert len(confirm_started) == 1
    assert set(confirm_started[0].filtered_metrics or ()) == {"total_ms", "alloc_bytes"}
    assert len(confirm_finished) == 1
    assert confirm_finished[0].reproduced is True


async def test_iterate_session_when_confirmation_without_filter_does_report_no_narrowing(
    repo: str, samples_mock: CollectSamplesRecorder
):
    """Without a filter template the rerun is the whole bench, so no metric list is claimed."""
    write_session_log(repo, session_record(repo))
    regressed_rounds = rounds(scaled(BASELINE_MS, 1.1), scaled(BASELINE_BYTES, 1.1))
    stub_runs(
        samples_mock,
        repo,
        [
            PairedRun(regressed_rounds, baseline_rounds()),
            PairedRun(regressed_rounds, baseline_rounds()),
        ],
    )
    events: list[ProgressEvent] = []

    await iterate_session(
        repo, resolved_config(filter=None), options=IterateOptions(on_progress=events.append)
    )

    confirm_started = [e for e in events if isinstance(e, ConfirmStarted)]
    assert len(confirm_started) == 1
    assert confirm_started[0].filtered_metrics is None


async def test_iterate_session_when_no_confirmation_triggers_does_emit_no_confirm_events(
    settled: str, samples_mock: CollectSamplesRecorder
):
    events: list[ProgressEvent] = []

    await iterate_session(
        settled, resolved_config(), options=IterateOptions(on_progress=events.append)
    )

    confirm_events = [e for e in events if isinstance(e, (ConfirmStarted, ConfirmFinished))]
    assert confirm_events == []


async def test_iterate_session_when_confirmation_rerun_does_tag_pass_events_as_confirm(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(repo, session_record(repo))
    regressed_rounds = rounds(scaled(BASELINE_MS, 1.1), scaled(BASELINE_BYTES, 1.1))
    stub_runs(
        samples_mock,
        repo,
        [
            PairedRun(regressed_rounds, baseline_rounds()),
            PairedRun(regressed_rounds, baseline_rounds()),
        ],
    )
    events: list[ProgressEvent] = []

    await iterate_session(
        repo, resolved_config(filter=FILTER), options=IterateOptions(on_progress=events.append)
    )

    rerun_callback = samples_mock.calls[1].options.on_progress
    assert rerun_callback is not None
    probe_started = PassStarted(
        round=1, total_rounds=1, target_index=0, target_count=1, label="x", at_ms=0
    )
    probe_finished = PassFinished(
        round=1, total_rounds=1, target_index=0, target_count=1, label="x", at_ms=0
    )
    rerun_callback(probe_started)
    rerun_callback(probe_finished)
    pass_events = [
        e for e in events if isinstance(e, (PassStarted, PassFinished)) and e.phase == "confirm"
    ]
    assert len(pass_events) >= 2


async def test_iterate_session_when_measuring_does_emit_iteration_recorded(
    settled: str, samples_mock: CollectSamplesRecorder
):
    events: list[ProgressEvent] = []

    result = await iterate_session(
        settled, resolved_config(), options=IterateOptions(on_progress=events.append)
    )

    recorded_events = [e for e in events if isinstance(e, IterationRecorded)]
    assert len(recorded_events) == 1
    assert recorded_events[0].seq == result.record.seq
    assert recorded_events[0].outcome == "improved"


async def test_iterate_session_when_improved_does_order_events_without_confirmation(
    settled: str, samples_mock: CollectSamplesRecorder
):
    events: list[ProgressEvent] = []

    await iterate_session(
        settled, resolved_config(), options=IterateOptions(on_progress=events.append)
    )

    event_types = [type(e).__name__ for e in events]
    judge_started_idx = event_types.index("JudgeStarted")
    judge_idx = event_types.index("JudgeFinished")
    recorded_idx = event_types.index("IterationRecorded")
    assert judge_started_idx < judge_idx
    assert judge_idx < recorded_idx
    assert "ConfirmStarted" not in event_types
    assert "ConfirmFinished" not in event_types


async def test_iterate_session_when_hooks_and_confirmation_does_order_all_events(
    repo: str, samples_mock: CollectSamplesRecorder
):
    experiment_dir = session_record(repo).worktrees.experiment
    _ensure_dir(experiment_dir)
    hooks = HookScripts(repo, experiment_dir)
    write_session_log(repo, session_record(repo), (iteration(1), committed_keep(1)))
    regressed_rounds = rounds(scaled(BASELINE_MS, 1.1), scaled(BASELINE_BYTES, 1.1))
    stub_runs(
        samples_mock,
        repo,
        [
            PairedRun(regressed_rounds, baseline_rounds()),
            PairedRun(regressed_rounds, baseline_rounds()),
        ],
    )
    events: list[ProgressEvent] = []
    config = resolved_config(
        filter=FILTER,
        hooks=HooksConfig(before=hooks.printing("hi"), after=hooks.printing("bye")),
    )

    await iterate_session(repo, config, options=IterateOptions(on_progress=events.append))

    def stage_index(event_type: type[HookStarted | HookFinished], stage: str) -> int:
        return next(
            i for i, e in enumerate(events) if isinstance(e, event_type) and e.stage == stage
        )

    event_types = [type(e).__name__ for e in events]
    assert event_types.index("HookStarted") < event_types.index("HookFinished")
    before_finished_idx = stage_index(HookFinished, "before")
    judge_started_idx = event_types.index("JudgeStarted")
    judge_idx = event_types.index("JudgeFinished")
    confirm_started_idx = event_types.index("ConfirmStarted")
    confirm_finished_idx = event_types.index("ConfirmFinished")
    recorded_idx = event_types.index("IterationRecorded")
    after_started_idx = stage_index(HookStarted, "after")
    assert before_finished_idx < judge_started_idx
    assert judge_started_idx < judge_idx
    assert judge_idx < confirm_started_idx
    assert confirm_started_idx < confirm_finished_idx
    assert confirm_finished_idx < recorded_idx
    assert recorded_idx < after_started_idx
