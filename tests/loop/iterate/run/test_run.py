"""Behavioral tests for ``iterate_session``: measuring one edit of an open session.

The one boundary these tests mock is sampling, which shells out to the
consumer's bench script; everything downstream of it — verdicts, aggregation,
the record, the report — runs for real. Sessions are laid down on disk with the
real record builders against a throwaway repository, so the suite is
order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat.config import HooksConfig, MetricEntry, StopConfig
from gymrat.errors import GymratError
from gymrat.loop.iterate.run import (
    BudgetExceededError,
    IterateOptions,
    LoopStopError,
    iterate_session,
)
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
from gymrat.session import workspace as _workspace
from gymrat.session.budget import Budget
from gymrat.session.records import PairedSamples
from gymrat.targets import InPlaceTarget
from tests._config import resolved_config
from tests.loop.iterate._fixtures import (
    BASELINE_BYTES,
    BASELINE_MS,
    MALFORMED_LINE_WARNING,
    PairedRun,
    as_logged,
    baseline_rounds,
    bench_malformed_once,
    improved_rounds,
    last_iteration_of,
    plain_report,
    rounds,
    sampling_call,
    scaled,
    session_record,
    stub_runs,
    stub_samples,
    trimmed_report_lines,
)
from tests.loop.iterate._hooks import HookScripts
from tests.session.records._fixtures import (
    committed_keep,
    discard_record,
    finalize_record,
    iteration_record,
    log_records,
    write_session_log,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from syrupy.assertion import SnapshotAssertion

    from tests.loop.iterate._fixtures import CollectSamplesRecorder


#: The confirm-rerun template a consumer configures when their bench can be narrowed.
FILTER = "npm run bench -- --filter {names}"


def _on_target_iteration(seq: int):
    """The iteration numbered ``seq``, measured at or past the configured target."""
    return iteration_record(seq=seq, target_reached=True)


@pytest.fixture
def settled(repo: str, samples_mock: CollectSamplesRecorder) -> str:
    """A settled session on disk — one kept iteration — with sampling stubbed improved."""
    write_session_log(repo, session_record(repo), (iteration_record(seq=1), committed_keep(1)))
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

    assert "gymrat start" in (exc.value.hint or "")
    assert samples_mock.call_count == 0


async def test_iterate_session_when_session_finalized_does_refuse_pointing_at_start(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(
        repo,
        session_record(repo),
        (iteration_record(seq=1), committed_keep(1), finalize_record()),
    )

    with pytest.raises(GymratError) as exc:
        await iterate_session(repo, resolved_config())

    assert "gymrat start" in (exc.value.hint or "")
    assert samples_mock.call_count == 0


async def test_iterate_session_when_last_iteration_unsettled_does_refuse_naming_both_paths(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(repo, session_record(repo), (iteration_record(seq=1),))

    with pytest.raises(GymratError) as exc:
        await iterate_session(repo, resolved_config())

    assert str(exc.value) == "Iteration 1 has not been settled"
    assert exc.value.hint == "Run gymrat keep or gymrat discard before measuring the next edit."
    assert samples_mock.call_count == 0


async def test_iterate_session_when_last_iteration_unsettled_does_carry_unsettled_reason(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(repo, session_record(repo), (iteration_record(seq=1),))

    with pytest.raises(GymratError) as exc:
        await iterate_session(repo, resolved_config())

    assert exc.value.reason == "unsettled"


async def test_iterate_session_when_adapter_unknown_does_refuse_before_sampling(
    settled: str, samples_mock: CollectSamplesRecorder
):
    with pytest.raises(GymratError, match="banana"):
        await iterate_session(settled, resolved_config(adapter="banana"))

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
        (iteration_record(seq=1), committed_keep(1), iteration_record(seq=2), committed_keep(2)),
    )
    stub_runs(samples_mock, repo, [])

    with pytest.raises(LoopStopError) as exc:
        await iterate_session(repo, resolved_config(stop=StopConfig(max_iterations=2)))

    assert str(exc.value) == "Stop condition met: max iterations (2 of 2)"
    assert samples_mock.call_count == 0
    assert len(log_records(repo)) == 5


async def test_iterate_session_when_target_kept_does_refuse_without_measuring(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(repo, session_record(repo), (_on_target_iteration(1), committed_keep(1)))
    stub_runs(samples_mock, repo, [])

    with pytest.raises(LoopStopError) as exc:
        await iterate_session(
            repo, resolved_config(primary="total_ms", stop=StopConfig(target_value=95))
        )

    assert str(exc.value) == "Stop condition met: target reached and kept"
    assert samples_mock.call_count == 0
    assert len(log_records(repo)) == 3


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
        (_on_target_iteration(1), committed_keep(1), iteration_record(seq=2), committed_keep(2)),
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
    assert isinstance(record.at, int)
    assert record.at > 0
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


async def test_iterate_session_when_measured_does_render_the_report_golden(
    settled: str, samples_mock: CollectSamplesRecorder, snapshot: SnapshotAssertion
):
    result = await iterate_session(settled, resolved_config(primary="total_ms"))

    assert result.report.split("\n") == snapshot


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

    assert "target reached" not in plain_report(result.report)


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

    plain = plain_report(result.report)
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
    write_session_log(repo, session_record(repo), (iteration_record(seq=1), committed_keep(1)))
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
    write_session_log(repo, session_record(repo), (iteration_record(seq=1), committed_keep(1)))
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
        for ctx in contexts:
            for event_type in (PassStarted, PassFinished):
                forward(
                    event_type(
                        round=1,
                        total_rounds=1,
                        target_count=len(contexts),
                        label=ctx.label,
                        at_ms=0,
                    )
                )
            collected.append(TargetSamples(ctx=ctx, samples=by_dir[ctx.dir]))
        return collected

    monkeypatch.setattr("gymrat.loop.iterate.confirm.collect_samples", sample_reporting_passes)
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
    probe_started = PassStarted(round=1, total_rounds=1, target_count=1, label="x", at_ms=0)
    probe_finished = PassFinished(round=1, total_rounds=1, target_count=1, label="x", at_ms=0)
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
    write_session_log(repo, session_record(repo), (iteration_record(seq=1), committed_keep(1)))
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


# ---------------------------------------------------------------------------
# the iteration record carries elapsed milliseconds (duration_ms)
# ---------------------------------------------------------------------------


async def test_iterate_session_when_measuring_does_record_duration_ms(
    repo: str, samples_mock: CollectSamplesRecorder, monkeypatch: pytest.MonkeyPatch
):
    experiment_dir = session_record(repo).worktrees.experiment
    _ensure_dir(experiment_dir)
    hooks = HookScripts(repo, experiment_dir)
    write_session_log(repo, session_record(repo), (iteration_record(seq=1), committed_keep(1)))
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())
    # The iteration reads the clock at start and end; the after hook times itself with two more reads.
    ticks = iter([1_000.0, 1_500.0, 2_000.0, 2_000.0])
    monkeypatch.setattr("gymrat.clock.monotonic_ms", lambda: next(ticks))
    config = resolved_config(hooks=HooksConfig(after=hooks.printing("bye")))

    result = await iterate_session(repo, config)

    assert result.record.duration_ms == 500


async def test_iterate_session_when_after_hook_sleeps_does_not_include_its_duration_in_duration_ms(
    repo: str, samples_mock: CollectSamplesRecorder, monkeypatch: pytest.MonkeyPatch
):
    experiment_dir = session_record(repo).worktrees.experiment
    _ensure_dir(experiment_dir)
    hooks = HookScripts(repo, experiment_dir)
    write_session_log(repo, session_record(repo), (iteration_record(seq=1), committed_keep(1)))
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())
    ticks = iter([0.0, 50.0, 100.0, 400.0])
    monkeypatch.setattr("gymrat.clock.monotonic_ms", lambda: next(ticks))
    config = resolved_config(
        hooks=HooksConfig(after=hooks.hook_command("import time\ntime.sleep(0.3)\n"))
    )

    result = await iterate_session(repo, config)

    assert result.record.duration_ms == 50


# ---------------------------------------------------------------------------
# the iteration record carries the experiment-tree fingerprint (measured_tree)
# ---------------------------------------------------------------------------


async def test_iterate_session_when_measuring_does_record_measured_tree_fingerprint(
    settled: str, samples_mock: CollectSamplesRecorder
):
    _ensure_dir(session_record(settled).worktrees.experiment)

    result = await iterate_session(settled, resolved_config())

    assert result.record.measured_tree is not None
    assert isinstance(result.record.measured_tree, str)
    assert len(result.record.measured_tree) > 0


async def test_iterate_session_when_bench_writes_file_does_change_measured_tree(
    repo: str, samples_mock: CollectSamplesRecorder
):
    experiment_dir = session_record(repo).worktrees.experiment
    _ensure_dir(experiment_dir)
    # The experiment dir sits inside the scratch repo: keep the growing session log
    # out of the fingerprint so only the bench's write can change it.
    _workspace.ensure_git_exclude(repo)
    write_session_log(repo, session_record(repo), (iteration_record(seq=1), committed_keep(1)))

    # First run: no extra file in the experiment worktree.
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())
    result_clean = await iterate_session(repo, resolved_config())
    tree_clean = result_clean.record.measured_tree

    write_session_log(
        repo,
        session_record(repo),
        (iteration_record(seq=1), committed_keep(1), iteration_record(seq=2), committed_keep(2)),
    )

    # Second run: write a file into the experiment worktree before fingerprinting.
    # The bench mock writes a marker file so the tree hash changes.
    original_answer = samples_mock._answer

    def writing_answer(targets: list[TargetContext]) -> list[TargetSamples]:
        Path(experiment_dir, "bench-artifact.txt").write_text("artifact", encoding="utf-8")
        return original_answer(targets)

    samples_mock._answer = writing_answer
    result_dirty = await iterate_session(repo, resolved_config())
    tree_dirty = result_dirty.record.measured_tree

    assert tree_clean is not None
    assert tree_dirty is not None
    assert tree_clean != tree_dirty


async def test_iterate_session_when_after_hook_writes_file_does_not_change_measured_tree(
    repo: str, samples_mock: CollectSamplesRecorder
):
    # The after-hook fires after the fingerprint, so its writes must not affect measured_tree.
    experiment_dir = session_record(repo).worktrees.experiment
    _ensure_dir(experiment_dir)
    _workspace.ensure_git_exclude(repo)
    hooks = HookScripts(repo, experiment_dir)
    write_session_log(repo, session_record(repo), (iteration_record(seq=1), committed_keep(1)))
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())

    artifact = Path(experiment_dir, "after-artifact.txt")
    write_body = (
        "import pathlib\n"
        f"pathlib.Path({experiment_dir!r}, 'after-artifact.txt')"
        ".write_text('artifact', encoding='utf-8')\n"
    )
    config = resolved_config(hooks=HooksConfig(after=hooks.hook_command(write_body)))
    result = await iterate_session(repo, config)

    assert artifact.is_file(), "after hook must have run"  # noqa: ASYNC240
    assert result.record.measured_tree is not None
    tree_after = _workspace.worktree_fingerprint(Path(experiment_dir))
    assert tree_after is not None
    assert result.record.measured_tree != tree_after


async def test_iterate_session_when_fingerprint_fails_does_omit_measured_tree_with_warning(
    settled: str,
    samples_mock: CollectSamplesRecorder,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "gymrat.session.workspace.worktree_fingerprint",
        lambda _directory: None,  # pyrefly: ignore
        raising=True,
    )

    result = await iterate_session(settled, resolved_config())

    assert result.record.measured_tree is None
    assert result.record.seq == 2
    captured = capsys.readouterr()
    assert captured.err == (
        "Could not fingerprint the experiment worktree; "
        "measured_tree is omitted from the iteration record.\n"
    )


# ---------------------------------------------------------------------------
# budget refusal: live budget + known estimate > remaining
# ---------------------------------------------------------------------------


def _install_live_budget(monkeypatch: pytest.MonkeyPatch, *, deadline_ms: float) -> None:
    """Make ``read_budget`` answer a 30-minute budget due at *deadline_ms*, with the clock at zero."""
    live_budget = Budget(max_minutes=30, deadline_ms=deadline_ms)
    monkeypatch.setattr(
        "gymrat.session.budget.read_budget",
        lambda _root, **_kw: live_budget,  # pyrefly: ignore
    )
    monkeypatch.setattr("gymrat.clock.now_ms", lambda: 0)


async def test_iterate_session_when_budget_exceeded_does_refuse_before_any_hook_or_bench(
    repo: str, samples_mock: CollectSamplesRecorder, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(
        repo,
        session_record(repo),
        (iteration_record(seq=1, duration_ms=840_000), committed_keep(1)),
    )
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())

    # Live budget with 12 min left, but last iteration took 14 min.
    _install_live_budget(monkeypatch, deadline_ms=720_000.0)

    with pytest.raises(LoopStopError) as exc:
        await iterate_session(repo, resolved_config())

    message = str(exc.value)
    assert "12m" in message
    hint = exc.value.hint or ""
    assert "report" in hint.lower() or "session" in hint.lower()
    assert samples_mock.call_count == 0


async def test_iterate_session_when_budget_exceeded_does_name_estimate_source_in_message(
    repo: str, samples_mock: CollectSamplesRecorder, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(
        repo,
        session_record(repo),
        (iteration_record(seq=1, duration_ms=840_000), committed_keep(1)),
    )
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())
    _install_live_budget(monkeypatch, deadline_ms=720_000.0)

    with pytest.raises(LoopStopError) as exc:
        await iterate_session(repo, resolved_config())

    message = str(exc.value)
    assert "iteration" in message.lower()


# ---------------------------------------------------------------------------
# no budget: manual session runs normally
# ---------------------------------------------------------------------------


async def test_iterate_session_when_budget_live_but_no_estimate_does_run_normally(
    repo: str, samples_mock: CollectSamplesRecorder, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, session_record(repo), (iteration_record(seq=1), committed_keep(1)))
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())

    _install_live_budget(monkeypatch, deadline_ms=1_800_000.0)

    result = await iterate_session(repo, resolved_config())

    assert result.record.seq == 2


async def test_iterate_session_when_no_budget_does_run_normally(
    settled: str, samples_mock: CollectSamplesRecorder
):
    result = await iterate_session(settled, resolved_config())

    assert result.record.seq == 2
    assert result.record.duration_ms is not None


async def test_iterate_session_when_stop_condition_met_does_report_stop_before_budget_check(
    repo: str, samples_mock: CollectSamplesRecorder, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, session_record(repo), (iteration_record(seq=1), committed_keep(1)))
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())

    _install_live_budget(monkeypatch, deadline_ms=720_000.0)

    config = resolved_config(stop=StopConfig(max_iterations=1))

    with pytest.raises(LoopStopError, match="max iterations") as exc:
        await iterate_session(repo, config)

    assert not isinstance(exc.value, BudgetExceededError)


# ---------------------------------------------------------------------------
# adapter warnings: a bench line the adapter cannot read
# ---------------------------------------------------------------------------


async def test_iterate_session_when_warn_sink_given_does_route_adapter_warnings_to_it(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    write_session_log(repo, session_record(repo))
    bench_malformed_once(monkeypatch)
    warnings: list[str] = []

    await iterate_session(repo, resolved_config(), options=IterateOptions(warn=warnings.append))

    assert warnings == [MALFORMED_LINE_WARNING]
    assert MALFORMED_LINE_WARNING not in capsys.readouterr().err


async def test_iterate_session_when_no_warn_sink_does_print_adapter_warnings_on_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    write_session_log(repo, session_record(repo))
    bench_malformed_once(monkeypatch)

    await iterate_session(repo, resolved_config())

    assert MALFORMED_LINE_WARNING in capsys.readouterr().err.splitlines()
