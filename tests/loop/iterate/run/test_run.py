"""Behavioral tests for ``iterate_session``: measuring one edit of an open session.

The one boundary these tests mock is sampling, which shells out to the
consumer's bench script; everything downstream of it — verdicts, aggregation,
the record, the report — runs for real. Sessions are laid down on disk with the
real record builders against a throwaway repository, so the suite is
order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat.config import HooksConfig, StopConfig
from gymrat.errors import GymratError
from gymrat.loop.iterate.run import (
    BudgetExceededError,
    IterateOptions,
    LoopStopError,
    iterate_session,
)
from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmSkipped,
    ConfirmStarted,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    JudgeStarted,
    PassFinished,
    ProgressEvent,
)
from gymrat.sampling import SamplingOptions, TargetContext, TargetSamples
from gymrat.session import workspace as _workspace
from gymrat.session.budget import Budget
from gymrat.session.records import PairedSamples
from gymrat.targets import InPlaceTarget
from tests._config import resolved_config
from tests.adapters._inputs import VALID_ADAPTERS_HINT
from tests.loop.iterate._fixtures import (
    MALFORMED_LINE_WARNING,
    as_logged,
    assert_permutation,
    baseline_rounds,
    bench_malformed_once,
    improved_rounds,
    iterate_session_header,
    last_iteration_of,
    report_a_pass_per_call,
    stub_samples,
)
from tests.loop.iterate._hooks import HookScripts
from tests.session.records._fixtures import (
    committed_keep,
    discard_record,
    iteration_record,
    log_records,
    write_session_log,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymrat.config import ResolvedConfig
    from gymrat.session.records import SessionLogRecord
    from tests.loop.iterate._fixtures import CollectSamplesRecorder


#: The events that mark the end of a bench pass and the judge, confirmation and
#: record stages of an iteration.
_MILESTONE_EVENTS = (
    PassFinished,
    JudgeStarted,
    JudgeFinished,
    ConfirmStarted,
    ConfirmFinished,
    ConfirmSkipped,
    IterationRecorded,
)

#: The hint every stop-condition refusal closes on.
_STOP_HINT = "The loop is done. Report what the session measured instead of measuring again."


# ---------------------------------------------------------------------------
# refusing to measure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("history", "config", "refusal"),
    [
        pytest.param(
            (iteration_record(seq=1),),
            resolved_config(),
            (
                "Iteration 1 has not been settled",
                "Run gymrat keep or gymrat discard before measuring the next edit.",
                "unsettled",
            ),
            id="last-iteration-unsettled",
        ),
        pytest.param(
            (iteration_record(seq=1), committed_keep(1)),
            resolved_config(adapter="banana"),
            ('Unknown adapter: "banana".', VALID_ADAPTERS_HINT, None),
            id="adapter-unknown",
        ),
        pytest.param(
            (
                iteration_record(seq=1),
                committed_keep(1),
                iteration_record(seq=2),
                committed_keep(2),
            ),
            resolved_config(stop=StopConfig(max_iterations=2)),
            ("Stop condition met: max iterations (2 of 2)", _STOP_HINT, "stop-condition"),
            id="max-iterations-reached",
        ),
        pytest.param(
            (iteration_record(seq=1, target_reached=True), committed_keep(1)),
            resolved_config(primary="total_ms", stop=StopConfig(target_value=95)),
            ("Stop condition met: target reached and kept", _STOP_HINT, "stop-condition"),
            id="target-kept",
        ),
    ],
)
async def test_iterate_session_when_not_ready_to_measure_does_refuse_before_sampling(
    repo: str,
    samples_mock: CollectSamplesRecorder,
    history: tuple[SessionLogRecord, ...],
    config: ResolvedConfig,
    refusal: tuple[str, str, str | None],
):
    write_session_log(repo, iterate_session_header(repo), history)

    with pytest.raises(GymratError) as exc:
        await iterate_session(repo, config)

    assert (str(exc.value), exc.value.hint, exc.value.reason) == refusal
    assert samples_mock.call_count == 0
    assert len(log_records(repo)) == len(history) + 1


# ---------------------------------------------------------------------------
# a configured stop condition no longer applies
# ---------------------------------------------------------------------------


async def test_iterate_session_when_target_iteration_discarded_does_measure_again(
    repo: str, samples_mock: CollectSamplesRecorder
):
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())
    write_session_log(
        repo,
        iterate_session_header(repo),
        (iteration_record(seq=1, target_reached=True), discard_record(1)),
    )

    result = await iterate_session(
        repo, resolved_config(primary="total_ms", stop=StopConfig(target_value=95))
    )

    assert result.record.seq == 2


async def test_iterate_session_when_no_stop_configured_does_measure_past_a_kept_target(
    repo: str, samples_mock: CollectSamplesRecorder
):
    write_session_log(
        repo,
        iterate_session_header(repo),
        (
            iteration_record(seq=1, target_reached=True),
            committed_keep(1),
            iteration_record(seq=2),
            committed_keep(2),
        ),
    )
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())

    result = await iterate_session(repo, resolved_config())

    assert result.record.seq == 3


# ---------------------------------------------------------------------------
# measuring a settled session
# ---------------------------------------------------------------------------


async def test_iterate_session_when_measuring_does_record_the_paired_iteration_after_last_settled(
    settled: str, samples_mock: CollectSamplesRecorder
):
    worktrees = iterate_session_header(settled).worktrees

    result = await iterate_session(settled, resolved_config())

    assert samples_mock.calls[0].targets == [
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
    record = last_iteration_of(settled)
    assert as_logged(result.record) == as_logged(record)
    assert record.seq == 2
    assert isinstance(record.at, int)
    assert record.at > 0
    assert record.samples == PairedSamples(
        experiment=tuple(improved_rounds()), baseline=tuple(baseline_rounds())
    )
    total = record.metrics["total_ms"]
    assert_permutation(total, delta=-10, verdict="improved", confirmed=False)
    alloc = record.metrics["alloc_bytes"]
    assert alloc.delta_pct == pytest.approx(-20, abs=1e-6)
    assert alloc.verdict == "improved"
    assert record.primary.kind == "geomean"
    assert record.primary.name is None
    assert record.primary.delta_pct == pytest.approx(-15.1472, abs=1e-3)
    assert record.outcome == "improved"
    assert record.target_reached is False


async def test_iterate_session_when_primary_is_named_metric_does_report_its_delta_alone(
    settled: str, samples_mock: CollectSamplesRecorder
):
    result = await iterate_session(settled, resolved_config(primary="total_ms"))

    assert result.record.primary.kind == "metric"
    assert result.record.primary.name == "total_ms"
    assert result.record.primary.delta_pct == pytest.approx(-10, abs=1e-6)


@pytest.mark.parametrize("color", [False, True])
async def test_iterate_session_when_color_given_does_emit_ansi_only_when_true(
    settled: str, samples_mock: CollectSamplesRecorder, *, color: bool
):
    result = await iterate_session(settled, resolved_config(), color=color)

    assert ("\x1b[" in result.report) is color


# ---------------------------------------------------------------------------
# progress events emitted by iterate_session
# ---------------------------------------------------------------------------


async def test_iterate_session_when_improved_does_judge_after_the_bench_then_record_without_confirming(
    settled: str, samples_mock: CollectSamplesRecorder, monkeypatch: pytest.MonkeyPatch
):
    report_a_pass_per_call(monkeypatch, samples_mock)
    events: list[ProgressEvent] = []

    result = await iterate_session(
        settled, resolved_config(), options=IterateOptions(on_progress=events.append)
    )

    milestones = [e for e in events if isinstance(e, _MILESTONE_EVENTS)]
    assert [type(e) for e in milestones] == [
        PassFinished,
        JudgeStarted,
        JudgeFinished,
        ConfirmSkipped,
        IterationRecorded,
    ]
    _, _, judge, _, recorded = milestones
    assert isinstance(judge, JudgeFinished)
    assert judge.primary_delta_pct == pytest.approx(-15.1472, abs=1e-3)
    assert judge.regressed == ()
    assert isinstance(recorded, IterationRecorded)
    assert (recorded.seq, recorded.outcome) == (result.record.seq, "improved")


# ---------------------------------------------------------------------------
# the iteration record carries elapsed milliseconds (duration_ms)
# ---------------------------------------------------------------------------


async def test_iterate_session_when_measuring_does_exclude_the_after_hook_from_duration_ms(
    hooks_setup: tuple[str, str, HookScripts],
    samples_mock: CollectSamplesRecorder,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, _experiment_dir, hooks = hooks_setup
    # The clock stands still except where the test moves it: the bench takes
    # 500 ms, and the after hook takes far longer once it starts.
    now_ms = [1_000.0]
    monkeypatch.setattr("gymrat.clock.monotonic_ms", lambda: now_ms[0])

    async def bench_taking_500_ms(
        adapter: object,
        targets: Sequence[TargetContext],
        options: SamplingOptions,
        abort: object,
    ) -> list[TargetSamples]:
        now_ms[0] += 500.0
        return await samples_mock(adapter, targets, options, abort)

    def slow_after_hook(event: ProgressEvent) -> None:
        if isinstance(event, HookStarted) and event.stage == "after":
            now_ms[0] += 60_000.0

    monkeypatch.setattr("gymrat.loop.iterate.confirm.collect_samples", bench_taking_500_ms)
    config = resolved_config(hooks=HooksConfig(after=hooks.printing("bye")))

    result = await iterate_session(
        repo, config, options=IterateOptions(on_progress=slow_after_hook)
    )

    assert result.record.duration_ms == 500


# ---------------------------------------------------------------------------
# the iteration record carries the experiment-tree fingerprint (measured_tree)
# ---------------------------------------------------------------------------


async def test_iterate_session_when_bench_writes_file_does_fingerprint_the_tree_it_left(
    hooks_setup: tuple[str, str, HookScripts],
    samples_mock: CollectSamplesRecorder,
    monkeypatch: pytest.MonkeyPatch,
):
    settled, experiment_dir, _hooks = hooks_setup
    # The experiment dir sits inside the scratch repo: keep the growing session log
    # out of the fingerprint so only the bench's write can change it.
    _workspace.ensure_git_exclude(settled)

    async def bench_writing_an_artifact(
        adapter: object,
        targets: Sequence[TargetContext],
        options: SamplingOptions,
        abort: object,
    ) -> list[TargetSamples]:
        await asyncio.to_thread(
            Path(experiment_dir, "bench-artifact.txt").write_text, "artifact", encoding="utf-8"
        )
        return await samples_mock(adapter, targets, options, abort)

    monkeypatch.setattr("gymrat.loop.iterate.confirm.collect_samples", bench_writing_an_artifact)

    result = await iterate_session(settled, resolved_config())

    assert result.record.measured_tree is not None
    assert result.record.measured_tree == _workspace.worktree_fingerprint(Path(experiment_dir))


async def test_iterate_session_when_after_hook_writes_file_does_not_change_measured_tree(
    hooks_setup: tuple[str, str, HookScripts],
):
    # The after-hook fires after the fingerprint, so its writes must not affect measured_tree.
    repo, experiment_dir, hooks = hooks_setup
    _workspace.ensure_git_exclude(repo)

    write_body = (
        "import pathlib\n"
        f"pathlib.Path({experiment_dir!r}, 'after-artifact.txt')"
        ".write_text('artifact', encoding='utf-8')\n"
    )
    config = resolved_config(hooks=HooksConfig(after=hooks.hook_command(write_body)))

    result = await iterate_session(repo, config)

    assert result.record.measured_tree is not None
    tree_after = _workspace.worktree_fingerprint(Path(experiment_dir))
    assert tree_after is not None
    assert result.record.measured_tree != tree_after


async def test_iterate_session_when_fingerprint_fails_does_omit_measured_tree_with_warning(
    settled: str,
    samples_mock: CollectSamplesRecorder,
    capsys: pytest.CaptureFixture[str],
):
    # The session names an experiment worktree that was never created, so git
    # has no tree there to fingerprint.
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

    def read_live_budget(root: str, *, now_ms: float) -> Budget | None:
        return live_budget

    monkeypatch.setattr("gymrat.session.budget.read_budget", read_live_budget)
    monkeypatch.setattr("gymrat.clock.now_ms", lambda: 0)


async def test_iterate_session_when_budget_exceeded_does_refuse_before_any_hook_or_bench(
    repo: str, samples_mock: CollectSamplesRecorder, monkeypatch: pytest.MonkeyPatch
):
    hooks = HookScripts(repo, iterate_session_header(repo).worktrees.experiment)
    write_session_log(
        repo,
        iterate_session_header(repo),
        (iteration_record(seq=1, duration_ms=840_000), committed_keep(1)),
    )
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())

    # Live budget with 12 min left, but last iteration took 14 min.
    _install_live_budget(monkeypatch, deadline_ms=720_000.0)
    config = resolved_config(hooks=HooksConfig(before=hooks.printing("hi")))

    with pytest.raises(LoopStopError) as exc:
        await iterate_session(repo, config)

    assert str(exc.value) == (
        "12m left; the last iteration took 14m and the cap would cut this one off."
    )
    assert exc.value.hint == "Report what the session measured instead of measuring again."
    assert samples_mock.call_count == 0
    assert [record.type for record in log_records(repo)] == ["session", "iteration", "keep"]


# ---------------------------------------------------------------------------
# no budget: manual session runs normally
# ---------------------------------------------------------------------------


async def test_iterate_session_when_budget_live_but_no_estimate_does_run_normally(
    settled: str, monkeypatch: pytest.MonkeyPatch
):
    _install_live_budget(monkeypatch, deadline_ms=1_800_000.0)

    result = await iterate_session(settled, resolved_config())

    assert result.record.seq == 2


async def test_iterate_session_when_stop_condition_met_does_report_stop_before_budget_check(
    repo: str, samples_mock: CollectSamplesRecorder, monkeypatch: pytest.MonkeyPatch
):
    # The last iteration took 14 minutes against 12 left, so the budget check
    # would refuse on its own: only the stop check running first names max iterations.
    write_session_log(
        repo,
        iterate_session_header(repo),
        (iteration_record(seq=1, duration_ms=840_000), committed_keep(1)),
    )
    _install_live_budget(monkeypatch, deadline_ms=720_000.0)
    config = resolved_config(stop=StopConfig(max_iterations=1))

    with pytest.raises(LoopStopError, match="max iterations") as exc:
        await iterate_session(repo, config)

    assert not isinstance(exc.value, BudgetExceededError)


# ---------------------------------------------------------------------------
# adapter warnings: a bench line the adapter cannot read
# ---------------------------------------------------------------------------


async def test_iterate_session_when_warn_sink_given_does_route_adapter_warnings_to_it(
    open_repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    bench_malformed_once(monkeypatch)
    warnings: list[str] = []

    await iterate_session(
        open_repo, resolved_config(), options=IterateOptions(warn=warnings.append)
    )

    assert warnings == [MALFORMED_LINE_WARNING]
    assert MALFORMED_LINE_WARNING not in capsys.readouterr().err


async def test_iterate_session_when_no_warn_sink_does_print_adapter_warnings_on_stderr(
    open_repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    bench_malformed_once(monkeypatch)

    await iterate_session(open_repo, resolved_config())

    assert MALFORMED_LINE_WARNING in capsys.readouterr().err.splitlines()
