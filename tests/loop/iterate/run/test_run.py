"""Behavioral tests for ``iterate_session``: measuring one edit of an open session.

Covers when it refuses to measure, what a measured iteration records, the
progress it emits, the elapsed time and tree fingerprint the record carries,
the before and after hooks around the measurement, the budget, and adapter
warnings. How the verdicts are judged lives in ``test_gating.py``.

The one boundary these tests mock is sampling, which shells out to the
consumer's bench script; everything downstream of it — verdicts, aggregation,
the record, the report, and the real hook subprocesses — runs for real. The
budget tests lay a real budget file down and freeze the wall clock it is read
against.
Sessions are laid down on disk with the real record builders against a
throwaway repository, so the suite is order-independent and safe under
``pytest-xdist`` / ``pytest-randomly``.
"""

from __future__ import annotations

import asyncio
import json
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
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    JudgeStarted,
    PassFinished,
    ProgressEvent,
)
from gymrat.sampling import SamplingOptions, TargetContext
from gymrat.session import workspace as _workspace
from gymrat.session.records import HookRecord, PairedSamples
from gymrat.targets import InPlaceTarget
from tests._clock import install_monotonic_clock
from tests._config import resolved_config
from tests.adapters._inputs import VALID_ADAPTERS_HINT, unknown_adapter_message
from tests.loop.iterate._fixtures import (
    FILTER,
    MALFORMED_LINE_WARNING,
    as_logged,
    assert_permutation,
    baseline_rounds,
    bench_malformed_once,
    improved_rounds,
    iterate_session_header,
    last_iteration_of,
    regressed_run,
    report_a_pass_per_call,
    run_before_each_call,
    stub_improved_samples,
    stub_runs,
    trimmed_report_lines,
    write_iterate_session,
    write_outlasted_session,
)
from tests.loop.iterate._hooks import HookScripts
from tests.session._budget import install_budget
from tests.session.records._fixtures import (
    committed_keep,
    discard_record,
    iteration_record,
    log_records,
    records_of_type,
    settled_history,
)

if TYPE_CHECKING:
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

#: The events that mark each stage of an iteration, hooks included, in the order they fire.
_STAGE_EVENTS = (
    HookStarted,
    HookFinished,
    JudgeStarted,
    JudgeFinished,
    ConfirmStarted,
    ConfirmFinished,
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
            settled_history(),
            resolved_config(adapter="banana"),
            (unknown_adapter_message("banana"), VALID_ADAPTERS_HINT, None),
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
    write_iterate_session(repo, history)

    with pytest.raises(GymratError) as exc:
        await iterate_session(repo, config)

    assert (str(exc.value), exc.value.hint, exc.value.reason) == refusal
    assert samples_mock.call_count == 0
    assert len(log_records(repo)) == len(history) + 1


# ---------------------------------------------------------------------------
# a configured stop condition no longer applies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("history", "config", "expected_seq"),
    [
        pytest.param(
            (iteration_record(seq=1, target_reached=True), discard_record(1)),
            resolved_config(primary="total_ms", stop=StopConfig(target_value=95)),
            2,
            id="target-iteration-discarded",
        ),
        pytest.param(
            (
                iteration_record(seq=1, target_reached=True),
                committed_keep(1),
                iteration_record(seq=2),
                committed_keep(2),
            ),
            resolved_config(),
            3,
            id="no-stop-configured",
        ),
    ],
)
async def test_iterate_session_when_stop_condition_no_longer_applies_does_measure_again(
    repo: str,
    samples_mock: CollectSamplesRecorder,
    history: tuple[SessionLogRecord, ...],
    config: ResolvedConfig,
    expected_seq: int,
):
    write_iterate_session(repo, history)
    stub_improved_samples(samples_mock, repo)

    result = await iterate_session(repo, config)

    assert result.record.seq == expected_seq


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
    clock = install_monotonic_clock(monkeypatch)

    async def bench_taking_500_ms(_options: SamplingOptions) -> None:
        clock.tick(500.0)

    def slow_after_hook(event: ProgressEvent) -> None:
        if isinstance(event, HookStarted) and event.stage == "after":
            clock.tick(60_000.0)

    run_before_each_call(monkeypatch, samples_mock, bench_taking_500_ms)
    config = resolved_config(hooks=HooksConfig(after=hooks.printing("bye")))

    result = await iterate_session(
        repo, config, options=IterateOptions(on_progress=slow_after_hook)
    )

    assert result.record.duration_ms == 500


# ---------------------------------------------------------------------------
# the iteration record carries the experiment-tree fingerprint (measured_tree)
# ---------------------------------------------------------------------------


async def test_iterate_session_when_experiment_worktree_missing_does_omit_measured_tree_with_a_warning(
    settled: str, capsys: pytest.CaptureFixture[str]
):
    # The session names an experiment worktree that was never created, so git
    # has no tree there to fingerprint.
    result = await iterate_session(settled, resolved_config())

    assert result.record.measured_tree is None
    assert last_iteration_of(settled).measured_tree is None
    assert capsys.readouterr().err == (
        "Could not fingerprint the experiment worktree; "
        "measured_tree is omitted from the iteration record.\n"
    )


async def test_iterate_session_when_bench_writes_file_does_fingerprint_the_tree_it_left(
    hooks_setup: tuple[str, str, HookScripts],
    samples_mock: CollectSamplesRecorder,
    monkeypatch: pytest.MonkeyPatch,
):
    settled, experiment_dir, _hooks = hooks_setup
    # The experiment dir sits inside the scratch repo: keep the growing session log
    # out of the fingerprint so only the bench's write can change it.
    _workspace.ensure_git_exclude(settled)

    async def bench_writing_an_artifact(_options: SamplingOptions) -> None:
        await asyncio.to_thread(
            Path(experiment_dir, "bench-artifact.txt").write_text, "artifact", encoding="utf-8"
        )

    run_before_each_call(monkeypatch, samples_mock, bench_writing_an_artifact)

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


# ---------------------------------------------------------------------------
# the config declares a command for a stage (hooks)
# ---------------------------------------------------------------------------


def _hook_events(events: list[ProgressEvent]) -> list[tuple[type[HookStarted | HookFinished], str]]:
    """Each hook event's type and stage, in the order they were emitted."""
    return [(type(e), e.stage) for e in events if isinstance(e, (HookStarted, HookFinished))]


def _capturing_payload(hooks: HookScripts, stage: str) -> str:
    """A hook command filing the payload it was handed away where assertions can read it."""
    body = (
        "import sys, pathlib\n"
        "data = sys.stdin.buffer.read()\n"
        f"pathlib.Path({json.dumps(stage + '.json')}).write_bytes(data)\n"
    )
    return hooks.hook_command(body)


async def test_iterate_session_when_hooks_configured_does_bracket_the_whole_measurement(
    hooks_setup: tuple[str, str, HookScripts], samples_mock: CollectSamplesRecorder
):
    repo, _experiment_dir, hooks = hooks_setup
    stub_runs(samples_mock, repo, [regressed_run(), regressed_run()])
    config = resolved_config(
        filter=FILTER,
        hooks=HooksConfig(before=hooks.printing("hi"), after=hooks.printing("bye")),
    )
    events: list[ProgressEvent] = []

    result = await iterate_session(repo, config, options=IterateOptions(on_progress=events.append))

    lines = trimmed_report_lines(result.report)
    assert (lines[0], lines[1], lines[-1]) == (
        "[before] hi",
        "iteration 2 · experiment vs baseline · 10 paired samples",
        "[after] bye",
    )
    assert [record.type for record in log_records(repo)] == [
        "session",
        "iteration",
        "keep",
        "hook",
        "iteration",
        "hook",
    ]
    assert [record.stage for record in records_of_type(repo, HookRecord)] == ["before", "after"]
    assert [
        (type(e), getattr(e, "stage", None)) for e in events if isinstance(e, _STAGE_EVENTS)
    ] == [
        (HookStarted, "before"),
        (HookFinished, "before"),
        (JudgeStarted, None),
        (JudgeFinished, None),
        (ConfirmStarted, None),
        (ConfirmFinished, None),
        (IterationRecorded, None),
        (HookStarted, "after"),
        (HookFinished, "after"),
    ]


def _iteration_fields(experiment_dir: str, stage: str) -> tuple[object, ...]:
    """The iteration-level fields of the payload the ``stage`` hook was handed.

    The capturing command names the file relatively, so reading it back out of
    the experiment worktree is also what proves the hook ran there.

    Args:
        experiment_dir: The experiment worktree the hook wrote its payload into.
        stage: The hook stage whose payload file to read.

    Returns:
        The payload's stage, experiment dir, seq, last iteration, and session
        iteration count, in that order.
    """
    payload = json.loads((Path(experiment_dir) / f"{stage}.json").read_text(encoding="utf-8"))
    return (
        payload["stage"],
        payload["experiment_dir"],
        payload["seq"],
        payload["last_iteration"],
        payload["session"]["iteration_count"],
    )


async def test_iterate_session_when_hooks_configured_does_tell_each_which_iteration(
    hooks_setup: tuple[str, str, HookScripts],
):
    repo, experiment_dir, hooks = hooks_setup
    config = resolved_config(
        hooks=HooksConfig(
            before=_capturing_payload(hooks, "before"), after=_capturing_payload(hooks, "after")
        )
    )

    result = await iterate_session(repo, config)

    assert _iteration_fields(experiment_dir, "before") == (
        "before",
        experiment_dir,
        2,
        as_logged(iteration_record(seq=1)),
        1,
    )
    assert _iteration_fields(experiment_dir, "after") == (
        "after",
        experiment_dir,
        2,
        as_logged(result.record),
        2,
    )


async def test_iterate_session_when_before_hook_fails_does_still_measure(
    hooks_setup: tuple[str, str, HookScripts],
):
    repo, _experiment_dir, hooks = hooks_setup
    before = hooks.failing_content_of("before-fails", "", "no warm copy\n")
    config = resolved_config(hooks=HooksConfig(before=before))

    result = await iterate_session(repo, config)

    assert [(record.stage, record.exit_code) for record in records_of_type(repo, HookRecord)] == [
        ("before", 3)
    ]
    assert last_iteration_of(repo).seq == 2
    assert result.record.outcome == "improved"


@pytest.mark.parametrize(
    ("with_after", "expected_stages", "expected_events"),
    [
        pytest.param(False, [], [], id="no-hooks"),
        pytest.param(
            True,
            ["after"],
            [(HookStarted, "after"), (HookFinished, "after")],
            id="only-after",
        ),
    ],
)
async def test_iterate_session_when_before_stage_absent_does_run_nothing_for_it(
    hooks_setup: tuple[str, str, HookScripts],
    *,
    with_after: bool,
    expected_stages: list[str],
    expected_events: list[tuple[type[HookStarted | HookFinished], str]],
):
    repo, _experiment_dir, hooks = hooks_setup
    hooks_config = HooksConfig(after=hooks.printing("bye")) if with_after else None
    config = resolved_config(hooks=hooks_config)
    events: list[ProgressEvent] = []

    result = await iterate_session(repo, config, options=IterateOptions(on_progress=events.append))

    assert [record.stage for record in records_of_type(repo, HookRecord)] == expected_stages
    assert _hook_events(events) == expected_events
    before_lines = [
        line for line in trimmed_report_lines(result.report) if line.startswith("[before]")
    ]
    assert before_lines == []


# ---------------------------------------------------------------------------
# budget refusal: live budget + known estimate > remaining
# ---------------------------------------------------------------------------


async def test_iterate_session_when_budget_exceeded_does_refuse_before_any_hook_or_bench(
    repo: str, samples_mock: CollectSamplesRecorder, monkeypatch: pytest.MonkeyPatch
):
    hooks = HookScripts.for_root(repo)
    write_outlasted_session(repo, monkeypatch)
    stub_improved_samples(samples_mock, repo)
    config = resolved_config(hooks=HooksConfig(before=hooks.printing("hi")))

    with pytest.raises(LoopStopError) as exc:
        await iterate_session(repo, config)

    assert str(exc.value) == (
        "5m left; the last iteration took 14m and the cap would cut this one off."
    )
    assert exc.value.hint == "Report what the session measured instead of measuring again."
    assert samples_mock.call_count == 0
    assert [record.type for record in log_records(repo)] == ["session", "iteration", "keep"]


# ---------------------------------------------------------------------------
# budget live: no estimate yet, and the stop check runs before the budget check
# ---------------------------------------------------------------------------


async def test_iterate_session_when_budget_live_but_no_estimate_does_run_normally(
    settled: str, monkeypatch: pytest.MonkeyPatch
):
    install_budget(settled, monkeypatch, deadline_ms=1_800_000.0, frozen_now_ms=0)

    result = await iterate_session(settled, resolved_config())

    assert result.record.seq == 2


async def test_iterate_session_when_stop_condition_met_does_report_stop_before_budget_check(
    repo: str, samples_mock: CollectSamplesRecorder, monkeypatch: pytest.MonkeyPatch
):
    # The last iteration took 14 minutes against 5 left, so the budget check
    # would refuse on its own: only the stop check running first names max iterations.
    write_outlasted_session(repo, monkeypatch)
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
