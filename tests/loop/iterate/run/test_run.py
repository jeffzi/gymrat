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

from gymrat.adapters.contract import TIME_SUFFIX
from gymrat.config import HooksConfig, StopConfig
from gymrat.errors import GymratError
from gymrat.loop.iterate.run import IterateOptions, iterate_session
from gymrat.loop.stop_condition import LoopStopError
from gymrat.progress_events import (
    ConfirmSkipped,
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    ProgressEvent,
)
from gymrat.session import workspace as _workspace
from gymrat.session.records import HookRecord
from tests._clock import install_monotonic_clock
from tests._config import resolved_config
from tests._pipeline import QUANTIZED_GAP, QUANTIZED_STEP_NS, is_quantized_warning
from tests.loop.iterate._fixtures import (
    MALFORMED_LINE_WARNING,
    PairedRun,
    as_logged,
    assert_permutation,
    baseline_rounds,
    bench_malformed_once,
    improved_rounds,
    last_iteration_of,
    run_before_each_call,
    stub_runs,
    trimmed_report_lines,
    write_iterate_session,
)
from tests.loop.iterate._hooks import HookScripts, labeled_lines
from tests.session._budget import install_budget
from tests.session.records._fixtures import (
    baseline_record,
    iteration_record,
    records_of_type,
    settled_history,
)

if TYPE_CHECKING:
    from gymrat.sampling import SamplingOptions
    from gymrat.session.records import SessionLogRecord
    from tests.loop.iterate._fixtures import CollectSamplesRecorder


# ---------------------------------------------------------------------------
# refusing to measure
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("samples_mock")
async def test_iterate_session_when_last_iteration_unsettled_does_refuse_before_sampling(repo: str):
    write_iterate_session(repo, (iteration_record(seq=1),))

    with pytest.raises(GymratError) as exc:
        await iterate_session(repo, resolved_config())

    assert exc.value.reason == "unsettled"


# ---------------------------------------------------------------------------
# measuring a settled session
# ---------------------------------------------------------------------------


async def test_iterate_session_when_measuring_does_record_the_paired_iteration_after_last_settled(
    settled: str, samples_mock: CollectSamplesRecorder
):
    result = await iterate_session(settled, resolved_config())

    record = last_iteration_of(settled)
    assert as_logged(result.record) == as_logged(record)
    assert record.seq == 2
    total = record.metrics["total_ms"]
    assert_permutation(total, delta=-10, verdict="improved", confirmed=False)
    assert total.gating is True
    assert total.p is not None
    assert total.noise_pct is not None
    assert record.primary.kind == "geomean"
    assert record.primary.name is None
    assert record.primary.delta_pct == pytest.approx(-15.1472, abs=1e-3)
    assert record.target_reached is False


# ---------------------------------------------------------------------------
# progress events emitted by iterate_session
# ---------------------------------------------------------------------------


async def test_iterate_session_when_improved_does_record_without_confirming(
    settled: str, samples_mock: CollectSamplesRecorder
):
    events: list[ProgressEvent] = []

    result = await iterate_session(
        settled, resolved_config(), options=IterateOptions(on_progress=events.append)
    )

    event_types = {type(e) for e in events}
    assert ConfirmSkipped in event_types
    (judge,) = [e for e in events if isinstance(e, JudgeFinished)]
    assert judge.primary_delta_pct == pytest.approx(-15.1472, abs=1e-3)
    assert judge.regressed == ()
    (recorded,) = [e for e in events if isinstance(e, IterationRecorded)]
    assert (recorded.seq, recorded.outcome) == (result.record.seq, "improved")


# ---------------------------------------------------------------------------
# the iteration record carries elapsed milliseconds (duration_ms)
# ---------------------------------------------------------------------------


async def test_iterate_session_when_measuring_does_record_the_elapsed_milliseconds(
    settled: str,
    samples_mock: CollectSamplesRecorder,
    monkeypatch: pytest.MonkeyPatch,
):
    # The clock stands still except where the test moves it: the bench takes 500 ms.
    clock = install_monotonic_clock(monkeypatch)

    async def bench_taking_500_ms(_options: SamplingOptions) -> None:
        clock.tick(500.0)

    run_before_each_call(monkeypatch, samples_mock, bench_taking_500_ms)

    result = await iterate_session(settled, resolved_config())

    assert result.record.duration_ms == 500


# ---------------------------------------------------------------------------
# the iteration record carries the experiment-tree fingerprint (measured_tree)
# ---------------------------------------------------------------------------


async def test_iterate_session_when_experiment_worktree_missing_does_omit_measured_tree_warning_the_sink(
    settled: str,
):
    # The session names an experiment worktree that was never created, so git
    # has no tree there to fingerprint.
    warnings: list[str] = []

    result = await iterate_session(
        settled, resolved_config(), options=IterateOptions(warn=warnings.append)
    )

    assert result.record.measured_tree is None
    assert len(warnings) == 1
    assert "measured_tree" in warnings[0]


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
    assert result.record.measured_tree == _workspace.worktree_fingerprint(experiment_dir)


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
    hooks_setup: tuple[str, str, HookScripts],
):
    repo, _experiment_dir, hooks = hooks_setup
    config = resolved_config(
        hooks=HooksConfig(before=hooks.printing("hi"), after=hooks.printing("bye")),
    )

    result = await iterate_session(repo, config)

    lines = trimmed_report_lines(result.report)
    assert labeled_lines(lines[0], "before") == ["hi"]
    assert lines[1].startswith("iteration 2")
    assert labeled_lines(lines[-1], "after") == ["bye"]


def _iteration_fields(experiment_dir: str, stage: str) -> tuple[object, ...]:
    """The iteration-level fields of the payload the ``stage`` hook was handed.

    Args:
        experiment_dir: The experiment worktree the hook wrote its payload into.
        stage: The hook stage whose payload file to read.

    Returns:
        The payload's seq, last iteration, and session iteration count, in that
        order.
    """
    payload = json.loads((Path(experiment_dir) / f"{stage}.json").read_text(encoding="utf-8"))
    return (payload["seq"], payload["last_iteration"], payload["session"]["iteration_count"])


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
        2,
        as_logged(iteration_record(seq=1)),
        1,
    )
    assert _iteration_fields(experiment_dir, "after") == (2, as_logged(result.record), 2)


async def test_iterate_session_when_before_hook_fails_does_still_measure(
    hooks_setup: tuple[str, str, HookScripts],
):
    repo, _experiment_dir, hooks = hooks_setup
    before = hooks.failing_content_of("before-fails", "", "no warm copy\n")
    config = resolved_config(hooks=HooksConfig(before=before))

    await iterate_session(repo, config)

    assert [record.stage for record in records_of_type(repo, HookRecord)] == ["before"]
    assert last_iteration_of(repo).seq == 2


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

    await iterate_session(repo, config, options=IterateOptions(on_progress=events.append))

    assert [record.stage for record in records_of_type(repo, HookRecord)] == expected_stages
    assert _hook_events(events) == expected_events


# ---------------------------------------------------------------------------
# budget refusal: live budget + known estimate > remaining
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("history", "computed"),
    [
        pytest.param(
            settled_history(duration_ms=870_000),
            ("1m 30s", "14m 30s"),
            id="estimate-from-last-iteration",
        ),
        pytest.param(
            (baseline_record(duration_ms=435_000),),
            ("1m 30s", "7m 15s", "14m 30s"),
            id="estimate-from-baseline",
        ),
    ],
)
@pytest.mark.usefixtures("samples_mock")
async def test_iterate_session_when_budget_exceeded_does_refuse_naming_the_estimate(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    history: tuple[SessionLogRecord, ...],
    computed: tuple[str, ...],
):
    write_iterate_session(repo, history)
    install_budget(repo, monkeypatch, deadline_ms=90_000.0, frozen_now_ms=0)

    with pytest.raises(LoopStopError) as exc:
        await iterate_session(repo, resolved_config())

    assert [part for part in computed if part not in str(exc.value)] == []


# ---------------------------------------------------------------------------
# a configured stop condition is met
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("samples_mock")
async def test_iterate_session_when_stop_condition_met_does_refuse_naming_it(repo: str):
    write_iterate_session(repo, settled_history())
    config = resolved_config(stop=StopConfig(max_iterations=1))

    with pytest.raises(LoopStopError) as exc:
        await iterate_session(repo, config)

    assert exc.value.reason == "stop-condition"


# ---------------------------------------------------------------------------
# adapter warnings: a bench line the adapter cannot read
# ---------------------------------------------------------------------------


def _warnings_about(warnings: list[str], metric: str) -> list[str]:
    return [warning for warning in warnings if warning.startswith(f"{metric}: ")]


async def test_iterate_session_when_warn_sink_given_does_route_adapter_warnings_to_it(
    open_repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    bench_malformed_once(monkeypatch)
    warnings: list[str] = []

    await iterate_session(
        open_repo, resolved_config(), options=IterateOptions(warn=warnings.append)
    )

    assert MALFORMED_LINE_WARNING in warnings
    assert MALFORMED_LINE_WARNING not in capsys.readouterr().err


async def test_iterate_session_when_time_readings_look_quantized_does_warn_the_sink_once_across_the_confirmation(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    time_metric = f"bench{TIME_SUFFIX}"
    base = 333.0
    quantized_slower = PairedRun(
        experiment=[{time_metric: base + QUANTIZED_STEP_NS}] * 9
        + [{time_metric: base + 2 * QUANTIZED_STEP_NS}],
        baseline=[{time_metric: base}] * 9 + [{time_metric: base + QUANTIZED_STEP_NS}],
    )
    stub_runs(samples_mock, open_repo, [quantized_slower, quantized_slower])
    warnings: list[str] = []

    await iterate_session(
        open_repo, resolved_config(), options=IterateOptions(warn=warnings.append)
    )

    metric_warnings = _warnings_about(warnings, time_metric)
    assert samples_mock.call_count == 2
    assert [
        is_quantized_warning(warning, time_metric, QUANTIZED_GAP) for warning in metric_warnings
    ] == [True]


async def test_iterate_session_when_a_round_is_one_sided_does_send_the_dropped_window_warning_to_the_sink(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    baseline = baseline_rounds()
    del baseline[0]["total_ms"]
    stub_runs(samples_mock, open_repo, [PairedRun(improved_rounds(), baseline)])
    warnings: list[str] = []

    await iterate_session(
        open_repo, resolved_config(), options=IterateOptions(warn=warnings.append)
    )

    assert _warnings_about(warnings, "total_ms") != []
