"""Gating, confirmation-rerun, and hook behavior of ``iterate_session``.

The one boundary these tests mock is sampling; everything downstream — verdicts,
the confirmation rerun, aggregation, the record, the report, and the real hook
subprocesses — runs against a throwaway repository. The suite is
order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat_py.config import HooksConfig, MetricEntry
from gymrat_py.errors import GymratError
from gymrat_py.loop.iterate import iterate_session
from gymrat_py.session import (
    Confirm,
    HookRecord,
    IterationPrimary,
    IterationRecord,
    MetricVerdict,
    PairedSamples,
    read_records,
    session_jsonl_path,
)
from gymrat_py.session import append_record as append_session_record
from tests.loop._hooks import HookScripts, expected_hook_record
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
    SESSION_ID,
    committed_keep,
    iteration_record,
    write_session_log,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.loop._iterate import CollectSamplesRecorder

#: The confirm-rerun template a consumer configures when their bench can be narrowed.
FILTER = "npm run bench -- --filter {names}"


def _jittered(values: list[float], up: float, down: float) -> list[float]:
    """Nudge alternate rounds up by ``up`` and the rest down by ``down``.

    The mixed signs leave the permutation test nothing to call, while the larger
    upward nudge still drags the median above the baseline's — a run that moved
    the wrong way without saying anything, which is what ``no-signal`` means.
    """
    return [value + up if index % 2 == 0 else value - down for index, value in enumerate(values)]


def _regressed_rounds() -> list[dict[str, float]]:
    """Ten rounds 10% slower and 10% fatter than the baseline's."""
    return rounds(scaled(BASELINE_MS, 1.1), scaled(BASELINE_BYTES, 1.1))


def _noisy_rounds() -> list[dict[str, float]]:
    """Ten rounds that drift half a percent the wrong way without ever settling."""
    return rounds(_jittered(BASELINE_MS, 2, 1), _jittered(BASELINE_BYTES, 20, 10))


def _filtered_rounds(name: str, values: list[float]) -> list[dict[str, float]]:
    """Rounds reporting ``name`` alone, the shape a bench filtered to that metric reports."""
    return [{name: value} for value in values]


def _plain(report: str) -> str:
    """The report stripped of color, as a terminal's visible text would read."""
    return "\n".join(trimmed_report_lines(report))


def _primary_line(report: str) -> str:
    """The report's ``primary:`` line, failing when there is none."""
    for line in trimmed_report_lines(report):
        if line.startswith("primary:"):
            return line
    message = f"no primary line in report:\n{report}"
    raise AssertionError(message)


def _assert_permutation(
    metric: MetricVerdict, *, delta: float, verdict: str, confirmed: bool
) -> None:
    """Assert a permutation metric moved ``delta`` percent and settled as ``verdict``."""
    assert metric.delta_pct == pytest.approx(delta, abs=1e-6)
    assert metric.verdict == verdict
    assert metric.method == "permutation"
    assert metric.p is not None
    assert metric.noise_pct is not None
    assert metric.gating is True
    assert metric.confirmed is confirmed


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str]) -> str:
    return create_scratch_repo()


@pytest.fixture
def samples_mock(monkeypatch: pytest.MonkeyPatch):
    return install_collect_samples(monkeypatch)


@pytest.fixture
def open_repo(repo: str, samples_mock: CollectSamplesRecorder):
    """A fresh open session on disk, no history, sampling left for the test to stub."""
    write_session_log(repo, session_record(repo))
    return repo


# ---------------------------------------------------------------------------
# a gating metric comes back regressed
# ---------------------------------------------------------------------------


async def test_iterate_session_when_gating_regression_does_rerun_through_the_filter_template(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    stub_runs(
        samples_mock,
        open_repo,
        [
            PairedRun(_regressed_rounds(), baseline_rounds()),
            PairedRun(_regressed_rounds(), baseline_rounds()),
        ],
    )

    await iterate_session(open_repo, resolved_config(filter=FILTER))

    assert samples_mock.call_count == 2
    assert sampling_call(samples_mock, 1).targets == sampling_call(samples_mock, 0).targets
    assert sampling_call(samples_mock, 1).bench == "npm run bench -- --filter total_ms alloc_bytes"


async def test_iterate_session_when_no_filter_configured_does_rerun_the_whole_bench(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    stub_runs(
        samples_mock,
        open_repo,
        [
            PairedRun(_regressed_rounds(), baseline_rounds()),
            PairedRun(_regressed_rounds(), baseline_rounds()),
        ],
    )

    await iterate_session(open_repo, resolved_config())

    assert sampling_call(samples_mock, 1).bench == "npm run bench"


async def test_iterate_session_when_gating_regression_does_record_the_rerun_raw_samples(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    rerun = PairedRun(
        experiment=_filtered_rounds("total_ms", scaled(BASELINE_MS, 1.2)),
        baseline=_filtered_rounds("total_ms", BASELINE_MS),
    )
    stub_runs(samples_mock, open_repo, [PairedRun(_regressed_rounds(), baseline_rounds()), rerun])
    resolved = resolved_config(filter=FILTER, metrics={"alloc_bytes": MetricEntry(gating=False)})

    result = await iterate_session(open_repo, resolved)

    assert result.record.confirm == Confirm(
        ran=True,
        filtered=("total_ms",),
        samples=PairedSamples(experiment=tuple(rerun.experiment), baseline=tuple(rerun.baseline)),
    )


async def test_iterate_session_when_rerun_agrees_does_confirm_and_read_regressed(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    stub_runs(
        samples_mock,
        open_repo,
        [
            PairedRun(_regressed_rounds(), baseline_rounds()),
            PairedRun(
                _filtered_rounds("total_ms", scaled(BASELINE_MS, 1.2)),
                _filtered_rounds("total_ms", BASELINE_MS),
            ),
        ],
    )
    resolved = resolved_config(filter=FILTER, metrics={"alloc_bytes": MetricEntry(gating=False)})

    result = await iterate_session(open_repo, resolved)

    _assert_permutation(
        result.record.metrics["total_ms"], delta=10, verdict="regressed", confirmed=True
    )
    assert result.record.outcome == "regressed"
    assert "total_ms: regression confirmed on rerun" in trimmed_report_lines(result.report)


@pytest.mark.parametrize(
    "experiment",
    [
        pytest.param(scaled(BASELINE_MS, 0.9), id="improved"),
        pytest.param(_jittered(BASELINE_MS, 2, 1), id="no-signal"),
    ],
)
async def test_iterate_session_when_rerun_disagrees_does_demote_to_no_signal(
    open_repo: str, samples_mock: CollectSamplesRecorder, experiment: list[float]
):
    stub_runs(
        samples_mock,
        open_repo,
        [
            PairedRun(_regressed_rounds(), baseline_rounds()),
            PairedRun(
                _filtered_rounds("total_ms", experiment),
                _filtered_rounds("total_ms", BASELINE_MS),
            ),
        ],
    )
    resolved = resolved_config(filter=FILTER, metrics={"alloc_bytes": MetricEntry(gating=False)})

    result = await iterate_session(open_repo, resolved)

    _assert_permutation(
        result.record.metrics["total_ms"], delta=10, verdict="no-signal", confirmed=False
    )
    assert result.record.outcome == "no-signal"
    assert "total_ms: regression not confirmed on rerun" in trimmed_report_lines(result.report)


async def test_iterate_session_when_rerun_bench_fails_does_fail_and_record_nothing(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    stub_runs(
        samples_mock,
        open_repo,
        [PairedRun(_regressed_rounds(), baseline_rounds()), GymratError("bench command failed")],
    )
    resolved = resolved_config(filter=FILTER, metrics={"alloc_bytes": MetricEntry(gating=False)})

    with pytest.raises(GymratError) as exc:
        await iterate_session(open_repo, resolved)

    assert str(exc.value) == "bench command failed"
    assert len(read_records(session_jsonl_path(open_repo))) == 1


# The filter command reaches a POSIX shell, which is what decides where one
# argument ends and the next begins; win32 is skipped for the same reason the
# exec suite is.
_ARGS_SCRIPT = '#!/bin/sh\nfor arg in "$@"; do\n  echo "$arg"\ndone\n'
_ARGS_FILTER = "sh args.sh {names}"


def _paired_with(name: str, values: list[float]) -> list[dict[str, float]]:
    """One round per entry, reporting ``name`` beside a plainly named metric."""
    return [{name: value, "total_ms": value} for value in values]


def _shell_args(directory: str, command: str) -> list[str]:
    """The arguments a POSIX shell hands the stand-in bench when it runs ``command``.

    The rerun's bench string is handed to a shell verbatim, so running it through
    one is the only assertion that speaks to what the bench is really given — a
    string comparison would pass for a command the shell refuses outright.
    """
    printed = subprocess.run(  # noqa: S603
        ["sh", "-c", command],  # noqa: S607
        cwd=directory,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in printed.split("\n") if line]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell argument splitting only")
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("sort(n=1000)/time", id="parentheses-and-equals"),
        pytest.param("decode large payload", id="space"),
        pytest.param("o'clock/time", id="single-quote"),  # cspell:disable-line
    ],
)
async def test_iterate_session_when_metric_name_needs_quoting_does_pass_it_as_one_argument(
    open_repo: str, samples_mock: CollectSamplesRecorder, name: str
):
    (Path(open_repo) / "args.sh").write_text(_ARGS_SCRIPT, encoding="utf-8")
    regressed = PairedRun(
        experiment=_paired_with(name, scaled(BASELINE_MS, 1.1)),
        baseline=_paired_with(name, BASELINE_MS),
    )
    stub_runs(samples_mock, open_repo, [regressed, regressed])

    await iterate_session(open_repo, resolved_config(filter=_ARGS_FILTER))

    assert _shell_args(open_repo, sampling_call(samples_mock, 1).bench) == [name, "total_ms"]


async def test_iterate_session_when_win32_does_double_quote_metric_names_in_filter(
    open_repo: str,
    samples_mock: CollectSamplesRecorder,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sys, "platform", "win32")
    name = "decode large payload"
    regressed = PairedRun(
        experiment=_paired_with(name, scaled(BASELINE_MS, 1.1)),
        baseline=_paired_with(name, BASELINE_MS),
    )
    stub_runs(samples_mock, open_repo, [regressed, regressed])

    await iterate_session(open_repo, resolved_config(filter=_ARGS_FILTER))

    bench = sampling_call(samples_mock, 1).bench
    # On win32 the metric name is double-quoted (cmd.exe convention), not
    # single-quoted the way shlex.quote would produce for a POSIX shell.
    assert f'"{name}"' in bench
    assert f"'{name}'" not in bench


# ---------------------------------------------------------------------------
# the rerun never measures one of the regressed metrics
# ---------------------------------------------------------------------------


def _partial_rerun() -> PairedRun:
    """The rerun a filtered bench reports when it only ever emits ``total_ms``.

    ``alloc_bytes`` was regressed on the first run and named in the filter, but
    the rerun comes back without it — silence, not disagreement.
    """
    return PairedRun(
        experiment=_filtered_rounds("total_ms", scaled(BASELINE_MS, 1.2)),
        baseline=_filtered_rounds("total_ms", BASELINE_MS),
    )


@pytest.fixture
def partial_rerun_repo(open_repo: str, samples_mock: CollectSamplesRecorder):
    partial = _partial_rerun()
    stub_runs(
        samples_mock,
        open_repo,
        [PairedRun(_regressed_rounds(), baseline_rounds()), partial],
    )
    return open_repo


async def test_iterate_session_when_rerun_silent_on_metric_does_leave_it_regressed(
    partial_rerun_repo: str,
):
    result = await iterate_session(partial_rerun_repo, resolved_config(filter=FILTER))

    _assert_permutation(
        result.record.metrics["alloc_bytes"], delta=10, verdict="regressed", confirmed=False
    )
    assert result.record.outcome == "regressed"


async def test_iterate_session_when_rerun_silent_on_metric_does_name_it_in_confirm(
    partial_rerun_repo: str,
):
    partial = _partial_rerun()
    expected = Confirm(
        ran=True,
        filtered=("total_ms", "alloc_bytes"),
        samples=PairedSamples(
            experiment=tuple(partial.experiment), baseline=tuple(partial.baseline)
        ),
        absent=("alloc_bytes",),
    )

    result = await iterate_session(partial_rerun_repo, resolved_config(filter=FILTER))

    assert result.record.confirm == expected
    assert last_iteration_of(partial_rerun_repo).confirm == expected


async def test_iterate_session_when_rerun_silent_on_metric_does_report_it_as_unmeasured(
    partial_rerun_repo: str,
):
    result = await iterate_session(partial_rerun_repo, resolved_config(filter=FILTER))

    lines = trimmed_report_lines(result.report)
    assert "alloc_bytes: not measured on rerun" in lines
    assert "total_ms: regression confirmed on rerun" in lines
    assert "alloc_bytes: regression not confirmed on rerun" not in lines


# ---------------------------------------------------------------------------
# the confirmation rerun produces no parsable metrics at all
# ---------------------------------------------------------------------------


async def test_iterate_session_when_rerun_produces_no_parsable_metrics_does_treat_all_filtered_as_absent(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    empty_rerun = PairedRun([{} for _ in range(10)], [{} for _ in range(10)])
    stub_runs(
        samples_mock,
        open_repo,
        [PairedRun(_regressed_rounds(), baseline_rounds()), empty_rerun],
    )

    result = await iterate_session(open_repo, resolved_config(filter=FILTER))

    assert result.record.confirm is not None
    assert result.record.confirm.ran is True
    assert result.record.confirm.absent is not None
    assert set(result.record.confirm.absent) == {"total_ms", "alloc_bytes"}
    # The iteration carries the first run's samples, not the empty rerun's.
    assert result.record.samples == PairedSamples(
        experiment=tuple(_regressed_rounds()),
        baseline=tuple(baseline_rounds()),
    )
    # Both regressions stand (the gate fails closed on absent metrics).
    assert result.record.metrics["total_ms"].verdict == "regressed"
    assert result.record.metrics["alloc_bytes"].verdict == "regressed"
    assert result.record.outcome == "regressed"


# ---------------------------------------------------------------------------
# a regressed metric a rerun cannot inform
# ---------------------------------------------------------------------------


async def test_iterate_session_when_metric_is_exact_does_gate_on_the_first_run_alone(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    stub_samples(samples_mock, open_repo, _regressed_rounds(), baseline_rounds())
    resolved = resolved_config(
        metrics={"total_ms": MetricEntry(exact=True), "alloc_bytes": MetricEntry(gating=False)}
    )

    result = await iterate_session(open_repo, resolved)

    assert samples_mock.call_count == 1
    total = result.record.metrics["total_ms"]
    assert total.delta_pct == pytest.approx(10, abs=1e-6)
    assert total.verdict == "regressed"
    assert total.method == "exact"
    assert total.gating is True
    assert total.confirmed is False
    assert total.p is None
    assert total.noise_pct is None
    assert result.record.outcome == "regressed"


async def test_iterate_session_when_metric_is_exact_does_leave_it_out_of_the_filter_list(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    stub_runs(
        samples_mock,
        open_repo,
        [
            PairedRun(_regressed_rounds(), baseline_rounds()),
            PairedRun(
                _filtered_rounds("alloc_bytes", scaled(BASELINE_BYTES, 1.2)),
                _filtered_rounds("alloc_bytes", BASELINE_BYTES),
            ),
        ],
    )

    await iterate_session(
        open_repo, resolved_config(filter=FILTER, metrics={"total_ms": MetricEntry(exact=True)})
    )

    assert sampling_call(samples_mock, 1).bench == "npm run bench -- --filter alloc_bytes"


async def test_iterate_session_when_metric_is_non_gating_does_inform_without_rerunning(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    experiment = rounds(scaled(BASELINE_MS, 0.9), scaled(BASELINE_BYTES, 1.1))
    stub_samples(samples_mock, open_repo, experiment, baseline_rounds())

    result = await iterate_session(
        open_repo, resolved_config(metrics={"alloc_bytes": MetricEntry(gating=False)})
    )

    assert samples_mock.call_count == 1
    assert result.record.confirm is None
    assert result.record.metrics["alloc_bytes"].verdict == "regressed"
    assert result.record.outcome == "improved"


# ---------------------------------------------------------------------------
# the bench names a metric after an Object.prototype member
# ---------------------------------------------------------------------------

_PROTO = "__proto__"


def _proto_rounds(total_ms: list[float], proto: list[float]) -> list[dict[str, float]]:
    """One round per entry, pairing ``total_ms`` with the metric named ``__proto__``."""
    return [
        {"total_ms": value, _PROTO: proto[index] if index < len(proto) else 0}
        for index, value in enumerate(total_ms)
    ]


@pytest.fixture
def proto_repo(open_repo: str, samples_mock: CollectSamplesRecorder):
    stub_samples(
        samples_mock,
        open_repo,
        _proto_rounds(scaled(BASELINE_MS, 0.9), scaled(BASELINE_BYTES, 0.8)),
        _proto_rounds(BASELINE_MS, BASELINE_BYTES),
    )
    return open_repo


async def test_iterate_session_when_metric_named_proto_does_keep_it_as_an_own_key(
    proto_repo: str,
):
    result = await iterate_session(proto_repo, resolved_config())

    assert set(result.record.metrics.keys()) == {"total_ms", _PROTO}
    _assert_permutation(
        result.record.metrics["total_ms"], delta=-10, verdict="improved", confirmed=False
    )
    _assert_permutation(
        result.record.metrics[_PROTO], delta=-20, verdict="improved", confirmed=False
    )


async def test_iterate_session_when_metric_named_proto_does_count_it_in_the_geomean(
    proto_repo: str,
):
    result = await iterate_session(proto_repo, resolved_config())

    assert result.record.primary.kind == "geomean"
    assert result.record.primary.name is None
    assert result.record.primary.delta_pct == pytest.approx(-15.1472, abs=1e-3)


# ---------------------------------------------------------------------------
# a metric's baseline median is zero
# ---------------------------------------------------------------------------


def _zero_baseline_rounds() -> list[dict[str, float]]:
    """The baseline's ten rounds, but with ``total_ms`` flat at zero."""
    return rounds([0.0 for _ in BASELINE_MS], BASELINE_BYTES)


def _undefined_delta_iteration(seq: int) -> IterationRecord:
    """The iteration numbered ``seq``, its deltas the nulls a zero median yields."""
    return iteration_record(
        seq=seq,
        metrics={
            "total_ms": MetricVerdict(
                delta_pct=None,
                verdict="no-signal",
                method="permutation",
                p=0.005,
                noise_pct=3,
                gating=True,
                confirmed=False,
            )
        },
        primary=IterationPrimary(kind="geomean", delta_pct=None),
        outcome="no-signal",
    )


@pytest.fixture
def zero_baseline_repo(open_repo: str, samples_mock: CollectSamplesRecorder):
    stub_samples(samples_mock, open_repo, improved_rounds(), _zero_baseline_rounds())
    return open_repo


async def test_iterate_session_when_baseline_median_zero_does_record_null_delta(
    zero_baseline_repo: str,
):
    result = await iterate_session(zero_baseline_repo, resolved_config())

    assert result.record.metrics["total_ms"].delta_pct is None
    assert result.record.metrics["alloc_bytes"].delta_pct == pytest.approx(-20, abs=1e-6)


async def test_iterate_session_when_baseline_median_zero_does_stay_readable_off_the_log(
    zero_baseline_repo: str,
):
    result = await iterate_session(zero_baseline_repo, resolved_config())

    assert as_logged(last_iteration_of(zero_baseline_repo)) == as_logged(result.record)


async def test_iterate_session_when_baseline_median_zero_does_record_primary_delta_null_and_state_no_percentage(
    zero_baseline_repo: str,
):
    result = await iterate_session(zero_baseline_repo, resolved_config(primary="total_ms"))

    assert result.record.primary == IterationPrimary(kind="metric", name="total_ms", delta_pct=None)
    assert result.record.outcome == "no-signal"

    primary = _primary_line(result.report)
    assert "verdict: NO-SIGNAL" in primary
    assert not re.search(r"NaN|null|%", primary)


async def test_iterate_session_when_log_holds_null_delta_does_measure_again(
    zero_baseline_repo: str,
):
    append_session_record(session_jsonl_path(zero_baseline_repo), _undefined_delta_iteration(1))
    append_session_record(session_jsonl_path(zero_baseline_repo), committed_keep(1))

    result = await iterate_session(zero_baseline_repo, resolved_config())

    assert result.record.seq == 2


# ---------------------------------------------------------------------------
# the named primary metric is one the bench never reported
# ---------------------------------------------------------------------------


@pytest.fixture
def improved_repo(open_repo: str, samples_mock: CollectSamplesRecorder):
    stub_samples(samples_mock, open_repo, improved_rounds(), baseline_rounds())
    return open_repo


async def test_iterate_session_when_primary_never_reported_does_record_delta_null_and_state_no_percentage(
    improved_repo: str,
):
    result = await iterate_session(improved_repo, resolved_config(primary="startup_ms"))

    assert result.record.primary == IterationPrimary(
        kind="metric", name="startup_ms", delta_pct=None
    )
    assert result.record.outcome == "no-signal"

    assert _primary_line(result.report) == "primary: · verdict: NO-SIGNAL"


# ---------------------------------------------------------------------------
# the geomean primary has no qualifying inputs
# ---------------------------------------------------------------------------


async def test_iterate_session_when_geomean_all_non_gating_does_record_null_delta_and_state_no_percentage(
    improved_repo: str,
):
    result = await iterate_session(
        improved_repo,
        resolved_config(
            metrics={
                "total_ms": MetricEntry(gating=False),
                "alloc_bytes": MetricEntry(gating=False),
            }
        ),
    )

    assert result.record.primary == IterationPrimary(kind="geomean", delta_pct=None)

    primary = _primary_line(result.report)
    assert "NO-SIGNAL" in primary
    assert not re.search(r"NaN|0\.0%", primary)


async def test_iterate_session_when_geomean_all_unstable_does_record_null_delta(
    improved_repo: str,
):
    result = await iterate_session(improved_repo, resolved_config(unstable_noise_pct=0.0))

    assert result.record.primary == IterationPrimary(kind="geomean", delta_pct=None)


# ---------------------------------------------------------------------------
# the named primary metric came back exactly where it started
# ---------------------------------------------------------------------------


async def test_iterate_session_when_primary_flat_does_record_zero_as_a_percentage(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    stub_samples(samples_mock, open_repo, baseline_rounds(), baseline_rounds())

    result = await iterate_session(open_repo, resolved_config(primary="total_ms"))

    assert result.record.primary == IterationPrimary(kind="metric", name="total_ms", delta_pct=0)
    assert _primary_line(result.report) == "primary: 0.0% · verdict: NO-SIGNAL"


# ---------------------------------------------------------------------------
# the report closes on the outcome's verdict and next step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "word", "experiment", "next_step"),
    [
        pytest.param("improved", "IMPROVED", improved_rounds(), "gymrat keep", id="improved"),
        pytest.param(
            "regressed",
            "REGRESSED",
            _regressed_rounds(),
            "fix or run gymrat discard",
            id="regressed",
        ),
        pytest.param(
            "no-signal",
            "NO-SIGNAL",
            _noisy_rounds(),
            "gymrat keep or gymrat discard",
            id="no-signal",
        ),
    ],
)
async def test_iterate_session_when_outcome_settles_does_close_report_on_verdict_and_next_step(
    open_repo: str,
    samples_mock: CollectSamplesRecorder,
    *,
    outcome: str,
    word: str,
    experiment: list[dict[str, float]],
    next_step: str,
):
    stub_samples(samples_mock, open_repo, experiment, baseline_rounds())

    result = await iterate_session(open_repo, resolved_config())

    lines = _plain(result.report).split("\n")
    assert result.record.outcome == outcome
    assert word in lines[-2]
    assert lines[-1] == next_step


# ---------------------------------------------------------------------------
# the config declares a command for a stage (hooks)
# ---------------------------------------------------------------------------


def _capturing_payload(hooks: HookScripts, stage: str) -> str:
    """A hook command filing the payload it was handed away where assertions can read it."""
    body = (
        "import sys, pathlib\n"
        "data = sys.stdin.buffer.read()\n"
        f"pathlib.Path({json.dumps(stage + '.json')}).write_bytes(data)\n"
    )
    return hooks.hook_command(body)


def _payload_of(experiment_dir: str, stage: str) -> object:
    """The payload the ``stage`` hook was handed, as the hook itself saw it.

    The capturing command names the file relatively, so reading it back out of
    the experiment worktree is also what proves the hook ran there.
    """
    return json.loads((Path(experiment_dir) / f"{stage}.json").read_text(encoding="utf-8"))


def _hook_records(root: str) -> list[HookRecord]:
    """Every hook record the session log holds, oldest first."""
    return [
        record
        for record in read_records(session_jsonl_path(root))
        if isinstance(record, HookRecord)
    ]


@pytest.fixture
def hooks_setup(repo: str, samples_mock: CollectSamplesRecorder):
    experiment_dir = session_record(repo).worktrees.experiment
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    scripts = HookScripts(repo, experiment_dir)
    write_session_log(repo, session_record(repo), (iteration(1), committed_keep(1)))
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())
    return repo, experiment_dir, scripts


async def test_iterate_session_when_hooks_configured_does_fire_before_then_after(
    hooks_setup: tuple[str, str, HookScripts],
):
    repo, _experiment_dir, hooks = hooks_setup
    config = resolved_config(
        hooks=HooksConfig(before=hooks.printing("hi"), after=hooks.printing("bye"))
    )

    await iterate_session(repo, config)

    records = read_records(session_jsonl_path(repo))
    assert [record.type for record in records] == [
        "session",
        "iteration",
        "keep",
        "hook",
        "iteration",
        "hook",
    ]
    hook_records = [record for record in records if isinstance(record, HookRecord)]
    assert [dataclasses.replace(record, duration_ms=0) for record in hook_records] == [
        expected_hook_record(stage="before", seq=2, exit_code=0, stdout_bytes=3),
        expected_hook_record(stage="after", seq=2, exit_code=0, stdout_bytes=4),
    ]


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

    session_payload = {
        "sessionId": SESSION_ID,
        "baseline": {"ref": "main", "sha": "a" * 40},
        "branch": f"gymrat/{SESSION_ID}",
    }
    assert _payload_of(experiment_dir, "before") == {
        "stage": "before",
        "experimentDir": experiment_dir,
        "seq": 2,
        "lastIteration": as_logged(iteration(1)),
        "session": {**session_payload, "iterationCount": 1},
    }
    assert _payload_of(experiment_dir, "after") == {
        "stage": "after",
        "experimentDir": experiment_dir,
        "seq": 2,
        "lastIteration": as_logged(result.record),
        "session": {**session_payload, "iterationCount": 2},
    }


async def test_iterate_session_when_hooks_configured_does_print_output_around_the_measurement(
    hooks_setup: tuple[str, str, HookScripts],
):
    repo, _experiment_dir, hooks = hooks_setup
    config = resolved_config(
        hooks=HooksConfig(
            before=hooks.printing("warmed the cache"), after=hooks.printing("archived the samples")
        )
    )

    result = await iterate_session(repo, config)

    lines = trimmed_report_lines(result.report)
    assert lines[0] == "[before] warmed the cache"
    assert lines[-1] == "[after] archived the samples"
    assert lines[1] == "iteration 2 · experiment vs baseline · 10 paired samples"


async def test_iterate_session_when_before_hook_fails_does_measure_on_reporting_the_failure(
    hooks_setup: tuple[str, str, HookScripts],
):
    repo, _experiment_dir, hooks = hooks_setup
    before = hooks.hook_command(
        'import sys\nsys.stderr.buffer.write(b"no warm copy\\n")\nsys.exit(3)\n'
    )
    config = resolved_config(hooks=HooksConfig(before=before))

    result = await iterate_session(repo, config)

    assert trimmed_report_lines(result.report)[:2] == [
        "[before] hook exited 3",
        "[before] no warm copy",
    ]
    hook_records = [dataclasses.replace(record, duration_ms=0) for record in _hook_records(repo)]
    assert hook_records == [
        expected_hook_record(stage="before", seq=2, exit_code=3, stdout_bytes=0, stderr_bytes=13)
    ]
    assert last_iteration_of(repo).seq == 2
    assert result.record.outcome == "improved"


@pytest.mark.parametrize(
    ("with_after", "expected_stages"),
    [
        pytest.param(False, [], id="no-hooks"),
        pytest.param(True, ["after"], id="only-after"),
    ],
)
async def test_iterate_session_when_before_stage_absent_does_run_nothing_for_it(
    hooks_setup: tuple[str, str, HookScripts], with_after: bool, expected_stages: list[str]
):
    repo, _experiment_dir, hooks = hooks_setup
    hooks_config = HooksConfig(after=hooks.printing("bye")) if with_after else None
    config = resolved_config(hooks=hooks_config)

    result = await iterate_session(repo, config)

    assert [record.stage for record in _hook_records(repo)] == expected_stages
    before_lines = [
        line for line in trimmed_report_lines(result.report) if line.startswith("[before]")
    ]
    assert before_lines == []
