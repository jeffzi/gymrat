"""Behavioral tests for the sampling core and progress reporting.

They also cover the per-metric median and spread computed from collected
samples.
"""

import asyncio
import itertools

import pytest

from gymrat.adapters.metric_lines import metric_lines_adapter
from gymrat.config import MetricEntry
from gymrat.errors import CommandError, GymratError
from gymrat.exec import OUTPUT_CAP, ExecResult, ExecTimeoutError
from gymrat.model import GEOMEAN_DIRECTION
from gymrat.progress_events import (
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressEvent,
)
from gymrat.sampling import (
    RunOptions,
    SamplingOptions,
    TargetContext,
    collect_samples,
    compute_metric_stats,
    primary_direction,
)
from gymrat.targets import InPlaceTarget, RefTarget
from tests._config import resolved_config
from tests._exec_fixtures import (
    CAPPED_STDERR_BYTES,
    CAPPED_STDOUT_BYTES,
    capped_result,
    expected_result,
    install_exec,
)


def make_success(stdout: str = "METRIC x=1") -> ExecResult:
    """Build a zero-exit result carrying ``stdout`` on the standard stream."""
    return expected_result(stdout)


#: The ``exec`` the sampling module calls, which every test here replaces.
SAMPLING_EXEC = "gymrat.sampling.exec"


def two_in_place_targets() -> list[TargetContext]:
    """Two in-place targets labelled old/new, rooted at distinct directories."""
    return [
        TargetContext(
            target=InPlaceTarget(dir="/a"),
            dir="/a",
            label="old",
            position="old",
        ),
        TargetContext(
            target=InPlaceTarget(dir="/b"),
            dir="/b",
            label="new",
            position="new",
        ),
    ]


def one_in_place_target() -> list[TargetContext]:
    """Single in-place target at /a, labelled old, in the old position."""
    return [
        TargetContext(target=InPlaceTarget(dir="/a"), dir="/a", label="old", position="old"),
    ]


def lettered_in_place_targets(count: int) -> list[TargetContext]:
    """``count`` in-place targets labelled a, b, c, ..., each rooted at ``/<label>``."""
    return [
        TargetContext(target=InPlaceTarget(dir=f"/{name}"), dir=f"/{name}", label=name)
        for name in "abcdefgh"[:count]
    ]


def numbered_sample(call: int) -> ExecResult:
    """Answer a bench run with ``METRIC x=<call>``, so each sample names the run it came from."""
    return make_success(f"METRIC x={call}")


# ---------------------------------------------------------------------------
# collect_samples
# ---------------------------------------------------------------------------


async def test_collect_samples_when_prepare_given_does_run_it_once_per_target_before_benching_under_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = install_exec(monkeypatch, SAMPLING_EXEC, make_success("METRIC x=1"))
    options = SamplingOptions(bench="run", prepare="prep", samples=2, timeout_seconds=2.5)

    await collect_samples(metric_lines_adapter, two_in_place_targets(), options, asyncio.Event())

    assert [(command, exec_options.cwd) for command, exec_options in recorder.calls] == [
        ("prep", "/a"),
        ("prep", "/b"),
        ("run", "/a"),
        ("run", "/b"),
        ("run", "/b"),
        ("run", "/a"),
    ]
    assert {exec_options.timeout_ms for _, exec_options in recorder.calls} == {2500}


async def test_collect_samples_when_two_targets_does_alternate_which_target_runs_first(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = install_exec(monkeypatch, SAMPLING_EXEC, numbered_sample).calls
    options = SamplingOptions(bench="run", prepare=None, samples=4, timeout_seconds=1.0)

    result = await collect_samples(
        metric_lines_adapter, two_in_place_targets(), options, asyncio.Event()
    )

    assert [exec_options.cwd for _, exec_options in calls] == [
        "/a",
        "/b",
        "/b",
        "/a",
        "/a",
        "/b",
        "/b",
        "/a",
    ]
    assert [[sample["x"] for sample in ts.samples] for ts in result] == [
        [1.0, 4.0, 5.0, 8.0],
        [2.0, 3.0, 6.0, 7.0],
    ]


#: Target counts of three or more with the shortest sample count that balances each.
_BALANCING_CYCLES = [
    pytest.param(4, 4, id="four-targets"),
    pytest.param(5, 10, id="five-targets"),
]


async def _collect_run_orders(
    monkeypatch: pytest.MonkeyPatch, target_count: int, samples: int
) -> tuple[list[list[str]], list[str]]:
    """Sample ``target_count`` lettered targets, returning each round's run order and the dirs."""
    calls = install_exec(monkeypatch, SAMPLING_EXEC, numbered_sample).calls
    targets = lettered_in_place_targets(target_count)
    options = SamplingOptions(bench="run", prepare=None, samples=samples, timeout_seconds=1.0)

    await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    ran = [exec_options.cwd for _, exec_options in calls]
    rounds = [ran[start : start + target_count] for start in range(0, len(ran), target_count)]
    return rounds, [ctx.dir for ctx in targets]


@pytest.mark.parametrize(("target_count", "cycle"), _BALANCING_CYCLES)
async def test_collect_samples_when_three_or_more_targets_does_give_each_target_every_position_equally_often(
    monkeypatch: pytest.MonkeyPatch, target_count: int, cycle: int
):
    rounds, dirs = await _collect_run_orders(monkeypatch, target_count, cycle)

    assert all(sorted(run_order) == dirs for run_order in rounds)
    slots = list(itertools.product(range(target_count), dirs))
    assert {
        (position, target): sum(run_order[position] == target for run_order in rounds)
        for position, target in slots
    } == dict.fromkeys(slots, cycle // target_count)


@pytest.mark.parametrize(("target_count", "cycle"), _BALANCING_CYCLES)
async def test_collect_samples_when_three_or_more_targets_does_run_each_target_after_every_other_equally_often(
    monkeypatch: pytest.MonkeyPatch, target_count: int, cycle: int
):
    rounds, dirs = await _collect_run_orders(monkeypatch, target_count, cycle)

    followings = [pair for run_order in rounds for pair in itertools.pairwise(run_order)]
    ordered_pairs = list(itertools.permutations(dirs, 2))
    assert {pair: followings.count(pair) for pair in ordered_pairs} == dict.fromkeys(
        ordered_pairs, cycle // target_count
    )


async def test_collect_samples_when_count_cannot_balance_the_targets_does_refuse_before_any_command_runs(
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = install_exec(monkeypatch, SAMPLING_EXEC, make_success())
    options = SamplingOptions(bench="run", prepare="prep", samples=3, timeout_seconds=1.0)

    with pytest.raises(GymratError):
        await collect_samples(
            metric_lines_adapter, lettered_in_place_targets(2), options, asyncio.Event()
        )

    assert recorder.calls == []


async def test_collect_samples_when_progress_given_does_emit_every_event_stamped_from_the_clock(
    monkeypatch: pytest.MonkeyPatch,
):
    install_exec(monkeypatch, SAMPLING_EXEC, make_success())
    events: list[ProgressEvent] = []
    clock_ms = 0.0

    def tick() -> float:
        nonlocal clock_ms
        clock_ms += 100
        return clock_ms

    options = SamplingOptions(
        bench="run",
        prepare="prep",
        samples=2,
        timeout_seconds=1.0,
        on_progress=events.append,
        clock=tick,
    )

    await collect_samples(metric_lines_adapter, two_in_place_targets(), options, asyncio.Event())

    assert events == [
        PrepareStarted("old", 100.0),
        PrepareFinished("old", 200.0),
        PrepareStarted("new", 300.0),
        PrepareFinished("new", 400.0),
        PassStarted(1, 2, 2, "old", 500.0),
        PassFinished(1, 2, 2, "old", 600.0),
        PassStarted(1, 2, 2, "new", 700.0),
        PassFinished(1, 2, 2, "new", 800.0),
        PassStarted(2, 2, 2, "new", 900.0),
        PassFinished(2, 2, 2, "new", 1000.0),
        PassStarted(2, 2, 2, "old", 1100.0),
        PassFinished(2, 2, 2, "old", 1200.0),
    ]


_IN_PLACE_NEW = TargetContext(
    target=InPlaceTarget(dir="/work"), dir="/work", label="main", position="new"
)
_REF_OLD = TargetContext(
    target=RefTarget(ref="feature", resolved_sha="deadbeef"),
    dir="/wt",
    label="base",
    position="old",
)
_EXIT_THREE = ExecResult(stdout="", stderr="boom", exit_code=3, stdout_bytes=4, stderr_bytes=4)
_TIMED_OUT = ExecTimeoutError(
    stdout="partial", stderr="", timeout_ms=1500, stdout_bytes=7, stderr_bytes=0
)
# A timeout whose stdout also overran the cap: the timeout is the diagnosis.
_TIMED_OUT_OVER_CAP = ExecTimeoutError(
    stdout="partial",
    stderr="",
    timeout_ms=1500,
    stdout_bytes=OUTPUT_CAP + 1,
    stderr_bytes=0,
    stdout_truncated=True,
)
_BENCH_ONLY = SamplingOptions(bench="run-bench", prepare=None, samples=1, timeout_seconds=1.5)
_WITH_PREPARE = SamplingOptions(bench="run", prepare="setup", samples=1, timeout_seconds=1.5)
_IN_PLACE_X = TargetContext(target=InPlaceTarget(dir="/work"), dir="/work", label="x")
_RUN_ONLY = SamplingOptions(bench="run", prepare=None, samples=1, timeout_seconds=1.0)


#: What a failed run of bench ``run`` in ``/work`` inserts into the error's opening lines.
_FAILED_HEAD = ("x", "1", "/work", "run", "1")

#: The capture cap, in MiB, that the stdout-overflow hint inserts.
_CAP_MIB = str(OUTPUT_CAP // (1024 * 1024))


def _failed(
    stdout: str,
    stderr: str,
    stdout_bytes: int,
    stderr_bytes: int,
    *,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
) -> ExecResult:
    return ExecResult(
        stdout, stderr, 1, stdout_bytes, stderr_bytes, stdout_truncated, stderr_truncated
    )


def _holds_in_order(text: str, parts: tuple[str, ...]) -> bool:
    position = 0
    for part in parts:
        found = text.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return True


@pytest.mark.parametrize(
    ("result", "target", "options", "inserted", "line_count", "hint_inserts"),
    [
        pytest.param(
            _EXIT_THREE,
            _IN_PLACE_NEW,
            _BENCH_ONLY,
            ("new", "main", "1", "/work", "run-bench", "3", "boom"),
            5,
            None,
            id="bench-fails-in-a-dir",
        ),
        pytest.param(
            _TIMED_OUT,
            _IN_PLACE_NEW,
            _BENCH_ONLY,
            ("new", "main", "1", "/work", "run-bench", "1500", "partial"),
            5,
            None,
            id="bench-times-out-in-a-dir",
        ),
        pytest.param(
            _TIMED_OUT_OVER_CAP,
            _IN_PLACE_NEW,
            _BENCH_ONLY,
            (
                "new",
                "main",
                "1",
                "/work",
                "run-bench",
                "1500",
                f"{OUTPUT_CAP + 1}",
                "partial",
            ),
            6,
            None,
            id="bench-times-out-over-the-cap-keeps-the-timeout",
        ),
        pytest.param(
            _EXIT_THREE,
            _REF_OLD,
            _WITH_PREPARE,
            ("old", "base", "feature", "/wt", "setup", "3", "boom"),
            6,
            (),
            id="prepare-fails-on-a-ref",
        ),
        pytest.param(
            _failed("std", "err", 3, 3),
            _IN_PLACE_X,
            _RUN_ONLY,
            (*_FAILED_HEAD, "err", "std"),
            8,
            None,
            id="both-labelled-stderr-first",
        ),
        pytest.param(
            _failed("s", "e", 1, 50, stderr_truncated=True),
            _IN_PLACE_X,
            _RUN_ONLY,
            (*_FAILED_HEAD, "50", "e", "s"),
            8,
            None,
            id="both-one-truncated",
        ),
        pytest.param(
            _failed("", "", 0, 0),
            _IN_PLACE_X,
            _RUN_ONLY,
            _FAILED_HEAD,
            4,
            None,
            id="neither-present",
        ),
        pytest.param(
            _failed("", "err", 0, 50),
            _IN_PLACE_X,
            _RUN_ONLY,
            (*_FAILED_HEAD, "err"),
            5,
            None,
            id="untruncated-count-above-kept-text-bare",
        ),
        pytest.param(
            capped_result(exit_code=1),
            _IN_PLACE_X,
            _RUN_ONLY,
            (
                *_FAILED_HEAD,
                f"{CAPPED_STDERR_BYTES}",
                "capped stderr",
                f"{CAPPED_STDOUT_BYTES}",
                "capped stdout",
            ),
            8,
            None,
            id="failed-bench-over-the-cap-keeps-the-failure",
        ),
        pytest.param(
            capped_result(),
            _IN_PLACE_NEW,
            _BENCH_ONLY,
            ("new", "main", "1", "/work", "run-bench", f"{CAPPED_STDOUT_BYTES}"),
            4,
            (_CAP_MIB,),
            id="bench-stdout-overran-the-cap-in-a-dir",
        ),
    ],
)
async def test_collect_samples_when_command_fails_does_raise_error_naming_what_failed_where(
    *,
    result: ExecResult | ExecTimeoutError,
    target: TargetContext,
    options: SamplingOptions,
    inserted: tuple[str, ...],
    line_count: int,
    hint_inserts: tuple[str, ...] | None,
    monkeypatch: pytest.MonkeyPatch,
):
    install_exec(monkeypatch, SAMPLING_EXEC, result)

    with pytest.raises(CommandError) as caught:
        await collect_samples(metric_lines_adapter, [target], options, asyncio.Event())

    message = str(caught.value)
    hint = caught.value.hint
    assert _holds_in_order(message, inserted)
    assert len(message.splitlines()) == line_count
    if hint_inserts is None:
        assert hint is None
    else:
        assert hint is not None
        assert [part for part in hint_inserts if part not in hint] == []


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(
            ExecResult("METRIC x=1", "capped", 0, 10, OUTPUT_CAP + 1, stderr_truncated=True),
            id="only-stderr-overran-the-cap",
        ),
    ],
)
async def test_collect_samples_when_bench_stdout_kept_whole_does_parse_the_sample(
    monkeypatch: pytest.MonkeyPatch, result: ExecResult
):
    install_exec(monkeypatch, SAMPLING_EXEC, result)

    samples = await collect_samples(
        metric_lines_adapter, one_in_place_target(), _RUN_ONLY, asyncio.Event()
    )

    assert [ts.samples for ts in samples] == [[{"x": 1.0}]]


def _prepare_overruns_the_cap(call: int) -> ExecResult:
    return capped_result() if call == 1 else numbered_sample(call)


async def test_collect_samples_when_prepare_stdout_overran_the_cap_does_still_sample(
    monkeypatch: pytest.MonkeyPatch,
):
    install_exec(monkeypatch, SAMPLING_EXEC, _prepare_overruns_the_cap)
    options = SamplingOptions(bench="run", prepare="prep", samples=1, timeout_seconds=1.0)

    samples = await collect_samples(
        metric_lines_adapter, one_in_place_target(), options, asyncio.Event()
    )

    assert [ts.samples for ts in samples] == [[{"x": 2.0}]]


# ---------------------------------------------------------------------------
# RunOptions.from_config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bench", "samples", "read", "expected"),
    [
        pytest.param("run --filter a", None, "bench", "run --filter a", id="bench"),
        pytest.param(None, 3, "samples", 3, id="samples"),
    ],
)
def test_run_options_from_config_when_override_given_does_replace_the_configured_value(
    bench: str | None, samples: int | None, read: str, expected: str | int
):
    config = resolved_config(bench="run", samples=7)

    run = RunOptions.from_config(config, samples=samples, bench=bench)

    assert getattr(run.sampling, read) == expected


# ---------------------------------------------------------------------------
# primary_direction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("primary", "expected"),
    [
        pytest.param("geomean", GEOMEAN_DIRECTION, id="geomean-ignores-a-geomean-metric-entry"),
        pytest.param("throughput", "higher", id="named-metric-resolves-its-entry"),
    ],
)
def test_primary_direction_when_primary_named_does_return_the_direction_it_resolves_to(
    primary: str, expected: str
):
    config = resolved_config(
        primary=primary,
        metrics={
            "throughput": MetricEntry(direction="higher"),
            "geomean": MetricEntry(direction="higher"),
        },
    )

    direction = primary_direction(config)

    assert direction == expected


def test_primary_direction_when_no_metric_entries_does_take_the_adapters_default():
    config = resolved_config(primary="throughput", metrics=None)

    direction = primary_direction(config)

    assert direction == "lower"


# ---------------------------------------------------------------------------
# sample summaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected_median", "expected_spread"),
    [
        pytest.param([5.0], 5.0, None, id="single-value"),
        pytest.param([10.0, 20.0, 30.0], 20.0, pytest.approx(50.0), id="odd-length-sorted"),
    ],
)
def test_compute_metric_stats_when_given_values_does_return_the_summary_stats(
    values: list[float],
    expected_median: float,
    expected_spread: object,
):
    stats = compute_metric_stats("x", values)

    assert (stats.median, stats.spread) == (expected_median, expected_spread)


def test_compute_metric_stats_when_no_values_does_raise_naming_the_metric():
    with pytest.raises(GymratError, match=r"\bdecode/time\b"):
        compute_metric_stats("decode/time", [])
