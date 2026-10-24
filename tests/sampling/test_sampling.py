"""Behavioral tests for the sampling core and progress reporting.

They also cover the per-metric median and spread computed from collected
samples.
"""

import asyncio
import itertools
import time
from collections.abc import Callable
from unittest.mock import create_autospec

import pytest

from gymrat.adapters import AdapterError, metric_lines_adapter
from gymrat.errors import CommandError, GymratError
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError
from gymrat.exec import exec as run_exec
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
)
from gymrat.targets import InPlaceTarget, RefTarget
from gymrat.utils import monotonic_ms
from tests._config import resolved_config
from tests._exec_fixtures import expected_result, install_exec
from tests.adapters._inputs import malformed_line_warning

REF_HINT = (
    "the worktree only contains files tracked at this ref; "
    "untracked, gitignored, or not-yet-committed files are absent"
)


def make_success(stdout: str = "METRIC x=1") -> ExecResult:
    """Build a zero-exit result carrying ``stdout`` on the standard stream."""
    return expected_result(stdout)


def make_failure(stdout: str = "", stderr: str = "boom") -> ExecResult:
    """Build an exit-code-1 result with byte counts computed from the given text."""
    return expected_result(stdout, stderr, exit_code=1)


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


def install_numbered_exec(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[int], ExecResult | ExecTimeoutError]
) -> list[tuple[str, ExecOptions]]:
    """Replace the sampling ``exec`` with a stand-in answering each call by its 1-based number.

    Args:
        monkeypatch: Patches the sampling ``exec`` for the duration of the test.
        answer: Builds the result for the n-th call.

    Returns:
        Each call's command and options, in call order.
    """
    calls: list[tuple[str, ExecOptions]] = []

    async def record(command: str, options: ExecOptions) -> ExecResult | ExecTimeoutError:
        calls.append((command, options))
        return answer(len(calls))

    monkeypatch.setattr(SAMPLING_EXEC, create_autospec(run_exec, side_effect=record))
    return calls


def numbered_sample(call: int) -> ExecResult:
    """Answer a bench run with ``METRIC x=<call>``, so each sample names the run it came from."""
    return make_success(f"METRIC x={call}")


# ---------------------------------------------------------------------------
# collect_samples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prepare", "expected_commands"),
    [
        pytest.param(
            "prep",
            [
                ("prep", "/a"),
                ("prep", "/b"),
                ("run", "/a"),
                ("run", "/b"),
                ("run", "/b"),
                ("run", "/a"),
            ],
            id="prepare-per-target-before-any-bench",
        ),
        pytest.param(
            None,
            [("run", "/a"), ("run", "/b"), ("run", "/b"), ("run", "/a")],
            id="bench-only",
        ),
    ],
)
async def test_collect_samples_when_targets_succeed_does_collect_each_targets_samples_in_schedule_order(
    monkeypatch: pytest.MonkeyPatch,
    prepare: str | None,
    expected_commands: list[tuple[str, str]],
):
    recorder = install_exec(monkeypatch, SAMPLING_EXEC, make_success("METRIC x=1"))
    targets = two_in_place_targets()
    options = SamplingOptions(bench="run", prepare=prepare, samples=2, timeout_seconds=2.5)

    result = await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert [(command, exec_options.cwd) for command, exec_options in recorder.calls] == (
        expected_commands
    )
    assert {exec_options.timeout_ms for _, exec_options in recorder.calls} == {2500}
    assert [ts.ctx for ts in result] == targets
    assert [ts.samples for ts in result] == [
        [{"x": 1.0}, {"x": 1.0}],
        [{"x": 1.0}, {"x": 1.0}],
    ]


@pytest.mark.parametrize(
    ("targets", "rounds", "expected_dirs", "expected_samples"),
    [
        pytest.param(
            one_in_place_target(),
            3,
            ["/a", "/a", "/a"],
            [[1.0, 2.0, 3.0]],
            id="one-target",
        ),
        pytest.param(
            one_in_place_target(),
            1,
            ["/a"],
            [[1.0]],
            id="one-target-single-round",
        ),
        pytest.param(
            two_in_place_targets(),
            4,
            ["/a", "/b", "/b", "/a", "/a", "/b", "/b", "/a"],
            [[1.0, 4.0, 5.0, 8.0], [2.0, 3.0, 6.0, 7.0]],
            id="two-targets",
        ),
    ],
)
async def test_collect_samples_when_one_or_two_targets_does_alternate_which_target_runs_first(
    monkeypatch: pytest.MonkeyPatch,
    targets: list[TargetContext],
    rounds: int,
    expected_dirs: list[str],
    expected_samples: list[list[float]],
):
    calls = install_numbered_exec(monkeypatch, numbered_sample)
    options = SamplingOptions(bench="run", prepare=None, samples=rounds, timeout_seconds=1.0)

    result = await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert [exec_options.cwd for _, exec_options in calls] == expected_dirs
    assert [ts.ctx for ts in result] == targets
    assert [[sample["x"] for sample in ts.samples] for ts in result] == expected_samples


#: Target counts of three or more with the shortest sample count that balances each.
_BALANCING_CYCLES = [
    pytest.param(3, 6, id="three-targets"),
    pytest.param(4, 4, id="four-targets"),
    pytest.param(5, 10, id="five-targets"),
    pytest.param(6, 6, id="six-targets"),
]


async def _collect_run_orders(
    monkeypatch: pytest.MonkeyPatch, target_count: int, samples: int
) -> tuple[list[list[str]], list[str]]:
    """Sample ``target_count`` lettered targets, returning each round's run order and the dirs."""
    calls = install_numbered_exec(monkeypatch, numbered_sample)
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

    assert len(rounds) == cycle
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


@pytest.mark.parametrize(("target_count", "cycle"), _BALANCING_CYCLES)
async def test_collect_samples_when_samples_span_two_cycles_does_repeat_the_cycle(
    monkeypatch: pytest.MonkeyPatch, target_count: int, cycle: int
):
    rounds, _ = await _collect_run_orders(monkeypatch, target_count, 2 * cycle)

    assert rounds[cycle:] == rounds[:cycle]


async def test_collect_samples_when_run_order_varies_does_keep_each_targets_samples_in_round_order(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = install_numbered_exec(monkeypatch, numbered_sample)
    targets = lettered_in_place_targets(3)
    options = SamplingOptions(bench="run", prepare=None, samples=12, timeout_seconds=1.0)

    result = await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert [ts.ctx for ts in result] == targets
    assert [[(int(sample["x"]) - 1) // 3 for sample in ts.samples] for ts in result] == [
        list(range(12))
    ] * 3
    assert [[int(sample["x"]) for sample in ts.samples] for ts in result] == [
        [call for call, (_, ran) in enumerate(calls, start=1) if ran.cwd == ctx.dir]
        for ctx in targets
    ]


@pytest.mark.parametrize(
    ("target_count", "samples"),
    [
        pytest.param(2, 3, id="two-targets-odd-count"),
        pytest.param(3, 9, id="three-targets-multiple-of-n-but-not-2n"),
        pytest.param(4, 6, id="four-targets-not-a-multiple"),
    ],
)
async def test_collect_samples_when_count_cannot_balance_the_targets_does_refuse_before_any_command_runs(
    monkeypatch: pytest.MonkeyPatch, target_count: int, samples: int
):
    recorder = install_exec(monkeypatch, SAMPLING_EXEC, make_success())
    options = SamplingOptions(bench="run", prepare="prep", samples=samples, timeout_seconds=1.0)

    with pytest.raises(GymratError) as caught:
        await collect_samples(
            metric_lines_adapter, lettered_in_place_targets(target_count), options, asyncio.Event()
        )

    assert caught.value.reason == "unbalanced-samples"
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
        PassStarted(1, 2, 2, "old", 500.0, phase="measure"),
        PassFinished(1, 2, 2, "old", 600.0, phase="measure"),
        PassStarted(1, 2, 2, "new", 700.0, phase="measure"),
        PassFinished(1, 2, 2, "new", 800.0, phase="measure"),
        PassStarted(2, 2, 2, "new", 900.0, phase="measure"),
        PassFinished(2, 2, 2, "new", 1000.0, phase="measure"),
        PassStarted(2, 2, 2, "old", 1100.0, phase="measure"),
        PassFinished(2, 2, 2, "old", 1200.0, phase="measure"),
    ]


async def test_collect_samples_when_three_targets_reorder_does_label_each_pass_with_the_target_it_ran(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = install_numbered_exec(monkeypatch, numbered_sample)
    targets = lettered_in_place_targets(3)
    label_of = {ctx.dir: ctx.label for ctx in targets}
    events: list[ProgressEvent] = []
    options = SamplingOptions(
        bench="run",
        prepare=None,
        samples=6,
        timeout_seconds=1.0,
        on_progress=events.append,
        clock=lambda: 0.0,
    )

    await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    passes = [event for event in events if isinstance(event, PassStarted | PassFinished)]
    assert len(passes) == len(events)
    assert [(type(event), event.round, event.label) for event in passes] == [
        (kind, index // 3 + 1, label_of[exec_options.cwd])
        for index, (_, exec_options) in enumerate(calls)
        for kind in (PassStarted, PassFinished)
    ]


async def test_collect_samples_when_clock_omitted_does_stamp_monotonic_milliseconds(
    monkeypatch: pytest.MonkeyPatch,
):
    install_exec(monkeypatch, SAMPLING_EXEC, make_success())
    monkeypatch.setattr(time, "perf_counter", lambda: 2.5)
    events: list[ProgressEvent] = []
    options = SamplingOptions(
        bench="run",
        prepare=None,
        samples=1,
        timeout_seconds=1.0,
        on_progress=events.append,
    )

    await collect_samples(metric_lines_adapter, one_in_place_target(), options, asyncio.Event())

    assert [e.at_ms for e in events] == [monotonic_ms(), monotonic_ms()]


async def test_collect_samples_when_bench_output_unreadable_does_warn_after_the_pass_finished(
    monkeypatch: pytest.MonkeyPatch,
):
    install_exec(monkeypatch, SAMPLING_EXEC, make_success("METRIC foo=bar\nMETRIC x=1"))
    log: list[object] = []
    options = SamplingOptions(
        bench="run",
        prepare=None,
        samples=1,
        timeout_seconds=1.0,
        on_progress=log.append,
        warn=log.append,
        clock=lambda: 0.0,
    )

    await collect_samples(metric_lines_adapter, one_in_place_target(), options, asyncio.Event())

    assert log == [
        PassStarted(round=1, total_rounds=1, target_count=1, label="old", at_ms=0.0),
        PassFinished(round=1, total_rounds=1, target_count=1, label="old", at_ms=0.0),
        malformed_line_warning("METRIC foo=bar"),
    ]


@pytest.mark.parametrize(
    ("prepare", "result", "error", "expected_commands", "expected_events"),
    [
        pytest.param(
            "prep",
            make_failure(),
            CommandError,
            ["prep"],
            [PrepareStarted(label="old", at_ms=0.0)],
            id="prepare-fails",
        ),
        pytest.param(
            None,
            make_failure(),
            CommandError,
            ["run"],
            [PassStarted(round=1, total_rounds=2, target_count=2, label="old", at_ms=0.0)],
            id="bench-exits-non-zero",
        ),
        pytest.param(
            None,
            make_success("METRIC a//b=1\nMETRIC /x=2"),
            AdapterError,
            ["run"],
            [
                PassStarted(round=1, total_rounds=2, target_count=2, label="old", at_ms=0.0),
                PassFinished(round=1, total_rounds=2, target_count=2, label="old", at_ms=0.0),
            ],
            id="bench-reports-only-malformed-names",
        ),
    ],
)
async def test_collect_samples_when_a_pass_cannot_yield_a_sample_does_stop_at_it_with_progress_up_to_the_failure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prepare: str | None,
    result: ExecResult,
    error: type[Exception],
    expected_commands: list[str],
    expected_events: list[ProgressEvent],
):
    recorder = install_exec(monkeypatch, SAMPLING_EXEC, result)
    events: list[ProgressEvent] = []
    options = SamplingOptions(
        bench="run",
        prepare=prepare,
        samples=2,
        timeout_seconds=1.0,
        on_progress=events.append,
        clock=lambda: 0.0,
    )

    with pytest.raises(error):
        await collect_samples(
            metric_lines_adapter, two_in_place_targets(), options, asyncio.Event()
        )

    assert [command for command, _ in recorder.calls] == expected_commands
    assert events == expected_events


def _malformed_at_call_three(call: int) -> ExecResult:
    return make_success("METRIC a//b=1") if call == 3 else numbered_sample(call)


def _fails_at_call_three(call: int) -> ExecResult:
    return make_failure() if call == 3 else numbered_sample(call)


def _times_out_at_call_three(call: int) -> ExecResult | ExecTimeoutError:
    return _TIMED_OUT if call == 3 else numbered_sample(call)


_ROUND_ONE_EVENTS: list[ProgressEvent] = [
    PassStarted(round=1, total_rounds=2, target_count=2, label="old", at_ms=0.0),
    PassFinished(round=1, total_rounds=2, target_count=2, label="old", at_ms=0.0),
    PassStarted(round=1, total_rounds=2, target_count=2, label="new", at_ms=0.0),
    PassFinished(round=1, total_rounds=2, target_count=2, label="new", at_ms=0.0),
]


@pytest.mark.parametrize(
    ("answer", "error", "expected_events"),
    [
        pytest.param(
            _fails_at_call_three,
            CommandError,
            [
                *_ROUND_ONE_EVENTS,
                PassStarted(round=2, total_rounds=2, target_count=2, label="new", at_ms=0.0),
            ],
            id="bench-exits-non-zero",
        ),
        pytest.param(
            _times_out_at_call_three,
            CommandError,
            [
                *_ROUND_ONE_EVENTS,
                PassStarted(round=2, total_rounds=2, target_count=2, label="new", at_ms=0.0),
            ],
            id="bench-times-out",
        ),
        pytest.param(
            _malformed_at_call_three,
            AdapterError,
            [
                *_ROUND_ONE_EVENTS,
                PassStarted(round=2, total_rounds=2, target_count=2, label="new", at_ms=0.0),
                PassFinished(round=2, total_rounds=2, target_count=2, label="new", at_ms=0.0),
            ],
            id="bench-reports-only-malformed-names",
        ),
    ],
)
async def test_collect_samples_when_a_later_round_pass_fails_does_stop_at_it_in_rotated_order(
    monkeypatch: pytest.MonkeyPatch,
    answer: Callable[[int], ExecResult | ExecTimeoutError],
    error: type[Exception],
    expected_events: list[ProgressEvent],
):
    calls = install_numbered_exec(monkeypatch, answer)
    events: list[ProgressEvent] = []
    options = SamplingOptions(
        bench="run",
        prepare=None,
        samples=2,
        timeout_seconds=1.0,
        on_progress=events.append,
        clock=lambda: 0.0,
    )

    with pytest.raises(error):
        await collect_samples(
            metric_lines_adapter, two_in_place_targets(), options, asyncio.Event()
        )

    assert [exec_options.cwd for _, exec_options in calls] == ["/a", "/b", "/b"]
    assert events == expected_events


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
_BENCH_ONLY = SamplingOptions(bench="run-bench", prepare=None, samples=1, timeout_seconds=1.5)
_WITH_PREPARE = SamplingOptions(bench="run", prepare="setup", samples=1, timeout_seconds=1.5)
_IN_PLACE_X = TargetContext(target=InPlaceTarget(dir="/work"), dir="/work", label="x")
_RUN_ONLY = SamplingOptions(bench="run", prepare=None, samples=1, timeout_seconds=1.0)


_FAILED_HEAD = [
    'bench command failed ("x", sample 1)',
    "  dir:       /work",
    "  command:   run",
    "  exit code: 1",
]


def _failed(stdout: str, stderr: str, stdout_bytes: int, stderr_bytes: int) -> ExecResult:
    return ExecResult(stdout, stderr, 1, stdout_bytes, stderr_bytes)


@pytest.mark.parametrize(
    ("result", "target", "options", "expected", "hint"),
    [
        pytest.param(
            _EXIT_THREE,
            _IN_PLACE_NEW,
            _BENCH_ONLY,
            'bench command failed (new, "main", sample 1)\n'
            "  dir:       /work\n"
            "  command:   run-bench\n"
            "  exit code: 3\n"
            "boom",
            None,
            id="bench-fails-in-a-dir",
        ),
        pytest.param(
            _TIMED_OUT,
            _IN_PLACE_NEW,
            _BENCH_ONLY,
            'bench command timed out (new, "main", sample 1)\n'
            "  dir:       /work\n"
            "  command:   run-bench\n"
            "  timeout:   1500ms\n"
            "partial",
            None,
            id="bench-times-out-in-a-dir",
        ),
        pytest.param(
            _EXIT_THREE,
            _REF_OLD,
            _WITH_PREPARE,
            'prepare command failed (old, "base")\n'
            "  ref:       feature\n"
            "  worktree:  /wt\n"
            "  command:   setup\n"
            "  exit code: 3\n"
            "boom",
            REF_HINT,
            id="prepare-fails-on-a-ref",
        ),
        pytest.param(
            _TIMED_OUT,
            _REF_OLD,
            _WITH_PREPARE,
            'prepare command timed out (old, "base")\n'
            "  ref:       feature\n"
            "  worktree:  /wt\n"
            "  command:   setup\n"
            "  timeout:   1500ms\n"
            "partial",
            REF_HINT,
            id="prepare-times-out-on-a-ref",
        ),
        pytest.param(
            _failed("std", "", 3, 0),
            _IN_PLACE_X,
            _RUN_ONLY,
            "\n".join([*_FAILED_HEAD, "std"]),
            None,
            id="stdout-only-bare",
        ),
        pytest.param(
            _failed("", "err", 0, 50),
            _IN_PLACE_X,
            _RUN_ONLY,
            "\n".join([*_FAILED_HEAD, "--- stderr (truncated, 50 bytes total) ---", "err"]),
            None,
            id="stderr-only-truncated",
        ),
        pytest.param(
            _failed("head", "", 100, 0),
            _IN_PLACE_X,
            _RUN_ONLY,
            "\n".join([*_FAILED_HEAD, "--- stdout (truncated, 100 bytes total) ---", "head"]),
            None,
            id="stdout-only-truncated",
        ),
        pytest.param(
            _failed("std", "err", 3, 3),
            _IN_PLACE_X,
            _RUN_ONLY,
            "\n".join([*_FAILED_HEAD, "--- stderr ---", "err", "--- stdout ---", "std"]),
            None,
            id="both-labelled-stderr-first",
        ),
        pytest.param(
            _failed("s", "e", 1, 50),
            _IN_PLACE_X,
            _RUN_ONLY,
            "\n".join([
                *_FAILED_HEAD,
                "--- stderr (truncated, 50 bytes total) ---",
                "e",
                "--- stdout ---",
                "s",
            ]),
            None,
            id="both-one-truncated",
        ),
        pytest.param(
            _failed("", "", 0, 0),
            _IN_PLACE_X,
            _RUN_ONLY,
            "\n".join(_FAILED_HEAD),
            None,
            id="neither-present",
        ),
    ],
)
async def test_collect_samples_when_command_fails_does_raise_error_with_full_shape(
    *,
    result: ExecResult | ExecTimeoutError,
    target: TargetContext,
    options: SamplingOptions,
    expected: str,
    hint: str | None,
    monkeypatch: pytest.MonkeyPatch,
):
    install_exec(monkeypatch, SAMPLING_EXEC, result)

    with pytest.raises(CommandError) as caught:
        await collect_samples(metric_lines_adapter, [target], options, asyncio.Event())

    assert (str(caught.value), caught.value.hint) == (expected, hint)


# ---------------------------------------------------------------------------
# RunOptions.from_config
# ---------------------------------------------------------------------------


def test_run_options_from_config_when_bench_and_samples_given_does_override_the_configured_ones():
    config = resolved_config(bench="run", samples=7)

    run = RunOptions.from_config(config, samples=3, bench="run --filter a")

    assert (run.sampling.bench, run.sampling.samples) == ("run --filter a", 3)


# ---------------------------------------------------------------------------
# sample summaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected_median", "expected_spread"),
    [
        pytest.param([5.0], 5.0, None, id="single-value"),
        pytest.param([10.0, 20.0, 30.0], 20.0, pytest.approx(50.0), id="odd-length-sorted"),
        pytest.param(
            [30.0, 10.0, 40.0, 20.0], 25.0, pytest.approx(60.0), id="even-length-unsorted"
        ),
        pytest.param([-1.0, 0.0, 1.0], 0.0, None, id="median-zero"),
        pytest.param([0.0, 5e-324, 1.0], 5e-324, None, id="ratio-overflows-to-infinity"),
    ],
)
def test_compute_metric_stats_when_given_values_does_return_median_and_percent_spread(
    values: list[float],
    expected_median: float,
    expected_spread: object,
):
    stats = compute_metric_stats("x", values)

    assert (stats.median, stats.spread) == (expected_median, expected_spread)


def test_compute_metric_stats_when_no_values_does_raise_naming_the_metric():
    with pytest.raises(GymratError, match=r"\bdecode/time\b"):
        compute_metric_stats("decode/time", [])
