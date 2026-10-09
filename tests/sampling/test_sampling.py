"""Behavioral tests for the sampling core and progress reporting.

They also cover the per-metric median and spread computed from collected
samples, and metric-meta resolution: adapter defaults, then kind, then metric.
"""

import asyncio
import time
from collections.abc import Callable

import pytest

from gymrat.adapters import AdapterError, MetricDefaults, metric_lines_adapter
from gymrat.config import KindEntry, MetricEntry, ResolvedConfig
from gymrat.errors import CommandError, GymratError
from gymrat.exec import ExecResult, ExecTimeoutError
from gymrat.model import ResolvedMetricMeta
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
    own_values,
    resolve_metric_meta,
    resolve_metric_meta_from_samples,
)
from gymrat.targets import InPlaceTarget, RefTarget
from tests._config import resolved_config
from tests._exec_fixtures import expected_result, install_exec
from tests.report._verdicts import metric_meta
from tests.sampling._adapters import make_adapter

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
                ("run", "/a"),
                ("run", "/b"),
            ],
            id="prepare-per-target-before-any-bench",
        ),
        pytest.param(
            None,
            [("run", "/a"), ("run", "/b"), ("run", "/a"), ("run", "/b")],
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

    assert [(command, options.cwd) for command, options in recorder.calls] == expected_commands
    assert {options.timeout_ms for _, options in recorder.calls} == {2500}
    assert [ts.ctx for ts in result] == targets
    assert [ts.samples for ts in result] == [
        [{"x": 1.0}, {"x": 1.0}],
        [{"x": 1.0}, {"x": 1.0}],
    ]


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
        samples=3,
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
        PassStarted(1, 3, 2, "old", 500.0, phase="measure"),
        PassFinished(1, 3, 2, "old", 600.0, phase="measure"),
        PassStarted(1, 3, 2, "new", 700.0, phase="measure"),
        PassFinished(1, 3, 2, "new", 800.0, phase="measure"),
        PassStarted(2, 3, 2, "old", 900.0, phase="measure"),
        PassFinished(2, 3, 2, "old", 1000.0, phase="measure"),
        PassStarted(2, 3, 2, "new", 1100.0, phase="measure"),
        PassFinished(2, 3, 2, "new", 1200.0, phase="measure"),
        PassStarted(3, 3, 2, "old", 1300.0, phase="measure"),
        PassFinished(3, 3, 2, "old", 1400.0, phase="measure"),
        PassStarted(3, 3, 2, "new", 1500.0, phase="measure"),
        PassFinished(3, 3, 2, "new", 1600.0, phase="measure"),
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

    assert [e.at_ms for e in events] == [2500.0, 2500.0]


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
        "Failed to parse METRIC line: METRIC foo=bar",
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


def _resolved_config() -> ResolvedConfig:
    """A resolved configuration with every run setting away from its default."""
    return resolved_config(
        bench="run",
        prepare="prep",
        adapter="mitata",
        samples=7,
        timeout_seconds=25,
        unstable_noise_pct=5.0,
        metrics={"decode/time": MetricEntry(direction="higher")},
        kinds={"memory": KindEntry(gating=False)},
    )


def test_run_options_from_config_when_no_overrides_given_does_copy_the_run_settings_leaving_the_default_clock():
    events: list[ProgressEvent] = []
    warnings: list[str] = []
    config = _resolved_config()

    run = RunOptions.from_config(config, on_progress=events.append, warn=warnings.append)

    assert run == RunOptions(
        sampling=SamplingOptions(
            bench="run",
            prepare="prep",
            samples=7,
            timeout_seconds=25,
            on_progress=events.append,
            warn=warnings.append,
        ),
        adapter="mitata",
        config_metrics=config.metrics,
        config_kinds=config.kinds,
    )


def test_run_options_from_config_when_bench_and_samples_given_does_override_the_configured_ones():
    run = RunOptions.from_config(_resolved_config(), samples=3, bench="run --filter a")

    assert (run.sampling.bench, run.sampling.samples) == ("run --filter a", 3)


# ---------------------------------------------------------------------------
# sample summaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected_median", "expected_spread"),
    [
        pytest.param([], None, None, id="empty"),
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
    expected_median: float | None,
    expected_spread: object,
):
    stats = compute_metric_stats(values)

    assert (stats.median, stats.spread) == (expected_median, expected_spread)


def test_own_values_when_rounds_missing_metric_does_skip_them():
    samples = [{"x": 1.0}, {"y": 2.0}, {"x": 3.0}]

    assert own_values(samples, "x") == [1.0, 3.0]


# ---------------------------------------------------------------------------
# resolve_metric_meta
# ---------------------------------------------------------------------------

_NAME = "bench-a/heap"
_HEAP = MetricDefaults(direction="higher", kind="memory", short_name="heap", unit="bytes")
_MEMORY_NOT_GATING = {"memory": KindEntry(gating=False)}


@pytest.mark.parametrize(
    ("defaults", "entry", "config_kinds", "expected"),
    [
        pytest.param(
            MetricDefaults(direction="higher", unit="ns"),
            None,
            None,
            metric_meta(_NAME, direction="higher", unit="ns"),
            id="adapter-direction-and-unit",
        ),
        pytest.param(
            MetricDefaults(direction="lower", kind="memory", short_name="heap"),
            None,
            None,
            metric_meta("heap", kind="memory"),
            id="adapter-kind-and-short-name",
        ),
        pytest.param(
            _HEAP,
            MetricEntry(direction="lower"),
            None,
            metric_meta("heap", direction="lower", kind="memory", unit="bytes"),
            id="entry-direction",
        ),
        pytest.param(
            _HEAP,
            MetricEntry(gating=False),
            None,
            metric_meta("heap", direction="higher", kind="memory", unit="bytes", gating=False),
            id="entry-gating",
        ),
        pytest.param(
            _HEAP,
            MetricEntry(exact=True),
            None,
            metric_meta("heap", direction="higher", kind="memory", unit="bytes", exact=True),
            id="entry-exact",
        ),
        pytest.param(
            _HEAP,
            MetricEntry(exact=True),
            _MEMORY_NOT_GATING,
            metric_meta(
                "heap", direction="higher", kind="memory", unit="bytes", gating=False, exact=True
            ),
            id="kind-gating-fills-an-entry-without-gating",
        ),
        pytest.param(
            _HEAP,
            None,
            _MEMORY_NOT_GATING,
            metric_meta("heap", direction="higher", kind="memory", unit="bytes", gating=False),
            id="kind-gating-without-an-entry",
        ),
        pytest.param(
            _HEAP,
            None,
            {"time": KindEntry(gating=False)},
            metric_meta("heap", direction="higher", kind="memory", unit="bytes"),
            id="other-kind-gating-ignored",
        ),
        pytest.param(
            _HEAP,
            MetricEntry(gating=True),
            _MEMORY_NOT_GATING,
            metric_meta("heap", direction="higher", kind="memory", unit="bytes"),
            id="entry-gating-beats-kind-gating",
        ),
        pytest.param(
            _HEAP,
            None,
            {"memory": KindEntry(gating=None)},
            metric_meta("heap", direction="higher", kind="memory", unit="bytes"),
            id="kind-entry-without-gating-keeps-the-default",
        ),
        pytest.param(
            MetricDefaults(direction="lower", short_name="heap"),
            None,
            {"other": KindEntry(gating=False)},
            metric_meta("heap", kind="other", gating=False),
            id="adapter-reports-no-kind",
        ),
    ],
)
def test_resolve_metric_meta_when_layers_given_does_layer_entry_over_kind_over_adapter(
    defaults: MetricDefaults,
    entry: MetricEntry | None,
    config_kinds: dict[str, KindEntry] | None,
    expected: ResolvedMetricMeta,
):
    adapter = make_adapter(lambda _name: defaults)

    result = resolve_metric_meta(_NAME, entry, adapter, config_kinds)

    assert result == expected


# ---------------------------------------------------------------------------
# resolve_metric_meta_from_samples
# ---------------------------------------------------------------------------


def _by_metric_name(name: str) -> MetricDefaults:
    if name == "response-time":
        return MetricDefaults(direction="lower", unit="ns")
    if name == "throughput":
        return MetricDefaults(direction="higher")
    return MetricDefaults(direction="lower")


def _by_metric_suffix(name: str) -> MetricDefaults:
    if name.endswith("/heap"):
        return MetricDefaults(direction="lower", kind="memory", short_name="heap")
    return MetricDefaults(direction="lower", kind="time", short_name="time")


def test_resolve_metric_meta_from_samples_when_no_set_reports_a_metric_does_raise():
    sample_sets = [[{}, {}], [{}, {}]]

    with pytest.raises(GymratError, match="^No metrics found in benchmark output$"):
        resolve_metric_meta_from_samples(sample_sets, None, make_adapter(), None)


@pytest.mark.parametrize(
    ("sample_sets", "defaults_fn", "config_metrics", "config_kinds", "expected"),
    [
        pytest.param(
            [[{"throughput": 1.0}], [{"response-time": 1.0}]],
            _by_metric_name,
            {
                "response-time": MetricEntry(gating=False),
                "throughput": MetricEntry(exact=True),
            },
            None,
            {
                "throughput": metric_meta("throughput", direction="higher", exact=True),
                "response-time": metric_meta("response-time", unit="ns", gating=False),
            },
            id="per-metric-entries-first-seen-in-a-later-set",
        ),
        pytest.param(
            [[{"bench-a/time": 1.0}, {"bench-a/time": 1.0, "bench-a/heap": 1.0}]],
            _by_metric_suffix,
            None,
            _MEMORY_NOT_GATING,
            {
                "bench-a/time": metric_meta("time", kind="time"),
                "bench-a/heap": metric_meta("heap", gating=False, kind="memory"),
            },
            id="kind-gating-first-seen-in-a-later-round",
        ),
    ],
)
def test_resolve_metric_meta_from_samples_when_several_names_reported_does_resolve_each_in_first_seen_order(
    sample_sets: list[list[dict[str, float]]],
    defaults_fn: Callable[[str], MetricDefaults],
    config_metrics: dict[str, MetricEntry] | None,
    config_kinds: dict[str, KindEntry] | None,
    expected: dict[str, ResolvedMetricMeta],
):
    result = resolve_metric_meta_from_samples(
        sample_sets, config_metrics, make_adapter(defaults_fn), config_kinds
    )

    assert list(result) == list(expected)
    assert result == expected
