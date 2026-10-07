"""Behavioral tests for the sampling core and progress reporting."""

import asyncio
import sys
import time
from pathlib import Path

import pytest

from gymrat import sampling
from gymrat.adapters import AdapterError, metric_lines_adapter
from gymrat.config import KindEntry, MetricEntry, ResolvedConfig
from gymrat.errors import CommandError
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError
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
)
from gymrat.targets import InPlaceTarget, RefTarget
from tests._exec_fixtures import expected_result

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


def patch_exec(
    monkeypatch: pytest.MonkeyPatch,
    result: ExecResult | ExecTimeoutError,
) -> list[tuple[str, str, int | None]]:
    """Patch the sampling exec seam to return ``result`` and record each call.

    Args:
        monkeypatch: Pytest fixture used to patch the exec seam.
        result: The result or timeout error to return from every patched call.

    Returns:
        The list of ``(command, cwd, timeout_ms)`` tuples in call order.
    """
    calls: list[tuple[str, str, int | None]] = []

    async def _exec(command: str, options: ExecOptions) -> ExecResult | ExecTimeoutError:
        calls.append((command, options.cwd, options.timeout_ms))
        return result

    monkeypatch.setattr(sampling, "exec", _exec)
    return calls


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


async def test_collect_samples_when_prepare_set_does_run_prepare_per_target_before_any_bench(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = patch_exec(monkeypatch, make_success())
    targets = two_in_place_targets()
    options = SamplingOptions(bench="run", prepare="prep", samples=2, timeout_seconds=1.0)

    await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    commands = [(command, cwd) for command, cwd, _ in calls]
    assert commands == [
        ("prep", "/a"),
        ("prep", "/b"),
        ("run", "/a"),
        ("run", "/b"),
        ("run", "/a"),
        ("run", "/b"),
    ]


async def test_collect_samples_when_prepare_absent_does_skip_prepare_and_run_bench_only(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = patch_exec(monkeypatch, make_success())
    targets = two_in_place_targets()
    options = SamplingOptions(bench="run", prepare=None, samples=2, timeout_seconds=1.0)

    await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    commands = [command for command, _, _ in calls]
    assert commands == ["run", "run", "run", "run"]


async def test_collect_samples_when_finished_does_return_samples_per_target_in_order(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_success("METRIC x=1"))
    targets = two_in_place_targets()
    options = SamplingOptions(bench="run", prepare=None, samples=2, timeout_seconds=1.0)

    result = await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert [ts.ctx for ts in result] == targets
    assert [ts.samples for ts in result] == [
        [{"x": 1.0}, {"x": 1.0}],
        [{"x": 1.0}, {"x": 1.0}],
    ]


async def test_collect_samples_when_timeout_seconds_given_does_pass_millisecond_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = patch_exec(monkeypatch, make_success())
    targets = one_in_place_target()
    options = SamplingOptions(bench="run", prepare=None, samples=1, timeout_seconds=2.5)

    await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert calls[0][2] == 2500


async def test_collect_samples_when_progress_given_does_fire_prepare_and_pass_events(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_success())
    events: list[ProgressEvent] = []
    targets = two_in_place_targets()
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

    await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    types_and_labels = [
        (type(e).__name__, e.label)
        for e in events
        if isinstance(e, (PrepareStarted, PrepareFinished, PassStarted, PassFinished))
    ]
    assert types_and_labels == [
        ("PrepareStarted", "old"),
        ("PrepareFinished", "old"),
        ("PrepareStarted", "new"),
        ("PrepareFinished", "new"),
        ("PassStarted", "old"),
        ("PassFinished", "old"),
        ("PassStarted", "new"),
        ("PassFinished", "new"),
        ("PassStarted", "old"),
        ("PassFinished", "old"),
        ("PassStarted", "new"),
        ("PassFinished", "new"),
    ]


async def test_collect_samples_when_progress_given_does_stamp_at_ms_from_clock(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_success())
    events: list[ProgressEvent] = []
    call_count = 0

    def deterministic_clock() -> float:
        nonlocal call_count
        call_count += 1
        return call_count * 10.0

    targets = one_in_place_target()
    options = SamplingOptions(
        bench="run",
        prepare="prep",
        samples=1,
        timeout_seconds=1.0,
        on_progress=events.append,
        clock=deterministic_clock,
    )

    await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    timestamps = [e.at_ms for e in events]
    assert timestamps == sorted(timestamps)
    assert all(t % 10.0 == 0.0 for t in timestamps), (
        "timestamps should come from the injected clock"
    )
    assert events[0].at_ms == 10.0


async def test_collect_samples_when_clock_omitted_does_stamp_monotonic_milliseconds(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_success())
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


async def test_collect_samples_when_progress_given_does_emit_pass_started_with_correct_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_success())
    events: list[ProgressEvent] = []
    targets = two_in_place_targets()
    options = SamplingOptions(
        bench="run",
        prepare=None,
        samples=2,
        timeout_seconds=1.0,
        on_progress=events.append,
        clock=lambda: 0.0,
    )

    await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    pass_started_events = [e for e in events if isinstance(e, PassStarted)]
    first = pass_started_events[0]
    assert first.round == 1
    assert first.total_rounds == 2
    assert first.target_count == 2
    assert first.label == "old"
    assert first.phase == "measure"

    second = pass_started_events[1]
    assert second.label == "new"


async def test_collect_samples_when_bench_fails_does_emit_started_but_not_finished(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_failure())
    events: list[ProgressEvent] = []
    targets = one_in_place_target()
    options = SamplingOptions(
        bench="run",
        prepare=None,
        samples=1,
        timeout_seconds=1.0,
        on_progress=events.append,
        clock=lambda: 0.0,
    )

    with pytest.raises(CommandError):
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert any(isinstance(e, PassStarted) for e in events)
    assert not any(isinstance(e, PassFinished) for e in events)


async def test_collect_samples_when_prepare_fails_does_emit_prepare_started_but_not_finished(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_failure())
    events: list[ProgressEvent] = []
    targets = one_in_place_target()
    options = SamplingOptions(
        bench="run",
        prepare="prep",
        samples=1,
        timeout_seconds=1.0,
        on_progress=events.append,
        clock=lambda: 0.0,
    )

    with pytest.raises(CommandError):
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert any(isinstance(e, PrepareStarted) for e in events)
    assert not any(isinstance(e, PrepareFinished) for e in events)


async def test_collect_samples_when_warn_sink_given_does_pass_it_through_to_parse(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_success("METRIC foo=bar\nMETRIC x=1"))
    warnings: list[str] = []
    targets = one_in_place_target()
    options = SamplingOptions(
        bench="run",
        prepare=None,
        samples=1,
        timeout_seconds=1.0,
        warn=warnings.append,
    )

    await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert warnings == ["Failed to parse METRIC line: METRIC foo=bar"]


async def test_collect_samples_when_bench_output_unreadable_does_warn_after_the_pass_finished(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_success("METRIC foo=bar\nMETRIC x=1"))
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


async def test_collect_samples_when_prepare_fails_does_stop_before_any_bench(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = patch_exec(monkeypatch, make_failure())
    targets = two_in_place_targets()
    options = SamplingOptions(bench="run", prepare="prep", samples=2, timeout_seconds=1.0)

    with pytest.raises(CommandError):
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert [command for command, _, _ in calls] == ["prep"]


@pytest.mark.parametrize(
    ("result", "error"),
    [
        pytest.param(make_failure(), CommandError, id="non-zero-exit"),
        pytest.param(
            make_success("METRIC a//b=1\nMETRIC /x=2"), AdapterError, id="only-malformed-names"
        ),
    ],
)
async def test_collect_samples_when_bench_fails_does_stop_mid_schedule(
    monkeypatch: pytest.MonkeyPatch, result: ExecResult, error: type[Exception]
):
    calls = patch_exec(monkeypatch, result)
    targets = two_in_place_targets()
    options = SamplingOptions(bench="run", prepare=None, samples=2, timeout_seconds=1.0)

    with pytest.raises(error):
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert [command for command, _, _ in calls] == ["run"]


async def test_collect_samples_when_bench_fails_with_empty_stderr_does_raise_command_error(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_failure(stderr=""))
    targets = one_in_place_target()
    options = SamplingOptions(bench="run", prepare=None, samples=1, timeout_seconds=1.0)

    with pytest.raises(CommandError):
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())


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
    ],
)
async def test_collect_samples_when_command_fails_does_raise_error_with_full_shape(  # noqa: PLR0917 -- one parameter per failure axis plus the fixture
    result: ExecResult | ExecTimeoutError,
    target: TargetContext,
    options: SamplingOptions,
    expected: str,
    hint: str | None,
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, result)

    with pytest.raises(CommandError) as caught:
        await collect_samples(metric_lines_adapter, [target], options, asyncio.Event())

    assert (str(caught.value), caught.value.hint) == (expected, hint)


async def test_collect_samples_when_no_position_does_omit_position_from_header(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(
        monkeypatch,
        ExecResult(stdout="", stderr="boom", exit_code=1, stdout_bytes=4, stderr_bytes=4),
    )
    targets = [
        TargetContext(target=InPlaceTarget(dir="/work"), dir="/work", label="solo"),
    ]
    options = SamplingOptions(bench="run", prepare=None, samples=1, timeout_seconds=1.0)

    with pytest.raises(CommandError) as caught:
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert str(caught.value).splitlines()[0] == 'bench command failed ("solo", sample 1)'


@pytest.mark.parametrize(
    ("streams", "expected_tail"),
    [
        pytest.param(("", "err", 0, 3), ["err"], id="stderr-only-bare"),
        pytest.param(("std", "", 3, 0), ["std"], id="stdout-only-bare"),
        pytest.param(
            ("", "err", 0, 50),
            ["--- stderr (truncated, 50 bytes total) ---", "err"],
            id="stderr-only-truncated",
        ),
        pytest.param(
            ("head", "", 100, 0),
            ["--- stdout (truncated, 100 bytes total) ---", "head"],
            id="stdout-only-truncated",
        ),
        pytest.param(
            ("std", "err", 3, 3),
            ["--- stderr ---", "err", "--- stdout ---", "std"],
            id="both-labelled-stderr-first",
        ),
        pytest.param(
            ("s", "e", 1, 50),
            ["--- stderr (truncated, 50 bytes total) ---", "e", "--- stdout ---", "s"],
            id="both-one-truncated",
        ),
        pytest.param(("", "", 0, 0), [], id="neither-present"),
    ],
)
async def test_collect_samples_when_bench_fails_does_render_captured_output(
    monkeypatch: pytest.MonkeyPatch,
    streams: tuple[str, str, int, int],
    expected_tail: list[str],
):
    stdout, stderr, stdout_bytes, stderr_bytes = streams
    patch_exec(
        monkeypatch,
        ExecResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=1,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
        ),
    )
    targets = [
        TargetContext(target=InPlaceTarget(dir="/work"), dir="/work", label="x"),
    ]
    options = SamplingOptions(bench="run", prepare=None, samples=1, timeout_seconds=1.0)

    with pytest.raises(CommandError) as caught:
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    head = [
        'bench command failed ("x", sample 1)',
        "  dir:       /work",
        "  command:   run",
        "  exit code: 1",
    ]
    assert str(caught.value) == "\n".join(head + expected_tail)


async def test_collect_samples_when_bench_times_out_does_render_both_captured_streams(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(
        monkeypatch,
        ExecTimeoutError(stdout="s", stderr="e", timeout_ms=1000, stdout_bytes=1, stderr_bytes=50),
    )
    targets = [
        TargetContext(target=InPlaceTarget(dir="/work"), dir="/work", label="x"),
    ]
    options = SamplingOptions(bench="run", prepare=None, samples=1, timeout_seconds=1.0)

    with pytest.raises(CommandError) as caught:
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    expected = [
        'bench command timed out ("x", sample 1)',
        "  dir:       /work",
        "  command:   run",
        "  timeout:   1000ms",
        "--- stderr (truncated, 50 bytes total) ---",
        "e",
        "--- stdout ---",
        "s",
    ]
    assert str(caught.value) == "\n".join(expected)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell")
async def test_collect_samples_when_driven_end_to_end_does_collect_parsed_metrics(tmp_path: Path):
    targets = [
        TargetContext(
            target=InPlaceTarget(dir=str(tmp_path)),
            dir=str(tmp_path),
            label="old",
            position="old",
        ),
    ]
    options = SamplingOptions(
        bench="printf 'METRIC x=1\\n'",
        prepare=None,
        samples=2,
        timeout_seconds=30.0,
    )

    result = await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert [ts.samples for ts in result] == [[{"x": 1.0}, {"x": 1.0}]]


# ---------------------------------------------------------------------------
# RunOptions.from_config
# ---------------------------------------------------------------------------


def _resolved_config() -> ResolvedConfig:
    """A resolved configuration with every run setting away from its default."""
    return ResolvedConfig(
        bench="run",
        prepare="prep",
        adapter="mitata",
        samples=7,
        timeout_seconds=25,
        unstable_noise_pct=5.0,
        primary="geomean",
        metrics={"decode/time": MetricEntry(direction="higher")},
        kinds={"memory": KindEntry(gating=False)},
    )


def test_run_options_from_config_when_called_does_copy_the_run_settings_and_default_clock():
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
            warn=run.sampling.warn,
        ),
        adapter="mitata",
        config_metrics=config.metrics,
        config_kinds=config.kinds,
    )
    run.sampling.warn("unreadable line")
    assert warnings == ["unreadable line"]


def test_run_options_from_config_when_bench_and_samples_given_does_override_the_configured_ones():
    run = RunOptions.from_config(_resolved_config(), samples=3, bench="run --filter a")

    assert (run.sampling.bench, run.sampling.samples) == ("run --filter a", 3)
