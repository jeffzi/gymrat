"""Behavioral tests for the sampling core, progress reporting, and metric stats."""

import asyncio
import dataclasses
import sys
from pathlib import Path

import pytest

from gymrat_py import sampling
from gymrat_py.adapters.metric_lines import metric_lines_adapter
from gymrat_py.errors import CommandError
from gymrat_py.exec import ExecOptions, ExecResult, ExecTimeoutError
from gymrat_py.sampling import (
    MetricStats,
    PrepareProgressStep,
    SampleProgressStep,
    SamplingOptions,
    TargetContext,
    TargetSamples,
    collect_samples,
    compute_metric_stats,
    own_values,
    paired_or_own_values,
)
from gymrat_py.targets import InPlaceTarget, RefTarget

REF_HINT = (
    "the worktree only contains files tracked at this ref; "
    "untracked, gitignored, or not-yet-committed files are absent"
)


def make_success(stdout: str = "METRIC x=1") -> ExecResult:
    """Build a zero-exit result carrying ``stdout`` on the standard stream."""
    return ExecResult(
        stdout=stdout,
        stderr="",
        exit_code=0,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=0,
    )


def make_failure(stdout: str = "", stderr: str = "boom") -> ExecResult:
    """Build an exit-code-1 result with byte counts computed from the given text."""
    return ExecResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=1,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
    )


def patch_exec(
    monkeypatch: pytest.MonkeyPatch,
    result: ExecResult | ExecTimeoutError,
) -> list[tuple[str, str, int | None]]:
    """Patch the sampling exec seam to return ``result`` and record each call.

    Returns the list of ``(command, cwd, timeout_ms)`` tuples in call order.
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


async def test_collect_samples_when_progress_given_does_fire_prepare_then_sample_steps(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_success())
    steps: list[PrepareProgressStep | SampleProgressStep] = []
    targets = two_in_place_targets()
    options = SamplingOptions(
        bench="run",
        prepare="prep",
        samples=2,
        timeout_seconds=1.0,
        on_progress=steps.append,
    )

    await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert steps == [
        PrepareProgressStep(label="old"),
        PrepareProgressStep(label="new"),
        SampleProgressStep(index=1, total=2, label="old"),
        SampleProgressStep(index=1, total=2, label="new"),
        SampleProgressStep(index=2, total=2, label="old"),
        SampleProgressStep(index=2, total=2, label="new"),
    ]


async def test_collect_samples_when_no_progress_callback_does_not_fail(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_success())
    targets = two_in_place_targets()
    options = SamplingOptions(bench="run", prepare=None, samples=1, timeout_seconds=1.0)

    result = await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert [ts.samples for ts in result] == [[{"x": 1.0}], [{"x": 1.0}]]


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


async def test_collect_samples_when_prepare_fails_does_stop_before_any_bench(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = patch_exec(monkeypatch, make_failure())
    targets = two_in_place_targets()
    options = SamplingOptions(bench="run", prepare="prep", samples=2, timeout_seconds=1.0)

    with pytest.raises(CommandError):
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert [command for command, _, _ in calls] == ["prep"]


async def test_collect_samples_when_bench_fails_does_stop_mid_schedule(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = patch_exec(monkeypatch, make_failure())
    targets = two_in_place_targets()
    options = SamplingOptions(bench="run", prepare=None, samples=2, timeout_seconds=1.0)

    with pytest.raises(CommandError):
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert [command for command, _, _ in calls] == ["run"]


async def test_collect_samples_when_abort_settles_run_as_exit_one_does_raise_command_error(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(monkeypatch, make_failure(stderr=""))
    targets = one_in_place_target()
    options = SamplingOptions(bench="run", prepare=None, samples=1, timeout_seconds=1.0)

    with pytest.raises(CommandError):
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())


async def test_collect_samples_when_bench_exits_non_zero_does_raise_error_with_full_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(
        monkeypatch,
        ExecResult(stdout="", stderr="boom", exit_code=3, stdout_bytes=4, stderr_bytes=4),
    )
    targets = [
        TargetContext(
            target=InPlaceTarget(dir="/work"),
            dir="/work",
            label="main",
            position="new",
        ),
    ]
    options = SamplingOptions(bench="run-bench", prepare=None, samples=1, timeout_seconds=1.0)

    with pytest.raises(CommandError) as caught:
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert str(caught.value) == (
        'bench command failed (new, "main", sample 1)\n'
        "  dir:       /work\n"
        "  command:   run-bench\n"
        "  exit code: 3\n"
        "boom"
    )
    assert caught.value.hint is None


async def test_collect_samples_when_prepare_times_out_on_ref_does_raise_error_with_full_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_exec(
        monkeypatch,
        ExecTimeoutError(
            stdout="partial",
            stderr="",
            timeout_ms=1500,
            stdout_bytes=7,
            stderr_bytes=0,
        ),
    )
    targets = [
        TargetContext(
            target=RefTarget(ref="feature", resolved_sha="deadbeef"),
            dir="/wt",
            label="base",
            position="old",
        ),
    ]
    options = SamplingOptions(bench="run", prepare="setup", samples=1, timeout_seconds=1.5)

    with pytest.raises(CommandError) as caught:
        await collect_samples(metric_lines_adapter, targets, options, asyncio.Event())

    assert str(caught.value) == (
        'prepare command timed out (old, "base")\n'
        "  ref:       feature\n"
        "  worktree:  /wt\n"
        "  command:   setup\n"
        "  timeout:   1500ms\n"
        "partial"
    )
    assert caught.value.hint == REF_HINT


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


def test_compute_metric_stats_when_empty_does_return_absent_median_and_spread():
    stats = compute_metric_stats([])

    assert stats == MetricStats(median=None, spread=None)


def test_compute_metric_stats_when_single_value_does_return_median_without_spread():
    stats = compute_metric_stats([5.0])

    assert stats.median == 5.0
    assert stats.spread is None


def test_compute_metric_stats_when_multiple_values_does_return_median_and_percent_spread():
    stats = compute_metric_stats([10.0, 20.0, 30.0])

    assert stats.median == 20.0
    assert stats.spread == 50.0


def test_compute_metric_stats_when_median_zero_does_omit_spread():
    stats = compute_metric_stats([-1.0, 0.0, 1.0])

    assert stats.median == 0.0
    assert stats.spread is None


def test_compute_metric_stats_when_ratio_overflows_to_infinity_does_omit_spread():
    stats = compute_metric_stats([0.0, 5e-324, 1.0])

    assert stats.median == 5e-324
    assert stats.spread is None


def test_metric_stats_when_field_assigned_does_raise_frozen():
    stats = compute_metric_stats([1.0, 2.0])

    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.median = 9.0  # type: ignore[misc]


def test_own_values_when_rounds_missing_metric_does_skip_them():
    samples = [{"x": 1.0}, {"y": 2.0}, {"x": 3.0}]

    assert own_values(samples, "x") == [1.0, 3.0]


def test_paired_or_own_values_when_paired_non_empty_does_return_paired():
    samples = [{"x": 1.0}, {"x": 3.0}]

    assert paired_or_own_values([7.0, 8.0], samples, "x") == [7.0, 8.0]


def test_paired_or_own_values_when_paired_empty_does_fall_back_to_own_values():
    samples = [{"x": 1.0}, {"x": 3.0}]

    assert paired_or_own_values([], samples, "x") == [1.0, 3.0]


def test_target_samples_when_constructed_does_pair_context_with_samples():
    ctx = TargetContext(target=InPlaceTarget(dir="/a"), dir="/a", label="old")

    bundle = TargetSamples(ctx=ctx, samples=[{"x": 1.0}])

    assert bundle.ctx is ctx
    assert bundle.samples == [{"x": 1.0}]
