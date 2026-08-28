"""Behavioral tests for the sampling core, progress reporting, and metric stats."""

import asyncio
import dataclasses
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat_py import sampling
from gymrat_py.adapters.metric_lines import metric_lines_adapter
from gymrat_py.errors import CommandError
from gymrat_py.exec import ExecOptions, ExecResult, ExecTimeoutError
from gymrat_py.progress_events import (
    PassFinished,
    PassStarted,
    PrepareFinished,
    PrepareStarted,
    ProgressEvent,
)
from gymrat_py.report.text import format_cleanup_failures
from gymrat_py.sampling import (
    MetricStats,
    SamplingOptions,
    TargetContext,
    TargetSamples,
    collect_samples,
    compute_metric_stats,
    own_values,
    paired_or_own_values,
    resolve_dir,
    resolve_label,
    run_with_worktrees,
)
from gymrat_py.targets import (
    CleanupResult,
    InPlaceTarget,
    RefTarget,
    WorktreeInfo,
    WorktreeRemovalFailure,
)

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

    assert all(e.at_ms > 0 for e in events)
    timestamps = [e.at_ms for e in events]
    assert timestamps == sorted(timestamps)


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
    assert first.target_index == 0
    assert first.target_count == 2
    assert first.label == "old"
    assert first.phase == "measure"

    second = pass_started_events[1]
    assert second.target_index == 1
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


# ---------------------------------------------------------------------------
# worktree run orchestration
# ---------------------------------------------------------------------------


class _InstallRecorder:
    """Capture install/uninstall of the termination cleanup plus event ordering."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.cleanup: Callable[[], None] | None = None

    def install(self, cleanup: Callable[[], None]) -> Callable[[], None]:
        self.events.append("install")
        self.cleanup = cleanup

        def uninstall() -> None:
            self.events.append("uninstall")

        return uninstall


def _capturing_install(
    captured: dict[str, Callable[[], None]],
) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """Build an install seam that records the registered cleanup into ``captured``.

    Returns a no-op uninstall, so a test can invoke ``captured["cleanup"]``
    directly to drive the termination path without a real signal.
    """

    def install(cleanup: Callable[[], None]) -> Callable[[], None]:
        captured["cleanup"] = cleanup
        return lambda: None

    return install


def _patch_cleanup(
    monkeypatch: pytest.MonkeyPatch, result: CleanupResult
) -> list[tuple[list[WorktreeInfo], str]]:
    """Patch the sampling cleanup seam to return ``result`` and record each sweep."""
    sweeps: list[tuple[list[WorktreeInfo], str]] = []

    def _cleanup(worktrees: list[WorktreeInfo], repo_dir: str) -> CleanupResult:
        sweeps.append((list(worktrees), repo_dir))
        return result

    monkeypatch.setattr(sampling, "cleanup_worktrees", _cleanup)
    return sweeps


def _clean_result() -> CleanupResult:
    """A sweep that removed everything with no failures."""
    return CleanupResult(removed=0, failures=(), prune_error=None)


def _dirty_result() -> CleanupResult:
    """A sweep that left a worktree behind and could not prune."""
    return CleanupResult(
        removed=1,
        failures=(WorktreeRemovalFailure(dir="/tmp/gymrat-wt", error="contains modified files"),),
        prune_error="could not prune",
    )


# resolve_dir


def test_resolve_dir_when_in_place_target_does_return_dir_and_leave_worktrees_untouched():
    worktrees: list[WorktreeInfo] = []

    result = resolve_dir(InPlaceTarget(dir="/bench"), "/repo", worktrees)

    assert result == "/bench"
    assert worktrees == []


def test_resolve_dir_when_ref_target_does_register_worktree_before_materialize(
    monkeypatch: pytest.MonkeyPatch,
):
    target = RefTarget(ref="feature", resolved_sha="deadbeef")
    stub = WorktreeInfo(dir="/tmp/gymrat-wt", sha="deadbeef", created=True)

    def fake_plan_worktree(_ref: RefTarget) -> WorktreeInfo:
        return stub

    monkeypatch.setattr(sampling, "plan_worktree", fake_plan_worktree)
    worktrees: list[WorktreeInfo] = []
    registered_before_materialize: list[bool] = []
    materialize_args: list[tuple[WorktreeInfo, str]] = []

    def _materialize(worktree: WorktreeInfo, repo_dir: str) -> None:
        registered_before_materialize.append(worktree in worktrees)
        materialize_args.append((worktree, repo_dir))

    monkeypatch.setattr(sampling, "materialize_worktree", _materialize)

    result = resolve_dir(target, "/repo", worktrees)

    assert result == "/tmp/gymrat-wt"
    assert worktrees == [stub]
    assert registered_before_materialize == [True]
    assert materialize_args == [(stub, "/repo")]


# resolve_label


@pytest.mark.parametrize(
    ("explicit", "target", "expected"),
    [
        pytest.param(
            "custom", RefTarget(ref="feature", resolved_sha="abc"), "custom", id="explicit-wins"
        ),
        pytest.param(None, RefTarget(ref="feature", resolved_sha="abc"), "feature", id="ref-name"),
        pytest.param(None, InPlaceTarget(dir="/some/path/bench"), "bench", id="in-place-basename"),
    ],
)
def test_resolve_label_when_given_inputs_does_return_expected(
    explicit: str | None, target: InPlaceTarget | RefTarget, expected: str
):
    assert resolve_label(explicit, target) == expected


# run_with_worktrees


async def test_run_with_worktrees_when_phase_succeeds_does_run_phase_then_sweep_once_and_build_result(
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = _InstallRecorder()
    monkeypatch.setattr(sampling, "install_termination_cleanup", recorder.install)
    cleanup = _clean_result()
    sweeps = _patch_cleanup(monkeypatch, cleanup)
    phase_args: dict[str, object] = {}

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        recorder.events.append("phase")
        phase_args["repo_dir"] = repo_dir
        phase_args["worktrees"] = worktrees
        phase_args["abort"] = abort
        return "measurement"

    result = await run_with_worktrees(phase, lambda m, c: (m, c))

    assert result == ("measurement", cleanup)
    assert len(sweeps) == 1
    assert recorder.events == ["install", "phase", "uninstall"]
    assert phase_args["repo_dir"] == str(Path.cwd())
    assert phase_args["worktrees"] == []
    assert isinstance(phase_args["abort"], asyncio.Event)


async def test_run_with_worktrees_when_phase_raises_and_cleanup_clean_does_reraise_original(
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = _InstallRecorder()
    monkeypatch.setattr(sampling, "install_termination_cleanup", recorder.install)
    sweeps = _patch_cleanup(monkeypatch, _clean_result())
    original = CommandError("bench command failed", hint="check the target")

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        recorder.events.append("phase")
        raise original

    with pytest.raises(CommandError) as caught:
        await run_with_worktrees(phase, lambda m, c: (m, c))

    assert caught.value is original
    assert len(sweeps) == 1
    assert recorder.events == ["install", "phase", "uninstall"]


async def test_run_with_worktrees_when_phase_raises_and_cleanup_dirty_does_wrap_preserving_subclass(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sampling, "install_termination_cleanup", _InstallRecorder().install)
    cleanup = _dirty_result()
    _patch_cleanup(monkeypatch, cleanup)
    original = CommandError("bench command failed", hint="check the target")

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        raise original

    with pytest.raises(CommandError) as caught:
        await run_with_worktrees(phase, lambda m, c: (m, c))

    details = format_cleanup_failures(cleanup.failures, cleanup.prune_error)
    assert caught.value is not original
    assert isinstance(caught.value, CommandError)
    assert str(caught.value) == "\n".join(
        ["bench command failed", "", "cleanup did not finish:", *details]
    )
    assert caught.value.hint == "check the target"
    assert caught.value.__cause__ is original


async def test_run_with_worktrees_when_termination_cleanup_invoked_does_abort_run_and_sweep(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Callable[[], None]] = {}
    monkeypatch.setattr(sampling, "install_termination_cleanup", _capturing_install(captured))
    sweeps = _patch_cleanup(monkeypatch, _clean_result())
    observed: dict[str, object] = {}

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        before = len(sweeps)
        captured["cleanup"]()
        observed["swept_by_cleanup"] = len(sweeps) - before
        observed["aborted"] = abort.is_set()
        return "measurement"

    await run_with_worktrees(phase, lambda m, c: (m, c))

    assert observed["aborted"] is True
    assert observed["swept_by_cleanup"] == 1


async def test_run_with_worktrees_when_termination_cleanup_invoked_does_kill_groups_before_sweep(
    monkeypatch: pytest.MonkeyPatch,
):
    order: list[str] = []
    captured: dict[str, Callable[[], None]] = {}
    monkeypatch.setattr(sampling, "install_termination_cleanup", _capturing_install(captured))
    monkeypatch.setattr(sampling, "kill_live_process_groups", lambda: order.append("kill"))

    def _cleanup(worktrees: list[WorktreeInfo], repo_dir: str) -> CleanupResult:
        order.append("sweep")
        return _clean_result()

    monkeypatch.setattr(sampling, "cleanup_worktrees", _cleanup)

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        captured["cleanup"]()
        return "measurement"

    await run_with_worktrees(phase, lambda m, c: (m, c))

    assert order[:2] == ["kill", "sweep"]
