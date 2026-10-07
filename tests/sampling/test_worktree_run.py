"""Behavioral tests for worktree run orchestration: target contexts, labels and the sweep."""

import asyncio
import shutil
import signal
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import pytest

from gymrat import sampling
from gymrat.errors import CommandError
from gymrat.report.text.render import format_cleanup_failures
from gymrat.sampling import (
    CleanupResult,
    TargetSpec,
    WorktreeInfo,
    resolve_label,
    run_with_worktrees,
    to_context,
)
from gymrat.targets import InPlaceTarget, RefTarget, WorktreeRemovalFailure
from tests._process_helpers import fake_install


class _InstallRecorder:
    """Capture install/uninstall of the termination cleanup plus event ordering."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def install(self, cleanup: Callable[[], None]) -> Callable[[], None]:
        self.events.append("install")

        def uninstall() -> None:
            self.events.append("uninstall")

        return uninstall


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


# ---------------------------------------------------------------------------
# to_context
# ---------------------------------------------------------------------------


def test_to_context_when_in_place_target_does_return_dir_and_leave_worktrees_untouched():
    worktrees: list[WorktreeInfo] = []

    result = to_context(
        TargetSpec(label=None, target="/bench"), InPlaceTarget(dir="/bench"), "/repo", worktrees
    )

    assert result.dir == "/bench"
    assert worktrees == []


def test_to_context_when_ref_target_does_register_worktree_before_materialize(
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

    result = to_context(TargetSpec(label=None, target="feature"), target, "/repo", worktrees)

    assert result.dir == "/tmp/gymrat-wt"
    assert worktrees == [stub]
    assert registered_before_materialize == [True]
    assert materialize_args == [(stub, "/repo")]


# ---------------------------------------------------------------------------
# resolve_label
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# run_with_worktrees
# ---------------------------------------------------------------------------


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


async def test_run_with_worktrees_when_phase_cancelled_does_skip_the_sweep_and_uninstall(
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = _InstallRecorder()
    monkeypatch.setattr(sampling, "install_termination_cleanup", recorder.install)
    sweeps = _patch_cleanup(monkeypatch, _clean_result())

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        recorder.events.append("phase")
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_with_worktrees(phase, lambda m, c: (m, c))

    assert sweeps == []
    assert recorder.events == ["install", "phase", "uninstall"]


async def test_run_with_worktrees_when_build_result_raises_does_not_sweep_again(
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = _InstallRecorder()
    monkeypatch.setattr(sampling, "install_termination_cleanup", recorder.install)
    sweeps = _patch_cleanup(monkeypatch, _clean_result())
    broken = RuntimeError("report failed")

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        return "measurement"

    def build_result(_measurement: str, _cleanup: CleanupResult) -> str:
        raise broken

    with pytest.raises(RuntimeError) as caught:
        await run_with_worktrees(phase, build_result)

    assert caught.value is broken
    assert len(sweeps) == 1
    assert recorder.events == ["install", "uninstall"]


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
    assert str(caught.value) == "\n".join([
        "bench command failed",
        "",
        "cleanup did not finish:",
        *details,
    ])
    assert caught.value.hint == "check the target"
    assert caught.value.__cause__ is original


async def test_run_with_worktrees_when_other_error_raised_and_cleanup_dirty_does_wrap_as_exception(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sampling, "install_termination_cleanup", _InstallRecorder().install)
    cleanup = _dirty_result()
    _patch_cleanup(monkeypatch, cleanup)
    original = RuntimeError("adapter exploded")

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        raise original

    with pytest.raises(Exception, match="adapter exploded") as caught:
        await run_with_worktrees(phase, lambda m, c: (m, c))

    details = format_cleanup_failures(cleanup.failures, cleanup.prune_error)
    assert type(caught.value) is Exception
    assert str(caught.value) == "\n".join([
        "adapter exploded",
        "",
        "cleanup did not finish:",
        *details,
    ])
    assert caught.value.__cause__ is original


async def test_run_with_worktrees_when_termination_cleanup_invoked_does_abort_run_and_sweep(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[Callable[[], None]] = []
    monkeypatch.setattr(sampling, "install_termination_cleanup", fake_install(captured))
    sweeps = _patch_cleanup(monkeypatch, _clean_result())
    observed: dict[str, object] = {}

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        before = len(sweeps)
        captured[0]()
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
    captured: list[Callable[[], None]] = []
    monkeypatch.setattr(sampling, "install_termination_cleanup", fake_install(captured))
    monkeypatch.setattr(sampling, "kill_live_process_groups", lambda: order.append("kill"))

    def _cleanup(worktrees: list[WorktreeInfo], repo_dir: str) -> CleanupResult:
        order.append("sweep")
        return _clean_result()

    monkeypatch.setattr(sampling, "cleanup_worktrees", _cleanup)

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        captured[0]()
        return "measurement"

    await run_with_worktrees(phase, lambda m, c: (m, c))

    assert order[:2] == ["kill", "sweep"]


async def test_run_with_worktrees_when_terminated_during_the_normal_sweep_does_sweep_each_worktree_once(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[Callable[[], None]] = []
    monkeypatch.setattr(sampling, "install_termination_cleanup", fake_install(captured))
    stub = WorktreeInfo(dir="/tmp/gymrat-wt", sha="deadbeef", created=True)
    swept: list[WorktreeInfo] = []

    def _cleanup(worktrees: list[WorktreeInfo], repo_dir: str) -> CleanupResult:
        first_sweep = not swept
        swept.extend(worktrees)
        if first_sweep:
            captured[0]()
        return _clean_result()

    monkeypatch.setattr(sampling, "cleanup_worktrees", _cleanup)

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        worktrees.append(stub)
        return "measurement"

    await run_with_worktrees(phase, lambda m, c: (m, c))

    assert swept == [stub]


async def test_run_with_worktrees_when_terminated_during_the_sweep_after_a_failed_phase_does_sweep_each_worktree_once(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[Callable[[], None]] = []
    monkeypatch.setattr(sampling, "install_termination_cleanup", fake_install(captured))
    stub = WorktreeInfo(dir="/tmp/gymrat-wt", sha="deadbeef", created=True)
    swept: list[WorktreeInfo] = []
    original = CommandError("bench command failed", hint="check the target")

    def _cleanup(worktrees: list[WorktreeInfo], repo_dir: str) -> CleanupResult:
        first_sweep = not swept
        swept.extend(worktrees)
        if first_sweep:
            captured[0]()
        return _clean_result()

    monkeypatch.setattr(sampling, "cleanup_worktrees", _cleanup)

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        worktrees.append(stub)
        raise original

    with pytest.raises(CommandError):
        await run_with_worktrees(phase, lambda m, c: (m, c))

    assert swept == [stub]


def _cleanup_block(cleanup: CleanupResult) -> str:
    """The stderr text reporting what ``cleanup`` left unfinished."""
    details = format_cleanup_failures(cleanup.failures, cleanup.prune_error)
    return "\n".join(["cleanup did not finish:", *details]) + "\n"


_LEFT_BEHIND = CleanupResult(
    removed=0,
    failures=(WorktreeRemovalFailure(dir="/tmp/gymrat-wt", error="contains modified files"),),
    prune_error=None,
)
_PRUNE_FAILED = CleanupResult(removed=1, failures=(), prune_error="could not prune")


@pytest.mark.parametrize(
    ("cleanup", "expected_stderr"),
    [
        pytest.param(_LEFT_BEHIND, _cleanup_block(_LEFT_BEHIND), id="worktree-left-behind"),
        pytest.param(_PRUNE_FAILED, _cleanup_block(_PRUNE_FAILED), id="prune-failed"),
        pytest.param(_clean_result(), "", id="clean-sweep"),
    ],
)
async def test_run_with_worktrees_when_signalled_does_report_unfinished_cleanup_on_stderr_before_exit(
    cleanup: CleanupResult,
    expected_stderr: str,
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
    capsys: pytest.CaptureFixture[str],
):
    _patch_cleanup(monkeypatch, cleanup)
    at_exit: list[tuple[int, str]] = []

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        code = raise_signal(signal.SIGTERM)
        at_exit.append((code, capsys.readouterr().err))
        return "measurement"

    await run_with_worktrees(phase, lambda m, c: (m, c))

    assert at_exit == [(128 + signal.SIGTERM, expected_stderr)]


# ---------------------------------------------------------------------------
# run_with_worktrees — a signal that takes over the normal sweep
# ---------------------------------------------------------------------------

_REFUSED = "contains modified files"
_PRUNE = ("prune",)


class _Exit(NamedTuple):
    """What the process had done by the moment the signal path exited."""

    code: int
    stderr: str
    git_calls: list[tuple[str, ...]]


def _removal(directory: Path) -> tuple[str, ...]:
    """The recorded shape of one targeted removal of ``directory``."""
    return ("remove", str(directory))


def _patch_git(
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
    capsys: pytest.CaptureFixture[str],
    *,
    refused: Path,
    signal_during: tuple[str, ...],
    prune_error: str | None = None,
) -> list[_Exit]:
    """Stub the sweep's git seam with scripted answers and one call that delivers SIGTERM.

    A removal of ``refused`` fails and leaves the directory as it was; any other
    removal deletes the directory and succeeds. A prune answers ``prune_error``.

    Args:
        monkeypatch: Patches the git seam the sweep calls.
        raise_signal: Delivers the signal from inside the stubbed call.
        capsys: Reads what the signal path wrote before it exited.
        refused: The worktree directory git refuses to remove.
        signal_during: The recorded call the signal lands in.
        prune_error: What a prune answers; ``None`` is success.

    Returns:
        The exits observed, one per delivered signal.
    """
    git_calls: list[tuple[str, ...]] = []
    exits: list[_Exit] = []

    def _try_git(args: Sequence[str], cwd: str) -> str | None:
        call = _PRUNE if "prune" in args else ("remove", args[-1])
        git_calls.append(call)
        if call == signal_during and not exits:
            code = raise_signal(signal.SIGTERM)
            exits.append(_Exit(code, capsys.readouterr().err, sorted(git_calls)))
        if call == _PRUNE:
            return prune_error
        if call == _removal(refused):
            return _REFUSED
        shutil.rmtree(args[-1], ignore_errors=True)
        return None

    monkeypatch.setattr(sampling, "try_git", _try_git)
    return exits


def _worktree_at(directory: Path, *, on_disk: bool = True) -> WorktreeInfo:
    """A created worktree at ``directory``, which exists unless ``on_disk`` is false."""
    if on_disk:
        directory.mkdir()
    return WorktreeInfo(dir=str(directory), sha="deadbeef", created=True)


def _phase_leaving(
    left: list[WorktreeInfo],
) -> Callable[[str, list[WorktreeInfo], asyncio.Event], Awaitable[str]]:
    """A phase that registers ``left`` and succeeds, handing them to the normal sweep."""

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        worktrees.extend(left)
        return "measurement"

    return phase


async def test_run_with_worktrees_when_signalled_after_the_normal_sweep_left_a_worktree_does_report_it_before_exit(
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    left, in_flight, remaining = tmp_path / "left", tmp_path / "in-flight", tmp_path / "remaining"
    worktrees = [_worktree_at(left), _worktree_at(in_flight), _worktree_at(remaining)]
    exits = _patch_git(
        monkeypatch, raise_signal, capsys, refused=left, signal_during=_removal(in_flight)
    )

    await run_with_worktrees(_phase_leaving(worktrees), lambda m, c: (m, c))

    reported = CleanupResult(
        removed=0,
        failures=(WorktreeRemovalFailure(dir=str(left), error=_REFUSED),),
        prune_error=None,
    )
    assert exits == [
        _Exit(
            code=128 + signal.SIGTERM,
            stderr=_cleanup_block(reported),
            git_calls=sorted([_removal(left), _removal(in_flight), _removal(remaining)]),
        )
    ]


@pytest.mark.parametrize(
    ("prune_error", "expected_stderr"),
    [
        pytest.param(None, "", id="prune-succeeds"),
        pytest.param("could not prune", _cleanup_block(_PRUNE_FAILED), id="prune-fails"),
    ],
)
async def test_run_with_worktrees_when_signalled_after_the_normal_sweep_met_a_stale_entry_does_prune_once_before_exit(
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    *,
    prune_error: str | None,
    expected_stderr: str,
):
    stale, in_flight = tmp_path / "stale", tmp_path / "in-flight"
    worktrees = [_worktree_at(stale, on_disk=False), _worktree_at(in_flight)]
    exits = _patch_git(
        monkeypatch,
        raise_signal,
        capsys,
        refused=stale,
        signal_during=_removal(in_flight),
        prune_error=prune_error,
    )

    await run_with_worktrees(_phase_leaving(worktrees), lambda m, c: (m, c))

    assert exits == [
        _Exit(
            code=128 + signal.SIGTERM,
            stderr=expected_stderr,
            git_calls=sorted([_removal(stale), _removal(in_flight), _PRUNE]),
        )
    ]


async def test_run_with_worktrees_when_signalled_during_the_normal_prune_does_exit_without_a_second_prune(
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    stale = tmp_path / "stale"
    exits = _patch_git(monkeypatch, raise_signal, capsys, refused=stale, signal_during=_PRUNE)

    await run_with_worktrees(
        _phase_leaving([_worktree_at(stale, on_disk=False)]), lambda m, c: (m, c)
    )

    assert [(exit_.code, exit_.git_calls) for exit_ in exits] == [
        (128 + signal.SIGTERM, sorted([_removal(stale), _PRUNE]))
    ]
