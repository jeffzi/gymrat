"""Behavioral tests for worktree runs: target contexts, labels, the sweep, and the lifecycle.

The lifecycle tests drive ``plan_worktree``, ``materialize_worktree`` and
``cleanup_worktrees`` against real git in a scratch repository, including the
awkward temp bases a host can hand the planner: a symlinked or slash-terminated
base resolves to its real path, and a read-only base fails without registering
anything.
"""

import asyncio
import contextlib
import os
import shutil
import signal
import sys
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from gymrat import sampling
from gymrat.errors import CommandError, GymratError
from gymrat.report.text.render import format_cleanup_failures
from gymrat.sampling import (
    CleanupResult,
    TargetSpec,
    WorktreeInfo,
    cleanup_worktrees,
    materialize_worktree,
    plan_worktree,
    resolve_label,
    run_with_worktrees,
    to_context,
)
from gymrat.targets import InPlaceTarget, RefTarget, WorktreeRemovalFailure
from tests._git import (
    head_of,
    kill_git_during_worktree_add,
    list_worktree_dirs,
    register_absent_worktree,
)
from tests._mode_bits import needs_mode_bits
from tests._pipeline import CLEAN_RESULT, DIRTY_RESULT, install_cleanup
from tests._process_helpers import fake_install, track_cleanups

# A sha no repository holds, so ``git worktree add`` rejects it outright.
UNKNOWN_SHA = "0" * 40


# ---------------------------------------------------------------------------
# to_context
# ---------------------------------------------------------------------------


def test_to_context_when_in_place_target_does_run_in_its_own_dir_without_a_worktree():
    worktrees: list[WorktreeInfo] = []

    result = to_context(
        TargetSpec(label=None, target="/bench"), InPlaceTarget(dir="/bench"), "/repo", worktrees
    )

    assert result.dir == "/bench"
    assert worktrees == []


def test_to_context_when_ref_target_does_run_in_its_registered_worktree(repo: str):
    sha = head_of(repo)
    worktrees: list[WorktreeInfo] = []

    result = to_context(
        TargetSpec(label=None, target=sha), RefTarget(ref=sha, resolved_sha=sha), repo, worktrees
    )

    assert [worktree.dir for worktree in worktrees] == [result.dir]
    assert (Path(result.dir) / "README.md").is_file()


def test_to_context_when_materialize_fails_does_leave_the_worktree_registered_for_the_sweep(
    repo: str,
):
    worktrees: list[WorktreeInfo] = []

    with pytest.raises(GymratError):
        to_context(
            TargetSpec(label=None, target="missing"),
            RefTarget(ref="missing", resolved_sha=UNKNOWN_SHA),
            repo,
            worktrees,
        )

    assert [worktree.sha for worktree in worktrees] == [UNKNOWN_SHA]


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
def test_resolve_label_when_given_target_does_prefer_explicit_label_then_ref_name_then_dir_basename(
    explicit: str | None, target: InPlaceTarget | RefTarget, expected: str
):
    assert resolve_label(explicit, target) == expected


# ---------------------------------------------------------------------------
# run_with_worktrees
# ---------------------------------------------------------------------------

_PRUNE = ("prune",)


def _removal(directory: Path) -> tuple[str, ...]:
    """The recorded shape of one targeted removal of ``directory``."""
    return ("remove", str(directory))


def _worktree_at(directory: Path, *, on_disk: bool = True) -> WorktreeInfo:
    """A created worktree at ``directory``, which exists unless ``on_disk`` is false."""
    if on_disk:
        directory.mkdir()
    return WorktreeInfo(dir=str(directory), sha="deadbeef", created=True)


def _record_git(
    monkeypatch: pytest.MonkeyPatch, *, on_first_call: Callable[[], object] = lambda: None
) -> list[tuple[str, ...]]:
    """Stub the sweep's git seam so every call succeeds, recording the shape of each.

    Args:
        monkeypatch: Patches the git seam the sweep calls.
        on_first_call: Runs inside the first call, before git answers, to land an
            event while the sweep is waiting on git.

    Returns:
        The recorded calls, in call order.
    """
    git_calls: list[tuple[str, ...]] = []

    def _try_git(args: Sequence[str], cwd: str) -> str | None:
        git_calls.append(_PRUNE if "prune" in args else ("remove", args[-1]))
        if len(git_calls) == 1:
            on_first_call()
        return None

    monkeypatch.setattr(sampling, "try_git", _try_git)
    return git_calls


async def test_run_with_worktrees_when_phase_succeeds_does_build_the_result_from_one_guarded_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    registry = track_cleanups(monkeypatch, "gymrat.sampling")
    git_calls = _record_git(monkeypatch)
    claimed = _worktree_at(tmp_path / "wt")
    phase_args: dict[str, object] = {}

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        phase_args["armed"] = len(registry.live())
        phase_args["repo_dir"] = repo_dir
        phase_args["abort"] = abort
        worktrees.append(claimed)
        return "measurement"

    result = await run_with_worktrees(phase, lambda m, c: (m, c))

    assert result == ("measurement", CleanupResult(removed=1, failures=(), prune_error=None))
    assert git_calls == [_removal(tmp_path / "wt")]
    assert (phase_args["armed"], registry.live()) == (1, [])
    assert phase_args["repo_dir"] == str(Path.cwd())
    assert isinstance(phase_args["abort"], asyncio.Event)


async def test_run_with_worktrees_when_phase_raises_and_cleanup_clean_does_reraise_original(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    registry = track_cleanups(monkeypatch, "gymrat.sampling")
    git_calls = _record_git(monkeypatch)
    claimed = _worktree_at(tmp_path / "wt")
    original = CommandError("bench command failed", hint="check the target")
    armed: list[int] = []

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        armed.append(len(registry.live()))
        worktrees.append(claimed)
        raise original

    with pytest.raises(CommandError) as caught:
        await run_with_worktrees(phase, lambda m, c: (m, c))

    assert caught.value is original
    assert git_calls == [_removal(tmp_path / "wt")]
    assert (armed, registry.live()) == ([1], [])


async def test_run_with_worktrees_when_phase_cancelled_does_tear_down_without_sweeping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    registry = track_cleanups(monkeypatch, "gymrat.sampling")
    git_calls = _record_git(monkeypatch)
    claimed = _worktree_at(tmp_path / "wt")
    armed: list[int] = []

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        armed.append(len(registry.live()))
        worktrees.append(claimed)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_with_worktrees(phase, lambda m, c: (m, c))

    assert git_calls == []
    assert (armed, registry.live()) == ([1], [])


async def test_run_with_worktrees_when_build_result_raises_does_propagate_it_without_cleanup_details(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = track_cleanups(monkeypatch, "gymrat.sampling")
    install_cleanup(monkeypatch, DIRTY_RESULT)
    broken = RuntimeError("report failed")

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        return "measurement"

    def build_result(_measurement: str, _cleanup: CleanupResult) -> str:
        raise broken

    with pytest.raises(RuntimeError) as caught:
        await run_with_worktrees(phase, build_result)

    assert caught.value is broken
    assert registry.live() == []


@pytest.mark.parametrize(
    ("original", "wrapped_type", "hint"),
    [
        pytest.param(
            CommandError("bench command failed", hint="check the target"),
            CommandError,
            "check the target",
            id="gymrat-error-keeps-its-class-and-hint",
        ),
        pytest.param(
            RuntimeError("adapter exploded"), Exception, None, id="other-error-becomes-exception"
        ),
    ],
)
async def test_run_with_worktrees_when_phase_raises_and_cleanup_dirty_does_wrap_with_the_unfinished_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    original: Exception,
    wrapped_type: type[Exception],
    hint: str | None,
):
    track_cleanups(monkeypatch, "gymrat.sampling")
    cleanup = DIRTY_RESULT
    install_cleanup(monkeypatch, cleanup)

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        raise original

    with pytest.raises(Exception) as caught:  # noqa: PT011 -- the row names the exact type below
        await run_with_worktrees(phase, lambda m, c: (m, c))

    details = format_cleanup_failures(cleanup.failures, cleanup.prune_error)
    assert type(caught.value) is wrapped_type
    assert str(caught.value) == "\n".join([
        str(original),
        "",
        "cleanup did not finish:",
        *details,
    ])
    assert getattr(caught.value, "hint", None) == hint
    assert caught.value.__cause__ is original


async def test_run_with_worktrees_when_termination_cleanup_invoked_does_abort_kill_then_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    captured: list[Callable[[], None]] = []
    monkeypatch.setattr(sampling, "install_termination_cleanup", fake_install(captured))
    git_calls = _record_git(monkeypatch)
    git_calls_at_kill: list[list[tuple[str, ...]]] = []
    monkeypatch.setattr(
        sampling, "kill_live_process_groups", lambda: git_calls_at_kill.append(list(git_calls))
    )
    claimed = _worktree_at(tmp_path / "wt")
    observed: dict[str, object] = {}

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        worktrees.append(claimed)
        captured[0]()
        observed["git_calls"] = list(git_calls)
        observed["aborted"] = abort.is_set()
        return "measurement"

    await run_with_worktrees(phase, lambda m, c: (m, c))

    assert git_calls_at_kill == [[]]
    assert observed == {"git_calls": [_removal(tmp_path / "wt")], "aborted": True}


async def _return_measurement() -> str:
    return "measurement"


async def _raise_bench_failure() -> str:
    message = "bench command failed"
    raise CommandError(message, hint="check the target")


@pytest.mark.parametrize(
    ("finish", "raised"),
    [
        pytest.param(_return_measurement, None, id="after-a-successful-phase"),
        pytest.param(_raise_bench_failure, CommandError, id="after-a-failed-phase"),
    ],
)
async def test_run_with_worktrees_when_terminated_during_the_sweep_does_sweep_each_worktree_once(
    monkeypatch: pytest.MonkeyPatch,
    finish: Callable[[], Awaitable[str]],
    raised: type[BaseException] | None,
    tmp_path: Path,
):
    captured: list[Callable[[], None]] = []
    monkeypatch.setattr(sampling, "install_termination_cleanup", fake_install(captured))

    def terminate() -> None:
        captured[0]()

    git_calls = _record_git(monkeypatch, on_first_call=terminate)
    claimed = _worktree_at(tmp_path / "wt")

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        worktrees.append(claimed)
        return await finish()

    with pytest.raises(raised) if raised is not None else contextlib.nullcontext():
        await run_with_worktrees(phase, lambda m, c: (m, c))

    assert git_calls == [_removal(tmp_path / "wt")]


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
        pytest.param(CLEAN_RESULT, "", id="clean-sweep"),
    ],
)
async def test_run_with_worktrees_when_signalled_does_report_unfinished_cleanup_on_stderr_before_exit(
    cleanup: CleanupResult,
    expected_stderr: str,
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
    capsys: pytest.CaptureFixture[str],
):
    install_cleanup(monkeypatch, cleanup)
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


@dataclass(frozen=True, slots=True)
class _Exit:
    """What the process had done by the moment the signal path exited."""

    code: int
    stderr: str
    git_calls: list[tuple[str, ...]]


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


# ---------------------------------------------------------------------------
# worktree lifecycle: plan, materialize, sweep
# ---------------------------------------------------------------------------

skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX signal delivery to git and real symlinks are required"
)


def _plan_and_attempt_materialize(target: RefTarget, repo_dir: str) -> tuple[WorktreeInfo, bool]:
    """Plan a worktree and materialize it, reporting failure instead of raising."""
    worktree = plan_worktree(target)
    try:
        materialize_worktree(worktree, repo_dir)
    except GymratError:
        return worktree, True
    return worktree, False


def _create_head_worktree(repo_dir: str) -> WorktreeInfo:
    sha = head_of(repo_dir)
    worktree = plan_worktree(RefTarget(ref=sha, resolved_sha=sha))
    materialize_worktree(worktree, repo_dir)
    return worktree


def _leave_interrupted_worktree(repo_dir: str) -> WorktreeInfo:
    """Leave a worktree a killed ``git worktree add`` never returned success for.

    Args:
        repo_dir: The repository the worktree is added from.

    Returns:
        The interrupted worktree, still on disk.

    Raises:
        AssertionError: When git cleaned up despite the kill, so the test that
            asked for this state fails instead of running half-arranged.
    """
    sha = head_of(repo_dir)
    worktree, failed = _plan_and_attempt_materialize(RefTarget(ref=sha, resolved_sha=sha), repo_dir)
    if failed and Path(worktree.dir).exists():
        return worktree
    shutil.rmtree(worktree.dir, ignore_errors=True)
    message = f"expected an interrupted worktree left at {worktree.dir}"
    raise AssertionError(message)


def _plan_rejected_worktree(repo_dir: str) -> WorktreeInfo:
    """Plan a worktree whose ``git worktree add`` fails before creating anything."""
    worktree, failed = _plan_and_attempt_materialize(
        RefTarget(ref="missing", resolved_sha=UNKNOWN_SHA), repo_dir
    )
    if failed and not Path(worktree.dir).exists():
        return worktree
    message = f"expected 'git worktree add' to create nothing at {worktree.dir}"
    raise AssertionError(message)


def _create_stray_worktree(tmp_path: Path) -> WorktreeInfo:
    """A real directory that is not a git worktree, so removal fails."""
    stray_dir = tmp_path / "stray"
    stray_dir.mkdir()
    return WorktreeInfo(dir=str(stray_dir), sha=UNKNOWN_SHA, created=True)


def _default_temp_base(_monkeypatch: pytest.MonkeyPatch, _tmp_path: Path) -> str:
    """Leave ``tempfile.gettempdir`` alone; the planner resolves the host's own base."""
    return str(Path(tempfile.gettempdir()).resolve())


def _symlinked_temp_base(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Route ``tempfile.gettempdir`` through a symlink to a real base under ``tmp_path``."""
    real_base = tmp_path / "real-base"
    real_base.mkdir()
    link = tmp_path / "link-real-base"
    link.symlink_to(real_base)
    # Patched directly, not through TMPDIR: gettempdir caches its first answer.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(link))
    return str(real_base.resolve())


def _slashed_temp_base(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Route ``tempfile.gettempdir`` at a real base spelled with a trailing separator."""
    real_base = tmp_path / "real-base"
    real_base.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(real_base) + os.sep)
    return str(real_base.resolve())


@pytest.mark.parametrize(
    "point_base",
    [
        pytest.param(_default_temp_base, id="host-temp-base"),
        pytest.param(_symlinked_temp_base, id="symlinked-base", marks=skip_on_windows),
        pytest.param(_slashed_temp_base, id="trailing-slash-base"),
    ],
)
def test_plan_worktree_when_given_ref_target_does_name_an_uncreated_dir_under_the_real_temp_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    point_base: Callable[[pytest.MonkeyPatch, Path], str],
):
    real_base = point_base(monkeypatch, tmp_path)

    worktree = plan_worktree(RefTarget(ref="my-tag", resolved_sha=UNKNOWN_SHA))

    assert (worktree.sha, worktree.created) == (UNKNOWN_SHA, False)
    assert str(Path(worktree.dir).parent) == real_base
    assert worktree.dir == str(Path(worktree.dir))
    assert not Path(worktree.dir).exists()


def test_plan_worktree_when_tmpdir_nonexistent_does_raise_naming_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    bogus_dir = str(tmp_path / "no-such-tmpdir")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: bogus_dir)

    with pytest.raises(GymratError) as exc_info:
        plan_worktree(RefTarget(ref="v1", resolved_sha=UNKNOWN_SHA))

    assert bogus_dir in str(exc_info.value)


def test_materialize_worktree_when_given_planned_worktree_does_check_out_ref_files(repo: str):
    sha = head_of(repo)
    worktree = plan_worktree(RefTarget(ref=sha, resolved_sha=sha))

    materialize_worktree(worktree, repo)

    assert (Path(worktree.dir) / "README.md").read_text(encoding="utf-8") == "# Test Repo\n"


def _plan_unknown_sha(
    repo: str, _monkeypatch: pytest.MonkeyPatch, _tmp_path: Path
) -> tuple[WorktreeInfo, str]:
    """A worktree for a sha the repository does not hold; git names the sha in its refusal."""
    return plan_worktree(RefTarget(ref="bad-sha", resolved_sha=UNKNOWN_SHA)), UNKNOWN_SHA


def _plan_under_read_only_base(
    repo: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[WorktreeInfo, str]:
    """A worktree planned under a temp base git cannot write into; the error names the base."""
    read_only_base = tmp_path / "read-only-base"
    read_only_base.mkdir()
    read_only_base.chmod(0o500)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(read_only_base))
    sha = head_of(repo)
    return plan_worktree(RefTarget(ref=sha, resolved_sha=sha)), str(read_only_base.resolve())


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_plan_unknown_sha, id="git-rejects-the-sha"),
        pytest.param(_plan_under_read_only_base, id="read-only-temp-base", marks=needs_mode_bits),
    ],
)
def test_materialize_worktree_when_git_refuses_does_fail_with_its_stderr_leaving_no_worktree(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arrange: Callable[[str, pytest.MonkeyPatch, Path], tuple[WorktreeInfo, str]],
):
    worktree, named = arrange(repo, monkeypatch, tmp_path)

    with pytest.raises(GymratError) as exc_info:
        materialize_worktree(worktree, repo)

    message = str(exc_info.value)
    assert message.startswith(f"git worktree add failed for {worktree.sha}: ")
    assert named in message
    assert "returned non-zero exit status" not in message
    assert worktree.created is False
    assert list_worktree_dirs(repo, include_main=False) == []


@pytest.mark.usefixtures("unusable_git")
def test_materialize_worktree_when_git_cannot_be_started_does_raise_gymrat_error(tmp_path: Path):
    worktree = plan_worktree(RefTarget(ref="main", resolved_sha=UNKNOWN_SHA))

    with pytest.raises(GymratError, match=rf"^git worktree add failed for {UNKNOWN_SHA}: .+"):
        materialize_worktree(worktree, str(tmp_path))


@skip_on_windows
def test_materialize_worktree_when_add_interrupted_does_set_created_from_disk_state(repo: str):
    kill_git_during_worktree_add(repo)
    sha = head_of(repo)
    worktree = plan_worktree(RefTarget(ref=sha, resolved_sha=sha))

    with contextlib.suppress(GymratError):
        materialize_worktree(worktree, repo)

    assert worktree.created is True
    assert Path(worktree.dir).exists()


def test_cleanup_worktrees_when_given_non_empty_list_does_report_removing_only_those(repo: str):
    absent = register_absent_worktree(repo)
    worktree = _create_head_worktree(repo)

    result = cleanup_worktrees([worktree], repo)

    assert result == CleanupResult(removed=1, failures=(), prune_error=None)
    assert not Path(worktree.dir).exists()
    assert list_worktree_dirs(repo, include_main=False) == [absent]


def test_cleanup_worktrees_when_list_empty_outside_repo_does_skip_prune_and_report_no_error(
    tmp_path: Path,
):
    result = cleanup_worktrees([], str(tmp_path))

    assert result == CleanupResult(removed=0, failures=(), prune_error=None)


@skip_on_windows
def test_cleanup_worktrees_when_add_was_killed_does_remove_like_a_normal_worktree(repo: str):
    kill_git_during_worktree_add(repo)
    worktree = _leave_interrupted_worktree(repo)

    try:
        result = cleanup_worktrees([worktree], repo)

        assert result.removed == 1
        assert not Path(worktree.dir).exists()
        assert list_worktree_dirs(repo, include_main=False) == []
    finally:
        shutil.rmtree(worktree.dir, ignore_errors=True)


def test_cleanup_worktrees_when_add_left_nothing_does_count_it_as_neither_removed_nor_failed(
    repo: str,
):
    absent = register_absent_worktree(repo)
    worktree = _plan_rejected_worktree(repo)

    result = cleanup_worktrees([worktree], repo)

    assert result == CleanupResult(removed=0, failures=(), prune_error=None)
    assert absent in list_worktree_dirs(repo)


def test_cleanup_worktrees_when_dir_gone_does_deregister_only_that_worktree(repo: str):
    absent = register_absent_worktree(repo)
    worktree = _create_head_worktree(repo)
    shutil.rmtree(worktree.dir, ignore_errors=True)

    result = cleanup_worktrees([worktree], repo)

    listed = list_worktree_dirs(repo)
    assert worktree.dir not in listed
    assert absent in listed
    assert result == CleanupResult(removed=0, failures=(), prune_error=None)


def test_cleanup_worktrees_when_removal_fails_does_report_git_error_and_still_remove_later_worktrees(
    repo: str, tmp_path: Path
):
    absent = register_absent_worktree(repo)
    stray = _create_stray_worktree(tmp_path)
    worktree = _create_head_worktree(repo)

    result = cleanup_worktrees([stray, worktree], repo)

    assert [failure.dir for failure in result.failures] == [stray.dir]
    assert "is not a working tree" in result.failures[0].error
    assert "returned non-zero exit status" not in result.failures[0].error
    assert result.removed == 1
    assert not Path(worktree.dir).exists()
    assert absent in list_worktree_dirs(repo)


def test_cleanup_worktrees_when_prune_sweep_fails_does_report_prune_error_instead_of_raising(
    tmp_path: Path,
):
    vanished = WorktreeInfo(dir=str(tmp_path / "gone"), sha=UNKNOWN_SHA, created=True)

    result = cleanup_worktrees([vanished], str(tmp_path))

    assert (result.removed, result.failures) == (0, ())
    assert result.prune_error is not None
    assert "not a git repository" in result.prune_error


def test_cleanup_worktrees_when_swept_twice_does_report_nothing_the_second_time(repo: str):
    worktree = _create_head_worktree(repo)
    absent = register_absent_worktree(repo)
    cleanup_worktrees([worktree], repo)

    second = cleanup_worktrees([worktree], repo)

    assert second == CleanupResult(removed=0, failures=(), prune_error=None)
    assert not Path(worktree.dir).exists()
    assert list_worktree_dirs(repo, include_main=False) == [absent]
