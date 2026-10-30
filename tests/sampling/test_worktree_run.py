"""Behavioral tests for worktree runs: target contexts, labels, the sweep, and the lifecycle.

The lifecycle tests drive ``plan_worktree``, ``materialize_worktree`` and
``cleanup_worktrees`` against real git in a scratch repository, including the
awkward temp bases a host can hand the planner: a symlinked or slash-terminated
base resolves to its real path, and a read-only base fails without registering
anything.
"""

import asyncio
import contextlib
import shutil
import signal
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from gymrat import sampling
from gymrat.errors import CommandError, GymratError
from gymrat.loop.stop_condition import LoopStopError
from gymrat.report.text.render import format_cleanup_failures
from gymrat.sampling import (
    CleanupResult,
    TargetSpec,
    WorktreeInfo,
    WorktreeRemovalFailure,
    cleanup_worktrees,
    materialize_worktree,
    plan_worktree,
    resolve_label,
    run_with_worktrees,
    to_context,
)
from gymrat.targets import InPlaceTarget, RefTarget
from tests._git import (
    head_of,
    kill_git_during_worktree_add,
    list_worktree_dirs,
    register_absent_worktree,
)
from tests._platform import needs_posix_kill, needs_symlinks
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
    result = resolve_label(explicit, target)

    assert result == expected


# ---------------------------------------------------------------------------
# run_with_worktrees
# ---------------------------------------------------------------------------

_PRUNE = ("prune",)

_REFUSED = "contains modified files"


def _removal(directory: Path) -> tuple[str, ...]:
    """The recorded shape of one targeted removal of ``directory``."""
    return ("remove", str(directory))


def _worktree_at(directory: Path, *, on_disk: bool = True) -> WorktreeInfo:
    """A created worktree at ``directory``, which exists unless ``on_disk`` is false."""
    if on_disk:
        directory.mkdir()
    return WorktreeInfo(dir=str(directory), sha="deadbeef", created=True)


def _git_answer(
    args: Sequence[str], *, refused: Sequence[Path], prune_error: str | None
) -> str | None:
    """How the scripted git seam answers one sweep call.

    A removal of a ``refused`` directory fails and leaves it as it was; any
    other removal deletes the directory and succeeds. A prune answers
    ``prune_error``.

    Args:
        args: The git arguments the sweep passed.
        refused: The worktree directories git refuses to remove.
        prune_error: What a prune answers; ``None`` is success.

    Returns:
        The error git reports, or ``None`` on success.
    """
    if "prune" in args:
        return prune_error
    if Path(args[-1]) in refused:
        return _REFUSED
    shutil.rmtree(args[-1], ignore_errors=True)
    return None


def _fake_git(
    monkeypatch: pytest.MonkeyPatch,
    *,
    refused: Sequence[Path] = (),
    prune_error: str | None = None,
    on_call: Callable[[tuple[str, ...]], object] = lambda _call: None,
) -> list[tuple[str, ...]]:
    """Stub the sweep's git seam with the scripted answers of :func:`_git_answer`.

    Each call is recorded by its shape, then handed to ``on_call`` before git
    answers, so a test can land an event while the sweep is waiting on git.

    Args:
        monkeypatch: Patches the git seam the sweep calls.
        refused: The worktree directories git refuses to remove.
        prune_error: What a prune answers; ``None`` is success.
        on_call: Runs inside each call with that call's recorded shape.

    Returns:
        The recorded calls, in call order.
    """
    git_calls: list[tuple[str, ...]] = []

    def _try_git(args: Sequence[str], cwd: str) -> str | None:
        call = _PRUNE if "prune" in args else ("remove", args[-1])
        git_calls.append(call)
        on_call(call)
        return _git_answer(args, refused=refused, prune_error=prune_error)

    monkeypatch.setattr(sampling, "try_git", _try_git)
    return git_calls


def _on_first_call(event: Callable[[], object]) -> Callable[[tuple[str, ...]], None]:
    """An ``on_call`` hook that runs ``event`` inside the first git call only."""
    fired: list[bool] = []

    def _hook(_call: tuple[str, ...]) -> None:
        if not fired:
            fired.append(True)
            event()

    return _hook


def _cleanup_block(cleanup: CleanupResult) -> list[str]:
    """The detail lines reporting what ``cleanup`` left unfinished."""
    return format_cleanup_failures(cleanup.failures, cleanup.prune_error)


def _reported(stderr: str) -> list[str] | None:
    """The detail lines under the unfinished-cleanup header, or ``None`` when nothing was written."""
    return stderr.splitlines()[1:] if stderr else None


@dataclass(frozen=True, slots=True)
class _Sweep:
    """Worktrees a phase claims, how git answers their sweep, and what that sweep reports."""

    worktrees: list[WorktreeInfo]
    refused: tuple[Path, ...]
    prune_error: str | None
    expected: CleanupResult
    expected_report: list[str] | None


def _nothing_claimed(_tmp_path: Path) -> _Sweep:
    """A phase that claims no worktree, so the sweep has nothing to do."""
    return _Sweep([], (), None, CleanupResult(removed=0, failures=(), prune_error=None), None)


def _one_left_behind(tmp_path: Path) -> _Sweep:
    """One worktree on disk that git refuses to remove."""
    left = tmp_path / "left"
    expected = CleanupResult(
        removed=0,
        failures=(WorktreeRemovalFailure(dir=str(left), error=_REFUSED),),
        prune_error=None,
    )
    return _Sweep([_worktree_at(left)], (left,), None, expected, _cleanup_block(expected))


_PRUNE_FAILED = CleanupResult(removed=1, failures=(), prune_error="could not prune")


def _removed_left_and_prune_failed(tmp_path: Path) -> _Sweep:
    """One worktree removed, one left behind, and a stale entry whose prune fails."""
    removed, left, stale = tmp_path / "removed", tmp_path / "left", tmp_path / "stale"
    expected = CleanupResult(
        removed=1,
        failures=(WorktreeRemovalFailure(dir=str(left), error=_REFUSED),),
        prune_error="could not prune",
    )
    return _Sweep(
        [_worktree_at(removed), _worktree_at(left), _worktree_at(stale, on_disk=False)],
        (left, stale),
        "could not prune",
        expected,
        _cleanup_block(expected),
    )


def _install_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arrange: Callable[[Path], _Sweep]
) -> _Sweep:
    """Lay down ``arrange``'s worktrees and script git to answer their sweep.

    Args:
        monkeypatch: Patches the git seam the sweep calls.
        tmp_path: Where the worktree directories are created.
        arrange: Builds the sweep scenario under ``tmp_path``.

    Returns:
        The scenario, carrying the worktrees a phase should claim.
    """
    sweep = arrange(tmp_path)
    _fake_git(monkeypatch, refused=sweep.refused, prune_error=sweep.prune_error)
    return sweep


def _phase_leaving(
    left: list[WorktreeInfo],
) -> Callable[[str, list[WorktreeInfo], asyncio.Event], Awaitable[str]]:
    """A phase that registers ``left`` and succeeds, handing them to the normal sweep."""

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        worktrees.extend(left)
        return "measurement"

    return phase


async def test_run_with_worktrees_when_phase_succeeds_does_build_the_result_from_one_guarded_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    registry = track_cleanups(monkeypatch, "gymrat.sampling")
    git_calls = _fake_git(monkeypatch)
    claimed = _worktree_at(tmp_path / "wt")
    phase_args: dict[str, object] = {}

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        phase_args["armed"] = len(registry.live())
        worktrees.append(claimed)
        return "measurement"

    result = await run_with_worktrees(phase, lambda m, c: (m, c))

    assert result == ("measurement", CleanupResult(removed=1, failures=(), prune_error=None))
    assert git_calls == [_removal(tmp_path / "wt")]
    assert (phase_args["armed"], registry.live()) == (1, [])


async def test_run_with_worktrees_when_phase_raises_and_cleanup_clean_does_reraise_original(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    registry = track_cleanups(monkeypatch, "gymrat.sampling")
    git_calls = _fake_git(monkeypatch)
    claimed = _worktree_at(tmp_path / "wt")
    original = CommandError("bench command failed", hint="check the target")

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        worktrees.append(claimed)
        raise original

    with pytest.raises(CommandError) as caught:
        await run_with_worktrees(phase, lambda m, c: (m, c))

    assert caught.value is original
    assert git_calls == [_removal(tmp_path / "wt")]
    assert registry.live() == []


async def test_run_with_worktrees_when_phase_cancelled_does_tear_down_without_sweeping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    registry = track_cleanups(monkeypatch, "gymrat.sampling")
    git_calls = _fake_git(monkeypatch)
    claimed = _worktree_at(tmp_path / "wt")

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        worktrees.append(claimed)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_with_worktrees(phase, lambda m, c: (m, c))

    assert git_calls == []
    assert registry.live() == []


async def test_run_with_worktrees_when_build_result_raises_does_propagate_it_without_cleanup_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    track_cleanups(monkeypatch, "gymrat.sampling")
    sweep = _install_sweep(monkeypatch, tmp_path, _removed_left_and_prune_failed)
    broken = RuntimeError("report failed")

    def build_result(_measurement: str, _cleanup: CleanupResult) -> str:
        raise broken

    with pytest.raises(RuntimeError) as caught:
        await run_with_worktrees(_phase_leaving(sweep.worktrees), build_result)

    assert caught.value is broken


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        pytest.param(
            CommandError("bench command failed", hint="check the target", reason="error"),
            (CommandError, "check the target", "error"),
            id="gymrat-error-keeps-its-class-hint-and-reason",
        ),
        pytest.param(
            LoopStopError("loop stopped", reason=None),
            (LoopStopError, None, None),
            id="subclass-keeps-an-explicit-none-over-its-default-reason",
        ),
        pytest.param(
            RuntimeError("adapter exploded"),
            (Exception, None, None),
            id="other-error-becomes-exception",
        ),
    ],
)
async def test_run_with_worktrees_when_phase_raises_and_cleanup_dirty_does_wrap_with_the_unfinished_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    original: Exception,
    expected: tuple[type[Exception], str | None, str | None],
):
    wrapped_type, hint, reason = expected
    track_cleanups(monkeypatch, "gymrat.sampling")
    sweep = _install_sweep(monkeypatch, tmp_path, _removed_left_and_prune_failed)
    details = _cleanup_block(sweep.expected)

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        worktrees.extend(sweep.worktrees)
        raise original

    with pytest.raises(Exception) as caught:  # noqa: PT011 -- the row names the exact type below
        await run_with_worktrees(phase, lambda m, c: (m, c))

    assert type(caught.value) is wrapped_type
    lines = str(caught.value).splitlines()
    assert lines[0] == str(original)
    assert lines[-len(details) :] == details
    assert getattr(caught.value, "hint", None) == hint
    assert getattr(caught.value, "reason", None) == reason
    assert caught.value.__cause__ is original


async def _return_measurement() -> str:
    return "measurement"


async def _raise_bench_failure() -> str:
    message = "bench command failed"
    raise CommandError(message, hint="check the target")


@pytest.mark.parametrize(
    ("finish", "expectation"),
    [
        pytest.param(_return_measurement, contextlib.nullcontext(), id="after-a-successful-phase"),
        pytest.param(_raise_bench_failure, pytest.raises(CommandError), id="after-a-failed-phase"),
    ],
)
async def test_run_with_worktrees_when_terminated_during_the_sweep_does_sweep_each_worktree_once(
    monkeypatch: pytest.MonkeyPatch,
    finish: Callable[[], Awaitable[str]],
    expectation: contextlib.AbstractContextManager[object],
    tmp_path: Path,
):
    captured: list[Callable[[], None]] = []
    monkeypatch.setattr(sampling, "install_termination_cleanup", fake_install(captured))

    def terminate() -> None:
        captured[0]()

    git_calls = _fake_git(monkeypatch, on_call=_on_first_call(terminate))
    claimed = _worktree_at(tmp_path / "wt")

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        worktrees.append(claimed)
        return await finish()

    with expectation:
        await run_with_worktrees(phase, lambda m, c: (m, c))

    assert git_calls == [_removal(tmp_path / "wt")]


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_one_left_behind, id="worktree-left-behind"),
        pytest.param(_nothing_claimed, id="clean-sweep"),
    ],
)
async def test_run_with_worktrees_when_signalled_does_report_unfinished_cleanup_on_stderr_before_exit(
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    *,
    arrange: Callable[[Path], _Sweep],
):
    sweep = _install_sweep(monkeypatch, tmp_path, arrange)
    at_exit: list[list[str] | None] = []

    async def phase(repo_dir: str, worktrees: list[WorktreeInfo], abort: asyncio.Event) -> str:
        worktrees.extend(sweep.worktrees)
        raise_signal(signal.SIGTERM)
        at_exit.append(_reported(capsys.readouterr().err))
        return "measurement"

    await run_with_worktrees(phase, lambda m, c: (m, c))

    assert at_exit == [sweep.expected_report]


# ---------------------------------------------------------------------------
# run_with_worktrees — a signal that takes over the normal sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Exit:
    """What the process had done by the moment the signal path exited."""

    report: list[str] | None
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

    Git answers as :func:`_fake_git` scripts it, with ``refused`` as the one
    directory it refuses to remove.

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
    exits: list[_Exit] = []

    def _deliver(call: tuple[str, ...]) -> None:
        if call == signal_during and not exits:
            raise_signal(signal.SIGTERM)
            exits.append(_Exit(_reported(capsys.readouterr().err), sorted(git_calls)))

    git_calls = _fake_git(
        monkeypatch, refused=(refused,), prune_error=prune_error, on_call=_deliver
    )
    return exits


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
    reported = CleanupResult(
        removed=0,
        failures=(WorktreeRemovalFailure(dir=str(left), error=_REFUSED),),
        prune_error=None,
    )

    await run_with_worktrees(_phase_leaving(worktrees), lambda m, c: (m, c))

    assert exits == [
        _Exit(
            report=_cleanup_block(reported),
            git_calls=sorted([_removal(left), _removal(in_flight), _removal(remaining)]),
        )
    ]


async def test_run_with_worktrees_when_signalled_after_the_normal_sweep_met_a_stale_entry_does_prune_once_before_exit(
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    stale, in_flight = tmp_path / "stale", tmp_path / "in-flight"
    worktrees = [_worktree_at(stale, on_disk=False), _worktree_at(in_flight)]
    exits = _patch_git(
        monkeypatch,
        raise_signal,
        capsys,
        refused=stale,
        signal_during=_removal(in_flight),
        prune_error="could not prune",
    )

    await run_with_worktrees(_phase_leaving(worktrees), lambda m, c: (m, c))

    assert exits == [
        _Exit(
            report=_cleanup_block(_PRUNE_FAILED),
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

    assert [exit_.git_calls for exit_ in exits] == [sorted([_removal(stale), _PRUNE])]


# ---------------------------------------------------------------------------
# worktree lifecycle: plan, materialize, sweep
# ---------------------------------------------------------------------------


def _plan_and_attempt_materialize(target: RefTarget, repo_dir: str) -> WorktreeInfo:
    """Plan a worktree and try to materialize it, swallowing a failed add."""
    worktree = plan_worktree(target)
    with contextlib.suppress(GymratError):
        materialize_worktree(worktree, repo_dir)
    return worktree


def _create_head_worktree(repo_dir: str) -> WorktreeInfo:
    sha = head_of(repo_dir)
    worktree = plan_worktree(RefTarget(ref=sha, resolved_sha=sha))
    materialize_worktree(worktree, repo_dir)
    return worktree


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


@pytest.mark.parametrize(
    "point_base",
    [
        pytest.param(_default_temp_base, id="host-temp-base"),
        pytest.param(_symlinked_temp_base, id="symlinked-base", marks=needs_symlinks),
    ],
)
def test_plan_worktree_when_given_ref_target_does_name_a_dir_under_the_real_temp_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    point_base: Callable[[pytest.MonkeyPatch, Path], str],
):
    real_base = point_base(monkeypatch, tmp_path)

    worktree = plan_worktree(RefTarget(ref="my-tag", resolved_sha=UNKNOWN_SHA))

    assert worktree.sha == UNKNOWN_SHA
    assert str(Path(worktree.dir).parent) == real_base


def test_plan_worktree_when_tmpdir_nonexistent_does_raise_naming_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    bogus_dir = str(tmp_path / "no-such-tmpdir")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: bogus_dir)

    with pytest.raises(GymratError) as exc_info:
        plan_worktree(RefTarget(ref="v1", resolved_sha=UNKNOWN_SHA))

    assert bogus_dir in str(exc_info.value)


def test_materialize_worktree_when_git_refuses_does_fail_leaving_no_worktree(repo: str):
    worktree = plan_worktree(RefTarget(ref="bad-sha", resolved_sha=UNKNOWN_SHA))

    with pytest.raises(GymratError) as exc_info:
        materialize_worktree(worktree, repo)

    assert worktree.sha in str(exc_info.value)
    assert worktree.created is False


@needs_posix_kill
def test_materialize_worktree_when_add_interrupted_does_set_created_from_disk_state(repo: str):
    kill_git_during_worktree_add(repo)
    sha = head_of(repo)
    worktree = plan_worktree(RefTarget(ref=sha, resolved_sha=sha))

    with pytest.raises(GymratError):
        materialize_worktree(worktree, repo)

    assert worktree.created is True
    assert Path(worktree.dir).exists()


def test_cleanup_worktrees_when_given_non_empty_list_does_report_removing_only_those(repo: str):
    absent = register_absent_worktree(repo)
    worktree = _create_head_worktree(repo)

    result = cleanup_worktrees([worktree], repo)

    assert result == CleanupResult(removed=1, failures=(), prune_error=None)
    assert list_worktree_dirs(repo, include_main=False) == [absent]


def test_cleanup_worktrees_when_add_left_nothing_does_count_it_as_neither_removed_nor_failed(
    repo: str,
):
    absent = register_absent_worktree(repo)
    worktree = _plan_and_attempt_materialize(
        RefTarget(ref="missing", resolved_sha=UNKNOWN_SHA), repo
    )

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


def test_cleanup_worktrees_when_removal_fails_does_record_the_failure_without_stopping_the_sweep(
    repo: str, tmp_path: Path
):
    absent = register_absent_worktree(repo)
    stray = _create_stray_worktree(tmp_path)
    worktree = _create_head_worktree(repo)

    result = cleanup_worktrees([stray, worktree], repo)

    assert [failure.dir for failure in result.failures] == [stray.dir]
    assert result.removed == 1
    assert absent in list_worktree_dirs(repo)


def test_cleanup_worktrees_when_prune_sweep_fails_does_report_prune_error_instead_of_raising(
    tmp_path: Path,
):
    vanished = WorktreeInfo(dir=str(tmp_path / "gone"), sha=UNKNOWN_SHA, created=True)

    result = cleanup_worktrees([vanished], str(tmp_path))

    assert (result.removed, result.failures) == (0, ())
    assert result.prune_error is not None


def test_cleanup_worktrees_when_swept_twice_does_report_nothing_the_second_time(repo: str):
    worktree = _create_head_worktree(repo)
    absent = register_absent_worktree(repo)
    cleanup_worktrees([worktree], repo)

    second = cleanup_worktrees([worktree], repo)

    assert second == CleanupResult(removed=0, failures=(), prune_error=None)
    assert list_worktree_dirs(repo, include_main=False) == [absent]
