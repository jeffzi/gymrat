"""Behavioral tests for containing a freshly spawned child, through ``spawn_contained`` and both exec forms.

A child that exists but could not be registered, contained, or resumed is torn
down before the spawn fails, and the failure always surfaces as a
:class:`~gymrat.exec.SpawnError` — or, through ``exec`` and ``exec_argv``, as a
failed :class:`~gymrat.exec.ExecResult` — even when the teardown fails too.

Real-subprocess tests are POSIX-only: process groups and ``os.killpg`` do not
exist on win32.
"""

import asyncio
import errno
import itertools
import sys
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import NoReturn

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError, exec_argv
from tests._process_helpers import (
    KILLPG_FAILED,
    SLEEPER_ARGV,
    capture_spawns,
    kill_surviving_groups,
    record_registry_sweep,
    refuse_resume,
)

if sys.platform == "win32":
    pytest.skip("POSIX-only process groups", allow_module_level=True)

# Upper bound each awaited run or spawn gets before the test fails outright.
_WAIT_TIMEOUT_S = 10


@pytest.fixture(autouse=True)
def _isolate_live_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the module-level live-group registry from bleeding across tests."""
    monkeypatch.setattr(exec_mod, "_live_process_groups", set())


@pytest.fixture
def options(tmp_path: Path) -> ExecOptions:
    """Run settings rooted at the test's ``tmp_path``."""
    return ExecOptions(cwd=str(tmp_path))


@pytest.fixture
def spawned_by_either_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[list[asyncio.subprocess.Process]]]:
    """Record every child ``exec_argv`` or ``exec`` spawns, and kill any survivor's group."""
    captured = [
        capture_spawns(monkeypatch, "create_subprocess_exec"),
        capture_spawns(monkeypatch, "create_subprocess_shell"),
    ]
    yield captured
    kill_surviving_groups(itertools.chain.from_iterable(captured))


async def run_argv_sleeper(options: ExecOptions) -> ExecResult | ExecTimeoutError:
    """Run a 30 s sleep through ``exec_argv``."""
    return await exec_argv(list(SLEEPER_ARGV), options)


async def run_shell_sleeper(options: ExecOptions) -> ExecResult | ExecTimeoutError:
    """Run a 30 s sleep through the shell form, ``exec``."""
    return await exec_mod.exec("sleep 30", options)


def _raise_on_call(error: Exception) -> Callable[[int], NoReturn]:
    """Build a stand-in that raises ``error`` when called with a pid."""

    def fail(_pid: int) -> NoReturn:
        raise error

    return fail


@pytest.mark.parametrize(
    ("failure", "expected_fragment"),
    [
        pytest.param(
            ("attach_process_group", _raise_on_call(RuntimeWarning("containment refused"))),
            "containment refused",
            id="attach-warning-escalated-to-error",
        ),
        pytest.param(
            (
                "resume_process_group",
                _raise_on_call(OSError(errno.EPERM, "containment refused")),
            ),
            "containment refused",
            id="resume-os-error",
        ),
        pytest.param(
            ("resume_process_group", refuse_resume),
            "could not be resumed",
            id="resume-refused",
        ),
    ],
)
@pytest.mark.parametrize(
    "run",
    [pytest.param(run_argv_sleeper, id="exec_argv"), pytest.param(run_shell_sleeper, id="exec")],
)
async def test_exec_when_containment_raises_after_spawn_does_fail_the_run_after_reaping_child(
    spawned_by_either_runner: list[list[asyncio.subprocess.Process]],
    options: ExecOptions,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: tuple[str, Callable[[int], bool]],
    expected_fragment: str,
    run: Callable[[ExecOptions], Awaitable[ExecResult | ExecTimeoutError]],
) -> None:
    seam, stand_in = failure

    monkeypatch.setattr(exec_mod, seam, stand_in)

    result = await asyncio.wait_for(run(options), _WAIT_TIMEOUT_S)

    assert isinstance(result, ExecResult)
    assert (result.stdout, result.exit_code) == ("", 1)
    assert expected_fragment in result.stderr
    (child,) = itertools.chain.from_iterable(spawned_by_either_runner)
    assert child.returncode is not None, (
        "the child left behind by the failed spawn was never reaped"
    )
    attempted = record_registry_sweep(monkeypatch)
    exec_mod.kill_live_process_groups()
    assert attempted == []


RELEASE_REFUSED = "release refused"


def _fail_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make dropping the child's container raise, as the win32 release can."""
    monkeypatch.setattr(
        exec_mod,
        "release_process_group",
        _raise_on_call(RuntimeWarning(RELEASE_REFUSED)),
    )


def _fail_group_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every group signal raise once sent, as an escalated killpg warning does."""
    for seam in ("terminate_process_group", "kill_process_group"):
        real: Callable[..., bool] = getattr(exec_mod, seam)

        def signal_then_raise(
            pid: int,
            *args: object,
            _real: Callable[..., bool] = real,
            **kwargs: object,
        ) -> NoReturn:
            _real(pid, *args, **kwargs)
            raise RuntimeWarning(KILLPG_FAILED)

        monkeypatch.setattr(exec_mod, seam, signal_then_raise)


TEARDOWN_FAILURES = pytest.mark.parametrize(
    ("install_teardown_failure", "teardown_fragment"),
    [
        pytest.param(_fail_group_kill, KILLPG_FAILED, id="group-kill-raises"),
        pytest.param(_fail_release, RELEASE_REFUSED, id="release-raises"),
    ],
)


async def _spawn_contained_sleeper() -> asyncio.subprocess.Process:
    """Spawn a 30 s sleep through ``spawn_contained``, with no stdio attached."""
    return await asyncio.wait_for(
        exec_mod.spawn_contained(
            asyncio.create_subprocess_exec,
            *SLEEPER_ARGV,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        ),
        _WAIT_TIMEOUT_S,
    )


@TEARDOWN_FAILURES
@pytest.mark.parametrize("containment_seam", ["attach_process_group", "resume_process_group"])
async def test_spawn_contained_when_containment_and_teardown_both_raise_does_raise_spawn_error_with_both_reasons(
    spawned_by_either_runner: list[list[asyncio.subprocess.Process]],
    monkeypatch: pytest.MonkeyPatch,
    *,
    containment_seam: str,
    install_teardown_failure: Callable[[pytest.MonkeyPatch], None],
    teardown_fragment: str,
) -> None:
    containment_error = RuntimeWarning("containment refused")
    monkeypatch.setattr(exec_mod, containment_seam, _raise_on_call(containment_error))
    install_teardown_failure(monkeypatch)

    with pytest.raises(exec_mod.SpawnError) as caught:
        await _spawn_contained_sleeper()

    assert "containment refused" in str(caught.value)
    assert teardown_fragment in str(caught.value)
    assert caught.value.__cause__ is containment_error
    (child,) = itertools.chain.from_iterable(spawned_by_either_runner)
    assert child.returncode is not None, (
        "the child left behind by the failed spawn was never reaped"
    )
    assert exec_mod._live_process_groups == set()


@TEARDOWN_FAILURES
async def test_spawn_contained_when_resume_refused_and_teardown_raises_does_raise_spawn_error_with_both_reasons(
    spawned_by_either_runner: list[list[asyncio.subprocess.Process]],
    monkeypatch: pytest.MonkeyPatch,
    *,
    install_teardown_failure: Callable[[pytest.MonkeyPatch], None],
    teardown_fragment: str,
) -> None:
    monkeypatch.setattr(exec_mod, "resume_process_group", refuse_resume)
    install_teardown_failure(monkeypatch)

    with pytest.raises(exec_mod.SpawnError) as caught:
        await _spawn_contained_sleeper()

    assert "could not be resumed" in str(caught.value)
    assert teardown_fragment in str(caught.value)
    (child,) = itertools.chain.from_iterable(spawned_by_either_runner)
    assert child.returncode is not None, (
        "the child left behind by the failed spawn was never reaped"
    )
    assert exec_mod._live_process_groups == set()


@TEARDOWN_FAILURES
@pytest.mark.parametrize(
    ("failure", "expected_fragment"),
    [
        pytest.param(
            ("attach_process_group", _raise_on_call(RuntimeWarning("containment refused"))),
            "containment refused",
            id="attach-raises",
        ),
        pytest.param(
            ("resume_process_group", refuse_resume),
            "could not be resumed",
            id="resume-refused",
        ),
    ],
)
@pytest.mark.parametrize(
    "run",
    [pytest.param(run_argv_sleeper, id="exec_argv"), pytest.param(run_shell_sleeper, id="exec")],
)
async def test_exec_when_containment_and_teardown_both_fail_does_resolve_with_containment_reason_on_stderr(
    spawned_by_either_runner: list[list[asyncio.subprocess.Process]],
    options: ExecOptions,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: tuple[str, Callable[[int], bool]],
    expected_fragment: str,
    run: Callable[[ExecOptions], Awaitable[ExecResult | ExecTimeoutError]],
    install_teardown_failure: Callable[[pytest.MonkeyPatch], None],
    teardown_fragment: str,
) -> None:
    seam, stand_in = failure
    monkeypatch.setattr(exec_mod, seam, stand_in)
    install_teardown_failure(monkeypatch)

    result = await asyncio.wait_for(run(options), _WAIT_TIMEOUT_S)

    assert isinstance(result, ExecResult)
    assert (result.stdout, result.exit_code) == ("", 1)
    assert expected_fragment in result.stderr
    assert teardown_fragment in result.stderr
