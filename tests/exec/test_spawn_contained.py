"""Behavioral tests for containing a freshly spawned child, through ``spawn_contained`` and both exec forms.

A child that exists but could not be registered, contained, or resumed is torn
down before the spawn fails, and the failure always surfaces as a
:class:`~gymrat.exec.SpawnError` — or, through ``exec`` and ``exec_argv``, as a
failed :class:`~gymrat.exec.ExecResult` — even when the teardown fails too.
A child that is contained leads its own session and runs one nesting level
deeper, in the environment the caller asked for.

Real-subprocess tests are POSIX-only: process groups and ``os.killpg`` do not
exist on win32.
"""

import asyncio
import contextlib
import errno
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from typing import NoReturn

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import ExecOptions, ExecResult, exec_argv
from tests._exec_fixtures import RUNNERS, Runner, expected_result
from tests._process_helpers import (
    KILLPG_FAILED,
    SLEEPER_ARGV,
    refuse_resume,
)

# Upper bound each awaited run or spawn gets before the test fails outright.
_WAIT_TIMEOUT_S = 10

CONTAINMENT_REFUSED = "containment refused"


def _raise_on_call(error: Exception) -> Callable[[int], NoReturn]:
    """Build a stand-in that raises ``error`` when called with a pid."""

    def fail(_pid: int) -> NoReturn:
        raise error

    return fail


@pytest.mark.parametrize(
    ("failure", "expected_message"),
    [
        pytest.param(
            ("attach_process_group", _raise_on_call(RuntimeWarning(CONTAINMENT_REFUSED))),
            CONTAINMENT_REFUSED,
            id="attach-warning-escalated-to-error",
        ),
        pytest.param(
            (
                "resume_process_group",
                _raise_on_call(OSError(errno.EPERM, CONTAINMENT_REFUSED)),
            ),
            "[Errno 1] containment refused",
            id="resume-os-error",
        ),
        pytest.param(
            ("resume_process_group", refuse_resume),
            "child process {pid} could not be resumed",
            id="resume-refused",
        ),
    ],
)
@pytest.mark.parametrize("runner", RUNNERS)
async def test_exec_when_containment_raises_after_spawn_does_fail_the_run_after_reaping_child(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: tuple[str, Callable[[int], bool]],
    expected_message: str,
    runner: Runner,
) -> None:
    seam, stand_in = failure
    monkeypatch.setattr(exec_mod, seam, stand_in)

    result = await asyncio.wait_for(runner.run(runner.sleeper, make_opts()), _WAIT_TIMEOUT_S)

    (child,) = spawned_processes
    stderr = expected_message.format(pid=child.pid) + "\n"
    assert result == expected_result("", stderr, exit_code=1)
    assert child.returncode is not None, (
        "the child left behind by the failed spawn was never reaped"
    )


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
@pytest.mark.parametrize(
    ("seam", "stand_in", "reason", "cause_from_teardown"),
    [
        pytest.param(
            "attach_process_group",
            _raise_on_call(RuntimeWarning(CONTAINMENT_REFUSED)),
            CONTAINMENT_REFUSED,
            False,
            id="attach-raises",
        ),
        pytest.param(
            "resume_process_group",
            _raise_on_call(RuntimeWarning(CONTAINMENT_REFUSED)),
            CONTAINMENT_REFUSED,
            False,
            id="resume-raises",
        ),
        pytest.param(
            "resume_process_group",
            refuse_resume,
            "child process {pid} could not be resumed",
            True,
            id="resume-refused",
        ),
    ],
)
async def test_spawn_contained_when_containment_and_teardown_both_fail_does_raise_spawn_error_with_both_reasons(
    spawned_processes: list[asyncio.subprocess.Process],
    monkeypatch: pytest.MonkeyPatch,
    *,
    seam: str,
    stand_in: Callable[[int], bool],
    reason: str,
    cause_from_teardown: bool,
    install_teardown_failure: Callable[[pytest.MonkeyPatch], None],
    teardown_fragment: str,
) -> None:
    monkeypatch.setattr(exec_mod, seam, stand_in)
    install_teardown_failure(monkeypatch)

    with pytest.raises(exec_mod.SpawnError) as caught:
        await _spawn_contained_sleeper()

    (child,) = spawned_processes
    expected_reason = reason.format(pid=child.pid)
    assert str(caught.value) == (
        f"{expected_reason} (tearing the child down also failed: {teardown_fragment})"
    )
    cause = caught.value.__cause__
    assert (type(cause), str(cause)) == (
        RuntimeWarning,
        teardown_fragment if cause_from_teardown else CONTAINMENT_REFUSED,
    )
    assert child.returncode is not None, (
        "the child left behind by the failed spawn was never reaped"
    )


def _teardown_succeeds(_monkeypatch: pytest.MonkeyPatch) -> None:
    pass


@pytest.mark.parametrize(
    "install_teardown_failure",
    [
        pytest.param(_teardown_succeeds, id="teardown-succeeds"),
        pytest.param(_fail_group_kill, id="group-kill-raises"),
        pytest.param(_fail_release, id="release-raises"),
    ],
)
@pytest.mark.usefixtures("spawned_processes")
async def test_kill_live_process_groups_when_spawn_contained_failed_does_not_target_the_child(
    monkeypatch: pytest.MonkeyPatch,
    record_killpg: Callable[[], list[int]],
    install_teardown_failure: Callable[[pytest.MonkeyPatch], None],
) -> None:
    monkeypatch.setattr(exec_mod, "resume_process_group", refuse_resume)
    install_teardown_failure(monkeypatch)
    with contextlib.suppress(exec_mod.SpawnError):
        await _spawn_contained_sleeper()
    attempted = record_killpg()

    exec_mod.kill_live_process_groups()

    assert attempted == []


@pytest.fixture
def sigterm_ignored() -> None:
    """Ignore SIGTERM in this process, so a child spawned meanwhile inherits ignoring it."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


@pytest.mark.usefixtures("sigterm_ignored")
async def test_spawn_contained_when_group_kill_raises_on_a_child_ignoring_the_request_does_kill_it(
    spawned_processes: list[asyncio.subprocess.Process],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exec_mod, "resume_process_group", refuse_resume)
    _fail_group_kill(monkeypatch)

    with pytest.raises(exec_mod.SpawnError):
        await _spawn_contained_sleeper()

    (child,) = spawned_processes
    assert child.returncode == -signal.SIGKILL


# ---------------------------------------------------------------------------
# env option
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "expected_marker"),
    [
        pytest.param(None, "inherited", id="inherited-env"),
        pytest.param({"GYMRAT_TEST_MARKER": "mapped"}, "mapped", id="explicit-env"),
        pytest.param({}, None, id="empty-env"),
    ],
)
async def test_exec_argv_when_env_given_or_omitted_does_hand_child_that_env_one_nesting_level_deeper(
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str] | None,
    expected_marker: str | None,
) -> None:
    monkeypatch.setattr(exec_mod, "NESTING_DEPTH", 2)
    monkeypatch.setenv("GYMRAT_TEST_MARKER", "inherited")

    result = await exec_argv(
        [sys.executable, "-c", "import os, json; print(json.dumps(dict(os.environ)))"],
        make_opts(env=env),
    )

    assert isinstance(result, ExecResult)
    child_env = json.loads(result.stdout.strip())
    assert {
        "GYMRAT_TEST_MARKER": child_env.get("GYMRAT_TEST_MARKER"),
        "GYMRAT_NESTING_DEPTH": child_env.get("GYMRAT_NESTING_DEPTH"),
    } == {"GYMRAT_TEST_MARKER": expected_marker, "GYMRAT_NESTING_DEPTH": "3"}


# ---------------------------------------------------------------------------
# nesting depth
# ---------------------------------------------------------------------------

_PRINT_NESTING_DEPTH = "from gymrat.exec import NESTING_DEPTH; print(NESTING_DEPTH)"


@pytest.mark.parametrize(
    ("depth_env", "expected_depth"),
    [
        pytest.param({}, 0, id="missing"),
        pytest.param({"GYMRAT_NESTING_DEPTH": "banana"}, 0, id="not-an-integer"),
        pytest.param({"GYMRAT_NESTING_DEPTH": "-2"}, 0, id="negative"),
        pytest.param({"GYMRAT_NESTING_DEPTH": "2"}, 2, id="valid"),
    ],
)
def test_nesting_depth_when_read_from_environment_does_fall_back_to_zero_unless_a_valid_depth(
    depth_env: dict[str, str],
    expected_depth: int,
) -> None:
    inherited = {
        name: value for name, value in os.environ.items() if name != "GYMRAT_NESTING_DEPTH"
    }
    env = {**inherited, **depth_env}

    probe = subprocess.run(  # noqa: S603 -- fixed interpreter; the script is test-written source
        [sys.executable, "-c", _PRINT_NESTING_DEPTH],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    assert probe.stdout == f"{expected_depth}\n"


# ---------------------------------------------------------------------------
# child session (POSIX)
# ---------------------------------------------------------------------------


async def test_exec_argv_when_child_spawned_does_make_it_lead_its_own_session(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await exec_argv(
        [sys.executable, "-c", "import os; print(os.getsid(0) == os.getpid())"],
        make_opts(),
    )

    assert isinstance(result, ExecResult)
    assert result.stdout == "True\n"
