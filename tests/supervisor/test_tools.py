"""Behavioral tests for the ToolHost handler logic and SDK wiring."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

import pytest

from gymrat.exec import (
    ExecOptions,
    ExecResult,
    ExecTimeoutError,
    _live_process_groups,
    kill_live_process_groups,
)
from gymrat.supervisor.tools import (
    ToolHost,
    create_gymrat_tools,
    gymrat_tool_definitions,
    gymrat_tools_factory,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_result(stdout: str = "", stderr: str = "", exit_code: int = 0) -> ExecResult:
    """Build an ``ExecResult`` with byte counts derived from the strings."""
    return ExecResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
    )


def _text_of(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


def _real_host(
    tmp_path: pathlib.Path,
    abort: asyncio.Event | None = None,
    script: str = "import time; time.sleep(60)",
) -> ToolHost:
    """Build a ToolHost with a real subprocess prefix (no mock)."""
    return ToolHost(
        root=str(tmp_path),
        abort=abort,
        extra_env={},
        argv_prefix=[sys.executable, "-c", script],
    )


def _mock_host(
    tmp_path: pathlib.Path,
    fake_exec: AsyncMock,
    *,
    abort: asyncio.Event | None = None,
    extra_env: dict[str, str] | None = None,
) -> ToolHost:
    """Build a ToolHost wired to *fake_exec* with a single-token argv prefix."""
    return ToolHost(
        root=str(tmp_path),
        abort=abort,
        extra_env=extra_env or {},
        argv_prefix=["gym"],
        _exec_fn=fake_exec,
    )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_exec() -> AsyncMock:
    """An ``exec_argv`` replacement that records calls and returns a canned result."""
    return AsyncMock(return_value=_make_result(stdout='{"ok": true}'))


@pytest.fixture
def host(tmp_path: pathlib.Path, fake_exec: AsyncMock) -> ToolHost:
    """A ToolHost wired to the fake executor, with a trivial prefix."""
    return ToolHost(
        root=str(tmp_path),
        abort=None,
        extra_env={},
        argv_prefix=["gym", "rat"],
        _exec_fn=fake_exec,
    )


# ---------------------------------------------------------------------------
# argv construction and child environment
# ---------------------------------------------------------------------------


async def test_probe_when_names_and_samples_given_does_build_correct_argv(
    host: ToolHost, fake_exec: AsyncMock
) -> None:
    await host.probe({"names": ["a", "b"], "samples": 6})

    argv = fake_exec.call_args[0][0]
    assert list(argv) == [
        "gym",
        "rat",
        "probe",
        "a",
        "b",
        "--samples",
        "6",
        "--format",
        "json",
    ]


async def test_probe_when_empty_input_does_build_minimal_argv(
    host: ToolHost, fake_exec: AsyncMock
) -> None:
    await host.probe({})

    argv = fake_exec.call_args[0][0]
    assert list(argv) == ["gym", "rat", "probe", "--format", "json"]


async def test_iterate_when_called_does_build_correct_argv(
    host: ToolHost, fake_exec: AsyncMock
) -> None:
    await host.iterate({})

    argv = fake_exec.call_args[0][0]
    assert list(argv) == ["gym", "rat", "iterate", "--format", "json"]


async def test_iterate_when_input_has_extras_does_ignore_them(
    host: ToolHost, fake_exec: AsyncMock
) -> None:
    await host.iterate({"something": "ignored"})

    argv = fake_exec.call_args[0][0]
    assert list(argv) == ["gym", "rat", "iterate", "--format", "json"]


async def test_probe_when_called_does_pass_cwd_as_root(
    tmp_path: pathlib.Path, fake_exec: AsyncMock
) -> None:
    host = _mock_host(tmp_path, fake_exec)

    await host.probe({})

    opts: ExecOptions = fake_exec.call_args[0][1]
    assert opts.cwd == str(tmp_path)


async def test_probe_when_called_does_set_env_with_no_color_and_extra_env(
    tmp_path: pathlib.Path, fake_exec: AsyncMock
) -> None:
    extra = {"GYMRAT_TRACEPARENT": "00-abc-def-01"}
    host = _mock_host(tmp_path, fake_exec, extra_env=extra)

    await host.probe({})

    opts: ExecOptions = fake_exec.call_args[0][1]
    assert opts.env is not None
    assert opts.env["NO_COLOR"] == "1"
    assert opts.env["GYMRAT_TRACEPARENT"] == "00-abc-def-01"
    # Supervisor env should be inherited (os.environ merged)
    assert opts.env.get("PATH") == os.environ.get("PATH")


async def test_probe_when_called_does_pass_abort_event(
    tmp_path: pathlib.Path, fake_exec: AsyncMock
) -> None:
    abort = asyncio.Event()
    host = _mock_host(tmp_path, fake_exec, abort=abort)

    await host.probe({})

    opts: ExecOptions = fake_exec.call_args[0][1]
    assert opts.abort is abort


# ---------------------------------------------------------------------------
# result mapping
# ---------------------------------------------------------------------------


async def test_probe_when_exit_0_with_json_stdout_does_return_content(
    tmp_path: pathlib.Path,
) -> None:
    host = _real_host(tmp_path, script="import sys, json; json.dump({'ok': True}, sys.stdout)")

    result = await host.probe({})

    assert result["is_error"] is False
    parsed = json.loads(_text_of(result))
    assert parsed == {"ok": True}


async def test_probe_when_exit_1_with_json_stdout_does_return_document(
    tmp_path: pathlib.Path,
) -> None:
    host = _real_host(
        tmp_path,
        script="import sys, json; json.dump({'gate': 'stop'}, sys.stdout); sys.exit(1)",
    )

    result = await host.probe({})

    assert result["is_error"] is False
    parsed = json.loads(_text_of(result))
    assert parsed == {"gate": "stop"}


async def test_probe_when_exit_2_with_stderr_does_return_error_with_stderr(
    tmp_path: pathlib.Path,
) -> None:
    host = _real_host(tmp_path, script="import sys; sys.stderr.write('bad config'); sys.exit(2)")

    result = await host.probe({})

    assert result == {"content": [{"type": "text", "text": "bad config"}], "is_error": True}


async def test_probe_when_exit_2_with_no_output_does_return_fallback_message(
    tmp_path: pathlib.Path,
) -> None:
    host = _real_host(tmp_path, script="import sys; sys.exit(2)")

    result = await host.probe({})

    assert result["is_error"] is True
    assert "gymrat probe exited 2 with no output" in _text_of(result)


async def test_iterate_when_exit_2_with_no_output_does_name_iterate_in_message(
    tmp_path: pathlib.Path,
) -> None:
    host = _real_host(tmp_path, script="import sys; sys.exit(2)")

    result = await host.iterate({})

    assert result["is_error"] is True
    assert "gymrat iterate exited 2 with no output" in _text_of(result)


async def test_probe_when_abort_set_and_no_json_does_return_killed_message(
    tmp_path: pathlib.Path,
) -> None:
    abort = asyncio.Event()
    abort.set()
    host = _real_host(tmp_path, abort=abort, script="import sys; sys.exit(1)")

    result = await host.probe({})

    assert result["is_error"] is True
    assert _text_of(result) == "killed by the supervisor"


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        pytest.param(
            "import sys; sys.stderr.write('bench crashed\\n'); sys.exit(1)",
            "bench crashed",
            id="stderr-exit-1",
        ),
        pytest.param(
            "import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
            "gymrat probe failed",
            id="signal-kill",
        ),
    ],
)
async def test_probe_when_child_fails_without_document_does_return_failure_text(
    tmp_path: pathlib.Path,
    script: str,
    expected: str,
) -> None:
    host = _real_host(tmp_path, script=script)

    result = await host.probe({})

    assert result == {"content": [{"type": "text", "text": expected}], "is_error": True}


async def test_probe_when_spawn_fails_does_return_spawn_error(tmp_path: pathlib.Path) -> None:
    missing = tmp_path / "missing-executable"
    host = ToolHost(root=str(tmp_path), abort=None, extra_env={}, argv_prefix=[str(missing)])

    result = await host.probe({})

    assert result == {
        "content": [{"type": "text", "text": f"[Errno 2] No such file or directory: '{missing}'"}],
        "is_error": True,
    }


@pytest.mark.parametrize("exit_code", [0, 1])
async def test_probe_when_json_object_padded_with_whitespace_does_return_document(
    tmp_path: pathlib.Path,
    fake_exec: AsyncMock,
    exit_code: int,
) -> None:
    stdout = '\n  {"ok": true}\n'
    fake_exec.return_value = _make_result(stdout=stdout, exit_code=exit_code)
    host = _mock_host(tmp_path, fake_exec)

    result = await host.probe({})

    assert result == {"content": [{"type": "text", "text": stdout}], "is_error": False}


@pytest.mark.parametrize(
    ("stdout", "exit_code"),
    [
        pytest.param('{"kind": "probe", ', 0, id="truncated-object-exit-0"),
        pytest.param('{"ok": true}', 3, id="object-exit-3"),
        pytest.param("[1, 2]", 0, id="array-exit-0"),
    ],
)
async def test_probe_when_stdout_is_not_a_document_does_return_stderr_error(
    tmp_path: pathlib.Path,
    fake_exec: AsyncMock,
    stdout: str,
    exit_code: int,
) -> None:
    fake_exec.return_value = _make_result(stdout=stdout, stderr="broken run", exit_code=exit_code)
    host = _mock_host(tmp_path, fake_exec)

    result = await host.probe({})

    assert result["is_error"] is True
    assert _text_of(result) == "broken run"


async def test_probe_when_invalid_json_and_no_stderr_does_return_stdout_error(
    tmp_path: pathlib.Path,
    fake_exec: AsyncMock,
) -> None:
    fake_exec.return_value = _make_result(stdout='{"kind": "probe", ', exit_code=0)
    host = _mock_host(tmp_path, fake_exec)

    result = await host.probe({})

    assert result["is_error"] is True
    assert _text_of(result) == '{"kind": "probe",'


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        pytest.param("bench ran past its deadline\n", "bench ran past its deadline", id="stderr"),
        pytest.param("", "gymrat probe timed out", id="no-output"),
    ],
)
async def test_probe_when_run_times_out_does_return_timeout_error(
    tmp_path: pathlib.Path,
    fake_exec: AsyncMock,
    stderr: str,
    expected: str,
) -> None:
    fake_exec.return_value = ExecTimeoutError(
        stdout='{"kind": "probe"}',
        stderr=stderr,
        timeout_ms=5,
        stdout_bytes=17,
        stderr_bytes=len(stderr.encode()),
    )
    host = _mock_host(tmp_path, fake_exec)

    result = await host.probe({})

    assert result == {"content": [{"type": "text", "text": expected}], "is_error": True}


async def test_iterate_when_no_output_and_no_abort_does_return_failed_message(
    tmp_path: pathlib.Path,
    fake_exec: AsyncMock,
) -> None:
    fake_exec.return_value = _make_result(exit_code=3)
    host = _mock_host(tmp_path, fake_exec)

    result = await host.iterate({})

    assert result["is_error"] is True
    assert _text_of(result) == "gymrat iterate failed"


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "names",
    [
        pytest.param("not-a-list", id="string"),
        pytest.param(123, id="integer"),
        pytest.param(["ok", 42], id="list-with-non-string"),
    ],
)
async def test_probe_when_names_invalid_does_return_error_without_spawn(
    host: ToolHost,
    fake_exec: AsyncMock,
    names: Any,
) -> None:
    result = await host.probe({"names": names})

    assert result["is_error"] is True
    assert "names" in _text_of(result)
    fake_exec.assert_not_called()


@pytest.mark.parametrize(
    "samples",
    [
        pytest.param("five", id="string"),
        pytest.param(-1, id="negative"),
        pytest.param(0, id="zero"),
        pytest.param(3.5, id="float"),
    ],
)
async def test_probe_when_samples_invalid_does_return_error_without_spawn(
    host: ToolHost,
    fake_exec: AsyncMock,
    samples: Any,
) -> None:
    result = await host.probe({"samples": samples})

    assert result["is_error"] is True
    assert "samples" in _text_of(result)
    fake_exec.assert_not_called()


# ---------------------------------------------------------------------------
# helpers: concurrency and gate-file child scripts
# ---------------------------------------------------------------------------


def _blocking_script(ready_file: str) -> str:
    """Return a Python script that signals readiness then sleeps."""
    return f"import pathlib, time; pathlib.Path({ready_file!r}).touch(); time.sleep(60)"


async def _wait_for_gate(gate: pathlib.Path, *, timeout: float = 5.0) -> None:  # noqa: ASYNC109 -- gate-file poll with cooperative yields, not a deadline
    """Poll until a gate file exists, raising on timeout."""
    elapsed = 0.0
    step = 0.05
    while not gate.exists():  # noqa: ASYNC240 -- brief sync check between async yields
        await asyncio.sleep(step)
        elapsed += step
        if elapsed >= timeout:
            msg = f"gate file {gate} not created within {timeout}s"
            raise TimeoutError(msg)


def _pid_is_dead(pid: int) -> bool:
    """Return True when *pid* no longer exists."""
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.ESRCH
    return False


def _assert_busy(result: dict[str, Any]) -> None:
    """Assert *result* is the error returned when the host is already running a call."""
    assert result["is_error"] is True
    assert "already running" in _text_of(result)


async def _cancel_and_await(task: asyncio.Task[dict[str, Any]]) -> None:
    """Cancel *task* and await it, swallowing the resulting CancelledError."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


async def test_probe_when_child_already_running_does_return_busy_error(
    tmp_path: pathlib.Path,
) -> None:
    ready = tmp_path / "ready"
    script = _blocking_script(str(ready))
    host = _real_host(tmp_path, script=script)

    first = asyncio.create_task(host.probe({}))
    await _wait_for_gate(ready)
    second_result = await host.probe({})

    _assert_busy(second_result)
    await _cancel_and_await(first)


async def test_iterate_when_child_already_running_does_return_busy_error(
    tmp_path: pathlib.Path,
) -> None:
    ready = tmp_path / "ready"
    script = _blocking_script(str(ready))
    host = _real_host(tmp_path, script=script)

    first = asyncio.create_task(host.iterate({}))
    await _wait_for_gate(ready)
    second_result = await host.iterate({})

    _assert_busy(second_result)
    await _cancel_and_await(first)


async def test_probe_when_previous_call_settled_does_run_normally(
    tmp_path: pathlib.Path,
) -> None:
    script = "import sys, json; json.dump({'n': 1}, sys.stdout)"
    host = _real_host(tmp_path, script=script)

    first = await host.probe({})
    second = await host.probe({})

    assert first["is_error"] is False
    assert second["is_error"] is False


async def test_probe_when_validation_fails_does_not_leave_host_busy(
    tmp_path: pathlib.Path,
) -> None:
    script = "import sys, json; json.dump({'n': 1}, sys.stdout)"
    host = _real_host(tmp_path, script=script)

    bad = await host.probe({"names": 123})
    assert bad["is_error"] is True

    good = await host.probe({})
    assert good["is_error"] is False


async def test_probe_when_concurrent_refused_does_not_leave_host_busy(
    host: ToolHost,
    fake_exec: AsyncMock,
) -> None:
    blocker: asyncio.Future[ExecResult] = asyncio.get_event_loop().create_future()
    call_count = 0

    async def counting_exec(*args: Any, **_kwargs: Any) -> ExecResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return await blocker
        return _make_result(stdout='{"ok": true}')

    fake_exec.side_effect = counting_exec

    first = asyncio.create_task(host.probe({}))
    await asyncio.sleep(0)
    refused = await host.iterate({})
    _assert_busy(refused)

    blocker.set_result(_make_result(stdout='{"ok": true}'))
    await first

    fake_exec.side_effect = None
    after = await host.probe({})
    assert after["is_error"] is False


# ---------------------------------------------------------------------------
# kill paths
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group semantics")
async def test_probe_when_abort_fires_mid_run_does_kill_child_and_return_killed(
    tmp_path: pathlib.Path,
) -> None:
    ready = tmp_path / "ready"
    abort = asyncio.Event()
    script = _blocking_script(str(ready))
    host = _real_host(tmp_path, abort=abort, script=script)

    task = asyncio.create_task(host.probe({}))
    await _wait_for_gate(ready)

    live_groups = set(_live_process_groups)
    assert len(live_groups) >= 1
    child_pid = next(iter(live_groups))

    abort.set()

    result = await task
    assert result["is_error"] is True
    assert _text_of(result) == "killed by the supervisor"
    assert child_pid not in _live_process_groups
    assert _pid_is_dead(child_pid)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group semantics")
async def test_probe_when_abort_fires_after_partial_json_does_return_killed(
    tmp_path: pathlib.Path,
) -> None:
    ready = tmp_path / "ready"
    abort = asyncio.Event()
    script = (
        "import pathlib, sys, time; "
        'sys.stdout.write(\'{"kind": "probe", \'); sys.stdout.flush(); '
        f"pathlib.Path({str(ready)!r}).touch(); time.sleep(60)"
    )
    host = _real_host(tmp_path, abort=abort, script=script)
    task = asyncio.create_task(host.probe({}))
    await _wait_for_gate(ready)

    abort.set()
    result = await task

    assert result["is_error"] is True
    assert _text_of(result) == "killed by the supervisor"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group semantics")
async def test_probe_when_task_cancelled_mid_run_does_kill_and_reap_child(
    tmp_path: pathlib.Path,
) -> None:
    ready = tmp_path / "ready"
    script = _blocking_script(str(ready))
    host = _real_host(tmp_path, script=script)

    task = asyncio.create_task(host.probe({}))
    await _wait_for_gate(ready)

    live_groups = set(_live_process_groups)
    assert len(live_groups) >= 1
    child_pid = next(iter(live_groups))

    await _cancel_and_await(task)

    assert child_pid not in _live_process_groups
    assert _pid_is_dead(child_pid)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group semantics")
async def test_probe_when_kill_live_process_groups_mid_run_does_kill_child(
    tmp_path: pathlib.Path,
) -> None:
    ready = tmp_path / "ready"
    script = _blocking_script(str(ready))
    host = _real_host(tmp_path, script=script)

    task = asyncio.create_task(host.probe({}))
    await _wait_for_gate(ready)

    live_groups = set(_live_process_groups)
    assert len(live_groups) >= 1
    child_pid = next(iter(live_groups))

    kill_live_process_groups()
    await task

    assert child_pid not in _live_process_groups
    assert _pid_is_dead(child_pid)


# ---------------------------------------------------------------------------
# pipe pressure
# ---------------------------------------------------------------------------


async def test_probe_when_child_writes_large_stderr_and_json_stdout_does_return_document(
    tmp_path: pathlib.Path,
) -> None:
    script = (
        "import sys, json; "
        "sys.stderr.write('x' * (1024 * 1024 + 1)); "
        "json.dump({'ok': True}, sys.stdout)"
    )
    host = _real_host(tmp_path, script=script)

    result = await host.probe({})

    assert result["is_error"] is False
    parsed = json.loads(_text_of(result))
    assert parsed == {"ok": True}


# ---------------------------------------------------------------------------
# SDK wiring
# ---------------------------------------------------------------------------


async def test_create_gymrat_tools_when_called_does_return_sdk_config_named_gymrat(
    host: ToolHost,
) -> None:
    config: dict[str, Any] = create_gymrat_tools(host)  # type: ignore[assignment]  # McpSdkServerConfig is a TypedDict

    assert config["type"] == "sdk"
    assert config["name"] == "gymrat"


async def test_gymrat_tool_definitions_when_called_does_return_probe_and_iterate(
    host: ToolHost,
) -> None:
    defs = gymrat_tool_definitions(host)

    names = [d.name for d in defs]
    assert names == ["probe", "iterate"]


async def test_gymrat_tool_definitions_when_called_does_set_probe_description_and_schema(
    host: ToolHost,
) -> None:
    defs = gymrat_tool_definitions(host)
    probe_def = defs[0]

    assert probe_def.description == (
        "Bench the experiment worktree and report each metric's delta against the current "
        "baseline. Pass metric names to scope the run. Records nothing."
    )
    schema: dict[str, Any] = probe_def.input_schema  # type: ignore[assignment]  # SdkMcpTool generic widens the type
    assert schema["type"] == "object"
    assert schema["properties"] == {
        "names": {"type": "array", "items": {"type": "string"}},
        "samples": {"type": "integer"},
    }
    assert schema.get("required", []) == []
    assert schema["additionalProperties"] is False


async def test_gymrat_tool_definitions_when_called_does_set_iterate_description_and_schema(
    host: ToolHost,
) -> None:
    defs = gymrat_tool_definitions(host)
    iterate_def = defs[1]

    assert iterate_def.description == (
        "Measure the experiment worktree against the baseline at the configured samples and "
        "record the verdict. The only way to measure before keep."
    )
    schema: dict[str, Any] = iterate_def.input_schema  # type: ignore[assignment]  # SdkMcpTool generic widens the type
    assert schema["type"] == "object"
    assert schema.get("properties", {}) == {}
    assert schema.get("required", []) == []
    assert schema["additionalProperties"] is False


async def test_gymrat_tool_definitions_when_probe_handler_called_does_invoke_host_probe(
    host: ToolHost,
    fake_exec: AsyncMock,
) -> None:
    defs = gymrat_tool_definitions(host)
    probe_def = defs[0]

    await probe_def.handler({"names": ["x"], "samples": 3})

    argv = fake_exec.call_args[0][0]
    assert "probe" in argv


async def test_gymrat_tool_definitions_when_iterate_handler_called_does_invoke_host_iterate(
    host: ToolHost,
    fake_exec: AsyncMock,
) -> None:
    defs = gymrat_tool_definitions(host)
    iterate_def = defs[1]

    await iterate_def.handler({})

    argv = fake_exec.call_args[0][0]
    assert "iterate" in argv


async def test_gymrat_tools_factory_when_called_does_build_host_from_root_abort_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, Any]] = []
    real_host_class = ToolHost

    def _recording_host(**kwargs: Any) -> ToolHost:
        built.append(kwargs)
        return real_host_class(**kwargs)

    monkeypatch.setattr("gymrat.supervisor.tools.ToolHost", _recording_host)
    factory = gymrat_tools_factory("/tmp/fake-root")
    abort = asyncio.Event()
    env = {"GYMRAT_TRACEPARENT": "00-abc-def-01"}

    config: dict[str, Any] = factory(abort, env)  # type: ignore[assignment]  # McpSdkServerConfig is a TypedDict

    assert (config["type"], config["name"]) == ("sdk", "gymrat")
    assert built == [{"root": "/tmp/fake-root", "abort": abort, "extra_env": env}]


# ---------------------------------------------------------------------------
# real CLI integration
# ---------------------------------------------------------------------------

pytestmark_posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only worktrees")


def _run_gymrat_cli(args: list[str], cwd: str) -> None:
    """Run a gymrat CLI command in *cwd*, blocking until done."""
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "gymrat", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytestmark_posix
async def test_probe_when_real_cli_on_scratch_repo_does_return_json_document(
    create_scratch_repo: Callable[[], str],
) -> None:
    from tests.loop._bench import commit_project

    repo = create_scratch_repo()
    commit_project(repo, samples=5)
    _run_gymrat_cli(["start", "--baseline", "main"], repo)
    _run_gymrat_cli(["measure", "--record"], repo)
    host = ToolHost(root=repo, abort=None, extra_env={})

    result = await host.probe({})

    assert result["is_error"] is False, f"probe error: {_text_of(result)}"
    doc = json.loads(_text_of(result))
    assert isinstance(doc, dict)


@pytestmark_posix
async def test_iterate_when_real_cli_on_scratch_repo_does_return_json_document(
    create_scratch_repo: Callable[[], str],
) -> None:
    from tests.loop._bench import commit_project

    repo = create_scratch_repo()
    commit_project(repo, samples=5)
    _run_gymrat_cli(["start", "--baseline", "main"], repo)
    host = ToolHost(root=repo, abort=None, extra_env={})

    result = await host.iterate({})

    assert result["is_error"] is False
    doc = json.loads(_text_of(result))
    assert isinstance(doc, dict)
