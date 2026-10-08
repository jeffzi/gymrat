"""Behavioral tests for the ToolHost handler logic and SDK wiring."""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, create_autospec

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.types import CallToolResult

import pytest
from mcp import Client
from mcp.types import TextContent

from gymrat.exec import (
    ExecOptions,
    ExecResult,
    ExecTimeoutError,
    exec_argv,
)
from gymrat.supervisor.tools import ToolHost, gymrat_tool_definitions, gymrat_tools_factory
from tests._cli import run_cli
from tests._exec_fixtures import expected_result
from tests.loop._bench import FILTER_TEMPLATE, commit_project, config_text

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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
        abort=abort or asyncio.Event(),
        extra_env={},
        argv_prefix=[sys.executable, "-c", script],
    )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_exec() -> AsyncMock:
    """An ``exec_argv`` replacement that records calls and returns a canned result."""
    return create_autospec(exec_argv, return_value=expected_result(stdout='{"ok": true}'))


@pytest.fixture
def host(request: pytest.FixtureRequest, tmp_path: pathlib.Path, fake_exec: AsyncMock) -> ToolHost:
    """A ToolHost wired to the fake executor, with a trivial prefix and any indirect extra env."""
    return ToolHost(
        root=str(tmp_path),
        abort=asyncio.Event(),
        extra_env=getattr(request, "param", {}),
        argv_prefix=["gym", "rat"],
        _exec_fn=fake_exec,
    )


# ---------------------------------------------------------------------------
# argv construction and child environment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", [{"GYMRAT_TRACEPARENT": "00-abc-def-01"}], indirect=True)
async def test_probe_when_called_does_run_in_the_root_with_tool_origin_no_color_and_extra_env(
    tmp_path: pathlib.Path, host: ToolHost, fake_exec: AsyncMock
) -> None:
    await host.probe({})

    opts: ExecOptions = fake_exec.call_args[0][1]
    assert opts.cwd == str(tmp_path)
    assert opts.env is not None
    assert opts.env["GYMRAT_COMMAND_ORIGIN"] == "tool"
    assert opts.env["NO_COLOR"] == "1"
    assert opts.env["GYMRAT_TRACEPARENT"] == "00-abc-def-01"
    assert opts.env.get("PATH") == os.environ.get("PATH")


@pytest.mark.parametrize("host", [{"NO_COLOR": "0"}], indirect=True)
async def test_probe_when_extra_env_names_a_fixed_variable_does_let_extra_env_win(
    host: ToolHost, fake_exec: AsyncMock
) -> None:
    await host.probe({})

    opts: ExecOptions = fake_exec.call_args[0][1]
    assert opts.env is not None
    assert opts.env["NO_COLOR"] == "0"


# ---------------------------------------------------------------------------
# result mapping
# ---------------------------------------------------------------------------


async def test_probe_when_exit_2_with_stderr_does_return_error_with_stderr(
    tmp_path: pathlib.Path,
) -> None:
    host = _real_host(tmp_path, script="import sys; sys.stderr.write('bad config'); sys.exit(2)")

    result = await host.probe({})

    assert result == {"content": [{"type": "text", "text": "bad config"}], "is_error": True}


@pytest.mark.parametrize("command", ["probe", "iterate"])
async def test_command_when_exit_2_with_no_output_does_return_fallback_naming_it(
    tmp_path: pathlib.Path, command: str
) -> None:
    host = _real_host(tmp_path, script="import sys; sys.exit(2)")

    result = await getattr(host, command)({})

    assert result == {
        "content": [{"type": "text", "text": f"gymrat {command} exited 2 with no output"}],
        "is_error": True,
    }


async def test_probe_when_abort_already_set_does_return_killed_without_running_child(
    tmp_path: pathlib.Path,
) -> None:
    marker = tmp_path / "child-ran"
    abort = asyncio.Event()
    abort.set()
    host = _real_host(
        tmp_path,
        abort=abort,
        script=f"import pathlib; pathlib.Path({str(marker)!r}).touch()",
    )

    result = await host.probe({})

    assert result["is_error"] is True
    assert _text_of(result) == "killed by the supervisor"
    assert not marker.exists()


@pytest.mark.parametrize(
    ("command", "script", "expected"),
    [
        pytest.param(
            "probe",
            "import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
            "gymrat probe failed",
            id="probe-signal-kill",
            marks=pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is POSIX-only"),
        ),
        pytest.param(
            "iterate", "import sys; sys.exit(3)", "gymrat iterate failed", id="iterate-no-output"
        ),
    ],
)
async def test_command_when_child_fails_without_document_does_return_failure_text(
    tmp_path: pathlib.Path,
    command: str,
    script: str,
    expected: str,
) -> None:
    host = _real_host(tmp_path, script=script)

    result = await getattr(host, command)({})

    assert result == {"content": [{"type": "text", "text": expected}], "is_error": True}


@pytest.fixture
async def missing_executable(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    """A path no executable sits at, and the text the host's spawn error gives for it."""
    missing = tmp_path / "missing-executable"
    try:
        await asyncio.create_subprocess_exec(str(missing))
    except FileNotFoundError as error:
        return missing, str(error)
    pytest.fail(f"spawning {missing} unexpectedly succeeded")


async def test_probe_when_spawn_fails_does_return_spawn_error(
    tmp_path: pathlib.Path, missing_executable: tuple[pathlib.Path, str]
) -> None:
    missing, spawn_error = missing_executable
    host = ToolHost(
        root=str(tmp_path), abort=asyncio.Event(), extra_env={}, argv_prefix=[str(missing)]
    )

    result = await host.probe({})

    assert result == {"content": [{"type": "text", "text": spawn_error}], "is_error": True}


@pytest.mark.parametrize("exit_code", [0, 1])
async def test_probe_when_json_object_padded_with_whitespace_does_return_document(
    host: ToolHost,
    fake_exec: AsyncMock,
    exit_code: int,
) -> None:
    stdout = '\n  {"ok": true}\n'
    fake_exec.return_value = expected_result(stdout=stdout, exit_code=exit_code)

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
    host: ToolHost,
    fake_exec: AsyncMock,
    stdout: str,
    exit_code: int,
) -> None:
    fake_exec.return_value = expected_result(
        stdout=stdout, stderr="broken run", exit_code=exit_code
    )

    result = await host.probe({})

    assert result["is_error"] is True
    assert _text_of(result) == "broken run"


async def test_probe_when_invalid_json_and_no_stderr_does_return_stdout_error(
    host: ToolHost,
    fake_exec: AsyncMock,
) -> None:
    fake_exec.return_value = expected_result(stdout='{"kind": "probe", ', exit_code=0)

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
    host: ToolHost,
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

    result = await host.probe({})

    assert result == {"content": [{"type": "text", "text": expected}], "is_error": True}


# ---------------------------------------------------------------------------
# helpers: concurrency and gate-file child scripts
# ---------------------------------------------------------------------------


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


def _assert_busy(result: dict[str, Any]) -> None:
    """Assert *result* is the error returned when the host is already running a call."""
    assert result["is_error"] is True
    assert "already running" in _text_of(result)


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def _block_first_call(fake_exec: AsyncMock) -> asyncio.Future[ExecResult]:
    """Make the first call to *fake_exec* wait on the returned future; later calls succeed."""
    blocker: asyncio.Future[ExecResult] = asyncio.get_event_loop().create_future()
    call_count = 0

    async def counting_exec(*_args: Any, **_kwargs: Any) -> ExecResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return await blocker
        return expected_result(stdout='{"ok": true}')

    fake_exec.side_effect = counting_exec
    return blocker


async def test_iterate_when_a_call_is_running_does_refuse_the_concurrent_one(
    host: ToolHost,
    fake_exec: AsyncMock,
) -> None:
    blocker = _block_first_call(fake_exec)
    first = asyncio.create_task(host.probe({}))
    await asyncio.sleep(0)

    refused = await host.iterate({})
    blocker.set_result(expected_result(stdout='{"ok": true}'))
    await first

    _assert_busy(refused)


async def test_probe_when_concurrent_refused_does_not_leave_host_busy(
    host: ToolHost,
    fake_exec: AsyncMock,
) -> None:
    blocker = _block_first_call(fake_exec)
    first = asyncio.create_task(host.probe({}))
    await asyncio.sleep(0)
    await host.iterate({})
    blocker.set_result(expected_result(stdout='{"ok": true}'))
    await first

    after = await host.probe({})

    assert after["is_error"] is False


# ---------------------------------------------------------------------------
# kill paths
# ---------------------------------------------------------------------------


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


async def test_gymrat_tool_definitions_when_called_does_describe_probe_then_iterate(
    host: ToolHost,
) -> None:
    defs = gymrat_tool_definitions(host)

    assert [(d.name, d.description) for d in defs] == [
        (
            "probe",
            (
                "Bench the experiment worktree and report each metric's delta against the current "
                "baseline. Pass metric names to scope the run. Records nothing."
            ),
        ),
        (
            "iterate",
            (
                "Measure the experiment worktree against the baseline at the configured samples "
                "and record the verdict. The only way to measure before keep."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# SDK server input validation (through an in-memory MCP client)
# ---------------------------------------------------------------------------


@pytest.fixture
def sdk_config(host: ToolHost, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The SDK server config the factory builds, with its tool host replaced by ``host``."""
    monkeypatch.setattr(
        "gymrat.supervisor.tools.ToolHost", create_autospec(ToolHost, return_value=host)
    )
    return gymrat_tools_factory("unused-root")(asyncio.Event(), {})  # type: ignore[return-value]  # McpSdkServerConfig is a TypedDict


async def _call_via_sdk(
    sdk_config: dict[str, Any], tool_name: str, arguments: dict[str, Any]
) -> CallToolResult:
    """Call *tool_name* through an in-memory MCP client on the SDK server."""
    async with Client(sdk_config["instance"]) as client:
        return await client.call_tool(tool_name, arguments)


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_argv"),
    [
        pytest.param(
            "iterate",
            {},
            ["gym", "rat", "iterate", "--format", "json"],
            id="iterate-empty",
        ),
        pytest.param(
            "probe",
            {},
            ["gym", "rat", "probe", "--format", "json", "--"],
            id="probe-no-args",
        ),
        pytest.param(
            "probe",
            {"names": []},
            ["gym", "rat", "probe", "--format", "json", "--"],
            id="probe-empty-names",
        ),
        pytest.param(
            "probe",
            {"samples": 6},
            ["gym", "rat", "probe", "--format", "json", "--samples", "6", "--"],
            id="probe-samples-only",
        ),
        pytest.param(
            "probe",
            {"names": ["a", "b"]},
            ["gym", "rat", "probe", "--format", "json", "--", "a", "b"],
            id="probe-names-only",
        ),
        pytest.param(
            "probe",
            {"names": ["a", "b"], "samples": 6},
            ["gym", "rat", "probe", "--format", "json", "--samples", "6", "--", "a", "b"],
            id="probe-names-and-samples",
        ),
    ],
)
async def test_gymrat_tools_factory_when_valid_arguments_given_does_run_child_with_expected_argv(
    sdk_config: dict[str, Any],
    fake_exec: AsyncMock,
    tool_name: str,
    arguments: dict[str, Any],
    expected_argv: list[str],
) -> None:
    result = await _call_via_sdk(sdk_config, tool_name, arguments)

    assert result.is_error is False
    assert list(fake_exec.call_args[0][0]) == expected_argv


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        pytest.param("iterate", {"anything": 1}, id="iterate-extra-key"),
        pytest.param("probe", {"samples": 0}, id="probe-zero-samples"),
        pytest.param("probe", {"samples": True}, id="probe-bool-samples"),
        pytest.param("probe", {"names": "a"}, id="probe-string-names"),
    ],
)
async def test_gymrat_tools_factory_when_invalid_arguments_given_does_reject_before_running_child(
    sdk_config: dict[str, Any],
    fake_exec: AsyncMock,
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    result = await _call_via_sdk(sdk_config, tool_name, arguments)

    assert result.is_error is True
    fake_exec.assert_not_called()


async def test_gymrat_tools_factory_when_called_does_run_tools_in_the_root_with_the_env(
    create_scratch_repo: Callable[[], str],
) -> None:
    factory = gymrat_tools_factory(create_scratch_repo())
    config: dict[str, Any] = factory(asyncio.Event(), {"GYMRAT_SAMPLES": "banana"})  # type: ignore[assignment]  # McpSdkServerConfig is a TypedDict

    result = await _call_via_sdk(config, "probe", {})

    assert (config["type"], config["name"], result.is_error) == ("sdk", "gymrat", True)
    assert result.content == [
        TextContent(
            type="text",
            text='Error: Invalid value for GYMRAT_SAMPLES: expected a positive integer, got "banana"',
        )
    ]


async def test_gymrat_tools_factory_when_abort_set_does_kill_the_tool_call(
    tmp_path: pathlib.Path,
) -> None:
    abort = asyncio.Event()
    abort.set()
    config: dict[str, Any] = gymrat_tools_factory(str(tmp_path))(abort, {})  # type: ignore[assignment]  # McpSdkServerConfig is a TypedDict

    result = await _call_via_sdk(config, "probe", {})

    assert result.is_error is True
    assert result.content == [TextContent(type="text", text="killed by the supervisor")]


# ---------------------------------------------------------------------------
# real CLI integration
# ---------------------------------------------------------------------------

pytestmark_posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only worktrees")


def _started_repo(create_scratch_repo: Callable[[], str], *, filter_template: str | None) -> str:
    """A scratch repository with a bench project, an open session, and a recorded baseline."""
    repo = create_scratch_repo()
    commit_project(repo, samples=2, filter_template=filter_template)
    run_cli(["start", "--baseline", "main"], repo, timeout=60)
    run_cli(["measure", "--record"], repo, timeout=60)
    return repo


@pytestmark_posix
async def test_probe_when_real_cli_given_names_and_samples_does_return_scoped_json_document(
    create_scratch_repo: Callable[[], str],
) -> None:
    repo = _started_repo(create_scratch_repo, filter_template=FILTER_TEMPLATE)
    host = ToolHost(root=repo, abort=asyncio.Event(), extra_env={})

    result = await host.probe({"names": ["latency"], "samples": 2})

    assert result["is_error"] is False, f"probe error: {_text_of(result)}"
    doc = json.loads(_text_of(result))
    assert doc["scoped"] is True
    assert doc["names"] == ["latency"]
    assert doc["samples"] == 2


def _scoped_config_path(repo: str) -> pathlib.Path:
    """Path of the alternate config whose filter would let a scoped probe run."""
    return pathlib.Path(repo) / "scoped.toml"


def _help_flag_name(_repo: str) -> list[str]:
    """A metric name spelled like the CLI's own help flag."""
    return ["--help"]


def _config_flag_names(repo: str) -> list[str]:
    """Metric names spelled like the config option and the path it would take."""
    return ["--config", str(_scoped_config_path(repo))]


@pytestmark_posix
@pytest.mark.parametrize(
    "names_of",
    [
        pytest.param(_help_flag_name, id="help-flag"),
        pytest.param(_config_flag_names, id="config-flag-and-path"),
    ],
)
async def test_probe_when_real_cli_given_option_like_names_does_return_the_cli_rejection(
    create_scratch_repo: Callable[[], str],
    names_of: Callable[[str], list[str]],
) -> None:
    repo = _started_repo(create_scratch_repo, filter_template=None)
    _scoped_config_path(repo).write_text(
        config_text(samples=2, filter_template=FILTER_TEMPLATE), encoding="utf-8"
    )
    host = ToolHost(root=repo, abort=asyncio.Event(), extra_env={})

    result = await host.probe({"names": names_of(repo)})

    text = _text_of(result)
    assert result["is_error"] is True
    assert "filter is not configured" in text
    assert "Usage:" not in text


@pytestmark_posix
async def test_iterate_when_real_cli_on_scratch_repo_does_return_json_document(
    create_scratch_repo: Callable[[], str],
) -> None:
    repo = create_scratch_repo()
    commit_project(repo, samples=5)
    run_cli(["start", "--baseline", "main"], repo, timeout=60)
    host = ToolHost(root=repo, abort=asyncio.Event(), extra_env={})

    result = await host.iterate({})

    assert result["is_error"] is False
    doc = json.loads(_text_of(result))
    assert isinstance(doc, dict)
