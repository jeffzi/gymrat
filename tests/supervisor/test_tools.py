"""Behavioral tests for the ToolHost handler logic and SDK wiring."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, create_autospec

if TYPE_CHECKING:
    import pathlib
    from collections.abc import AsyncIterator, Awaitable, Callable

    from claude_agent_sdk import McpSdkServerConfig
    from mcp.types import CallToolResult

import pytest
from claude_agent_sdk import create_sdk_mcp_server
from mcp import Client

from gymrat.exec import (
    ExecOptions,
    ExecResult,
    ExecTimeoutError,
    exec_argv,
)
from gymrat.supervisor.tools import ToolHost, gymrat_tool_definitions
from tests._cli import ENTRY
from tests._exec_fixtures import expected_result

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _text_of(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_exec() -> AsyncMock:
    """An ``exec_argv`` replacement that records calls and returns a canned result."""
    return create_autospec(exec_argv, return_value=expected_result(stdout='{"ok": true}'))


@pytest.fixture
def host(tmp_path: pathlib.Path, fake_exec: AsyncMock) -> ToolHost:
    """A ToolHost wired to the fake executor, with a trivial prefix and no extra env."""
    return ToolHost(
        root=str(tmp_path),
        abort=asyncio.Event(),
        extra_env={},
        argv_prefix=["gym", "rat"],
        _exec_fn=fake_exec,
    )


# ---------------------------------------------------------------------------
# argv construction and child environment
# ---------------------------------------------------------------------------


async def test_probe_when_extra_env_names_a_fixed_key_does_let_the_extra_value_win(
    tmp_path: pathlib.Path, fake_exec: AsyncMock
) -> None:
    host = ToolHost(
        root=str(tmp_path),
        abort=asyncio.Event(),
        extra_env={"NO_COLOR": "0"},
        argv_prefix=["gym", "rat"],
        _exec_fn=fake_exec,
    )

    await host.probe({})

    opts: ExecOptions = fake_exec.call_args[0][1]
    assert opts.env is not None
    assert opts.env.get("NO_COLOR") == "0"


@pytest.mark.parametrize(
    "force_color",
    [pytest.param("1", id="force-color-set"), pytest.param(None, id="force-color-unset")],
)
async def test_probe_when_child_fails_does_return_error_text_without_ansi(
    tmp_path: pathlib.Path,
    color_env: Callable[[str | None, str | None], None],
    force_color: str | None,
) -> None:
    color_env(force_color, None)
    host = ToolHost(
        root=str(tmp_path),
        abort=asyncio.Event(),
        extra_env={},
        argv_prefix=ENTRY,
        _exec_fn=exec_argv,
    )

    result = await host.probe({})

    text = _text_of(result)
    assert (result["is_error"], text != "", "\x1b[" in text) == (True, True, False)


async def test_probe_when_run_does_hand_the_child_the_supervisor_environment(
    host: ToolHost, fake_exec: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GYMRAT_TEST_INHERITED", "banana")

    await host.probe({})

    opts: ExecOptions = fake_exec.call_args[0][1]
    assert opts.env is not None
    assert opts.env.get("GYMRAT_TEST_INHERITED") == "banana"


# ---------------------------------------------------------------------------
# result mapping
# ---------------------------------------------------------------------------


async def test_probe_when_child_exits_two_with_stderr_does_return_the_stderr(
    host: ToolHost, fake_exec: AsyncMock
) -> None:
    fake_exec.return_value = expected_result(stderr="bad config", exit_code=2)

    result = await host.probe({})

    assert result == {"content": [{"type": "text", "text": "bad config"}], "is_error": True}


@pytest.mark.parametrize(
    ("command", "exit_code", "fragments"),
    [
        pytest.param("probe", 2, ("probe", "2"), id="probe-exit-2"),
        pytest.param("iterate", 3, ("iterate",), id="iterate-exit-3"),
    ],
)
async def test_tool_host_when_child_fails_without_output_does_return_an_error_naming_the_command(
    host: ToolHost,
    fake_exec: AsyncMock,
    command: str,
    exit_code: int,
    fragments: tuple[str, ...],
) -> None:
    fake_exec.return_value = expected_result(exit_code=exit_code)

    result = await getattr(host, command)({})

    assert result["is_error"] is True
    assert [fragment for fragment in fragments if fragment not in _text_of(result)] == []


async def test_probe_when_child_exits_on_the_gate_with_a_document_does_return_it(
    host: ToolHost,
    fake_exec: AsyncMock,
) -> None:
    stdout = '{"ok": true}'
    fake_exec.return_value = expected_result(stdout=stdout, exit_code=1)

    result = await host.probe({})

    assert result == {"content": [{"type": "text", "text": stdout}], "is_error": False}


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        pytest.param(
            expected_result(stdout='{"kind": "probe", ', stderr="broken run"),
            "broken run",
            id="truncated-object-exit-0",
        ),
        pytest.param(
            expected_result(stdout='{"ok": true}', stderr="broken run", exit_code=3),
            "broken run",
            id="object-exit-3",
        ),
        pytest.param(
            expected_result(stdout="[1, 2]", stderr="broken run"),
            "broken run",
            id="array-exit-0",
        ),
        pytest.param(
            expected_result(stdout='{"kind": "probe", '),
            '{"kind": "probe",',
            id="truncated-object-no-stderr",
        ),
    ],
)
async def test_probe_when_stdout_is_not_a_document_does_return_the_error_output(
    host: ToolHost,
    fake_exec: AsyncMock,
    run: ExecResult,
    expected: str,
) -> None:
    fake_exec.return_value = run

    result = await host.probe({})

    assert result == {"content": [{"type": "text", "text": expected}], "is_error": True}


def _timed_out(stderr: str) -> ExecTimeoutError:
    """A timeout that left a probe document on stdout and ``stderr`` on stderr."""
    return ExecTimeoutError(
        stdout='{"kind": "probe"}',
        stderr=stderr,
        timeout_ms=5,
        stdout_bytes=17,
        stderr_bytes=len(stderr.encode()),
    )


async def test_probe_when_run_times_out_with_stderr_does_return_the_stderr(
    host: ToolHost,
    fake_exec: AsyncMock,
) -> None:
    fake_exec.return_value = _timed_out("bench ran past its deadline\n")

    result = await host.probe({})

    assert result == {
        "content": [{"type": "text", "text": "bench ran past its deadline"}],
        "is_error": True,
    }


async def test_probe_when_run_times_out_without_stderr_does_return_an_error_naming_probe(
    host: ToolHost,
    fake_exec: AsyncMock,
) -> None:
    fake_exec.return_value = _timed_out("")

    result = await host.probe({})

    assert (result["is_error"], "probe" in _text_of(result)) == (True, True)


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


@dataclass(frozen=True, slots=True)
class _BusyHost:
    host: ToolHost
    release: Callable[[], Awaitable[None]]


@pytest.fixture
async def busy_host(host: ToolHost, fake_exec: AsyncMock) -> AsyncIterator[_BusyHost]:
    """A host with a ``probe`` call still running; teardown lets that call finish."""
    blocker = _block_first_call(fake_exec)
    first = asyncio.create_task(host.probe({}))
    await asyncio.sleep(0)

    async def release() -> None:
        if not blocker.done():
            blocker.set_result(expected_result(stdout='{"ok": true}'))
        await first

    yield _BusyHost(host, release)
    await release()


async def test_iterate_when_a_call_is_running_does_refuse_the_concurrent_one(
    busy_host: _BusyHost,
) -> None:
    refused = await busy_host.host.iterate({})

    assert refused["is_error"] is True


async def test_probe_when_concurrent_refused_does_not_leave_host_busy(busy_host: _BusyHost) -> None:
    await busy_host.host.iterate({})
    await busy_host.release()

    after = await busy_host.host.probe({})

    assert after["is_error"] is False


# ---------------------------------------------------------------------------
# kill paths
# ---------------------------------------------------------------------------


async def test_probe_when_abort_fires_after_partial_json_does_return_killed(
    tmp_path: pathlib.Path, fake_exec: AsyncMock
) -> None:
    abort = asyncio.Event()
    abort.set()
    fake_exec.return_value = expected_result(stdout='{"kind": "probe", ', exit_code=1)
    host = ToolHost(
        root=str(tmp_path),
        abort=abort,
        extra_env={},
        argv_prefix=["gym", "rat"],
        _exec_fn=fake_exec,
    )

    result = await host.probe({})

    assert result["is_error"] is True
    assert '{"kind": "probe"' not in _text_of(result)


# ---------------------------------------------------------------------------
# SDK server input validation (through an in-memory MCP client)
# ---------------------------------------------------------------------------


@pytest.fixture
def sdk_config(host: ToolHost) -> McpSdkServerConfig:
    """An SDK server config exposing ``host``'s tool definitions."""
    return create_sdk_mcp_server("gymrat", "0.1.0", gymrat_tool_definitions(host))


async def _call_via_sdk(
    sdk_config: McpSdkServerConfig, tool_name: str, arguments: dict[str, Any]
) -> CallToolResult:
    """Call *tool_name* through an in-memory MCP client on the SDK server."""
    async with Client(sdk_config["instance"]) as client:
        return await client.call_tool(tool_name, arguments)


@pytest.mark.parametrize(
    ("arguments", "expected_options", "expected_names"),
    [
        pytest.param(
            {"samples": 6},
            ["probe", "--format", "json", "--samples", "6"],
            [],
            id="probe-samples-only",
        ),
        pytest.param(
            {"names": ["a", "b"]},
            ["probe", "--format", "json"],
            ["a", "b"],
            id="probe-names-only",
        ),
    ],
)
async def test_gymrat_tool_definitions_when_valid_arguments_given_does_put_names_after_the_separator(
    sdk_config: McpSdkServerConfig,
    fake_exec: AsyncMock,
    arguments: dict[str, Any],
    expected_options: list[str],
    expected_names: list[str],
) -> None:
    result = await _call_via_sdk(sdk_config, "probe", arguments)

    argv = list(fake_exec.call_args[0][0])
    split = argv.index("--")
    assert result.is_error is False
    assert (argv[split - len(expected_options) : split], argv[split + 1 :]) == (
        expected_options,
        expected_names,
    )
