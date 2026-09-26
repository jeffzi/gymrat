"""MCP tool host: translates handler calls into gymrat child-process runs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError, exec_argv

if TYPE_CHECKING:
    from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool

ToolsFactory = Callable[[asyncio.Event, Mapping[str, str]], object]

_ExecFn = Callable[[Sequence[str], ExecOptions], Awaitable[ExecResult | ExecTimeoutError]]

# click's default exit code for a UsageError (e.g. a rejected CLI argument).
_USAGE_ERROR_EXIT_CODE = 2

# gymrat emits its JSON document on success (0) and on a stop gate (1).
_DOCUMENT_EXIT_CODES = frozenset({0, 1})

_JSON_FORMAT_ARGS = ("--format", "json")


def _error_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def _ok_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": False}


def _validate_probe_input(input_data: dict[str, Any]) -> str | None:
    """Return an error message if ``input_data`` has invalid fields, else ``None``."""
    if "names" in input_data:
        names = input_data["names"]
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            return "names must be a list of strings"
    if "samples" in input_data:
        samples = input_data["samples"]
        if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
            return "samples must be a positive integer"
    return None


def _is_json_document(outcome: ExecResult) -> bool:
    """Return whether the run exited 0 or 1 with stdout holding a complete JSON object."""
    if outcome.exit_code not in _DOCUMENT_EXIT_CODES:
        return False
    try:
        parsed = json.loads(outcome.stdout.strip())
    except ValueError:
        return False
    return isinstance(parsed, dict)


class ToolHost:
    """Runs gymrat subcommands as child processes and maps results for MCP.

    Args:
        root: Session root directory, used as the child's working directory.
        abort: Event that, once set, kills in-flight child runs.
        extra_env: Extra environment variables forwarded to the child
            (e.g. ``GYMRAT_TRACEPARENT``).
        argv_prefix: Command prefix before the subcommand name. Defaults to
            ``[sys.executable, "-m", "gymrat"]``; override for testing.
        _exec_fn: Subprocess executor, exposed as a test seam. Defaults to
            :func:`gymrat.exec.exec_argv`.
    """

    def __init__(
        self,
        *,
        root: str,
        abort: asyncio.Event | None,
        extra_env: Mapping[str, str],
        argv_prefix: Sequence[str] = (sys.executable, "-m", "gymrat"),
        _exec_fn: _ExecFn = exec_argv,
    ) -> None:
        self._root = root
        self._abort = abort
        self._extra_env = extra_env
        self._prefix = list(argv_prefix)
        self._exec_fn = _exec_fn
        self._busy = False

    def _build_env(self) -> dict[str, str]:
        # The origin marks the run as one the agent made through a tool, so the
        # child can tell it apart from a command a person typed.
        return {
            **os.environ,
            "NO_COLOR": "1",
            "GYMRAT_COMMAND_ORIGIN": "tool",
            **self._extra_env,
        }

    def _build_options(self) -> ExecOptions:
        return ExecOptions(cwd=self._root, abort=self._abort, env=self._build_env())

    def _map_result(self, outcome: ExecResult | ExecTimeoutError, cmd: str) -> dict[str, Any]:
        if isinstance(outcome, ExecTimeoutError):
            text = outcome.stderr.strip() or f"gymrat {cmd} timed out"
            return _error_result(text)

        if outcome.exit_code == _USAGE_ERROR_EXIT_CODE:
            text = outcome.stderr.strip() or f"gymrat {cmd} exited 2 with no output"
            return _error_result(text)

        if _is_json_document(outcome):
            return _ok_result(outcome.stdout)

        if self._abort is not None and self._abort.is_set():
            return _error_result("killed by the supervisor")

        text = outcome.stderr.strip() or outcome.stdout.strip() or f"gymrat {cmd} failed"
        return _error_result(text)

    async def _run_command(self, argv: list[str], cmd: str) -> dict[str, Any]:
        """Execute *argv* as a child process, serializing against concurrent calls.

        Args:
            argv: Full argument vector for the child process.
            cmd: Subcommand name used in error messages (``"probe"`` or
                ``"iterate"``).

        Returns:
            MCP tool result dict with ``content`` and ``is_error``.
        """
        if self._busy:
            return _error_result("a gymrat command is already running")
        self._busy = True
        try:
            outcome = await self._exec_fn(argv, self._build_options())
            return self._map_result(outcome, cmd)
        finally:
            self._busy = False

    async def probe(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run ``gymrat probe`` with optional name filtering and sample count.

        Args:
            input_data: Tool input dict; may contain ``names`` (list of strings)
                and ``samples`` (positive integer).

        Returns:
            MCP tool result dict with ``content`` and ``is_error``.
        """
        err = _validate_probe_input(input_data)
        if err is not None:
            return _error_result(err)

        argv: list[str] = [*self._prefix, "probe"]

        samples = input_data.get("samples")
        if samples is not None:
            argv.extend(["--samples", str(samples)])

        argv.extend(_JSON_FORMAT_ARGS)

        # Every option precedes the separator: names come from the agent and may
        # look like flags, and only after ``--`` does the CLI read them as names.
        argv.append("--")
        names = input_data.get("names")
        if names:
            argv.extend(names)

        return await self._run_command(argv, "probe")

    async def iterate(self, _input_data: dict[str, Any]) -> dict[str, Any]:
        """Run ``gymrat iterate`` to advance the optimization loop.

        Args:
            _input_data: Tool input dict (currently unused).

        Returns:
            MCP tool result dict with ``content`` and ``is_error``.
        """
        argv = [*self._prefix, "iterate", *_JSON_FORMAT_ARGS]

        return await self._run_command(argv, "iterate")


def gymrat_tool_definitions(host: ToolHost) -> list[SdkMcpTool[dict[str, Any]]]:
    """Build the probe and iterate SDK tool definitions wired to *host*.

    The ``claude_agent_sdk`` import lives here so the module never pulls in the
    SDK (and its transitive dependencies) at import time.

    Args:
        host: The tool host whose handlers back the tool definitions.

    Returns:
        Two-element list of ``SdkMcpTool`` definitions, probe then iterate.
    """
    from claude_agent_sdk import (  # noqa: PLC0415 -- deferred to avoid import-time SDK load
        SdkMcpTool as _SdkMcpTool,
    )

    probe = _SdkMcpTool(
        name="probe",
        description=(
            "Bench the experiment worktree and report each metric's delta "
            "against the current baseline. Pass metric names to scope the run. "
            "Records nothing."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "names": {"type": "array", "items": {"type": "string"}},
                "samples": {"type": "integer"},
            },
            "additionalProperties": False,
        },
        handler=host.probe,
    )

    iterate = _SdkMcpTool(
        name="iterate",
        description=(
            "Measure the experiment worktree against the baseline at the "
            "configured samples and record the verdict. The only way to "
            "measure before keep."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=host.iterate,
    )

    return [probe, iterate]


def create_gymrat_tools(host: ToolHost) -> McpSdkServerConfig:
    """Create an SDK MCP server config exposing the gymrat tools.

    The ``claude_agent_sdk`` import lives here so the module never pulls in the
    SDK at import time.

    Args:
        host: The tool host whose handlers back the tool definitions.

    Returns:
        An ``McpSdkServerConfig`` with ``name="gymrat"`` and ``type="sdk"``.
    """
    from claude_agent_sdk import (  # noqa: PLC0415 -- deferred to avoid import-time SDK load
        create_sdk_mcp_server,
    )

    tools = gymrat_tool_definitions(host)
    return create_sdk_mcp_server("gymrat", "0.1.0", tools)


def gymrat_tools_factory(root: str) -> ToolsFactory:
    """Return a factory that builds the gymrat SDK server config on demand.

    The returned callable accepts an abort event and an environment mapping,
    constructs a :class:`ToolHost`, and wraps it in an ``McpSdkServerConfig``.

    Args:
        root: Session root directory for the tool host.

    Returns:
        A callable ``(abort, env) -> McpSdkServerConfig``.
    """

    def factory(abort: asyncio.Event, env: Mapping[str, str]) -> McpSdkServerConfig:
        host = ToolHost(root=root, abort=abort, extra_env=env)
        return create_gymrat_tools(host)

    return factory
