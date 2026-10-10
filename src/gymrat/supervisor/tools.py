"""MCP tool host: translates handler calls into gymrat child-process runs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from gymrat.agent_env import COMMAND_ORIGIN_ENV, TOOL_ORIGIN
from gymrat.errors import GATE_EXIT_CODE, TOOL_FAILURE_EXIT_CODE
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError, exec_argv
from gymrat.supervisor.tool_names import ITERATE_TOOL_NAME, MCP_SERVER, PROBE_TOOL_NAME

if TYPE_CHECKING:
    from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool

type ToolsFactory = Callable[[asyncio.Event, Mapping[str, str]], McpSdkServerConfig]

_ExecFn = Callable[[Sequence[str], ExecOptions], Awaitable[ExecResult | ExecTimeoutError]]

# gymrat emits its JSON document on success and on a stop gate.
_DOCUMENT_EXIT_CODES = frozenset({0, GATE_EXIT_CODE})

_JSON_FORMAT_ARGS = ("--format", "json")


def _result(text: str, *, is_error: bool) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}


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
        abort: asyncio.Event,
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

    def _map_result(self, outcome: ExecResult | ExecTimeoutError, cmd: str) -> dict[str, Any]:
        if isinstance(outcome, ExecTimeoutError):
            text = outcome.stderr.strip() or f"gymrat {cmd} timed out"
            return _result(text, is_error=True)

        # A tool failure, which includes click's UsageError for a rejected argument.
        if outcome.exit_code == TOOL_FAILURE_EXIT_CODE:
            text = (
                outcome.stderr.strip()
                or f"gymrat {cmd} exited {TOOL_FAILURE_EXIT_CODE} with no output"
            )
            return _result(text, is_error=True)

        if _is_json_document(outcome):
            return _result(outcome.stdout, is_error=False)

        if self._abort.is_set():
            return _result("killed by the supervisor", is_error=True)

        text = outcome.stderr.strip() or outcome.stdout.strip() or f"gymrat {cmd} failed"
        return _result(text, is_error=True)

    async def _run_command(self, cmd: str, args: Sequence[str] = ()) -> dict[str, Any]:
        """Run ``gymrat <cmd>`` as a child process, serializing against concurrent calls.

        Args:
            cmd: Subcommand to run (``"probe"`` or ``"iterate"``), also named
                in error messages.
            args: Arguments placed after the JSON format flag.

        Returns:
            MCP tool result dict with ``content`` and ``is_error``.
        """
        if self._busy:
            return _result("a gymrat command is already running", is_error=True)
        self._busy = True
        argv = [*self._prefix, cmd, *_JSON_FORMAT_ARGS, *args]
        try:
            # The origin marks the run as one the agent made through a tool, so the
            # child can tell it apart from a command a person typed.
            env = {
                **os.environ,
                "NO_COLOR": "1",
                COMMAND_ORIGIN_ENV: TOOL_ORIGIN,
                **self._extra_env,
            }
            exec_options = ExecOptions(cwd=self._root, abort=self._abort, env=env)
            outcome = await self._exec_fn(argv, exec_options)
            return self._map_result(outcome, cmd)
        finally:
            self._busy = False

    async def probe(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run ``gymrat probe`` with optional name filtering and sample count.

        Args:
            input_data: Tool input dict, already validated against the tool
                schema; may contain ``names`` (list of strings) and ``samples``
                (positive integer).

        Returns:
            MCP tool result dict with ``content`` and ``is_error``.
        """
        samples = input_data.get("samples")
        options = () if samples is None else ("--samples", str(samples))

        # Every option precedes the separator: names come from the agent and may
        # look like flags, and only after ``--`` does the CLI read them as names.
        return await self._run_command("probe", (*options, "--", *(input_data.get("names") or ())))

    async def iterate(self, _input_data: dict[str, Any]) -> dict[str, Any]:
        """Run ``gymrat iterate`` to advance the optimization loop.

        Args:
            _input_data: Tool input dict (currently unused).

        Returns:
            MCP tool result dict with ``content`` and ``is_error``.
        """
        return await self._run_command("iterate")


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
        name=PROBE_TOOL_NAME,
        description=(
            "Bench the experiment worktree and report each metric's delta "
            "against the current baseline. Pass metric names to scope the run. "
            "Records nothing."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "names": {"type": "array", "items": {"type": "string"}},
                "samples": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        handler=host.probe,
    )

    iterate = _SdkMcpTool(
        name=ITERATE_TOOL_NAME,
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


def gymrat_tools_factory(root: str) -> ToolsFactory:
    """Return a factory that builds the gymrat SDK server config on demand.

    The ``claude_agent_sdk`` import happens when the returned callable runs, so
    importing this module never loads the SDK.

    Args:
        root: Session root directory for the tool host.

    Returns:
        A callable ``(abort, env)`` that constructs a :class:`ToolHost` and
        returns an ``McpSdkServerConfig`` with ``name="gymrat"`` and
        ``type="sdk"`` exposing its tools.
    """

    def factory(abort: asyncio.Event, env: Mapping[str, str]) -> McpSdkServerConfig:
        from claude_agent_sdk import (  # noqa: PLC0415 -- deferred to avoid import-time SDK load
            create_sdk_mcp_server,
        )

        host = ToolHost(root=root, abort=abort, extra_env=env)
        return create_sdk_mcp_server(MCP_SERVER, "0.1.0", gymrat_tool_definitions(host))

    return factory
