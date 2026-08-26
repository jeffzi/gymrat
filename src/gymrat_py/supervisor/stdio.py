"""A subprocess stdio driver: an agent session run in a child process.

``create_stdio_driver`` returns a :class:`~gymrat_py.supervisor.driver.Driver`
that runs each session in a child process (the supplied ``argv``). The driver
and the child speak a line-delimited JSON protocol over the child's stdio:

- The driver writes one ``start`` command line to stdin, carrying the prompt in
  camelCase, and may later write an ``interrupt`` command line.
- The child writes session-event lines — the same camelCase wire form
  :func:`~gymrat_py.supervisor.events.to_json_line` produces — followed by a
  terminal ``{"type": "outcome", ...}`` line. Its stderr is never relayed.

The child is spawned into its own process group so an abort can tree-kill every
descendant. The session never raises: a spawn failure, a nonzero exit without an
outcome, an interrupt, and an abort each settle a :class:`SessionOutcome`.
"""

import asyncio
import contextlib
import json
import warnings
from collections.abc import Sequence
from typing import Any, cast

from gymrat_py.errors import message_of
from gymrat_py.process_group import current_platform, kill_process_group
from gymrat_py.supervisor.driver import (
    Driver,
    DriverSession,
    SessionEndReason,
    SessionObserver,
    SessionOutcome,
    SessionPrompt,
)
from gymrat_py.supervisor.events import UsageUpdateEvent, event_from_wire

_READ_CHUNK = 65536
"""Bytes requested per read while draining the child's stderr."""

_STREAM_LIMIT = 8 * 1024 * 1024
"""Max bytes buffered for a single child stdout line before the reader overruns.

Event lines can legitimately carry large tool payloads, so the limit sits well
above asyncio's 64 KiB default; the overrun guard in ``_read_events`` still stops
an unbounded, unterminated line from escaping as an unhandled read error.
"""

_TEARDOWN_GRACE_SECONDS = 2.0
"""How long teardown waits for a killed child to exit before abandoning the reap.

A healthy child dies within milliseconds of the group kill, so the grace only
elapses for one stuck in an uninterruptible state — which no longer wait would
save — and it keeps teardown from wedging forever on such a child.
"""


def _start_command(prompt: SessionPrompt) -> dict[str, object]:
    """Build the ``start`` command, omitting prompt optionals that are ``None``."""
    wire: dict[str, object] = {"kickoff": prompt.kickoff, "cwd": prompt.cwd}
    if prompt.system_prompt_append is not None:
        wire["systemPromptAppend"] = prompt.system_prompt_append
    if prompt.model is not None:
        wire["model"] = prompt.model
    return {"type": "start", "prompt": wire}


def _outcome_from_wire(wire: dict[str, Any], cost_usd: float) -> SessionOutcome:
    """Read a terminal ``outcome`` line, falling back to the running cost."""
    reason = cast("SessionEndReason", wire.get("reason", "error"))
    raw_cost = wire.get("costUsd")
    cost = float(raw_cost) if isinstance(raw_cost, (int, float)) else cost_usd
    message = wire.get("message")
    return SessionOutcome(reason=reason, cost_usd=cost, message=message)


class _StdioSession:
    """Runs one child-process session as a task; ``outcome`` settles when it ends.

    ``interrupt`` and an external ``abort`` both drive the session to an
    ``interrupted`` outcome, and they win over any terminal ``outcome`` line the
    child sends afterwards. The running cost is the value from the child's most
    recent ``usage_update`` event; an interrupt or abort settles with that cost.
    """

    def __init__(
        self,
        argv: Sequence[str],
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None,
    ) -> None:
        self._argv = tuple(argv)
        self._prompt = prompt
        self._observer = observer
        self._abort = abort
        self._proc: asyncio.subprocess.Process | None = None
        self._abort_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._cost_usd = 0.0
        self._interrupt_requested = False
        self._outcome: SessionOutcome | None = None
        self._task: asyncio.Task[SessionOutcome] = asyncio.create_task(self._run())

    @property
    def outcome(self) -> asyncio.Task[SessionOutcome]:
        return self._task

    async def interrupt(self) -> None:
        self._interrupt_requested = True
        await self._write_line({"type": "interrupt"})

    async def _write_line(self, obj: dict[str, object]) -> None:
        """Write one JSON command line to the child's stdin, best-effort.

        A closed pipe means the child is already gone, which is an expected
        end-of-run condition rather than an error to surface.
        """
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write((json.dumps(obj, separators=(",", ":")) + "\n").encode())
            await proc.stdin.drain()
        except (OSError, ConnectionError):
            pass

    async def _spawn(self) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            *self._argv,
            cwd=self._prompt.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=current_platform() != "win32",
            limit=_STREAM_LIMIT,
        )

    async def _run(self) -> SessionOutcome:
        try:
            proc = await self._spawn()
        except OSError as err:
            return SessionOutcome(reason="error", cost_usd=0.0, message=message_of(err))
        self._proc = proc
        if self._abort is not None:
            self._abort_task = asyncio.create_task(self._watch_abort(self._abort, proc))
        if proc.stderr is not None:
            self._stderr_task = asyncio.create_task(self._drain(proc.stderr))
        try:
            await self._write_line(_start_command(self._prompt))
            await self._read_events(proc)
            return await self._resolve(proc)
        finally:
            await self._teardown(proc)

    async def _read_events(self, proc: asyncio.subprocess.Process) -> None:
        """Relay event lines from the child's stdout until a terminal outcome or EOF.

        A child line longer than ``_STREAM_LIMIT`` makes the reader overrun rather
        than yield a line; catch that so an oversized, unterminated line settles a
        clean error outcome instead of escaping as an unhandled read error. The
        child is left as-is here; ``_teardown`` kills and reaps it.
        """
        if proc.stdout is None:
            return
        try:
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").strip()
                if line and self._consume(line):
                    return
        except (asyncio.LimitOverrunError, ValueError) as err:
            self._outcome = SessionOutcome(
                reason="error",
                cost_usd=self._cost_usd,
                message=f"child output line exceeded the read limit: {message_of(err)}",
            )

    def _consume(self, line: str) -> bool:
        """Handle one stdout line; return ``True`` once the terminal outcome arrives."""
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            return False
        if isinstance(decoded, dict) and decoded.get("type") == "outcome":
            self._outcome = _outcome_from_wire(cast("dict[str, Any]", decoded), self._cost_usd)
            return True
        event = event_from_wire(decoded)
        if event is None:
            return False
        if isinstance(event, UsageUpdateEvent):
            self._cost_usd = event.cost_usd
        self._observer(event)
        return False

    async def _resolve(self, proc: asyncio.subprocess.Process) -> SessionOutcome:
        # Settle from a stop signal or a terminal line without awaiting the child's
        # exit: a child that ignores the kill must not block the outcome. Teardown
        # reaps it under a bound. Only the "exited without an outcome" path needs
        # the return code, and there the child has already ended its stdout.
        if self._interrupt_requested or (self._abort is not None and self._abort.is_set()):
            return SessionOutcome(reason="interrupted", cost_usd=self._cost_usd)
        if self._outcome is not None:
            return self._outcome
        returncode = await proc.wait()
        return SessionOutcome(
            reason="error",
            cost_usd=self._cost_usd,
            message=f"child process exited with code {returncode}",
        )

    async def _watch_abort(self, abort: asyncio.Event, proc: asyncio.subprocess.Process) -> None:
        await abort.wait()
        # Tree-kill the group so the blocked stdout read unblocks; the resolution
        # then settles ``interrupted`` because the abort is set.
        kill_process_group(proc.pid)

    async def _drain(self, reader: asyncio.StreamReader) -> None:
        """Consume the child's stderr to EOF and discard it; it is never relayed.

        Draining keeps a chatty child from blocking on a full stderr pipe.
        """
        with contextlib.suppress(OSError):
            while await reader.read(_READ_CHUNK):
                pass

    async def _teardown(self, proc: asyncio.subprocess.Process) -> None:
        for task in (self._abort_task, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if proc.returncode is None:
            kill_process_group(proc.pid)
        if proc.stdin is not None:
            proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), _TEARDOWN_GRACE_SECONDS)
        except TimeoutError:
            warnings.warn(
                f"child process {proc.pid} did not exit within "
                f"{_TEARDOWN_GRACE_SECONDS:g}s of the group kill; abandoning the wait",
                RuntimeWarning,
                stacklevel=2,
            )


class _StdioDriver:
    def __init__(self, argv: Sequence[str]) -> None:
        self._argv = tuple(argv)

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None = None,
    ) -> DriverSession:
        return _StdioSession(self._argv, prompt, observer, abort)


def create_stdio_driver(argv: Sequence[str]) -> Driver:
    """Build a :class:`Driver` that runs each session as the ``argv`` child process.

    Args:
        argv: The command (program plus arguments) to spawn per session. The
            child speaks the line-delimited JSON protocol over its stdio.

    Returns:
        A driver whose ``start`` launches one child process per call.
    """
    return _StdioDriver(argv)
