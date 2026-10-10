"""The command-run seam: repository lock and command-trace recording.

Wraps a command body in the single-flight lock, then appends a
:class:`CommandRecord` once the body settles. The exit codes it records live in
:mod:`gymrat.errors`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from gymrat.session.schema import CommandOrigin, CommandReason

import typer

from gymrat import clock as _clock
from gymrat.agent_env import COMMAND_ORIGIN_ENV, TOOL_ORIGIN, TRACEPARENT_ENV
from gymrat.errors import GATE_EXIT_CODE, TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.loop.stop_condition import LoopStopError
from gymrat.session.lock import acquire_lock
from gymrat.session.paths import (
    NotAGitRepositoryError,
    lockfile_path,
    repo_root,
    session_jsonl_path,
)
from gymrat.session.records import CommandRecord, SessionRecord, add_record_events
from gymrat.session.store import (
    append_record,
    read_records,
    recover_torn_tail,
    session_header,
)
from gymrat.utils import otlp_endpoint_from_env, warn_to_stderr

# ---------------------------------------------------------------------------
# Trace bookkeeping
# ---------------------------------------------------------------------------


def command_origin() -> CommandOrigin:
    """The running command's origin: ``tool`` only when the supervisor says so, else ``cli``."""
    return "tool" if os.environ.get(COMMAND_ORIGIN_ENV) == TOOL_ORIGIN else "cli"


@dataclass(slots=True)
class CommandTrace:
    """Mutable bag the body of :func:`with_repo_lock` writes into.

    The seam reads ``gate``, ``reason``, and ``seq`` after the body settles to
    populate the :class:`CommandRecord` it appends.  The body sets them freely;
    they carry no meaning to the seam beyond the exit-code / reason mapping.

    Attributes:
        seq: The iteration the command acted on, if any.
        gate: Whether the body's outcome gates the exit code.
        reason: Why the command settled as it did.
    """

    seq: int | None = None
    gate: bool = False
    reason: CommandReason | None = None


# ---------------------------------------------------------------------------
# Lock lifecycle
# ---------------------------------------------------------------------------


def _resolve_exit(  # noqa: PLR0911 -- flat branch per exception type, each an exit-code mapping
    trace: CommandTrace,
    caught: BaseException | None,
) -> tuple[Literal[0, 1, 2], CommandReason | None]:
    """Map a body outcome to the ``(exit_code, reason)`` pair for the command record."""
    if caught is None:
        if trace.gate:
            return GATE_EXIT_CODE, trace.reason
        return 0, None

    if isinstance(caught, LoopStopError):
        reason: CommandReason = caught.reason or "stop-condition"
        return GATE_EXIT_CODE, reason

    if isinstance(caught, typer.Exit):
        if caught.exit_code == 0:
            return 0, trace.reason
        if caught.exit_code == GATE_EXIT_CODE:
            return GATE_EXIT_CODE, trace.reason
        return TOOL_FAILURE_EXIT_CODE, trace.reason or "error"

    if isinstance(caught, GymratError):
        gym_reason: CommandReason = caught.reason or "error"
        return TOOL_FAILURE_EXIT_CODE, gym_reason

    return TOOL_FAILURE_EXIT_CODE, "error"


async def with_repo_lock[T](
    command: str,
    body: Callable[[CommandTrace], Awaitable[T]],
    *,
    args: dict[str, object] | None = None,
    root: str | None = None,
) -> T:
    """Hold the repository's single-flight lock for the length of ``body``.

    The lock is acquired around ``body`` and released however it settles —
    including on exception, and always before the caller renders its report.

    With no ``root``, a run outside every git repository executes ``body`` with
    no lock at all.

    A caller that passes ``root`` has already chosen the repository, so the
    working directory is never consulted and the unlocked "not a git
    repository" answer never applies — the lock is taken for that root whether
    or not it is a repository.

    Holding the lock is what makes repairing the session log safe: a torn final
    line can only belong to a writer the previous run left dead, so the tail is
    dropped here — once per command, before ``body`` reads or appends anything.

    After the body settles the seam appends a :class:`CommandRecord` to the
    session log when one exists and is non-empty.  A failure to append warns to
    stderr and never masks the body's result or exception.

    With an OTLP endpoint set, the appended record is also exported as a
    command span. Tracing is configured only then, for the session whose log
    holds the record, so a body that opens a new session is traced under it.
    A session log that cannot be read before the body, or read and parsed
    after it, warns to stderr and emits no span.

    Args:
        command: The command name recorded in the :class:`CommandRecord`.
        body: The async callable to run under the lock.
        args: Extra trace arguments to include in the command record.
        root: The repository to lock, record, and repair; ``None`` discovers it
            from the process working directory.

    Returns:
        The value returned by ``body``.

    Raises:
        LockContentionError: When another process already holds the lock.
        GymratError: When ``root`` is ``None`` and the repository root cannot
            be resolved for a reason other than not being inside a git
            repository, or when the lock file cannot be opened. The
            root-resolution failure is raised before ``body`` runs, the lock is
            taken, or a record is written.
        Exception: Whatever ``body`` raises, propagated unchanged.
    """
    trace = CommandTrace()
    start = _clock.monotonic_ms()

    if root is None:
        try:
            root = repo_root()
        except NotAGitRepositoryError:
            return await body(trace)

    jsonl = session_jsonl_path(root)
    start_ns = _clock.now_ns()
    pre_body_session_id, pre_body_lines = _pre_body_log(root, jsonl)

    release = acquire_lock(lockfile_path(root), command)
    caught: BaseException | None = None
    result: T
    try:
        recover_torn_tail(jsonl)
        result = await body(trace)
    except BaseException as exc:
        caught = exc
        raise
    finally:
        elapsed_ms = int(_clock.monotonic_ms() - start)
        exit_code, reason = _resolve_exit(trace, caught)
        try:
            appended = _try_append_command_record(
                jsonl=jsonl,
                command=command,
                args=args if args is not None else {},
                seq=trace.seq,
                exit_code=exit_code,
                reason=reason,
                elapsed_ms=elapsed_ms,
            )
            if appended is not None and pre_body_lines is not None:
                try:
                    _, tracing_active = _maybe_configure_tracing(root)
                    if tracing_active:
                        _emit_command_span(
                            appended,
                            jsonl=jsonl,
                            pre_body_session_id=pre_body_session_id,
                            start_ns=start_ns,
                            pre_body_lines=pre_body_lines,
                        )
                except Exception as span_error:  # noqa: BLE001 -- must not mask the command outcome
                    warn_to_stderr(f"failed to emit command span: {span_error}")
        finally:
            release()
    return result


# ---------------------------------------------------------------------------
# Command-record persistence
# ---------------------------------------------------------------------------


def _try_append_command_record(  # noqa: PLR0913 -- one parameter per distinct concern of the command record
    *,
    jsonl: str,
    command: str,
    args: dict[str, object],
    seq: int | None,
    exit_code: Literal[0, 1, 2],
    reason: CommandReason | None,
    elapsed_ms: int,
) -> CommandRecord | None:
    """Append a :class:`CommandRecord` when the session log exists and is non-empty.

    A failure to read the log's size, or to build or append the record, warns
    to stderr instead of raising.

    Args:
        jsonl: The session log to append to.
        command: The command name.
        args: The command's trace arguments.
        seq: The iteration the command acted on, if any.
        exit_code: The exit code the command settled with.
        reason: Why the command settled as it did, if recorded.
        elapsed_ms: How long the command ran, in milliseconds.

    Returns:
        The appended record, or ``None`` when nothing was appended.
    """
    try:
        if _jsonl_is_empty(jsonl):
            return None
        record = CommandRecord(
            type="command",
            at=_clock.now_ns(),
            name=command,
            args=args,
            exit_code=exit_code,
            reason=reason,
            duration_ms=elapsed_ms,
            origin=command_origin(),
            traceparent=os.environ.get(TRACEPARENT_ENV) or os.environ.get("TRACEPARENT"),
            seq=seq,
        )
        append_record(jsonl, record)
    except Exception as error:  # noqa: BLE001 -- construction or IO failure must not mask the command outcome
        warn_to_stderr(f"failed to append command record: {error}")
        return None
    return record


# ---------------------------------------------------------------------------
# Telemetry tracing
# ---------------------------------------------------------------------------


def _pre_body_log(root: str, jsonl: str) -> tuple[str, int | None]:
    """Read what the command span needs from the session log before the body runs.

    The span's events are the records the body appends, so the line count must
    be taken before the body runs. Without an endpoint no span is emitted and
    the log is not read. A log that exists but cannot be read warns to stderr:
    without its line count the span cannot tell the body's records apart, so
    none is emitted.

    Args:
        root: The repository whose session header names the session.
        jsonl: The session log to count lines of.

    Returns:
        The session id ``jsonl`` belongs to (empty when there is none, or no
        endpoint) and its line count: ``0`` without an endpoint or a log,
        ``None`` when the log could not be read.
    """
    if otlp_endpoint_from_env() is None:
        return "", 0
    header = session_header(root)
    session_id = header.session_id if header is not None else ""
    try:
        return session_id, _count_lines(jsonl)
    except OSError as error:
        warn_to_stderr(f"failed to emit command span: {error}")
        return session_id, None


def _maybe_configure_tracing(root: str) -> tuple[str, bool]:
    """Configure tracing for the session whose log is at ``root`` now.

    Runs after the body, so a body that opens a session (``start``, the
    supervise preflight) traces under the session it opened, the same one
    every later span in the process uses.

    Args:
        root: The repository whose session header names the session.

    Returns:
        The session id (empty when there is no endpoint or no session) and
        whether a tracer provider is now active.

    Raises:
        ValueError: When tracing is already configured for another session.
    """
    if otlp_endpoint_from_env() is None:
        return "", False
    header = session_header(root)
    if header is None:
        return "", False

    from gymrat.telemetry.provider import (  # noqa: PLC0415 -- deferred: the telemetry stack and the optional otel extra stay off the CLI import path
        configure_tracing,
    )

    active = configure_tracing(header.session_id)
    return header.session_id, active


def _jsonl_is_empty(jsonl_path: str) -> bool:
    """True when the session log doesn't exist or has no content yet."""
    try:
        return Path(jsonl_path).stat().st_size == 0
    except FileNotFoundError:
        return True


def _count_lines(jsonl_path: str) -> int:
    """Count newline-terminated lines in ``jsonl_path``, 0 if absent."""
    try:
        data = Path(jsonl_path).read_bytes()
    except FileNotFoundError:
        return 0
    return data.count(b"\n")


def _emit_command_span(
    cmd_record: CommandRecord,
    *,
    jsonl: str,
    pre_body_session_id: str,
    start_ns: int,
    pre_body_lines: int,
) -> None:
    """Create a retroactive span for ``cmd_record``, the last record of ``jsonl``.

    The span belongs to the session whose log holds the record, read after the
    body ran: a body such as ``start`` may have replaced the log it found. The
    records the body appended to that log become the span's events.

    Args:
        cmd_record: The command record just appended to ``jsonl``.
        jsonl: The session log holding ``cmd_record``.
        pre_body_session_id: The session ``jsonl`` belonged to before the body ran.
        start_ns: When the command started, in nanoseconds since the epoch.
        pre_body_lines: How many lines ``jsonl`` held before the body ran.
    """
    from opentelemetry.trace import (  # noqa: PLC0415 -- deferred: the telemetry stack and the optional otel extra stay off the CLI import path
        NonRecordingSpan,
        set_span_in_context,
    )

    from gymrat.telemetry.command_span import (  # noqa: PLC0415 -- deferred: the telemetry stack and the optional otel extra stay off the CLI import path
        start_command_span,
    )
    from gymrat.telemetry.provider import (  # noqa: PLC0415 -- deferred: the telemetry stack and the optional otel extra stay off the CLI import path
        flush_tracing,
        parse_traceparent,
    )
    from gymrat.telemetry.session_span import (  # noqa: PLC0415 -- deferred: the telemetry stack and the optional otel extra stay off the CLI import path
        existing_session_span,
    )

    records = read_records(jsonl)
    if not records or not isinstance(records[0], SessionRecord):
        return
    session_id = records[0].session_id
    first_body_line = pre_body_lines if session_id == pre_body_session_id else 0

    gymrat_tp = os.environ.get(TRACEPARENT_ENV)
    parent_span_ctx = parse_traceparent(gymrat_tp) if gymrat_tp else None
    if parent_span_ctx is not None:
        parent_ctx = set_span_in_context(NonRecordingSpan(parent_span_ctx))
    else:
        parent_ctx = set_span_in_context(existing_session_span(session_id))

    with start_command_span(
        cmd_record,
        session_id=session_id,
        line_number=len(records),
        context=parent_ctx,
        start_time=start_ns,
    ) as span:
        add_record_events(span, records[first_body_line:-1])

    flush_tracing()
