"""The ``gymrat export`` command: replay a finished session's spans to a collector."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from gymrat.cli.console import apply_command_flags
from gymrat.cli.exit import exit_with_error, run_guarded, write_and_flush
from gymrat.cli.options import (  # noqa: TC001 -- typer resolves these annotations at runtime
    ColorOption,
    DebugOption,
)
from gymrat.errors import GymratError
from gymrat.session.paths import repo_root, session_jsonl_path, supervisor_log_name
from gymrat.session.store import first_line_json, read_session_header
from gymrat.utils import ENDPOINT_ENV, otlp_endpoint, warn_to_stderr

_SessionLogArgument = Annotated[
    str | None, typer.Argument(metavar="[SESSION_LOG]", help="path to session.jsonl")
]
_EndpointOption = Annotated[
    str | None,
    typer.Option("--endpoint", envvar=ENDPOINT_ENV, help="OTLP HTTP endpoint URL"),
]

_OTEL_MISSING = (
    "OpenTelemetry SDK or OTLP exporter not available. Install with: uv tool install 'gymrat[otel]'"
)


def _matching_supervisor_logs(session_dir: Path, session_id: str) -> list[str]:
    """Find the supervisor logs whose launch line names one session.

    Reads only the first line of each file and checks the ``session_id`` field
    from the raw JSON dict. An entry the OS refuses to read, such as a directory
    or a file without read permission, is skipped with a warning on stderr so
    one bad entry never blocks the export of the others.

    Args:
        session_dir: The directory holding the supervisor logs.
        session_id: The session the launch line must name.

    Returns:
        The matching log paths, in sorted order.
    """
    matched: list[str] = []
    for path in sorted(session_dir.glob(supervisor_log_name("*"))):
        try:
            entry = first_line_json(path)
        except OSError as error:
            warn_to_stderr(f"warning: skipped {path}: {error.strerror or error}")
            continue
        if entry is None:
            continue
        if entry.get("type") == "launch" and entry.get("session_id") == session_id:
            matched.append(str(path))
    return matched


def export_command(
    session_log: _SessionLogArgument = None,
    endpoint: _EndpointOption = None,
    *,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Export a finished session's spans to an OpenTelemetry collector.

    Replays the session log, plus every supervisor log in its directory whose
    launch line names the same session, as spans sent over OTLP HTTP.

    Args:
        session_log: Path to the session's ``session.jsonl``, or ``None`` for
            the current repository's session log.
        endpoint: OTLP HTTP endpoint URL from ``--endpoint`` or
            ``OTEL_EXPORTER_OTLP_ENDPOINT``; surrounding whitespace is trimmed,
            and ``None`` or a blank value is an error.
        color: The ``--color``/``--no-color`` flag, or ``None`` when neither
            was given.
        debug: The ``--debug`` flag; shows stack traces on errors.

    Raises:
        typer.Exit: With the tool-failure code once any failure has been
            reported on stderr.
    """
    apply_command_flags(debug=debug, color=color)

    run_guarded(lambda: _export(session_log, endpoint))


def _export(session_log: str | None, endpoint: str | None) -> None:
    if session_log is None:
        session_log = session_jsonl_path(repo_root())

    session_path = Path(session_log)
    session_dir = session_path.parent
    header = read_session_header(str(session_path))
    if header is None:
        exit_with_error(f"No session found in {session_log}")

    session_id = header.session_id

    from gymrat.telemetry.provider import (  # noqa: PLC0415 -- lazy import keeps CLI startup off the telemetry stack
        configure_tracing,
        export_failed,
        flush_tracing,
        session_span_dropped,
    )
    from gymrat.telemetry.replay import (  # noqa: PLC0415 -- lazy import keeps CLI startup off the telemetry stack
        replay_session,
    )

    endpoint = otlp_endpoint(endpoint)
    if endpoint is None:
        exit_with_error(f"No endpoint: pass --endpoint or set {ENDPOINT_ENV}")

    if not configure_tracing(session_id, endpoint=endpoint):
        exit_with_error(_OTEL_MISSING)

    supervisor_logs = _matching_supervisor_logs(session_dir, session_id)
    count = replay_session(session_log, supervisor_logs)
    if session_span_dropped():
        msg = f"No spans recorded for session {session_id}: the tracer drops them."
        raise GymratError(
            msg,
            hint="Unset OTEL_SDK_DISABLED and set OTEL_TRACES_SAMPLER to a sampler that keeps "
            "the session trace.",
        )
    flush_tracing()
    if export_failed():
        exit_with_error(
            f"Could not export spans to {endpoint}: the collector is unreachable or rejected "
            "a batch. The exporter's log lines above give the cause."
        )

    write_and_flush(
        sys.stderr,
        f"exported {count} spans for session {session_id} to {endpoint}\n",
    )
