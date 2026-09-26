"""The ``gymrat export`` command: replay a finished session's spans to a collector."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from gymrat.cli.shared import (
    ColorOption,
    DebugOption,
    apply_color_override,
    apply_debug,
    exit_with_error,
    write_and_flush,
)
from gymrat.session.paths import repo_root, session_jsonl_path, supervisor_log_name
from gymrat.session.store import first_line_json, read_session_header
from gymrat.warn import warn_to_stderr

SessionLogArg = Annotated[
    str | None, typer.Argument(metavar="[SESSION_LOG]", help="path to session.jsonl")
]
EndpointOption = Annotated[str | None, typer.Option("--endpoint", help="OTLP HTTP endpoint URL")]

_SDK_MISSING = "OpenTelemetry SDK not available. Install with: pip install 'gymrat[otel]'"


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
    session_log: SessionLogArg = None,
    endpoint: EndpointOption = None,
    color: ColorOption = None,
    debug: DebugOption = False,  # noqa: FBT002 -- 1:1 pass-through of the --debug flag
) -> None:
    """Export a finished session's spans to an OpenTelemetry collector."""
    apply_debug(debug)
    apply_color_override(color)

    try:
        _export(session_log, endpoint)
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 -- CLI boundary: route any failure through the formatter
        exit_with_error(error)


def _export(session_log: str | None, endpoint: str | None) -> None:
    if session_log is None:
        session_log = session_jsonl_path(repo_root())

    session_path = Path(session_log)
    session_dir = session_path.parent
    header = read_session_header(str(session_path))
    if header is None:
        exit_with_error(f"No session found in {session_log}")

    session_id = header.session_id

    resolved_endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not resolved_endpoint:
        exit_with_error("No endpoint: pass --endpoint or set OTEL_EXPORTER_OTLP_ENDPOINT")

    if endpoint:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint

    from gymrat.telemetry.provider import configure_tracing, flush_tracing  # noqa: PLC0415
    from gymrat.telemetry.replay import replay_session  # noqa: PLC0415

    try:
        if not configure_tracing(session_id):
            exit_with_error(_SDK_MISSING)
    except ImportError:
        exit_with_error(_SDK_MISSING)

    supervisor_logs = _matching_supervisor_logs(session_dir, session_id)
    count = replay_session(session_log, supervisor_logs)
    flush_tracing()

    write_and_flush(
        sys.stderr,
        f"exported {count} spans for session {session_id} to {resolved_endpoint}\n",
    )
