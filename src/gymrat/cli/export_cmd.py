"""The ``gymrat export`` command: replay a finished session's spans to a collector."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from gymrat.cli.shared import DebugOption, apply_debug, exit_with_error, write_and_flush
from gymrat.errors import GymratError
from gymrat.session.paths import repo_root, session_jsonl_path, supervisor_log_name
from gymrat.session.records.models import SessionRecord
from gymrat.session.records.parse import parse_record

SessionLogArg = Annotated[str | None, typer.Argument(help="path to session.jsonl")]
EndpointOption = Annotated[str | None, typer.Option("--endpoint", help="OTLP HTTP endpoint URL")]

_SDK_MISSING = "OpenTelemetry SDK not available. Install with: pip install 'gymrat[otel]'"


def _matching_supervisor_logs(session_dir: Path, session_id: str) -> list[str]:
    """Return supervisor logs in ``session_dir`` whose launch line names ``session_id``.

    Reads only the first line of each file and checks the ``session_id`` field
    from the raw JSON dict.
    """
    matched: list[str] = []
    for path in sorted(session_dir.glob(supervisor_log_name("*"))):
        entry = _first_line_json(path)
        if entry is None:
            continue
        if entry.get("type") == "launch" and entry.get("session_id") == session_id:
            matched.append(str(path))
    return matched


def _read_first_line(path: Path) -> str | None:
    """Return the first line of *path*, or ``None`` when the file is absent."""
    try:
        with path.open(encoding="utf-8") as fh:
            return fh.readline()
    except FileNotFoundError:
        return None


def _first_line_json(path: Path) -> dict[str, object] | None:
    """Return the first line of *path* parsed as a JSON object, or ``None``."""
    first_line = _read_first_line(path)
    if first_line is None:
        return None
    try:
        parsed = json.loads(first_line)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _session_header(session_path: Path) -> SessionRecord | None:
    """Return the session header from *session_path*, or ``None`` when absent.

    Returns:
        The session record, or ``None`` when the log does not exist.

    Raises:
        GymratError: When the first line is present but malformed JSON, an
            unrecognized record, or not a session header.
    """
    first_line = _read_first_line(session_path)
    if first_line is None or not first_line.strip():
        return None

    location = f"{session_path}:1"

    try:
        parsed = json.loads(first_line)
    except json.JSONDecodeError as error:
        message = f"Invalid JSON at {location}"
        raise GymratError(message, hint="Line 1 is not a JSON object.") from error

    record = parse_record(parsed)

    if not isinstance(record, SessionRecord):
        message = f"Expected session header at {location}, got a {record.type} record"
        raise GymratError(
            message,
            hint="Line 1 is not a session header. The session log is corrupt; start a new session.",
        )

    return record


def export_command(
    session_log: SessionLogArg = None,
    endpoint: EndpointOption = None,
    debug: DebugOption = False,  # noqa: FBT002 -- 1:1 pass-through of the --debug flag
) -> None:
    """Export a finished session's spans to an OpenTelemetry collector."""
    apply_debug(debug)

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
    header = _session_header(session_path)
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
