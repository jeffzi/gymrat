"""The ``gymrat export`` command: replay a finished session's spans to a collector."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from gymrat.cli.console import apply_command_flags
from gymrat.cli.exit import run_guarded
from gymrat.cli.options import (  # noqa: TC001 -- typer resolves these annotations at runtime
    ColorOption,
    DebugOption,
)
from gymrat.errors import GymratError
from gymrat.session.paths import repo_root, session_jsonl_path, supervisor_log_name
from gymrat.session.store import read_session_header
from gymrat.utils import ENDPOINT_ENV, TRACES_ENDPOINT_ENV, otlp_traces_url, write_and_flush

_SessionLogArgument = Annotated[
    str | None, typer.Argument(metavar="[SESSION_LOG]", help="path to session.jsonl")
]
# No ``envvar=`` on the option: configure_tracing must see whether the endpoint
# came from the flag, which beats OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, or from
# OTEL_EXPORTER_OTLP_ENDPOINT, which yields to it.
_EndpointOption = Annotated[
    str | None,
    typer.Option(
        "--endpoint",
        help="OTLP HTTP endpoint URL; overrides OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, "
        f"defaults to ${ENDPOINT_ENV}",
    ),
]

_OTEL_MISSING = (
    "OpenTelemetry SDK or OTLP exporter not available. Install with: uv tool install 'gymrat[otel]'"
)


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

    \f
    Args:
        session_log: Path to the session's ``session.jsonl``, or ``None`` for
            the current repository's session log.
        endpoint: OTLP HTTP endpoint URL from ``--endpoint``, or ``None`` to
            read ``OTEL_EXPORTER_OTLP_ENDPOINT``; surrounding whitespace is
            trimmed, and no endpoint or a blank one is an error.
        color: The ``--color``/``--no-color`` flag, or ``None`` when neither
            was given.
        debug: The ``--debug`` flag; shows stack traces on errors.

    Raises:
        typer.Exit: With the tool-failure code once any failure has been
            reported on stderr.
    """  # noqa: D301 -- the form feed is deliberate: click cuts --help at it, hiding these sections
    apply_command_flags(debug=debug, color=color)

    run_guarded(lambda: _export(session_log, endpoint))


def _export(session_log: str | None, endpoint: str | None) -> None:
    if session_log is None:
        session_log = session_jsonl_path(repo_root())

    header = read_session_header(session_log)
    if header is None:
        msg = f"No session found in {session_log}"
        raise GymratError(msg)
    session_id = header.session_id

    from gymrat.telemetry.provider import (  # noqa: PLC0415 -- lazy import keeps CLI startup off the telemetry stack
        configure_tracing,
        export_failed,
        exported_span_count,
        flush_tracing,
        session_span_dropped,
    )
    from gymrat.telemetry.replay import (  # noqa: PLC0415 -- lazy import keeps CLI startup off the telemetry stack
        replay_session,
    )

    destination = otlp_traces_url(endpoint)
    if destination is None:
        msg = f"No endpoint: pass --endpoint or set {ENDPOINT_ENV} or {TRACES_ENDPOINT_ENV}"
        raise GymratError(msg)

    if not configure_tracing(session_id, endpoint=endpoint):
        raise GymratError(_OTEL_MISSING)

    session_dir = Path(session_log).parent
    supervisor_logs = [str(path) for path in sorted(session_dir.glob(supervisor_log_name("*")))]
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
        msg = (
            f"Could not export spans to {destination}: the collector is unreachable or rejected "
            "a batch. The exporter's log lines above give the cause."
        )
        raise GymratError(msg)
    exported = exported_span_count()
    if exported < count:
        msg = (
            f"Only {exported} of {count} spans reached the exporter for {destination}: "
            "the span queue filled up while the collector was slow to accept batches."
        )
        raise GymratError(msg, hint="Raise OTEL_BSP_MAX_QUEUE_SIZE and run the export again.")

    write_and_flush(
        sys.stderr,
        f"exported {count} spans for session {session_id} to {destination}\n",
    )
