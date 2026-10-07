"""Command-span export tests for :mod:`gymrat.command_run`.

These cover the spans ``with_repo_lock`` emits when tracing is enabled: span
identity, attributes, status, events, trace-context parenting and linking, and
the flush before return. They share the session-seeding helpers with
``test_command_run``.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from gymrat.command_run import CommandTrace, with_repo_lock
from gymrat.errors import GymratError
from gymrat.session.paths import archived_session_path, repo_root, session_jsonl_path
from gymrat.session.records import CommandRecord, SessionRecord
from gymrat.session.store import append_record, session_header
from tests._command_run_fixtures import ok_body as _ok_body
from tests._command_run_fixtures import seeded_session as _seeded_session
from tests.session.records._fixtures import (
    baseline_record,
    iteration_record,
    log_records,
    session_record,
    write_session_log,
)
from tests.telemetry._fixtures import (
    isolate_tracing_provider as _isolate_tracing_provider,  # noqa: F401 -- registers the autouse fixture
)

#: The id of the session a ``start`` opens over the previous one.
_FRESH_SESSION_ID = "20260809-090000-b7e4"

# ---------------------------------------------------------------------------
# with_repo_lock — command span export (tracing enabled)
# ---------------------------------------------------------------------------


def _command_span(
    exporter: InMemorySpanExporter, name: str = "gymrat.command.measure"
) -> ReadableSpan:
    """Return the single finished span with the given name."""
    return next(s for s in exporter.get_finished_spans() if s.name == name)


async def test_with_repo_lock_when_tracing_enabled_does_export_command_span(
    repo: str,
):
    from gymrat.telemetry.provider import trace_id_of
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    spans = exporter.get_finished_spans()
    command_spans = [s for s in spans if s.name == "gymrat.command.measure"]
    assert len(command_spans) == 1
    assert command_spans[0].context.trace_id == trace_id_of(header.session_id)  # pyrefly: ignore[missing-attribute]


async def test_with_repo_lock_when_tracing_enabled_does_use_deterministic_span_id(
    repo: str,
):
    from gymrat.telemetry.provider import span_id_of
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    records = log_records(repo_root())
    cmd_line = len(records)
    assert command_span.context.span_id == span_id_of(header.session_id, f"command:{cmd_line}")  # pyrefly: ignore[missing-attribute]


async def test_with_repo_lock_when_tracing_enabled_does_set_command_attributes(
    repo: str,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body, args={"samples": 5})

    command_span = _command_span(exporter)
    attrs = dict(command_span.attributes or {})
    assert attrs["gymrat.session.id"] == header.session_id
    assert attrs["gymrat.command.name"] == "measure"
    assert attrs["gymrat.command.exit_code"] == 0
    assert attrs["gymrat.command.args.samples"] == 5


async def test_with_repo_lock_when_exit_zero_does_set_span_status_ok(
    repo: str,
):
    from opentelemetry.trace import StatusCode

    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert command_span.status.status_code == StatusCode.OK


async def test_with_repo_lock_when_exit_two_does_set_span_status_error_with_reason(
    repo: str,
):
    from opentelemetry.trace import StatusCode

    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:

        async def body(trace: CommandTrace) -> str:
            msg = "boom"
            raise GymratError(msg, reason="no-session")

        with pytest.raises(GymratError):
            await with_repo_lock("measure", body)

    command_span = _command_span(exporter)
    assert command_span.status.status_code == StatusCode.ERROR
    assert command_span.status.description == "no-session"


async def test_with_repo_lock_when_exit_one_does_set_span_status_unset(
    repo: str,
):
    from opentelemetry.trace import StatusCode

    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:

        async def body(trace: CommandTrace) -> str:
            trace.gate = True
            trace.reason = "gating-regression"
            return "gated"

        await with_repo_lock("compare", body)

    command_span = _command_span(exporter, "gymrat.command.compare")
    assert command_span.status.status_code == StatusCode.UNSET


async def test_with_repo_lock_when_body_appends_records_does_add_span_events(
    repo: str,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:

        async def body(trace: CommandTrace) -> str:
            jsonl_path = session_jsonl_path(repo_root())
            iteration = iteration_record()
            append_record(jsonl_path, iteration)
            return "ok"

        await with_repo_lock("iterate", body)

    command_span = _command_span(exporter, "gymrat.command.iterate")
    event_names = [e.name for e in command_span.events]
    assert "gymrat.iteration" in event_names


async def test_with_repo_lock_when_gymrat_traceparent_set_does_use_as_parent(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)
    valid_traceparent = "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01"
    monkeypatch.setenv("GYMRAT_TRACEPARENT", valid_traceparent)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert command_span.parent is not None
    assert command_span.parent.span_id == 0x1112131415161718


async def test_with_repo_lock_when_gymrat_traceparent_absent_does_parent_under_session(
    repo: str,
):
    from gymrat.telemetry.provider import span_id_of, trace_id_of
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert command_span.parent is not None
    assert command_span.context.trace_id == trace_id_of(header.session_id)  # pyrefly: ignore[missing-attribute]
    assert command_span.parent.span_id == span_id_of(header.session_id, "session")


async def test_with_repo_lock_when_gymrat_traceparent_malformed_does_parent_under_session(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from gymrat.telemetry.provider import span_id_of, trace_id_of
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)
    monkeypatch.setenv("GYMRAT_TRACEPARENT", "not-a-valid-traceparent")

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert command_span.parent is not None
    assert command_span.context.trace_id == trace_id_of(header.session_id)  # pyrefly: ignore[missing-attribute]
    assert command_span.parent.span_id == span_id_of(header.session_id, "session")


async def test_with_repo_lock_when_traceparent_valid_does_add_link(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)
    valid_traceparent = "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01"
    monkeypatch.setenv("TRACEPARENT", valid_traceparent)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert len(command_span.links) == 1
    assert command_span.links[0].context.span_id == 0x1112131415161718


async def test_with_repo_lock_when_traceparent_absent_does_not_add_link(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)
    monkeypatch.delenv("TRACEPARENT", raising=False)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert len(command_span.links) == 0


async def test_with_repo_lock_when_traceparent_malformed_does_not_add_link(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)
    monkeypatch.setenv("TRACEPARENT", "not-valid")

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert len(command_span.links) == 0


async def test_with_repo_lock_when_otlp_exporter_missing_does_run_body_without_tracing(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    from tests.telemetry._collector import otlp_collector
    from tests.telemetry._fixtures import hide_otlp_exporter

    _seeded_session(repo)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    hide_otlp_exporter(monkeypatch)

    with otlp_collector() as collector:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint)
        result = await with_repo_lock("measure", _ok_body)

    records = log_records(repo_root())
    assert result == "ok"
    assert isinstance(records[-1], CommandRecord)
    assert records[-1].name == "measure"
    assert collector.received == []
    assert capsys.readouterr().err == ""


async def test_with_repo_lock_when_tracing_enabled_does_flush_before_return(
    repo: str,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id, buffered=True) as exporter:
        await with_repo_lock("measure", _ok_body)

        spans_before_exit = exporter.get_finished_spans()

    assert any(s.name == "gymrat.command.measure" for s in spans_before_exit)


# ---------------------------------------------------------------------------
# with_repo_lock — a start that opens a new session log
# ---------------------------------------------------------------------------


def _replace_session_log(previous: SessionRecord, fresh: SessionRecord) -> None:
    """Archive the open session log and write a fresh one holding ``fresh`` and a baseline."""
    root = repo_root()
    archive = Path(archived_session_path(root, previous.session_id))
    archive.parent.mkdir(parents=True, exist_ok=True)
    Path(session_jsonl_path(root)).rename(archive)
    write_session_log(root, fresh, (baseline_record(),))


def _start_body(previous: SessionRecord) -> Callable[[CommandTrace], Awaitable[str]]:
    """A ``start`` body that replaces ``previous``'s log with a fresh session's."""

    async def body(trace: CommandTrace) -> str:
        _replace_session_log(previous, session_record(session_id=_FRESH_SESSION_ID))
        return "started"

    return body


def _seeded_long_session(repo: str) -> SessionRecord:
    """Write a session header followed by three iterations, longer than a fresh session's log."""
    header = session_record()
    write_session_log(repo, header, tuple(iteration_record(seq=seq) for seq in (1, 2, 3)))
    return header


def _start_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    """Every finished ``start`` command span."""
    return [s for s in exporter.get_finished_spans() if s.name == "gymrat.command.start"]


async def test_with_repo_lock_when_start_opens_a_new_session_does_key_span_on_the_log_holding_the_record(
    repo: str,
):
    from gymrat.telemetry.provider import span_id_of, trace_id_of
    from tests.telemetry._fixtures import memory_tracing

    previous = _seeded_long_session(repo)

    with memory_tracing(_FRESH_SESSION_ID) as exporter:
        await with_repo_lock("start", _start_body(previous))

    records = log_records(repo_root())
    assert isinstance(records[-1], CommandRecord)
    holder = session_header(repo_root())
    assert holder is not None
    spans = _start_spans(exporter)
    assert len(spans) == 1
    span = spans[0]
    assert dict(span.attributes or {})["gymrat.session.id"] == holder.session_id
    assert span.context.trace_id == trace_id_of(holder.session_id)  # pyrefly: ignore[missing-attribute]
    assert span.context.span_id == span_id_of(holder.session_id, f"command:{len(records)}")  # pyrefly: ignore[missing-attribute]


async def test_with_repo_lock_when_start_opens_the_first_session_does_key_span_on_it(
    repo: str,
):
    from gymrat.telemetry.provider import span_id_of, trace_id_of
    from tests.telemetry._fixtures import memory_tracing

    async def first_start(trace: CommandTrace) -> str:
        write_session_log(repo_root(), session_record(session_id=_FRESH_SESSION_ID))
        return "started"

    with memory_tracing(_FRESH_SESSION_ID) as exporter:
        await with_repo_lock("start", first_start)

    records = log_records(repo_root())
    spans = _start_spans(exporter)
    assert [dict(span.attributes or {})["gymrat.session.id"] for span in spans] == [
        _FRESH_SESSION_ID
    ]
    assert spans[0].context.trace_id == trace_id_of(_FRESH_SESSION_ID)  # pyrefly: ignore[missing-attribute]
    assert spans[0].context.span_id == span_id_of(_FRESH_SESSION_ID, f"command:{len(records)}")  # pyrefly: ignore[missing-attribute]


async def test_with_repo_lock_when_start_opens_a_new_session_under_gymrat_traceparent_does_parent_on_it(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from gymrat.telemetry.provider import span_id_of
    from tests.telemetry._fixtures import memory_tracing

    previous = _seeded_long_session(repo)
    env_trace_id = 0x0102030405060708090A0B0C0D0E0F10
    env_span_id = 0x1112131415161718
    monkeypatch.setenv("GYMRAT_TRACEPARENT", f"00-{env_trace_id:032x}-{env_span_id:016x}-01")

    with memory_tracing(_FRESH_SESSION_ID) as exporter:
        await with_repo_lock("start", _start_body(previous))

    records = log_records(repo_root())
    (span,) = _start_spans(exporter)
    assert span.parent is not None
    assert (span.parent.trace_id, span.parent.span_id) == (env_trace_id, env_span_id)
    assert span.context.trace_id == env_trace_id  # pyrefly: ignore[missing-attribute]
    assert dict(span.attributes or {})["gymrat.session.id"] == _FRESH_SESSION_ID
    assert span.context.span_id == span_id_of(_FRESH_SESSION_ID, f"command:{len(records)}")  # pyrefly: ignore[missing-attribute]


async def test_with_repo_lock_when_start_opens_a_new_session_does_add_that_logs_records_as_events(
    repo: str,
):
    from gymrat.telemetry.provider import record_event
    from tests.telemetry._fixtures import memory_tracing

    previous = _seeded_long_session(repo)

    with memory_tracing(_FRESH_SESSION_ID) as exporter:
        await with_repo_lock("start", _start_body(previous))

    (span,) = _start_spans(exporter)
    event_names = [e.name for e in span.events]
    baseline_event, _attrs = record_event(baseline_record())
    iteration_event, _attrs = record_event(iteration_record())
    assert baseline_event in event_names
    assert iteration_event not in event_names


async def test_with_repo_lock_when_start_cannot_append_its_record_does_emit_no_command_span(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from tests.telemetry._fixtures import memory_tracing

    previous = _seeded_long_session(repo)

    def broken_append(path: str, record: object) -> None:
        msg = "disk full"
        raise GymratError(msg)

    monkeypatch.setattr("gymrat.command_run.append_record", broken_append)

    with memory_tracing(_FRESH_SESSION_ID) as exporter:
        await with_repo_lock("start", _start_body(previous))

    assert _start_spans(exporter) == []


# ---------------------------------------------------------------------------
# with_repo_lock — command span export to a real collector
# ---------------------------------------------------------------------------


async def test_with_repo_lock_when_endpoint_padded_does_export_command_span_to_trimmed_endpoint(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from tests.telemetry._collector import otlp_collector

    _seeded_session(repo)

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    with otlp_collector() as collector:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"  {collector.endpoint} ")
        await with_repo_lock("measure", _ok_body)

    assert {export.path for export in collector.received} == {"/v1/traces"}
    assert "gymrat.command.measure" in collector.span_names


async def test_with_repo_lock_when_collector_rejects_the_export_does_return_body_result(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from gymrat.telemetry.provider import export_failed
    from tests.telemetry._collector import otlp_collector

    _seeded_session(repo)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

    with otlp_collector(statuses=[400]) as collector:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint)
        result = await with_repo_lock("measure", _ok_body)

    assert (result, export_failed(), "gymrat.command.measure" in collector.span_names) == (
        "ok",
        True,
        True,
    )


async def test_with_repo_lock_when_first_start_traced_only_by_endpoint_does_export_its_span_keyed_on_the_new_session(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from tests.telemetry._collector import otlp_collector

    async def first_start(trace: CommandTrace) -> str:
        write_session_log(repo_root(), session_record(session_id=_FRESH_SESSION_ID))
        return "started"

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    with otlp_collector() as collector:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint)
        await with_repo_lock("start", first_start)

    start_sessions = [
        span.attributes.get("gymrat.session.id")
        for span in collector.spans
        if span.name == "gymrat.command.start"
    ]
    assert start_sessions == [_FRESH_SESSION_ID]


async def test_with_repo_lock_when_endpoint_whitespace_only_does_export_nothing(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from tests.telemetry._collector import otlp_collector

    _seeded_session(repo)

    with otlp_collector() as collector:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", " \t ")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"{collector.endpoint}/v1/traces")
        await with_repo_lock("measure", _ok_body)

    assert collector.received == []
