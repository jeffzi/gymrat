"""A local OTLP/HTTP trace collector for tests that export real spans.

``otlp_collector`` serves on a free loopback port in a background thread and
records every export request it receives, so a test can drive the real OTLP
exporter end to end and check where spans went, and what happens when the
collector rejects a batch.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, override

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

_OK = 200


@dataclass
class ReceivedExport:
    """One export request the collector received."""

    path: str
    span_names: list[str]


@dataclass
class OtlpCollector:
    """Handle on a running local collector."""

    endpoint: str
    received: list[ReceivedExport] = field(default_factory=list)

    @property
    def span_names(self) -> list[str]:
        """Names of every span received, in arrival order."""
        return [name for export in self.received for name in export.span_names]


def _span_names(body: bytes) -> list[str]:
    request = ExportTraceServiceRequest()
    request.ParseFromString(body)
    return [
        span.name
        for resource_spans in request.resource_spans
        for scope_spans in resource_spans.scope_spans
        for span in scope_spans.spans
    ]


@contextmanager
def otlp_collector(statuses: Iterable[int] = ()) -> Generator[OtlpCollector]:
    """Serve OTLP/HTTP traces on a free loopback port for the duration of the block.

    Args:
        statuses: HTTP statuses to answer the first export requests with, in
            order. Every request past the end of ``statuses`` gets ``200``.

    Yields:
        The collector, whose ``endpoint`` is the base URL to export to.
    """
    replies = list(statuses)
    lock = threading.Lock()
    received: list[ReceivedExport] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with lock:
                received.append(ReceivedExport(path=self.path, span_names=_span_names(body)))
                status = replies.pop(0) if replies else _OK
            self.send_response(status)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", "0")
            self.end_headers()

        @override
        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    collector = OtlpCollector(endpoint=f"http://{host!s}:{port}", received=received)
    try:
        yield collector
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
