"""Shared session-seeding and tracing-isolation fixtures for the lock test files.

``test_lock.py`` and ``test_lock_tracing.py`` both exercise ``with_repo_lock``
and need the same starting point: a session log seeded with a header record,
and a telemetry provider reset between tests. Both modules import from here
instead of duplicating the definitions.
"""

import warnings
from collections.abc import Iterator

import pytest

from gymrat.session import SessionRecord
from tests.session.records._fixtures import session_record, write_session_log

__all__ = [
    "isolate_tracing_provider",
    "seeded_session",
    "seeded_session_no_trace_context",
]


@pytest.fixture(autouse=True)
def isolate_tracing_provider() -> Iterator[None]:
    """Reset the telemetry provider singleton between tests."""
    yield
    from gymrat.telemetry.provider import _reset_for_tests

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _reset_for_tests()


def seeded_session(repo: str, monkeypatch: pytest.MonkeyPatch) -> SessionRecord:
    """Write a session header and clear TRACEPARENT so tests start without ambient trace context."""
    header = session_record()
    write_session_log(repo, header)
    monkeypatch.delenv("TRACEPARENT", raising=False)
    return header


def seeded_session_no_trace_context(repo: str, monkeypatch: pytest.MonkeyPatch) -> SessionRecord:
    """Like :func:`seeded_session`, also clearing GYMRAT_TRACEPARENT."""
    header = seeded_session(repo, monkeypatch)
    monkeypatch.delenv("GYMRAT_TRACEPARENT", raising=False)
    return header
