"""Behavioral tests for ``stop_session``.

Appending a stop record to the session log, with refusals for an unsettled
iteration, a gating block, and an already-stopped session. The no-session and
finalized refusals come from the shared session guard, pinned in
``tests/session/test_store.py``.

Every test drives the real ``stop_session`` against a throwaway repository from
the shared ``create_scratch_repo`` factory, so the suite is order-independent
and safe under ``pytest-xdist`` / ``pytest-randomly``.
"""

import pytest

from gymrat.loop.stop import stop_session
from gymrat.session.records import SessionLogRecord, StopRecord
from tests.loop._settle import (
    capture_error,
    gating_blocked_history,
    settling_record_of,
    start_with,
)
from tests.session.records._fixtures import (
    iteration_record,
    log_records,
    settled_history,
    stop_record,
)


def _record_count(repo: str) -> int:
    """How many records the session log at ``repo`` currently holds."""
    return len(log_records(repo))


# ---------------------------------------------------------------------------
# when the session is open and settled
# ---------------------------------------------------------------------------


def test_stop_session_when_open_does_append_a_stop_record(repo: str):
    start_with(repo, settled_history())

    stop_session(repo, "switched to a different approach\nsecond line")

    record = settling_record_of(repo)
    assert isinstance(record, StopRecord)
    assert record.message == "switched to a different approach\nsecond line"


def test_stop_session_when_message_multiline_does_report_only_its_first_line(repo: str):
    start_with(repo, settled_history())

    result = stop_session(repo, "switched to a different approach\nsecond line")

    assert "switched to a different approach" in result.report
    assert "second line" not in result.report


# ---------------------------------------------------------------------------
# when the session cannot be stopped: unsettled, blocked, or already stopped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("history", "reason"),
    [
        pytest.param(gating_blocked_history(), "gating-block", id="gating-block-stands"),
        pytest.param((*settled_history(), stop_record()), "already-stopped", id="already-stopped"),
        pytest.param((iteration_record(seq=1),), "unsettled", id="last-iteration-unsettled"),
    ],
)
def test_stop_session_when_session_cannot_be_stopped_does_refuse_writing_no_record(
    repo: str, history: tuple[SessionLogRecord, ...], reason: str
):
    start_with(repo, history)
    before = _record_count(repo)

    error = capture_error(lambda: stop_session(repo, "done"))

    assert error.reason == reason
    assert _record_count(repo) == before
