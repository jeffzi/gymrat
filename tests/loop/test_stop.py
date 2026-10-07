"""Behavioral tests for ``stop_session``.

Appending a stop record to the session log, with refusals for an unsettled
iteration, a gating block, and an already-stopped session. The no-session and
finalized refusals come from the shared session guard, pinned for every loop
command in ``test_loop_integration``.

Every test drives the real ``stop_session`` against a throwaway repository from
the shared ``create_scratch_repo`` factory, so the suite is order-independent
and safe under ``pytest-xdist`` / ``pytest-randomly``.
"""

import pytest

from gymrat.loop.stop import StopResult, stop_session
from gymrat.session.records import KeepChecks, SessionLogRecord, StopRecord
from tests.loop._settle import (
    capture_error,
    confirmed_regression,
    settling_record_of,
    start_with,
)
from tests.session.records._fixtures import (
    blocked_keep,
    committed_keep,
    iteration_record,
    log_records,
    stop_record,
)


def _record_count(repo: str) -> int:
    """How many records the session log at ``repo`` currently holds."""
    return len(log_records(repo))


# ---------------------------------------------------------------------------
# when the session is open and settled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "history",
    [
        pytest.param((iteration_record(seq=1), committed_keep(1)), id="settled-iteration"),
        pytest.param((), id="no-iterations"),
    ],
)
def test_stop_session_when_open_does_append_a_stop_record_and_return_a_report(
    repo: str, history: tuple[SessionLogRecord, ...]
):
    start_with(repo, history)

    result = stop_session(repo, "switched to a different approach\nsecond line")

    record = settling_record_of(repo)
    assert isinstance(record, StopRecord)
    assert record.message == "switched to a different approach\nsecond line"
    assert record.at > 0
    assert isinstance(result, StopResult)
    assert result.report == "Stopped: switched to a different approach"


# ---------------------------------------------------------------------------
# when the last iteration is unsettled
# ---------------------------------------------------------------------------


_SETTLE_FIRST = "Run gymrat keep or gymrat discard before stopping."


@pytest.mark.parametrize(
    ("history", "message", "hint", "reason"),
    [
        pytest.param(
            (iteration_record(seq=1),),
            "Iteration 1 has not been settled",
            _SETTLE_FIRST,
            "unsettled",
            id="last-iteration-unsettled",
        ),
        pytest.param(
            (
                confirmed_regression(1),
                blocked_keep(1, reason="gating-regression", checks=KeepChecks(configured=True)),
            ),
            "Iteration 1 is blocked by a gating regression",
            _SETTLE_FIRST,
            "gating-block",
            id="gating-block-stands",
        ),
        pytest.param(
            (iteration_record(seq=1), committed_keep(1), stop_record()),
            "Already stopped",
            "Run iterate, keep, or discard to continue.",
            "already-stopped",
            id="already-stopped",
        ),
    ],
)
def test_stop_session_when_nothing_is_left_to_stop_cleanly_does_refuse_writing_no_record(
    repo: str, history: tuple[SessionLogRecord, ...], message: str, hint: str, reason: str
):
    start_with(repo, history)
    before = _record_count(repo)

    error = capture_error(lambda: stop_session(repo, "done"))

    assert (str(error), error.hint, error.reason) == (message, hint, reason)
    assert _record_count(repo) == before
