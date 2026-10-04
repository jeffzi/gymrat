"""Behavioral tests for ``stop_session``.

Appending a stop record to the session log, with refusals for every invalid
state (no session, finalized, unsettled iteration, gating block, already
stopped).

Every test drives the real ``stop_session`` against a throwaway repository from
the shared ``create_scratch_repo`` factory, so the suite is order-independent
and safe under ``pytest-xdist`` / ``pytest-randomly``.
"""

import re
from pathlib import Path

import pytest

from gymrat.loop.finalize import finalize_session
from gymrat.loop.stop import StopResult, stop_session
from gymrat.session.paths import experiment_worktree_dir, session_jsonl_path
from gymrat.session.records import KeepChecks, SessionLogRecord, StopRecord
from gymrat.session.store import append_record
from tests._git import head_of, run_git
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


def _mentions_keep_or_discard(hint: str) -> bool:
    """Whether ``hint`` points at either settling command."""
    return bool(
        re.search(r"keep", hint, re.IGNORECASE) or re.search(r"discard", hint, re.IGNORECASE)
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
    assert isinstance(record.at, int)
    assert record.at > 0
    assert isinstance(result, StopResult)
    assert "Stopped" in result.report
    assert "switched to a different approach" in result.report


# ---------------------------------------------------------------------------
# when no session is open
# ---------------------------------------------------------------------------


def test_stop_session_when_no_session_does_refuse_pointing_at_the_command_that_opens_one(
    repo: str,
):
    error = capture_error(lambda: stop_session(repo, "done"))

    assert error.hint is not None
    assert "gymrat start" in error.hint


# ---------------------------------------------------------------------------
# when the session is finalized
# ---------------------------------------------------------------------------


def test_stop_session_when_finalized_does_refuse(repo: str):
    start_with(repo, (iteration_record(seq=1), committed_keep(1)))
    worktree = experiment_worktree_dir(repo)
    (Path(worktree) / "step.txt").write_text("cache the regex\n", encoding="utf-8")
    run_git(["add", "-A"], worktree)
    run_git(["commit", "-m", "cache the regex"], worktree)
    commit = head_of(worktree)
    append_record(session_jsonl_path(repo), committed_keep(1, commit=commit))
    finalize_session(repo)

    before = _record_count(repo)

    error = capture_error(lambda: stop_session(repo, "too late"))

    assert "finalized" in str(error)
    assert error.hint is not None
    assert "gymrat start" in error.hint
    assert _record_count(repo) == before


# ---------------------------------------------------------------------------
# when the last iteration is unsettled
# ---------------------------------------------------------------------------


def test_stop_session_when_last_iteration_unsettled_does_refuse_naming_settle_hint(repo: str):
    start_with(repo, (iteration_record(seq=1),))
    before = _record_count(repo)

    error = capture_error(lambda: stop_session(repo, "done"))

    assert str(error) == "Iteration 1 has not been settled"
    assert error.hint == "Run gymrat keep or gymrat discard before stopping."
    assert _record_count(repo) == before


def test_stop_session_when_last_iteration_unsettled_does_carry_unsettled_reason(repo: str):
    start_with(repo, (iteration_record(seq=1),))

    error = capture_error(lambda: stop_session(repo, "done"))

    assert error.reason == "unsettled"


# ---------------------------------------------------------------------------
# when a gating block stands
# ---------------------------------------------------------------------------


def test_stop_session_when_gating_block_stands_does_refuse_with_settle_hint(repo: str):
    start_with(
        repo,
        (
            confirmed_regression(1),
            blocked_keep(1, reason="gating-regression", checks=KeepChecks(configured=True)),
        ),
    )
    before = _record_count(repo)

    error = capture_error(lambda: stop_session(repo, "done"))

    assert error.hint is not None
    assert _mentions_keep_or_discard(error.hint)
    assert _record_count(repo) == before


def test_stop_session_when_gating_block_stands_does_carry_gating_block_reason(repo: str):
    start_with(
        repo,
        (
            confirmed_regression(1),
            blocked_keep(1, reason="gating-regression", checks=KeepChecks(configured=True)),
        ),
    )

    error = capture_error(lambda: stop_session(repo, "done"))

    assert error.reason == "gating-block"


# ---------------------------------------------------------------------------
# when the log already ends on a stop
# ---------------------------------------------------------------------------


def test_stop_session_when_already_stopped_does_refuse_with_hint(repo: str):
    start_with(repo, (iteration_record(seq=1), committed_keep(1)))
    append_record(session_jsonl_path(repo), stop_record())
    before = _record_count(repo)

    error = capture_error(lambda: stop_session(repo, "stop again"))

    assert "already stopped" in str(error).lower()
    assert error.hint == "Run iterate, keep, or discard to continue."
    assert _record_count(repo) == before


def test_stop_session_when_already_stopped_does_carry_already_stopped_reason(repo: str):
    start_with(repo, (iteration_record(seq=1), committed_keep(1)))
    append_record(session_jsonl_path(repo), stop_record())

    error = capture_error(lambda: stop_session(repo, "stop again"))

    assert error.reason == "already-stopped"
