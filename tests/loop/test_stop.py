"""Behavioral tests for ``stop_session``.

Appending a stop record to the session log, with refusals for every invalid
state (no session, finalized, unsettled iteration, gating block, already
stopped).

Every test drives the real ``stop_session`` against a throwaway repository from
the shared ``create_scratch_repo`` factory, so the suite is order-independent
and safe under ``pytest-xdist`` / ``pytest-randomly``.
"""

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.loop.finalize import finalize_session
from gymrat.loop.stop import StopResult, stop_session
from gymrat.session import (
    SessionLogRecord,
    StopRecord,
    append_record,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from tests.loop.settle._fixtures import (
    ISO_PATTERN,
    capture_error,
    confirmed_regression,
    gating_block,
    git,
    head_of,
    iteration,
    last_record_of,
    start_with,
)
from tests.session.records._fixtures import (
    committed_keep,
    stop_record,
)


def _mentions_keep_or_discard(hint: str) -> bool:
    """Whether ``hint`` points at either settling command."""
    return bool(
        re.search(r"keep", hint, re.IGNORECASE) or re.search(r"discard", hint, re.IGNORECASE)
    )


def _record_count(repo: str) -> int:
    """How many records the session log at ``repo`` currently holds."""
    return len(read_records(session_jsonl_path(repo)))


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str]) -> str:
    """A fresh scratch git repository for one stop test."""
    return create_scratch_repo()


# ---------------------------------------------------------------------------
# when the session is open and settled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "history",
    [
        pytest.param((iteration(1), committed_keep(1)), id="settled-iteration"),
        pytest.param((), id="no-iterations"),
    ],
)
def test_stop_session_when_open_does_append_a_stop_record_and_return_a_report(
    repo: str, history: tuple[SessionLogRecord, ...]
):
    start_with(repo, history)

    result = stop_session(repo, "switched to a different approach\nsecond line")

    record = last_record_of(repo)
    assert isinstance(record, StopRecord)
    assert record.message == "switched to a different approach\nsecond line"
    assert ISO_PATTERN.match(record.at)
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
    start_with(repo, (iteration(1), committed_keep(1)))
    worktree = experiment_worktree_dir(repo)
    (Path(worktree) / "step.txt").write_text("cache the regex\n", encoding="utf-8")
    git(["add", "-A"], worktree)
    git(["commit", "-m", "cache the regex"], worktree)
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
    start_with(repo, (iteration(1),))
    before = _record_count(repo)

    error = capture_error(lambda: stop_session(repo, "done"))

    assert error.hint is not None
    assert _mentions_keep_or_discard(error.hint)
    assert _record_count(repo) == before


# ---------------------------------------------------------------------------
# when a gating block stands
# ---------------------------------------------------------------------------


def test_stop_session_when_gating_block_stands_does_refuse_with_settle_hint(repo: str):
    start_with(repo, (confirmed_regression(1), gating_block(1)))
    before = _record_count(repo)

    error = capture_error(lambda: stop_session(repo, "done"))

    assert error.hint is not None
    assert _mentions_keep_or_discard(error.hint)
    assert _record_count(repo) == before


# ---------------------------------------------------------------------------
# when the log already ends on a stop
# ---------------------------------------------------------------------------


def test_stop_session_when_already_stopped_does_refuse_with_hint(repo: str):
    start_with(repo, (iteration(1), committed_keep(1)))
    append_record(session_jsonl_path(repo), stop_record())
    before = _record_count(repo)

    error = capture_error(lambda: stop_session(repo, "stop again"))

    assert "already stopped" in str(error).lower()
    assert error.hint is not None
    assert "iterate" in error.hint.lower()
    assert _mentions_keep_or_discard(error.hint)
    assert _record_count(repo) == before
