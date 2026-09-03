"""Behavioral tests for ``discard_session``.

Throwing away edits (measured or unmeasured), numbering past blocks, resetting
to baseline or kept commit, clean-worktree refusals, and the result shape on
each path.

Every test drives the real settle functions against a throwaway repository from
the shared ``create_scratch_repo`` factory, so the suite is order-independent and
safe under ``pytest-xdist`` / ``pytest-randomly``. Every git operation is real.
"""

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.loop.settle import discard_session, keep_session
from gymrat.session import (
    SessionLogRecord,
    append_record,
    baseline_worktree_dir,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from tests.loop.settle._fixtures import (
    ISO_PATTERN,
    assert_settling_record,
    checks_config,
    checks_pass,
    commit_experiment_directly,
    confirmed_regression,
    edit_experiment,
    gating_block,
    git,
    head_of,
    iteration,
    last_record_of,
    nothing_measured_block,
    start_with,
    status_of,
    undefined_delta,
    unmeasured_regression,
)
from tests.session.records._fixtures import committed_keep, discard_record


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str]) -> str:
    """A fresh scratch git repository for one settle test."""
    return create_scratch_repo()


# ---------------------------------------------------------------------------
# discard_session
# ---------------------------------------------------------------------------


def test_discard_session_when_no_session_does_refuse_pointing_at_start(repo: str):
    with pytest.raises(GymratError) as excinfo:
        discard_session(repo)

    assert excinfo.value.hint is not None
    assert "gymrat start" in excinfo.value.hint


def test_discard_session_when_unsettled_edit_does_throw_away_tracked_and_untracked(repo: str):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)

    discard_session(repo)

    worktree = experiment_worktree_dir(repo)
    assert (Path(worktree) / "README.md").read_text(encoding="utf-8") == "# Test Repo\n"
    assert not (Path(worktree) / "scratch.txt").exists()
    assert status_of(worktree) == ""


def test_discard_session_when_unsettled_edit_does_append_discard_naming_iteration(repo: str):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)

    result = discard_session(repo)

    assert result.record is not None
    assert_settling_record(result.record, discard_record(1))
    assert last_record_of(repo) == result.record
    assert result.at == result.record.at


def test_discard_session_when_primary_delta_undefined_does_record_discard(repo: str):
    start_with(repo, (undefined_delta(1),))
    edit_experiment(repo)

    result = discard_session(repo)

    assert result.record is not None
    assert result.record.seq == 1


def test_discard_session_when_worktree_clean_does_record_discard_anyway(repo: str):
    start_with(repo, (iteration(1),))

    result = discard_session(repo)

    assert last_record_of(repo) == result.record


def test_discard_session_when_gating_block_stands_does_throw_away_the_edit(repo: str):
    start_with(repo, (confirmed_regression(1), gating_block(1)))
    edit_experiment(repo)

    discard_session(repo)

    worktree = experiment_worktree_dir(repo)
    assert (Path(worktree) / "README.md").read_text(encoding="utf-8") == "# Test Repo\n"
    assert not (Path(worktree) / "scratch.txt").exists()
    assert status_of(worktree) == ""


def test_discard_session_when_gating_block_stands_does_number_discard_past_it(repo: str):
    start_with(repo, (confirmed_regression(1), gating_block(1)))
    edit_experiment(repo)

    result = discard_session(repo)

    # The block already settled iteration 1, so the discard takes the number no
    # iteration has used yet, leaving the block in history.
    assert result.record is not None
    assert_settling_record(result.record, discard_record(2))
    tail = read_records(session_jsonl_path(repo))[-2:]
    assert tail == [gating_block(1), result.record]


def test_discard_session_when_unmeasured_regression_block_stands_does_number_discard_past_it(
    repo: str,
):
    start_with(repo, (unmeasured_regression(1), gating_block(1)))
    edit_experiment(repo)

    result = discard_session(repo)

    assert status_of(experiment_worktree_dir(repo)) == ""
    assert result.record is not None
    assert_settling_record(result.record, discard_record(2))


def test_discard_session_when_gating_block_then_nothing_measured_keep_does_report_reverted_iteration(
    repo: str,
):
    start_with(repo, (confirmed_regression(1), gating_block(1), nothing_measured_block(2)))
    edit_experiment(repo)

    result = discard_session(repo)

    # The report names iteration 1 — the one whose edit was actually thrown away —
    # not the nothing-measured keep's number (2) or the discard's own seq (3).
    assert re.search(r"iteration 1\b", result.report, re.IGNORECASE)
    assert not re.search(r"iteration [23]\b", result.report, re.IGNORECASE)


async def test_discard_session_when_keep_retried_after_block_does_throw_away_standing_edit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (confirmed_regression(1), gating_block(1)))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    await keep_session(repo, checks_config())

    discard_session(repo)

    worktree = experiment_worktree_dir(repo)
    assert (Path(worktree) / "README.md").read_text(encoding="utf-8") == "# Test Repo\n"
    assert not (Path(worktree) / "scratch.txt").exists()
    assert status_of(worktree) == ""


async def test_discard_session_when_keep_retried_after_block_does_append_after_the_refusal(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (confirmed_regression(1), gating_block(1)))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    await keep_session(repo, checks_config())

    result = discard_session(repo)

    tail = read_records(session_jsonl_path(repo))[-2:]
    assert tail[0].type == "keep"
    assert tail[0].status == "blocked"
    assert tail[0].reason == "nothing-measured"
    assert tail[1] == result.record


# ---------------------------------------------------------------------------
# discard_session resets to last kept commit or baseline SHA (D6)
# ---------------------------------------------------------------------------


def test_discard_session_when_nothing_kept_and_agent_committed_does_reset_to_baseline_sha(
    repo: str,
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    commit_experiment_directly(repo)
    worktree = experiment_worktree_dir(repo)
    baseline_sha = head_of(baseline_worktree_dir(repo))
    assert head_of(worktree) != baseline_sha

    discard_session(repo)

    assert head_of(worktree) == baseline_sha
    assert status_of(worktree) == ""


async def test_discard_session_when_keep_committed_then_agent_committed_does_reset_to_kept_commit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    keep_result = await keep_session(repo, checks_config())
    kept_commit = keep_result.record.commit

    worktree = experiment_worktree_dir(repo)
    append_record(session_jsonl_path(repo), iteration(2))
    (Path(worktree) / "post-keep.txt").write_text("after keep\n", encoding="utf-8")
    git(["add", "-A"], worktree)
    git(["commit", "-m", "agent commit after keep"], worktree)
    assert head_of(worktree) != kept_commit

    discard_session(repo)

    assert head_of(worktree) == kept_commit
    assert status_of(worktree) == ""


def test_discard_session_when_resetting_does_report_the_commit_it_landed_on(
    repo: str,
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    commit_experiment_directly(repo)

    result = discard_session(repo)

    worktree = experiment_worktree_dir(repo)
    assert head_of(worktree)[:7] in result.report


# ---------------------------------------------------------------------------
# discard_session unmeasured revert (dirty worktree, nothing to settle)
# ---------------------------------------------------------------------------

NOTHING_MEASURED_HISTORIES = [
    pytest.param((), id="no-iteration-ever-recorded"),
    pytest.param((iteration(1), committed_keep(1)), id="last-iteration-already-kept"),
    pytest.param(
        (confirmed_regression(1), gating_block(1), discard_record(2)),
        id="gating-block-already-discarded",
    ),
]


@pytest.mark.parametrize("history", NOTHING_MEASURED_HISTORIES)
def test_discard_session_when_nothing_measured_and_dirty_does_revert_and_return_unmeasured_result(
    repo: str, history: tuple[SessionLogRecord, ...]
):
    start_with(repo, history)
    edit_experiment(repo)
    records_before = len(read_records(session_jsonl_path(repo)))
    baseline_sha = head_of(baseline_worktree_dir(repo))

    result = discard_session(repo)

    worktree = experiment_worktree_dir(repo)
    assert (Path(worktree) / "README.md").read_text(encoding="utf-8") == "# Test Repo\n"
    assert not (Path(worktree) / "scratch.txt").exists()
    assert status_of(worktree) == ""
    assert len(read_records(session_jsonl_path(repo))) == records_before
    assert result.record is None
    assert ISO_PATTERN.match(result.at)
    assert (
        result.report
        == f"Reverted 2 unmeasured edits: the experiment worktree is back at {baseline_sha[:7]}"
    )


def test_discard_session_when_nothing_measured_and_agent_committed_does_report_reverted_edit_count(
    repo: str,
):
    start_with(repo, ())
    edit_experiment(repo)
    commit_experiment_directly(repo)
    worktree = experiment_worktree_dir(repo)
    (Path(worktree) / "extra.txt").write_text("more\n", encoding="utf-8")

    result = discard_session(repo)

    baseline_sha = head_of(baseline_worktree_dir(repo))
    assert (
        result.report
        == f"Reverted 3 unmeasured edits: the experiment worktree is back at {baseline_sha[:7]}"
    )


# ---------------------------------------------------------------------------
# discard_session when nothing was measured and worktree is clean (refusal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("history", NOTHING_MEASURED_HISTORIES)
def test_discard_session_when_nothing_measured_and_clean_does_refuse(
    repo: str, history: tuple[SessionLogRecord, ...]
):
    start_with(repo, history)
    before = len(read_records(session_jsonl_path(repo)))

    with pytest.raises(GymratError, match="Discard refused"):
        discard_session(repo)

    assert len(read_records(session_jsonl_path(repo))) == before
