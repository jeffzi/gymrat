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
from gymrat.loop.discard import discard_session
from gymrat.loop.keep import keep_session
from gymrat.session.paths import baseline_worktree_dir, experiment_worktree_dir
from gymrat.session.records import KeepRecord, SessionLogRecord
from tests._git import commit_all, head_of, status_of
from tests.loop._settle import (
    NOTHING_MEASURED_HISTORIES,
    assert_settling_record,
    checks_config,
    checks_pass,
    commit_and_keep,
    commit_experiment_directly,
    edit_experiment,
    gating_blocked_history,
    settling_record_of,
    start_with,
)
from tests.session.records._fixtures import (
    COMMIT,
    append_records,
    committed_keep,
    discard_record,
    iteration_record,
    log_records,
)

# ---------------------------------------------------------------------------
# discard_session
# ---------------------------------------------------------------------------


def _assert_reverted(worktree: str) -> None:
    """Assert the experiment worktree is back at the committed tree, edits and untracked files gone."""
    assert (Path(worktree) / "README.md").read_text(encoding="utf-8") == "# Test Repo\n"
    assert not (Path(worktree) / "scratch.txt").exists()
    assert status_of(worktree) == ""


def _leave_worktree_clean(repo_dir: str) -> None:
    """Leave the experiment worktree untouched."""


def _edit_and_commit(repo_dir: str) -> None:
    """Edit the experiment worktree, then commit the edit as the agent would."""
    edit_experiment(repo_dir)
    commit_experiment_directly(repo_dir)


@pytest.mark.parametrize(
    "arrange_worktree",
    [
        pytest.param(edit_experiment, id="dirty-edit"),
        pytest.param(_leave_worktree_clean, id="clean-worktree"),
        pytest.param(_edit_and_commit, id="agent-committed-edit"),
    ],
)
def test_discard_session_when_iteration_unsettled_does_settle_it_as_discarded(
    repo: str, arrange_worktree: Callable[[str], None]
):
    start_with(repo, (iteration_record(seq=1),))
    arrange_worktree(repo)
    baseline_sha = head_of(baseline_worktree_dir(repo))

    result = discard_session(repo)

    _assert_reverted(experiment_worktree_dir(repo))
    assert head_of(experiment_worktree_dir(repo)) == baseline_sha
    assert baseline_sha[:7] in result.report
    assert result.record is not None
    assert_settling_record(result.record, discard_record(1))
    assert settling_record_of(repo) == result.record
    assert result.at == result.record.at


def test_discard_session_when_gating_block_stands_does_throw_away_the_edit_numbering_past_it(
    repo: str,
):
    history = gating_blocked_history()
    start_with(repo, history)
    edit_experiment(repo)

    result = discard_session(repo)

    _assert_reverted(experiment_worktree_dir(repo))
    # The block already settled iteration 1, so the discard takes the number no
    # iteration has used yet, leaving the block in history.
    assert result.record is not None
    assert_settling_record(result.record, discard_record(2))
    assert log_records(repo)[-2:] == [history[-1], result.record]


async def test_discard_session_when_keep_retried_after_block_does_throw_away_the_edit_after_the_refusal(
    repo: str,
):
    start_with(repo, gating_blocked_history())
    edit_experiment(repo)
    await keep_session(repo, checks_config())

    result = discard_session(repo)

    _assert_reverted(experiment_worktree_dir(repo))
    refusal, discard = log_records(repo)[-2:]
    assert isinstance(refusal, KeepRecord)
    assert (refusal.status, refusal.reason) == ("blocked", "nothing-measured")
    assert discard == result.record
    # The report names iteration 1 — the one whose edit was actually thrown away —
    # not the nothing-measured keep's number (2) or the discard's own seq (3).
    assert re.search(r"iteration 1\b", result.report, re.IGNORECASE)
    assert not re.search(r"iteration [23]\b", result.report, re.IGNORECASE)


# ---------------------------------------------------------------------------
# discard_session resets to last kept commit or baseline SHA
# ---------------------------------------------------------------------------


async def test_discard_session_when_keep_committed_then_agent_committed_does_reset_to_kept_commit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    keep_result = await keep_session(repo, checks_config())
    kept_commit = keep_result.record.commit
    worktree = experiment_worktree_dir(repo)
    append_records(repo, iteration_record(seq=2))
    commit_all(worktree, "agent commit after keep", file="post-keep.txt", content="after keep\n")

    discard_session(repo)

    assert head_of(worktree) == kept_commit
    assert status_of(worktree) == ""


def test_discard_session_when_kept_commit_unreachable_does_fall_back_to_the_worktree_head_with_a_warning(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    # The keep names COMMIT, a commit that exists nowhere in the repository.
    start_with(repo, (iteration_record(seq=1), committed_keep(1)))
    edit_experiment(repo)
    worktree = experiment_worktree_dir(repo)
    head = head_of(worktree)

    result = discard_session(repo)

    stderr = capsys.readouterr().err
    assert f"Commit {COMMIT} is unreachable in the experiment worktree: " in stderr
    assert "reverting to the worktree's HEAD instead" in stderr
    _assert_reverted(worktree)
    assert head_of(worktree) == head
    assert result.record is None


# ---------------------------------------------------------------------------
# discard_session unmeasured revert (dirty worktree, nothing to settle)
# ---------------------------------------------------------------------------


def _edit_after_no_iteration(repo_dir: str) -> str:
    """Edit a session that never recorded an iteration, returning the baseline it reverts to."""
    start_with(repo_dir)
    edit_experiment(repo_dir)
    return head_of(baseline_worktree_dir(repo_dir))


def _edit_after_a_real_keep(repo_dir: str) -> str:
    """Edit a session whose last iteration was kept onto a real commit, returning that commit."""
    start_with(repo_dir)
    kept = commit_and_keep(repo_dir, 1, "cache the regex")
    edit_experiment(repo_dir)
    return kept


def _edit_after_a_discarded_block(repo_dir: str) -> str:
    """Edit a session whose gating block was already discarded, returning the baseline."""
    start_with(repo_dir, (*gating_blocked_history(), discard_record(2)))
    edit_experiment(repo_dir)
    return head_of(baseline_worktree_dir(repo_dir))


def _commit_and_add_a_file(repo_dir: str) -> str:
    """Commit an edit as the agent would, then leave a fresh untracked file beside it."""
    start_with(repo_dir)
    _edit_and_commit(repo_dir)
    (Path(experiment_worktree_dir(repo_dir)) / "extra.txt").write_text("more\n", encoding="utf-8")
    return head_of(baseline_worktree_dir(repo_dir))


@pytest.mark.parametrize(
    ("arrange", "reverted"),
    [
        pytest.param(_edit_after_no_iteration, 2, id="no-iteration-ever-recorded"),
        pytest.param(_edit_after_a_real_keep, 2, id="last-iteration-already-kept"),
        pytest.param(_edit_after_a_discarded_block, 2, id="gating-block-already-discarded"),
        pytest.param(_commit_and_add_a_file, 3, id="agent-committed-plus-untracked"),
    ],
)
def test_discard_session_when_nothing_measured_and_dirty_does_revert_without_recording(
    repo: str, arrange: Callable[[str], str], reverted: int
):
    target = arrange(repo)
    records_before = len(log_records(repo))

    result = discard_session(repo)

    _assert_reverted(experiment_worktree_dir(repo))
    assert head_of(experiment_worktree_dir(repo)) == target
    assert len(log_records(repo)) == records_before
    assert result.record is None
    assert isinstance(result.at, int)
    assert result.at > 0
    assert result.report == (
        f"Reverted {reverted} unmeasured edits: the experiment worktree is back at {target[:7]}"
    )


# ---------------------------------------------------------------------------
# discard_session when nothing was measured and worktree is clean (refusal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("history", NOTHING_MEASURED_HISTORIES)
def test_discard_session_when_nothing_measured_and_clean_does_refuse(
    repo: str, history: tuple[SessionLogRecord, ...]
):
    start_with(repo, history)
    before = len(log_records(repo))

    with pytest.raises(GymratError) as excinfo:
        discard_session(repo)

    assert "Discard refused" in str(excinfo.value)
    assert excinfo.value.hint == "Run iterate to measure an edit before settling it."
    assert excinfo.value.reason == "nothing-to-discard"
    assert len(log_records(repo)) == before


def test_discard_session_when_session_id_mismatches_does_carry_stale_session_reason(repo: str):
    start_with(repo, (iteration_record(seq=1),))

    with pytest.raises(GymratError) as excinfo:
        discard_session(repo, expected_session_id="wrong-id")

    assert excinfo.value.reason == "stale-session"
