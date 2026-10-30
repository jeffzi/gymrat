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
from gymrat.session.records import KeepRecord
from tests._git import commit_all, head_of, status_of
from tests.loop._settle import (
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
    assert status_of(worktree) == ""


def _edit_and_commit(repo_dir: str) -> None:
    """Edit the experiment worktree, then commit the edit as the agent would."""
    edit_experiment(repo_dir)
    commit_experiment_directly(repo_dir)


def test_discard_session_when_iteration_unsettled_does_settle_it_as_discarded(repo: str):
    start_with(repo, (iteration_record(seq=1),))
    _edit_and_commit(repo)
    baseline_sha = head_of(baseline_worktree_dir(repo))

    result = discard_session(repo)

    assert head_of(experiment_worktree_dir(repo)) == baseline_sha
    assert baseline_sha[:7] in result.report
    assert result.record is not None
    assert_settling_record(result.record, discard_record(1))
    assert settling_record_of(repo) == result.record
    assert result.at == result.record.at


def test_discard_session_when_gating_block_stands_does_throw_away_the_edit_numbering_past_it(
    repo: str,
):
    start_with(repo, gating_blocked_history())
    edit_experiment(repo)

    result = discard_session(repo)

    _assert_reverted(experiment_worktree_dir(repo))
    # The block already settled iteration 1, so the discard takes the number no
    # iteration has used yet.
    assert result.record is not None
    assert_settling_record(result.record, discard_record(2))


async def test_discard_session_when_keep_retried_after_block_does_throw_away_the_edit_after_the_refusal(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, gating_blocked_history())
    edit_experiment(repo)
    checks_pass(monkeypatch)
    await keep_session(repo, checks_config())

    result = discard_session(repo)

    _assert_reverted(experiment_worktree_dir(repo))
    refusal, discard = log_records(repo)[-2:]
    assert isinstance(refusal, KeepRecord)
    assert refusal.status == "blocked"
    assert discard == result.record
    # The report names iteration 1 — the one whose edit was actually thrown away.
    assert re.search(r"iteration 1\b", result.report, re.IGNORECASE)


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


def test_discard_session_when_kept_commit_unreachable_does_fall_back_to_the_worktree_head_with_a_warning(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    # The keep names COMMIT, a commit that exists nowhere in the repository; the
    # worktree's HEAD sits on a commit past the baseline, so the fallback is visible.
    start_with(repo, (iteration_record(seq=1), committed_keep(1)))
    edit_experiment(repo)
    commit_experiment_directly(repo)
    worktree = experiment_worktree_dir(repo)
    (Path(worktree) / "after-commit.txt").write_text("more\n", encoding="utf-8")
    head = head_of(worktree)

    result = discard_session(repo)

    stderr = capsys.readouterr().err
    assert COMMIT in stderr
    assert status_of(worktree) == ""
    assert head_of(worktree) == head
    assert result.record is None


# ---------------------------------------------------------------------------
# discard_session unmeasured revert (dirty worktree, nothing to settle)
# ---------------------------------------------------------------------------


def _edit_after_a_real_keep(repo_dir: str) -> str:
    """Edit a session whose last iteration was kept onto a real commit, returning that commit."""
    start_with(repo_dir)
    kept = commit_and_keep(repo_dir, 1, "cache the regex")
    edit_experiment(repo_dir)
    return kept


def _commit_and_add_a_file(repo_dir: str) -> str:
    """Commit an edit as the agent would, then leave a fresh untracked file beside it."""
    start_with(repo_dir)
    _edit_and_commit(repo_dir)
    (Path(experiment_worktree_dir(repo_dir)) / "extra.txt").write_text("more\n", encoding="utf-8")
    return head_of(baseline_worktree_dir(repo_dir))


@pytest.mark.parametrize(
    ("arrange", "reverted"),
    [
        pytest.param(_edit_after_a_real_keep, 2, id="last-iteration-already-kept"),
        pytest.param(_commit_and_add_a_file, 3, id="agent-committed-plus-untracked"),
    ],
)
def test_discard_session_when_nothing_measured_and_dirty_does_revert_without_recording(
    repo: str, arrange: Callable[[str], str], reverted: int
):
    target = arrange(repo)

    result = discard_session(repo)

    _assert_reverted(experiment_worktree_dir(repo))
    assert head_of(experiment_worktree_dir(repo)) == target
    assert result.record is None
    assert re.search(rf"\b{reverted}\b", result.report)
    assert target[:7] in result.report


# ---------------------------------------------------------------------------
# discard_session when nothing was measured and worktree is clean (refusal)
# ---------------------------------------------------------------------------


def test_discard_session_when_nothing_measured_and_clean_does_refuse(repo: str):
    start_with(repo)

    with pytest.raises(GymratError) as excinfo:
        discard_session(repo)

    assert excinfo.value.reason == "nothing-to-discard"


def test_discard_session_when_session_id_mismatches_does_carry_stale_session_reason(repo: str):
    start_with(repo, (iteration_record(seq=1),))

    with pytest.raises(GymratError) as excinfo:
        discard_session(repo, expected_session_id="wrong-id")

    assert excinfo.value.reason == "stale-session"
