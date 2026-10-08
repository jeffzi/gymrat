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
from gymrat.session.records import IterationRecord, KeepRecord, SessionLogRecord
from tests._git import head_of, run_git, status_of
from tests.loop._settle import (
    assert_settling_record,
    checks_config,
    checks_pass,
    commit_experiment_directly,
    confirmed_regression,
    edit_experiment,
    settling_record_of,
    start_with,
    unmeasured_regression,
)
from tests.session.records._fixtures import (
    append_records,
    committed_keep,
    discard_record,
    gate_block,
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


@pytest.mark.parametrize(
    "arrange_worktree",
    [
        pytest.param(edit_experiment, id="dirty-edit"),
        pytest.param(_leave_worktree_clean, id="clean-worktree"),
    ],
)
def test_discard_session_when_iteration_unsettled_does_settle_it_as_discarded(
    repo: str, arrange_worktree: Callable[[str], None]
):
    start_with(repo, (iteration_record(seq=1),))
    arrange_worktree(repo)

    result = discard_session(repo)

    _assert_reverted(experiment_worktree_dir(repo))
    assert result.record is not None
    assert_settling_record(result.record, discard_record(1))
    assert settling_record_of(repo) == result.record
    assert result.at == result.record.at


_GATING_BLOCK = gate_block(1, "gating-regression")


@pytest.mark.parametrize(
    "blocked_iteration",
    [
        pytest.param(confirmed_regression(1), id="confirmed-regression"),
        pytest.param(unmeasured_regression(1), id="unmeasured-regression"),
    ],
)
def test_discard_session_when_gating_block_stands_does_throw_away_the_edit_numbering_past_it(
    repo: str, blocked_iteration: IterationRecord
):
    start_with(repo, (blocked_iteration, _GATING_BLOCK))
    edit_experiment(repo)

    result = discard_session(repo)

    _assert_reverted(experiment_worktree_dir(repo))
    # The block already settled iteration 1, so the discard takes the number no
    # iteration has used yet, leaving the block in history.
    assert result.record is not None
    assert_settling_record(result.record, discard_record(2))
    assert log_records(repo)[-2:] == [_GATING_BLOCK, result.record]


async def test_discard_session_when_keep_retried_after_block_does_throw_away_the_edit_after_the_refusal(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(
        repo,
        (
            confirmed_regression(1),
            _GATING_BLOCK,
        ),
    )
    edit_experiment(repo)
    checks_pass(monkeypatch)
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


def test_discard_session_when_nothing_kept_and_agent_committed_does_reset_to_the_named_baseline_sha(
    repo: str,
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    commit_experiment_directly(repo)
    worktree = experiment_worktree_dir(repo)
    baseline_sha = head_of(baseline_worktree_dir(repo))

    result = discard_session(repo)

    assert head_of(worktree) == baseline_sha
    assert status_of(worktree) == ""
    assert baseline_sha[:7] in result.report


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
    (Path(worktree) / "post-keep.txt").write_text("after keep\n", encoding="utf-8")
    run_git(["add", "-A"], worktree)
    run_git(["commit", "-m", "agent commit after keep"], worktree)

    discard_session(repo)

    assert head_of(worktree) == kept_commit
    assert status_of(worktree) == ""


# ---------------------------------------------------------------------------
# discard_session unmeasured revert (dirty worktree, nothing to settle)
# ---------------------------------------------------------------------------

NOTHING_MEASURED_HISTORIES = [
    pytest.param((), id="no-iteration-ever-recorded"),
    pytest.param((iteration_record(seq=1), committed_keep(1)), id="last-iteration-already-kept"),
    pytest.param(
        (
            confirmed_regression(1),
            _GATING_BLOCK,
            discard_record(2),
        ),
        id="gating-block-already-discarded",
    ),
]


@pytest.mark.parametrize("history", NOTHING_MEASURED_HISTORIES)
def test_discard_session_when_nothing_measured_and_dirty_does_revert_without_recording(
    repo: str, history: tuple[SessionLogRecord, ...]
):
    start_with(repo, history)
    edit_experiment(repo)
    records_before = len(log_records(repo))
    baseline_sha = head_of(baseline_worktree_dir(repo))

    result = discard_session(repo)

    _assert_reverted(experiment_worktree_dir(repo))
    assert len(log_records(repo)) == records_before
    assert result.record is None
    assert isinstance(result.at, int)
    assert result.at > 0
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
    baseline_sha = head_of(baseline_worktree_dir(repo))

    result = discard_session(repo)

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
