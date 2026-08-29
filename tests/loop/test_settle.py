"""Behavioral tests for settling an edit — ``keep_session`` and ``discard_session``.

Every test drives the real settle functions against a throwaway repository from
the shared ``create_scratch_repo`` factory, so the suite is order-independent and
safe under ``pytest-xdist`` / ``pytest-randomly``. The only mocked boundary is the
checks command (the consumer's own test suite); every git operation is real, and
the assertions read commit SHAs and porcelain status straight out of the worktrees
git laid down.
"""

# cspell:ignore gitdir -- the literal content of a worktree's .git pointer file

import re
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError
from gymrat.loop.settle import KeepOptions, discard_session, keep_session
from gymrat.session import (
    Confirm,
    DiscardRecord,
    IterationPrimary,
    IterationRecord,
    KeepChecks,
    KeepRecord,
    MetricVerdict,
    PairedSamples,
    SessionLogRecord,
    append_record,
    baseline_worktree_dir,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from tests._ansi import SGR_RE, strip_ansi
from tests.loop._settle import (
    CHECKS,
    CHECKS_STDERR,
    CHECKS_STDOUT,
    ExecRecorder,
    checks_config,
    checks_fail,
    checks_pass,
    edit_experiment,
    git,
    head_of,
    install_exec,
    iteration,
    last_record_of,
    start_with,
    status_of,
)
from tests.session._records import blocked_keep, committed_keep, discard_record

ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

#: The run timeout from ``checks_config().timeout_seconds``, in milliseconds.
TIMEOUT_MS = 1_800_000

#: A result the no-checks path must never reach; installed only to prove exec stayed unused.
UNUSED_EXEC = ExecResult(stdout="", stderr="", exit_code=0, stdout_bytes=0, stderr_bytes=0)

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only .git pointer sabotage")


# ---------------------------------------------------------------------------
# record builders — the iteration and keep shapes the engine produces
# ---------------------------------------------------------------------------

#: The paired rerun samples a filtered bench reports when it only emits ``total_ms``.
RERUN_SAMPLES = PairedSamples(experiment=({"total_ms": 14_120},), baseline=({"total_ms": 15_170},))


def metric(**overrides: object) -> MetricVerdict:
    """A metric verdict the engine produces, improved and gating unless overridden."""
    defaults: dict[str, object] = {
        "delta_pct": -7.2,
        "verdict": "improved",
        "method": "permutation",
        "p": 0.002,
        "noise_pct": 1.4,
        "gating": True,
        "confirmed": False,
    }
    defaults.update(overrides)
    return MetricVerdict(**defaults)  # type: ignore[arg-type]


def undefined_delta(seq: int) -> IterationRecord:
    """An iteration numbered ``seq`` whose deltas a zero baseline median left undefined."""
    return iteration(
        seq,
        metrics={"total_ms": metric(delta_pct=None, verdict="no-signal")},
        primary=IterationPrimary(kind="geomean", delta_pct=None),
        outcome="no-signal",
    )


def confirmed_regression(seq: int) -> IterationRecord:
    """An iteration whose gating metric regressed and stayed regressed on the rerun."""
    return iteration(
        seq,
        metrics={"total_ms": metric(delta_pct=9.4, verdict="regressed", confirmed=True)},
        primary=IterationPrimary(kind="geomean", delta_pct=9.4),
        outcome="regressed",
    )


def unmeasured_regression(seq: int) -> IterationRecord:
    """An iteration whose gating ``alloc_bytes`` regressed then went missing from the rerun."""
    return iteration(
        seq,
        metrics={
            "total_ms": metric(),
            "alloc_bytes": metric(delta_pct=9.4, verdict="regressed"),
        },
        primary=IterationPrimary(kind="geomean", delta_pct=9.4),
        outcome="regressed",
        confirm=Confirm(
            ran=True,
            filtered=("total_ms", "alloc_bytes"),
            absent=("alloc_bytes",),
            samples=RERUN_SAMPLES,
        ),
    )


def gating_block(seq: int) -> KeepRecord:
    """The keep a gating regression refused, numbered with the iteration it refused."""
    return blocked_keep(seq, reason="gating-regression", checks=KeepChecks(configured=True))


def nothing_to_commit_block(seq: int) -> KeepRecord:
    """The keep blocked because the experiment worktree had nothing to commit."""
    return blocked_keep(seq, reason="nothing-to-commit", checks=KeepChecks(configured=True))


def nothing_measured_block(seq: int) -> KeepRecord:
    """The keep refusing because nothing has been measured since the last settle."""
    return blocked_keep(seq, reason="nothing-measured", checks=KeepChecks(configured=True))


def failed_checks(stdout: str, stderr: str) -> KeepChecks:
    """The ``checks`` a blocked keep records for a failing run that printed both streams."""
    return KeepChecks(
        configured=True,
        passed=False,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
    )


def long_output(prefix: str) -> str:
    """200 lines of exactly 100 bytes each, every one numbered behind ``prefix``.

    The uniform line width puts the relay's byte budget on a line a test can name:
    81 lines are 8100 bytes and fit the 8192-byte budget the hook relay uses, an
    82nd would take it to 8200 and overrun it.
    """
    return "".join(f"{prefix}-{index:03d}".ljust(99, ".") + "\n" for index in range(200))


LONG_STDOUT = long_output("out")
LONG_STDERR = long_output("err")


def assert_settling_record(
    actual: KeepRecord | DiscardRecord, expected: KeepRecord | DiscardRecord
) -> None:
    """Assert ``actual`` equals ``expected`` once its stamped ``at`` is normalized.

    The settle stamps a real timestamp the fixtures cannot predict, so the ``at``
    is checked against the ISO shape and then aligned before the structural compare.
    """
    assert ISO_PATTERN.match(actual.at)
    assert replace(actual, at=expected.at) == expected


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str]) -> str:
    """A fresh scratch git repository for one settle test."""
    return create_scratch_repo()


# ---------------------------------------------------------------------------
# keep_session preconditions and checks
# ---------------------------------------------------------------------------


async def test_keep_session_when_no_session_does_refuse_pointing_at_start(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    recorder = checks_pass(monkeypatch)

    with pytest.raises(GymratError) as excinfo:
        await keep_session(repo, checks_config())

    assert excinfo.value.hint is not None
    assert "gymrat start" in excinfo.value.hint
    assert recorder.calls == []


async def test_keep_session_when_checks_pass_does_run_them_in_experiment_under_timeout(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)

    await keep_session(repo, checks_config())

    assert recorder.calls == [
        (CHECKS, ExecOptions(cwd=experiment_worktree_dir(repo), timeout_ms=TIMEOUT_MS))
    ]


async def test_keep_session_when_checks_pass_does_commit_tracked_and_untracked_changes(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    worktree = experiment_worktree_dir(repo)
    before = head_of(worktree)

    await keep_session(repo, checks_config())

    assert status_of(worktree) == ""
    assert git(["rev-parse", "HEAD~1"], worktree) == before


async def test_keep_session_when_checks_pass_does_append_committed_keep_with_commit_and_message(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config(), KeepOptions(message="cache the regex"))

    record = result.record
    assert ISO_PATTERN.match(record.at)
    assert (record.type, record.seq, record.status) == ("keep", 1, "committed")
    assert record.commit == head_of(experiment_worktree_dir(repo))
    assert record.message == "cache the regex"
    assert record.checks == KeepChecks(configured=True, passed=True)
    assert last_record_of(repo) == record


async def test_keep_session_when_checks_pass_does_advance_baseline_to_kept_commit_detached(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    baseline = baseline_worktree_dir(repo)
    assert head_of(baseline) == result.record.commit
    assert git(["rev-parse", "--abbrev-ref", "HEAD"], baseline) == "HEAD"


async def test_keep_session_when_checks_pass_does_report_the_commit_it_made(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert head_of(experiment_worktree_dir(repo))[:7] in result.report


async def test_keep_session_when_no_message_given_does_generate_one_naming_iteration_and_delta(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    subject = git(["log", "-1", "--format=%s"], experiment_worktree_dir(repo))
    assert "iteration 1" in (result.record.message or "")
    assert "-7.2" in (result.record.message or "")
    assert subject == result.record.message


async def test_keep_session_when_primary_delta_undefined_does_generate_message_that_says_so(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (undefined_delta(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    subject = git(["log", "-1", "--format=%s"], experiment_worktree_dir(repo))
    assert result.record.message == "iteration 1: geomean delta undefined"
    assert subject == result.record.message


async def test_keep_session_when_no_checks_configured_does_keep_and_record_gate_off(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    recorder = install_exec(monkeypatch, UNUSED_EXEC)

    result = await keep_session(repo, checks_config(checks=None))

    assert recorder.calls == []
    record = result.record
    assert ISO_PATTERN.match(record.at)
    assert (record.type, record.seq, record.status) == ("keep", 1, "committed")
    assert record.commit == head_of(experiment_worktree_dir(repo))
    assert isinstance(record.message, str)
    assert record.checks == KeepChecks(configured=False)


async def test_keep_session_when_checks_fail_does_append_blocked_keep(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_fail(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert_settling_record(
        result.record,
        blocked_keep(1, reason="checks-failed", checks=failed_checks(CHECKS_STDOUT, CHECKS_STDERR)),
    )
    assert last_record_of(repo) == result.record


async def test_keep_session_when_checks_fail_does_report_both_streams(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_fail(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert CHECKS_STDOUT in result.report
    assert CHECKS_STDERR in result.report


async def test_keep_session_when_checks_fail_does_leave_experiment_uncommitted(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_fail(monkeypatch)
    worktree = experiment_worktree_dir(repo)
    before = head_of(worktree)

    await keep_session(repo, checks_config())

    assert head_of(worktree) == before
    assert status_of(worktree) != ""


async def test_keep_session_when_checks_time_out_does_block_like_a_failure(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    install_exec(
        monkeypatch,
        ExecTimeoutError(
            stdout=CHECKS_STDOUT,
            stderr=CHECKS_STDERR,
            timeout_ms=TIMEOUT_MS,
            stdout_bytes=len(CHECKS_STDOUT.encode()),
            stderr_bytes=len(CHECKS_STDERR.encode()),
        ),
    )

    result = await keep_session(repo, checks_config())

    assert result.record.status == "blocked"
    assert result.record.reason == "checks-failed"
    assert result.record.checks == failed_checks(CHECKS_STDOUT, CHECKS_STDERR)


@pytest.mark.parametrize("prefix", ["out", "err"])
async def test_keep_session_when_output_over_relay_budget_does_cut_to_last_whole_line(
    repo: str, monkeypatch: pytest.MonkeyPatch, prefix: str
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    install_exec(
        monkeypatch,
        ExecResult(
            stdout=LONG_STDOUT,
            stderr=LONG_STDERR,
            exit_code=1,
            stdout_bytes=len(LONG_STDOUT.encode()),
            stderr_bytes=len(LONG_STDERR.encode()),
        ),
    )

    result = await keep_session(repo, checks_config())

    # 81 of the 100-byte lines fit the byte budget the hook relay uses, an 82nd
    # overruns it, so the cut lands between the two.
    assert f"{prefix}-000" in result.report
    assert f"{prefix}-080" in result.report
    assert f"{prefix}-081" not in result.report


async def test_keep_session_when_output_over_relay_budget_does_record_true_byte_counts(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    install_exec(
        monkeypatch,
        ExecResult(
            stdout=LONG_STDOUT,
            stderr=LONG_STDERR,
            exit_code=1,
            stdout_bytes=len(LONG_STDOUT.encode()),
            stderr_bytes=len(LONG_STDERR.encode()),
        ),
    )

    result = await keep_session(repo, checks_config())

    # The counts are what the command printed rather than what the report
    # relayed, so a reader of the log can tell the relay was cut.
    assert result.record.checks == failed_checks(LONG_STDOUT, LONG_STDERR)
    assert last_record_of(repo) == result.record


async def test_keep_session_when_output_exceeded_exec_cap_does_record_pre_cap_byte_counts(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    pre_cap_stdout_bytes = 200_000
    pre_cap_stderr_bytes = 150_000
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    install_exec(
        monkeypatch,
        ExecResult(
            stdout="capped stdout",
            stderr="capped stderr",
            exit_code=1,
            stdout_bytes=pre_cap_stdout_bytes,
            stderr_bytes=pre_cap_stderr_bytes,
        ),
    )

    result = await keep_session(repo, checks_config())

    # The keep record carries the original byte counts so a log reader can
    # tell the output was truncated by the exec cap, not the capped lengths.
    assert result.record.checks.stdout_bytes == pre_cap_stdout_bytes
    assert result.record.checks.stderr_bytes == pre_cap_stderr_bytes


# ---------------------------------------------------------------------------
# keep_session regression gate
# ---------------------------------------------------------------------------


async def test_keep_session_when_gating_regression_confirmed_does_block_before_checks(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (confirmed_regression(1),))
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert status_of(experiment_worktree_dir(repo)) != ""
    assert_settling_record(result.record, gating_block(1))


async def test_keep_session_when_gating_regression_confirmed_does_not_claim_unmeasured(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (confirmed_regression(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert not re.search(r"not measured", result.report, re.IGNORECASE)
    assert not re.search(r"filter", result.report, re.IGNORECASE)


async def test_keep_session_when_log_predates_absent_field_does_block_on_confirmation_alone(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(
        repo,
        (
            iteration(
                1,
                metrics={"total_ms": metric(delta_pct=9.4, verdict="regressed", confirmed=True)},
                primary=IterationPrimary(kind="geomean", delta_pct=9.4),
                outcome="regressed",
                confirm=Confirm(ran=True, filtered=("total_ms",), samples=RERUN_SAMPLES),
            ),
        ),
    )
    edit_experiment(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert result.record.status == "blocked"
    assert result.record.reason == "gating-regression"
    assert not re.search(r"not measured", result.report, re.IGNORECASE)


async def test_keep_session_when_rerun_did_not_confirm_regression_does_keep(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(
        repo,
        (
            iteration(
                1,
                metrics={"total_ms": metric(delta_pct=9.4, verdict="regressed")},
                outcome="no-signal",
            ),
        ),
    )
    edit_experiment(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert result.record.status == "committed"


async def test_keep_session_when_gating_exact_metric_regressed_does_block_though_unconfirmed(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(
        repo,
        (
            iteration(
                1,
                metrics={"total_ms": metric(delta_pct=9.4, verdict="regressed", method="exact")},
                primary=IterationPrimary(kind="geomean", delta_pct=9.4),
                outcome="regressed",
            ),
        ),
    )
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert_settling_record(result.record, gating_block(1))


async def test_keep_session_when_rerun_never_measured_regression_does_block_before_checks(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (unmeasured_regression(1),))
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert_settling_record(result.record, gating_block(1))


async def test_keep_session_when_rerun_never_measured_regression_does_name_metric_and_filter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (unmeasured_regression(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert "alloc_bytes" in result.report
    assert re.search(r"not measured on the confirmation rerun", result.report, re.IGNORECASE)
    assert re.search(r"filter", result.report, re.IGNORECASE)
    assert re.search(r"discard", result.report, re.IGNORECASE)


@pytest.mark.parametrize(
    "confirm",
    [
        pytest.param(
            Confirm(ran=True, filtered=("total_ms",), absent=(), samples=RERUN_SAMPLES),
            id="rerun-reported-the-metric",
        ),
        pytest.param(
            Confirm(ran=True, filtered=("total_ms",), samples=RERUN_SAMPLES),
            id="log-predates-absent-field",
        ),
    ],
)
async def test_keep_session_when_rerun_measured_regression_away_does_keep(
    repo: str, monkeypatch: pytest.MonkeyPatch, confirm: Confirm
):
    start_with(
        repo,
        (
            iteration(
                1,
                metrics={"total_ms": metric(delta_pct=9.4, verdict="no-signal")},
                outcome="no-signal",
                confirm=confirm,
            ),
        ),
    )
    edit_experiment(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert result.record.status == "committed"


# ---------------------------------------------------------------------------
# keep_session commit edge cases
# ---------------------------------------------------------------------------


@posix_only
async def test_keep_session_when_baseline_cannot_advance_does_write_no_keep_record(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    # Sabotage the baseline so git commands inside it fail: overwrite the .git
    # pointer to a path that does not resolve.
    baseline = baseline_worktree_dir(repo)
    (Path(baseline) / ".git").write_text("gitdir: /nonexistent\n", encoding="utf-8")

    with pytest.raises(GymratError, match="baseline"):
        await keep_session(repo, checks_config())

    last = last_record_of(repo)
    assert last.type == "iteration"
    assert last.seq == 1


@posix_only
async def test_keep_session_when_commit_landed_but_advance_failed_does_recover_on_retry(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    worktree = experiment_worktree_dir(repo)
    baseline = baseline_worktree_dir(repo)
    git_pointer = (Path(baseline) / ".git").read_text(encoding="utf-8")
    head_before = head_of(worktree)

    (Path(baseline) / ".git").write_text("gitdir: /nonexistent\n", encoding="utf-8")
    with pytest.raises(GymratError, match="baseline"):
        await keep_session(repo, checks_config())

    # The commit landed — worktree is clean and HEAD moved forward.
    committed = head_of(worktree)
    assert status_of(worktree) == ""
    assert committed != head_before

    (Path(baseline) / ".git").write_text(git_pointer, encoding="utf-8")
    result = await keep_session(repo, checks_config())

    assert result.record.status == "committed"
    assert result.record.commit == committed
    assert head_of(baseline) == committed


def _commit_experiment_directly(repo: str) -> str:
    """Commit the experiment worktree outside a keep, returning the standing commit."""
    worktree = experiment_worktree_dir(repo)
    git(["add", "-A"], worktree)
    git(["commit", "-m", "committed outside the keep"], worktree)
    return head_of(worktree)


async def test_keep_session_when_clean_and_ahead_does_keep_standing_commit_on_passing_checks(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    committed = _commit_experiment_directly(repo)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == [
        (CHECKS, ExecOptions(cwd=experiment_worktree_dir(repo), timeout_ms=TIMEOUT_MS))
    ]
    assert result.record.status == "committed"
    assert result.record.commit == committed
    assert result.record.checks == KeepChecks(configured=True, passed=True)
    assert head_of(baseline_worktree_dir(repo)) == committed


async def test_keep_session_when_clean_and_ahead_does_refuse_standing_commit_on_failing_checks(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    _commit_experiment_directly(repo)
    checks_fail(monkeypatch)
    baseline = baseline_worktree_dir(repo)
    baseline_before = head_of(baseline)

    result = await keep_session(repo, checks_config())

    assert result.record.status == "blocked"
    assert result.record.reason == "checks-failed"
    assert result.record.checks == failed_checks(CHECKS_STDOUT, CHECKS_STDERR)
    assert CHECKS_STDOUT in result.report
    assert CHECKS_STDERR in result.report
    assert head_of(baseline) == baseline_before
    assert last_record_of(repo) == result.record


async def test_keep_session_when_clean_and_ahead_and_no_checks_does_keep_standing_commit_unchecked(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    committed = _commit_experiment_directly(repo)
    recorder = install_exec(monkeypatch, UNUSED_EXEC)

    result = await keep_session(repo, checks_config(checks=None))

    assert recorder.calls == []
    assert result.record.status == "committed"
    assert result.record.checks == KeepChecks(configured=False)
    assert head_of(baseline_worktree_dir(repo)) == committed


async def test_keep_session_when_head_matches_baseline_does_append_nothing_to_commit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    # The iteration measured something but the agent made no changes.
    start_with(repo, (iteration(1),))
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert_settling_record(result.record, nothing_to_commit_block(1))


async def test_keep_session_when_nothing_new_after_prior_keep_does_append_nothing_to_commit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    await keep_session(repo, checks_config())
    append_record(session_jsonl_path(repo), iteration(2))

    result = await keep_session(repo, checks_config())

    assert_settling_record(result.record, nothing_to_commit_block(2))


@pytest.mark.parametrize(
    ("history", "seq"),
    [
        pytest.param((), 1, id="no-iteration-ever-recorded"),
        pytest.param((iteration(1), committed_keep(1)), 2, id="last-iteration-already-kept"),
    ],
)
async def test_keep_session_when_nothing_measured_does_refuse_with_nothing_measured_keep(
    repo: str, monkeypatch: pytest.MonkeyPatch, history: tuple[SessionLogRecord, ...], seq: int
):
    start_with(repo, history)
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert_settling_record(result.record, nothing_measured_block(seq))


async def test_keep_session_when_second_refusal_does_number_past_the_first(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    edit_experiment(repo)
    checks_pass(monkeypatch)
    await keep_session(repo, checks_config())

    result = await keep_session(repo, checks_config())

    # A consumer walking the raw log sees two distinct records, not one number
    # written twice.
    keeps = [record for record in read_records(session_jsonl_path(repo)) if record.type == "keep"]
    assert [record.seq for record in keeps] == [1, 2]
    assert result.record.seq == 2


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

    assert_settling_record(result.record, discard_record(1))
    assert last_record_of(repo) == result.record


def test_discard_session_when_primary_delta_undefined_does_throw_it_away(repo: str):
    start_with(repo, (undefined_delta(1),))
    edit_experiment(repo)

    result = discard_session(repo)

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
    _commit_experiment_directly(repo)
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
    _commit_experiment_directly(repo)

    result = discard_session(repo)

    worktree = experiment_worktree_dir(repo)
    assert head_of(worktree)[:7] in result.report


# ---------------------------------------------------------------------------
# discard_session when nothing was measured
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "history",
    [
        pytest.param((), id="no-iteration-ever-recorded"),
        pytest.param((iteration(1), committed_keep(1)), id="last-iteration-already-kept"),
        pytest.param(
            (confirmed_regression(1), gating_block(1), discard_record(2)),
            id="gating-block-already-discarded",
        ),
    ],
)
def test_discard_session_when_nothing_measured_does_refuse_settling_again(
    repo: str, history: tuple[SessionLogRecord, ...]
):
    start_with(repo, history)
    edit_experiment(repo)
    before = len(read_records(session_jsonl_path(repo)))

    with pytest.raises(GymratError):
        discard_session(repo)

    assert len(read_records(session_jsonl_path(repo))) == before
    assert status_of(experiment_worktree_dir(repo)) != ""


# ---------------------------------------------------------------------------
# the hint the refusals close on
# ---------------------------------------------------------------------------


def _nothing_measured(repo: str) -> None:
    """An edited experiment with no iteration measured behind it."""
    start_with(repo, ())
    edit_experiment(repo)


def _nothing_to_commit(repo: str) -> None:
    """A measured iteration the agent left the experiment untouched under."""
    start_with(repo, (iteration(1),))


def _edited_after_iteration(repo: str) -> None:
    """The ordinary keep shape: one measured iteration and an edit to commit."""
    start_with(repo, (iteration(1),))
    edit_experiment(repo)


def _standing_gating_regression(repo: str) -> None:
    """An edit standing behind a gating regression the rerun confirmed."""
    start_with(repo, (confirmed_regression(1),))
    edit_experiment(repo)


def _unmeasured_gating_regression(repo: str) -> None:
    """An edit standing behind a gating regression the rerun never measured."""
    start_with(repo, (unmeasured_regression(1),))
    edit_experiment(repo)


@pytest.mark.parametrize(
    ("arrange", "install_checks"),
    [
        pytest.param(_nothing_measured, checks_pass, id="nothing-measured"),
        pytest.param(_nothing_to_commit, checks_pass, id="nothing-to-commit"),
        pytest.param(_edited_after_iteration, checks_fail, id="checks-failed"),
        pytest.param(_standing_gating_regression, checks_pass, id="gating-regression"),
        pytest.param(_unmeasured_gating_regression, checks_pass, id="unmeasured-regression"),
    ],
)
async def test_keep_session_when_refusing_does_close_on_a_hint_carrying_no_label(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[str], None],
    install_checks: Callable[[pytest.MonkeyPatch], ExecRecorder],
):
    arrange(repo)
    install_checks(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert "Hint" not in result.report
    assert "`" not in result.report


async def test_keep_session_when_nothing_measured_does_name_iterate_in_bare_prose(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _nothing_measured(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert "run gymrat iterate first" in result.report


async def test_keep_session_when_colored_does_dim_the_hint_and_paint_the_command(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _nothing_measured(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config(), color=True)

    hint = next(line for line in result.report.split("\n") if "gymrat iterate" in strip_ansi(line))
    assert hint.startswith("\x1b[2m")
    assert any("34" in run.split(";") for run in SGR_RE.findall(hint))


async def test_keep_session_when_checks_output_holds_markup_metacharacters_does_report_it_literally(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _edited_after_iteration(repo)
    noisy = "FAIL [i] parse_config"
    install_exec(
        monkeypatch,
        ExecResult(
            stdout=noisy, stderr="", exit_code=1, stdout_bytes=len(noisy.encode()), stderr_bytes=0
        ),
    )

    result = await keep_session(repo, checks_config(), color=True)

    assert noisy in strip_ansi(result.report)


async def test_keep_session_when_no_checks_configured_does_warn_on_stderr_without_a_label(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _edited_after_iteration(repo)
    install_exec(monkeypatch, UNUSED_EXEC)

    await keep_session(repo, checks_config(checks=None))

    warning = capsys.readouterr().err
    assert "gymrat.toml" in warning
    assert "Hint" not in warning
    assert "`" not in warning


async def test_keep_session_when_no_checks_configured_and_color_forced_does_dim_the_hint(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    _edited_after_iteration(repo)
    install_exec(monkeypatch, UNUSED_EXEC)

    await keep_session(repo, checks_config(checks=None))

    hint = capsys.readouterr().err.splitlines()[1]
    assert hint.startswith("\x1b[2m")
