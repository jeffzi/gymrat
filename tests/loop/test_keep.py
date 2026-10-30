"""Behavioral tests for ``keep_session``.

Preconditions, checks, the regression gate, the outcome gate, commit edge cases,
nothing-measured refusals, and the hints that refusals close on.

Every test drives the real settle functions against a throwaway repository from
the shared ``create_scratch_repo`` factory, so the suite is order-independent and
safe under ``pytest-xdist`` / ``pytest-randomly``. The only mocked boundary is the
checks command (the consumer's own test suite); every git operation is real.
"""

import re
from collections.abc import Awaitable, Callable

import pytest

from gymrat.exec import ExecOptions, ExecTimeoutError
from gymrat.loop.keep import KeepOptions, keep_session
from gymrat.report.format import format_percent_delta
from gymrat.session.paths import baseline_worktree_dir, experiment_worktree_dir
from gymrat.session.records import (
    BaselineRecord,
    IterationRecord,
    KeepChecks,
)
from gymrat.session.schema import Outcome
from gymrat.utils import SHORT_SHA_LENGTH
from tests._ansi import SGR_RE, strip_ansi
from tests._exec_fixtures import (
    CAPPED_STDERR_BYTES,
    CAPPED_STDOUT_BYTES,
    ExecRecorder,
    capped_result,
    expected_result,
    install_exec,
)
from tests._git import head_of, run_git
from tests._streams import FakeStream
from tests.loop._settle import (
    CHECKS,
    CHECKS_STDERR,
    CHECKS_STDOUT,
    KEEP_EXEC,
    UNUSED_EXEC,
    assert_settling_record,
    checks_config,
    checks_fail,
    checks_pass,
    commit_experiment_directly,
    confirmed_regression,
    edit_experiment,
    exact_regression,
    settling_record_of,
    start_with,
    undefined_delta,
    unimproved,
    unmeasured_regression,
)
from tests.session.records._fixtures import (
    append_records,
    blocked_keep,
    gate_block,
    iteration_record,
    log_records,
    settled_history,
)

#: The run timeout from ``checks_config().timeout_seconds``, in milliseconds.
TIMEOUT_MS = 1_800_000


def failed_checks(stdout: str, stderr: str) -> KeepChecks:
    """The ``checks`` a blocked keep records for a failing run that printed both streams."""
    return KeepChecks(
        configured=True,
        passed=False,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
    )


def long_output(prefix: str) -> str:
    """Build 200 numbered lines of 100 bytes each, 20000 bytes in all.

    That is well past the relay's byte budget, so the report must drop the
    closing lines.

    Args:
        prefix: The word each line opens on, ahead of its three-digit number.

    Returns:
        The lines joined, each ending in a newline.
    """
    return "".join(f"{prefix}-{index:03d}".ljust(99, ".") + "\n" for index in range(200))


LONG_STDOUT = long_output("out")
LONG_STDERR = long_output("err")


# ---------------------------------------------------------------------------
# keep_session preconditions and checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "inserted"),
    [
        pytest.param("cache the regex", ("cache the regex",), id="message-given"),
        pytest.param(None, ("1", "geomean", format_percent_delta(-7.2)), id="message-generated"),
    ],
)
async def test_keep_session_when_checks_pass_does_settle_the_edit_as_committed(
    repo: str, monkeypatch: pytest.MonkeyPatch, message: str | None, inserted: tuple[str, ...]
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)
    worktree = experiment_worktree_dir(repo)
    baseline = baseline_worktree_dir(repo)
    before = head_of(worktree)

    result = await keep_session(repo, checks_config(), KeepOptions(message=message))

    record = result.record
    assert recorder.calls == [(CHECKS, ExecOptions(cwd=worktree, timeout_ms=TIMEOUT_MS))]
    assert run_git(["rev-parse", "HEAD~1"], worktree) == before
    assert (record.seq, record.status) == (1, "committed")
    assert record.commit == head_of(worktree)
    assert record.checks == KeepChecks(configured=True, passed=True)
    assert record.message is not None
    assert [part for part in inserted if part not in record.message] == []
    assert run_git(["log", "-1", "--format=%s"], worktree) == record.message
    assert settling_record_of(repo) == record
    assert head_of(baseline) == record.commit
    assert head_of(worktree)[:7] in result.report


async def test_keep_session_when_primary_delta_undefined_does_generate_message_that_says_so(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (undefined_delta(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config(), KeepOptions(allow_unimproved=True))

    assert result.record.message is not None
    assert result.record.message.endswith("delta undefined")


async def test_keep_session_when_checks_fail_does_block_reporting_both_streams_leaving_the_edit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    checks_fail(monkeypatch)
    worktree = experiment_worktree_dir(repo)
    before = head_of(worktree)

    result = await keep_session(repo, checks_config())

    assert_settling_record(
        result.record,
        blocked_keep(1, reason="checks-failed", checks=failed_checks(CHECKS_STDOUT, CHECKS_STDERR)),
    )
    assert settling_record_of(repo) == result.record
    assert (CHECKS_STDOUT in result.report, CHECKS_STDERR in result.report) == (True, True)
    assert head_of(worktree) == before


async def test_keep_session_when_checks_time_out_does_block_like_a_failure(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    install_exec(
        monkeypatch,
        KEEP_EXEC,
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


async def test_keep_session_when_output_over_relay_budget_does_truncate_the_report(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    install_exec(monkeypatch, KEEP_EXEC, expected_result(LONG_STDOUT, LONG_STDERR, exit_code=1))

    result = await keep_session(repo, checks_config())

    assert "out-000" in result.report
    assert "out-199" not in result.report
    assert "err-000" in result.report
    assert "err-199" not in result.report


async def test_keep_session_when_output_exceeded_exec_cap_does_record_pre_cap_byte_counts(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    install_exec(monkeypatch, KEEP_EXEC, capped_result(exit_code=1))

    result = await keep_session(repo, checks_config())

    # The keep record carries the original byte counts so a log reader can
    # tell the output was truncated by the exec cap, not the capped lengths.
    assert result.record.checks.stdout_bytes == CAPPED_STDOUT_BYTES
    assert result.record.checks.stderr_bytes == CAPPED_STDERR_BYTES


# ---------------------------------------------------------------------------
# keep_session regression gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "measured",
    [
        pytest.param(confirmed_regression(1), id="confirmed-regression"),
        pytest.param(exact_regression(1), id="exact-method-unconfirmed"),
    ],
)
async def test_keep_session_when_gating_regression_stands_does_block_before_checks(
    repo: str, monkeypatch: pytest.MonkeyPatch, measured: IterationRecord
):
    start_with(repo, (measured,))
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert_settling_record(result.record, gate_block(1, "gating-regression"))
    assert not re.search(r"filter", result.report, re.IGNORECASE)


async def test_keep_session_when_rerun_never_measured_regression_does_block_naming_the_metric(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (unmeasured_regression(1),))
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert_settling_record(result.record, gate_block(1, "gating-regression"))
    assert "alloc_bytes" in result.report


# ---------------------------------------------------------------------------
# keep_session outcome gate
# ---------------------------------------------------------------------------

#: The two outcomes the gate refuses without ``allow_unimproved``.
UNIMPROVED = [
    pytest.param("no-signal", id="no-signal"),
    pytest.param("regressed", id="regressed"),
]


@pytest.mark.parametrize("outcome", UNIMPROVED)
async def test_keep_session_when_outcome_not_improved_does_block_before_checks(
    repo: str, monkeypatch: pytest.MonkeyPatch, outcome: Outcome
):
    start_with(repo, (unimproved(1, outcome),))
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert_settling_record(result.record, gate_block(1, "not-improved"))
    assert outcome in result.report


# ---------------------------------------------------------------------------
# keep_session commit edge cases
# ---------------------------------------------------------------------------


def _checks_skipped(monkeypatch: pytest.MonkeyPatch) -> ExecRecorder:
    return install_exec(monkeypatch, KEEP_EXEC, UNUSED_EXEC)


@pytest.mark.parametrize(
    ("install_checks", "checks", "status", "recorded_checks", "baseline_advanced", "checks_ran"),
    [
        pytest.param(
            checks_pass,
            CHECKS,
            "committed",
            KeepChecks(configured=True, passed=True),
            True,
            True,
            id="passing-checks-keep-it",
        ),
        pytest.param(
            checks_fail,
            CHECKS,
            "blocked",
            failed_checks(CHECKS_STDOUT, CHECKS_STDERR),
            False,
            True,
            id="failing-checks-refuse-it",
        ),
        pytest.param(
            _checks_skipped,
            None,
            "committed",
            KeepChecks(configured=False),
            True,
            False,
            id="no-checks-keep-it-unchecked",
        ),
    ],
)
async def test_keep_session_when_clean_and_ahead_does_settle_the_standing_commit_on_the_checks(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    install_checks: Callable[[pytest.MonkeyPatch], ExecRecorder],
    checks: str | None,
    status: str,
    recorded_checks: KeepChecks,
    baseline_advanced: bool,
    checks_ran: bool,
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    committed = commit_experiment_directly(repo)
    recorder = install_checks(monkeypatch)

    result = await keep_session(repo, checks_config(checks=checks))

    assert (result.record.status, result.record.checks) == (status, recorded_checks)
    assert (head_of(baseline_worktree_dir(repo)) == committed) is baseline_advanced
    assert bool(recorder.calls) is checks_ran


async def _measured_without_an_edit(repo: str, monkeypatch: pytest.MonkeyPatch) -> int:
    """Measure iteration 1 with no edit made; return its seq."""
    start_with(repo, (iteration_record(seq=1),))
    return 1


async def _measured_nothing_new_after_a_keep(repo: str, monkeypatch: pytest.MonkeyPatch) -> int:
    """Keep an edit, then measure iteration 2 with nothing new on top; return its seq."""
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    await keep_session(repo, checks_config())
    append_records(repo, iteration_record(seq=2))
    return 2


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_measured_without_an_edit, id="nothing-kept-yet"),
        pytest.param(_measured_nothing_new_after_a_keep, id="after-a-prior-keep"),
    ],
)
async def test_keep_session_when_head_matches_baseline_does_block_as_nothing_to_commit(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[str, pytest.MonkeyPatch], Awaitable[int]],
):
    seq = await arrange(repo, monkeypatch)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert_settling_record(result.record, gate_block(seq, "nothing-to-commit"))


async def test_keep_session_when_nothing_measured_does_refuse_with_nothing_measured_keep(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, settled_history())
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert_settling_record(result.record, gate_block(2, "nothing-measured"))


# ---------------------------------------------------------------------------
# keep_session hint tests
# ---------------------------------------------------------------------------


def _edited_after_iteration(repo: str) -> None:
    """The ordinary keep shape: one measured iteration and an edit to commit."""
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)


async def test_keep_session_when_checks_output_holds_markup_metacharacters_does_report_it_literally(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _edited_after_iteration(repo)
    noisy = "FAIL [i] parse_config"
    install_exec(monkeypatch, KEEP_EXEC, expected_result(noisy, exit_code=1))

    result = await keep_session(repo, checks_config(), color=True)

    assert noisy in strip_ansi(result.report)


async def test_keep_session_when_no_checks_configured_does_send_a_styled_hint_to_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    stderr = FakeStream(tty=True)
    monkeypatch.setattr("sys.stderr", stderr)
    _edited_after_iteration(repo)
    install_exec(monkeypatch, KEEP_EXEC, UNUSED_EXEC)

    await keep_session(repo, checks_config(checks=None))

    assert SGR_RE.search(stderr.getvalue()) is not None


async def test_keep_session_when_no_checks_and_warn_sink_does_send_plain_hint_to_the_sink(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    _edited_after_iteration(repo)
    install_exec(monkeypatch, KEEP_EXEC, UNUSED_EXEC)
    warned: list[str] = []

    await keep_session(repo, checks_config(checks=None), KeepOptions(warn=warned.append))

    assert len(warned) == 1
    assert SGR_RE.search(warned[0]) is None
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# keep_session baseline bookkeeping
# ---------------------------------------------------------------------------


async def test_keep_session_when_committed_does_append_the_kept_samples_as_a_baseline(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    kept = iteration_record(seq=1)
    start_with(repo, (kept,))
    edit_experiment(repo)
    checks_pass(monkeypatch)

    await keep_session(repo, checks_config())

    records = log_records(repo)
    baseline = records[-1]
    assert isinstance(baseline, BaselineRecord)
    assert baseline.label == head_of(experiment_worktree_dir(repo))[:SHORT_SHA_LENGTH]
    assert baseline.samples == kept.samples.experiment
