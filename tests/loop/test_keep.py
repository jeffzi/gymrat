"""Behavioral tests for ``keep_session``.

Preconditions, checks, the regression gate, the outcome gate, commit edge cases,
nothing-measured refusals, and the hints that refusals close on.

Every test drives the real settle functions against a throwaway repository from
the shared ``create_scratch_repo`` factory, so the suite is order-independent and
safe under ``pytest-xdist`` / ``pytest-randomly``. The only mocked boundary is the
checks command (the consumer's own test suite); every git operation is real.
"""

# cspell:ignore gitdir -- the literal content of a worktree's .git pointer file

import re
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError
from gymrat.git import SHORT_SHA_LENGTH
from gymrat.loop.keep import KeepOptions, keep_session
from gymrat.session.paths import baseline_worktree_dir, experiment_worktree_dir
from gymrat.session.records import (
    BaselineRecord,
    IterationPrimary,
    IterationRecord,
    KeepChecks,
    KeepRecord,
    SessionLogRecord,
)
from gymrat.session.schema import Outcome
from gymrat.session.store import latest_baseline
from tests._ansi import SGR_RE, strip_ansi
from tests._exec_fixtures import (
    ExecRecorder,
    expected_result,
    install_exec,
)
from tests._git import head_of, run_git, status_of
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
    measured_rounds,
    settling_record_of,
    start_with,
    undefined_delta,
    unimproved,
    unmeasured_regression,
)
from tests.session.records._fixtures import (
    append_records,
    blocked_keep,
    committed_keep,
    iteration_record,
    log_records,
    metric_verdict,
    records_of_type,
)

#: The run timeout from ``checks_config().timeout_seconds``, in milliseconds.
TIMEOUT_MS = 1_800_000

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only .git pointer sabotage")


def failed_checks(stdout: str, stderr: str) -> KeepChecks:
    """The ``checks`` a blocked keep records for a failing run that printed both streams."""
    return KeepChecks(
        configured=True,
        passed=False,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
    )


def long_output(prefix: str) -> str:
    """Build 200 numbered lines of exactly 100 bytes each.

    The uniform line width puts the relay's byte budget on a line a test can name:
    81 lines are 8100 bytes and fit the 8192-byte budget the hook relay uses, an
    82nd would take it to 8200 and overrun it.

    Args:
        prefix: The word each line opens on, ahead of its three-digit number.

    Returns:
        The lines joined, each ending in a newline.
    """
    return "".join(f"{prefix}-{index:03d}".ljust(99, ".") + "\n" for index in range(200))


LONG_STDOUT = long_output("out")
LONG_STDERR = long_output("err")


def _hint_line(report: str, action: str) -> str:
    """The report line naming ``action``, with ANSI styling stripped."""
    return next(line for line in map(strip_ansi, report.split("\n")) if action in line)


def _assert_closes_on_a_bare_hint(repo: str, report: str) -> None:
    """Assert a refusal names no ``gymrat`` command or markup and appends no baseline."""
    assert "Hint" not in report
    assert "`" not in report
    assert "gymrat " not in report
    assert latest_baseline(log_records(repo)) is None


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


@pytest.mark.parametrize(
    ("message", "expected_message"),
    [
        pytest.param("cache the regex", "cache the regex", id="message-given"),
        pytest.param(None, "iteration 1: geomean -7.2%", id="message-generated"),
    ],
)
async def test_keep_session_when_checks_pass_does_settle_the_edit_as_committed(
    repo: str, monkeypatch: pytest.MonkeyPatch, message: str | None, expected_message: str
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
    assert status_of(worktree) == ""
    assert run_git(["rev-parse", "HEAD~1"], worktree) == before
    assert isinstance(record.at, int)
    assert record.at > 0
    assert (record.type, record.seq, record.status) == ("keep", 1, "committed")
    assert record.commit == head_of(worktree)
    assert record.checks == KeepChecks(configured=True, passed=True)
    assert record.message == expected_message
    assert run_git(["log", "-1", "--format=%s"], worktree) == expected_message
    assert settling_record_of(repo) == record
    assert head_of(baseline) == record.commit
    assert run_git(["rev-parse", "--abbrev-ref", "HEAD"], baseline) == "HEAD"
    assert head_of(worktree)[:7] in result.report


async def test_keep_session_when_primary_delta_undefined_does_generate_message_that_says_so(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (undefined_delta(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config(), KeepOptions(allow_unimproved=True))

    subject = run_git(["log", "-1", "--format=%s"], experiment_worktree_dir(repo))
    assert result.record.message == "iteration 1: geomean delta undefined"
    assert subject == result.record.message


async def test_keep_session_when_no_checks_configured_does_keep_it_unchecked_with_a_stderr_warning(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    recorder = install_exec(monkeypatch, KEEP_EXEC, UNUSED_EXEC)

    result = await keep_session(repo, checks_config(checks=None))

    warning = capsys.readouterr().err
    assert recorder.calls == []
    record = result.record
    assert isinstance(record.at, int)
    assert record.at > 0
    assert (record.type, record.seq, record.status) == ("keep", 1, "committed")
    assert record.commit == head_of(experiment_worktree_dir(repo))
    assert isinstance(record.message, str)
    assert record.checks == KeepChecks(configured=False)
    assert "gymrat.toml" in warning
    assert "Hint" not in warning
    assert "`" not in warning


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
    assert status_of(worktree) != ""
    _assert_closes_on_a_bare_hint(repo, result.report)


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


async def test_keep_session_when_output_over_relay_budget_does_cut_report_but_record_true_counts(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    install_exec(monkeypatch, KEEP_EXEC, expected_result(LONG_STDOUT, LONG_STDERR, exit_code=1))

    result = await keep_session(repo, checks_config())

    # 81 of the 100-byte lines fit the byte budget the hook relay uses, an 82nd
    # overruns it, so the cut lands between the two.
    for prefix in ("out", "err"):
        assert f"{prefix}-000" in result.report
        assert f"{prefix}-080" in result.report
        assert f"{prefix}-081" not in result.report
    assert result.record.checks == failed_checks(LONG_STDOUT, LONG_STDERR)
    assert settling_record_of(repo) == result.record


async def test_keep_session_when_output_exceeded_exec_cap_does_record_pre_cap_byte_counts(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    pre_cap_stdout_bytes = 200_000
    pre_cap_stderr_bytes = 150_000
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    install_exec(
        monkeypatch,
        KEEP_EXEC,
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
    assert_settling_record(
        result.record,
        blocked_keep(1, reason="gating-regression", checks=KeepChecks(configured=True)),
    )
    assert not re.search(r"not measured", result.report, re.IGNORECASE)
    assert not re.search(r"filter", result.report, re.IGNORECASE)
    _assert_closes_on_a_bare_hint(repo, result.report)


async def test_keep_session_when_gating_exact_metric_regressed_does_block_though_unconfirmed(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(
        repo,
        (
            iteration_record(
                seq=1,
                metrics={
                    "total_ms": metric_verdict(delta_pct=9.4, verdict="regressed", method="exact")
                },
                primary=IterationPrimary(kind="geomean", delta_pct=9.4),
                outcome="regressed",
            ),
        ),
    )
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert_settling_record(
        result.record,
        blocked_keep(1, reason="gating-regression", checks=KeepChecks(configured=True)),
    )


async def test_keep_session_when_rerun_never_measured_regression_does_block_naming_metric_and_filter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (unmeasured_regression(1),))
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert_settling_record(
        result.record,
        blocked_keep(1, reason="gating-regression", checks=KeepChecks(configured=True)),
    )
    assert "alloc_bytes" in result.report
    assert re.search(r"not measured on the confirmation rerun", result.report, re.IGNORECASE)
    assert re.search(r"filter", result.report, re.IGNORECASE)
    assert re.search(r"discard", result.report, re.IGNORECASE)
    _assert_closes_on_a_bare_hint(repo, result.report)


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
    worktree = experiment_worktree_dir(repo)
    experiment_before = head_of(worktree)
    baseline_before = head_of(baseline_worktree_dir(repo))

    result = await keep_session(repo, checks_config())

    assert recorder.calls == []
    assert_settling_record(
        result.record, blocked_keep(1, reason="not-improved", checks=KeepChecks(configured=True))
    )
    assert settling_record_of(repo) == result.record
    assert head_of(worktree) == experiment_before
    assert status_of(worktree) != ""
    assert head_of(baseline_worktree_dir(repo)) == baseline_before
    assert f"Keep refused: the iteration was {outcome}, not improved." in result.report
    assert "discard it, or pass --allow-unimproved to keep it anyway." in result.report
    _assert_closes_on_a_bare_hint(repo, result.report)


@pytest.mark.parametrize(
    "measured",
    [
        pytest.param(unimproved(1, "no-signal"), id="no-signal"),
        pytest.param(unimproved(1, "regressed"), id="regressed"),
        pytest.param(iteration_record(seq=1), id="improved"),
    ],
)
async def test_keep_session_when_allow_unimproved_does_commit_on_passing_checks(
    repo: str, monkeypatch: pytest.MonkeyPatch, measured: IterationRecord
):
    start_with(repo, (measured,))
    edit_experiment(repo)
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config(), KeepOptions(allow_unimproved=True))

    assert recorder.calls == [
        (CHECKS, ExecOptions(cwd=experiment_worktree_dir(repo), timeout_ms=TIMEOUT_MS))
    ]
    assert result.record.status == "committed"
    assert result.record.commit == head_of(experiment_worktree_dir(repo))
    assert result.record.checks == KeepChecks(configured=True, passed=True)
    assert head_of(baseline_worktree_dir(repo)) == result.record.commit


async def test_keep_session_when_override_follows_a_not_improved_refusal_does_commit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (unimproved(1, "no-signal"),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    await keep_session(repo, checks_config())

    result = await keep_session(repo, checks_config(), KeepOptions(allow_unimproved=True))

    keeps = records_of_type(repo, KeepRecord)
    assert [(record.status, record.reason) for record in keeps] == [
        ("blocked", "not-improved"),
        ("committed", None),
    ]
    assert result.record.commit == head_of(experiment_worktree_dir(repo))


# ---------------------------------------------------------------------------
# keep_session commit edge cases
# ---------------------------------------------------------------------------


@posix_only
async def test_keep_session_when_baseline_advance_fails_does_leave_the_commit_landed_unrecorded(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    worktree = experiment_worktree_dir(repo)
    head_before = head_of(worktree)
    (Path(baseline_worktree_dir(repo)) / ".git").write_text(
        "gitdir: /nonexistent\n", encoding="utf-8"
    )

    with pytest.raises(GymratError, match="baseline"):
        await keep_session(repo, checks_config())

    last = settling_record_of(repo)
    assert isinstance(last, IterationRecord)
    assert last.seq == 1
    assert status_of(worktree) == ""
    assert head_of(worktree) != head_before


def _checks_skipped(monkeypatch: pytest.MonkeyPatch) -> ExecRecorder:
    return install_exec(monkeypatch, KEEP_EXEC, UNUSED_EXEC)


@pytest.mark.parametrize(
    ("install_checks", "checks", "status", "recorded_checks", "baseline_advanced"),
    [
        pytest.param(
            checks_pass,
            CHECKS,
            "committed",
            KeepChecks(configured=True, passed=True),
            True,
            id="passing-checks-keep-it",
        ),
        pytest.param(
            checks_fail,
            CHECKS,
            "blocked",
            failed_checks(CHECKS_STDOUT, CHECKS_STDERR),
            False,
            id="failing-checks-refuse-it",
        ),
        pytest.param(
            _checks_skipped,
            None,
            "committed",
            KeepChecks(configured=False),
            True,
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
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    committed = commit_experiment_directly(repo)
    install_checks(monkeypatch)

    result = await keep_session(repo, checks_config(checks=checks))

    assert (result.record.status, result.record.checks) == (status, recorded_checks)
    assert (head_of(baseline_worktree_dir(repo)) == committed) is baseline_advanced
    assert settling_record_of(repo) == result.record


async def test_keep_session_when_head_matches_baseline_does_append_nothing_to_commit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    # The iteration measured something but the agent made no changes.
    start_with(repo, (iteration_record(seq=1),))
    recorder = checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config())

    hint = _hint_line(result.report, "iterate")
    assert recorder.calls == []
    assert_settling_record(
        result.record,
        blocked_keep(1, reason="nothing-to-commit", checks=KeepChecks(configured=True)),
    )
    assert "gymrat" not in hint
    assert "keep" not in hint
    _assert_closes_on_a_bare_hint(repo, result.report)


async def test_keep_session_when_nothing_new_after_prior_keep_does_append_nothing_to_commit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    await keep_session(repo, checks_config())
    append_records(repo, iteration_record(seq=2))

    result = await keep_session(repo, checks_config())

    assert_settling_record(
        result.record,
        blocked_keep(2, reason="nothing-to-commit", checks=KeepChecks(configured=True)),
    )


@pytest.mark.parametrize(
    ("history", "seq"),
    [
        pytest.param((), 1, id="no-iteration-ever-recorded"),
        pytest.param(
            (iteration_record(seq=1), committed_keep(1)), 2, id="last-iteration-already-kept"
        ),
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
    assert_settling_record(
        result.record,
        blocked_keep(seq, reason="nothing-measured", checks=KeepChecks(configured=True)),
    )
    assert "run iterate first" in result.report
    _assert_closes_on_a_bare_hint(repo, result.report)


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
    keeps = records_of_type(repo, KeepRecord)
    assert [record.seq for record in keeps] == [1, 2]
    assert result.record.seq == 2


# ---------------------------------------------------------------------------
# keep_session hint tests
# ---------------------------------------------------------------------------


def _nothing_measured(repo: str) -> None:
    """An edited experiment with no iteration measured behind it."""
    start_with(repo, ())
    edit_experiment(repo)


def _edited_after_iteration(repo: str) -> None:
    """The ordinary keep shape: one measured iteration and an edit to commit."""
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)


async def test_keep_session_when_colored_does_paint_the_hint_dim_around_a_blue_command(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _nothing_measured(repo)
    checks_pass(monkeypatch)

    result = await keep_session(repo, checks_config(), color=True)

    hint = next(line for line in result.report.split("\n") if "iterate" in strip_ansi(line))
    assert hint.startswith("\x1b[2m")
    assert any("34" in run.split(";") for run in SGR_RE.findall(hint))


async def test_keep_session_when_checks_output_holds_markup_metacharacters_does_report_it_literally(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _edited_after_iteration(repo)
    noisy = "FAIL [i] parse_config"
    install_exec(monkeypatch, KEEP_EXEC, expected_result(noisy, exit_code=1))

    result = await keep_session(repo, checks_config(), color=True)

    assert noisy in strip_ansi(result.report)


@pytest.mark.parametrize(
    ("variables", "tty", "expect_dim"),
    [
        pytest.param(["FORCE_COLOR"], False, True, id="force-color-without-tty"),
        pytest.param(["NO_COLOR"], True, False, id="no-color-on-a-tty"),
        pytest.param([], True, True, id="tty"),
        pytest.param([], False, False, id="no-tty"),
    ],
)
async def test_keep_session_when_no_checks_and_no_color_given_does_dim_hint_per_env_and_tty(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    variables: list[str],
    tty: bool,
    expect_dim: bool,
):
    for name in variables:
        monkeypatch.setenv(name, "1")
    stderr = FakeStream(tty=tty)
    monkeypatch.setattr("sys.stderr", stderr)
    _edited_after_iteration(repo)
    install_exec(monkeypatch, KEEP_EXEC, UNUSED_EXEC)

    await keep_session(repo, checks_config(checks=None))

    hint = stderr.getvalue().splitlines()[1]
    assert hint.startswith("\x1b[2m") is expect_dim


async def test_keep_session_when_no_checks_and_warn_sink_does_send_plain_hint_to_the_sink(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    _edited_after_iteration(repo)
    install_exec(monkeypatch, KEEP_EXEC, UNUSED_EXEC)
    warned: list[str] = []

    await keep_session(repo, checks_config(checks=None), KeepOptions(warn=warned.append))

    message = "\n".join(warned)
    assert "no checks command is configured" in message
    assert "gymrat.toml" in message
    assert SGR_RE.search(message) is None
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# keep_session baseline bookkeeping
# ---------------------------------------------------------------------------


async def test_keep_session_when_committed_does_append_the_kept_samples_as_a_baseline(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    kept = measured_rounds(1)
    start_with(repo, (kept,))
    edit_experiment(repo)
    checks_pass(monkeypatch)

    await keep_session(repo, checks_config())

    records = log_records(repo)
    baseline = records[-1]
    assert [record.type for record in records[-2:]] == ["keep", "baseline"]
    assert isinstance(baseline, BaselineRecord)
    assert baseline.label == head_of(experiment_worktree_dir(repo))[:SHORT_SHA_LENGTH]
    assert baseline.samples == kept.samples.experiment
    assert baseline.duration_ms is None
