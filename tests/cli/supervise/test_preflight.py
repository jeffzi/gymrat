"""Tests for the supervise pre-flight module.

The pre-flight owns everything between the doctor gate and the budget file:
doctor check, checks-configured warning, session open/resume under the
repository lock, stop-condition refusal, baseline measurement, and feasibility
check. Each behavior is tested through the public ``run_preflight`` entry
point, with seams patched at the names ``preflight`` imports them under.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gymrat.cli.supervise.preflight import doctor_gate, run_preflight
from gymrat.config import ResolvedConfig, StopConfig
from gymrat.doctor.checks import (
    Check,
    CheckSection,
    DoctorReport,
    EnvironmentInfo,
    create_doctor_report,
)
from gymrat.errors import GymratError
from gymrat.loop.finalize import finalize_session
from gymrat.loop.iterate.run import stop_condition
from gymrat.loop.start import StartResult, start_session
from gymrat.session import (
    BaselineRecord,
    FinalizeRecord,
    SessionRecord,
    append_record,
    read_records,
    session_jsonl_path,
)
from gymrat.session.lock import acquire_lock
from gymrat.session.paths import lockfile_path
from tests._git import run_git
from tests.cli.supervise._fixtures import (
    baseline_record,
    seed_session_with_baseline,
    seed_session_with_iteration,
    start_open_session,
)
from tests.loop.iterate._fixtures import resolved_config
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import committed_keep, iteration_record, tear_final_line

_MODULE = "gymrat.cli.supervise.preflight"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _env() -> EnvironmentInfo:
    return EnvironmentInfo(gymrat_version="0.1.0", python_version="3.14.0", platform="darwin")


def _ok_check(name: str = "git") -> Check:
    return Check(name=name, status="ok", detail="found")


def _failing_check(name: str = "git") -> Check:
    return Check(name=name, status="fail", detail="missing", hint="install git")


def _ok_report() -> DoctorReport:
    return create_doctor_report(
        _env(),
        [CheckSection(title="Environment", checks=[_ok_check()])],
    )


def _failing_report() -> DoctorReport:
    return create_doctor_report(
        _env(),
        [CheckSection(title="Environment", checks=[_failing_check()])],
    )


def _run_preflight(
    repo: str,
    *,
    config: ResolvedConfig | None = None,
    baseline_ref: str | None = None,
    max_minutes: float = 60,
    force: bool = False,
) -> StartResult:
    return run_preflight(
        root=repo,
        config=config if config is not None else resolved_config(),
        baseline_ref=baseline_ref,
        max_minutes=max_minutes,
        force=force,
    )


def _probe_lock(lock_path: str) -> str | None:
    """Return the holder ``command`` if the repo lock is held, else ``None``.

    Attempts to acquire the lock; when the current process already holds it,
    ``acquire_lock`` raises, confirming the lock is active. The holder's
    ``command`` field is then read from the lock file.
    """
    try:
        release = acquire_lock(lock_path, "probe")
        release()
    except GymratError:
        return json.loads(Path(lock_path).read_text(encoding="utf-8")).get("command")
    else:
        return None


def _assert_lock_released(repo: str) -> None:
    """Verify the repo lock is free by acquiring and immediately releasing it."""
    release = acquire_lock(lockfile_path(repo), "probe")
    release()


# ---------------------------------------------------------------------------
# seam installation
# ---------------------------------------------------------------------------


def _install_doctor_seam(
    monkeypatch: pytest.MonkeyPatch,
    report: DoctorReport | None = None,
) -> list[dict[str, object]]:
    """Replace the doctor gate's ``build_doctor_report`` with a fixed report.

    Returns the list of ``(flags, cwd)`` pairs it was called with.
    """
    calls: list[dict[str, object]] = []
    result = report if report is not None else _ok_report()

    def fake_build(flags: object, cwd: object) -> DoctorReport:
        calls.append({"flags": flags, "cwd": cwd})
        return result

    monkeypatch.setattr(f"{_MODULE}.build_doctor_report", fake_build)
    return calls


def _install_baseline_seam(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the baseline measurement path so no real bench runs.

    Returns a list that records each call's keyword arguments.
    """
    calls: list[dict[str, Any]] = []

    async def fake_measure(target: object, run_options: object) -> Any:
        calls.append({"target": target, "run_options": run_options})
        record = baseline_record(duration_ms=5000)
        result = create_measurement_result(label=record.label, samples=1, rounds=record.samples)
        return result, record

    monkeypatch.setattr(f"{_MODULE}.measure_baseline", fake_measure)
    return calls


# ---------------------------------------------------------------------------
# doctor gate
# ---------------------------------------------------------------------------


def test_doctor_gate_when_check_fails_does_exit_two_with_report(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _install_doctor_seam(monkeypatch, report=_failing_report())

    with pytest.raises(SystemExit) as exc:
        doctor_gate(repo)

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "install git" in captured.err


def test_doctor_gate_when_all_pass_does_not_print(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _install_doctor_seam(monkeypatch, report=_ok_report())

    doctor_gate(repo)

    captured = capsys.readouterr()
    assert captured.err == ""


# ---------------------------------------------------------------------------
# checks warning
# ---------------------------------------------------------------------------


def test_preflight_when_checks_not_configured_does_warn_on_stderr(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    config = resolved_config(checks=None)

    _run_preflight(repo, config=config)

    captured = capsys.readouterr()
    assert "checks is not configured" in captured.err
    assert "gate off" in captured.err


def test_preflight_when_checks_configured_does_not_warn(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    config = resolved_config(checks="npm test")

    _run_preflight(repo, config=config)

    captured = capsys.readouterr()
    assert "checks is not configured" not in captured.err


# ---------------------------------------------------------------------------
# session step
# ---------------------------------------------------------------------------


def test_preflight_when_no_session_does_open_and_print_summary_to_stdout(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _install_baseline_seam(monkeypatch)

    result = _run_preflight(repo)

    captured = capsys.readouterr()
    records = read_records(session_jsonl_path(repo))
    header = records[0]
    assert isinstance(header, SessionRecord)
    assert header.branch in captured.out
    assert result.state.session is not None


def test_preflight_when_open_session_does_resume_and_print_history(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    append_record(session_jsonl_path(repo), iteration_record(seq=1))

    result = _run_preflight(repo)

    captured = capsys.readouterr()
    assert result.state.iteration_count == 1
    assert "1 iteration" in captured.out


def test_preflight_when_open_session_and_baseline_given_does_warn_ref_ignored(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)

    _run_preflight(repo, baseline_ref="some-branch")

    captured = capsys.readouterr()
    assert "ignored" in captured.err.lower()


def test_preflight_when_finalized_session_does_archive_and_open_fresh(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_baseline_seam(monkeypatch)
    start_open_session(repo)
    worktree = Path(repo) / ".gymrat" / "worktrees" / "experiment"
    (worktree / "README.md").write_text("# edit\n", encoding="utf-8")

    run_git(["add", "README.md"], str(worktree))
    run_git(["commit", "-m", "edit"], str(worktree))
    commit = run_git(["rev-parse", "HEAD"], str(worktree)).strip()
    append_record(session_jsonl_path(repo), iteration_record(seq=1))
    append_record(session_jsonl_path(repo), committed_keep(1, commit=commit))
    finalize_session(repo)

    result = _run_preflight(repo)

    assert result.state.session is not None
    assert result.state.iteration_count == 0


# ---------------------------------------------------------------------------
# lock span
# ---------------------------------------------------------------------------


def test_preflight_when_running_does_hold_lock_from_session_through_feasibility(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    lock_path = lockfile_path(repo)
    holders: dict[str, str | None] = {}
    original_start = start_session

    def spying_start(root: str, ref: str | None, config: ResolvedConfig) -> Any:
        holders["start_session"] = _probe_lock(lock_path)
        return original_start(root, ref, config)

    def spying_stop(config: ResolvedConfig, state: Any) -> Any:
        holders["stop_condition"] = _probe_lock(lock_path)
        return stop_condition(config, state)

    async def spying_measure(target: object, run_options: object) -> Any:
        holders["measure_baseline"] = _probe_lock(lock_path)
        record = baseline_record(duration_ms=5000)
        result = create_measurement_result(
            label=record.label,
            samples=1,
            rounds=record.samples,
        )
        return result, record

    import gymrat.cli.supervise.preflight as _pm

    original_feasibility = _pm._check_feasibility

    def spying_feasibility(root: str, *, max_minutes: float, force: bool) -> None:
        holders["feasibility"] = _probe_lock(lock_path)
        original_feasibility(root, max_minutes=max_minutes, force=force)

    monkeypatch.setattr(f"{_MODULE}.start_session", spying_start)
    monkeypatch.setattr(f"{_MODULE}.stop_condition", spying_stop)
    monkeypatch.setattr(f"{_MODULE}.measure_baseline", spying_measure)
    monkeypatch.setattr(f"{_MODULE}._check_feasibility", spying_feasibility)

    _run_preflight(repo)

    assert holders == {
        "start_session": "supervise",
        "stop_condition": "supervise",
        "measure_baseline": "supervise",
        "feasibility": "supervise",
    }


# ---------------------------------------------------------------------------
# torn-tail repair
# ---------------------------------------------------------------------------


def test_preflight_when_log_has_torn_tail_does_truncate_before_session_opens(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    start_open_session(repo)
    log_path = session_jsonl_path(repo)
    tear_final_line(log_path)
    _install_baseline_seam(monkeypatch)

    _run_preflight(repo)

    records = read_records(log_path)
    assert isinstance(records[0], SessionRecord)
    baseline_records = [r for r in records if isinstance(r, BaselineRecord)]
    assert len(baseline_records) == 1


# ---------------------------------------------------------------------------
# stop condition
# ---------------------------------------------------------------------------


def test_preflight_when_stop_condition_met_does_raise_with_message_and_hint(repo: str):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    config = resolved_config(stop=StopConfig(max_iterations=0))

    with pytest.raises(GymratError, match="Stop condition met") as exc:
        _run_preflight(repo, config=config)

    assert exc.value.hint is not None
    assert "new session" in exc.value.hint.lower()


def test_preflight_when_stop_condition_met_and_force_does_warn_and_proceed(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    config = resolved_config(stop=StopConfig(max_iterations=0))

    result = _run_preflight(repo, config=config, force=True)

    captured = capsys.readouterr()
    assert "Stop condition met" in captured.err
    assert result.state.session is not None


# ---------------------------------------------------------------------------
# baseline measurement
# ---------------------------------------------------------------------------


def test_preflight_when_no_baseline_record_does_measure_and_append(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    measure_calls = _install_baseline_seam(monkeypatch)
    start_open_session(repo)

    _run_preflight(repo)

    assert len(measure_calls) == 1
    assert measure_calls[0]["target"].label == ".gymrat/worktrees/baseline"
    records = read_records(session_jsonl_path(repo))
    baseline_records = [r for r in records if isinstance(r, BaselineRecord)]
    assert len(baseline_records) == 1
    assert baseline_records[0].duration_ms is not None


def test_preflight_when_baseline_already_recorded_does_not_measure(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    measure_calls = _install_baseline_seam(monkeypatch)
    seed_session_with_baseline(repo, baseline_duration_ms=5000)

    _run_preflight(repo)

    assert len(measure_calls) == 0


# ---------------------------------------------------------------------------
# feasibility check
# ---------------------------------------------------------------------------


def test_preflight_when_cap_cannot_fit_one_iterate_does_raise_with_arithmetic_and_hint(repo: str):
    seed_session_with_baseline(repo, baseline_duration_ms=1_440_000)

    with pytest.raises(GymratError) as exc:
        _run_preflight(repo, max_minutes=30)

    text = str(exc.value)
    assert "24m" in text
    assert "48m" in text
    assert "30m" in text
    assert exc.value.hint is not None
    assert "--max-minutes" in exc.value.hint
    assert "--force" in exc.value.hint


def test_preflight_when_session_has_baseline_does_need_one_iterate(repo: str):
    seed_session_with_iteration(repo, iteration_duration_ms=2_880_000, include_baseline=True)

    with pytest.raises(GymratError) as exc:
        _run_preflight(repo, max_minutes=47)

    assert "48m" in str(exc.value)


def test_preflight_when_session_lacks_baseline_does_measure_then_charge_one_iterate(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    measure_calls = _install_baseline_seam(monkeypatch)
    seed_session_with_iteration(repo, iteration_duration_ms=2_880_000, include_baseline=False)

    result = _run_preflight(repo, max_minutes=50)

    assert len(measure_calls) == 1
    assert result.state.session is not None


def test_preflight_when_force_passed_does_bypass_feasibility_check(repo: str):
    seed_session_with_baseline(repo, baseline_duration_ms=1_440_000)

    result = _run_preflight(repo, max_minutes=30, force=True)

    assert result.state.session is not None


def test_preflight_when_infeasible_does_leave_session_open(repo: str):
    seed_session_with_baseline(repo, baseline_duration_ms=1_440_000)

    with pytest.raises(GymratError):
        _run_preflight(repo, max_minutes=30)

    records = read_records(session_jsonl_path(repo))
    assert not any(isinstance(r, FinalizeRecord) for r in records)


def test_preflight_when_no_estimate_available_does_print_info_on_stderr_and_proceed(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    start_open_session(repo)
    append_record(session_jsonl_path(repo), baseline_record())

    result = _run_preflight(repo, max_minutes=10)

    captured = capsys.readouterr()
    assert "iterate" in captured.err.lower()
    assert result.state.session is not None


# ---------------------------------------------------------------------------
# result type
# ---------------------------------------------------------------------------


def test_preflight_when_new_session_does_return_a_start_result(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_baseline_seam(monkeypatch)

    result = _run_preflight(repo)

    assert isinstance(result, StartResult)


# ---------------------------------------------------------------------------
# doctor gate color forwarding
# ---------------------------------------------------------------------------


def test_doctor_gate_when_color_true_does_produce_ansi_on_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _install_doctor_seam(monkeypatch, report=_failing_report())

    with pytest.raises(SystemExit):
        doctor_gate(repo, color=True)

    captured = capsys.readouterr()
    assert "\x1b[" in captured.err
