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
from typing import TYPE_CHECKING, Any

import pytest
import typer
from opentelemetry.trace import StatusCode

from gymrat.cli.supervise.preflight import PreflightFlags, doctor_gate, run_preflight
from gymrat.config import ResolvedConfig, StopConfig
from gymrat.doctor import (
    Check,
    CheckSection,
    DoctorReport,
    EnvironmentInfo,
    create_doctor_report,
)
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.loop.iterate.run import stop_condition
from gymrat.loop.start import StartResult, start_session
from gymrat.session.lock import acquire_lock
from gymrat.session.paths import experiment_worktree_dir, lockfile_path, session_jsonl_path
from gymrat.session.records import BaselineRecord, FinalizeRecord
from gymrat.session.store import append_record
from tests._config import resolved_config
from tests.cli._session import last_command_record
from tests.cli.supervise._fixtures import (
    finalized_session,
    install_baseline_seam,
    start_open_session,
)
from tests.conftest import hold_lock
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import (
    baseline_record,
    iteration_record,
    log_records,
    session_header_of,
    tear_final_line,
)
from tests.telemetry._fixtures import (
    isolate_tracing_provider as _isolate_tracing_provider,  # noqa: F401 -- registers the autouse fixture
)
from tests.telemetry._fixtures import memory_tracing

if TYPE_CHECKING:
    from collections.abc import Callable

_MODULE = "gymrat.cli.supervise.preflight"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def seed_session_with_baseline(
    repo: str, *, baseline_duration_ms: float, label: str = ".gymrat/worktrees/baseline"
) -> None:
    """Open a session and append a single baseline record with the given duration."""
    start_open_session(repo)
    log = session_jsonl_path(repo)
    append_record(log, baseline_record(label=label, duration_ms=baseline_duration_ms))


def seed_session_with_iteration(
    repo: str,
    *,
    iteration_duration_ms: float,
    include_baseline: bool = True,
    label: str = ".gymrat/worktrees/baseline",
) -> None:
    """Seed a session whose iteration carries the given duration.

    The seeded baseline (when included) gets no ``duration_ms`` of its own —
    there is no parameter to set one — so any feasibility math a test exercises
    is driven entirely by ``iteration_duration_ms``.
    """
    start_open_session(repo)
    log = session_jsonl_path(repo)
    if include_baseline:
        append_record(log, baseline_record(label=label))
    append_record(log, iteration_record(duration_ms=iteration_duration_ms))


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


def _warning_report() -> DoctorReport:
    return create_doctor_report(
        _env(),
        [
            CheckSection(
                title="Environment",
                checks=[Check(name="skill", status="warn", detail="missing", hint="run init")],
            )
        ],
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
    allow_dirty: bool = False,
) -> StartResult:
    return run_preflight(
        root=repo,
        config=config if config is not None else resolved_config(),
        flags=PreflightFlags(
            baseline_ref=baseline_ref,
            max_minutes=max_minutes,
            force=force,
            allow_dirty=allow_dirty,
        ),
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


# ---------------------------------------------------------------------------
# doctor gate
# ---------------------------------------------------------------------------


def test_doctor_gate_when_check_fails_does_exit_two_with_report(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _install_doctor_seam(monkeypatch, report=_failing_report())

    with pytest.raises(typer.Exit) as exc:
        doctor_gate(repo)

    assert exc.value.exit_code == TOOL_FAILURE_EXIT_CODE
    captured = capsys.readouterr()
    assert "install git" in captured.err


@pytest.mark.parametrize(
    "make_report",
    [
        pytest.param(_ok_report, id="all-pass"),
        pytest.param(_warning_report, id="warning-only"),
    ],
)
def test_doctor_gate_when_nothing_fails_does_not_print(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    make_report: Callable[[], DoctorReport],
):
    _install_doctor_seam(monkeypatch, report=make_report())

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
    install_baseline_seam(monkeypatch)

    result = _run_preflight(repo)

    captured = capsys.readouterr()
    assert session_header_of(repo).branch in captured.out
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
    install_baseline_seam(monkeypatch)
    finalized_session(repo)

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


def test_preflight_when_repository_lock_held_does_refuse_with_the_contention_error_without_opening_a_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_baseline_seam(monkeypatch)
    lock_path = lockfile_path(repo)
    blocker = hold_lock(lock_path, "iterate")
    try:
        with pytest.raises(GymratError) as rival:
            acquire_lock(lock_path, "measure")

        with pytest.raises(GymratError) as refused:
            _run_preflight(repo)
    finally:
        blocker.release()

    assert (str(refused.value), refused.value.hint) == (str(rival.value), rival.value.hint)
    assert not Path(session_jsonl_path(repo)).exists()


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
    install_baseline_seam(monkeypatch)

    _run_preflight(repo)

    session_header_of(repo)
    baseline_records = [r for r in log_records(repo) if isinstance(r, BaselineRecord)]
    assert len(baseline_records) == 1


# ---------------------------------------------------------------------------
# command record
# ---------------------------------------------------------------------------


def test_preflight_when_it_opens_a_session_does_append_a_supervise_record_for_the_preflight_stage(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_baseline_seam(monkeypatch)

    _run_preflight(repo)

    command = last_command_record(repo)
    assert (command.name, command.args, command.exit_code) == (
        "supervise",
        {"stage": "preflight"},
        0,
    )


def _met_stop_condition(_repo: str) -> ResolvedConfig:
    """A config whose stop condition the seeded session already meets."""
    return resolved_config(stop=StopConfig(max_iterations=0))


def _dirty_working_tree(repo: str) -> ResolvedConfig:
    """Leave an uncommitted file in the main working tree; return the default config."""
    (Path(repo) / "uncommitted.txt").write_text("dirty\n", encoding="utf-8")
    return resolved_config()


def _dirty_experiment_worktree(repo: str) -> ResolvedConfig:
    """Leave an unmeasured edit in the experiment worktree; return the default config."""
    (Path(experiment_worktree_dir(repo)) / "scratch.txt").write_text("dirty\n", encoding="utf-8")
    return resolved_config()


_REFUSALS = [
    pytest.param(_met_stop_condition, id="stop-condition"),
    pytest.param(_dirty_working_tree, id="dirty-working-tree"),
    pytest.param(_dirty_experiment_worktree, id="dirty-experiment-worktree"),
]


@pytest.mark.parametrize("refusal", _REFUSALS)
def test_preflight_when_it_refuses_does_record_the_refusal_on_the_command_record(
    repo: str, refusal: Callable[[str], ResolvedConfig]
):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    config = refusal(repo)

    with pytest.raises(GymratError):
        _run_preflight(repo, config=config)

    command = last_command_record(repo)
    assert (command.name, command.args, command.exit_code) == (
        "supervise",
        {"stage": "preflight"},
        2,
    )


def test_preflight_when_tracing_enabled_does_export_a_supervise_command_span(repo: str):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    session_id = session_header_of(repo).session_id

    with memory_tracing(session_id) as exporter:
        _run_preflight(repo)

    names = [span.name for span in exporter.get_finished_spans()]
    assert names.count("gymrat.command.supervise") == 1


@pytest.mark.parametrize("refusal", _REFUSALS)
def test_preflight_when_it_refuses_with_tracing_enabled_does_export_the_supervise_command_span_with_error_status(
    repo: str, refusal: Callable[[str], ResolvedConfig]
):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    session_id = session_header_of(repo).session_id
    config = refusal(repo)

    with memory_tracing(session_id) as exporter, pytest.raises(GymratError):
        _run_preflight(repo, config=config)

    statuses = [
        span.status.status_code
        for span in exporter.get_finished_spans()
        if span.name == "gymrat.command.supervise"
    ]
    assert statuses == [StatusCode.ERROR]


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
# warning sink
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config_overrides", "preflight_kwargs", "expected"),
    [
        pytest.param(
            {"checks": None},
            {},
            "warning: checks is not configured — keep will commit with the gate off",
            id="checks-not-configured",
        ),
        pytest.param(
            {"checks": "npm test"},
            {"baseline_ref": "some-branch"},
            "warning: --baseline some-branch ignored because the session was resumed",
            id="baseline-ref-ignored",
        ),
        pytest.param(
            {"checks": "npm test", "stop": StopConfig(max_iterations=0)},
            {"force": True},
            "warning: Stop condition met: max iterations (0 of 0)",
            id="stop-condition-forced",
        ),
    ],
)
def test_preflight_when_warning_raised_does_route_it_to_the_warn_sink(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    config_overrides: dict[str, Any],
    preflight_kwargs: dict[str, Any],
    expected: str,
):
    warnings: list[str] = []
    monkeypatch.setattr(f"{_MODULE}.warn_to_stderr", warnings.append)
    seed_session_with_baseline(repo, baseline_duration_ms=1000)

    _run_preflight(repo, config=resolved_config(**config_overrides), **preflight_kwargs)

    assert warnings == [expected]


def test_preflight_when_tree_dirty_and_allowed_does_route_warning_to_the_warn_sink(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    warnings: list[str] = []
    monkeypatch.setattr(f"{_MODULE}.warn_to_stderr", warnings.append)
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    _dirty_working_tree(repo)

    _run_preflight(repo, config=resolved_config(checks="npm test"), allow_dirty=True)

    assert warnings == [
        "warning: working tree has 1 dirty file — proceeding because --allow-dirty was set"
    ]


# ---------------------------------------------------------------------------
# baseline measurement
# ---------------------------------------------------------------------------


def test_preflight_when_no_baseline_record_does_measure_and_append(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    measure_calls = install_baseline_seam(monkeypatch)
    start_open_session(repo)

    _run_preflight(repo)

    assert len(measure_calls) == 1
    assert measure_calls[0]["target"].label == ".gymrat/worktrees/baseline"
    records = log_records(repo)
    baseline_records = [r for r in records if isinstance(r, BaselineRecord)]
    assert len(baseline_records) == 1
    assert baseline_records[0].duration_ms is not None


def test_preflight_when_baseline_already_recorded_does_not_measure(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    measure_calls = install_baseline_seam(monkeypatch)
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
    measure_calls = install_baseline_seam(monkeypatch)
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

    records = log_records(repo)
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
    install_baseline_seam(monkeypatch)

    result = _run_preflight(repo)

    assert isinstance(result, StartResult)


# ---------------------------------------------------------------------------
# doctor gate color forwarding
# ---------------------------------------------------------------------------


def test_doctor_gate_when_color_true_does_produce_ansi_on_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _install_doctor_seam(monkeypatch, report=_failing_report())

    with pytest.raises(typer.Exit):
        doctor_gate(repo, color=True)

    captured = capsys.readouterr()
    assert "\x1b[" in captured.err
