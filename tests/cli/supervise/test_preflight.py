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

from gymrat.cli.supervise.preflight import (
    PreflightFlags,
    doctor_gate,
    run_preflight,
    validate_experiment_worktree,
)
from gymrat.config import ResolvedConfig, StopConfig
from gymrat.doctor import Check, DoctorReport
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.loop.iterate.run import stop_condition
from gymrat.loop.start import StartResult, start_session
from gymrat.session.lock import acquire_lock
from gymrat.session.paths import (
    archived_session_path,
    experiment_worktree_dir,
    lockfile_path,
    session_jsonl_path,
)
from gymrat.session.records import BaselineRecord, FinalizeRecord
from tests._ansi import strip_ansi
from tests._config import resolved_config
from tests._doctor_fixtures import single_check_report
from tests._lock import hold_lock
from tests.cli._session import (
    close_session_with_one_keep,
    last_command_record,
)
from tests.cli.supervise._fixtures import (
    install_baseline_seam,
    start_open_session,
)
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import (
    append_records,
    baseline_record,
    committed_keep,
    finalize_record,
    iteration_record,
    log_records,
    records_of_type,
    session_header_of,
    tear_final_line,
)
from tests.telemetry._fixtures import memory_tracing

if TYPE_CHECKING:
    from collections.abc import Callable

_MODULE = "gymrat.cli.supervise.preflight"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def seed_session_with_baseline(repo: str, *, baseline_duration_ms: float) -> None:
    """Open a session and append a single baseline record with the given duration."""
    start_open_session(repo)
    append_records(
        repo,
        baseline_record(label=".gymrat/worktrees/baseline", duration_ms=baseline_duration_ms),
    )


def seed_session_with_iteration(
    repo: str,
    *,
    iteration_duration_ms: float,
    include_baseline: bool = True,
) -> None:
    """Seed a session whose iteration carries the given duration.

    The seeded baseline gets no ``duration_ms`` of its own, so any feasibility
    math a test exercises is driven entirely by the iteration's duration.

    Args:
        repo: The repository to open the session in.
        iteration_duration_ms: The duration recorded on the seeded iteration.
        include_baseline: Whether to seed a baseline record before the iteration.
    """
    start_open_session(repo)
    if include_baseline:
        append_records(repo, baseline_record(label=".gymrat/worktrees/baseline"))
    append_records(repo, iteration_record(duration_ms=iteration_duration_ms))


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
    """Report which command holds the repo lock, by trying to take it.

    When the current process already holds the lock, ``acquire_lock`` raises,
    confirming the lock is active, and the holder record names the command.

    Args:
        lock_path: The repository lock file to probe.

    Returns:
        The holder's ``command`` while the lock is held, else ``None``.
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


def _install_doctor_seam(monkeypatch: pytest.MonkeyPatch, report: DoctorReport) -> None:
    """Replace the doctor gate's ``build_doctor_report`` with one returning *report*."""

    def fake_build(_flags: object, _cwd: object) -> DoctorReport:
        return report

    monkeypatch.setattr(f"{_MODULE}.build_doctor_report", fake_build)


# ---------------------------------------------------------------------------
# doctor gate
# ---------------------------------------------------------------------------


def test_doctor_gate_when_check_fails_does_exit_two_with_a_colored_report(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    failing = Check(name="git", status="fail", detail="missing", hint="install git")
    _install_doctor_seam(monkeypatch, report=single_check_report(failing))

    with pytest.raises(typer.Exit) as exc:
        doctor_gate(repo, color=True)

    assert exc.value.exit_code == TOOL_FAILURE_EXIT_CODE
    captured = capsys.readouterr()
    assert "install git" in strip_ansi(captured.err)
    assert "\x1b[" in captured.err


@pytest.mark.parametrize(
    "check",
    [
        pytest.param(Check(name="git", status="ok", detail="found"), id="all-pass"),
        pytest.param(
            Check(name="skill", status="warn", detail="missing", hint="run init"),
            id="warning-only",
        ),
    ],
)
def test_doctor_gate_when_nothing_fails_does_not_print(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    check: Check,
):
    _install_doctor_seam(monkeypatch, report=single_check_report(check))

    doctor_gate(repo)

    captured = capsys.readouterr()
    assert captured.err == ""


# ---------------------------------------------------------------------------
# session step
# ---------------------------------------------------------------------------


def test_preflight_when_no_session_does_open_one_announcing_its_branch(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    install_baseline_seam(monkeypatch)

    result = _run_preflight(repo)

    captured = capsys.readouterr()
    assert session_header_of(repo).branch in captured.out
    assert result.state.session is not None


def test_preflight_when_open_session_does_resume_it_with_its_history(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    append_records(repo, iteration_record(seq=1))

    result = _run_preflight(repo)

    captured = capsys.readouterr()
    assert result.state.iteration_count == 1
    assert "1 iteration" in captured.out


def test_preflight_when_finalized_session_does_replace_it_with_a_fresh_one(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_baseline_seam(monkeypatch)
    close_session_with_one_keep(repo)
    finalized_id = session_header_of(repo).session_id

    result = _run_preflight(repo)

    assert result.state.session is not None
    assert result.state.iteration_count == 0
    assert Path(archived_session_path(repo, finalized_id)).is_file()


# ---------------------------------------------------------------------------
# lock span
# ---------------------------------------------------------------------------


def test_preflight_when_running_does_hold_lock_from_session_through_baseline_measurement(
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

    monkeypatch.setattr(f"{_MODULE}.start_session", spying_start)
    monkeypatch.setattr(f"{_MODULE}.stop_condition", spying_stop)
    monkeypatch.setattr(f"{_MODULE}.measure_baseline", spying_measure)

    _run_preflight(repo)

    assert holders == {
        "start_session": "supervise",
        "stop_condition": "supervise",
        "measure_baseline": "supervise",
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
    baseline_records = records_of_type(repo, BaselineRecord)
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


def test_preflight_when_tracing_enabled_does_export_a_supervise_command_span(repo: str):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    session_id = session_header_of(repo).session_id

    with memory_tracing(session_id) as exporter:
        _run_preflight(repo)

    names = [span.name for span in exporter.get_finished_spans()]
    assert names.count("gymrat.command.supervise") == 1


@pytest.mark.parametrize("refusal", _REFUSALS)
def test_preflight_when_it_refuses_does_mark_the_supervise_command_failed(
    repo: str, refusal: Callable[[str], ResolvedConfig]
):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    session_id = session_header_of(repo).session_id
    config = refusal(repo)

    with memory_tracing(session_id) as exporter, pytest.raises(GymratError):
        _run_preflight(repo, config=config)

    command = last_command_record(repo)
    assert (command.name, command.args, command.exit_code) == (
        "supervise",
        {"stage": "preflight"},
        2,
    )
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


def test_preflight_when_no_baseline_record_does_record_a_measured_baseline(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    measure_calls = install_baseline_seam(monkeypatch)
    start_open_session(repo)

    _run_preflight(repo)

    assert len(measure_calls) == 1
    assert measure_calls[0]["target"].label == ".gymrat/worktrees/baseline"
    baseline_records = records_of_type(repo, BaselineRecord)
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


def test_preflight_when_cap_cannot_fit_one_iterate_does_raise_with_arithmetic_and_hint_leaving_the_session_open(
    repo: str,
):
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
    assert not any(isinstance(record, FinalizeRecord) for record in log_records(repo))


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

    with pytest.raises(GymratError) as exc:
        _run_preflight(repo, max_minutes=47)

    assert len(measure_calls) == 1
    assert "48m" in str(exc.value)


def test_preflight_when_force_passed_does_bypass_feasibility_check(repo: str):
    seed_session_with_baseline(repo, baseline_duration_ms=1_440_000)

    result = _run_preflight(repo, max_minutes=30, force=True)

    assert result.state.session is not None


def test_preflight_when_no_estimate_available_does_proceed_with_a_notice(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    start_open_session(repo)
    append_records(repo, baseline_record())

    result = _run_preflight(repo, max_minutes=10)

    captured = capsys.readouterr()
    assert "iterate" in captured.err.lower()
    assert result.state.session is not None


# ---------------------------------------------------------------------------
# experiment worktree validation
# ---------------------------------------------------------------------------


def test_validate_experiment_worktree_when_session_finalized_does_not_refuse_its_leftover_edits(
    repo: str,
):
    start_open_session(repo)
    append_records(repo, iteration_record(seq=1), committed_keep(seq=1), finalize_record())
    _dirty_experiment_worktree(repo)

    validate_experiment_worktree(repo)
