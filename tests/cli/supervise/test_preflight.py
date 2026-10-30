"""Tests for the supervise pre-flight module.

The pre-flight owns everything between the doctor gate and the budget file:
checks-configured warning, session open/resume under the repository lock,
stop-condition refusal, baseline measurement, and feasibility check. Each of
these is tested through the public ``run_preflight`` entry point, with seams
patched at the names ``preflight`` imports them under. The doctor gate is a
separate public entry point, ``doctor_gate``, and is tested directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import typer

from gymrat.cli.supervise.preflight import (
    PreflightFlags,
    doctor_gate,
    run_preflight,
    validate_experiment_worktree,
)
from gymrat.config import ResolvedConfig, StopConfig
from gymrat.doctor import Check, DoctorReport
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.session.paths import (
    experiment_worktree_dir,
)
from gymrat.session.records import BaselineRecord, FinalizeRecord
from tests._ansi import strip_ansi
from tests._config import resolved_config
from tests._doctor_fixtures import single_check_report
from tests.cli._session import make_discard_repo
from tests.loop._settle import start_with
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import (
    append_records,
    baseline_record,
    committed_keep,
    finalize_record,
    iteration_record,
    records_of_type,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.loop.start import StartResult

_MODULE = "gymrat.cli.supervise.preflight"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def start_open_session(repo: str) -> None:
    """Start a gymrat session so the experiment worktree and session log exist."""
    start_with(repo, config=resolved_config())


def install_baseline_seam(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the baseline measurement path so no real bench runs.

    Args:
        monkeypatch: The fixture that installs the stand-in measurement.

    Returns:
        A list that records each call's keyword arguments.
    """
    calls: list[dict[str, Any]] = []

    async def fake_measure(target: object, run_options: object) -> Any:
        calls.append({"target": target, "run_options": run_options})
        record = baseline_record(duration_ms=5000)
        result = create_measurement_result(label=record.label, samples=1, rounds=record.samples)
        return result, record

    monkeypatch.setattr("gymrat.cli.supervise.preflight.measure_baseline", fake_measure)
    return calls


def seed_session_with_baseline(repo: str, *, baseline_duration_ms: float) -> None:
    """Open a session and append a single baseline record with the given duration."""
    start_open_session(repo)
    append_records(
        repo,
        baseline_record(label=".gymrat/worktrees/baseline", duration_ms=baseline_duration_ms),
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


def test_doctor_gate_when_nothing_fails_does_not_print(
    repo: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    passing = Check(name="git", status="ok", detail="found")
    _install_doctor_seam(monkeypatch, report=single_check_report(passing))

    doctor_gate(repo)

    captured = capsys.readouterr()
    assert captured.err == ""


# ---------------------------------------------------------------------------
# command record
# ---------------------------------------------------------------------------


def _met_stop_condition(_repo: str) -> ResolvedConfig:
    """A config whose stop condition the seeded session already meets."""
    return resolved_config(stop=StopConfig(max_iterations=0))


def _dirty_working_tree(repo: str) -> ResolvedConfig:
    """Leave an uncommitted file in the main working tree; return the default config."""
    (Path(repo) / "uncommitted.txt").write_text("dirty\n", encoding="utf-8")
    return resolved_config()


def _dirty_experiment_worktree(repo: str) -> ResolvedConfig:
    """Leave one unmeasured edit in the experiment worktree; return the default config."""
    (Path(experiment_worktree_dir(repo)) / "scratch.txt").write_text("dirty\n", encoding="utf-8")
    return resolved_config()


_REFUSALS = [
    pytest.param(_met_stop_condition, "(0 of 0)", id="stop-condition"),
    pytest.param(_dirty_working_tree, "1 uncommitted file", id="dirty-working-tree"),
    pytest.param(_dirty_experiment_worktree, "1 unmeasured edit", id="dirty-experiment-worktree"),
]


@pytest.mark.parametrize(("refusal", "named"), _REFUSALS)
def test_preflight_when_it_refuses_does_fail_the_supervise_command_with_the_refusal(
    repo: str,
    refusal: Callable[[str], ResolvedConfig],
    named: str,
):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    config = refusal(repo)

    with pytest.raises(GymratError) as exc:
        _run_preflight(repo, config=config)

    assert named in str(exc.value)


# ---------------------------------------------------------------------------
# warning sink
# ---------------------------------------------------------------------------


def _leave_tree_clean(_repo: str) -> None:
    """Leave the seeded repository untouched."""


@pytest.mark.parametrize(
    ("arrange", "config_overrides", "preflight_kwargs", "named"),
    [
        pytest.param(
            _leave_tree_clean,
            {"checks": None},
            {},
            "checks",
            id="checks-not-configured",
        ),
        pytest.param(
            _leave_tree_clean,
            {"checks": "npm test", "stop": StopConfig(max_iterations=0)},
            {"force": True},
            "(0 of 0)",
            id="stop-condition-forced",
        ),
        pytest.param(
            _dirty_working_tree,
            {"checks": "npm test"},
            {"allow_dirty": True},
            "1 dirty file",
            id="dirty-tree-allowed",
        ),
    ],
)
def test_preflight_when_warning_raised_does_print_it_to_stderr(
    repo: str,
    capsys: pytest.CaptureFixture[str],
    *,
    arrange: Callable[[str], object],
    config_overrides: dict[str, Any],
    preflight_kwargs: dict[str, Any],
    named: str,
):
    seed_session_with_baseline(repo, baseline_duration_ms=1000)
    arrange(repo)

    _run_preflight(repo, config=resolved_config(**config_overrides), **preflight_kwargs)

    (warning,) = capsys.readouterr().err.splitlines()
    assert named in warning


# ---------------------------------------------------------------------------
# dirty-tree guards
# ---------------------------------------------------------------------------


def test_preflight_when_experiment_worktree_dirty_with_unsettled_does_refuse_with_settle_hint(
    repo: str,
):
    make_discard_repo(repo)
    _dirty_experiment_worktree(repo)

    with pytest.raises(GymratError) as exc:
        _run_preflight(repo)

    assert "unsettled" in str(exc.value)


# ---------------------------------------------------------------------------
# baseline measurement
# ---------------------------------------------------------------------------


def test_preflight_when_no_baseline_record_does_record_a_measured_baseline(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_baseline_seam(monkeypatch)
    start_open_session(repo)

    _run_preflight(repo)

    assert records_of_type(repo, BaselineRecord) == [baseline_record(duration_ms=5000)]


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


def test_preflight_when_cap_cannot_fit_one_iterate_does_refuse_with_the_arithmetic_leaving_the_session_open(
    repo: str,
):
    seed_session_with_baseline(repo, baseline_duration_ms=1_440_000)

    with pytest.raises(GymratError) as exc:
        _run_preflight(repo, max_minutes=30)

    assert all(part in str(exc.value) for part in ("baseline", "24m", "48m", "30m"))
    assert records_of_type(repo, FinalizeRecord) == []


def test_preflight_when_force_passed_does_bypass_feasibility_check(repo: str):
    seed_session_with_baseline(repo, baseline_duration_ms=1_440_000)

    _run_preflight(repo, max_minutes=30, force=True)


def test_preflight_when_no_estimate_available_does_proceed_with_a_notice(
    repo: str, capsys: pytest.CaptureFixture[str]
):
    start_open_session(repo)
    append_records(repo, baseline_record())

    _run_preflight(repo, config=resolved_config(checks="npm test"), max_minutes=10)

    assert len(capsys.readouterr().err.splitlines()) == 1


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
