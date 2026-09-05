"""Tests for the doctor report assembly module.

The report assembly (``build_doctor_report``) coordinates the section builders,
config inspection, and git environment probe to produce a ``DoctorReport``.
These tests patch the section builders and ``inspect_config`` at their
``gymrat.doctor.report`` import targets and verify the assembly contract:

- Sections are collected in the correct order.
- Config problems and adapter flags are forwarded to the bench section.
- The ``cwd`` parameter flows through to the git probe.
- ``detect_git_environment`` maps git-missing, outside-repo, and unresolvable
  root to distinct ``GitEnvironment`` outcomes without raising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from types import SimpleNamespace

import pytest

from gymrat.config import CliFlags
from gymrat.doctor.checks import Check, DoctorReport, EnvironmentInfo
from gymrat.doctor.report import GitEnvironment, build_doctor_report, detect_git_environment
from tests.doctor._fixtures import fixed_section, patch_common_seams

_MODULE = "gymrat.doctor.report"


def _flags(**overrides: object) -> CliFlags:
    return CliFlags(**overrides)  # pyrefly: ignore


def _patch_report(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_failure: bool = False,
    bench_fail: bool = False,
    cwd_calls: list[str] | None = None,
) -> SimpleNamespace:
    """Replace every report seam and return recorded calls."""
    handles = patch_common_seams(
        monkeypatch,
        config_failure=config_failure,
        bench_fail=bench_fail,
        problems=["Config file not found"] if config_failure else [],
    )

    def fake_env_info() -> EnvironmentInfo:
        return EnvironmentInfo(gymrat_version="0.1.0", python_version="3.13.0", platform="darwin")

    monkeypatch.setattr(f"{_MODULE}._environment_info", fake_env_info)

    def fake_detect_git(cwd: str) -> GitEnvironment:
        if cwd_calls is not None:
            cwd_calls.append(cwd)
        return GitEnvironment(git_available=True, inside_git_repo=True, repo_root_dir=cwd)

    monkeypatch.setattr(f"{_MODULE}.detect_git_environment", fake_detect_git)

    monkeypatch.setattr(
        f"{_MODULE}.build_environment_section",
        fixed_section("Environment", [Check("git", "ok", "available")]),
    )

    return handles


# ---------------------------------------------------------------------------
# build_doctor_report — assembly
# ---------------------------------------------------------------------------


def test_build_doctor_report_when_healthy_does_return_four_sections(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_report(monkeypatch)

    report = build_doctor_report(_flags(), cwd="/project")

    assert isinstance(report, DoctorReport)
    titles = [s.title for s in report.sections]
    assert titles == ["Environment", "Configuration", "Workflow", "Bench"]


def test_build_doctor_report_when_called_does_pass_cwd_to_git_probe(
    monkeypatch: pytest.MonkeyPatch,
):
    cwd_calls: list[str] = []
    _patch_report(monkeypatch, cwd_calls=cwd_calls)

    build_doctor_report(_flags(), cwd="/my/project")

    assert cwd_calls == ["/my/project"]


def test_build_doctor_report_when_config_failed_and_adapter_flag_does_forward_flag_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    handles = _patch_report(monkeypatch, config_failure=True)

    build_doctor_report(_flags(adapter="custom-adapter"), cwd="/project")

    assert len(handles.bench_calls) == 1
    assert handles.bench_calls[0]["adapter"] == "custom-adapter"


@pytest.mark.parametrize(
    ("config_failure", "expected"),
    [
        pytest.param(False, False, id="config-ok"),
        pytest.param(True, True, id="config-problems"),
    ],
)
def test_build_doctor_report_when_run_does_forward_config_problems_to_bench_section(
    monkeypatch: pytest.MonkeyPatch, config_failure: bool, expected: bool
):
    handles = _patch_report(monkeypatch, config_failure=config_failure)

    build_doctor_report(_flags(), cwd="/project")

    assert len(handles.bench_calls) == 1
    assert handles.bench_calls[0]["config_problems"] is expected


def test_build_doctor_report_when_bench_fails_does_report_has_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_report(monkeypatch, bench_fail=True)

    report = build_doctor_report(_flags(), cwd="/project")

    assert report.has_failures


# ---------------------------------------------------------------------------
# detect_git_environment — three distinct non-raising outcomes
# ---------------------------------------------------------------------------


def _try_git_missing(*_args: object, **_kwargs: object) -> str:
    return "git: command not found"


def _try_git_ok(*_args: object, **_kwargs: object) -> None:
    return None


def test_detect_git_environment_when_git_missing_does_report_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(f"{_MODULE}.try_git", _try_git_missing)

    result = detect_git_environment("/some/dir")

    assert result.git_available is False
    assert result.inside_git_repo is False
    assert result.repo_root_dir is None


def test_detect_git_environment_when_outside_repo_does_report_not_in_repo(
    monkeypatch: pytest.MonkeyPatch,
):
    from gymrat.git import NotAGitRepositoryError

    monkeypatch.setattr(f"{_MODULE}.try_git", _try_git_ok)

    def fake_repo_root(cwd: str | None = None) -> str:
        msg = "not a git repository"
        raise NotAGitRepositoryError(msg)

    monkeypatch.setattr(f"{_MODULE}.repo_root", fake_repo_root)

    result = detect_git_environment("/some/dir")

    assert result.git_available is True
    assert result.inside_git_repo is False
    assert result.repo_root_dir is None


def test_detect_git_environment_when_root_unresolvable_does_report_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from gymrat.errors import GymratError

    monkeypatch.setattr(f"{_MODULE}.try_git", _try_git_ok)

    def fake_repo_root(cwd: str | None = None) -> str:
        msg = "cannot resolve"
        raise GymratError(msg)

    monkeypatch.setattr(f"{_MODULE}.repo_root", fake_repo_root)

    result = detect_git_environment(str(tmp_path))

    assert result.git_available is True
    assert result.repo_root_dir is None
    assert result.git_error is not None
