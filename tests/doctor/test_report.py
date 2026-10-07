"""Tests for the doctor report assembly module.

The report assembly (``build_doctor_report``) coordinates the section builders,
config inspection, and git environment probe to produce a ``DoctorReport``.
These tests patch the section builders and ``inspect_config`` at their
``gymrat.doctor`` import targets and verify the assembly contract:

- Sections are collected in the correct order.
- Config problems and adapter flags are forwarded to the bench section.
- The ``cwd`` parameter flows through to the git probe.
- ``detect_git_environment`` maps git-missing, outside-repo, and unresolvable
  root to distinct ``GitEnvironment`` outcomes without raising.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import SimpleNamespace

import pytest

from gymrat.config import CliFlags
from gymrat.doctor import (
    Check,
    DoctorReport,
    EnvironmentInfo,
    GitEnvironment,
    build_doctor_report,
    detect_git_environment,
)
from tests.config._toml import DEEP_NESTING_DOCUMENT, DIGIT_LIMIT_DOCUMENT, write_raw
from tests.doctor._fixtures import fixed_section, patch_common_seams

_MODULE = "gymrat.doctor"


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


# ---------------------------------------------------------------------------
# config inspection — real gymrat.toml
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("toml", "expected"),
    [
        pytest.param(
            "samples = 0\n",
            Check(
                name="config",
                status="fail",
                detail="Invalid config value for samples: expected a number at or above 1, got 0",
            ),
            id="invalid-value",
        ),
        pytest.param(
            None,
            Check(
                name="config",
                status="ok",
                detail="No config file found; operating with defaults only",
            ),
            id="no-file",
        ),
    ],
)
def test_build_doctor_report_when_config_read_for_real_does_report_its_findings(
    tmp_path: Path, toml: str | None, expected: Check
):
    if toml is not None:
        (tmp_path / "gymrat.toml").write_text(toml, encoding="utf-8")

    report = build_doctor_report(_flags(), str(tmp_path))

    config_section = next(
        section for section in report.sections if section.title == "Configuration"
    )
    assert config_section.checks == [expected]


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(DIGIT_LIMIT_DOCUMENT, id="integer-past-digit-limit"),
        pytest.param(DEEP_NESTING_DOCUMENT, id="nesting-past-recursion-limit"),
    ],
)
def test_build_doctor_report_when_config_parser_hits_interpreter_limit_does_fail_config_check(
    tmp_path: Path, document: str
):
    config_path = write_raw(tmp_path, document)

    report = build_doctor_report(_flags(), str(tmp_path))

    config_section = next(
        section for section in report.sections if section.title == "Configuration"
    )
    [check] = config_section.checks
    assert check.status == "fail"
    assert check.detail.startswith(f"Failed to parse config file at {config_path}: ")


def _check(report: DoctorReport, title: str, name: str) -> Check:
    section = next(section for section in report.sections if section.title == title)
    return next(check for check in section.checks if check.name == name)


@pytest.mark.parametrize(
    ("toml", "expected"),
    [
        pytest.param(
            "samples = 5\n",
            'Add `runbook = "gymrat-runbook.md"` to gymrat.toml. '
            "Without one, supervise has no instructions to follow.",
            id="config-file-lacks-the-key",
        ),
        pytest.param(
            None,
            "Run `gymrat init` to create a runbook, or add `runbook` to gymrat.toml. "
            "Without one, supervise has no instructions to follow.",
            id="no-config-file",
        ),
    ],
)
def test_build_doctor_report_when_runbook_unset_does_hint_the_step_that_fits_the_config_file(
    tmp_path: Path, toml: str | None, expected: str
):
    if toml is not None:
        (tmp_path / "gymrat.toml").write_text(toml, encoding="utf-8")

    report = build_doctor_report(_flags(), str(tmp_path))

    assert _check(report, "Workflow", "runbook").hint == expected


# ---------------------------------------------------------------------------
# bench executable — path-shaped token, run from a subdirectory
# ---------------------------------------------------------------------------

_posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="the executable bit is a POSIX file mode"
)


@pytest.fixture
def repo_subdirectory(repo: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A subdirectory of a scratch repository, made the process working directory."""
    subdirectory = Path(repo) / "packages" / "app"
    subdirectory.mkdir(parents=True)
    monkeypatch.chdir(subdirectory)
    return subdirectory


@pytest.mark.parametrize(
    ("bench", "script", "mode", "expected"),
    [
        pytest.param(
            "./scripts/bench.sh",
            "scripts/bench.sh",
            0o755,
            Check(name="executable", status="ok", detail="./scripts/bench.sh is executable"),
            id="executable-under-root",
        ),
        pytest.param(
            '"my scripts/bench.sh" --fast',
            "my scripts/bench.sh",
            0o755,
            Check(name="executable", status="ok", detail="my scripts/bench.sh is executable"),
            id="quoted-path-with-spaces",
        ),
        pytest.param(
            "./scripts/bench.sh",
            None,
            None,
            Check(name="executable", status="warn", detail="./scripts/bench.sh was not found"),
            id="missing",
        ),
        pytest.param(
            "./scripts/bench.sh",
            "scripts/bench.sh",
            0o644,
            Check(name="executable", status="warn", detail="./scripts/bench.sh is not executable"),
            id="not-executable",
            marks=_posix_only,
        ),
    ],
)
def test_build_doctor_report_when_bench_is_a_path_does_resolve_it_against_the_repository_root(
    repo_subdirectory: Path, bench: str, script: str | None, mode: int | None, expected: Check
):
    root = repo_subdirectory.parent.parent
    (root / "gymrat.toml").write_text(f"bench = '{bench}'\n", encoding="utf-8")
    if script is not None and mode is not None:
        script_path = root / script
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("#!/bin/sh\n", encoding="utf-8")
        script_path.chmod(mode)

    report = build_doctor_report(_flags(), str(repo_subdirectory))

    assert _check(report, "Bench", "executable") == expected
