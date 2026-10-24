"""Tests for the ``gymrat doctor`` command wiring.

These drive the assembled app through :class:`typer.testing.CliRunner` with the
report builder and both renderers replaced; the tests that depend on what the
real report finds keep the real builder, and those that read the JSON document
keep the real renderer. They cover
the exit-code contract (a missing ``--config`` surfacing as a config failure
rather than a crash), the JSON path, and ``--no-color`` leaving the color env
untouched; ``--color`` is pinned with every command's in ``test_app``.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from gymrat.cli.app import app
from gymrat.doctor import GitEnvironment, detect_git_environment
from gymrat.scaffold import SKILL_RELATIVE_PATH
from tests.cli._doctor_seams import patch_doctor
from tests.cli._runner import (
    runner,
)
from tests.config._toml import write_config

# ---------------------------------------------------------------------------
# exit-code contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bench_fail", "real_report", "argv", "exit_code"),
    [
        pytest.param(False, False, ["doctor"], 0, id="no-failures-exit-zero"),
        pytest.param(True, False, ["doctor"], 1, id="failures-exit-one"),
        pytest.param(
            False,
            True,
            ["doctor", "--config", "missing/gymrat.toml"],
            1,
            id="missing-config-is-a-config-failure",
        ),
    ],
)
# Outside a repository the real report can fail on nothing but the config.
@pytest.mark.usefixtures("_in_non_repo")
def test_doctor_when_report_written_does_exit_on_its_failures_after_writing_it(
    *,
    bench_fail: bool,
    real_report: bool,
    argv: list[str],
    exit_code: int,
    monkeypatch: pytest.MonkeyPatch,
):
    patch_doctor(monkeypatch, bench_fail=bench_fail, real_report=real_report)

    result = runner.invoke(app, argv)

    assert result.exit_code == exit_code
    assert "doctor text report" in result.stdout


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def test_doctor_when_format_json_does_write_the_json_document_once(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_doctor(monkeypatch, stub_json=False)

    result = runner.invoke(app, ["doctor", "--format", "json"])

    assert result.exit_code == 0
    assert "sections" in json.loads(result.stdout)
    assert result.stdout.endswith("}\n")


# ---------------------------------------------------------------------------
# color control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        pytest.param({}, id="color-env-unset"),
        pytest.param({"FORCE_COLOR": "1"}, id="force-color-set"),
    ],
)
def test_doctor_when_no_color_flag_does_leave_the_color_env_as_it_was(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    patch_doctor(monkeypatch)

    result = runner.invoke(app, ["doctor", "--no-color"])

    assert result.exit_code == 0
    assert (os.environ.get("NO_COLOR"), os.environ.get("FORCE_COLOR")) == (
        None,
        env.get("FORCE_COLOR"),
    )


# ---------------------------------------------------------------------------
# unexpected crash
# ---------------------------------------------------------------------------


def test_doctor_when_command_crashes_does_exit_two_with_message_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_doctor(monkeypatch, report_error=RuntimeError("unexpected doctor crash"))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 2
    assert "unexpected doctor crash" in result.stderr


# ---------------------------------------------------------------------------
# skill file: only a regular file at the path counts as installed
# ---------------------------------------------------------------------------


def _skill_directory(path: Path) -> None:
    """Put a directory where the skill file belongs."""
    path.mkdir(parents=True, exist_ok=True)


def _skill_file(path: Path) -> None:
    """Put a regular skill file at its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("skill", encoding="utf-8")


@pytest.mark.parametrize(
    ("make_entry", "status"),
    [
        pytest.param(_skill_directory, "warn", id="directory"),
        pytest.param(_skill_file, "ok", id="file"),
    ],
)
def test_doctor_when_skill_path_checked_does_report_installed_only_for_a_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_entry: Callable[[Path], None],
    status: str,
):
    git_env = GitEnvironment(git_available=True, inside_git_repo=True, repo_root_dir=str(tmp_path))
    make_entry(tmp_path / SKILL_RELATIVE_PATH)
    write_config(tmp_path, {"bench": "npm run bench"})
    monkeypatch.setattr(
        "gymrat.doctor.detect_git_environment",
        create_autospec(detect_git_environment, return_value=git_env),
    )
    patch_doctor(monkeypatch, real_report=True, stub_json=False)

    result = runner.invoke(app, ["doctor", "--format", "json"])

    statuses = {
        check["name"]: check["status"]
        for section in json.loads(result.stdout)["sections"]
        for check in section["checks"]
    }
    assert result.exit_code == 0
    assert statuses["skill file"] == status


# ---------------------------------------------------------------------------
# samples: an odd count fails the report in both formats
# ---------------------------------------------------------------------------

_ODD_SAMPLES_HINT = (
    "`iterate` and `compare` refuse an odd count; `measure` and `probe` still run — use 10 or 12"
)


@pytest.fixture
def _ready_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A repository root with a bench configured, run through the real report and renderers."""
    git_env = GitEnvironment(git_available=True, inside_git_repo=True, repo_root_dir=str(tmp_path))
    write_config(tmp_path, {"bench": "npm run bench"})
    monkeypatch.setattr(
        "gymrat.doctor.detect_git_environment",
        create_autospec(detect_git_environment, return_value=git_env),
    )
    patch_doctor(monkeypatch, real_report=True, stub_text=False, stub_json=False)


@pytest.mark.parametrize(
    ("samples", "expected", "exit_code"),
    [
        pytest.param(
            "11",
            {
                "name": "samples",
                "status": "fail",
                "detail": "samples: 11 — balances one target only",
                "hint": _ODD_SAMPLES_HINT,
            },
            1,
            id="odd-fails",
        ),
        pytest.param(
            "12",
            {
                "name": "samples",
                "status": "ok",
                "detail": "samples: 12 — balances 2, 3, 4 or 6 targets",
                "hint": None,
            },
            0,
            id="even-passes",
        ),
    ],
)
@pytest.mark.usefixtures("_ready_project")
def test_doctor_when_samples_flag_given_does_report_its_check_in_json_and_exit_on_it(
    samples: str, expected: dict[str, object], exit_code: int
):
    result = runner.invoke(app, ["doctor", "--samples", samples, "--format", "json"])

    checks = [
        check
        for section in json.loads(result.stdout)["sections"]
        if section["title"] == "Workflow"
        for check in section["checks"]
        if check["name"] == "samples"
    ]
    assert checks == [expected]
    assert result.exit_code == exit_code


@pytest.mark.usefixtures("_ready_project")
def test_doctor_when_samples_odd_does_render_the_failed_check_and_hint_as_text():
    result = runner.invoke(app, ["doctor", "--samples", "11", "--no-color"])

    lines = result.stdout.split("\n")
    failed = lines.index("  ✗ samples: 11 — balances one target only")
    assert lines[failed + 1] == f"    {_ODD_SAMPLES_HINT.replace('`', '')}"
    assert result.exit_code == 1


@pytest.mark.usefixtures("_ready_project")
def test_doctor_when_samples_even_does_render_the_passing_check_as_text():
    result = runner.invoke(app, ["doctor", "--samples", "12", "--no-color"])

    assert "  ✓ samples: 12 — balances 2, 3, 4 or 6 targets" in result.stdout.split("\n")
    assert result.exit_code == 0
