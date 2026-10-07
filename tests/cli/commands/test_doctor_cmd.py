"""Tests for the ``gymrat doctor`` command wiring.

These drive the assembled app through :class:`typer.testing.CliRunner` with the
section builders, both renderers, and ``inspect_config`` replaced; the tests
that read the rendered report or the JSON document keep the real renderer. They cover
the exit-code contract (a missing ``--config`` surfacing as a config failure
rather than a crash), the JSON path, and ``--color``/``--no-color``.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.config import inspect_config
from gymrat.doctor import (
    Check,
    CheckSection,
    GitEnvironment,
    build_config_section,
    build_workflow_section,
)
from gymrat.scaffold import SKILL_RELATIVE_PATH
from tests._doctor_fixtures import fixed_section, patch_common_seams
from tests.cli._session import (
    FailingStdoutRunner,
    closed_stdout_error,
    runner,
)


def _patch_doctor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bench_fail: bool = False,
    env_error: Exception | None = None,
    stub_text: bool = True,
    stub_json: bool = True,
) -> None:
    """Replace every doctor seam.

    Args:
        monkeypatch: The fixture that installs the stand-ins.
        bench_fail: Whether the bench section reports a failed check.
        env_error: An error the environment section raises instead of returning.
        stub_text: ``False`` leaves the real text renderer in place.
        stub_json: ``False`` leaves the real JSON renderer in place.
    """
    patch_common_seams(monkeypatch, config_failure=False, bench_fail=bench_fail, problems=[])

    def env_section(*_a: object, **_k: object) -> CheckSection:
        if env_error is not None:
            raise env_error
        return CheckSection(title="Environment", checks=[Check("git", "ok", "available")])

    monkeypatch.setattr("gymrat.doctor.build_environment_section", env_section)

    def fake_text(_report: object, **_kwargs: object) -> str:
        return "doctor text report"

    def fake_json(_report: object) -> str:
        return '{"doctor": true}'

    if stub_text:
        monkeypatch.setattr("gymrat.cli.commands.doctor.render_doctor_report", fake_text)
    if stub_json:
        monkeypatch.setattr("gymrat.cli.commands.doctor.render_doctor_json", fake_json)


# ---------------------------------------------------------------------------
# exit-code contract
# ---------------------------------------------------------------------------


def _keep_the_stubbed_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the stubbed config inspection and config section in place."""


def _inspect_the_real_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the real config inspection and config section over the stubs."""
    monkeypatch.setattr("gymrat.doctor.inspect_config", inspect_config)
    monkeypatch.setattr("gymrat.doctor.build_config_section", build_config_section)


@pytest.mark.parametrize(
    ("bench_fail", "config_seams", "argv", "exit_code"),
    [
        pytest.param(False, _keep_the_stubbed_config, ["doctor"], 0, id="no-failures-exit-zero"),
        pytest.param(True, _keep_the_stubbed_config, ["doctor"], 1, id="failures-exit-one"),
        pytest.param(
            False,
            _inspect_the_real_config,
            ["doctor", "--config", "missing/gymrat.toml"],
            1,
            id="missing-config-is-a-config-failure",
        ),
    ],
)
def test_doctor_when_report_written_does_exit_on_its_failures_after_writing_it(
    *,
    bench_fail: bool,
    config_seams: Callable[[pytest.MonkeyPatch], None],
    argv: list[str],
    exit_code: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    _patch_doctor(monkeypatch, bench_fail=bench_fail)
    config_seams(monkeypatch)

    result = runner.invoke(app, argv)

    assert result.exit_code == exit_code
    assert "doctor text report" in result.stdout


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def test_doctor_when_format_json_does_write_indented_document_with_every_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_doctor(monkeypatch, stub_json=False)
    monkeypatch.setattr(
        "gymrat.doctor.build_workflow_section",
        fixed_section(
            "Workflow", [Check("skill file", "warn", "not installed", hint="run gymrat init")]
        ),
    )

    result = runner.invoke(app, ["doctor", "--format", "json"])

    document = json.loads(result.stdout)
    hints = {
        check["name"]: check["hint"]
        for section in document["sections"]
        for check in section["checks"]
    }
    assert result.exit_code == 0
    assert result.stdout == json.dumps(document, indent=2) + "\n"
    assert (hints["git"], hints["skill file"]) == (None, "run gymrat init")


# ---------------------------------------------------------------------------
# color control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("option", "env", "expected"),
    [
        pytest.param("--color", "NO_COLOR", True, id="color-flag-outranks-no-color-env"),
        pytest.param("--no-color", "FORCE_COLOR", False, id="no-color-flag-outranks-force-color"),
    ],
)
def test_doctor_when_color_flag_given_does_style_the_text_report_to_match(
    option: str, env: str, expected: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(env, "1")
    _patch_doctor(monkeypatch, stub_text=False)

    result = runner.invoke(app, ["doctor", option])

    assert result.exit_code == 0
    assert ("\x1b[" in result.stdout) is expected


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
    _patch_doctor(monkeypatch)

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
    _patch_doctor(monkeypatch, env_error=RuntimeError("unexpected doctor crash"))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 2
    assert "unexpected doctor crash" in result.stderr


# ---------------------------------------------------------------------------
# closed stdout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["text", "json"])
def test_doctor_when_stdout_reader_closed_does_exit_zero_without_stderr(
    monkeypatch: pytest.MonkeyPatch, fmt: str
):
    _patch_doctor(monkeypatch)

    result = FailingStdoutRunner(closed_stdout_error()).invoke(app, ["doctor", "--format", fmt])

    assert (result.exit_code, result.stderr) == (0, "")


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
    make_entry(tmp_path / SKILL_RELATIVE_PATH)

    git_env = GitEnvironment(git_available=True, inside_git_repo=True, repo_root_dir=str(tmp_path))
    monkeypatch.setattr(
        "gymrat.doctor.detect_git_environment",
        lambda _cwd: git_env,  # pyrefly: ignore
    )

    _patch_doctor(monkeypatch, stub_json=False)
    monkeypatch.setattr("gymrat.doctor.build_workflow_section", build_workflow_section)

    result = runner.invoke(app, ["doctor", "--format", "json"])

    statuses = {
        check["name"]: check["status"]
        for section in json.loads(result.stdout)["sections"]
        for check in section["checks"]
    }
    assert result.exit_code == 0
    assert statuses["skill file"] == status
