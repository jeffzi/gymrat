"""Tests for the ``gymrat doctor`` command wiring.

These drive the assembled app through :class:`typer.testing.CliRunner` with the
section builders, both renderers, and ``inspect_config`` replaced; the one test
that reads the JSON document itself keeps the real JSON renderer. They cover
registration and help, the exit-code contract, the JSON path, ``--no-color``,
and a missing ``--config`` surfacing as a config failure rather than a crash.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from gymrat.cli.app import app
from gymrat.doctor import Check, CheckSection, GitEnvironment
from gymrat.scaffold import SKILL_RELATIVE_PATH
from tests.cli._help import help_output
from tests.cli._session import FailingStdoutRunner, closed_stdout_error
from tests.doctor._fixtures import fixed_section, patch_common_seams

runner = CliRunner()


def _patch_doctor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_failure: bool = False,
    bench_fail: bool = False,
    env_error: Exception | None = None,
    stub_json: bool = True,
) -> SimpleNamespace:
    """Replace every doctor seam and return the recorded bench calls.

    ``stub_json=False`` leaves the real JSON renderer in place.
    """
    handles = patch_common_seams(
        monkeypatch,
        config_failure=config_failure,
        bench_fail=bench_fail,
        problems=["Config file not found at /missing/gymrat.json"] if config_failure else [],
    )

    def env_section(*_a: object, **_k: object) -> CheckSection:
        if env_error is not None:
            raise env_error
        return CheckSection(title="Environment", checks=[Check("git", "ok", "available")])

    monkeypatch.setattr("gymrat.doctor.build_environment_section", env_section)

    def fake_text(_report: object, **_kwargs: object) -> str:
        return "doctor text report"

    def fake_json(_report: object) -> str:
        return '{"doctor": true}'

    monkeypatch.setattr("gymrat.cli.commands.doctor.render_doctor_report", fake_text)
    if stub_json:
        monkeypatch.setattr("gymrat.cli.commands.doctor.render_doctor_json", fake_json)

    return handles


# ---------------------------------------------------------------------------
# registration and help
# ---------------------------------------------------------------------------


def test_doctor_when_root_help_does_list_doctor():
    assert "doctor" in help_output()


def test_doctor_when_help_does_document_no_color_and_format():
    out = help_output("doctor")

    assert "--no-color" in out
    assert "--format" in out


# ---------------------------------------------------------------------------
# exit-code contract
# ---------------------------------------------------------------------------


def test_doctor_when_no_failures_does_exit_zero_and_write_text_report(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_doctor(monkeypatch)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "doctor text report" in result.stdout


def test_doctor_when_report_has_failures_does_exit_one_after_writing_report(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_doctor(monkeypatch, bench_fail=True)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "doctor text report" in result.stdout


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def test_doctor_when_format_json_does_write_only_the_json_line(monkeypatch: pytest.MonkeyPatch):
    _patch_doctor(monkeypatch)

    result = runner.invoke(app, ["doctor", "--format", "json"])

    assert result.exit_code == 0
    assert "doctor text report" not in result.stdout
    assert json.loads(result.stdout) == {"doctor": True}


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
def test_doctor_when_color_flag_given_does_hand_it_to_the_text_renderer(
    option: str, env: str, expected: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(env, "1")
    _patch_doctor(monkeypatch)
    rendered_with: list[bool] = []

    def render(_report: object, *, color: bool) -> str:
        rendered_with.append(color)
        return "doctor text report"

    monkeypatch.setattr("gymrat.cli.commands.doctor.render_doctor_report", render)

    runner.invoke(app, ["doctor", option])

    assert rendered_with == [expected]


def test_doctor_when_no_color_flag_does_not_mutate_color_env(monkeypatch: pytest.MonkeyPatch):
    _patch_doctor(monkeypatch)

    result = runner.invoke(app, ["doctor", "--no-color"])

    assert result.exit_code == 0
    assert os.environ.get("NO_COLOR") is None
    assert os.environ.get("FORCE_COLOR") is None


def test_doctor_when_no_color_flag_and_force_color_set_does_preserve_force_color_env(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    _patch_doctor(monkeypatch)

    result = runner.invoke(app, ["doctor", "--no-color"])

    assert result.exit_code == 0
    assert os.environ.get("FORCE_COLOR") == "1"


# ---------------------------------------------------------------------------
# missing --config is a config failure, not a crash
# ---------------------------------------------------------------------------


def test_doctor_when_config_path_missing_does_render_config_failure_not_crash(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_doctor(monkeypatch, config_failure=True)

    result = runner.invoke(app, ["doctor", "--config", "/missing/gymrat.json"])

    assert result.exit_code == 1
    assert "doctor text report" in result.stdout


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
    ("make_entry", "installed"),
    [
        pytest.param(_skill_directory, False, id="directory"),
        pytest.param(_skill_file, True, id="file"),
    ],
)
def test_doctor_when_skill_path_checked_does_report_installed_only_for_a_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_entry: Callable[[Path], None],
    installed: bool,
):
    make_entry(tmp_path / SKILL_RELATIVE_PATH)

    git_env = GitEnvironment(git_available=True, inside_git_repo=True, repo_root_dir=str(tmp_path))
    monkeypatch.setattr(
        "gymrat.doctor.detect_git_environment",
        lambda _cwd: git_env,  # pyrefly: ignore
    )

    _patch_doctor(monkeypatch)

    workflow_calls: list[dict[str, object]] = []

    def workflow_section(*_a: object, **kwargs: object) -> CheckSection:
        workflow_calls.append(dict(kwargs))
        return CheckSection(title="Workflow", checks=[Check("skill file", "ok", "found")])

    monkeypatch.setattr("gymrat.doctor.build_workflow_section", workflow_section)

    runner.invoke(app, ["doctor"])

    assert len(workflow_calls) >= 1
    assert workflow_calls[0].get("skill_file_exists") is installed
