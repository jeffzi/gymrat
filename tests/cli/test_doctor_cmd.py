"""Tests for the ``gymrat doctor`` command wiring.

These drive the assembled app through :class:`typer.testing.CliRunner` with the
section builders, both renderers, and ``inspect_config`` replaced. They cover
registration and help, the exit-code contract, the JSON path, ``--no-color``,
a missing ``--config`` surfacing as a config failure rather than a crash, and
the adapter flag reaching the bench section.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from gymrat.cli.app import app
from gymrat.config import BenchlessConfig
from gymrat.config_inspect import ConfigInspection
from gymrat.doctor.checks import Check, CheckSection

runner = CliRunner()


def _config() -> BenchlessConfig:
    return BenchlessConfig(
        adapter="metric-lines",
        samples=10,
        timeout_seconds=1800,
        unstable_noise_pct=200,
        primary="geomean",
    )


def _patch_doctor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_failure: bool = False,
    bench_fail: bool = False,
    env_error: Exception | None = None,
) -> SimpleNamespace:
    """Replace every doctor seam and return the recorded bench calls."""
    inspection = ConfigInspection(
        config_path="/missing/gymrat.json" if config_failure else "/project/gymrat.json",
        problems=["Config file not found at /missing/gymrat.json"] if config_failure else [],
        config=None if config_failure else _config(),
        bench="node bench.js",
    )

    def fake_inspect(*_a: object, **_k: object) -> ConfigInspection:
        return inspection

    monkeypatch.setattr("gymrat.cli.doctor_cmd.inspect_config", fake_inspect)

    def env_section(*_a: object, **_k: object) -> CheckSection:
        if env_error is not None:
            raise env_error
        return CheckSection(title="Environment", checks=[Check("git", "ok", "available")])

    monkeypatch.setattr("gymrat.cli.doctor_cmd.build_environment_section", env_section)

    config_checks = (
        [Check("config", "fail", "not found", hint="create gymrat.json")]
        if config_failure
        else [Check("config", "ok", "/project/gymrat.json")]
    )

    def config_section(*_a: object, **_k: object) -> CheckSection:
        return CheckSection(title="Configuration", checks=config_checks)

    monkeypatch.setattr("gymrat.cli.doctor_cmd.build_config_section", config_section)

    def workflow_section(*_a: object, **_k: object) -> CheckSection:
        return CheckSection(title="Workflow", checks=[Check("skill file", "ok", "found")])

    monkeypatch.setattr("gymrat.cli.doctor_cmd.build_workflow_section", workflow_section)

    bench_calls: list[dict[str, object]] = []
    bench_check = (
        Check("bench", "fail", "bench crashed")
        if bench_fail
        else Check("bench", "ok", "1 metric found")
    )

    def bench_section(*, bench: object, adapter: object, **kwargs: object) -> CheckSection:
        bench_calls.append({"bench": bench, "adapter": adapter, **kwargs})
        return CheckSection(title="Bench", checks=[bench_check])

    monkeypatch.setattr("gymrat.cli.doctor_cmd.build_bench_section", bench_section)

    def fake_text(_report: object, **_kwargs: object) -> str:
        return "doctor text report"

    def fake_json(_report: object) -> str:
        return '{"doctor": true}'

    monkeypatch.setattr("gymrat.cli.doctor_cmd.render_doctor_report", fake_text)
    monkeypatch.setattr("gymrat.cli.doctor_cmd.render_doctor_json", fake_json)

    return SimpleNamespace(bench_calls=bench_calls)


@pytest.fixture(autouse=True)
def _preserve_color_env(monkeypatch: pytest.MonkeyPatch):
    for name in ("NO_COLOR", "FORCE_COLOR"):
        value = os.environ.get(name)
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


# ---------------------------------------------------------------------------
# registration and help
# ---------------------------------------------------------------------------


def test_doctor_when_root_help_does_list_doctor():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_doctor_when_help_does_document_no_color_and_format():
    result = runner.invoke(app, ["doctor", "--help"])

    assert result.exit_code == 0
    out = result.stdout
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


# ---------------------------------------------------------------------------
# color control
# ---------------------------------------------------------------------------


def test_doctor_when_no_color_flag_does_not_mutate_color_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
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
# config failure + adapter flag
# ---------------------------------------------------------------------------


def test_doctor_when_config_failed_and_adapter_flag_given_does_forward_flag_adapter_to_bench_section(
    monkeypatch: pytest.MonkeyPatch,
):
    handles = _patch_doctor(monkeypatch, config_failure=True)

    result = runner.invoke(app, ["doctor", "--adapter", "custom-adapter"])

    assert result.exit_code in (0, 1)
    assert len(handles.bench_calls) == 1
    assert handles.bench_calls[0]["adapter"] == "custom-adapter"


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
# config_problems forwarded to bench section
# ---------------------------------------------------------------------------


def test_doctor_when_config_problems_does_forward_config_problems_true_to_bench_section(
    monkeypatch: pytest.MonkeyPatch,
):
    handles = _patch_doctor(monkeypatch, config_failure=True)

    runner.invoke(app, ["doctor"])

    assert len(handles.bench_calls) == 1
    assert handles.bench_calls[0]["config_problems"] is True


def test_doctor_when_config_ok_does_forward_config_problems_false_to_bench_section(
    monkeypatch: pytest.MonkeyPatch,
):
    handles = _patch_doctor(monkeypatch, config_failure=False)

    runner.invoke(app, ["doctor"])

    assert len(handles.bench_calls) == 1
    assert handles.bench_calls[0]["config_problems"] is False


# ---------------------------------------------------------------------------
# skill file: directory at path
# ---------------------------------------------------------------------------


def test_doctor_when_skill_path_is_directory_does_not_report_installed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """A directory at the skill path must not count as 'installed'."""
    from gymrat.cli.doctor_cmd import GitEnvironment
    from gymrat.init.scaffold import SKILL_RELATIVE_PATH

    skill_dir = tmp_path / SKILL_RELATIVE_PATH
    skill_dir.mkdir(parents=True, exist_ok=True)

    git_env = GitEnvironment(git_available=True, inside_git_repo=True, repo_root_dir=str(tmp_path))
    monkeypatch.setattr(
        "gymrat.cli.doctor_cmd.detect_git_environment",
        lambda _cwd: git_env,  # pyrefly: ignore
    )

    _patch_doctor(monkeypatch)

    workflow_calls: list[dict[str, object]] = []

    def workflow_section(*_a: object, **kwargs: object) -> CheckSection:
        workflow_calls.append(dict(kwargs))
        return CheckSection(title="Workflow", checks=[Check("skill file", "ok", "found")])

    monkeypatch.setattr("gymrat.cli.doctor_cmd.build_workflow_section", workflow_section)

    runner.invoke(app, ["doctor"])

    assert len(workflow_calls) >= 1
    assert workflow_calls[0].get("skill_file_exists") is False
