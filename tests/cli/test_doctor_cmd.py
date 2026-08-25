"""Tests for the ``gymrat doctor`` command wiring.

These drive the assembled app through :class:`typer.testing.CliRunner` with the
section builders, the bench smoke run, both renderers, ``inspect_config``, and
the repository lock replaced. They cover registration and help, the exit-code
contract, the JSON path, ``--no-color``, the ``--no-bench`` lock-free path, a
missing ``--config`` surfacing as a config failure rather than a crash, and the
abort event reaching the bench section.
"""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from gymrat_py.cli.app import app
from gymrat_py.config import BenchlessConfig
from gymrat_py.config_inspect import ConfigInspection
from gymrat_py.doctor.checks import Check, CheckSection

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
    """Replace every doctor seam and return the recorded bench inputs and lock calls."""
    inspection = ConfigInspection(
        config_path="/missing/gymrat.json" if config_failure else "/project/gymrat.json",
        config_exists=not config_failure,
        problems=["Config file not found at /missing/gymrat.json"] if config_failure else [],
        config=None if config_failure else _config(),
        bench="node bench.js",
    )

    def fake_inspect(*_a: object, **_k: object) -> ConfigInspection:
        return inspection

    monkeypatch.setattr("gymrat_py.cli.doctor_cmd.inspect_config", fake_inspect)

    def env_section(*_a: object, **_k: object) -> CheckSection:
        if env_error is not None:
            raise env_error
        return CheckSection(title="Environment", checks=[Check("git", "ok", "available")])

    monkeypatch.setattr("gymrat_py.cli.doctor_cmd.build_environment_section", env_section)

    config_checks = (
        [Check("config", "fail", "not found", hint="create gymrat.json")]
        if config_failure
        else [Check("config", "ok", "/project/gymrat.json")]
    )

    def config_section(*_a: object, **_k: object) -> CheckSection:
        return CheckSection(title="Configuration", checks=config_checks)

    monkeypatch.setattr("gymrat_py.cli.doctor_cmd.build_config_section", config_section)

    def workflow_section(*_a: object, **_k: object) -> CheckSection:
        return CheckSection(title="Workflow", checks=[Check("skill file", "ok", "found")])

    monkeypatch.setattr("gymrat_py.cli.doctor_cmd.build_workflow_section", workflow_section)

    bench_inputs: list[object] = []
    bench_check = (
        Check("smoke", "fail", "bench crashed")
        if bench_fail
        else Check("smoke", "ok", "1 metric found")
    )

    async def bench_section(bench_input: object) -> CheckSection:
        bench_inputs.append(bench_input)
        return CheckSection(title="Bench", checks=[bench_check])

    monkeypatch.setattr("gymrat_py.cli.doctor_cmd.build_bench_section", bench_section)

    def fake_text(_report: object) -> str:
        return "doctor text report"

    def fake_json(_report: object) -> str:
        return '{"doctor": true}'

    monkeypatch.setattr("gymrat_py.cli.doctor_cmd.render_doctor_report", fake_text)
    monkeypatch.setattr("gymrat_py.cli.doctor_cmd.render_doctor_json", fake_json)

    lock_calls: list[str] = []

    async def fake_lock(command: str, body: object) -> object:
        lock_calls.append(command)
        return await body()  # pyrefly: ignore

    monkeypatch.setattr("gymrat_py.cli.doctor_cmd.with_repo_lock", fake_lock)

    return SimpleNamespace(bench_inputs=bench_inputs, lock_calls=lock_calls)


@pytest.fixture
def _preserve_color_env():
    saved = {name: os.environ.get(name) for name in ("NO_COLOR", "FORCE_COLOR")}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


# ---------------------------------------------------------------------------
# registration and help
# ---------------------------------------------------------------------------


def test_doctor_when_root_help_does_list_doctor():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_doctor_when_help_does_document_no_bench_no_color_and_format():
    result = runner.invoke(app, ["doctor", "--help"])

    assert result.exit_code == 0
    out = result.stdout
    assert "--no-bench" in out
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


@pytest.mark.usefixtures("_preserve_color_env")
def test_doctor_when_no_color_does_set_no_color_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    _patch_doctor(monkeypatch)

    result = runner.invoke(app, ["doctor", "--no-color"])

    assert result.exit_code == 0
    assert os.environ.get("NO_COLOR") == "1"


# ---------------------------------------------------------------------------
# --no-bench and the repository lock
# ---------------------------------------------------------------------------


def test_doctor_when_no_bench_does_reach_bench_section_as_skip_without_the_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    handles = _patch_doctor(monkeypatch)

    result = runner.invoke(app, ["doctor", "--no-bench"])

    assert result.exit_code == 0
    assert len(handles.bench_inputs) == 1
    assert handles.bench_inputs[0].no_bench is True
    assert handles.lock_calls == []


def test_doctor_when_bench_runs_does_hold_the_repository_lock(monkeypatch: pytest.MonkeyPatch):
    handles = _patch_doctor(monkeypatch)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert len(handles.bench_inputs) == 1
    assert handles.bench_inputs[0].no_bench is False
    assert handles.lock_calls == ["doctor"]


def test_doctor_when_bench_runs_does_forward_an_abort_event_to_the_bench_section(
    monkeypatch: pytest.MonkeyPatch,
):
    handles = _patch_doctor(monkeypatch)

    runner.invoke(app, ["doctor"])

    assert isinstance(handles.bench_inputs[0].abort, asyncio.Event)


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
