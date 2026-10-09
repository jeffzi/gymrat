"""Tests for the ``gymrat doctor`` command wiring.

These drive the assembled app through :class:`typer.testing.CliRunner` with the
section builders, both renderers, and ``inspect_config`` replaced; the tests
that read the rendered report or the JSON document keep the real renderer. They cover
the exit-code contract (a missing ``--config`` surfacing as a config failure
rather than a crash), the JSON path, and ``--no-color`` leaving the color env
untouched; ``--color`` is pinned with every command's in ``test_app``.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.config import inspect_config
from gymrat.doctor import (
    GitEnvironment,
    build_config_section,
    build_workflow_section,
)
from gymrat.scaffold import SKILL_RELATIVE_PATH
from tests.cli._doctor_seams import patch_doctor
from tests.cli._session import runner

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
    patch_doctor(monkeypatch, bench_fail=bench_fail)
    config_seams(monkeypatch)

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
    patch_doctor(monkeypatch, env_error=RuntimeError("unexpected doctor crash"))

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
    def fake_detect(_cwd: object) -> GitEnvironment:
        return GitEnvironment(git_available=True, inside_git_repo=True, repo_root_dir=str(tmp_path))

    make_entry(tmp_path / SKILL_RELATIVE_PATH)
    monkeypatch.setattr("gymrat.doctor.detect_git_environment", fake_detect)
    patch_doctor(monkeypatch, stub_json=False)
    monkeypatch.setattr("gymrat.doctor.build_workflow_section", build_workflow_section)

    result = runner.invoke(app, ["doctor", "--format", "json"])

    statuses = {
        check["name"]: check["status"]
        for section in json.loads(result.stdout)["sections"]
        for check in section["checks"]
    }
    assert result.exit_code == 0
    assert statuses["skill file"] == status
