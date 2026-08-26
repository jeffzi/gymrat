"""Command-level tests for ``gymrat init``.

Each case drives the assembled app through :class:`typer.testing.CliRunner`. The
non-interactive command takes ``--bench`` and optional ``--no-runbook`` /
``--no-skill`` flags, builds a :class:`ScaffoldRequest`, and delegates to
:func:`scaffold`. Usage errors, the already-exists refusal, and the
base-directory resolution are exercised the way a shell would invoke them.
"""

import re
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gymrat_py.cli.app import app

runner = CliRunner()

# These flags must never be advertised in --help or accepted by the command.
OLD_WIZARD_FLAGS = [
    pytest.param("--adapter", id="adapter"),
    pytest.param("--checks", id="checks"),
    pytest.param("--stop-target", id="stop-target"),
    pytest.param("--stop-max-iterations", id="stop-max-iterations"),
    pytest.param("--primary", id="primary"),
    pytest.param("--yes", id="yes"),
    pytest.param("-y", id="y"),
]


@pytest.fixture
def non_repo_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run the command from a fresh, non-git directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_init_when_root_help_does_list_init():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "init" in result.stdout


def test_init_when_help_does_describe_scaffolding_a_toml_config():
    result = runner.invoke(app, ["init", "--help"])

    assert result.exit_code == 0
    assert "gymrat.toml" in result.stdout


@pytest.mark.parametrize(
    "flag",
    [
        pytest.param("--bench", id="bench"),
        pytest.param("--no-runbook", id="no-runbook"),
        pytest.param("--no-skill", id="no-skill"),
        pytest.param("--debug", id="debug"),
    ],
)
def test_init_when_help_does_list_new_flags(flag: str):
    result = runner.invoke(app, ["init", "--help"])

    assert result.exit_code == 0
    assert flag in result.stdout


@pytest.mark.parametrize("flag", OLD_WIZARD_FLAGS)
def test_init_when_help_does_not_list_old_wizard_flags(flag: str):
    result = runner.invoke(app, ["init", "--help"])

    assert result.exit_code == 0
    assert flag not in result.stdout


@pytest.mark.parametrize(
    "flag", [pytest.param("runbook", id="runbook"), pytest.param("skill", id="skill")]
)
def test_init_when_help_does_not_list_standalone_flag(flag: str):
    """``--no-<flag>`` is legitimate; a standalone ``--<flag>`` option is not."""
    result = runner.invoke(app, ["init", "--help"])

    assert result.exit_code == 0
    assert not re.search(rf"(?<!no-)--{flag}\b", result.stdout)


@pytest.mark.parametrize(
    "flag",
    [
        *OLD_WIZARD_FLAGS,
        pytest.param("--runbook", id="runbook"),
        pytest.param("--skill", id="skill"),
    ],
)
@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_old_flag_given_does_reject_it(flag: str):
    result = runner.invoke(app, ["init", flag, "dummy"])

    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# missing --bench
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_bench_missing_does_exit_two_naming_bench():
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 2
    assert "--bench" in result.stderr


# ---------------------------------------------------------------------------
# existing gymrat.toml at the resolved base
# ---------------------------------------------------------------------------


def test_init_when_config_already_exists_does_exit_two_pointing_at_doctor(non_repo_cwd: Path):
    (non_repo_cwd / "gymrat.toml").write_text("", encoding="utf-8")

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 2
    assert re.search(r"already exists", result.stderr, re.IGNORECASE)
    assert "gymrat doctor" in result.stderr


# ---------------------------------------------------------------------------
# success summary
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_scaffolding_succeeds_does_write_summary_and_doctor_pointer():
    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    out = result.stdout
    assert "Config: created at gymrat.toml" in out
    assert "gymrat doctor" in out
    assert result.stderr == ""


def test_init_when_runbook_already_existed_does_report_it(non_repo_cwd: Path):
    (non_repo_cwd / "gymrat-runbook.md").write_text("# Existing\n", encoding="utf-8")

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert re.search(r"runbook.*already exist", result.stdout, re.IGNORECASE)


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_runbook_declined_does_report_it():
    result = runner.invoke(app, ["init", "--bench", "npm run bench", "--no-runbook"])

    assert result.exit_code == 0
    assert re.search(r"runbook.*(decline|skip)", result.stdout, re.IGNORECASE)


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_skill_declined_does_report_it():
    result = runner.invoke(app, ["init", "--bench", "npm run bench", "--no-skill"])

    assert result.exit_code == 0
    assert re.search(r"skill.*(decline|skip)", result.stdout, re.IGNORECASE)


# ---------------------------------------------------------------------------
# usage errors — unknown flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        pytest.param("--config", "gymrat.toml", id="config"),
        pytest.param("--samples", "5", id="samples"),
        pytest.param("--timeout", "300", id="timeout"),
    ],
)
@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_unknown_config_command_flag_given_does_reject_it(flag: str, value: str):
    result = runner.invoke(app, ["init", flag, value])

    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# base-directory resolution
# ---------------------------------------------------------------------------


def test_init_when_run_in_a_git_repo_subdirectory_does_scaffold_at_the_repo_root(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    root = create_scratch_repo()
    nested = Path(root) / "packages" / "core"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert (Path(root) / "gymrat.toml").exists()
    assert not (nested / "gymrat.toml").exists()


def test_init_when_run_outside_a_git_repo_does_scaffold_in_cwd(non_repo_cwd: Path):
    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert (non_repo_cwd / "gymrat.toml").exists()


# ---------------------------------------------------------------------------
# shared.py — stop-target helpers are not exported
# ---------------------------------------------------------------------------


def test_shared_when_stop_target_removed_does_not_export_helpers():
    from gymrat_py.cli import shared

    assert not hasattr(shared, "parse_stop_target_value")
    assert not hasattr(shared, "_STOP_TARGET_RE")
