"""Command-level tests for ``gymrat init``.

Each case drives the assembled app through :class:`typer.testing.CliRunner`. The
success paths run the real wizard and scaffold end to end from a throwaway cwd,
so the summary, the artifact statuses, and the base-directory resolution are
exercised the way a shell would invoke them. Usage errors and the
already-exists refusal assert the exit code and the routed message.
"""

import re
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gymrat_py.cli.app import app

runner = CliRunner()


def _read_config(base: Path) -> dict[str, object]:
    return tomllib.loads((base / "gymrat.toml").read_text(encoding="utf-8"))


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


def test_init_when_help_does_document_its_flags():
    result = runner.invoke(app, ["init", "--help"])

    assert result.exit_code == 0
    out = result.stdout
    assert "--bench" in out
    assert "--adapter" in out
    assert "--runbook" in out
    assert "--skill" in out
    assert "--yes" in out
    assert "-y" in out


def test_init_when_help_does_describe_scaffolding_a_toml_config():
    result = runner.invoke(app, ["init", "--help"])

    assert result.exit_code == 0
    assert "gymrat.toml" in result.stdout


# ---------------------------------------------------------------------------
# existing gymrat.toml at the resolved base
# ---------------------------------------------------------------------------


def test_init_when_config_already_exists_does_exit_two_pointing_at_doctor(non_repo_cwd: Path):
    (non_repo_cwd / "gymrat.toml").write_text("", encoding="utf-8")

    result = runner.invoke(app, ["init", "--bench", "npm run bench", "--yes"])

    assert result.exit_code == 2
    assert re.search(r"already exists", result.stderr, re.IGNORECASE)
    assert "gymrat doctor" in result.stderr


# ---------------------------------------------------------------------------
# wizard rejection
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_bench_missing_non_interactive_does_exit_two_naming_bench():
    result = runner.invoke(app, ["init", "--yes"])

    assert result.exit_code == 2
    assert "--bench" in result.stderr


# ---------------------------------------------------------------------------
# success summary
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_scaffolding_succeeds_does_write_summary_and_doctor_pointer():
    result = runner.invoke(app, ["init", "--bench", "npm run bench", "--yes"])

    assert result.exit_code == 0
    out = result.stdout
    assert "Config: created at gymrat.toml" in out
    assert "gymrat doctor" in out
    assert result.stderr == ""


def test_init_when_runbook_already_existed_does_report_it(non_repo_cwd: Path):
    (non_repo_cwd / "gymrat-runbook.md").write_text("# Existing\n", encoding="utf-8")

    result = runner.invoke(app, ["init", "--bench", "npm run bench", "--yes"])

    assert result.exit_code == 0
    assert re.search(r"runbook.*already exist", result.stdout, re.IGNORECASE)


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_runbook_declined_does_report_it():
    result = runner.invoke(app, ["init", "--bench", "npm run bench", "--yes", "--no-runbook"])

    assert result.exit_code == 0
    assert re.search(r"runbook.*(decline|skip)", result.stdout, re.IGNORECASE)


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_skill_declined_does_report_it():
    result = runner.invoke(app, ["init", "--bench", "npm run bench", "--yes", "--no-skill"])

    assert result.exit_code == 0
    assert re.search(r"skill.*(decline|skip)", result.stdout, re.IGNORECASE)


# ---------------------------------------------------------------------------
# flags forwarded to the wizard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "key", "expected"),
    [
        pytest.param(["--bench", "my-bench.sh"], "bench", "my-bench.sh", id="bench"),
        pytest.param(
            ["--bench", "bench.sh", "--adapter", "mitata"], "adapter", "mitata", id="adapter"
        ),
        pytest.param(
            ["--bench", "bench.sh", "--stop-target", "1.5", "--primary", "latency"],
            "stop",
            {"target_value": 1.5},
            id="stop-target",
        ),
        pytest.param(
            ["--bench", "bench.sh", "--runbook=custom-runbook.md"],
            "runbook",
            "custom-runbook.md",
            id="runbook-path",
        ),
    ],
)
def test_init_when_flags_given_does_forward_them_into_the_config(
    non_repo_cwd: Path, argv: list[str], key: str, expected: object
):
    result = runner.invoke(app, ["init", *argv, "--yes"])

    assert result.exit_code == 0
    assert _read_config(non_repo_cwd)[key] == expected


# ---------------------------------------------------------------------------
# usage errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("not-a-number", id="non-numeric"),
        pytest.param("1.5x", id="trailing-garbage"),
        pytest.param("Infinity", id="infinity"),
        pytest.param("-Infinity", id="negative-infinity"),
    ],
)
@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_stop_target_invalid_does_exit_two_naming_flag(value: str):
    result = runner.invoke(app, ["init", "--stop-target", value])

    assert result.exit_code == 2
    assert "stop-target" in (result.stdout + result.stderr)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("abc", id="non-numeric"),
        pytest.param("-1", id="negative"),
        pytest.param("1.5", id="non-integer"),
    ],
)
@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_stop_max_iterations_invalid_does_exit_two_naming_flag(value: str):
    result = runner.invoke(app, ["init", "--stop-max-iterations", value])

    assert result.exit_code == 2
    assert "stop-max-iterations" in (result.stdout + result.stderr)


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

    result = runner.invoke(app, ["init", "--bench", "npm run bench", "--yes"])

    assert result.exit_code == 0
    assert (Path(root) / "gymrat.toml").exists()
    assert not (nested / "gymrat.toml").exists()


def test_init_when_run_outside_a_git_repo_does_scaffold_in_cwd(non_repo_cwd: Path):
    result = runner.invoke(app, ["init", "--bench", "npm run bench", "--yes"])

    assert result.exit_code == 0
    assert (non_repo_cwd / "gymrat.toml").exists()
