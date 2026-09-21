"""Command-level tests for ``gymrat init``.

Each case drives the assembled app through :class:`typer.testing.CliRunner`. The
non-interactive command takes ``--bench`` and optional ``--no-runbook`` /
``--no-skill`` flags, builds a :class:`ScaffoldRequest`, and delegates to
:func:`scaffold`. Usage errors, the re-run over an existing ``gymrat.toml``,
and the base-directory resolution are exercised the way a shell would invoke
them.
"""

import re
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gymrat.cli.app import app
from gymrat.loop.start import start_session
from gymrat.session import experiment_worktree_dir, read_records, session_jsonl_path
from gymrat.session.budget import write_budget
from tests._ansi import strip_ansi
from tests.cli._budget import (
    LIVE_BUDGET,
    set_origin,
)
from tests.cli._help import help_output
from tests.loop.iterate._fixtures import resolved_config

runner = CliRunner()

EXISTING_CONFIG = 'bench = "old"\n'

LIVE_REFUSAL = "a supervised run is live; init is not part of the loop"

#: Every file init can write at a base directory.
INIT_ARTIFACTS = ("gymrat.toml", "gymrat-runbook.md", ".claude/skills/gymrat/SKILL.md")

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


@pytest.fixture
def existing_config_cwd(non_repo_cwd: Path) -> Path:
    """A non-repo cwd with a pre-existing ``gymrat.toml`` already written."""
    (non_repo_cwd / "gymrat.toml").write_text(EXISTING_CONFIG, encoding="utf-8")
    return non_repo_cwd


def _write_live_budget(repo: str) -> None:
    """Write a live supervised-run budget file at ``repo``."""
    Path(repo, ".gymrat").mkdir(exist_ok=True)
    write_budget(repo, LIVE_BUDGET)


@pytest.fixture
def live_repo(repo: str, supervise_lock: None) -> str:
    """A chdir'd scratch repository with a live supervised-run budget."""
    _write_live_budget(repo)
    return repo


def _written_artifacts(base: str) -> list[str]:
    """The init artifacts present under ``base``."""
    return [name for name in INIT_ARTIFACTS if (Path(base) / name).exists()]


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_init_when_root_help_does_list_init():
    assert "init" in help_output()


def test_init_when_help_does_describe_scaffolding_a_toml_config():
    assert "gymrat.toml" in help_output("init")


@pytest.mark.parametrize(
    "flag",
    [
        pytest.param("--bench", id="bench"),
        pytest.param("--no-runbook", id="no-runbook"),
        pytest.param("--no-skill", id="no-skill"),
        pytest.param("--no-color", id="no-color"),
        pytest.param("--debug", id="debug"),
    ],
)
def test_init_when_help_does_list_new_flags(flag: str):
    assert flag in help_output("init")


def test_init_when_help_does_show_bench_short_form():
    out = help_output("init")

    assert "--bench" in out
    assert re.search(r"(?<!\w)-b(?!\w)", out)


@pytest.mark.parametrize("flag", OLD_WIZARD_FLAGS)
def test_init_when_help_does_not_list_old_wizard_flags(flag: str):
    assert flag not in help_output("init")


@pytest.mark.parametrize(
    "flag", [pytest.param("runbook", id="runbook"), pytest.param("skill", id="skill")]
)
def test_init_when_help_does_not_list_standalone_flag(flag: str):
    # --no-<flag> is legitimate; a standalone --<flag> option is not.
    assert not re.search(rf"(?<!no-)--{flag}\b", help_output("init"))


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
def test_init_when_short_bench_flag_does_scaffold_same_as_long_form(non_repo_cwd: Path):
    result = runner.invoke(app, ["init", "-b", "npm run bench"])

    assert result.exit_code == 0
    assert (non_repo_cwd / "gymrat.toml").exists()


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_bench_missing_does_exit_two_naming_bench():
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 2
    assert "--bench" in result.stderr


# ---------------------------------------------------------------------------
# existing gymrat.toml at the resolved base
# ---------------------------------------------------------------------------


def test_init_when_config_already_exists_does_report_existing_config_and_new_artifacts(
    existing_config_cwd: Path,
):
    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert re.search(r"config.*already exists at gymrat\.toml", result.stdout, re.IGNORECASE)
    assert re.search(r"runbook.*created at", result.stdout, re.IGNORECASE)


def test_init_when_config_already_exists_does_not_require_bench(existing_config_cwd: Path):
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert (existing_config_cwd / "gymrat-runbook.md").exists()


def test_init_when_config_already_exists_and_bench_given_does_not_rewrite_it(
    existing_config_cwd: Path,
):
    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert (existing_config_cwd / "gymrat.toml").read_text(encoding="utf-8") == EXISTING_CONFIG


# ---------------------------------------------------------------------------
# blocked artifact path exits with code 2
# ---------------------------------------------------------------------------


def test_init_when_skill_path_is_a_directory_does_exit_two(existing_config_cwd: Path):
    (existing_config_cwd / ".claude" / "skills" / "gymrat" / "SKILL.md").mkdir(parents=True)

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 2


def test_init_when_runbook_path_is_a_directory_does_exit_two_naming_path(non_repo_cwd: Path):
    (non_repo_cwd / "gymrat-runbook.md").mkdir()

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 2
    assert "gymrat-runbook.md" in result.stderr


def test_init_when_runbook_is_a_symlink_does_exit_two(non_repo_cwd: Path):
    target = non_repo_cwd / "real.md"
    target.write_text("# target\n", encoding="utf-8")
    (non_repo_cwd / "gymrat-runbook.md").symlink_to(target)

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 2
    assert "gymrat-runbook.md" in result.stderr


# ---------------------------------------------------------------------------
# success summary
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_scaffolding_succeeds_does_write_summary_and_doctor_pointer():
    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    out = result.stdout
    assert "Config: created at gymrat.toml" in out
    assert result.stderr == ""
    lines = strip_ansi(out).rstrip("\n").split("\n")
    pointer_index = next(i for i, line in enumerate(lines) if "gymrat doctor" in line)
    # The hint closes the artifact block directly — no blank line before it.
    assert lines[pointer_index - 1].strip().startswith("Skill:")


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


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_colored_does_dim_the_doctor_pointer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert "`" not in result.stdout
    pointer = next(
        line for line in result.stdout.split("\n") if "gymrat doctor" in strip_ansi(line)
    )
    assert pointer.startswith("\x1b[2m")


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_no_color_flag_does_suppress_ansi_in_summary():
    result = runner.invoke(app, ["init", "--bench", "npm run bench", "--no-color"])

    assert result.exit_code == 0
    assert "\x1b[" not in result.stdout


# ---------------------------------------------------------------------------
# broken pipe on stdout
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("non_repo_cwd")
def test_init_when_stdout_broken_pipe_does_exit_zero(monkeypatch: pytest.MonkeyPatch):
    def broken_write(_stream: object, _data: str) -> None:
        raise BrokenPipeError

    monkeypatch.setattr("gymrat.cli.init_cmd.write_and_flush", broken_write)

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# init summary prints cwd-relative paths
# ---------------------------------------------------------------------------


def test_init_when_run_from_subdirectory_does_print_cwd_relative_paths_in_summary(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    root = create_scratch_repo()
    nested = Path(root) / "packages" / "core"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    out = result.stdout
    # Paths must be navigable from cwd, not bare filenames relative to repo root.
    # From packages/core/, gymrat.toml is at ../../gymrat.toml.
    assert "gymrat.toml" in out
    relative = str(Path("../..") / "gymrat.toml")
    assert relative in out or str(Path(root) / "gymrat.toml") in out


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
# refusal while a supervised run is live
# ---------------------------------------------------------------------------


ORIGINS = [
    pytest.param(None, id="origin-unset"),
    pytest.param("cli", id="origin-cli"),
    pytest.param("tool", id="origin-tool"),
]


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--bench", "npm run bench"], id="with-bench"),
        pytest.param([], id="without-bench"),
    ],
)
def test_init_when_supervised_run_live_does_refuse_without_writing(
    live_repo: str, monkeypatch: pytest.MonkeyPatch, origin: str | None, args: list[str]
):
    set_origin(monkeypatch, origin)

    result = runner.invoke(app, ["init", *args])

    assert result.exit_code == 2
    assert LIVE_REFUSAL in result.stderr
    assert _written_artifacts(live_repo) == []
    assert not Path(session_jsonl_path(live_repo)).exists()


def test_init_when_run_from_experiment_worktree_during_live_run_does_refuse(
    live_repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_session(live_repo, "main", resolved_config())
    records_before = read_records(session_jsonl_path(live_repo))
    worktree = experiment_worktree_dir(live_repo)
    monkeypatch.chdir(worktree)

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 2
    assert LIVE_REFUSAL in result.stderr
    assert _written_artifacts(live_repo) == []
    assert _written_artifacts(worktree) == []
    assert read_records(session_jsonl_path(live_repo)) == records_before


def test_init_when_budget_left_behind_without_supervise_lock_does_scaffold(repo: str):
    _write_live_budget(repo)

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert _written_artifacts(repo) == list(INIT_ARTIFACTS)
