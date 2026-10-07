"""Command-level tests for ``gymrat init``.

Each case drives the assembled app through :class:`typer.testing.CliRunner`. The
non-interactive command takes ``--bench`` and optional ``--no-runbook`` /
``--no-skill`` flags, builds a :class:`ScaffoldRequest`, and delegates to
:func:`scaffold`. Usage errors, the re-run over an existing ``gymrat.toml``,
and the base-directory resolution are exercised the way a shell would invoke
them.
"""

import re
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.loop.start import start_session
from gymrat.session.paths import (
    experiment_worktree_dir,
    session_jsonl_path,
)
from tests._ansi import strip_ansi
from tests._config import resolved_config
from tests._lock import hold_supervise_lock
from tests.cli._budget import set_origin, write_budget_file
from tests.cli._session import (
    FailingStdoutRunner,
    closed_stdout_error,
    runner,
)
from tests.session.records._fixtures import log_records

EXISTING_CONFIG = 'bench = "old"\n'

LIVE_REFUSAL = "a supervised run is live; init is not part of the loop"

#: Every file init can write at a base directory.
INIT_ARTIFACTS = ("gymrat.toml", "gymrat-runbook.md", ".claude/skills/gymrat/SKILL.md")


@pytest.fixture
def supervise_lock(repo: str) -> Iterator[None]:
    """Hold the real supervise lock for ``repo`` for the duration of the test."""
    lock = hold_supervise_lock(repo)
    yield
    lock.release()


@pytest.fixture
def existing_config_cwd(_in_non_repo: None, tmp_path: Path) -> Path:
    """A non-repo cwd with a pre-existing ``gymrat.toml`` already written."""
    (tmp_path / "gymrat.toml").write_text(EXISTING_CONFIG, encoding="utf-8")
    return tmp_path


@pytest.fixture
def live_repo(repo: str, supervise_lock: None) -> str:
    """A chdir'd scratch repository with a live supervised-run budget."""
    write_budget_file(repo)
    return repo


def _written_artifacts(base: str) -> list[str]:
    """The init artifacts present under ``base``."""
    return [name for name in INIT_ARTIFACTS if (Path(base) / name).exists()]


# ---------------------------------------------------------------------------
# --bench flag forms
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_init_when_short_bench_flag_does_scaffold_the_config(tmp_path: Path):
    result = runner.invoke(app, ["init", "-b", "npm run bench"])

    assert result.exit_code == 0
    assert (tmp_path / "gymrat.toml").exists()


# ---------------------------------------------------------------------------
# missing --bench
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_init_when_bench_missing_does_exit_two_naming_bench():
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 2
    assert "--bench" in result.stderr


# ---------------------------------------------------------------------------
# existing gymrat.toml at the resolved base
# ---------------------------------------------------------------------------


def test_init_when_config_already_exists_does_keep_it_while_scaffolding_the_rest(
    existing_config_cwd: Path,
):
    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert (existing_config_cwd / "gymrat.toml").read_text(encoding="utf-8") == EXISTING_CONFIG
    assert re.search(r"config.*already exists at gymrat\.toml", result.stdout, re.IGNORECASE)
    assert re.search(r"runbook.*created at", result.stdout, re.IGNORECASE)


def test_init_when_config_already_exists_does_not_require_bench(existing_config_cwd: Path):
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert (existing_config_cwd / "gymrat-runbook.md").exists()


# ---------------------------------------------------------------------------
# blocked artifact path exits with code 2
# ---------------------------------------------------------------------------


def _skill_path_directory(base: Path) -> None:
    """Make the skill file path a directory, beside an existing config."""
    (base / "gymrat.toml").write_text(EXISTING_CONFIG, encoding="utf-8")
    (base / ".claude" / "skills" / "gymrat" / "SKILL.md").mkdir(parents=True)


def _runbook_path_directory(base: Path) -> None:
    """Make the runbook path a directory."""
    (base / "gymrat-runbook.md").mkdir()


def _runbook_symlink(base: Path) -> None:
    """Make the runbook path a symlink to a regular file."""
    target = base / "real.md"
    target.write_text("# target\n", encoding="utf-8")
    (base / "gymrat-runbook.md").symlink_to(target)


@pytest.mark.parametrize(
    ("block", "argv", "named"),
    [
        pytest.param(_skill_path_directory, ["init"], "SKILL.md", id="skill-path-is-a-directory"),
        pytest.param(
            _runbook_path_directory,
            ["init", "--bench", "npm run bench"],
            "gymrat-runbook.md",
            id="runbook-path-is-a-directory",
        ),
        pytest.param(
            _runbook_symlink,
            ["init", "--bench", "npm run bench"],
            "gymrat-runbook.md",
            id="runbook-is-a-symlink",
        ),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_init_when_artifact_path_blocked_does_exit_two_naming_it(
    block: Callable[[Path], None], argv: list[str], named: str, tmp_path: Path
):
    block(tmp_path)

    result = runner.invoke(app, argv)

    assert result.exit_code == 2
    assert named in result.stderr


# ---------------------------------------------------------------------------
# success summary
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_init_when_scaffolding_succeeds_does_close_the_summary_on_the_doctor_pointer():
    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    out = result.stdout
    assert "Config: created at gymrat.toml" in out
    assert result.stderr == ""
    lines = strip_ansi(out).rstrip("\n").split("\n")
    pointer_index = next(i for i, line in enumerate(lines) if "gymrat doctor" in line)
    assert lines[pointer_index].strip() == "Run gymrat doctor to verify the setup."
    # The hint closes the artifact block directly — no blank line before it.
    assert lines[pointer_index - 1].strip().startswith("Skill:")


@pytest.mark.usefixtures("_in_non_repo")
def test_init_when_runbook_already_existed_does_report_it(tmp_path: Path):
    (tmp_path / "gymrat-runbook.md").write_text("# Existing\n", encoding="utf-8")

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert re.search(r"runbook.*already exist", result.stdout, re.IGNORECASE)


@pytest.mark.parametrize(
    ("flag", "report"),
    [
        pytest.param("--no-runbook", r"runbook.*(decline|skip)", id="runbook"),
        pytest.param("--no-skill", r"skill.*(decline|skip)", id="skill"),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_init_when_artifact_declined_does_report_it(flag: str, report: str):
    result = runner.invoke(app, ["init", "--bench", "npm run bench", flag])

    assert result.exit_code == 0
    assert re.search(report, result.stdout, re.IGNORECASE)


@pytest.mark.usefixtures("_in_non_repo")
def test_init_when_colored_does_dim_the_doctor_pointer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORCE_COLOR", "1")

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert "`" not in result.stdout
    pointer = next(
        line for line in result.stdout.split("\n") if "gymrat doctor" in strip_ansi(line)
    )
    assert pointer.startswith("\x1b[2m")


@pytest.mark.parametrize(
    ("env", "flag", "styled"),
    [
        pytest.param("FORCE_COLOR", "--no-color", False, id="no-color-beats-force-color-env"),
        pytest.param("NO_COLOR", "--color", True, id="color-beats-no-color-env"),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_init_when_color_flag_given_does_style_the_summary_despite_the_env(
    env: str, flag: str, styled: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(env, "1")

    result = runner.invoke(app, ["init", "--bench", "npm run bench", flag])

    assert result.exit_code == 0
    assert ("\x1b[" in result.stdout) is styled


# ---------------------------------------------------------------------------
# broken pipe on stdout
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_init_when_stdout_reader_closed_does_exit_zero_without_stderr():
    result = FailingStdoutRunner(closed_stdout_error()).invoke(
        app, ["init", "--bench", "npm run bench"]
    )

    assert (result.exit_code, result.stderr) == (0, "")


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
    # The summary paths are navigable from cwd: packages/core/ reaches ../../gymrat.toml.
    assert str(Path("../..") / "gymrat.toml") in result.stdout


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
    records_before = log_records(live_repo)
    worktree = experiment_worktree_dir(live_repo)
    monkeypatch.chdir(worktree)

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 2
    assert LIVE_REFUSAL in result.stderr
    assert _written_artifacts(live_repo) == []
    assert _written_artifacts(worktree) == []
    assert log_records(live_repo) == records_before
