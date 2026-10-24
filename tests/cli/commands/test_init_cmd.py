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
    supervise_lockfile_path,
)
from tests._ansi import strip_ansi
from tests._config import resolved_config
from tests._lock import lock_probe_error
from tests.cli._origin import set_origin
from tests.cli._runner import (
    runner,
)
from tests.config._toml import EXISTING_CONFIG, write_raw
from tests.session._budget import block_supervise_lock, install_budget, write_budget_file
from tests.session.records._fixtures import log_records

LIVE_REFUSAL = "a supervised run is live; init is not part of the loop"

#: Every file init can write at a base directory.
INIT_ARTIFACTS = ("gymrat.toml", "gymrat-runbook.md", ".claude/skills/gymrat/SKILL.md")


@pytest.fixture
def existing_config_cwd(_in_non_repo: None, tmp_path: Path) -> Path:
    """A non-repo cwd with a pre-existing ``gymrat.toml`` already written."""
    write_raw(tmp_path, EXISTING_CONFIG)
    return tmp_path


@pytest.fixture
def live_repo(repo: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """A chdir'd scratch repository with a live supervised-run budget."""
    install_budget(repo, monkeypatch)
    return repo


@pytest.fixture
def blocked_supervise_lock(repo: str) -> Iterator[str]:
    """A live-budget repository whose supervise OS lock file is a directory, yielding its path."""
    write_budget_file(repo)
    with block_supervise_lock(repo) as blocker:
        yield blocker


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
# success summary
# ---------------------------------------------------------------------------


_CONFIG_CREATED = "  Config: created at gymrat.toml\n"
_RUNBOOK_CREATED = "  Runbook: created at gymrat-runbook.md\n"
_SKILL_CREATED = f"  Skill: created at {Path('.claude/skills/gymrat/SKILL.md')}\n"
_DOCTOR_POINTER = "Run gymrat doctor to verify the setup.\n"


def _leave_base_empty(_base: Path) -> None:
    """Start from a base directory holding no init artifact."""


def _plant_runbook(base: Path) -> None:
    """Leave a runbook at ``base`` from an earlier run."""
    (base / "gymrat-runbook.md").write_text("# Existing\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("arrange", "flags", "summary"),
    [
        pytest.param(
            _leave_base_empty,
            [],
            _CONFIG_CREATED + _RUNBOOK_CREATED + _SKILL_CREATED,
            id="all-created",
        ),
        pytest.param(
            _plant_runbook,
            [],
            _CONFIG_CREATED + "  Runbook: already exists at gymrat-runbook.md\n" + _SKILL_CREATED,
            id="runbook-already-existed",
        ),
        pytest.param(
            _leave_base_empty,
            ["--no-runbook"],
            _CONFIG_CREATED + "  Runbook: declined\n" + _SKILL_CREATED,
            id="runbook-declined",
        ),
        pytest.param(
            _leave_base_empty,
            ["--no-skill"],
            _CONFIG_CREATED + _RUNBOOK_CREATED + "  Skill: declined\n",
            id="skill-declined",
        ),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_init_when_scaffolding_succeeds_does_list_each_artifact_closing_on_the_doctor_pointer(
    tmp_path: Path, arrange: Callable[[Path], None], flags: list[str], summary: str
):
    arrange(tmp_path)

    result = runner.invoke(app, ["init", "--bench", "npm run bench", *flags])

    # The pointer closes the artifact block directly — no blank line before it.
    assert (result.exit_code, strip_ansi(result.stdout), result.stderr) == (
        0,
        summary + _DOCTOR_POINTER,
        "",
    )


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


@pytest.mark.usefixtures("blocked_supervise_lock")
def test_init_when_supervise_lock_cannot_be_opened_does_exit_two_without_writing(repo: str):
    expected = lock_probe_error(supervise_lockfile_path(repo))

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert result.exit_code == 2
    assert " ".join(strip_ansi(result.stderr).split()).startswith(
        " ".join(f"Error: {expected}".split())
    )
    assert result.stdout == ""
    assert _written_artifacts(repo) == []
