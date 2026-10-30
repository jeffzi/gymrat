"""Command-level tests for ``gymrat init``.

Each case drives the assembled app through :class:`typer.testing.CliRunner`. The
non-interactive command takes ``--bench`` and optional ``--no-runbook`` /
``--no-skill`` flags, builds a :class:`ScaffoldRequest`, and delegates to
:func:`scaffold`. Usage errors, the re-run over an existing ``gymrat.toml``,
and the base-directory resolution are exercised the way a shell would invoke
them.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli.app import app
from tests._ansi import strip_ansi
from tests.cli._origin import set_origin
from tests.cli._runner import (
    runner,
)
from tests.config._toml import EXISTING_CONFIG, write_raw
from tests.session._budget import install_budget

LIVE_REFUSAL = "a supervised run is live; init is not part of the loop"


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


def test_init_when_config_already_exists_does_not_require_bench(existing_config_cwd: Path):
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert (existing_config_cwd / "gymrat-runbook.md").exists()


# ---------------------------------------------------------------------------
# success summary
# ---------------------------------------------------------------------------


_CONFIG_CREATED = "created at gymrat.toml"
_RUNBOOK_CREATED = "created at gymrat-runbook.md"
_SKILL_CREATED = f"created at {Path('.claude/skills/gymrat/SKILL.md')}"


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
            [_CONFIG_CREATED, _RUNBOOK_CREATED, _SKILL_CREATED],
            id="all-created",
        ),
        pytest.param(
            _plant_runbook,
            [],
            [_CONFIG_CREATED, "already exists at gymrat-runbook.md", _SKILL_CREATED],
            id="runbook-already-existed",
        ),
        pytest.param(
            _leave_base_empty,
            ["--no-runbook"],
            [_CONFIG_CREATED, "declined", _SKILL_CREATED],
            id="runbook-declined",
        ),
        pytest.param(
            _leave_base_empty,
            ["--no-skill"],
            [_CONFIG_CREATED, _RUNBOOK_CREATED, "declined"],
            id="skill-declined",
        ),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_init_when_scaffolding_succeeds_does_list_each_artifact_with_its_status(
    tmp_path: Path, arrange: Callable[[Path], None], flags: list[str], summary: list[str]
):
    arrange(tmp_path)

    result = runner.invoke(app, ["init", "--bench", "npm run bench", *flags])

    artifact_lines = strip_ansi(result.stdout).splitlines()[:3]
    assert result.exit_code == 0
    assert [line.split(": ", 1)[1] for line in artifact_lines] == summary


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


def test_init_when_run_in_a_git_repo_subdirectory_does_print_summary_paths_relative_to_cwd(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    root = create_scratch_repo()
    nested = Path(root) / "packages" / "core"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    result = runner.invoke(app, ["init", "--bench", "npm run bench"])

    assert str(Path("../..") / "gymrat.toml") in result.stdout


# ---------------------------------------------------------------------------
# refusal while a supervised run is live
# ---------------------------------------------------------------------------


def test_init_when_supervised_run_live_does_refuse(live_repo: str, monkeypatch: pytest.MonkeyPatch):
    # Run from the tool and without a bench: the live check alone decides the refusal.
    set_origin(monkeypatch, "tool")

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 2
    assert LIVE_REFUSAL in result.stderr
