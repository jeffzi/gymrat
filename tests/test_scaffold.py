"""Tests for the ``init`` scaffold that writes the config, runbook, and skill.

Each case uses ``tmp_path`` as the base directory and drives ``scaffold`` with a
``ScaffoldRequest``. The suite pins the config key ordering, the hand-written
TOML format (``json.dumps`` for string escaping), the validate-before-any-write
ordering (a broken bench leaves nothing behind), the runbook/skill status
reporting, and the re-run behavior over an existing ``gymrat.toml`` (left
byte-identical, remaining artifacts still filled in).
"""

import os
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.config import load_config_file_collecting
from gymrat.errors import GymratError
from gymrat.scaffold import (
    SKILL_RELATIVE_PATH,
    ScaffoldArtifact,
    ScaffoldRequest,
    scaffold,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_config(base: Path) -> dict[str, object]:
    return tomllib.loads((base / "gymrat.toml").read_text(encoding="utf-8"))


EXISTING_CONFIG = 'bench = "old"\n'


@pytest.fixture
def existing_config_dir(tmp_path: Path) -> Path:
    """A ``tmp_path`` with a pre-existing ``gymrat.toml`` already written."""
    (tmp_path / "gymrat.toml").write_text(EXISTING_CONFIG, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# basic scaffold with defaults
# ---------------------------------------------------------------------------


def test_scaffold_when_defaults_does_write_a_loadable_config_beside_the_runbook_stub(
    tmp_path: Path,
):
    scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    raw = (tmp_path / "gymrat.toml").read_bytes()
    assert raw == b'bench = "npm run bench"\nrunbook = "gymrat-runbook.md"\n'
    assert load_config_file_collecting(tmp_path / "gymrat.toml", required=True).problems == []
    content = (tmp_path / "gymrat-runbook.md").read_text(encoding="utf-8")
    assert content == (
        "# Optimization Runbook\n"
        "\n"
        "## Goal\n"
        "\n"
        "<!-- Describe the optimization goal here. -->\n"
        "\n"
        "## Gating metrics\n"
        "\n"
        "<!-- List the metrics that must not regress. -->\n"
        "\n"
        "## Constraints\n"
        "\n"
        "<!-- List any constraints on the optimization. -->\n"
        "\n"
        "## Approaches to try\n"
        "\n"
        "<!-- List strategies for the agent to explore. -->\n"
        "\n"
        "`gymrat supervise` injects this file into the agent's instructions.\n"
    )


# ---------------------------------------------------------------------------
# runbook=False omits runbook
# ---------------------------------------------------------------------------


def test_scaffold_when_runbook_false_does_omit_runbook_and_report_declined(tmp_path: Path):
    result = scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", runbook=False))

    config = _read_config(tmp_path)
    assert config == {"bench": "npm run bench"}
    assert not (tmp_path / "gymrat-runbook.md").exists()
    assert result.runbook.status == "declined"


# ---------------------------------------------------------------------------
# special character round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bench",
    [
        pytest.param('echo "hello"', id="double-quotes"),
        pytest.param("path\\to\\bench", id="backslashes"),
        pytest.param("bench éàü ☃", id="non-ascii"),
        pytest.param('tricky "val\\ue"', id="mixed-quotes-backslash"),
    ],
)
def test_scaffold_when_bench_has_special_chars_does_round_trip(tmp_path: Path, bench: str):
    scaffold(str(tmp_path), ScaffoldRequest(bench=bench, runbook=False))

    parsed = _read_config(tmp_path)
    assert parsed["bench"] == bench


#: What each artifact holds when a test plants it before the run.
_PLANTED = {
    "gymrat.toml": EXISTING_CONFIG,
    "gymrat-runbook.md": "# My Custom Runbook\n",
    SKILL_RELATIVE_PATH: "# Custom Skill\n",
}


@pytest.mark.parametrize(
    ("planted", "statuses"),
    [
        pytest.param(("gymrat.toml",), ("exists", "created", "created"), id="config"),
        pytest.param(("gymrat-runbook.md",), ("created", "exists", "created"), id="runbook"),
        pytest.param((SKILL_RELATIVE_PATH,), ("created", "created", "exists"), id="skill"),
        pytest.param(tuple(_PLANTED), ("exists", "exists", "exists"), id="every-artifact"),
    ],
)
def test_scaffold_when_artifacts_already_exist_does_leave_them_and_report_exists(
    tmp_path: Path, planted: tuple[str, ...], statuses: tuple[str, str, str]
):
    for relative in planted:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(_PLANTED[relative], encoding="utf-8")

    result = scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert (result.config.status, result.runbook.status, result.skill.status) == statuses
    assert {
        relative: (tmp_path / relative).read_text(encoding="utf-8") for relative in planted
    } == {relative: _PLANTED[relative] for relative in planted}


# ---------------------------------------------------------------------------
# skill file behavior
# ---------------------------------------------------------------------------


def test_scaffold_when_skill_declined_does_not_create_skill_and_report_declined(tmp_path: Path):
    result = scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=False))

    assert not (tmp_path / ".claude" / "skills" / "gymrat" / "SKILL.md").exists()
    assert result.skill.status == "declined"


# ---------------------------------------------------------------------------
# failure ordering
# ---------------------------------------------------------------------------


def test_scaffold_when_bench_empty_does_raise_before_writing(tmp_path: Path):
    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench=""))

    assert not (tmp_path / "gymrat.toml").exists()


def _break_bundled_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make reading the bundled skill fail, standing in for a broken install."""

    def raise_missing() -> str:
        message = "bundled skill missing"
        raise GymratError(message)

    monkeypatch.setattr("gymrat.scaffold.read_bundled_skill", raise_missing)


def test_scaffold_when_skill_read_fails_does_not_leave_config_or_runbook_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _break_bundled_skill(monkeypatch)

    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert not (tmp_path / "gymrat.toml").exists()
    assert not (tmp_path / "gymrat-runbook.md").exists()


def test_scaffold_when_skill_read_fails_does_not_delete_a_pre_existing_config(
    existing_config_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _break_bundled_skill(monkeypatch)

    with pytest.raises(GymratError):
        scaffold(str(existing_config_dir), ScaffoldRequest(install_skill=True))

    assert (existing_config_dir / "gymrat.toml").read_text(encoding="utf-8") == EXISTING_CONFIG


# ---------------------------------------------------------------------------
# re-run over an existing gymrat.toml
# ---------------------------------------------------------------------------


def test_scaffold_when_config_already_exists_does_still_create_runbook_and_skill(
    existing_config_dir: Path,
):
    result = scaffold(str(existing_config_dir), ScaffoldRequest(install_skill=True))

    assert result.runbook == ScaffoldArtifact(path="gymrat-runbook.md", status="created")
    assert result.skill == ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="created")
    assert (existing_config_dir / "gymrat-runbook.md").exists()
    assert (existing_config_dir / ".claude" / "skills" / "gymrat" / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# blocked artifact path raises GymratError
# ---------------------------------------------------------------------------


def _directory_at(*relatives: str) -> Callable[[Path], None]:
    def plant(base: Path) -> None:
        for relative in relatives:
            (base / relative).mkdir(parents=True)

    return plant


def _symlink_at(relative: str, *, dangling: bool = False) -> Callable[[Path], None]:
    def plant(base: Path) -> None:
        target = base / "symlink-target.md"
        if not dangling:
            target.write_text("# target\n", encoding="utf-8")
        (base / relative).parent.mkdir(parents=True, exist_ok=True)
        (base / relative).symlink_to(target)

    return plant


def _config_then_skill_directory(base: Path) -> None:
    (base / "gymrat.toml").write_text(EXISTING_CONFIG, encoding="utf-8")
    (base / SKILL_RELATIVE_PATH).mkdir(parents=True)


def _config_text(base: Path) -> str | None:
    config = base / "gymrat.toml"
    return config.read_text(encoding="utf-8") if config.is_file() else None


@pytest.mark.parametrize(
    ("plant", "blocked", "config"),
    [
        pytest.param(
            _directory_at("gymrat-runbook.md"), "gymrat-runbook.md", None, id="runbook-directory"
        ),
        pytest.param(
            _directory_at(SKILL_RELATIVE_PATH), SKILL_RELATIVE_PATH, None, id="skill-directory"
        ),
        pytest.param(
            _directory_at("gymrat.toml", "gymrat-runbook.md", SKILL_RELATIVE_PATH),
            f"gymrat.toml, gymrat-runbook.md, {SKILL_RELATIVE_PATH}",
            None,
            id="every-path-a-directory-named-in-write-order",
        ),
        pytest.param(
            _symlink_at("gymrat-runbook.md"), "gymrat-runbook.md", None, id="runbook-symlink"
        ),
        pytest.param(
            _symlink_at(SKILL_RELATIVE_PATH), SKILL_RELATIVE_PATH, None, id="skill-symlink"
        ),
        pytest.param(
            _symlink_at("gymrat-runbook.md", dangling=True),
            "gymrat-runbook.md",
            None,
            id="runbook-dangling-symlink",
        ),
        pytest.param(
            _config_then_skill_directory,
            SKILL_RELATIVE_PATH,
            EXISTING_CONFIG,
            id="existing-config-left-untouched",
        ),
    ],
)
def test_scaffold_when_an_artifact_path_is_blocked_does_raise_naming_it_before_writing(
    tmp_path: Path, plant: Callable[[Path], None], blocked: str, config: str | None
):
    plant(tmp_path)

    with pytest.raises(GymratError) as caught:
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert (str(caught.value), caught.value.hint) == (
        f"Blocked path: {blocked}",
        "Remove or rename the blocking entry and re-run.",
    )
    assert _config_text(tmp_path) == config


def test_scaffold_when_skipped_artifact_paths_are_directories_does_write_the_config(
    tmp_path: Path,
):
    (tmp_path / "gymrat-runbook.md").mkdir()
    (tmp_path / ".claude" / "skills" / "gymrat" / "SKILL.md").mkdir(parents=True)

    result = scaffold(
        str(tmp_path),
        ScaffoldRequest(bench="npm run bench", runbook=False, install_skill=False),
    )

    assert result.config.status == "created"


def test_scaffold_when_config_path_cannot_be_checked_does_raise_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def denied_exists(_path: Path, **_kwargs: object) -> bool:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "exists", denied_exists)

    with pytest.raises(GymratError) as caught:
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    assert (str(caught.value), caught.value.hint) == (
        f"Cannot access {tmp_path / 'gymrat.toml'}: [Errno 13] Permission denied",
        "Check directory permissions.",
    )


# ---------------------------------------------------------------------------
# atomic config write
# ---------------------------------------------------------------------------


def test_scaffold_when_config_write_fails_does_raise_with_the_os_reason_and_leave_no_partial_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def exploding_replace(src: object, dst: object) -> None:
        msg = "Read-only file system"
        raise OSError(msg)

    monkeypatch.setattr("os.replace", exploding_replace)

    with pytest.raises(GymratError) as exc_info:
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    # The report-a-bug footer must never appear for a filesystem error.
    assert (str(exc_info.value), exc_info.value.hint) == (
        f"Cannot write gymrat.toml in {tmp_path}",
        "Read-only file system",
    )
    assert list(tmp_path.glob("gymrat.toml*")) == []


# ---------------------------------------------------------------------------
# filesystem failures surface as GymratError
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX file modes; root bypasses them",
)
def test_scaffold_when_base_dir_not_writable_does_raise_gymrat_error_with_path(
    tmp_path: Path,
):
    read_only = tmp_path / "locked"
    read_only.mkdir()
    read_only.chmod(0o444)

    try:
        with pytest.raises(GymratError, match="locked"):
            scaffold(str(read_only), ScaffoldRequest(bench="npm run bench"))
    finally:
        read_only.chmod(0o755)


STRAY_NAME = "stray.txt"
STRAY_CONTENT = "left by someone else\n"


def _fail_write_of(monkeypatch: pytest.MonkeyPatch, name: str, *, drop_stray: bool = False) -> None:
    """Make the rename that puts the file called ``name`` in place fail like a full disk.

    With ``drop_stray``, a foreign file appears beside the destination just before the failure.
    """
    real_replace = os.replace

    def replace_that_fails_on_name(
        src: str | os.PathLike[str], dst: str | os.PathLike[str]
    ) -> None:
        if Path(dst).name == name:
            if drop_stray:
                (Path(dst).parent / STRAY_NAME).write_text(STRAY_CONTENT, encoding="utf-8")
            msg = "No space left on device"
            raise OSError(msg)
        real_replace(src, dst)

    monkeypatch.setattr("os.replace", replace_that_fails_on_name)


def _files(base: Path) -> dict[str, str]:
    return {
        path.relative_to(base).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def _tree(base: Path) -> dict[str, str | None]:
    """Every entry under ``base``: a file maps to its text, a directory to ``None``."""
    return {
        path.relative_to(base).as_posix(): (
            path.read_text(encoding="utf-8") if path.is_file() else None
        )
        for path in sorted(base.rglob("*"))
    }


def _plant_tree(base: Path, tree: dict[str, str | None]) -> None:
    """Create the entries of ``tree`` under ``base``, parents listed before children."""
    for relative, content in tree.items():
        if content is None:
            (base / relative).mkdir()
        else:
            (base / relative).write_text(content, encoding="utf-8")


def test_scaffold_when_runbook_write_fails_does_raise_naming_it_and_remove_the_config_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _fail_write_of(monkeypatch, "gymrat-runbook.md")

    with pytest.raises(GymratError) as exc_info:
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    assert str(exc_info.value) == f"Cannot write gymrat-runbook.md in {tmp_path}"
    assert exc_info.value.hint == "No space left on device"
    assert not (tmp_path / "gymrat.toml").exists()


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("gymrat-runbook.md", id="runbook"),
        pytest.param("SKILL.md", id="skill"),
    ],
)
def test_scaffold_when_artifact_write_fails_does_leave_no_partial_or_temporary_file(
    existing_config_dir: Path, monkeypatch: pytest.MonkeyPatch, name: str
):
    _fail_write_of(monkeypatch, name)

    with pytest.raises(GymratError):
        scaffold(str(existing_config_dir), ScaffoldRequest(install_skill=True))

    assert _files(existing_config_dir) == {"gymrat.toml": EXISTING_CONFIG}


# ---------------------------------------------------------------------------
# a failed run removes what it created and nothing else
# ---------------------------------------------------------------------------

EXISTING_RUNBOOK = "# My Custom Runbook\n"
BLOCKING_FILE = "not a directory\n"


@pytest.mark.parametrize(
    "already_there",
    [
        pytest.param({}, id="nothing-existed"),
        pytest.param({"gymrat.toml": EXISTING_CONFIG}, id="config-existed"),
        pytest.param({"gymrat-runbook.md": EXISTING_RUNBOOK}, id="runbook-existed"),
    ],
)
def test_scaffold_when_skill_write_fails_does_remove_only_the_artifacts_this_run_created(
    tmp_path: Path, already_there: dict[str, str]
):
    before = {".claude": BLOCKING_FILE, **already_there}
    for relative, content in before.items():
        (tmp_path / relative).write_text(content, encoding="utf-8")

    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert _files(tmp_path) == before


@pytest.mark.parametrize(
    "already_there",
    [
        pytest.param({}, id="nothing-existed"),
        pytest.param({".claude": None}, id="empty-claude-directory-existed"),
        pytest.param(
            {".claude": None, ".claude/notes.md": STRAY_CONTENT},
            id="claude-directory-with-unrelated-file-existed",
        ),
    ],
)
def test_scaffold_when_skill_write_fails_does_remove_the_directories_this_run_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, already_there: dict[str, str | None]
):
    _plant_tree(tmp_path, already_there)
    _fail_write_of(monkeypatch, "SKILL.md")

    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert _tree(tmp_path) == already_there


def test_scaffold_when_created_directory_is_no_longer_empty_does_leave_it_and_its_ancestors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _fail_write_of(monkeypatch, "SKILL.md", drop_stray=True)

    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert _tree(tmp_path) == {
        ".claude": None,
        ".claude/skills": None,
        ".claude/skills/gymrat": None,
        f".claude/skills/gymrat/{STRAY_NAME}": STRAY_CONTENT,
    }


def test_scaffold_when_directory_appears_before_its_creation_does_leave_that_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Another process wins the race for ``.claude/skills``: the directory is
    # already there when this run's own creation attempt reaches the filesystem.
    real_mkdir = os.mkdir

    def mkdir_that_loses_the_race_for_skills(
        path: str | os.PathLike[str], mode: int = 0o777
    ) -> None:
        if Path(path).name == "skills":
            real_mkdir(path, mode)
        real_mkdir(path, mode)

    monkeypatch.setattr("os.mkdir", mkdir_that_loses_the_race_for_skills)
    _fail_write_of(monkeypatch, "SKILL.md")

    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert _tree(tmp_path) == {".claude": None, ".claude/skills": None}


# ---------------------------------------------------------------------------
# rollback must not mask the original error
# ---------------------------------------------------------------------------


def test_scaffold_when_rollback_unlink_fails_does_propagate_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The runbook write fails after the config is written, so the except-block
    # tries to unlink the config; that unlink also raises, and the original
    # error must still propagate.
    _fail_write_of(monkeypatch, "gymrat-runbook.md")
    original_unlink = Path.unlink

    def unlink_that_fails_on_config(self: Path, *, missing_ok: bool = False) -> None:
        if self.name == "gymrat.toml":
            msg = "device removed"
            raise OSError(msg)
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink_that_fails_on_config)

    with pytest.raises(GymratError) as exc_info:
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    assert (str(exc_info.value), exc_info.value.hint) == (
        f"Cannot write gymrat-runbook.md in {tmp_path}",
        "No space left on device",
    )


def test_scaffold_when_rollback_directory_removal_fails_does_propagate_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def exploding_rmdir(path: str | os.PathLike[str]) -> None:
        msg = "device removed"
        raise OSError(msg)

    monkeypatch.setattr("os.rmdir", exploding_rmdir)
    _fail_write_of(monkeypatch, "SKILL.md")

    with pytest.raises(GymratError) as exc_info:
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert exc_info.value.hint == "No space left on device"


# ---------------------------------------------------------------------------
# returned artifact statuses (created)
# ---------------------------------------------------------------------------


def test_scaffold_when_skill_requested_and_absent_does_create_every_artifact(
    tmp_path: Path,
):
    result = scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert result.config == ScaffoldArtifact(path="gymrat.toml", status="created")
    assert result.runbook == ScaffoldArtifact(path="gymrat-runbook.md", status="created")
    assert result.skill == ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="created")
    skill_text = (tmp_path / SKILL_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "# Driving a gymrat optimization session" in skill_text
