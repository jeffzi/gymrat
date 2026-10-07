"""Tests for the ``init`` scaffold that writes the config, runbook, and skill.

Each case uses ``tmp_path`` as the base directory and drives ``scaffold`` with a
``ScaffoldRequest``. The suite pins the config key ordering, the hand-written
TOML format (``json.dumps`` for string escaping), the validate-before-any-write
ordering (a broken bench leaves nothing behind), the runbook/skill status
reporting, and the re-run behavior over an existing ``gymrat.toml`` (left
byte-identical, remaining artifacts still filled in).
"""

import ast
import os
import sys
import tomllib
from pathlib import Path

import pytest

import gymrat.scaffold as scaffold_module
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


def test_scaffold_when_defaults_does_write_bench_and_runbook(tmp_path: Path):
    scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    config = _read_config(tmp_path)
    assert config == {"bench": "npm run bench", "runbook": "gymrat-runbook.md"}


def test_scaffold_when_defaults_does_reload_through_config_loader(tmp_path: Path):
    scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    loaded = load_config_file_collecting(tmp_path / "gymrat.toml", required=True)
    config = loaded.config_file
    assert loaded.problems == []
    assert config is not None
    assert config.bench == "npm run bench"
    assert config.runbook == "gymrat-runbook.md"


def test_scaffold_when_defaults_does_produce_one_key_per_line_with_trailing_newline(
    tmp_path: Path,
):
    scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    raw = (tmp_path / "gymrat.toml").read_bytes()
    assert raw == b'bench = "npm run bench"\nrunbook = "gymrat-runbook.md"\n'


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


# ---------------------------------------------------------------------------
# runbook stub behavior
# ---------------------------------------------------------------------------


def test_scaffold_when_runbook_true_does_create_stub_with_expected_sections(
    tmp_path: Path,
):
    scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

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


def test_scaffold_when_runbook_already_exists_does_leave_it_and_report_exists(
    tmp_path: Path,
):
    existing = "# My Custom Runbook\n"
    (tmp_path / "gymrat-runbook.md").write_text(existing, encoding="utf-8")

    result = scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    assert (tmp_path / "gymrat-runbook.md").read_text(encoding="utf-8") == existing
    assert result.runbook == ScaffoldArtifact(path="gymrat-runbook.md", status="exists")
    assert _read_config(tmp_path)["runbook"] == "gymrat-runbook.md"


# ---------------------------------------------------------------------------
# skill file behavior
# ---------------------------------------------------------------------------


def test_scaffold_when_skill_requested_and_absent_does_copy_bundled_skill(
    tmp_path: Path,
):
    scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    skill_path = tmp_path / ".claude" / "skills" / "gymrat" / "SKILL.md"
    assert skill_path.exists()
    assert "# Driving a gymrat optimization session" in skill_path.read_text(encoding="utf-8")


def test_scaffold_when_skill_already_exists_does_leave_it_untouched_and_report_exists(
    tmp_path: Path,
):
    skill_dir = tmp_path / ".claude" / "skills" / "gymrat"
    skill_dir.mkdir(parents=True)
    existing = "# Custom Skill\n"
    (skill_dir / "SKILL.md").write_text(existing, encoding="utf-8")

    result = scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == existing
    assert result.skill == ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="exists")


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


def test_scaffold_when_config_already_exists_does_leave_it_and_report_exists(
    existing_config_dir: Path,
):
    result = scaffold(
        str(existing_config_dir), ScaffoldRequest(bench="npm run bench", install_skill=True)
    )

    assert (existing_config_dir / "gymrat.toml").read_text(encoding="utf-8") == EXISTING_CONFIG
    assert result.config == ScaffoldArtifact(path="gymrat.toml", status="exists")


def test_scaffold_when_config_already_exists_does_still_create_runbook_and_skill(
    existing_config_dir: Path,
):
    result = scaffold(str(existing_config_dir), ScaffoldRequest(install_skill=True))

    assert result.runbook == ScaffoldArtifact(path="gymrat-runbook.md", status="created")
    assert result.skill == ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="created")
    assert (existing_config_dir / "gymrat-runbook.md").exists()
    assert (existing_config_dir / ".claude" / "skills" / "gymrat" / "SKILL.md").exists()


def test_scaffold_when_every_artifact_already_exists_does_report_all_of_them_exists(
    existing_config_dir: Path,
):
    (existing_config_dir / "gymrat-runbook.md").write_text("# Existing\n", encoding="utf-8")
    skill_dir = existing_config_dir / ".claude" / "skills" / "gymrat"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Custom\n", encoding="utf-8")

    result = scaffold(str(existing_config_dir), ScaffoldRequest(install_skill=True))

    assert result.config.status == "exists"
    assert result.runbook.status == "exists"
    assert result.skill.status == "exists"


# ---------------------------------------------------------------------------
# blocked artifact path raises GymratError
# ---------------------------------------------------------------------------


def test_scaffold_when_runbook_path_is_a_directory_does_raise_and_not_write_config(
    tmp_path: Path,
):
    (tmp_path / "gymrat-runbook.md").mkdir()

    with pytest.raises(GymratError) as caught:
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    assert (str(caught.value), caught.value.hint) == (
        "Blocked path: gymrat-runbook.md",
        "Remove or rename the blocking entry and re-run.",
    )
    assert not (tmp_path / "gymrat.toml").exists()


def test_scaffold_when_every_artifact_path_is_a_directory_does_raise_naming_each_in_write_order(
    tmp_path: Path,
):
    for relative in ("gymrat.toml", "gymrat-runbook.md", ".claude/skills/gymrat/SKILL.md"):
        (tmp_path / relative).mkdir(parents=True)

    with pytest.raises(GymratError) as caught:
        scaffold(str(tmp_path), ScaffoldRequest())

    assert str(caught.value) == (
        "Blocked path: gymrat.toml, gymrat-runbook.md, .claude/skills/gymrat/SKILL.md"
    )


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


def test_scaffold_when_skill_path_is_a_directory_does_raise_and_not_write_config(
    tmp_path: Path,
):
    (tmp_path / ".claude" / "skills" / "gymrat" / "SKILL.md").mkdir(parents=True)

    with pytest.raises(GymratError, match="SKILL.md"):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert not (tmp_path / "gymrat.toml").exists()


def test_scaffold_when_blocked_and_config_exists_does_not_touch_existing_config(
    existing_config_dir: Path,
):
    (existing_config_dir / ".claude" / "skills" / "gymrat" / "SKILL.md").mkdir(parents=True)

    with pytest.raises(GymratError):
        scaffold(str(existing_config_dir), ScaffoldRequest(install_skill=True))

    assert (existing_config_dir / "gymrat.toml").read_text(encoding="utf-8") == EXISTING_CONFIG


# ---------------------------------------------------------------------------
# symlinks at artifact paths are blocked
# ---------------------------------------------------------------------------


def test_scaffold_when_runbook_path_is_a_symlink_does_raise_and_not_write_config(
    tmp_path: Path,
):
    target = tmp_path / "some-file.md"
    target.write_text("# target\n", encoding="utf-8")
    (tmp_path / "gymrat-runbook.md").symlink_to(target)

    with pytest.raises(GymratError, match="gymrat-runbook.md"):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    assert not (tmp_path / "gymrat.toml").exists()


def test_scaffold_when_skill_path_is_a_symlink_does_raise_gymrat_error(
    tmp_path: Path,
):
    skill_dir = tmp_path / ".claude" / "skills" / "gymrat"
    skill_dir.mkdir(parents=True)
    target = tmp_path / "real-skill.md"
    target.write_text("# skill\n", encoding="utf-8")
    (skill_dir / "SKILL.md").symlink_to(target)

    with pytest.raises(GymratError, match="SKILL.md"):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))


def test_scaffold_when_runbook_path_is_a_dangling_symlink_does_raise_gymrat_error(
    tmp_path: Path,
):
    (tmp_path / "gymrat-runbook.md").symlink_to(tmp_path / "nonexistent")

    with pytest.raises(GymratError, match="gymrat-runbook.md"):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))


# ---------------------------------------------------------------------------
# atomic config write
# ---------------------------------------------------------------------------


def test_scaffold_when_config_write_fails_does_not_leave_partial_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def exploding_replace(src: object, dst: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr("os.replace", exploding_replace)

    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    assert not (tmp_path / "gymrat.toml").exists()
    tmp_files = list(tmp_path.glob("gymrat.toml.*"))
    assert tmp_files == []


def test_scaffold_when_artifacts_renamed_into_place_does_fsync_each_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real_fsync = os.fsync
    real_replace = os.replace
    synced_files: set[int] = set()
    synced_at_rename: dict[str, bool] = {}

    def recording_fsync(fd: int) -> None:
        real_fsync(fd)
        synced_files.add(os.fstat(fd).st_ino)

    def observing_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        synced_at_rename[Path(dst).name] = Path(src).stat().st_ino in synced_files
        real_replace(src, dst)

    monkeypatch.setattr("os.fsync", recording_fsync)
    monkeypatch.setattr("os.replace", observing_replace)

    scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert synced_at_rename == {
        "gymrat.toml": True,
        "gymrat-runbook.md": True,
        "SKILL.md": True,
    }


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


def test_scaffold_when_filesystem_error_does_include_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def exploding_replace(src: object, dst: object) -> None:
        msg = "Read-only file system"
        raise OSError(msg)

    monkeypatch.setattr("os.replace", exploding_replace)

    with pytest.raises(GymratError) as exc_info:
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    # The report-a-bug footer must never appear for a filesystem error.
    assert str(exc_info.value) == f"Cannot write gymrat.toml in {tmp_path}"
    assert exc_info.value.hint == "Read-only file system"


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


def test_scaffold_when_runbook_write_fails_does_raise_naming_the_artifact_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _fail_write_of(monkeypatch, "gymrat-runbook.md")

    with pytest.raises(GymratError) as exc_info:
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    assert str(exc_info.value) == f"Cannot write gymrat-runbook.md in {tmp_path}"
    assert exc_info.value.hint == "No space left on device"


def test_scaffold_when_runbook_write_fails_does_remove_the_config_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _fail_write_of(monkeypatch, "gymrat-runbook.md")

    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

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
# hand-rolled TOML encoding (no tomli_w dependency)
# ---------------------------------------------------------------------------


def test_scaffold_module_does_not_import_tomli_w():
    source = Path(scaffold_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("tomli_w"), "scaffold must not depend on tomli_w"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("tomli_w"), "scaffold must not depend on tomli_w"


# ---------------------------------------------------------------------------
# rollback must not mask the original error
# ---------------------------------------------------------------------------


def test_scaffold_when_rollback_unlink_fails_does_propagate_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The runbook write fails after the config is written, so the except-block
    # tries to unlink the config; that unlink also raises, and the original
    # error must still propagate.
    def exploding_write_artifact(*args: object, **kwargs: object) -> None:
        msg = "runbook write failed"
        raise GymratError(msg)

    monkeypatch.setattr(scaffold_module, "_write_artifact", exploding_write_artifact)

    original_unlink = Path.unlink

    def unlink_that_fails_on_config(self: Path, *, missing_ok: bool = False) -> None:
        if self.name == "gymrat.toml":
            msg = "device removed"
            raise OSError(msg)
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink_that_fails_on_config)

    with pytest.raises(GymratError, match="runbook write failed"):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))


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


def test_scaffold_when_all_artifacts_created_does_return_created_statuses(
    tmp_path: Path,
):
    result = scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert result.config == ScaffoldArtifact(path="gymrat.toml", status="created")
    assert result.runbook == ScaffoldArtifact(path="gymrat-runbook.md", status="created")
    assert result.skill == ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="created")
