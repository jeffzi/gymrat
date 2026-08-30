"""Tests for the ``init`` scaffold that writes the config, runbook, and skill.

Each case uses ``tmp_path`` as the base directory and drives ``scaffold`` with a
``ScaffoldRequest``. The suite pins the config key ordering, the hand-written
TOML format (``json.dumps`` for string escaping), the validate-before-any-write
ordering (a broken bench leaves nothing behind), the runbook/skill status
reporting, and the re-run behavior over an existing ``gymrat.toml`` (left
byte-identical, remaining artifacts still filled in).
"""

import sys
import tomllib
from pathlib import Path

import pytest

import gymrat.init.scaffold as scaffold_module
from gymrat.config import load_config_file
from gymrat.errors import GymratError
from gymrat.init.scaffold import (
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

    config = load_config_file(tmp_path / "gymrat.toml")
    assert config.bench == "npm run bench"
    assert config.runbook == "gymrat-runbook.md"
    assert config.adapter is None
    assert config.checks is None
    assert config.primary is None
    assert config.stop is None


def test_scaffold_when_defaults_does_produce_one_key_per_line_with_trailing_newline(
    tmp_path: Path,
):
    scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    raw = (tmp_path / "gymrat.toml").read_text(encoding="utf-8")
    assert raw == 'bench = "npm run bench"\nrunbook = "gymrat-runbook.md"\n'


# ---------------------------------------------------------------------------
# runbook=False omits runbook
# ---------------------------------------------------------------------------


def test_scaffold_when_runbook_false_does_omit_runbook_key(tmp_path: Path):
    scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", runbook=False))

    config = _read_config(tmp_path)
    assert config == {"bench": "npm run bench"}


def test_scaffold_when_runbook_false_does_not_create_runbook_file(tmp_path: Path):
    scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", runbook=False))

    assert not (tmp_path / "gymrat-runbook.md").exists()


def test_scaffold_when_runbook_false_does_report_declined(tmp_path: Path):
    result = scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", runbook=False))

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
    assert "# Optimization Runbook" in content
    assert "## Goal" in content
    assert "## Gating metrics" in content
    assert "## Constraints" in content
    assert "## Approaches to try" in content
    assert "gymrat supervise" in content


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


def test_scaffold_when_skill_already_exists_does_leave_it_untouched(tmp_path: Path):
    skill_dir = tmp_path / ".claude" / "skills" / "gymrat"
    skill_dir.mkdir(parents=True)
    existing = "# Custom Skill\n"
    (skill_dir / "SKILL.md").write_text(existing, encoding="utf-8")

    scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == existing


def test_scaffold_when_skill_already_exists_does_report_exists(tmp_path: Path):
    skill_dir = tmp_path / ".claude" / "skills" / "gymrat"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Custom\n", encoding="utf-8")

    result = scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert result.skill == ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="exists")


def test_scaffold_when_skill_declined_does_not_create_skill_file(tmp_path: Path):
    scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=False))

    assert not (tmp_path / ".claude" / "skills" / "gymrat" / "SKILL.md").exists()


def test_scaffold_when_skill_declined_does_report_declined(tmp_path: Path):
    result = scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=False))

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

    monkeypatch.setattr("gymrat.init.scaffold.read_bundled_skill", raise_missing)


def test_scaffold_when_skill_read_fails_does_not_leave_config_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _break_bundled_skill(monkeypatch)

    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

    assert not (tmp_path / "gymrat.toml").exists()


def test_scaffold_when_skill_read_fails_does_not_leave_runbook_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _break_bundled_skill(monkeypatch)

    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))

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


def test_scaffold_when_runbook_path_is_a_directory_does_raise_gymrat_error(
    tmp_path: Path,
):
    (tmp_path / "gymrat-runbook.md").mkdir()

    with pytest.raises(GymratError, match="gymrat-runbook.md"):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))


def test_scaffold_when_skill_path_is_a_directory_does_raise_gymrat_error(
    tmp_path: Path,
):
    (tmp_path / ".claude" / "skills" / "gymrat" / "SKILL.md").mkdir(parents=True)

    with pytest.raises(GymratError, match="SKILL.md"):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench", install_skill=True))


def test_scaffold_when_runbook_path_blocked_does_not_write_config(tmp_path: Path):
    (tmp_path / "gymrat-runbook.md").mkdir()

    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    assert not (tmp_path / "gymrat.toml").exists()


def test_scaffold_when_skill_path_blocked_does_not_write_config(tmp_path: Path):
    (tmp_path / ".claude" / "skills" / "gymrat" / "SKILL.md").mkdir(parents=True)

    with pytest.raises(GymratError):
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


def test_scaffold_when_runbook_path_is_a_symlink_does_raise_gymrat_error(
    tmp_path: Path,
):
    target = tmp_path / "some-file.md"
    target.write_text("# target\n", encoding="utf-8")
    (tmp_path / "gymrat-runbook.md").symlink_to(target)

    with pytest.raises(GymratError, match="gymrat-runbook.md"):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))


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


def test_scaffold_when_symlink_blocked_does_not_write_config(tmp_path: Path):
    target = tmp_path / "some-file.md"
    target.write_text("# target\n", encoding="utf-8")
    (tmp_path / "gymrat-runbook.md").symlink_to(target)

    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    assert not (tmp_path / "gymrat.toml").exists()


# ---------------------------------------------------------------------------
# atomic config write
# ---------------------------------------------------------------------------


def test_scaffold_when_config_write_fails_does_not_leave_partial_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A crash during the config write must not leave a truncated gymrat.toml."""

    def exploding_replace(src: object, dst: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr("os.replace", exploding_replace)

    with pytest.raises(GymratError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    assert not (tmp_path / "gymrat.toml").exists()
    # Also verify no .tmp sibling was left behind.
    tmp_files = list(tmp_path.glob("gymrat.toml.*"))
    assert tmp_files == []


# ---------------------------------------------------------------------------
# filesystem failures surface as GymratError
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_scaffold_when_base_dir_not_writable_does_raise_gymrat_error_with_path(
    tmp_path: Path,
):
    read_only = tmp_path / "locked"
    read_only.mkdir()
    read_only.chmod(0o444)

    with pytest.raises(GymratError, match="locked"):
        scaffold(str(read_only), ScaffoldRequest(bench="npm run bench"))

    read_only.chmod(0o755)


def test_scaffold_when_filesystem_error_does_include_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """OSError during writing surfaces as GymratError with a hint.

    The report-a-bug footer must never appear.
    """
    from gymrat.errors import hint_of

    def exploding_replace(src: object, dst: object) -> None:
        msg = "Read-only file system"
        raise OSError(msg)

    monkeypatch.setattr("os.replace", exploding_replace)

    with pytest.raises(GymratError) as exc_info:
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))

    assert hint_of(exc_info.value) is not None


# ---------------------------------------------------------------------------
# hand-rolled TOML encoding (no tomli_w dependency)
# ---------------------------------------------------------------------------


def test_scaffold_module_does_not_import_tomli_w():
    source = Path(scaffold_module.__file__).read_text(encoding="utf-8")
    assert "tomli_w" not in source


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
