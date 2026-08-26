"""Tests for the ``init`` scaffold that writes the config, runbook, and skill.

Each case uses ``tmp_path`` as the base directory and drives ``scaffold`` with a
``ScaffoldRequest``. The suite pins the config key ordering, the hand-written
TOML format (``json.dumps`` for string escaping), the validate-before-any-write
ordering (a broken bench leaves nothing behind), the runbook/skill status
reporting, and the exclusive-create refusal to overwrite an existing
``gymrat.toml``.
"""

import tomllib
from pathlib import Path

import pytest

import gymrat_py.init.scaffold as scaffold_module
from gymrat_py.config import load_config_file
from gymrat_py.errors import GymratError
from gymrat_py.init.scaffold import (
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

    monkeypatch.setattr("gymrat_py.init.scaffold.read_bundled_skill", raise_missing)


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


def test_scaffold_when_config_already_exists_does_raise_instead_of_overwriting(
    tmp_path: Path,
):
    (tmp_path / "gymrat.toml").write_text('bench = "old"\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        scaffold(str(tmp_path), ScaffoldRequest(bench="npm run bench"))


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
