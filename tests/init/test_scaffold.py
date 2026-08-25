"""Tests for the ``init`` scaffold that writes the config, runbook, and skill.

Each case uses ``tmp_path`` as the base directory and drives ``scaffold`` with a
``WizardResult``. The suite pins the config key ordering, the validate-before-any-write
ordering (a broken bench or a geomean stop target leaves nothing behind), the
runbook/skill status reporting, and the exclusive-create refusal to overwrite an
existing ``gymrat.json``.
"""

import json
from pathlib import Path

import pytest

from gymrat_py.errors import GymratError
from gymrat_py.init.scaffold import (
    SKILL_RELATIVE_PATH,
    ScaffoldArtifact,
    scaffold,
)
from gymrat_py.init.wizard import RunbookChoice, WizardResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(**overrides: object) -> WizardResult:
    """Build a minimal ``WizardResult`` with only the required fields set."""
    base: dict[str, object] = {"bench": "npm run bench", "runbook": False, "install_skill": False}
    base.update(overrides)
    return WizardResult(**base)  # type: ignore[arg-type]


def _read_config(base: Path) -> dict[str, object]:
    return json.loads((base / "gymrat.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# config object construction
# ---------------------------------------------------------------------------


def test_scaffold_when_only_bench_does_write_config_with_only_bench(tmp_path: Path):
    scaffold(str(tmp_path), _result())

    assert _read_config(tmp_path) == {"bench": "npm run bench"}


@pytest.mark.parametrize(
    ("override", "field", "expected"),
    [
        pytest.param({"adapter": "mitata"}, "adapter", "mitata", id="adapter"),
        pytest.param({"checks": "npm test"}, "checks", "npm test", id="checks"),
        pytest.param({"primary": "latency"}, "primary", "latency", id="primary"),
    ],
)
def test_scaffold_when_field_set_does_include_it_in_config(
    tmp_path: Path, override: dict[str, object], field: str, expected: str
):
    scaffold(str(tmp_path), _result(**override))

    assert _read_config(tmp_path)[field] == expected


def test_scaffold_when_stop_target_set_does_write_stop_with_only_target(tmp_path: Path):
    scaffold(str(tmp_path), _result(stop_target=1.5, primary="latency"))

    assert _read_config(tmp_path)["stop"] == {"targetValue": 1.5}


def test_scaffold_when_max_iterations_set_does_write_stop_with_only_max(tmp_path: Path):
    scaffold(str(tmp_path), _result(stop_max_iterations=10))

    assert _read_config(tmp_path)["stop"] == {"maxIterations": 10}


def test_scaffold_when_both_stop_fields_set_does_write_both(tmp_path: Path):
    scaffold(str(tmp_path), _result(stop_target=2.0, stop_max_iterations=20, primary="latency"))

    assert _read_config(tmp_path)["stop"] == {"targetValue": 2.0, "maxIterations": 20}


def test_scaffold_when_runbook_path_set_does_include_runbook_key(tmp_path: Path):
    scaffold(str(tmp_path), _result(runbook=RunbookChoice(path="gymrat-runbook.md")))

    assert _read_config(tmp_path)["runbook"] == "gymrat-runbook.md"


def test_scaffold_writes_config_keys_in_the_declared_order(tmp_path: Path):
    scaffold(
        str(tmp_path),
        _result(
            adapter="mitata",
            checks="npm test",
            primary="latency",
            stop_target=1.5,
            stop_max_iterations=20,
            runbook=RunbookChoice(path="gymrat-runbook.md"),
        ),
    )

    assert list(_read_config(tmp_path).keys()) == [
        "bench",
        "adapter",
        "checks",
        "primary",
        "stop",
        "runbook",
    ]


# ---------------------------------------------------------------------------
# validation before any write
# ---------------------------------------------------------------------------


def test_scaffold_when_schema_invalid_does_raise_before_writing_any_file(tmp_path: Path):
    with pytest.raises(GymratError):
        scaffold(str(tmp_path), _result(bench=""))

    assert not (tmp_path / "gymrat.json").exists()


def test_scaffold_when_stop_target_with_geomean_primary_does_raise_before_writing(tmp_path: Path):
    with pytest.raises(GymratError):
        scaffold(str(tmp_path), _result(stop_target=1.5))

    assert not (tmp_path / "gymrat.json").exists()


# ---------------------------------------------------------------------------
# runbook stub
# ---------------------------------------------------------------------------


def test_scaffold_when_runbook_requested_does_create_stub_with_expected_sections(tmp_path: Path):
    scaffold(str(tmp_path), _result(runbook=RunbookChoice(path="gymrat-runbook.md")))

    content = (tmp_path / "gymrat-runbook.md").read_text(encoding="utf-8")
    assert "# Optimization Runbook" in content
    assert "## Goal" in content
    assert "## Gating metrics" in content
    assert "## Constraints" in content
    assert "## Approaches to try" in content
    assert "gymrat supervise" in content


def test_scaffold_when_runbook_in_nested_directory_does_create_parents(tmp_path: Path):
    scaffold(str(tmp_path), _result(runbook=RunbookChoice(path="docs/runbooks/bench.md")))

    assert (tmp_path / "docs" / "runbooks" / "bench.md").exists()


def test_scaffold_when_runbook_path_exists_does_leave_it_and_still_record_key(tmp_path: Path):
    existing = "# My Custom Runbook\n"
    (tmp_path / "my-runbook.md").write_text(existing, encoding="utf-8")

    scaffold(str(tmp_path), _result(runbook=RunbookChoice(path="my-runbook.md")))

    assert (tmp_path / "my-runbook.md").read_text(encoding="utf-8") == existing
    assert _read_config(tmp_path)["runbook"] == "my-runbook.md"


def test_scaffold_when_runbook_path_absolute_does_create_there_and_record_it(tmp_path: Path):
    abs_runbook = tmp_path / "elsewhere" / "rb.md"

    scaffold(str(tmp_path), _result(runbook=RunbookChoice(path=str(abs_runbook))))

    assert "# Optimization Runbook" in abs_runbook.read_text(encoding="utf-8")
    assert _read_config(tmp_path)["runbook"] == str(abs_runbook)


def test_scaffold_when_absolute_runbook_exists_does_report_exists_and_preserve(tmp_path: Path):
    abs_runbook = tmp_path / "existing-rb.md"
    existing = "# My Absolute Runbook\n"
    abs_runbook.write_text(existing, encoding="utf-8")

    result = scaffold(str(tmp_path), _result(runbook=RunbookChoice(path=str(abs_runbook))))

    assert result.runbook == ScaffoldArtifact(path=str(abs_runbook), status="exists")
    assert abs_runbook.read_text(encoding="utf-8") == existing


def test_scaffold_when_runbook_declined_does_create_no_file_and_omit_key(tmp_path: Path):
    scaffold(str(tmp_path), _result(runbook=False))

    assert not (tmp_path / "gymrat-runbook.md").exists()
    assert "runbook" not in _read_config(tmp_path)


# ---------------------------------------------------------------------------
# skill install
# ---------------------------------------------------------------------------


def test_scaffold_when_skill_requested_and_absent_does_copy_bundled_skill(tmp_path: Path):
    scaffold(str(tmp_path), _result(install_skill=True))

    skill_path = tmp_path / ".claude" / "skills" / "gymrat" / "SKILL.md"
    assert skill_path.exists()
    assert "# Driving a gymrat optimization session" in skill_path.read_text(encoding="utf-8")


def test_scaffold_when_skill_already_exists_does_leave_it_untouched(tmp_path: Path):
    skill_dir = tmp_path / ".claude" / "skills" / "gymrat"
    skill_dir.mkdir(parents=True)
    existing = "# Custom Skill\n"
    (skill_dir / "SKILL.md").write_text(existing, encoding="utf-8")

    scaffold(str(tmp_path), _result(install_skill=True))

    assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == existing


def test_scaffold_when_skill_declined_does_not_create_skill_file(tmp_path: Path):
    scaffold(str(tmp_path), _result(install_skill=False))

    assert not (tmp_path / ".claude" / "skills" / "gymrat" / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# config write format
# ---------------------------------------------------------------------------


def test_scaffold_writes_pretty_printed_json_with_trailing_newline(tmp_path: Path):
    scaffold(str(tmp_path), _result())

    raw = (tmp_path / "gymrat.json").read_text(encoding="utf-8")
    assert raw == json.dumps({"bench": "npm run bench"}, indent=2) + "\n"


# ---------------------------------------------------------------------------
# write ordering (config last)
# ---------------------------------------------------------------------------


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
        scaffold(str(tmp_path), _result(install_skill=True))

    assert not (tmp_path / "gymrat.json").exists()


def test_scaffold_when_skill_read_fails_does_not_leave_runbook_stub_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _break_bundled_skill(monkeypatch)

    with pytest.raises(GymratError):
        scaffold(
            str(tmp_path),
            _result(install_skill=True, runbook=RunbookChoice(path="gymrat-runbook.md")),
        )

    assert not (tmp_path / "gymrat-runbook.md").exists()


# ---------------------------------------------------------------------------
# exclusive create (no silent overwrite)
# ---------------------------------------------------------------------------


def test_scaffold_when_config_already_exists_does_raise_instead_of_overwriting(tmp_path: Path):
    (tmp_path / "gymrat.json").write_text('{"bench":"old"}\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        scaffold(str(tmp_path), _result())


# ---------------------------------------------------------------------------
# returned artifact statuses
# ---------------------------------------------------------------------------


def test_scaffold_when_all_artifacts_created_does_return_created_statuses(tmp_path: Path):
    result = scaffold(
        str(tmp_path),
        _result(runbook=RunbookChoice(path="gymrat-runbook.md"), install_skill=True),
    )

    assert result.config == ScaffoldArtifact(path="gymrat.json", status="created")
    assert result.runbook == ScaffoldArtifact(path="gymrat-runbook.md", status="created")
    assert result.skill == ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="created")


def test_scaffold_when_runbook_declined_does_report_declined_status(tmp_path: Path):
    result = scaffold(str(tmp_path), _result(runbook=False, install_skill=True))

    assert result.runbook.status == "declined"


def test_scaffold_when_runbook_exists_does_report_exists_status(tmp_path: Path):
    (tmp_path / "my-runbook.md").write_text("# Existing\n", encoding="utf-8")

    result = scaffold(str(tmp_path), _result(runbook=RunbookChoice(path="my-runbook.md")))

    assert result.runbook == ScaffoldArtifact(path="my-runbook.md", status="exists")


def test_scaffold_when_skill_declined_does_report_declined_status(tmp_path: Path):
    result = scaffold(str(tmp_path), _result(install_skill=False))

    assert result.skill.status == "declined"


def test_scaffold_when_skill_exists_does_report_exists_status(tmp_path: Path):
    skill_dir = tmp_path / ".claude" / "skills" / "gymrat"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Custom\n", encoding="utf-8")

    result = scaffold(str(tmp_path), _result(install_skill=True))

    assert result.skill == ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="exists")
