"""Tests for the doctor report model and the pure section builders.

These exercise ``create_doctor_report`` count aggregation and the environment,
config, and workflow section builders with no mocks — every input is a plain
dataclass, so a builder's status/detail/hint output is asserted directly.
"""

import pytest

from gymrat.config import ConfigInspection, StopConfig
from gymrat.doctor import (
    Check,
    CheckSection,
    build_config_section,
    build_environment_section,
    build_workflow_section,
    create_doctor_report,
)
from tests._config import benchless_config as _config
from tests.doctor._fixtures import environment_info as _env


def _inspection(**overrides: object) -> ConfigInspection:
    base: dict[str, object] = {
        "config_path": "/project/gymrat.json",
        "problems": [],
        "config": _config(),
    }
    base.update(overrides)
    return ConfigInspection(**base)  # pyrefly: ignore


def _find(section: CheckSection, name: str) -> Check:
    match = next((check for check in section.checks if check.name == name), None)
    assert match is not None, f"no check named {name!r}"
    return match


# ---------------------------------------------------------------------------
# create_doctor_report — count aggregation
# ---------------------------------------------------------------------------


def test_create_doctor_report_when_all_ok_does_report_zero_warnings_and_no_failures():
    sections = [
        CheckSection(title="Environment", checks=[Check(name="a", status="ok", detail="")]),
        CheckSection(title="Config", checks=[Check(name="b", status="ok", detail="")]),
    ]

    report = create_doctor_report(_env(), sections)

    assert report.ok_count == 2
    assert report.warn_count == 0
    assert report.fail_count == 0
    assert report.has_failures is False


def test_create_doctor_report_when_mixed_statuses_does_count_each_and_flag_failures():
    sections = [
        CheckSection(
            title="Mixed",
            checks=[
                Check(name="a", status="ok", detail=""),
                Check(name="b", status="warn", detail=""),
                Check(name="c", status="fail", detail=""),
                Check(name="d", status="fail", detail=""),
            ],
        )
    ]

    report = create_doctor_report(_env(), sections)

    assert report.ok_count == 1
    assert report.warn_count == 1
    assert report.fail_count == 2
    assert report.has_failures is True


def test_create_doctor_report_when_counts_span_sections_does_aggregate_across_all():
    sections = [
        CheckSection(title="A", checks=[Check(name="a", status="ok", detail="")]),
        CheckSection(title="B", checks=[Check(name="b", status="warn", detail="")]),
        CheckSection(title="C", checks=[Check(name="c", status="fail", detail="")]),
    ]

    report = create_doctor_report(_env(), sections)

    assert (report.ok_count, report.warn_count, report.fail_count) == (1, 1, 1)
    assert report.has_failures is True


def test_create_doctor_report_when_no_sections_does_report_all_zeros():
    report = create_doctor_report(_env(), [])

    assert (report.ok_count, report.warn_count, report.fail_count) == (0, 0, 0)
    assert report.has_failures is False


def test_create_doctor_report_when_built_does_preserve_environment_and_sections():
    env = _env()
    sections = [CheckSection(title="Test", checks=[Check(name="x", status="ok", detail="")])]

    report = create_doctor_report(env, sections)

    assert report.environment == env
    assert report.sections == sections


# ---------------------------------------------------------------------------
# build_environment_section
# ---------------------------------------------------------------------------


def test_build_environment_section_has_title_environment():
    section = build_environment_section(git_available=True, inside_git_repo=True)

    assert section.title == "Environment"


def test_build_environment_section_when_git_available_does_produce_ok_git_check():
    section = build_environment_section(git_available=True, inside_git_repo=True)

    assert _find(section, "git") == Check(
        name="git", status="ok", detail="git is available on PATH"
    )


def test_build_environment_section_when_git_missing_does_fail_with_install_hint():
    section = build_environment_section(git_available=False, inside_git_repo=False)

    assert _find(section, "git") == Check(
        name="git",
        status="fail",
        detail="git is not available on PATH",
        hint="Install git: https://git-scm.com/downloads",
    )


def test_build_environment_section_when_inside_repo_does_produce_ok_repository_check():
    section = build_environment_section(git_available=True, inside_git_repo=True)

    assert _find(section, "git repository") == Check(
        name="git repository",
        status="ok",
        detail="current directory is inside a git repository",
    )


def test_build_environment_section_when_outside_repo_does_warn_with_compare_hint():
    section = build_environment_section(git_available=True, inside_git_repo=False)

    assert _find(section, "git repository") == Check(
        name="git repository",
        status="warn",
        detail="current directory is not inside a git repository",
        hint="The compare command resolves refs against a git repository",
    )


def test_build_environment_section_when_git_error_given_does_warn_naming_the_error():
    section = build_environment_section(
        git_available=True, inside_git_repo=True, git_error="permission denied"
    )

    root = _find(section, "git repository root")
    assert root.status == "warn"
    assert "permission denied" in root.detail
    assert (
        root.hint == "Falling back to the current directory; commands may operate on the wrong path"
    )


def test_build_environment_section_when_no_git_error_does_omit_repository_root_check():
    section = build_environment_section(git_available=True, inside_git_repo=True)

    names = [check.name for check in section.checks]
    assert "git repository root" not in names


# ---------------------------------------------------------------------------
# build_config_section
# ---------------------------------------------------------------------------


def test_build_config_section_has_title_configuration():
    section = build_config_section(_inspection())

    assert section.title == "Configuration"


def test_build_config_section_when_clean_does_produce_single_ok_naming_the_path():
    section = build_config_section(_inspection(config_path="/my/project/gymrat.json", problems=[]))

    assert section.checks == [
        Check(name="config", status="ok", detail="Config file loaded: /my/project/gymrat.json")
    ]


def test_build_config_section_when_no_config_file_does_produce_single_ok_defaults_only():
    section = build_config_section(_inspection(config_path=None, config=None, problems=[]))

    assert section.checks == [
        Check(
            name="config",
            status="ok",
            detail="No config file found; operating with defaults only",
        )
    ]


def test_build_config_section_when_problems_present_does_produce_one_fail_per_problem_verbatim():
    problems = [
        'Invalid value for "samples": expected a positive integer, got "abc"',
        'Invalid value for "adapter": expected a string, got 42',
    ]
    section = build_config_section(_inspection(problems=problems, config=None))

    fails = [check for check in section.checks if check.status == "fail"]
    assert [check.detail for check in fails] == problems


# ---------------------------------------------------------------------------
# build_workflow_section
# ---------------------------------------------------------------------------


def test_build_workflow_section_has_title_workflow():
    section = build_workflow_section(_config(), config_has_problems=False, skill_file_exists=True)

    assert section.title == "Workflow"


def test_build_workflow_section_when_problems_present_does_return_single_ok_skip_check():
    section = build_workflow_section(_config(), config_has_problems=True, skill_file_exists=True)

    assert section.checks == [
        Check(name="workflow", status="ok", detail="Skipped — fix config errors first")
    ]


def test_build_workflow_section_when_problems_present_does_omit_individual_workflow_checks():
    section = build_workflow_section(_config(), config_has_problems=True, skill_file_exists=False)

    names = {check.name for check in section.checks}
    assert names.isdisjoint({"skill file", "checks", "stop", "runbook"})


def test_build_workflow_section_when_skill_file_present_does_produce_ok_skill_check():
    section = build_workflow_section(_config(), config_has_problems=False, skill_file_exists=True)

    assert _find(section, "skill file") == Check(
        name="skill file", status="ok", detail="Skill file is installed"
    )


def test_build_workflow_section_when_skill_file_missing_does_warn_with_init_only_hint():
    section = build_workflow_section(_config(), config_has_problems=False, skill_file_exists=False)

    assert _find(section, "skill file") == Check(
        name="skill file",
        status="warn",
        detail="No skill file — Claude Code agents won't have gymrat's workflow instructions",
        hint="Run `gymrat init` to scaffold the project.",
    )


def test_build_workflow_section_when_checks_set_does_produce_ok_echoing_value():
    section = build_workflow_section(
        _config(checks="npm test"), config_has_problems=False, skill_file_exists=True
    )

    assert _find(section, "checks") == Check(name="checks", status="ok", detail="checks: npm test")


def test_build_workflow_section_when_checks_unset_does_warn_about_keep_gating():
    section = build_workflow_section(_config(), config_has_problems=False, skill_file_exists=True)

    assert _find(section, "checks") == Check(
        name="checks",
        status="warn",
        detail="checks is not configured",
        hint="Without checks, keep cannot gate commits",
    )


@pytest.mark.parametrize(
    ("stop", "detail"),
    [
        pytest.param(StopConfig(target_value=1.5), "stop: target_value: 1.5", id="target-only"),
        pytest.param(StopConfig(max_iterations=20), "stop: max_iterations: 20", id="max-only"),
        pytest.param(
            StopConfig(target_value=1.5, max_iterations=20),
            "stop: target_value: 1.5, max_iterations: 20",
            id="both",
        ),
    ],
)
def test_build_workflow_section_when_stop_set_does_produce_ok_echoing_its_keys(
    stop: StopConfig, detail: str
):
    section = build_workflow_section(
        _config(stop=stop), config_has_problems=False, skill_file_exists=True
    )

    assert _find(section, "stop") == Check(name="stop", status="ok", detail=detail)


@pytest.mark.parametrize(
    "stop",
    [
        pytest.param(None, id="unset"),
        pytest.param(StopConfig(), id="empty"),
    ],
)
def test_build_workflow_section_when_stop_absent_or_empty_does_warn(stop: StopConfig | None):
    section = build_workflow_section(
        _config(stop=stop), config_has_problems=False, skill_file_exists=True
    )

    assert _find(section, "stop") == Check(
        name="stop",
        status="warn",
        detail="stop is not configured",
        hint="Without stop, a session has no finish line",
    )


def test_build_workflow_section_when_runbook_set_does_produce_ok_echoing_path():
    section = build_workflow_section(
        _config(runbook="./RUNBOOK.md"), config_has_problems=False, skill_file_exists=True
    )

    assert _find(section, "runbook") == Check(
        name="runbook", status="ok", detail="runbook: ./RUNBOOK.md"
    )


def test_build_workflow_section_when_runbook_unset_does_warn_about_supervise():
    section = build_workflow_section(_config(), config_has_problems=False, skill_file_exists=True)

    assert _find(section, "runbook") == Check(
        name="runbook",
        status="warn",
        detail="runbook is not configured",
        hint=(
            "Run `gymrat init` to create a runbook, or add `runbook` to gymrat.toml. "
            "Without one, supervise has no instructions to follow."
        ),
    )
