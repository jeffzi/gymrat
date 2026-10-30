"""Tests for ``gymrat doctor``'s report model, section builders, assembly and renderers.

The report model (``create_doctor_report`` count aggregation) and the pure
environment, config, and workflow section builders run with no mocks — every
input is a plain dataclass, so a builder's status/detail/hint output is
asserted directly. The bench section uses the real adapter registry and
patches ``shutil.which`` to control executable availability.

The report assembly (``build_doctor_report``) coordinates the section builders,
config inspection, and git environment probe to produce a ``DoctorReport``;
those tests run it against a real directory and verify that config problems
and adapter flags reach the bench section, and that ``detect_git_environment``
maps git-missing, outside-repo, and unresolvable root to distinct
``GitEnvironment`` outcomes without raising.

The text renderer is pinned by one golden per named scenario (glyphs, hint
and multi-line indentation, caveat note, summary counts); the JSON renderer
by the failure flag it adds.
"""

import json
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import create_autospec

import pytest
from syrupy.assertion import SnapshotAssertion

from gymrat.config import BenchlessConfig, CliFlags, ConfigInspection, StopConfig
from gymrat.doctor import (
    Check,
    CheckSection,
    DoctorReport,
    GitEnvironment,
    build_bench_section,
    build_config_section,
    build_doctor_report,
    build_environment_section,
    build_workflow_section,
    create_doctor_report,
    detect_git_environment,
    render_doctor_json,
    render_doctor_report,
)
from gymrat.scaffold import DEFAULT_RUNBOOK_PATH, SKILL_RELATIVE_PATH
from tests._config import benchless_config as _config
from tests._doctor_fixtures import doctor_report, environment_info
from tests._git import refuse_git_lookups
from tests.adapters._inputs import unknown_adapter_error
from tests.config._toml import write_raw

_DEFAULT_CONFIG = _config()


def _inspection(
    *,
    config_path: str = "/project/gymrat.json",
    problems: list[str] | None = None,
    config: BenchlessConfig | None = _DEFAULT_CONFIG,
) -> ConfigInspection:
    return ConfigInspection(config_path=config_path, problems=problems or [], config=config)


def _find(section: CheckSection, name: str) -> Check:
    match = next((check for check in section.checks if check.name == name), None)
    assert match is not None, f"no check named {name!r}"
    return match


# ---------------------------------------------------------------------------
# create_doctor_report — count aggregation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sections", "counts", "has_failures"),
    [
        pytest.param(
            [
                CheckSection(title="Environment", checks=[Check(name="a", status="ok", detail="")]),
                CheckSection(title="Config", checks=[Check(name="b", status="ok", detail="")]),
            ],
            (2, 0, 0),
            False,
            id="all-ok",
        ),
        pytest.param(
            [
                CheckSection(
                    title="Mixed",
                    checks=[
                        Check(name="a", status="ok", detail=""),
                        Check(name="b", status="warn", detail=""),
                        Check(name="c", status="fail", detail=""),
                        Check(name="d", status="fail", detail=""),
                    ],
                )
            ],
            (1, 1, 2),
            True,
            id="mixed-statuses",
        ),
    ],
)
def test_create_doctor_report_when_statuses_vary_does_aggregate_the_status_summary(
    sections: list[CheckSection], counts: tuple[int, int, int], has_failures: bool
):
    report = create_doctor_report(environment_info(), sections)

    assert (report.ok_count, report.warn_count, report.fail_count) == counts
    assert report.has_failures is has_failures


# ---------------------------------------------------------------------------
# build_environment_section
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("git_available", "inside_git_repo", "checks"),
    [
        pytest.param(True, True, [("git", "ok"), ("git repository", "ok")], id="inside-repo"),
        pytest.param(True, False, [("git", "ok"), ("git repository", "warn")], id="outside-repo"),
        pytest.param(False, False, [("git", "fail"), ("git repository", "warn")], id="git-missing"),
    ],
)
def test_build_environment_section_when_git_state_varies_does_set_each_check_status(
    git_available: bool, inside_git_repo: bool, checks: list[tuple[str, str]]
):
    section = build_environment_section(
        git_available=git_available, inside_git_repo=inside_git_repo
    )

    assert [(check.name, check.status) for check in section.checks] == checks


def test_build_environment_section_when_git_error_given_does_warn_naming_the_error():
    section = build_environment_section(
        git_available=True, inside_git_repo=True, git_error="permission denied"
    )

    root = _find(section, "git repository root")
    assert root.status == "warn"
    assert "permission denied" in root.detail


# ---------------------------------------------------------------------------
# build_config_section
# ---------------------------------------------------------------------------


def test_build_config_section_when_clean_does_produce_single_ok_naming_the_path():
    section = build_config_section(_inspection(config_path="/my/project/gymrat.json", problems=[]))

    assert [(check.name, check.status) for check in section.checks] == [("config", "ok")]
    assert "/my/project/gymrat.json" in section.checks[0].detail


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


def test_build_workflow_section_when_problems_present_does_return_single_ok_skip_check():
    section = build_workflow_section(
        _config(), config_has_problems=True, skill_file_exists=True, config_file_exists=False
    )

    assert [(check.name, check.status) for check in section.checks] == [("workflow", "ok")]


def test_build_workflow_section_when_config_unset_does_warn_for_each_missing_piece():
    section = build_workflow_section(
        _config(),
        config_has_problems=False,
        skill_file_exists=False,
        config_file_exists=False,
    )

    assert [(check.name, check.status) for check in section.checks] == [
        ("skill file", "warn"),
        ("checks", "warn"),
        ("stop", "warn"),
        ("runbook", "warn"),
        ("samples", "ok"),
    ]


@pytest.mark.parametrize(
    ("config", "name", "inserted"),
    [
        pytest.param(_config(checks="npm test"), "checks", "npm test", id="checks-set"),
        pytest.param(
            _config(stop=StopConfig(target_value=1.5)),
            "stop",
            "target_value: 1.5",
            id="stop-target-only",
        ),
        pytest.param(
            _config(stop=StopConfig(max_iterations=20)),
            "stop",
            "max_iterations: 20",
            id="stop-max-only",
        ),
        pytest.param(
            _config(stop=StopConfig(target_value=1.5, max_iterations=20)),
            "stop",
            "target_value: 1.5, max_iterations: 20",
            id="stop-both",
        ),
        pytest.param(_config(runbook="./RUNBOOK.md"), "runbook", "./RUNBOOK.md", id="runbook-set"),
    ],
)
def test_build_workflow_section_when_field_configured_does_report_its_check(
    config: BenchlessConfig, name: str, inserted: str
):
    section = build_workflow_section(
        config,
        config_has_problems=False,
        skill_file_exists=True,
        config_file_exists=False,
    )

    check = _find(section, name)
    assert (check.status, inserted in check.detail) == ("ok", True)


@pytest.mark.parametrize(
    ("samples", "numbers"),
    [
        pytest.param(2, ["2", "2"], id="samples-even-one-target-count"),
        pytest.param(8, ["8", "2", "4", "8"], id="samples-even-up-to-eight-targets"),
        pytest.param(18, ["18", "2", "3", "6"], id="samples-even-no-further-than-eight-targets"),
    ],
)
def test_build_workflow_section_when_samples_even_does_list_the_target_counts_it_balances(
    samples: int, numbers: list[str]
):
    section = build_workflow_section(
        _config(samples=samples),
        config_has_problems=False,
        skill_file_exists=True,
        config_file_exists=False,
    )

    check = _find(section, "samples")
    assert (check.status, re.findall(r"\d+", check.detail)) == ("ok", numbers)


def test_build_workflow_section_when_stop_has_no_keys_does_warn():
    section = build_workflow_section(
        _config(stop=StopConfig()),
        config_has_problems=False,
        skill_file_exists=True,
        config_file_exists=False,
    )

    assert _find(section, "stop").status == "warn"


def test_build_workflow_section_when_samples_odd_does_fail_naming_the_nearest_balanced_counts():
    section = build_workflow_section(
        _config(samples=11),
        config_has_problems=False,
        skill_file_exists=True,
        config_file_exists=False,
    )

    samples = _find(section, "samples")
    assert (samples.status, "11" in samples.detail, "10 or 12" in (samples.hint or "")) == (
        "fail",
        True,
        True,
    )


# ---------------------------------------------------------------------------
# whole section, per path
# ---------------------------------------------------------------------------

_UNKNOWN_ADAPTER_ERROR = unknown_adapter_error("banana")


@pytest.mark.parametrize(
    ("config_problems", "checks"),
    [
        pytest.param(True, [("bench", "ok")], id="config-problems"),
        pytest.param(False, [("adapter", "ok"), ("bench", "fail")], id="bench-unresolved"),
    ],
)
def test_build_bench_section_when_bench_unresolved_does_skip_or_fail_the_bench_check(
    config_problems: bool, checks: list[tuple[str, str]]
):
    section = build_bench_section(
        bench=None, adapter="metric-lines", config_problems=config_problems, base_dir="."
    )

    assert [(check.name, check.status) for check in section.checks] == checks


@pytest.mark.parametrize(
    ("bench", "adapter", "on_path", "expected"),
    [
        pytest.param(
            "node bench.js", "banana", None, [("adapter", "fail", "")], id="adapter-unknown"
        ),
        pytest.param(
            "npx tsx bench.ts",
            "metric-lines",
            "/usr/bin/npx",
            [
                ("adapter", "ok", ""),
                ("bench", "ok", "npx tsx bench.ts"),
                ("executable", "ok", "npx"),
            ],
            id="executable-found",
        ),
        pytest.param(
            "node bench.js",
            "metric-lines",
            None,
            [
                ("adapter", "ok", ""),
                ("bench", "ok", "node bench.js"),
                ("executable", "warn", "node"),
            ],
            id="executable-missing",
        ),
        pytest.param(
            "VAR=1 make test",
            "metric-lines",
            "/usr/bin/x",
            [
                ("adapter", "ok", ""),
                ("bench", "ok", "VAR=1 make test"),
                ("executable", "ok", "make"),
            ],
            id="env-var-prefix",
        ),
        pytest.param(
            "cd src && make bench",
            "metric-lines",
            None,
            [("adapter", "ok", ""), ("bench", "ok", "cd src && make bench")],
            id="shell-operator-cd",
        ),
        pytest.param(
            "~/bin/bench.sh",
            "metric-lines",
            None,
            [("adapter", "ok", ""), ("bench", "ok", "~/bin/bench.sh")],
            id="home-shorthand",
        ),
        pytest.param(
            "$HOME/bin/bench.sh --fast",
            "metric-lines",
            None,
            [("adapter", "ok", ""), ("bench", "ok", "$HOME/bin/bench.sh --fast")],
            id="variable-expansion",
        ),
        pytest.param(
            "`pwd`/bench.sh",
            "metric-lines",
            None,
            [("adapter", "ok", ""), ("bench", "ok", "`pwd`/bench.sh")],
            id="command-substitution",
        ),
    ],
)
def test_build_bench_section_when_inputs_vary_does_produce_the_matching_checks(
    bench: str,
    adapter: str,
    on_path: str | None,
    expected: list[tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
):
    def which(_cmd: str) -> str | None:
        return on_path

    monkeypatch.setattr("shutil.which", create_autospec(shutil.which, side_effect=which))

    section = build_bench_section(bench=bench, adapter=adapter, base_dir=".")

    assert [(check.name, check.status) for check in section.checks] == [
        (name, status) for name, status, _ in expected
    ]
    assert all(
        part in check.detail for check, (_, _, part) in zip(section.checks, expected, strict=True)
    )


# ---------------------------------------------------------------------------
# detect_git_environment — three distinct non-raising outcomes
# ---------------------------------------------------------------------------


def test_detect_git_environment_when_git_missing_does_report_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("PATH", str(tmp_path))

    result = detect_git_environment(str(tmp_path))

    assert result == GitEnvironment(git_available=False, inside_git_repo=False)


def test_detect_git_environment_when_outside_repo_does_report_not_in_repo(tmp_path: Path):
    result = detect_git_environment(str(tmp_path))

    assert result == GitEnvironment(git_available=True, inside_git_repo=False)


def test_detect_git_environment_when_root_unresolvable_does_report_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    refuse_git_lookups(monkeypatch)

    result = detect_git_environment(str(tmp_path))

    assert (result.git_available, result.inside_git_repo, result.repo_root_dir) == (
        True,
        True,
        None,
    )
    assert str(tmp_path) in (result.git_error or "")


# ---------------------------------------------------------------------------
# config inspection — real gymrat.toml
# ---------------------------------------------------------------------------


def _check(report: DoctorReport, title: str, name: str) -> Check:
    section = next((section for section in report.sections if section.title == title), None)
    assert section is not None, f"no section named {title!r}"
    return _find(section, name)


@pytest.mark.parametrize(
    ("toml", "bench_checks"),
    [
        pytest.param("samples = 0\n", [("bench", "ok")], id="invalid-value"),
        pytest.param(None, [("adapter", "ok"), ("bench", "fail")], id="no-file"),
    ],
)
def test_build_doctor_report_when_config_file_invalid_or_absent_does_set_the_bench_checks(
    tmp_path: Path, toml: str | None, bench_checks: list[tuple[str, str]]
):
    if toml is not None:
        write_raw(tmp_path, toml)

    report = build_doctor_report(CliFlags(), str(tmp_path))

    bench = next(section for section in report.sections if section.title == "Bench")
    assert [(check.name, check.status) for check in bench.checks] == bench_checks


def test_build_doctor_report_when_adapter_flag_given_does_check_it_in_the_bench_section(
    tmp_path: Path,
):
    report = build_doctor_report(CliFlags(adapter="banana"), str(tmp_path))

    assert _check(report, "Bench", "adapter") == Check(
        name="adapter",
        status="fail",
        detail=str(_UNKNOWN_ADAPTER_ERROR),
        hint=_UNKNOWN_ADAPTER_ERROR.hint,
    )


def _skill_directory(path: Path) -> None:
    """Put a directory where the skill file belongs."""
    path.mkdir(parents=True, exist_ok=True)


def _skill_file(path: Path) -> None:
    """Put a regular skill file at its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("skill", encoding="utf-8")


@pytest.mark.parametrize(
    ("make_entry", "status"),
    [
        pytest.param(_skill_directory, "warn", id="directory"),
        pytest.param(_skill_file, "ok", id="file"),
    ],
)
def test_build_doctor_report_when_skill_path_checked_does_report_installed_only_for_a_file(
    repo: str,
    make_entry: Callable[[Path], None],
    status: str,
):
    make_entry(Path(repo) / SKILL_RELATIVE_PATH)

    report = build_doctor_report(CliFlags(), repo)

    assert _check(report, "Workflow", "skill file").status == status


def test_build_doctor_report_when_runbook_unset_does_hint_adding_the_key_to_the_config_file(
    tmp_path: Path,
):
    write_raw(tmp_path, "samples = 5\n")

    report = build_doctor_report(CliFlags(), str(tmp_path))

    assert DEFAULT_RUNBOOK_PATH in (_check(report, "Workflow", "runbook").hint or "")


# ---------------------------------------------------------------------------
# bench executable — path-shaped token, run from a subdirectory
# ---------------------------------------------------------------------------

_posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="the executable bit is a POSIX file mode"
)


@pytest.fixture
def repo_subdirectory(repo: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A subdirectory of a scratch repository, made the process working directory."""
    subdirectory = Path(repo) / "packages" / "app"
    subdirectory.mkdir(parents=True)
    monkeypatch.chdir(subdirectory)
    return subdirectory


@pytest.mark.parametrize(
    ("bench", "script", "mode", "status", "path"),
    [
        pytest.param(
            '"my scripts/bench.sh" --fast',
            "my scripts/bench.sh",
            0o755,
            "ok",
            "my scripts/bench.sh",
            id="quoted-path-with-spaces",
        ),
        pytest.param("./scripts/bench.sh", None, None, "warn", "./scripts/bench.sh", id="missing"),
        pytest.param(
            "./scripts/bench.sh",
            "scripts/bench.sh",
            0o644,
            "warn",
            "./scripts/bench.sh",
            id="not-executable",
            marks=_posix_only,
        ),
    ],
)
def test_build_doctor_report_when_bench_is_a_path_does_resolve_it_against_the_repository_root(
    repo_subdirectory: Path,
    *,
    bench: str,
    script: str | None,
    mode: int | None,
    status: str,
    path: str,
):
    root = repo_subdirectory.parent.parent
    write_raw(root, f"bench = '{bench}'\n")
    if script is not None and mode is not None:
        script_path = root / script
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("#!/bin/sh\n", encoding="utf-8")
        script_path.chmod(mode)

    report = build_doctor_report(CliFlags(), str(repo_subdirectory))

    check = _check(report, "Bench", "executable")
    assert check.status == status
    assert path in check.detail


# ---------------------------------------------------------------------------
# text rendering — one golden per named scenario
# ---------------------------------------------------------------------------


def _mixed_status_report() -> DoctorReport:
    """A report with a passing, a warning and a failing check, each kind with a hint where it applies."""
    return doctor_report([
        CheckSection(
            title="Environment",
            checks=[
                Check("git", "ok", "git 2.45.0"),
                Check("skill", "warn", "skill not installed", hint="run gymrat init"),
            ],
        ),
        CheckSection(
            title="Bench",
            checks=[Check("bench", "fail", "bench not set", hint="set bench in gymrat.toml")],
        ),
    ])


def _multiline_detail_report() -> DoctorReport:
    """A report with one check whose detail spans three lines."""
    return doctor_report([
        CheckSection(
            title="Bench",
            checks=[Check("bench", "ok", "line one\nline two\nline three")],
        )
    ])


def _workflow_skipped_report() -> DoctorReport:
    """A report whose workflow checks were skipped behind config errors."""
    return doctor_report([
        CheckSection(
            title="Workflow",
            checks=[Check("workflow", "ok", "Skipped — fix config errors first")],
        )
    ])


def _plural_counts_report() -> DoctorReport:
    """A report with one passing, two warning and three failing checks."""
    return doctor_report([
        CheckSection(
            title="All",
            checks=[
                Check("a", "ok", ""),
                Check("b", "warn", ""),
                Check("c", "warn", ""),
                Check("d", "fail", ""),
                Check("e", "fail", ""),
                Check("f", "fail", ""),
            ],
        )
    ])


@pytest.mark.parametrize(
    "make_report",
    [
        pytest.param(_mixed_status_report, id="ok-warn-fail"),
        pytest.param(_multiline_detail_report, id="multiline-detail"),
        pytest.param(_workflow_skipped_report, id="workflow-skipped"),
        pytest.param(_plural_counts_report, id="plural-counts"),
    ],
)
def test_render_doctor_report_when_scenario_rendered_does_match_its_golden(
    make_report: Callable[[], DoctorReport], snapshot: SnapshotAssertion
):
    report = make_report()

    output = render_doctor_report(report, color=False)

    assert output.split("\n") == snapshot


# ---------------------------------------------------------------------------
# JSON rendering
# ---------------------------------------------------------------------------


def test_render_doctor_json_when_report_has_a_failure_does_flag_it():
    report = _mixed_status_report()

    output = render_doctor_json(report)

    document = json.loads(output)
    assert (document["fail_count"], document["has_failures"]) == (1, True)
