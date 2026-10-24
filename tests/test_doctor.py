"""Tests for ``gymrat doctor``'s report model, section builders, assembly and renderers.

The report model (``create_doctor_report`` count aggregation) and the pure
environment, config, and workflow section builders run with no mocks — every
input is a plain dataclass, so a builder's status/detail/hint output is
asserted directly. The bench section uses the real adapter registry and
patches ``shutil.which`` to control executable availability.

The report assembly (``build_doctor_report``) coordinates the section builders,
config inspection, and git environment probe to produce a ``DoctorReport``;
those tests run it against a real directory and verify that sections are
collected in order, that config problems and adapter flags reach the bench
section, and that ``detect_git_environment`` maps git-missing, outside-repo,
and unresolvable root to distinct ``GitEnvironment`` outcomes without raising.

The text renderer is pinned by one golden per named scenario (glyphs, hint
and multi-line indentation, caveat note, summary counts); the JSON renderer
by its exact document.
"""

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
from tests._ansi import sgr_codes, strip_ansi
from tests._config import benchless_config as _config
from tests._doctor_fixtures import doctor_report, environment_info
from tests._git import refuse_git_lookups
from tests.adapters._inputs import VALID_ADAPTERS_HINT, unknown_adapter_message
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
        pytest.param(
            [
                CheckSection(title="A", checks=[Check(name="a", status="ok", detail="")]),
                CheckSection(title="B", checks=[Check(name="b", status="warn", detail="")]),
                CheckSection(title="C", checks=[Check(name="c", status="fail", detail="")]),
            ],
            (1, 1, 1),
            True,
            id="across-sections",
        ),
        pytest.param([], (0, 0, 0), False, id="no-sections"),
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


_GIT_OK = Check(name="git", status="ok", detail="git is available on PATH")
_INSIDE_REPO = Check(
    name="git repository", status="ok", detail="current directory is inside a git repository"
)
_OUTSIDE_REPO = Check(
    name="git repository",
    status="warn",
    detail="current directory is not inside a git repository",
    hint="The compare command resolves refs against a git repository",
)


@pytest.mark.parametrize(
    ("git_available", "inside_git_repo", "checks"),
    [
        pytest.param(True, True, [_GIT_OK, _INSIDE_REPO], id="inside-repo"),
        pytest.param(True, False, [_GIT_OK, _OUTSIDE_REPO], id="outside-repo"),
        pytest.param(
            False,
            False,
            [
                Check(
                    name="git",
                    status="fail",
                    detail="git is not available on PATH",
                    hint="Install git: https://git-scm.com/downloads",
                ),
                _OUTSIDE_REPO,
            ],
            id="git-missing",
        ),
    ],
)
def test_build_environment_section_when_git_state_varies_does_produce_the_exact_section(
    git_available: bool, inside_git_repo: bool, checks: list[Check]
):
    section = build_environment_section(
        git_available=git_available, inside_git_repo=inside_git_repo
    )

    assert section == CheckSection(title="Environment", checks=checks)


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


# ---------------------------------------------------------------------------
# build_config_section
# ---------------------------------------------------------------------------


def test_build_config_section_when_clean_does_produce_single_ok_naming_the_path():
    section = build_config_section(_inspection(config_path="/my/project/gymrat.json", problems=[]))

    assert section.title == "Configuration"
    assert section.checks == [
        Check(name="config", status="ok", detail="Config file loaded: /my/project/gymrat.json")
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


def test_build_workflow_section_when_problems_present_does_return_single_ok_skip_check():
    section = build_workflow_section(
        _config(), config_has_problems=True, skill_file_exists=True, config_file_exists=False
    )

    assert section.title == "Workflow"
    assert section.checks == [
        Check(name="workflow", status="ok", detail="Skipped — fix config errors first")
    ]


_STOP_UNSET = Check(
    name="stop",
    status="warn",
    detail="stop is not configured",
    hint="Without stop, a session has no finish line",
)

_UNSET_WORKFLOW_CHECKS = [
    Check(
        name="checks",
        status="warn",
        detail="checks is not configured",
        hint="Without checks, keep cannot gate commits",
    ),
    _STOP_UNSET,
    Check(
        name="runbook",
        status="warn",
        detail="runbook is not configured",
        hint=(
            "Run `gymrat init` to create a runbook, or add `runbook` to gymrat.toml. "
            "Without one, supervise has no instructions to follow."
        ),
    ),
    Check(name="samples", status="ok", detail="samples: 12 — balances 2, 3, 4 or 6 targets"),
]


@pytest.mark.parametrize(
    ("skill_file_exists", "skill_check"),
    [
        pytest.param(
            True,
            Check(name="skill file", status="ok", detail="Skill file is installed"),
            id="skill-file-present",
        ),
        pytest.param(
            False,
            Check(
                name="skill file",
                status="warn",
                detail="No skill file — Claude Code agents won't have gymrat's workflow instructions",
                hint="Run `gymrat init` to scaffold the project.",
            ),
            id="skill-file-missing",
        ),
    ],
)
def test_build_workflow_section_when_config_unset_does_produce_the_exact_section(
    skill_file_exists: bool, skill_check: Check
):
    section = build_workflow_section(
        _config(),
        config_has_problems=False,
        skill_file_exists=skill_file_exists,
        config_file_exists=False,
    )

    assert section == CheckSection(title="Workflow", checks=[skill_check, *_UNSET_WORKFLOW_CHECKS])


@pytest.mark.parametrize(
    ("config", "name", "expected"),
    [
        pytest.param(
            _config(checks="npm test"),
            "checks",
            Check(name="checks", status="ok", detail="checks: npm test"),
            id="checks-set",
        ),
        pytest.param(
            _config(stop=StopConfig(target_value=1.5)),
            "stop",
            Check(name="stop", status="ok", detail="stop: target_value: 1.5"),
            id="stop-target-only",
        ),
        pytest.param(
            _config(stop=StopConfig(max_iterations=20)),
            "stop",
            Check(name="stop", status="ok", detail="stop: max_iterations: 20"),
            id="stop-max-only",
        ),
        pytest.param(
            _config(stop=StopConfig(target_value=1.5, max_iterations=20)),
            "stop",
            Check(name="stop", status="ok", detail="stop: target_value: 1.5, max_iterations: 20"),
            id="stop-both",
        ),
        pytest.param(_config(stop=StopConfig()), "stop", _STOP_UNSET, id="stop-empty"),
        pytest.param(
            _config(runbook="./RUNBOOK.md"),
            "runbook",
            Check(name="runbook", status="ok", detail="runbook: ./RUNBOOK.md"),
            id="runbook-set",
        ),
        pytest.param(
            _config(samples=10),
            "samples",
            Check(name="samples", status="ok", detail="samples: 10 — balances 2 or 5 targets"),
            id="samples-even-two-targets-counts",
        ),
        pytest.param(
            _config(samples=2),
            "samples",
            Check(name="samples", status="ok", detail="samples: 2 — balances 2 targets"),
            id="samples-even-one-target-count",
        ),
        pytest.param(
            _config(samples=8),
            "samples",
            Check(name="samples", status="ok", detail="samples: 8 — balances 2, 4 or 8 targets"),
            id="samples-even-up-to-eight-targets",
        ),
        pytest.param(
            _config(samples=18),
            "samples",
            Check(name="samples", status="ok", detail="samples: 18 — balances 2, 3 or 6 targets"),
            id="samples-even-no-further-than-eight-targets",
        ),
        pytest.param(
            _config(samples=11),
            "samples",
            Check(
                name="samples",
                status="fail",
                detail="samples: 11 — balances one target only",
                hint=(
                    "`iterate` and `compare` refuse an odd count; "
                    "`measure` and `probe` still run — use 10 or 12"
                ),
            ),
            id="samples-odd",
        ),
        pytest.param(
            _config(samples=1),
            "samples",
            Check(
                name="samples",
                status="fail",
                detail="samples: 1 — balances one target only",
                hint=(
                    "`iterate` and `compare` refuse an odd count; "
                    "`measure` and `probe` still run — use 2"
                ),
            ),
            id="samples-one",
        ),
    ],
)
def test_build_workflow_section_when_field_configured_does_report_its_check(
    config: BenchlessConfig, name: str, expected: Check
):
    section = build_workflow_section(
        config,
        config_has_problems=False,
        skill_file_exists=True,
        config_file_exists=False,
    )

    assert _find(section, name) == expected


# ---------------------------------------------------------------------------
# whole section, per path
# ---------------------------------------------------------------------------

_ADAPTER_OK = Check(name="adapter", status="ok", detail="adapter: metric-lines")


@pytest.mark.parametrize(
    ("bench", "adapter", "config_problems", "on_path", "expected"),
    [
        pytest.param(
            None,
            "metric-lines",
            True,
            None,
            [Check(name="bench", status="ok", detail="Skipped — fix config errors first")],
            id="config-problems",
        ),
        pytest.param(
            "node bench.js",
            "banana",
            False,
            None,
            [
                Check(
                    name="adapter",
                    status="fail",
                    detail=unknown_adapter_message("banana"),
                    hint=VALID_ADAPTERS_HINT,
                )
            ],
            id="adapter-unknown",
        ),
        pytest.param(
            None,
            "metric-lines",
            False,
            None,
            [
                _ADAPTER_OK,
                Check(
                    name="bench",
                    status="fail",
                    detail="No bench command configured",
                    hint='Set the bench command with --bench or the "bench" config key',
                ),
            ],
            id="bench-unresolved",
        ),
        pytest.param(
            "npx tsx bench.ts",
            "metric-lines",
            False,
            "/usr/bin/npx",
            [
                _ADAPTER_OK,
                Check(name="bench", status="ok", detail="bench: npx tsx bench.ts"),
                Check(name="executable", status="ok", detail="npx is available on PATH"),
            ],
            id="executable-found",
        ),
        pytest.param(
            "node bench.js",
            "metric-lines",
            False,
            None,
            [
                _ADAPTER_OK,
                Check(name="bench", status="ok", detail="bench: node bench.js"),
                Check(name="executable", status="warn", detail="node was not found on PATH"),
            ],
            id="executable-missing",
        ),
        pytest.param(
            '"node" bench.js',
            "metric-lines",
            False,
            "/usr/bin/x",
            [
                _ADAPTER_OK,
                Check(name="bench", status="ok", detail='bench: "node" bench.js'),
                Check(name="executable", status="ok", detail="node is available on PATH"),
            ],
            id="quoted-exe",
        ),
        pytest.param(
            "VAR=1 make test",
            "metric-lines",
            False,
            "/usr/bin/x",
            [
                _ADAPTER_OK,
                Check(name="bench", status="ok", detail="bench: VAR=1 make test"),
                Check(name="executable", status="ok", detail="make is available on PATH"),
            ],
            id="env-var-prefix",
        ),
        pytest.param(
            "VAR=1 FOO=2 node bench.js",
            "metric-lines",
            False,
            "/usr/bin/x",
            [
                _ADAPTER_OK,
                Check(name="bench", status="ok", detail="bench: VAR=1 FOO=2 node bench.js"),
                Check(name="executable", status="ok", detail="node is available on PATH"),
            ],
            id="multiple-env-vars",
        ),
        pytest.param(
            "cd src && make bench",
            "metric-lines",
            False,
            None,
            [_ADAPTER_OK, Check(name="bench", status="ok", detail="bench: cd src && make bench")],
            id="shell-operator-cd",
        ),
        pytest.param(
            "{ make bench; }",
            "metric-lines",
            False,
            None,
            [_ADAPTER_OK, Check(name="bench", status="ok", detail="bench: { make bench; }")],
            id="shell-brace-group",
        ),
        pytest.param(
            "~/bin/bench.sh",
            "metric-lines",
            False,
            None,
            [_ADAPTER_OK, Check(name="bench", status="ok", detail="bench: ~/bin/bench.sh")],
            id="home-shorthand",
        ),
        pytest.param(
            "$HOME/bin/bench.sh --fast",
            "metric-lines",
            False,
            None,
            [
                _ADAPTER_OK,
                Check(name="bench", status="ok", detail="bench: $HOME/bin/bench.sh --fast"),
            ],
            id="variable-expansion",
        ),
        pytest.param(
            "`pwd`/bench.sh",
            "metric-lines",
            False,
            None,
            [_ADAPTER_OK, Check(name="bench", status="ok", detail="bench: `pwd`/bench.sh")],
            id="command-substitution",
        ),
    ],
)
def test_build_bench_section_when_inputs_vary_does_produce_the_matching_checks(  # noqa: PLR0917 -- one parameter per input plus the expected section and fixture
    bench: str | None,
    adapter: str,
    config_problems: bool,
    on_path: str | None,
    expected: list[Check],
    monkeypatch: pytest.MonkeyPatch,
):
    def which(_cmd: str) -> str | None:
        return on_path

    monkeypatch.setattr("shutil.which", create_autospec(shutil.which, side_effect=which))

    section = build_bench_section(
        bench=bench, adapter=adapter, config_problems=config_problems, base_dir="."
    )

    assert section == CheckSection(title="Bench", checks=expected)


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

    assert result == GitEnvironment(
        git_available=True,
        inside_git_repo=True,
        git_error=(
            f"Cannot determine the git repository at {tmp_path}: fatal: detected dubious ownership"
        ),
    )


# ---------------------------------------------------------------------------
# config inspection — real gymrat.toml
# ---------------------------------------------------------------------------


def _check(report: DoctorReport, title: str, name: str) -> Check:
    section = next((section for section in report.sections if section.title == title), None)
    assert section is not None, f"no section named {title!r}"
    return _find(section, name)


@pytest.mark.parametrize(
    ("toml", "expected", "bench_checks"),
    [
        pytest.param(
            "samples = 0\n",
            Check(
                name="config",
                status="fail",
                detail="Invalid config value for samples: expected a number at or above 1, got 0",
            ),
            [Check(name="bench", status="ok", detail="Skipped — fix config errors first")],
            id="invalid-value",
        ),
        pytest.param(
            None,
            Check(
                name="config",
                status="ok",
                detail="No config file found; operating with defaults only",
            ),
            [
                Check(name="adapter", status="ok", detail="adapter: metric-lines"),
                Check(
                    name="bench",
                    status="fail",
                    detail="No bench command configured",
                    hint='Set the bench command with --bench or the "bench" config key',
                ),
            ],
            id="no-file",
        ),
    ],
)
def test_build_doctor_report_when_config_file_invalid_or_absent_does_report_its_effect_per_section(
    tmp_path: Path, toml: str | None, expected: Check, bench_checks: list[Check]
):
    if toml is not None:
        write_raw(tmp_path, toml)

    report = build_doctor_report(CliFlags(), str(tmp_path))

    sections = {section.title: section.checks for section in report.sections}
    assert list(sections) == ["Environment", "Configuration", "Workflow", "Bench"]
    assert sections["Configuration"] == [expected]
    assert sections["Bench"] == bench_checks


def test_build_doctor_report_when_adapter_flag_given_does_check_it_in_the_bench_section(
    tmp_path: Path,
):
    report = build_doctor_report(CliFlags(adapter="banana"), str(tmp_path))

    assert _check(report, "Bench", "adapter") == Check(
        name="adapter",
        status="fail",
        detail=unknown_adapter_message("banana"),
        hint=VALID_ADAPTERS_HINT,
    )


def test_build_doctor_report_when_runbook_unset_does_hint_adding_the_key_to_the_config_file(
    tmp_path: Path,
):
    write_raw(tmp_path, "samples = 5\n")

    report = build_doctor_report(CliFlags(), str(tmp_path))

    assert _check(report, "Workflow", "runbook").hint == (
        'Add `runbook = "gymrat-runbook.md"` to gymrat.toml. '
        "Without one, supervise has no instructions to follow."
    )


@pytest.mark.parametrize(
    ("flag", "env", "detail"),
    [
        pytest.param(2, "10", "samples: 2 — balances 2 targets", id="flag-wins"),
        pytest.param(None, "10", "samples: 10 — balances 2 or 5 targets", id="env-beats-file"),
        pytest.param(None, None, "samples: 11 — balances one target only", id="file-alone"),
    ],
)
def test_build_doctor_report_when_samples_set_at_several_levels_does_check_the_resolved_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: int | None,
    env: str | None,
    detail: str,
):
    write_raw(tmp_path, "samples = 11\n")
    if env is not None:
        monkeypatch.setenv("GYMRAT_SAMPLES", env)

    report = build_doctor_report(CliFlags(samples=flag), str(tmp_path))

    assert _check(report, "Workflow", "samples").detail == detail


def test_build_doctor_report_when_samples_set_nowhere_does_check_the_default_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("GYMRAT_SAMPLES", raising=False)

    report = build_doctor_report(CliFlags(), str(tmp_path))

    assert _check(report, "Workflow", "samples").detail == (
        "samples: 12 — balances 2, 3, 4 or 6 targets"
    )


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
    ("bench", "script", "mode", "expected"),
    [
        pytest.param(
            "./scripts/bench.sh",
            "scripts/bench.sh",
            0o755,
            Check(name="executable", status="ok", detail="./scripts/bench.sh is executable"),
            id="executable-under-root",
        ),
        pytest.param(
            '"my scripts/bench.sh" --fast',
            "my scripts/bench.sh",
            0o755,
            Check(name="executable", status="ok", detail="my scripts/bench.sh is executable"),
            id="quoted-path-with-spaces",
        ),
        pytest.param(
            "./scripts/bench.sh",
            None,
            None,
            Check(name="executable", status="warn", detail="./scripts/bench.sh was not found"),
            id="missing",
        ),
        pytest.param(
            "./scripts/bench.sh",
            "scripts/bench.sh",
            0o644,
            Check(name="executable", status="warn", detail="./scripts/bench.sh is not executable"),
            id="not-executable",
            marks=_posix_only,
        ),
    ],
)
def test_build_doctor_report_when_bench_is_a_path_does_resolve_it_against_the_repository_root(
    repo_subdirectory: Path, bench: str, script: str | None, mode: int | None, expected: Check
):
    root = repo_subdirectory.parent.parent
    write_raw(root, f"bench = '{bench}'\n")
    if script is not None and mode is not None:
        script_path = root / script
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("#!/bin/sh\n", encoding="utf-8")
        script_path.chmod(mode)

    report = build_doctor_report(CliFlags(), str(repo_subdirectory))

    assert _check(report, "Bench", "executable") == expected


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


def _warning_only_report() -> DoctorReport:
    """A report whose only problem is a warning."""
    return doctor_report([
        CheckSection(
            title="Environment",
            checks=[
                Check("git", "ok", "git 2.45.0"),
                Check("skill", "warn", "skill not installed", hint="run gymrat init"),
            ],
        )
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
        pytest.param(_warning_only_report, id="warn-only"),
        pytest.param(_multiline_detail_report, id="multiline-detail"),
        pytest.param(_workflow_skipped_report, id="workflow-skipped"),
        pytest.param(_plural_counts_report, id="plural-counts"),
    ],
)
def test_render_doctor_report_when_statuses_vary_does_render_glyphs_hints_and_summary_counts(
    make_report: Callable[[], DoctorReport], snapshot: SnapshotAssertion
):
    report = make_report()

    output = render_doctor_report(report, color=False)

    assert output.split("\n") == snapshot


def test_render_doctor_report_when_colored_does_style_each_line_by_its_role():
    report = _mixed_status_report()

    output = render_doctor_report(report, color=True)

    assert [(strip_ansi(line), sgr_codes(line)) for line in output.split("\n")] == [
        ("gymrat v0.5.0 · python 3.13.0 · darwin", {"1", "2"}),
        ("", set()),
        ("Environment", {"1"}),
        ("  ✓ git 2.45.0", {"32"}),
        ("  ⚠ skill not installed", {"33"}),
        ("    run gymrat init", {"2"}),
        ("", set()),
        ("Bench", {"1"}),
        ("  ✗ bench not set", {"31"}),
        ("    set bench in gymrat.toml", {"2"}),
        ("", set()),
        (
            (
                "Note: prepare scripts were not run; only the Claude skill file location was"
                " checked (presence ≠ loaded)."
            ),
            {"2"},
        ),
        ("", set()),
        ("1 ok · 1 warning · 1 failure", {"32", "33", "31"}),
    ]


# ---------------------------------------------------------------------------
# JSON rendering
# ---------------------------------------------------------------------------


def test_render_doctor_json_when_report_has_a_warning_does_emit_two_space_indented_document():
    report = _warning_only_report()

    output = render_doctor_json(report)

    assert output.split("\n") == [
        "{",
        '  "schema_version": 1,',
        '  "environment": {',
        '    "gymrat_version": "0.5.0",',
        '    "python_version": "3.13.0",',
        '    "platform": "darwin"',
        "  },",
        '  "sections": [',
        "    {",
        '      "title": "Environment",',
        '      "checks": [',
        "        {",
        '          "name": "git",',
        '          "status": "ok",',
        '          "detail": "git 2.45.0",',
        '          "hint": null',
        "        },",
        "        {",
        '          "name": "skill",',
        '          "status": "warn",',
        '          "detail": "skill not installed",',
        '          "hint": "run gymrat init"',
        "        }",
        "      ]",
        "    }",
        "  ],",
        '  "ok_count": 1,',
        '  "warn_count": 1,',
        '  "fail_count": 0,',
        '  "has_failures": false',
        "}",
    ]
