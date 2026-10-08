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

import sys
from collections.abc import Callable
from pathlib import Path

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
from gymrat.errors import GymratError
from tests._config import benchless_config as _config
from tests._doctor_fixtures import doctor_report, environment_info
from tests.adapters._inputs import VALID_ADAPTERS_HINT
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


_UNSET_WORKFLOW_CHECKS = [
    Check(
        name="checks",
        status="warn",
        detail="checks is not configured",
        hint="Without checks, keep cannot gate commits",
    ),
    Check(
        name="stop",
        status="warn",
        detail="stop is not configured",
        hint="Without stop, a session has no finish line",
    ),
    Check(
        name="runbook",
        status="warn",
        detail="runbook is not configured",
        hint=(
            "Run `gymrat init` to create a runbook, or add `runbook` to gymrat.toml. "
            "Without one, supervise has no instructions to follow."
        ),
    ),
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


_STOP_UNSET = Check(
    name="stop",
    status="warn",
    detail="stop is not configured",
    hint="Without stop, a session has no finish line",
)


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
        pytest.param(_config(stop=None), "stop", _STOP_UNSET, id="stop-unset"),
        pytest.param(_config(stop=StopConfig()), "stop", _STOP_UNSET, id="stop-empty"),
        pytest.param(
            _config(runbook="./RUNBOOK.md"),
            "runbook",
            Check(name="runbook", status="ok", detail="runbook: ./RUNBOOK.md"),
            id="runbook-set",
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


def _found_under_usr_bin(cmd: str) -> str:
    return f"/usr/bin/{cmd}"


def _not_on_path(_cmd: str) -> None:
    return None


# ---------------------------------------------------------------------------
# PATH probe with shell command strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bench", "expected_exe"),
    [
        pytest.param('"node" bench.js', "node", id="quoted-exe"),
        pytest.param("VAR=1 make test", "make", id="env-var-prefix"),
        pytest.param("VAR=1 FOO=2 node bench.js", "node", id="multiple-env-vars"),
    ],
)
def test_build_bench_section_when_shell_command_does_extract_real_executable(
    monkeypatch: pytest.MonkeyPatch,
    bench: str,
    expected_exe: str,
):
    monkeypatch.setattr("shutil.which", _found_under_usr_bin)

    section = build_bench_section(bench=bench, adapter="metric-lines", base_dir=".")

    exe_check = next(c for c in section.checks if c.name == "executable")
    assert exe_check == Check(
        name="executable", status="ok", detail=f"{expected_exe} is available on PATH"
    )


@pytest.mark.parametrize(
    "bench",
    [
        pytest.param("cd src && make bench", id="shell-operator-cd"),
        pytest.param("{ make bench; }", id="shell-brace-group"),
        pytest.param("~/bin/bench.sh", id="home-shorthand"),
        pytest.param("$HOME/bin/bench.sh --fast", id="variable-expansion"),
        pytest.param("`pwd`/bench.sh", id="command-substitution"),
    ],
)
def test_build_bench_section_when_shell_metacharacters_does_skip_path_check(
    monkeypatch: pytest.MonkeyPatch,
    bench: str,
):
    monkeypatch.setattr("shutil.which", _not_on_path)

    section = build_bench_section(bench=bench, adapter="metric-lines", base_dir=".")

    assert not any(c.name == "executable" for c in section.checks)


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
                    detail='Unknown adapter: "banana".',
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
    ],
)
def test_build_bench_section_when_built_does_produce_the_exact_section(  # noqa: PLR0917 -- one parameter per input plus the expected section and fixture
    bench: str | None,
    adapter: str,
    config_problems: bool,
    on_path: str | None,
    expected: list[Check],
    monkeypatch: pytest.MonkeyPatch,
):
    def which(_cmd: str) -> str | None:
        return on_path

    monkeypatch.setattr("shutil.which", which)

    section = build_bench_section(
        bench=bench, adapter=adapter, config_problems=config_problems, base_dir="."
    )

    assert section == CheckSection(title="Bench", checks=expected)


_MODULE = "gymrat.doctor"


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
    def fake_repo_root(cwd: str | None = None) -> str:
        msg = "cannot resolve"
        raise GymratError(msg)

    monkeypatch.setattr(f"{_MODULE}.repo_root", fake_repo_root)

    result = detect_git_environment(str(tmp_path))

    assert result == GitEnvironment(
        git_available=True, inside_git_repo=True, git_error="cannot resolve"
    )


# ---------------------------------------------------------------------------
# config inspection — real gymrat.toml
# ---------------------------------------------------------------------------


def _check(report: DoctorReport, title: str, name: str) -> Check:
    section = next(section for section in report.sections if section.title == title)
    return next(check for check in section.checks if check.name == name)


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
def test_build_doctor_report_when_config_read_for_real_does_report_its_findings(
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
        detail='Unknown adapter: "banana".',
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


# CSI introducer; present in the output iff any ANSI escape was emitted.
_ESCAPE_PREFIX = "\x1b["


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
    ("make_report", "color"),
    [
        pytest.param(_mixed_status_report, False, id="ok-warn-fail-color-off"),
        pytest.param(_mixed_status_report, True, id="ok-warn-fail-color-on"),
        pytest.param(_warning_only_report, False, id="warn-only-color-off"),
        pytest.param(_multiline_detail_report, False, id="multiline-detail-color-off"),
        pytest.param(_workflow_skipped_report, False, id="workflow-skipped-color-off"),
        pytest.param(_plural_counts_report, False, id="plural-counts-color-off"),
    ],
)
def test_render_doctor_report_when_rendered_does_match_the_snapshot(
    make_report: Callable[[], DoctorReport], color: bool, snapshot: SnapshotAssertion
):
    report = make_report()

    output = render_doctor_report(report, color=color)

    assert output.split("\n") == snapshot


# ---------------------------------------------------------------------------
# JSON rendering
# ---------------------------------------------------------------------------


def test_render_doctor_json_when_rendered_does_emit_two_space_indented_document():
    report = _warning_only_report()

    output = render_doctor_json(report)

    assert output.split("\n") == [
        "{",
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
