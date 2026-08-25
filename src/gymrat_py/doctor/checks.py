"""The doctor report model and its pure diagnostic section builders.

A :class:`DoctorReport` is a titled list of :class:`CheckSection`s over a shared
:class:`EnvironmentInfo`, with ok/warn/fail counts derived from every check. The
section builders here are pure functions of their inputs — the environment probe,
the config inspection, and the resolved workflow config — so the bench smoke run
(which touches the filesystem) lives in its own module.
"""

from dataclasses import dataclass
from typing import Literal

from gymrat_py.config import BenchlessConfig, StopConfig
from gymrat_py.config_inspect import ConfigInspection

CheckStatus = Literal["ok", "warn", "fail"]
"""The outcome severity of a single diagnostic check."""


@dataclass(frozen=True, slots=True)
class Check:
    """One diagnostic probe: a status, a human detail line, and an optional fix hint."""

    name: str
    status: CheckStatus
    detail: str
    hint: str | None = None


@dataclass(frozen=True, slots=True)
class CheckSection:
    """A titled group of related checks (e.g. "Environment", "Configuration")."""

    title: str
    checks: list[Check]


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """Version and platform context printed at the top of the doctor report."""

    gymrat_version: str
    python_version: str
    platform: str


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """The assembled report: sections, environment, and derived counts.

    ``has_failures`` is ``fail_count > 0`` — the CLI reads it to choose exit code
    0 versus 1.
    """

    environment: EnvironmentInfo
    sections: list[CheckSection]
    ok_count: int
    warn_count: int
    fail_count: int
    has_failures: bool


def _ok(name: str, detail: str) -> Check:
    return Check(name=name, status="ok", detail=detail)


def _issue(name: str, status: CheckStatus, detail: str, hint: str) -> Check:
    return Check(name=name, status=status, detail=detail, hint=hint)


def create_doctor_report(
    environment: EnvironmentInfo, sections: list[CheckSection]
) -> DoctorReport:
    """Assemble ``sections`` into a report, deriving ok/warn/fail counts from every check."""
    counts: dict[CheckStatus, int] = {"ok": 0, "warn": 0, "fail": 0}
    for section in sections:
        for check in section.checks:
            counts[check.status] += 1

    return DoctorReport(
        environment=environment,
        sections=sections,
        ok_count=counts["ok"],
        warn_count=counts["warn"],
        fail_count=counts["fail"],
        has_failures=counts["fail"] > 0,
    )


# ---------------------------------------------------------------------------
# Environment section
# ---------------------------------------------------------------------------


def build_environment_section(
    *, git_available: bool, inside_git_repo: bool, git_error: str | None = None
) -> CheckSection:
    """FAIL when git is absent from PATH; WARN outside a repo or when the root won't resolve."""
    checks: list[Check] = []

    checks.append(
        _ok("git", "git is available on PATH")
        if git_available
        else _issue(
            "git",
            "fail",
            "git is not available on PATH",
            "Install git: https://git-scm.com/downloads",
        )
    )

    checks.append(
        _ok("git repository", "current directory is inside a git repository")
        if inside_git_repo
        else _issue(
            "git repository",
            "warn",
            "current directory is not inside a git repository",
            "The compare command resolves refs against a git repository",
        )
    )

    if git_error is not None:
        checks.append(
            _issue(
                "git repository root",
                "warn",
                f"could not determine the repository root: {git_error}",
                "Falling back to the current directory; commands may operate on the wrong path",
            )
        )

    return CheckSection(title="Environment", checks=checks)


# ---------------------------------------------------------------------------
# Config section
# ---------------------------------------------------------------------------


def build_config_section(inspection: ConfigInspection) -> CheckSection:
    """One FAIL per collected config problem; a single OK when clean or absent."""
    if inspection.problems:
        checks = [
            Check(name="config", status="fail", detail=problem) for problem in inspection.problems
        ]
    elif inspection.config_path is None:
        checks = [_ok("config", "No config file found; operating with defaults only")]
    else:
        checks = [_ok("config", f"Config file loaded: {inspection.config_path}")]

    return CheckSection(title="Configuration", checks=checks)


# ---------------------------------------------------------------------------
# Workflow section
# ---------------------------------------------------------------------------

_SKILL_MISSING_HINT = "Run `gymrat init` to scaffold the project."
_CHECKS_MISSING_HINT = "Without checks, keep cannot gate commits"
_STOP_MISSING_HINT = "Without stop, a session has no finish line"
_RUNBOOK_MISSING_HINT = (
    "Run `gymrat init` to create a runbook, or add `runbook` to gymrat.json. "
    "Without one, supervise has no instructions to follow."
)


def build_workflow_section(
    config: BenchlessConfig, *, problems: list[str], skill_file_exists: bool
) -> CheckSection:
    """WARN for each missing workflow piece (skill file, checks, stop, runbook) with a fix hint.

    When ``problems`` is non-empty the individual checks are meaningless — the
    config never settled — so the section collapses to a single skip placeholder.
    """
    if problems:
        return CheckSection(
            title="Workflow",
            checks=[_ok("workflow", "Skipped — fix config errors first")],
        )

    checks: list[Check] = []

    checks.append(
        _ok("skill file", "Skill file is installed")
        if skill_file_exists
        else _issue(
            "skill file",
            "warn",
            "No skill file — Claude Code agents won't have gymrat's workflow instructions",
            _SKILL_MISSING_HINT,
        )
    )

    checks.append(
        _ok("checks", f"checks: {config.checks}")
        if config.checks is not None
        else _issue("checks", "warn", "checks is not configured", _CHECKS_MISSING_HINT)
    )

    checks.append(_build_stop_check(config.stop))

    checks.append(
        _ok("runbook", f"runbook: {config.runbook}")
        if config.runbook is not None
        else _issue("runbook", "warn", "runbook is not configured", _RUNBOOK_MISSING_HINT)
    )

    return CheckSection(title="Workflow", checks=checks)


def _build_stop_check(stop: StopConfig | None) -> Check:
    """OK echoing whichever stop keys are set; WARN when stop is absent or empty."""
    if stop is not None and (stop.target_value is not None or stop.max_iterations is not None):
        parts: list[str] = []
        if stop.target_value is not None:
            parts.append(f"targetValue: {stop.target_value}")
        if stop.max_iterations is not None:
            parts.append(f"maxIterations: {stop.max_iterations}")
        return _ok("stop", f"stop: {', '.join(parts)}")

    return _issue("stop", "warn", "stop is not configured", _STOP_MISSING_HINT)
