"""The ``doctor`` diagnostic: report model, section builders, assembly, and renderers.

A :class:`DoctorReport` is a titled list of :class:`CheckSection`s over a shared
:class:`EnvironmentInfo`, with ok/warn/fail counts derived from every check. The
environment, config and workflow section builders are pure functions of their
inputs — the environment probe, the config inspection, and the resolved workflow
config.

:func:`build_doctor_report` is the single entry point that coordinates the git
probe, config inspection, and section builders into an assembled report. The CLI
command layer calls it with an explicit working directory rather than reading
``Path.cwd()`` itself. Its bench section, built by :func:`build_bench_section`,
validates bench configuration without executing the bench command: the adapter
name resolves, a bench command is set, and the command's executable is on PATH
or, when given as a path, present under the repository root. That section looks
the executable up on ``PATH`` or on disk, so it is the one builder that is not
pure.

The text renderer styles each check with a status glyph, indents continuation
lines and hints under it, and closes with a caveat note and a status summary.
Color follows the project's :func:`render_lines` resolution — ``NO_COLOR`` /
``FORCE_COLOR`` and stdout's TTY status decide whether ANSI escapes appear. The
JSON renderer writes its document through :func:`gymrat.report.json_doc.render_document`,
the serializer shared with the compare, measure, probe, and loop JSON documents, so the output
is two-space-indented and carries no ANSI whatever the color settings.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import shlex
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from rich.markup import escape

from gymrat.adapters import get_adapter
from gymrat.config import (
    CONFIG_DEFAULTS,
    BenchlessConfig,
    CliFlags,
    ConfigInspection,
    StopConfig,
    inspect_config,
)
from gymrat.errors import GymratError
from gymrat.git import try_git
from gymrat.report.json_doc import render_document
from gymrat.report.style import (
    format_hint,
    markup,
    render_lines,
)
from gymrat.scaffold import DEFAULT_RUNBOOK_PATH, SKILL_RELATIVE_PATH
from gymrat.session.paths import NotAGitRepositoryError, repo_root

# ---------------------------------------------------------------------------
# report model and pure section builders
# ---------------------------------------------------------------------------


CheckStatus = Literal["ok", "warn", "fail"]
"""The outcome severity of a single diagnostic check."""

_WORKFLOW_SECTION_TITLE = "Workflow"

# The synthetic check name emitted when config errors collapse the workflow
# section; the renderer detects the skip by it.
_WORKFLOW_SKIP_CHECK_NAME = "workflow"


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
    """The assembled report: sections, environment, and derived counts."""

    environment: EnvironmentInfo
    sections: list[CheckSection]
    ok_count: int
    warn_count: int
    fail_count: int

    @property
    def has_failures(self) -> bool:
        """Whether any check failed — the CLI reads this to choose exit code 0 versus 1."""
        return self.fail_count > 0


def create_doctor_report(
    environment: EnvironmentInfo, sections: list[CheckSection]
) -> DoctorReport:
    """Assemble ``sections`` into a report, deriving ok/warn/fail counts from every check."""
    counts: Counter[CheckStatus] = Counter(
        check.status for section in sections for check in section.checks
    )

    return DoctorReport(
        environment=environment,
        sections=sections,
        ok_count=counts["ok"],
        warn_count=counts["warn"],
        fail_count=counts["fail"],
    )


def build_environment_section(
    *, git_available: bool, inside_git_repo: bool, git_error: str | None = None
) -> CheckSection:
    """FAIL when git is absent from PATH; WARN outside a repo or when the root won't resolve."""
    checks = [
        _presence_check(
            Check(
                name="git",
                status="fail",
                detail="git is not available on PATH",
                hint="Install git: https://git-scm.com/downloads",
            ),
            present=git_available,
            ok_detail="git is available on PATH",
        ),
        _presence_check(
            Check(
                name="git repository",
                status="warn",
                detail="current directory is not inside a git repository",
                hint="The compare command resolves refs against a git repository",
            ),
            present=inside_git_repo,
            ok_detail="current directory is inside a git repository",
        ),
    ]

    if git_error is not None:
        checks.append(
            Check(
                name="git repository root",
                status="warn",
                detail=f"could not determine the repository root: {git_error}",
                hint=(
                    "Falling back to the current directory; commands may operate on the wrong path"
                ),
            )
        )

    return CheckSection(title="Environment", checks=checks)


def build_config_section(inspection: ConfigInspection) -> CheckSection:
    """One FAIL per collected config problem; a single OK when clean or absent."""
    if inspection.problems:
        checks = [
            Check(name="config", status="fail", detail=problem) for problem in inspection.problems
        ]
    else:
        detail = (
            "No config file found; operating with defaults only"
            if inspection.config_path is None
            else f"Config file loaded: {inspection.config_path}"
        )
        checks = [Check(name="config", status="ok", detail=detail)]

    return CheckSection(title="Configuration", checks=checks)


def _presence_check(missing: Check, *, present: bool, ok_detail: str) -> Check:
    """``missing`` when the piece it checks is absent, else an OK check of the same name."""
    return Check(name=missing.name, status="ok", detail=ok_detail) if present else missing


def _skipped_section(title: str, check_name: str) -> CheckSection:
    """The placeholder a section collapses to when config errors leave nothing to check."""
    return CheckSection(
        title=title,
        checks=[Check(name=check_name, status="ok", detail="Skipped — fix config errors first")],
    )


def build_workflow_section(
    config: BenchlessConfig,
    *,
    config_has_problems: bool,
    skill_file_exists: bool,
    config_file_exists: bool,
) -> CheckSection:
    """WARN for each missing workflow piece (skill file, checks, stop, runbook) with a fix hint.

    When ``config_has_problems`` is true the individual checks are meaningless — the
    config never settled — so the section collapses to a single skip placeholder.

    Args:
        config: The resolved benchless configuration to check against.
        config_has_problems: Whether config inspection already found problems.
        skill_file_exists: Whether the project's skill file is installed.
        config_file_exists: Whether a config file was found. It picks the runbook
            fix: ``gymrat init`` leaves an existing config file as it is, so the
            key has to be added to that file by hand.

    Returns:
        The assembled workflow check section.
    """
    if config_has_problems:
        return _skipped_section(_WORKFLOW_SECTION_TITLE, _WORKFLOW_SKIP_CHECK_NAME)

    runbook_fix = (
        f'Add `runbook = "{DEFAULT_RUNBOOK_PATH}"` to gymrat.toml.'
        if config_file_exists
        else "Run `gymrat init` to create a runbook, or add `runbook` to gymrat.toml."
    )
    checks = [
        _presence_check(
            Check(
                name="skill file",
                status="warn",
                detail=(
                    "No skill file — Claude Code agents won't have gymrat's workflow instructions"
                ),
                hint="Run `gymrat init` to scaffold the project.",
            ),
            present=skill_file_exists,
            ok_detail="Skill file is installed",
        ),
        _presence_check(
            Check(
                name="checks",
                status="warn",
                detail="checks is not configured",
                hint="Without checks, keep cannot gate commits",
            ),
            present=config.checks is not None,
            ok_detail=f"checks: {config.checks}",
        ),
        _build_stop_check(config.stop),
        _presence_check(
            Check(
                name="runbook",
                status="warn",
                detail="runbook is not configured",
                hint=f"{runbook_fix} Without one, supervise has no instructions to follow.",
            ),
            present=config.runbook is not None,
            ok_detail=f"runbook: {config.runbook}",
        ),
    ]

    return CheckSection(title=_WORKFLOW_SECTION_TITLE, checks=checks)


def _build_stop_check(stop: StopConfig | None) -> Check:
    """OK echoing whichever stop keys are set; WARN when stop is absent or empty."""
    parts: list[str] = []
    if stop is not None:
        if stop.target_value is not None:
            parts.append(f"target_value: {stop.target_value}")
        if stop.max_iterations is not None:
            parts.append(f"max_iterations: {stop.max_iterations}")
    if parts:
        return Check(name="stop", status="ok", detail=f"stop: {', '.join(parts)}")

    return Check(
        name="stop",
        status="warn",
        detail="stop is not configured",
        hint="Without stop, a session has no finish line",
    )


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------


_STATUS_GLYPHS: dict[CheckStatus, str] = {"ok": "✓", "warn": "⚠", "fail": "✗"}
_STATUS_STYLES: dict[CheckStatus, str] = {"ok": "green", "warn": "yellow", "fail": "red"}

# Continuation lines and hints align under the status glyph, two spaces past its
# two-space indent.
_DETAIL_INDENT = "    "

_NOTE_WORKFLOW_RAN = (
    "Note: prepare scripts were not run; only the Claude skill file location "
    "was checked (presence ≠ loaded)."
)
_NOTE_WORKFLOW_SKIPPED = (
    "Note: prepare scripts were not run; workflow checks (including the Claude "
    "skill file check) were skipped because config errors block them."
)


def _workflow_was_skipped(report: DoctorReport) -> bool:
    """Whether config problems made the workflow section collapse to the skip placeholder.

    When that happened the skill file was never looked at, so the caveat note
    drops the skill-file claim.

    Args:
        report: The doctor report whose workflow section is checked.

    Returns:
        ``True`` when the report carries the skip placeholder.
    """
    return any(
        check.name == _WORKFLOW_SKIP_CHECK_NAME
        for section in report.sections
        for check in section.checks
    )


def _header_line(report: DoctorReport) -> str:
    env = report.environment
    name = markup(f"gymrat v{env.gymrat_version}", "bold")
    rest = markup(f" · python {env.python_version} · {env.platform}", "dim")
    return f"{name}{rest}"


def _check_lines(status: CheckStatus, detail: str, hint: str | None) -> list[str]:
    glyph = markup(_STATUS_GLYPHS[status], _STATUS_STYLES[status])
    first, *continuations = detail.split("\n")
    lines = [f"  {glyph} {escape(first)}"]
    lines.extend(f"{_DETAIL_INDENT}{escape(line)}" for line in continuations)
    if hint is not None:
        lines.append(f"{_DETAIL_INDENT}{format_hint(hint)}")
    return lines


def render_doctor_report(report: DoctorReport, *, color: bool | None = None) -> str:
    """Render a doctor report as styled text for the terminal.

    Args:
        report: The assembled doctor report.
        color: Explicit color choice — ``True`` forces ANSI, ``False``
            suppresses it, ``None`` defers to the environment and TTY.

    Returns:
        The rendered report string, with or without ANSI escapes.
    """
    lines: list[str] = [_header_line(report), ""]

    for section in report.sections:
        lines.append(markup(section.title, "bold"))
        for check in section.checks:
            lines.extend(_check_lines(check.status, check.detail, check.hint))
        lines.append("")

    note = _NOTE_WORKFLOW_SKIPPED if _workflow_was_skipped(report) else _NOTE_WORKFLOW_RAN
    lines.append(markup(note, "dim"))
    lines.append("")

    segments: list[tuple[int, tuple[str, str], CheckStatus]] = [
        (report.ok_count, ("ok", "ok"), "ok"),
        (report.warn_count, ("warning", "warnings"), "warn"),
        (report.fail_count, ("failure", "failures"), "fail"),
    ]
    parts: list[str] = []
    for count, (singular, plural), status in segments:
        if count == 0:
            continue
        word = singular if count == 1 else plural
        colored_count = markup(str(count), _STATUS_STYLES[status])
        parts.append(f"{colored_count} {escape(word)}")
    lines.append(" · ".join(parts))

    return render_lines(*lines, color=color)


def render_doctor_json(report: DoctorReport) -> str:
    """Serialize the report as JSON for machine consumption, keyed in snake_case.

    Args:
        report: The assembled doctor report.

    Returns:
        The two-space-indented JSON document: fields in model declaration
        order followed by ``has_failures``, a check without a hint carrying
        ``"hint": null``.
    """
    document: dict[str, object] = {**asdict(report), "has_failures": report.has_failures}
    return render_document(document)


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------


_BENCH_TITLE = "Bench"
_SHELL_OPERATOR_RE = re.compile(r"[;&|(){}<>]")


@dataclass(frozen=True, slots=True)
class GitEnvironment:
    """A probe of git's availability and repository status, resolved without raising.

    Attributes:
        git_available: Whether a git binary runs at all.
        inside_git_repo: Whether ``cwd`` sits inside a git repository.
        repo_root_dir: The repository root, or ``None`` outside a repository.
        git_error: Why repository detection failed, set only for a reason other
            than "not a git repository".
    """

    git_available: bool
    inside_git_repo: bool
    repo_root_dir: str | None = None
    git_error: str | None = None


def detect_git_environment(cwd: str) -> GitEnvironment:
    """Probe git's availability and repository status from ``cwd``, without raising."""
    git_available = try_git(["--version"], cwd) is None
    if not git_available:
        return GitEnvironment(git_available=False, inside_git_repo=False)

    try:
        return GitEnvironment(
            git_available=True, inside_git_repo=True, repo_root_dir=repo_root(cwd)
        )
    except NotAGitRepositoryError:
        return GitEnvironment(git_available=True, inside_git_repo=False)
    except GymratError as error:
        return GitEnvironment(git_available=True, inside_git_repo=True, git_error=str(error))


def _first_command_word(bench: str) -> str | None:
    """Extract the first real executable from a shell command string.

    Skips env-var assignments (``VAR=val``). The executable check is
    meaningless for compound shell expressions and for a word only the shell
    can resolve (``~``, a variable, a command substitution), so those yield no
    word.

    Args:
        bench: The shell command string to extract the first token from.

    Returns:
        The first non-assignment token, or ``None`` when no executable can be
        checked statically.
    """
    if _SHELL_OPERATOR_RE.search(bench):
        return None

    try:
        tokens = shlex.split(bench)
    except ValueError:
        return None

    word = next((token for token in tokens if "=" not in token), None)
    if word is None or word.startswith("~") or "$" in word or "`" in word:
        return None
    return word


def _executable_check(executable: str, base_dir: str) -> Check:
    """A bare name is looked up on PATH; a path is resolved against ``base_dir``."""
    if not any(sep and sep in executable for sep in (os.sep, os.altsep)):
        found = shutil.which(executable) is not None
        return Check(
            name="executable",
            status="ok" if found else "warn",
            detail=f"{executable} {'is available' if found else 'was not found'} on PATH",
        )

    target = Path(base_dir, executable)
    if not target.is_file():
        return Check(name="executable", status="warn", detail=f"{executable} was not found")
    if not os.access(target, os.X_OK):
        return Check(name="executable", status="warn", detail=f"{executable} is not executable")
    return Check(name="executable", status="ok", detail=f"{executable} is executable")


def build_bench_section(
    *, bench: str | None, adapter: str, config_problems: bool = False, base_dir: str
) -> CheckSection:
    """Build the "Bench" section by validating config, without running anything.

    When ``config_problems`` is True and ``bench`` is None the section collapses
    to a single skip placeholder — the bench value was never resolved, so a FAIL
    would be misleading.

    Args:
        bench: The configured bench command, or ``None`` if unresolved.
        adapter: The name of the adapter to validate.
        config_problems: Whether config inspection already found problems.
        base_dir: The directory a bench executable given as a path is resolved
            against.

    Returns:
        The assembled bench check section.
    """
    if config_problems and bench is None:
        return _skipped_section(_BENCH_TITLE, "bench")

    try:
        get_adapter(adapter)
    except GymratError as error:
        return CheckSection(
            title=_BENCH_TITLE,
            checks=[Check(name="adapter", status="fail", detail=str(error), hint=error.hint)],
        )
    checks: list[Check] = [Check(name="adapter", status="ok", detail=f"adapter: {adapter}")]

    if bench is None:
        checks.append(
            Check(
                name="bench",
                status="fail",
                detail="No bench command configured",
                hint='Set the bench command with --bench or the "bench" config key',
            )
        )
        return CheckSection(title=_BENCH_TITLE, checks=checks)

    checks.append(Check(name="bench", status="ok", detail=f"bench: {bench}"))

    executable = _first_command_word(bench)
    if executable is not None:
        checks.append(_executable_check(executable, base_dir))

    return CheckSection(title=_BENCH_TITLE, checks=checks)


def _environment_info() -> EnvironmentInfo:
    return EnvironmentInfo(
        gymrat_version=importlib.metadata.version("gymrat"),
        python_version=platform.python_version(),
        platform=sys.platform,
    )


def build_doctor_report(flags: CliFlags, cwd: str) -> DoctorReport:
    """Coordinate the git probe, config inspection, and section builders into a single report.

    Config inspection yields no config only when it found problems. The config
    defaults then stand in so the bench section still has an adapter to name;
    the workflow section collapses to its skip check.

    Args:
        flags: The command-line overrides to apply during config inspection.
        cwd: The working directory to probe git and config from.

    Returns:
        The assembled doctor report with environment, config, workflow, and bench
        sections.
    """
    git_env = detect_git_environment(cwd)
    base_dir = git_env.repo_root_dir or cwd

    inspection = inspect_config(flags, base_dir)

    env_section = build_environment_section(
        git_available=git_env.git_available,
        inside_git_repo=git_env.inside_git_repo,
        git_error=git_env.git_error,
    )
    config_section = build_config_section(inspection)
    resolved = inspection.config or CONFIG_DEFAULTS
    workflow_section = build_workflow_section(
        resolved,
        config_has_problems=bool(inspection.problems),
        skill_file_exists=(Path(base_dir) / SKILL_RELATIVE_PATH).is_file(),
        config_file_exists=inspection.config_path is not None,
    )
    bench_section = build_bench_section(
        bench=inspection.bench,
        adapter=flags.adapter or resolved.adapter,
        config_problems=bool(inspection.problems),
        base_dir=base_dir,
    )

    return create_doctor_report(
        _environment_info(),
        [env_section, config_section, workflow_section, bench_section],
    )
