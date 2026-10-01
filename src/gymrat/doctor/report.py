"""Doctor report assembly: probe the environment and build a ``DoctorReport``.

This module owns ``build_doctor_report``, the single entry point that coordinates
the git probe, config inspection, and section builders into an assembled report.
The CLI command layer calls it with an explicit working directory rather than
reading ``Path.cwd()`` itself.

The bench section, built by ``build_bench_section``, validates bench
configuration without executing the bench command: the adapter name resolves, a
bench command is set, and the command's executable is on PATH.
"""

from __future__ import annotations

import importlib.metadata
import platform
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from gymrat.adapters import get_adapter
from gymrat.config import CONFIG_DEFAULTS, BenchlessConfig, CliFlags, inspect_config
from gymrat.doctor.checks import (
    Check,
    CheckSection,
    DoctorReport,
    EnvironmentInfo,
    build_config_section,
    build_environment_section,
    build_workflow_section,
    create_doctor_report,
)
from gymrat.errors import GymratError, hint_of
from gymrat.git import NotAGitRepositoryError, try_git
from gymrat.scaffold import SKILL_RELATIVE_PATH
from gymrat.session.paths import repo_root

_NO_BENCH_HINT = 'Set the bench command with --bench or the "bench" config key'
_BENCH_TITLE = "Bench"
_SHELL_OPERATOR_RE = re.compile(r"[;&|(){}<>]")


@dataclass(frozen=True, slots=True)
class GitEnvironment:
    """A probe of git's availability and repository status, resolved without raising.

    ``git_error`` is set only when repository detection failed for a reason other
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

    Skips env-var assignments (``VAR=val``). A command with shell
    metacharacters yields no word, since the PATH check is meaningless for
    compound shell expressions.

    Args:
        bench: The shell command string to extract the first token from.

    Returns:
        The first non-assignment token, or ``None`` when the command contains
        shell metacharacters or has no executable token.
    """
    if _SHELL_OPERATOR_RE.search(bench):
        return None

    try:
        tokens = shlex.split(bench)
    except ValueError:
        return None

    for token in tokens:
        if "=" in token:
            continue
        return token
    return None


def build_bench_section(
    *, bench: str | None, adapter: str, config_problems: bool = False
) -> CheckSection:
    """Build the "Bench" section by validating config, without running anything.

    When ``config_problems`` is True and ``bench`` is None the section collapses
    to a single skip placeholder — the bench value was never resolved, so a FAIL
    would be misleading.

    Args:
        bench: The configured bench command, or ``None`` if unresolved.
        adapter: The name of the adapter to validate.
        config_problems: Whether config inspection already found problems.

    Returns:
        The assembled bench check section.
    """
    if config_problems and bench is None:
        return CheckSection(
            title=_BENCH_TITLE,
            checks=[Check(name="bench", status="ok", detail="Skipped — fix config errors first")],
        )

    try:
        get_adapter(adapter)
    except GymratError as error:
        return CheckSection(
            title=_BENCH_TITLE,
            checks=[Check(name="adapter", status="fail", detail=str(error), hint=hint_of(error))],
        )
    checks: list[Check] = [Check(name="adapter", status="ok", detail=f"adapter: {adapter}")]

    if bench is None:
        checks.append(
            Check(
                name="bench",
                status="fail",
                detail="No bench command configured",
                hint=_NO_BENCH_HINT,
            )
        )
        return CheckSection(title=_BENCH_TITLE, checks=checks)

    checks.append(Check(name="bench", status="ok", detail=f"bench: {bench}"))

    executable = _first_command_word(bench)
    if executable is not None:
        if shutil.which(executable) is not None:
            checks.append(
                Check(name="executable", status="ok", detail=f"{executable} is available on PATH")
            )
        else:
            checks.append(
                Check(
                    name="executable",
                    status="warn",
                    detail=f"{executable} was not found on PATH",
                )
            )

    return CheckSection(title=_BENCH_TITLE, checks=checks)


def _defaults_as_benchless() -> BenchlessConfig:
    """A benchless config carrying only the settled defaults.

    Stands in whenever config inspection yields no config: either the config
    has problems, in which case the workflow section collapses to a skip check,
    or no config file exists, in which case the unset ``checks``, ``stop``, and
    ``runbook`` surface as workflow warnings and ``adapter`` feeds the bench
    section.

    Returns:
        A :class:`BenchlessConfig` populated from :data:`CONFIG_DEFAULTS`.
    """
    return BenchlessConfig(
        adapter=CONFIG_DEFAULTS.adapter,
        samples=CONFIG_DEFAULTS.samples,
        timeout_seconds=CONFIG_DEFAULTS.timeout_seconds,
        unstable_noise_pct=CONFIG_DEFAULTS.unstable_noise_pct,
        primary=CONFIG_DEFAULTS.primary,
    )


def _environment_info() -> EnvironmentInfo:
    return EnvironmentInfo(
        gymrat_version=importlib.metadata.version("gymrat"),
        python_version=platform.python_version(),
        platform=sys.platform,
    )


def build_doctor_report(flags: CliFlags, cwd: str) -> DoctorReport:
    """Coordinate the git probe, config inspection, and section builders into a single report.

    Falls back to config defaults for the workflow and bench sections whenever
    config inspection yields no config — on config problems or when no config
    file exists.

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
    resolved = inspection.config or _defaults_as_benchless()
    workflow_section = build_workflow_section(
        resolved,
        config_has_problems=bool(inspection.problems),
        skill_file_exists=(Path(base_dir) / SKILL_RELATIVE_PATH).is_file(),
    )
    bench_section = build_bench_section(
        bench=inspection.bench,
        adapter=flags.adapter or resolved.adapter,
        config_problems=bool(inspection.problems),
    )

    return create_doctor_report(
        _environment_info(),
        [env_section, config_section, workflow_section, bench_section],
    )
