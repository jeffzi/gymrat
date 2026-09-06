"""Doctor report assembly: probe the environment and build a ``DoctorReport``.

This module owns ``build_doctor_report``, the single entry point that coordinates
the git probe, config inspection, and section builders into an assembled report.
The CLI command layer calls it with an explicit working directory rather than
reading ``Path.cwd()`` itself.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from gymrat.config import CONFIG_DEFAULTS, BenchlessConfig, CliFlags
from gymrat.config.inspect import inspect_config
from gymrat.doctor.bench import build_bench_section
from gymrat.doctor.checks import (
    DoctorReport,
    EnvironmentInfo,
    build_config_section,
    build_environment_section,
    build_workflow_section,
    create_doctor_report,
)
from gymrat.errors import GymratError
from gymrat.git import NotAGitRepositoryError, try_git
from gymrat.init.scaffold import SKILL_RELATIVE_PATH
from gymrat.session.paths import repo_root


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


def _defaults_as_benchless() -> BenchlessConfig:
    """A benchless config carrying only the settled defaults.

    Used for the workflow section when the config failed to settle: that path
    collapses to a skip check before any config field is read, so the missing
    loop keys never matter.
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

    Falls back to config defaults for the workflow section when config inspection fails.
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
