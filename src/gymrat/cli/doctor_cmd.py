"""The ``gymrat doctor`` command: probe the project setup and report problems.

Doctor validates the project's configuration — environment, config file, workflow
keys, bench command, and adapter — without running any benchmarks. Any check
failure exits 1; an unexpected crash exits 2.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

from gymrat.cli.shared import (
    GATE_EXIT_CODE,
    AdapterOption,
    BenchOption,
    ConfigOption,
    DebugOption,
    FormatOption,
    NoColorOption,
    OutputFormat,
    PrepareOption,
    SamplesOption,
    SharedFlags,
    TimeoutOption,
    apply_debug,
    color_override_of,
    resolve_stream_color,
    run_cli,
    wants_json,
    write_and_flush,
)
from gymrat.config import CONFIG_DEFAULTS, BenchlessConfig, CliFlags
from gymrat.config_inspect import inspect_config
from gymrat.doctor.bench import build_bench_section
from gymrat.doctor.checks import (
    DoctorReport,
    EnvironmentInfo,
    build_config_section,
    build_environment_section,
    build_workflow_section,
    create_doctor_report,
)
from gymrat.doctor.render import (
    render_doctor_json,
    render_doctor_report,
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


def doctor_command(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    *,
    bench: BenchOption = None,
    prepare: PrepareOption = None,
    adapter: AdapterOption = None,
    samples: SamplesOption = None,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    output_format: FormatOption = OutputFormat.text,
    no_color: NoColorOption = False,
    debug: DebugOption = False,
) -> None:
    """Check the project setup and report any problems."""
    apply_debug(debug)
    flags = SharedFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
        color=not no_color,
        format=output_format.value,
    )
    color_override = color_override_of(flags.color)

    def _build_report() -> DoctorReport:
        cwd = str(Path.cwd())
        git_env = detect_git_environment(cwd)
        base_dir = git_env.repo_root_dir or cwd

        config_flags = CliFlags(
            bench=bench,
            prepare=prepare,
            adapter=adapter,
            samples=samples,
            timeout=timeout,
            config=config,
        )
        inspection = inspect_config(config_flags, base_dir)
        config_resolved = inspection.config

        env_section = build_environment_section(
            git_available=git_env.git_available,
            inside_git_repo=git_env.inside_git_repo,
            git_error=git_env.git_error,
        )
        config_section = build_config_section(inspection)
        resolved = config_resolved or _defaults_as_benchless()
        workflow_section = build_workflow_section(
            resolved,
            problems=inspection.problems,
            skill_file_exists=(Path(base_dir) / SKILL_RELATIVE_PATH).is_file(),
        )
        bench_section = build_bench_section(
            bench=inspection.bench,
            adapter=adapter or resolved.adapter,
            config_problems=bool(inspection.problems),
        )

        return create_doctor_report(
            _environment_info(),
            [env_section, config_section, workflow_section, bench_section],
        )

    async def run() -> None:
        report = _build_report()

        if wants_json(flags):
            write_and_flush(sys.stdout, render_doctor_json(report) + "\n")
        else:
            color = resolve_stream_color(color_override, sys.stdout)
            output = render_doctor_report(report, color=color)
            write_and_flush(sys.stdout, output + "\n")

        if report.has_failures:
            raise typer.Exit(GATE_EXIT_CODE)

    run_cli(run)
