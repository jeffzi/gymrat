"""The ``gymrat doctor`` command: probe the project setup and report problems.

Doctor assembles an environment, config, and workflow section from pure builders,
then runs a one-shot bench smoke run. The smoke run is the only part that touches
the repository, so it holds the repository lock — unless ``--no-bench`` skips it,
which then runs lock-free. Any check failure exits 1; an unexpected crash exits 2.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from gymrat_py.cli.shared import (
    AdapterOption,
    BenchOption,
    ConfigOption,
    DebugOption,
    FormatOption,
    NoColorOption,
    OutputFormat,
    PrepareOption,
    ReportRenderers,
    SamplesOption,
    SharedFlags,
    TimeoutOption,
    color_override_of,
    emit_report,
    exit_with_error,
    run_with_signal_abort,
    set_debug_mode,
    suppress_color,
    with_repo_lock,
)
from gymrat_py.config import CONFIG_DEFAULTS, BenchlessConfig, CliFlags
from gymrat_py.config_inspect import inspect_config
from gymrat_py.doctor.bench import BenchSectionInput, build_bench_section
from gymrat_py.doctor.checks import (
    DoctorReport,
    EnvironmentInfo,
    build_config_section,
    build_environment_section,
    build_workflow_section,
    create_doctor_report,
)
from gymrat_py.doctor.render import render_doctor_json, render_doctor_report
from gymrat_py.errors import GymratError, message_of
from gymrat_py.git import NotAGitRepositoryError, try_git
from gymrat_py.init.scaffold import SKILL_RELATIVE_PATH
from gymrat_py.report.types import ReportOptions
from gymrat_py.session.paths import repo_root

NoBenchOption = Annotated[bool, typer.Option("--no-bench", help="skip the smoke-run bench section")]

GATE_EXIT_CODE = 1


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
        return GitEnvironment(git_available=True, inside_git_repo=True, git_error=message_of(error))


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
        gymrat_version=importlib.metadata.version("gymrat-py"),
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
    no_bench: NoBenchOption = False,
    output_format: FormatOption = OutputFormat.text,
    no_color: NoColorOption = False,
    debug: DebugOption = False,
) -> None:
    """Check the project setup and report any problems."""
    set_debug_mode(debug)
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
    if not flags.color:
        suppress_color()
    color_override = color_override_of(flags.color)

    async def build_report(abort: asyncio.Event) -> DoctorReport:
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
            skill_file_exists=(Path(base_dir) / SKILL_RELATIVE_PATH).exists(),
        )
        bench_section = await build_bench_section(
            BenchSectionInput(
                bench=inspection.bench,
                adapter=resolved.adapter,
                timeout_seconds=resolved.timeout_seconds,
                primary=resolved.primary,
                metrics=config_resolved.metrics if config_resolved else None,
                kinds=config_resolved.kinds if config_resolved else None,
                repo_root=base_dir,
                abort=abort,
                no_bench=no_bench,
                config_failed=bool(inspection.problems),
            )
        )

        return create_doctor_report(
            _environment_info(),
            [env_section, config_section, workflow_section, bench_section],
        )

    async def run() -> None:
        # The smoke run is the only part that touches the repository, so
        # `--no-bench` stays lock-free and can run alongside whatever holds the lock.
        report = (
            await run_with_signal_abort(build_report)
            if no_bench
            else await with_repo_lock("doctor", lambda: run_with_signal_abort(build_report))
        )

        emit_report(
            report,
            flags,
            ReportRenderers(
                text=lambda result, _opts: render_doctor_report(result),
                json=render_doctor_json,
            ),
            ReportOptions(color=color_override),
        )

        if report.has_failures:
            raise typer.Exit(GATE_EXIT_CODE)

    try:
        asyncio.run(run())
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 -- CLI boundary: route any failure through the formatter
        exit_with_error(error)
