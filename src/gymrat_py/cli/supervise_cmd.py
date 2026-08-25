"""The ``gymrat supervise`` command: a supervised agent session under caps.

The action guards against a dirty working tree, takes the supervise lock (never
the repository lock a bench holds), resolves where the JSONL event log lands,
and hands a Claude driver, a progress reporter, and the composed kickoff to the
supervisor. The supervisor's backend is imported lazily inside the driver, so
assembling the CLI never pulls the agent SDK; every collaborator this module
touches stays light enough to import when ``gymrat --help`` runs.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from gymrat_py.cli.shared import (
    GATE_EXIT_CODE,
    TOOL_FAILURE_EXIT_CODE,
    DebugOption,
    NoColorOption,
    exit_with_error,
    parse_max_minutes,
    parse_positive_number,
    resolve_render_mode,
    set_debug_mode,
    write_and_flush,
)
from gymrat_py.cli.supervise_progress import create_supervise_reporter
from gymrat_py.config import CliFlags, resolve_benchless_config
from gymrat_py.errors import GymratError
from gymrat_py.eta import format_duration
from gymrat_py.git import run_git
from gymrat_py.report.format import pluralize
from gymrat_py.session.clock import now_ms
from gymrat_py.session.lock import acquire_lock
from gymrat_py.session.paths import repo_root, session_dir, supervise_lockfile_path
from gymrat_py.session.workspace import ensure_git_exclude
from gymrat_py.signals import install_termination_cleanup
from gymrat_py.supervisor import (
    KickoffResult,
    SessionPrompt,
    SupervisionResult,
    compose_kickoff,
    create_claude_driver,
    supervise,
)
from gymrat_py.supervisor.events import DirtyInfo, LaunchEvent, summarize

_PromptArgument = Annotated[
    str | None,
    typer.Argument(metavar="[PROMPT]", help="optimization prompt for the agent"),
]
_MaxMinutesOption = Annotated[
    float,
    typer.Option("--max-minutes", parser=parse_max_minutes, help="wall-clock cap in minutes"),
]
_MaxUsdOption = Annotated[
    float | None,
    typer.Option("--max-usd", parser=parse_positive_number, help="spend cap in USD"),
]
_LogOption = Annotated[str | None, typer.Option("--log", help="path for the JSONL event log")]
_ModelOption = Annotated[
    str | None, typer.Option("--model", help="model to use for the agent session")
]
_AllowDirtyOption = Annotated[
    bool, typer.Option("--allow-dirty", help="allow launching with uncommitted changes")
]


def _dirty_file_count(root: str) -> int:
    """Count the working-tree entries git reports, expanding untracked directories.

    ``--untracked-files=all`` lists each file inside an untracked directory
    rather than the directory alone, so a new folder of three files counts as
    three.
    """
    output = run_git(["status", "--porcelain", "--untracked-files=all"], root).strip()
    if not output:
        return 0
    return len(output.split("\n"))


def _validate_working_tree(root: str, *, allow_dirty: bool) -> int:
    """Refuse a dirty tree unless ``allow_dirty`` was set, warning when it was."""
    count = _dirty_file_count(root)
    if count > 0 and not allow_dirty:
        message = f"Working tree has {pluralize(count, 'uncommitted file')}."
        hint = "Commit or stash your changes, or pass --allow-dirty to proceed anyway."
        exit_with_error(GymratError(message, hint=hint))
    if count > 0:
        write_and_flush(
            sys.stderr,
            f"warning: working tree has {pluralize(count, 'dirty file')} — "
            "proceeding because --allow-dirty was set\n",
        )
    return count


def _resolve_log_path(root: str, explicit: str | None) -> str:
    """The caller's ``--log`` verbatim, or a timestamped path under the session dir.

    Only the default path is written under ``.gymrat/``, so only that branch
    ensures the directory is git-excluded; a caller-supplied path is left to the
    caller to place and ignore.
    """
    if explicit is not None:
        return explicit
    ensure_git_exclude(root)
    return str(Path(session_dir(root)) / f"supervisor-{now_ms()}.jsonl")


@dataclass(frozen=True, slots=True)
class _SessionContext:
    """Everything the session run needs, assembled once the lock is held."""

    root: str
    log_path: str
    launch: LaunchEvent
    kickoff: KickoffResult
    max_minutes: float
    max_usd: float | None
    max_iterations: int | None
    model: str | None
    color: bool


def _report_result(result: SupervisionResult, log_path: str) -> None:
    """Print the closing summary to stdout, then exit per how the run ended.

    A session that ended on its own returns normally (exit 0). A cap trip exits
    on the gate code. An error outcome exits on the tool-failure code, surfacing
    its message to stderr only when one is present.
    """
    summary = "\n".join(
        [
            f"outcome: {result.outcome.reason}",
            f"ended by: {result.ended_by}",
            f"duration: {format_duration(result.duration_ms)}",
            f"cost: ${result.cost_usd:.2f}",
            f"log: {log_path}",
        ]
    )
    write_and_flush(sys.stdout, f"{summary}\n")

    if result.outcome.reason == "error":
        if result.outcome.message:
            exit_with_error(GymratError(result.outcome.message))
        raise typer.Exit(TOOL_FAILURE_EXIT_CODE)

    if result.ended_by != "session":
        raise typer.Exit(GATE_EXIT_CODE)


def _run_session(ctx: _SessionContext) -> None:
    """Drive the supervised session, reporting progress and stopping it cleanly."""
    driver = create_claude_driver()
    mode = resolve_render_mode(ctx.color)
    reporter = create_supervise_reporter(
        root=ctx.root,
        max_minutes=ctx.max_minutes,
        max_usd=ctx.max_usd,
        max_iterations=ctx.max_iterations,
        mode=mode,
    )
    uninstall_cleanup = install_termination_cleanup(reporter.stop)

    write_and_flush(sys.stderr, f"log: {ctx.log_path}\n")

    prompt = SessionPrompt(
        kickoff=ctx.kickoff.kickoff,
        cwd=ctx.root,
        system_prompt_append=ctx.kickoff.system_prompt_append,
        model=ctx.model,
    )
    try:
        result = asyncio.run(
            supervise(
                driver=driver,
                prompt=prompt,
                max_minutes=ctx.max_minutes,
                max_usd=ctx.max_usd,
                log_path=ctx.log_path,
                launch=ctx.launch,
                observer=reporter.observer,
            )
        )
        _report_result(result, ctx.log_path)
    finally:
        reporter.stop()
        uninstall_cleanup()


@dataclass(frozen=True, slots=True)
class _Options:
    """The parsed flag surface, gathered so the run helpers take one argument."""

    prompt: str | None
    max_minutes: float
    max_usd: float | None
    log: str | None
    model: str | None
    allow_dirty: bool
    color: bool


def _execute(options: _Options) -> None:
    """Guard the tree, hold the supervise lock, and run one session under it."""
    root = repo_root()
    dirty_count = _validate_working_tree(root, allow_dirty=options.allow_dirty)

    release = acquire_lock(supervise_lockfile_path(root), "supervise")
    try:
        log_path = _resolve_log_path(root, options.log)
        config = resolve_benchless_config(CliFlags(), root)
        kickoff = compose_kickoff(config, options.prompt)
        head_sha = run_git(["rev-parse", "HEAD"], root).strip()

        launch = LaunchEvent(
            timestamp=now_ms(),
            head_sha=head_sha,
            dirty=DirtyInfo(file_count=dirty_count) if dirty_count > 0 else False,
            max_minutes=options.max_minutes,
            max_usd=options.max_usd,
            model=options.model,
            runbook_path=config.runbook or "",
            kickoff_summary=summarize(kickoff.kickoff),
        )

        _run_session(
            _SessionContext(
                root=root,
                log_path=log_path,
                launch=launch,
                kickoff=kickoff,
                max_minutes=options.max_minutes,
                max_usd=options.max_usd,
                max_iterations=config.stop.max_iterations if config.stop is not None else None,
                model=options.model,
                color=options.color,
            )
        )
    finally:
        release()


def supervise_command(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the option surface
    prompt: _PromptArgument = None,
    *,
    max_minutes: _MaxMinutesOption,
    max_usd: _MaxUsdOption = None,
    log: _LogOption = None,
    model: _ModelOption = None,
    allow_dirty: _AllowDirtyOption = False,
    no_color: NoColorOption = False,
    debug: DebugOption = False,
) -> None:
    """Run a supervised agent session with wall-clock and spend caps."""
    set_debug_mode(debug)
    options = _Options(
        prompt=prompt,
        max_minutes=max_minutes,
        max_usd=max_usd,
        log=log,
        model=model,
        allow_dirty=allow_dirty,
        color=not no_color,
    )
    try:
        _execute(options)
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 -- CLI boundary: route any failure through the formatter
        exit_with_error(error)
