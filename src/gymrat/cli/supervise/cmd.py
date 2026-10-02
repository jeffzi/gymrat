"""The ``gymrat supervise`` command: a supervised agent session under caps.

Guards against a dirty working tree, takes the supervise lock, resolves the JSONL
event log, and hands a driver, reporter, and kickoff to the supervisor. Every run
the supervisor returns from then passes through the exit sequence, which settles
and (unless ``--no-finalize``) finalizes the session before the summary prints.
The agent SDK is imported lazily inside the driver so ``gymrat --help`` stays fast.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from gymrat.cli.supervise.types import ReadSessionResult, SuperviseReporter
    from gymrat.supervisor.events import SessionObserver
    from gymrat.supervisor.supervise import SupervisionResult

from gymrat.cli.console import apply_color_override, apply_debug, resolve_stream_color
from gymrat.cli.options import (
    BaselineOption,
    ColorOption,
    DebugOption,
    parse_max_minutes,
    parse_positive_number,
)
from gymrat.cli.shared import exit_with_error, resolve_render_mode, write_and_flush, write_stdout
from gymrat.cli.supervise.preflight import doctor_gate, run_preflight, validate_experiment_worktree
from gymrat.cli.supervise.progress import create_supervise_reporter
from gymrat.cli.supervise.summary import SessionLabels, build_summary
from gymrat.clock import now_ms, now_ns
from gymrat.config.resolve import resolve_config
from gymrat.config.types import CliFlags, Effort, ResolvedConfig, SuperviseConfig
from gymrat.display_path import abbreviate_home
from gymrat.errors import GATE_EXIT_CODE, TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.exec import kill_live_process_groups
from gymrat.git import run_git
from gymrat.plural import pluralize
from gymrat.report.style import RENDER_WIDTH, render_lines
from gymrat.session.budget import (
    Budget,
    clear_budget,
    minutes_to_ms,
    write_budget,
)
from gymrat.session.lock import acquire_lock
from gymrat.session.paths import (
    lockfile_path,
    repo_root,
    session_dir,
    supervise_lockfile_path,
    supervisor_log_name,
)
from gymrat.session.workspace import dirty_file_count, ensure_git_exclude
from gymrat.signals import install_termination_cleanup
from gymrat.supervisor.claude import create_claude_driver
from gymrat.supervisor.context import SupervisedSession
from gymrat.supervisor.driver import Driver, SessionPrompt
from gymrat.supervisor.events import (
    DirtyInfo,
    LaunchEvent,
    combine_observers,
    create_event_log_writer,
    probe_event_log_path,
    summarize,
)
from gymrat.supervisor.exit_sequence import ExitReport, run_exit_sequence
from gymrat.supervisor.hooks import supervise_hooks_factory
from gymrat.supervisor.kickoff import KickoffResult, compose_kickoff
from gymrat.supervisor.supervise import supervise
from gymrat.supervisor.tools import gymrat_tools_factory
from gymrat.warn import warn_to_stderr

# ---------------------------------------------------------------------------
# Flag surface
# ---------------------------------------------------------------------------

PromptArgument = Annotated[
    str | None,
    typer.Argument(metavar="[PROMPT]", help="optimization prompt for the agent"),
]
MaxMinutesOption = Annotated[
    float,
    typer.Option(
        "--max-minutes",
        parser=parse_max_minutes,
        metavar="<float>",
        help="wall-clock cap in minutes, counted from when the baseline is recorded",
    ),
]
MaxUsdOption = Annotated[
    float | None,
    typer.Option(
        "--max-usd", parser=parse_positive_number, metavar="<float>", help="spend cap in USD"
    ),
]
LogOption = Annotated[str | None, typer.Option("--log", help="path for the JSONL event log")]
ModelOption = Annotated[
    str | None, typer.Option("--model", help="model to use for the agent session")
]
AllowDirtyOption = Annotated[
    bool, typer.Option("--allow-dirty", help="allow launching with uncommitted changes")
]
ForceOption = Annotated[
    bool,
    typer.Option(
        "--force",
        help="launch even when the cap cannot fit one iteration or a stop condition is already met",
    ),
]
NoFinalizeOption = Annotated[
    bool,
    typer.Option("--no-finalize", help="leave the session open instead of finalizing it on exit"),
]
EffortOption = Annotated[
    Effort | None,
    typer.Option("--effort", metavar="<level>", help="effort level"),
]


@dataclass(frozen=True, slots=True)
class Options:
    """The parsed flag surface, gathered so the run helpers take one argument."""

    prompt: str | None
    max_minutes: float
    max_usd: float | None
    log: str | None
    baseline: str | None
    model: str | None
    effort: Effort | None
    allow_dirty: bool
    force: bool
    color: bool | None
    finalize: bool


# ---------------------------------------------------------------------------
# Pre-flight guards
# ---------------------------------------------------------------------------


def _validate_working_tree(root: str, *, allow_dirty: bool) -> int:
    """Refuse a dirty tree unless ``allow_dirty`` was set, warning when it was."""
    count = dirty_file_count(root)
    if count == 0:
        return count

    if not allow_dirty:
        message = f"Working tree has {pluralize(count, 'uncommitted file')}."
        hint = "Commit or stash your changes, or pass --allow-dirty to proceed anyway."
        exit_with_error(GymratError(message, hint=hint))

    warn_to_stderr(
        f"warning: working tree has {pluralize(count, 'dirty file')} — "
        "proceeding because --allow-dirty was set"
    )
    return count


def _resolve_log_path(root: str, explicit: str | None) -> str:
    """The caller's ``--log`` verbatim, or a timestamped path under the session dir.

    Only the default path is written under ``.gymrat/``, so only that branch
    ensures the directory is git-excluded; a caller-supplied path is left to the
    caller to place and ignore.

    Args:
        root: The repository root path.
        explicit: The caller's ``--log`` value, or ``None`` to use the default path.

    Returns:
        The resolved absolute path for the event log.
    """
    if explicit is not None:
        return explicit
    ensure_git_exclude(root)
    return str(Path(session_dir(root)) / supervisor_log_name(now_ms()))


# ---------------------------------------------------------------------------
# Session budget and run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SessionContext:
    """Everything the session run needs, assembled once the lock is held."""

    root: str
    log_path: str
    launch: LaunchEvent
    kickoff: KickoffResult
    config: ResolvedConfig
    max_minutes: float
    max_usd: float | None
    max_iterations: int | None
    model: str | None
    effort: Effort | None
    color: bool | None
    branch: str
    finalize: bool

    def session_prompt(self) -> SessionPrompt:
        return SessionPrompt(
            kickoff=self.kickoff.kickoff,
            cwd=self.root,
            system_prompt_append=self.kickoff.system_prompt_append,
            model=self.model,
            effort=self.effort,
            command_timeout_ms=minutes_to_ms(self.max_minutes),
            max_budget_usd=self.max_usd,
        )


def _report_result(
    result: SupervisionResult,
    *,
    ctx: _SessionContext,
    session_result: ReadSessionResult | None,
    final_text: str | None,
    exit_report: ExitReport,
) -> None:
    """Print the closing summary, then exit with the code the run end maps to.

    A driver error wins (its message goes to stderr), then an exit-sequence error
    (already shown as the summary's ``exit  error`` row, so nothing more is
    printed), then how the run ended; a clean end returns normally.

    Args:
        result: What the supervisor returned.
        ctx: The session run's context.
        session_result: The session as the exit sequence left it.
        final_text: The agent's last message, if any.
        exit_report: What the exit sequence did.

    Raises:
        typer.Exit: With the tool-failure code on a driver or exit-sequence
            error, or the gate code when anything but the session or a stop
            condition ended the run.
    """
    summary = render_lines(
        build_summary(
            result,
            log_path=ctx.log_path,
            session_result=session_result,
            final_text=final_text,
            labels=SessionLabels(model=ctx.model, effort=ctx.effort),
            exit_report=exit_report,
        ),
        color=resolve_stream_color(ctx.color, sys.stdout),
        width=RENDER_WIDTH,
    )
    write_stdout(f"{summary}\n")

    if result.outcome.reason == "error":
        if result.outcome.message:
            exit_with_error(GymratError(result.outcome.message))
        raise typer.Exit(TOOL_FAILURE_EXIT_CODE)

    if exit_report.error is not None:
        raise typer.Exit(TOOL_FAILURE_EXIT_CODE)

    if result.ended_by not in ("session", "stop-condition"):
        raise typer.Exit(GATE_EXIT_CODE)


def _init_budget(root: str, max_minutes: float) -> tuple[float, Callable[[], None]]:
    """Create, persist, and arm cleanup for the session time budget.

    Args:
        root: Session root directory.
        max_minutes: Maximum session duration in minutes.

    Returns:
        A ``(deadline_ms, release)`` pair: the absolute deadline and a callback
        that removes the budget file and uninstalls the termination hook,
        uninstalling even when the removal raises. The callback is idempotent,
        so the run's own teardown and the unwind of a session whose setup
        failed can both call it without clearing twice.
    """
    started_at_ms = now_ms()
    deadline_ms = started_at_ms + minutes_to_ms(max_minutes)
    budget = Budget(
        started_at_ms=started_at_ms,
        max_minutes=max_minutes,
        deadline_ms=deadline_ms,
    )
    Path(session_dir(root)).mkdir(parents=True, exist_ok=True)
    write_budget(root, budget)
    uninstall = install_termination_cleanup(lambda: clear_budget(root))
    released = False

    def release() -> None:
        nonlocal released
        if released:
            return
        released = True
        try:
            clear_budget(root)
        finally:
            uninstall()

    return deadline_ms, release


def _create_reporter(ctx: _SessionContext, mode: Literal["live", "plain"]) -> SuperviseReporter:
    return create_supervise_reporter(
        root=ctx.root,
        max_minutes=ctx.max_minutes,
        max_usd=ctx.max_usd,
        max_iterations=ctx.max_iterations,
        mode=mode,
        log_path=ctx.log_path,
        color=ctx.color,
        model=ctx.model,
        effort=ctx.effort,
        session_id=ctx.launch.session_id,
        branch=ctx.branch,
    )


def _supervised_session(ctx: _SessionContext, deadline_ms: float) -> SupervisedSession:
    return SupervisedSession(
        root=ctx.root,
        log_path=ctx.log_path,
        lock_path=lockfile_path(ctx.root),
        config=ctx.config,
        deadline_ms=deadline_ms,
        max_minutes=ctx.max_minutes,
        max_usd=ctx.max_usd,
    )


async def _supervise_then_release(
    pending: Coroutine[object, object, SupervisionResult],
    release_budget: Callable[[], None],
) -> SupervisionResult:
    """Await the supervisor, releasing the budget the moment it stops.

    Args:
        pending: The supervisor run.
        release_budget: Removes the budget file and its termination cleanup.

    Returns:
        What the supervisor returned.

    Raises:
        Exception: Whatever the supervisor raised, unchanged.
    """
    try:
        return await pending
    finally:
        release_budget()


async def _supervise_and_exit_sequence(
    reporter: SuperviseReporter,
    pending: Coroutine[object, object, SupervisionResult],
    *,
    ctx: _SessionContext,
    context: SupervisedSession,
    observer: SessionObserver,
) -> tuple[SupervisionResult, ExitReport]:
    """Run the supervisor, then the exit sequence, inside one running loop.

    The exit sequence runs only when the supervisor returned; its phases and
    warnings go through the still-running reporter, and its decisions land in
    the same event log and observer the supervisor wrote to.

    Args:
        reporter: The dashboard, already showing when the supervisor is awaited.
        pending: The supervisor run, which releases the budget when it stops so
            the exit sequence never runs under a live budget.
        ctx: The session run's context.
        context: The supervised session the supervisor and exit sequence share.
        observer: The observer the supervisor was handed.

    Returns:
        What the supervisor returned and what the exit sequence did.

    Raises:
        Exception: Whatever the supervisor raised, unchanged.
    """
    result = await pending
    exit_report = await run_exit_sequence(
        context,
        ended_by=result.ended_by,
        finalize=ctx.finalize,
        progress=reporter.exit_phase,
        log=combine_observers(create_event_log_writer(ctx.log_path), observer),
        warn=reporter.warn,
    )
    # A step that writes the session log and then fails emits no event, so the
    # reporter would otherwise summarize the session as it was before that step.
    reporter.refresh_session()
    return result, exit_report


def _create_driver(root: str) -> Driver:
    """The Claude driver with gymrat's tools and the worktree-guard hooks for ``root``."""
    return create_claude_driver(
        tools=gymrat_tools_factory(root), hooks=supervise_hooks_factory(Path(root))
    )


def _run_session(ctx: _SessionContext) -> None:
    """Drive the supervised session, reporting progress and stopping it cleanly.

    Everything the session arms — the reporter, the budget file and its cleanup,
    the process-group kill cleanup — is released before this returns, including
    when the setup between arming them and starting the supervisor fails, so a
    failed run leaves nothing registered behind.

    The reporter is built before either cleanup is installed: its live display
    installs its own signal erase as it starts, and that erase must run before
    the budget release and the process-group kill.

    Args:
        ctx: Everything the run needs, assembled once the lock is held.
    """
    from gymrat.cli.supervise.span_lifecycle import finalize_tracing, setup_tracing  # noqa: PLC0415

    driver = _create_driver(ctx.root)
    mode = resolve_render_mode()
    reporter = _create_reporter(ctx, mode)

    with ExitStack() as armed:
        # Stops the display exactly once: here when setup below fails, before the
        # error reaches the terminal, or when the run path closes it early.
        display = armed.enter_context(ExitStack())
        display.callback(reporter.stop)
        deadline_ms, release_budget = _init_budget(ctx.root, ctx.max_minutes)
        armed.callback(release_budget)
        # A signal mid-exit-sequence exits the process before the loop can cancel
        # the checks command it is running, so its process group is killed here.
        armed.callback(install_termination_cleanup(kill_live_process_groups))
        if mode == "plain":
            write_and_flush(sys.stderr, f"log: {abbreviate_home(ctx.log_path)}\n")

        prompt, observer, tracing = setup_tracing(
            session_id=ctx.launch.session_id,
            branch=ctx.branch,
            launch_at=ctx.launch.at,
            head_sha=ctx.launch.head_sha,
            max_minutes=ctx.max_minutes,
            max_usd=ctx.max_usd,
            effort=ctx.effort,
            model=ctx.model,
            prompt=ctx.session_prompt(),
            reporter_observer=reporter.observer,
        )

        context = _supervised_session(ctx, deadline_ms)
        result: SupervisionResult | None = None
        try:
            try:
                result, exit_report = asyncio.run(
                    _supervise_and_exit_sequence(
                        reporter,
                        _supervise_then_release(
                            supervise(
                                driver=driver,
                                prompt=prompt,
                                context=context,
                                launch=ctx.launch,
                                observer=observer,
                            ),
                            release_budget,
                        ),
                        ctx=ctx,
                        context=context,
                        observer=observer,
                    )
                )
            finally:
                display.close()
            _report_result(
                result,
                ctx=ctx,
                session_result=reporter.session_result(),
                final_text=reporter.final_text(),
                exit_report=exit_report,
            )
        finally:
            if tracing.active:
                finalize_tracing(tracing, result)


# ---------------------------------------------------------------------------
# Command entry point
# ---------------------------------------------------------------------------


def _execute(options: Options) -> None:
    """Run the full supervised-session pipeline.

    Step order: doctor gate, working-tree guard, experiment-worktree guard,
    supervise lock, pre-flight (session under the repository lock, stop
    condition, baseline, feasibility), then log-path resolution, kickoff,
    launch event, and the session run.

    Args:
        options: The parsed ``supervise`` flags.
    """
    root = repo_root()
    doctor_gate(root, color=options.color)
    dirty_count = _validate_working_tree(root, allow_dirty=options.allow_dirty)
    validate_experiment_worktree(root)

    release = acquire_lock(supervise_lockfile_path(root), "supervise")
    try:
        resolved = resolve_config(CliFlags(), root)
        preflight = run_preflight(
            root=root,
            config=resolved,
            baseline_ref=options.baseline,
            max_minutes=options.max_minutes,
            force=options.force,
        )
        worktrees = preflight.session.worktrees

        log_path = _resolve_log_path(root, options.log)
        probe_event_log_path(log_path)
        kickoff = compose_kickoff(
            resolved,
            options.prompt,
            experiment_worktree=worktrees.experiment,
        )
        head_sha = run_git(["rev-parse", "HEAD"], root).strip()

        supervise_config = (
            resolved.supervise if resolved.supervise is not None else SuperviseConfig()
        )
        model = options.model if options.model is not None else supervise_config.model
        effort = options.effort if options.effort is not None else supervise_config.effort

        launch = LaunchEvent(
            at=now_ns(),
            schema_version=1,
            session_id=preflight.session.session_id,
            head_sha=head_sha,
            dirty=DirtyInfo(file_count=dirty_count) if dirty_count > 0 else False,
            max_minutes=options.max_minutes,
            max_usd=options.max_usd,
            model=model,
            effort=effort,
            runbook_path=resolved.runbook or "",
            kickoff_summary=summarize(kickoff.kickoff),
        )

        _run_session(
            _SessionContext(
                root=root,
                log_path=log_path,
                launch=launch,
                kickoff=kickoff,
                config=resolved,
                max_minutes=options.max_minutes,
                max_usd=options.max_usd,
                max_iterations=resolved.stop.max_iterations if resolved.stop is not None else None,
                model=model,
                effort=effort,
                color=options.color,
                branch=preflight.session.branch,
                finalize=options.finalize,
            )
        )
    finally:
        release()


def supervise_command(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the option surface
    prompt: PromptArgument = None,
    *,
    max_minutes: MaxMinutesOption,
    max_usd: MaxUsdOption = None,
    log: LogOption = None,
    baseline: BaselineOption = None,
    model: ModelOption = None,
    effort: EffortOption = None,
    allow_dirty: AllowDirtyOption = False,
    force: ForceOption = False,
    no_finalize: NoFinalizeOption = False,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Run a supervised agent session with wall-clock and spend caps."""
    apply_debug(debug)
    apply_color_override(color)
    options = Options(
        prompt=prompt,
        max_minutes=max_minutes,
        max_usd=max_usd,
        log=log,
        baseline=baseline,
        model=model,
        effort=effort,
        allow_dirty=allow_dirty,
        force=force,
        color=color,
        finalize=not no_finalize,
    )
    try:
        _execute(options)
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 -- CLI boundary: route any failure through the formatter
        exit_with_error(error)
