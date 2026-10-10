"""The ``gymrat supervise`` command: a supervised agent session under caps.

Takes the supervise lock, runs the pre-flight (which guards against a dirty
working tree), resolves the JSONL event log, and hands a driver, reporter, and
kickoff to the supervisor. Every run the supervisor returns from then passes
through the exit sequence, which settles
and (unless ``--no-finalize``) finalizes the session before the summary prints.
The agent SDK is imported lazily inside the driver so ``gymrat --help`` stays fast.
"""

from __future__ import annotations

import asyncio
import math
import re
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from gymrat.cli.supervise.progress import SuperviseReporter
    from gymrat.model import Direction
    from gymrat.session.store import ReadSessionResult
    from gymrat.supervisor.events import SessionObserver
    from gymrat.supervisor.supervise import SupervisionResult

from gymrat.adapters import get_adapter
from gymrat.cli.console import apply_command_flags, resolve_stream_color
from gymrat.cli.exit import exit_with_error, run_guarded, write_and_flush, write_stdout
from gymrat.cli.options import (  # noqa: TC001 -- typer resolves these annotations at runtime
    BaselineOption,
    ColorOption,
    DebugOption,
)
from gymrat.cli.run_setup import resolve_render_mode
from gymrat.cli.supervise.preflight import PreflightFlags, doctor_gate, run_preflight
from gymrat.cli.supervise.progress import create_supervise_reporter
from gymrat.cli.supervise.summary import build_summary
from gymrat.clock import now_ms, now_ns
from gymrat.config import (
    GEOMEAN_PRIMARY,
    MAX_TIMEOUT_SECONDS,
    CliFlags,
    Effort,
    ResolvedConfig,
    SuperviseConfig,
    resolve_config,
)
from gymrat.errors import GATE_EXIT_CODE, TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.exec import kill_live_process_groups
from gymrat.report.style import render_lines
from gymrat.sampling import resolve_metric_meta
from gymrat.session.budget import (
    Budget,
    clear_budget,
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
from gymrat.session.workspace import dirty_file_count, ensure_git_exclude, worktree_head
from gymrat.signals import install_termination_cleanup
from gymrat.supervisor.claude import create_claude_driver
from gymrat.supervisor.driver import SessionPrompt
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
from gymrat.supervisor.supervise import SupervisedSession, supervise
from gymrat.supervisor.tools import gymrat_tools_factory
from gymrat.utils import ASCII_DECIMAL_PATTERN, SECONDS_PER_MINUTE, abbreviate_home, minutes_to_ms

# ---------------------------------------------------------------------------
# Flag surface
# ---------------------------------------------------------------------------

_POSITIVE_NUMBER_RE = re.compile(ASCII_DECIMAL_PATTERN)
_POSITIVE_NUMBER_MESSAGE = "must be a positive number."


def parse_positive_number(value: str) -> float:
    """Parse a strictly positive finite decimal.

    Args:
        value: The raw flag value.

    Returns:
        The parsed number.

    Raises:
        typer.BadParameter: When the value is not a bare decimal written in
            ASCII digits, or is zero or not finite.
    """
    if _POSITIVE_NUMBER_RE.fullmatch(value) is None:
        raise typer.BadParameter(_POSITIVE_NUMBER_MESSAGE)
    parsed = float(value)
    if parsed <= 0 or not math.isfinite(parsed):
        raise typer.BadParameter(_POSITIVE_NUMBER_MESSAGE)
    return parsed


def parse_max_minutes(value: str) -> float:
    """Parse a positive number of minutes bounded by the 32-bit timer ceiling.

    Args:
        value: The raw flag value.

    Returns:
        The parsed number of minutes.

    Raises:
        typer.BadParameter: When the value is not a positive number, or is above
            the ceiling.
    """
    parsed = parse_positive_number(value)
    max_minutes = MAX_TIMEOUT_SECONDS // SECONDS_PER_MINUTE
    if parsed > max_minutes:
        message = f"must be at most {max_minutes} minutes."
        raise typer.BadParameter(message)
    return parsed


_PromptArgument = Annotated[
    str | None,
    typer.Argument(metavar="[PROMPT]", help="optimization prompt for the agent"),
]
_MaxMinutesOption = Annotated[
    float,
    typer.Option(
        "--max-minutes",
        parser=parse_max_minutes,
        metavar="<float>",
        help="wall-clock cap in minutes, counted from when the baseline is recorded",
    ),
]
_MaxUsdOption = Annotated[
    float | None,
    typer.Option(
        "--max-usd", parser=parse_positive_number, metavar="<float>", help="spend cap in USD"
    ),
]
_LogOption = Annotated[str | None, typer.Option("--log", help="path for the JSONL event log")]
_ModelOption = Annotated[
    str | None, typer.Option("--model", help="model to use for the agent session")
]
_AllowDirtyOption = Annotated[
    bool, typer.Option("--allow-dirty", help="allow launching with uncommitted changes")
]
_ForceOption = Annotated[
    bool,
    typer.Option(
        "--force",
        help="launch even when the cap cannot fit one iteration or a stop condition is already met",
    ),
]
_NoFinalizeOption = Annotated[
    bool,
    typer.Option("--no-finalize", help="leave the session open instead of finalizing it on exit"),
]
_EffortOption = Annotated[
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


def _resolve_log_path(root: str, explicit: str | None) -> str:
    """Resolve where the supervisor writes its event log.

    Only the default path is written under ``.gymrat/``, so only that branch
    ensures the directory is git-excluded; a caller-supplied path is left to the
    caller to place and ignore.

    Args:
        root: The repository root path.
        explicit: The caller's ``--log`` value, or ``None`` to use the default path.

    Returns:
        The caller's ``--log`` path verbatim, or a timestamped path under the
        session directory.
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
    max_iterations: int | None
    color: bool | None
    branch: str
    resumed: bool
    finalize: bool

    def session_prompt(self) -> SessionPrompt:
        return SessionPrompt(
            kickoff=self.kickoff.kickoff,
            cwd=self.root,
            system_prompt_append=self.kickoff.system_prompt_append,
            model=self.launch.model,
            effort=self.launch.effort,
            command_timeout_ms=minutes_to_ms(self.launch.max_minutes),
            max_budget_usd=self.launch.max_usd,
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
            model=ctx.launch.model,
            effort=ctx.launch.effort,
            exit_report=exit_report,
        ),
        color=resolve_stream_color(ctx.color, sys.stdout),
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
    deadline_ms = now_ms() + minutes_to_ms(max_minutes)
    budget = Budget(max_minutes=max_minutes, deadline_ms=deadline_ms)
    Path(session_dir(root)).mkdir(parents=True, exist_ok=True)
    write_budget(root, budget)
    release = ExitStack()
    # Callbacks unwind last-in first-out: the budget file goes, then the hook.
    release.callback(install_termination_cleanup(lambda: clear_budget(root)))
    release.callback(clear_budget, root)
    return deadline_ms, release.close


def _primary_direction(config: ResolvedConfig) -> Direction:
    """Whether a lower or a higher value of the configured primary is the better outcome."""
    if config.primary == GEOMEAN_PRIMARY:
        return "lower"
    entry = config.metrics.get(config.primary) if config.metrics is not None else None
    return resolve_metric_meta(
        config.primary, entry, get_adapter(config.adapter), config.kinds
    ).direction


def _create_reporter(ctx: _SessionContext, mode: Literal["live", "plain"]) -> SuperviseReporter:
    return create_supervise_reporter(
        primary_direction=_primary_direction(ctx.config),
        root=ctx.root,
        max_minutes=ctx.launch.max_minutes,
        max_usd=ctx.launch.max_usd,
        max_iterations=ctx.max_iterations,
        mode=mode,
        log_path=ctx.log_path,
        color=ctx.color,
        model=ctx.launch.model,
        effort=ctx.launch.effort,
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
        max_minutes=ctx.launch.max_minutes,
        max_usd=ctx.launch.max_usd,
    )


async def _supervise_and_exit_sequence(  # noqa: PLR0913 -- the run and everything the exit sequence shares with it
    reporter: SuperviseReporter,
    pending: Coroutine[object, object, SupervisionResult],
    release_budget: Callable[[], None],
    *,
    ctx: _SessionContext,
    context: SupervisedSession,
    observer: SessionObserver,
) -> tuple[SupervisionResult, ExitReport]:
    """Run the supervisor, then the exit sequence, inside one running loop.

    The budget is released the moment the supervisor stops, so the exit
    sequence never runs under a live budget. The exit sequence runs only when
    the supervisor returned; its phases and warnings go through the
    still-running reporter, and its decisions land in the same event log and
    observer the supervisor wrote to.

    Args:
        reporter: The dashboard, already showing when the supervisor is awaited.
        pending: The supervisor run.
        release_budget: Removes the budget file and its termination cleanup.
        ctx: The session run's context.
        context: The supervised session the supervisor and exit sequence share.
        observer: The observer the supervisor was handed.

    Returns:
        What the supervisor returned and what the exit sequence did.

    Raises:
        Exception: Whatever the supervisor raised, unchanged.
    """
    try:
        result = await pending
    finally:
        release_budget()
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
    from gymrat.telemetry.run_spans import (  # noqa: PLC0415 -- lazy import keeps CLI startup off the telemetry stack
        finalize_tracing,
        setup_tracing,
    )

    launch = ctx.launch
    driver = create_claude_driver(
        tools=gymrat_tools_factory(ctx.root), hooks=supervise_hooks_factory(Path(ctx.root))
    )
    mode = resolve_render_mode()
    reporter = _create_reporter(ctx, mode)

    with ExitStack() as armed:
        # Stops the display exactly once: here when setup below fails, before the
        # error reaches the terminal, or when the run path closes it early.
        display = armed.enter_context(ExitStack())
        display.callback(reporter.stop)
        deadline_ms, release_budget = _init_budget(ctx.root, launch.max_minutes)
        armed.callback(release_budget)
        # A signal mid-exit-sequence exits the process before the loop can cancel
        # the checks command it is running, so its process group is killed here.
        armed.callback(install_termination_cleanup(kill_live_process_groups))
        if mode == "plain":
            write_and_flush(sys.stderr, f"log: {abbreviate_home(ctx.log_path)}\n")

        prompt, observer, tracing = setup_tracing(
            launch,
            branch=ctx.branch,
            resumed=ctx.resumed,
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
                        supervise(
                            driver=driver,
                            prompt=prompt,
                            context=context,
                            launch=launch,
                            observer=observer,
                        ),
                        release_budget,
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
            if tracing.run_span is not None:
                finalize_tracing(tracing, result)


# ---------------------------------------------------------------------------
# Command entry point
# ---------------------------------------------------------------------------


def _execute(options: Options) -> None:
    """Run the full supervised-session pipeline.

    Step order: doctor gate, supervise lock, pre-flight (under the repository
    lock: working-tree guard, experiment-worktree guard, session, stop
    condition, baseline, feasibility), then log-path resolution, kickoff,
    launch event, and the session run.

    Args:
        options: The parsed ``supervise`` flags.
    """
    root = repo_root()
    doctor_gate(root, color=options.color)

    release = acquire_lock(supervise_lockfile_path(root), "supervise")
    try:
        resolved = resolve_config(CliFlags(), root)
        preflight = run_preflight(
            root=root,
            config=resolved,
            flags=PreflightFlags(
                baseline_ref=options.baseline,
                max_minutes=options.max_minutes,
                force=options.force,
                allow_dirty=options.allow_dirty,
            ),
        )
        dirty_count = dirty_file_count(root)

        log_path = _resolve_log_path(root, options.log)
        probe_event_log_path(log_path)
        kickoff = compose_kickoff(
            resolved,
            options.prompt,
            experiment_worktree=preflight.session.worktrees.experiment,
        )
        head_sha = worktree_head(root)

        supervise_config = resolved.supervise or SuperviseConfig()
        model = options.model if options.model is not None else supervise_config.model
        effort = options.effort if options.effort is not None else supervise_config.effort
        max_iterations = resolved.stop.max_iterations if resolved.stop is not None else None

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
                max_iterations=max_iterations,
                color=options.color,
                branch=preflight.session.branch,
                resumed=preflight.resumed,
                finalize=options.finalize,
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
    baseline: BaselineOption = None,
    model: _ModelOption = None,
    effort: _EffortOption = None,
    allow_dirty: _AllowDirtyOption = False,
    force: _ForceOption = False,
    no_finalize: _NoFinalizeOption = False,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Run a supervised agent session with wall-clock and spend caps."""
    apply_command_flags(debug=debug, color=color)
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
    run_guarded(lambda: _execute(options))
