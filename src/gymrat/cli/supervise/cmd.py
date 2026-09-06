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
from typing import TYPE_CHECKING, Annotated, cast

import typer

if TYPE_CHECKING:
    from collections.abc import Callable

from gymrat.cli.shared import (
    GATE_EXIT_CODE,
    TOOL_FAILURE_EXIT_CODE,
    ColorOption,
    DebugOption,
    apply_color_override,
    apply_debug,
    exit_with_error,
    parse_max_minutes,
    parse_positive_number,
    resolve_render_mode,
    resolve_stream_color,
    write_and_flush,
)
from gymrat.cli.supervise.frame import SessionLabels, build_summary
from gymrat.cli.supervise.preflight import doctor_gate, run_preflight
from gymrat.cli.supervise.progress import ReadSessionResult, create_supervise_reporter
from gymrat.config import (
    EFFORT_LEVELS,
    EFFORT_PHRASE,
    CliFlags,
    Effort,
    ResolvedConfig,
    SuperviseConfig,
    resolve_config,
)
from gymrat.errors import GymratError
from gymrat.git import run_git
from gymrat.paths import abbreviate_home
from gymrat.plural import pluralize
from gymrat.report.style import RENDER_WIDTH, render_lines
from gymrat.session.budget import (
    Budget,
    clear_budget,
    minutes_to_ms,
    write_budget,
)
from gymrat.session.clock import now_ms
from gymrat.session.lock import acquire_lock
from gymrat.session.paths import (
    experiment_worktree_dir,
    lockfile_path,
    repo_root,
    session_dir,
    session_jsonl_path,
    supervise_lockfile_path,
)
from gymrat.session.store import fold_session, last_kept_position, read_records
from gymrat.session.workspace import changed_file_count, dirty_file_count, ensure_git_exclude
from gymrat.signals import install_termination_cleanup
from gymrat.supervisor import (
    KickoffResult,
    SessionPrompt,
    SupervisedSession,
    SupervisionResult,
    compose_kickoff,
    create_claude_driver,
    supervise,
)
from gymrat.supervisor.event_log import probe_event_log_path
from gymrat.supervisor.events import DirtyInfo, LaunchEvent, summarize

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
_BaselineOption = Annotated[
    str | None,
    typer.Option(
        "--baseline",
        metavar="<ref>",
        help=(
            "git ref that pins a freshly opened session; "
            "defaults to HEAD and is ignored when a session is resumed"
        ),
    ),
]


def _parse_effort(value: str) -> Effort:
    if value not in EFFORT_LEVELS:
        raise typer.BadParameter(EFFORT_PHRASE)
    return cast("Effort", value)


_EffortOption = Annotated[
    Effort | None,
    typer.Option("--effort", parser=_parse_effort, metavar="<level>", help="effort level"),
]


def _validate_working_tree(root: str, *, allow_dirty: bool) -> int:
    """Refuse a dirty tree unless ``allow_dirty`` was set, warning when it was."""
    count = dirty_file_count(root)
    if count == 0:
        return count

    if not allow_dirty:
        message = f"Working tree has {pluralize(count, 'uncommitted file')}."
        hint = "Commit or stash your changes, or pass --allow-dirty to proceed anyway."
        exit_with_error(GymratError(message, hint=hint))

    write_and_flush(
        sys.stderr,
        f"warning: working tree has {pluralize(count, 'dirty file')} — "
        "proceeding because --allow-dirty was set\n",
    )
    return count


def _validate_experiment_worktree(root: str) -> None:
    """Refuse to launch when the experiment worktree has unmeasured changes.

    An unsettled iteration needs settling first; unmeasured edits — committed or
    still uncommitted — need measuring or reverting. The check runs regardless of
    ``--allow-dirty``, which covers only the main working tree.
    """
    records = read_records(session_jsonl_path(root))
    if not records:
        return

    state = fold_session(records)
    if state.finalized is not None or state.session is None:
        return

    worktree = experiment_worktree_dir(root)
    target = last_kept_position(state, state.session.baseline.sha)
    count = changed_file_count(worktree, target)
    if count == 0:
        return

    if state.unsettled:
        message = "The experiment worktree has an unsettled iteration with uncommitted changes."
        hint = "Run gymrat keep or gymrat discard first."
    elif state.ends_on_gating_block:
        message = "The last keep was refused for a gating regression."
        hint = "Run gymrat discard to revert it."
    else:
        message = f"The experiment worktree has {pluralize(count, 'unmeasured edit')}."
        hint = "Measure them with gymrat iterate or revert them with gymrat discard."
    exit_with_error(GymratError(message, hint=hint))


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
    config: ResolvedConfig
    max_minutes: float
    max_usd: float | None
    max_iterations: int | None
    model: str | None
    effort: Effort | None
    color: bool | None


def _report_result(
    result: SupervisionResult,
    *,
    ctx: _SessionContext,
    session_result: ReadSessionResult | None,
    final_text: str | None,
) -> None:
    """Print the closing summary to stdout, then exit per how the run ended.

    A session that ended on its own returns normally (exit 0). A cap trip exits
    on the gate code. An error outcome exits on the tool-failure code, surfacing
    its message to stderr only when one is present.
    """
    summary = render_lines(
        build_summary(
            result,
            log_path=ctx.log_path,
            session_result=session_result,
            final_text=final_text,
            labels=SessionLabels(model=ctx.model, effort=ctx.effort),
        ),
        color=resolve_stream_color(ctx.color, sys.stdout),
        width=RENDER_WIDTH,
    )
    write_and_flush(sys.stdout, f"{summary}\n")

    if result.outcome.reason == "error":
        if result.outcome.message:
            exit_with_error(GymratError(result.outcome.message))
        raise typer.Exit(TOOL_FAILURE_EXIT_CODE)

    if result.ended_by != "session":
        raise typer.Exit(GATE_EXIT_CODE)


def _init_budget(root: str, max_minutes: float) -> tuple[float, Callable[[], None]]:
    """Create, persist, and arm cleanup for the session time budget.

    Returns the deadline in epoch milliseconds and a callback that
    removes the budget file and uninstalls the termination hook.
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
    return deadline_ms, uninstall


def _run_session(ctx: _SessionContext) -> None:
    """Drive the supervised session, reporting progress and stopping it cleanly."""
    driver = create_claude_driver()
    mode = resolve_render_mode()
    reporter = create_supervise_reporter(
        root=ctx.root,
        max_minutes=ctx.max_minutes,
        max_usd=ctx.max_usd,
        max_iterations=ctx.max_iterations,
        mode=mode,
        log_path=ctx.log_path,
        color=ctx.color,
        model=ctx.model,
        effort=ctx.effort,
    )
    uninstall_cleanup = install_termination_cleanup(reporter.stop)

    deadline_ms, uninstall_budget_cleanup = _init_budget(ctx.root, ctx.max_minutes)

    if mode == "plain":
        write_and_flush(sys.stderr, f"log: {abbreviate_home(ctx.log_path)}\n")

    prompt = SessionPrompt(
        kickoff=ctx.kickoff.kickoff,
        cwd=ctx.root,
        system_prompt_append=ctx.kickoff.system_prompt_append,
        model=ctx.model,
        effort=ctx.effort,
        command_timeout_ms=minutes_to_ms(ctx.max_minutes),
    )
    context = SupervisedSession(
        root=ctx.root,
        log_path=ctx.log_path,
        lock_path=lockfile_path(ctx.root),
        config=ctx.config,
        deadline_ms=deadline_ms,
        max_minutes=ctx.max_minutes,
        max_usd=ctx.max_usd,
    )
    try:
        result = asyncio.run(
            supervise(
                driver=driver,
                prompt=prompt,
                context=context,
                launch=ctx.launch,
                observer=reporter.observer,
            )
        )
    finally:
        clear_budget(ctx.root)
        reporter.stop()
        uninstall_cleanup()
        uninstall_budget_cleanup()
    _report_result(
        result,
        ctx=ctx,
        session_result=reporter.session_result(),
        final_text=reporter.final_text(),
    )


@dataclass(frozen=True, slots=True)
class _Options:
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


def _execute(options: _Options) -> None:
    """Run the full supervised-session pipeline.

    Step order: doctor gate, working-tree guard, experiment-worktree guard,
    supervise lock, pre-flight (session under the repository lock, stop
    condition, baseline, feasibility), then log-path resolution, kickoff,
    launch event, and the session run.
    """
    root = repo_root()
    doctor_gate(root, color=options.color)
    dirty_count = _validate_working_tree(root, allow_dirty=options.allow_dirty)
    _validate_experiment_worktree(root)

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
            timestamp=now_ms(),
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
    baseline: _BaselineOption = None,
    model: _ModelOption = None,
    effort: _EffortOption = None,
    allow_dirty: _AllowDirtyOption = False,
    force: _ForceOption = False,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Run a supervised agent session with wall-clock and spend caps."""
    apply_debug(debug)
    apply_color_override(color)
    options = _Options(
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
    )
    try:
        _execute(options)
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 -- CLI boundary: route any failure through the formatter
        exit_with_error(error)
