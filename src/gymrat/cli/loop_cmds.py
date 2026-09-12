"""The optimization-loop subcommands: start, iterate, keep, discard, finalize, stop, status, sync.

Each command resolves configuration at the repository root and holds the
single-flight lock for the duration. ``discard`` prompts before taking the lock
so the repository is not held hostage to a reader who never answers; the session
id from prompt time guards the locked revert. ``iterate`` routes SIGINT/SIGTERM
into an abort event so an interrupted iteration abandons the current sample.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from collections.abc import Callable

import typer

from gymrat.cli.iterate import IterateRenderer
from gymrat.cli.lock import GATE_EXIT_CODE
from gymrat.cli.shared import (
    AdapterOption,
    AllowUnimprovedOption,
    BenchOption,
    BranchOption,
    ColorOption,
    CommandTrace,
    ConfigOption,
    DebugOption,
    ForceOption,
    FormatOption,
    MessageOption,
    OutputFormat,
    PrepareOption,
    SamplesOption,
    TimeoutOption,
    VerboseOption,
    apply_color_override,
    apply_debug,
    broken_pipe_guard,
    budget_snapshot,
    config_trace_args,
    exit_with_error,
    is_tty,
    resolve_render_mode,
    resolve_stream_color,
    run_cli,
    run_with_signal_abort,
    with_repo_lock,
    write_and_flush,
)
from gymrat.config import CliFlags, resolve_benchless_config, resolve_config
from gymrat.confirm import confirm_action
from gymrat.loop.finalize import FinalizeOptions, FinalizeResult, finalize_session
from gymrat.loop.iterate import IterateOptions, IterateResult, LoopStopError, iterate_session
from gymrat.loop.settle import DiscardResult, KeepOptions, KeepResult, discard_session, keep_session
from gymrat.loop.start import StartResult, start_session
from gymrat.loop.status import status_data, status_session
from gymrat.loop.stop import StopResult, stop_session
from gymrat.loop.sync import SyncResult, sync_to_experiment
from gymrat.plural import pluralize
from gymrat.progress_events import create_fan_out
from gymrat.report.json_doc import (
    BudgetSummary,
    render_discard_json,
    render_iterate_json,
    render_iterate_stop_json,
    render_keep_json,
    render_status_json,
    render_stop_json,
)
from gymrat.report.loop import format_start_summary
from gymrat.session.paths import repo_root
from gymrat.session.progress_file import clear_progress, create_sidecar_writer
from gymrat.session.store import require_open_session
from gymrat.signals import install_termination_cleanup


def _write_budget_report(
    root: str,
    *,
    use_json: bool,
    render_json: Callable[[BudgetSummary | None], str],
    text_report: str,
) -> None:
    """Render the JSON or text report from a single budget read, then write it once."""
    trailer, summary = budget_snapshot(root)
    report = render_json(summary) if use_json else text_report + trailer
    write_and_flush(sys.stdout, report + "\n")


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------


def start(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    ref: str | None = typer.Argument(
        default=None, metavar="[REF]", help="ref the baseline is pinned to; defaults to HEAD"
    ),
    *,
    bench: BenchOption = None,
    prepare: PrepareOption = None,
    adapter: AdapterOption = None,
    samples: SamplesOption = None,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    debug: DebugOption = False,
) -> None:
    """Create or resume this repository's optimization session."""
    apply_debug(debug)

    flags = CliFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
    )

    start_args: dict[str, object] = config_trace_args(flags)
    if ref is not None:
        start_args["ref"] = ref

    async def run() -> None:
        async def body(_trace: CommandTrace) -> tuple[StartResult, str | None]:
            root = repo_root()
            resolved = resolve_config(flags, root)
            return start_session(root, ref, resolved), resolved.runbook

        result, runbook = await with_repo_lock("start", body, args=start_args)
        write_and_flush(sys.stdout, format_start_summary(result, runbook) + "\n")

    run_cli(run)


# ---------------------------------------------------------------------------
# Iterate
# ---------------------------------------------------------------------------


async def _iterate_body(
    trace: CommandTrace,
    flags: CliFlags,
    *,
    color: bool | None,
    verbose: bool,
    resolved_color: bool,
) -> IterateResult:
    root = repo_root()
    resolved = resolve_config(flags, root)
    required = require_open_session(root, "iterate")

    from gymrat.cli.console import stderr_console  # noqa: PLC0415 -- avoids circular import

    mode = resolve_render_mode()
    console = stderr_console(color_flag=color)
    seq = required.state.last_seq + 1
    # Pre-set to the unsettled seq so refusals carry it; success overwrites below.
    trace.seq = required.state.last_seq
    metric_count = len(resolved.metrics) if resolved.metrics is not None else 0
    renderer = IterateRenderer(
        mode,
        console,
        seq,
        required.session.session_id,
        resolved.samples,
        metric_count,
        resolved.primary,
        verbose=verbose,
        clock=time.perf_counter,
        checks_cmd=resolved.checks,
        has_before_hook=resolved.hooks is not None and resolved.hooks.before is not None,
        has_after_hook=resolved.hooks is not None and resolved.hooks.after is not None,
    )
    sidecar_writer = create_sidecar_writer(root)
    fan_out = create_fan_out([renderer.report, sidecar_writer])
    uninstall_progress_cleanup = install_termination_cleanup(lambda: clear_progress(root))
    try:
        result = await run_with_signal_abort(
            lambda abort: iterate_session(
                root,
                resolved,
                IterateOptions(abort=abort, on_progress=fan_out),
                color=resolved_color,
            )
        )
        trace.seq = result.record.seq
        return result
    finally:
        renderer.stop()
        uninstall_progress_cleanup()
        clear_progress(root)


def iterate(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    *,
    bench: BenchOption = None,
    prepare: PrepareOption = None,
    adapter: AdapterOption = None,
    samples: SamplesOption = None,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    color: ColorOption = None,
    verbose: VerboseOption = False,
    format: FormatOption = OutputFormat.text,  # noqa: A002 -- shadows builtin to match the CLI flag name
    debug: DebugOption = False,
) -> None:
    """Measure the session's experiment worktree against its baseline."""
    apply_debug(debug)
    color_override = apply_color_override(color)

    use_json = format == OutputFormat.json
    resolved_color = resolve_stream_color(color_override, sys.stdout)
    flags = CliFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
    )

    iterate_args = config_trace_args(flags)

    async def run() -> None:
        root = repo_root()
        try:
            result = await with_repo_lock(
                "iterate",
                lambda trace: _iterate_body(
                    trace, flags, color=color, verbose=verbose, resolved_color=resolved_color
                ),
                args=iterate_args,
            )
        except LoopStopError as error:
            trailer, summary = budget_snapshot(root)
            if use_json:
                write_and_flush(
                    sys.stdout,
                    render_iterate_stop_json(str(error), budget=summary) + "\n",
                )
                raise typer.Exit(GATE_EXIT_CODE) from None
            if trailer:
                write_and_flush(sys.stderr, trailer.lstrip("\n") + "\n")
            exit_with_error(error, GATE_EXIT_CODE)
        _write_budget_report(
            root,
            use_json=use_json,
            render_json=lambda summary: render_iterate_json(result, budget=summary),
            text_report=result.report,
        )

    run_cli(run)


# ---------------------------------------------------------------------------
# Keep
# ---------------------------------------------------------------------------


def keep(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    *,
    bench: BenchOption = None,
    prepare: PrepareOption = None,
    adapter: AdapterOption = None,
    samples: SamplesOption = None,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    message: MessageOption = None,
    allow_unimproved: AllowUnimprovedOption = False,
    format: FormatOption = OutputFormat.text,  # noqa: A002 -- shadows builtin to match the CLI flag name
    debug: DebugOption = False,
) -> None:
    """Commit the session's measured edit once its checks pass."""
    apply_debug(debug)

    use_json = format == OutputFormat.json
    resolved_color = resolve_stream_color(None, sys.stdout)
    flags = CliFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
    )

    keep_args: dict[str, object] = config_trace_args(flags)
    if message is not None:
        keep_args["message"] = message
    if allow_unimproved:
        keep_args["allow_unimproved"] = True

    async def run() -> None:
        async def body(trace: CommandTrace) -> KeepResult:
            root = repo_root()
            keep_result = await keep_session(
                root,
                resolve_benchless_config(flags, root),
                KeepOptions(message=message, allow_unimproved=allow_unimproved),
                color=resolved_color,
            )
            trace.seq = keep_result.record.seq
            if keep_result.record.status == "blocked":
                trace.gate = True
                trace.reason = keep_result.record.reason
            return keep_result

        result = await with_repo_lock("keep", body, args=keep_args)
        root = repo_root()
        _write_budget_report(
            root,
            use_json=use_json,
            render_json=lambda summary: render_keep_json(result, budget=summary),
            text_report=result.report,
        )
        if result.record.status == "blocked":
            raise typer.Exit(GATE_EXIT_CODE)

    run_cli(run)


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------


def discard(
    *,
    force: ForceOption = False,
    format: FormatOption = OutputFormat.text,  # noqa: A002 -- shadows builtin to match the CLI flag name
    debug: DebugOption = False,
) -> None:
    """Revert the session's experiment worktree to its last commit."""
    apply_debug(debug)

    use_json = format == OutputFormat.json

    async def run() -> None:
        root = repo_root()
        confirmed_session_id: str | None = None
        if is_tty(sys.stdin) and not force:
            required = require_open_session(root, "discard")
            confirmed_session_id = required.session.session_id
            confirmed = confirm_action(
                "discard will revert uncommitted changes in "
                f"{required.session.worktrees.experiment}.\nProceed?",
                sys.stdin,
            )
            if not confirmed:
                write_and_flush(sys.stderr, "discard cancelled\n")
                raise typer.Exit(GATE_EXIT_CODE)

        async def body(trace: CommandTrace) -> DiscardResult:
            discard_result = discard_session(root, confirmed_session_id)
            if discard_result.record is not None:
                trace.seq = discard_result.record.seq
            return discard_result

        result = await with_repo_lock("discard", body, args={"force": force})
        _write_budget_report(
            root,
            use_json=use_json,
            render_json=lambda summary: render_discard_json(result, budget=summary),
            text_report=result.report,
        )

    run_cli(run)


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


def finalize(
    *,
    message: MessageOption = None,
    branch: BranchOption = None,
    debug: DebugOption = False,
) -> None:
    """Collapse the session's kept iterations into one commit and close it."""
    apply_debug(debug)

    finalize_args: dict[str, object] = {}
    if branch is not None:
        finalize_args["branch"] = branch
    if message is not None:
        finalize_args["message"] = message

    async def run() -> None:
        async def body(_trace: CommandTrace) -> FinalizeResult:
            return finalize_session(repo_root(), FinalizeOptions(message=message, branch=branch))

        result = await with_repo_lock("finalize", body, args=finalize_args)
        write_and_flush(sys.stdout, result.report + "\n")

    run_cli(run)


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


_StopMessageOption = Annotated[
    str,
    typer.Option("--message", "-m", help="why the session is being stopped"),
]


def stop(
    *,
    message: _StopMessageOption,
    format: FormatOption = OutputFormat.text,  # noqa: A002 -- shadows builtin to match the CLI flag name
    debug: DebugOption = False,
) -> None:
    """Record a stop in the session log without reverting or committing."""
    apply_debug(debug)
    if not message.strip():
        msg = "message must not be empty"
        raise typer.BadParameter(msg)

    use_json = format == OutputFormat.json

    async def run() -> None:
        root = repo_root()

        async def body(_trace: CommandTrace) -> StopResult:
            return stop_session(root, message)

        result = await with_repo_lock("stop", body)
        _write_budget_report(
            root,
            use_json=use_json,
            render_json=lambda summary: render_stop_json(
                at=result.record.at, message=result.record.message, budget=summary
            ),
            text_report=result.report,
        )

    run_cli(run)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def status(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    *,
    bench: BenchOption = None,
    prepare: PrepareOption = None,
    adapter: AdapterOption = None,
    samples: SamplesOption = None,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    format: FormatOption = OutputFormat.text,  # noqa: A002 -- shadows builtin to match the CLI flag name
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Show this repository's session history, read from its log."""
    apply_debug(debug)
    color_override = apply_color_override(color)

    use_json = format == OutputFormat.json
    resolved_color = resolve_stream_color(color_override, sys.stdout)
    flags = CliFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
    )

    async def run() -> None:
        async def body(_trace: CommandTrace) -> str:
            root = repo_root()
            trailer, summary = budget_snapshot(root)
            if use_json:
                return render_status_json(status_data(root), budget=summary)
            return (
                status_session(root, resolve_benchless_config(flags, root), color=resolved_color)
                + trailer
            )

        report = await with_repo_lock("status", body)
        with broken_pipe_guard():
            write_and_flush(sys.stdout, report + "\n")

    run_cli(run)


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def sync(*, debug: DebugOption = False) -> None:
    """Sync uncommitted main-tree changes into the experiment worktree."""
    apply_debug(debug)

    async def run() -> None:
        async def body(_trace: CommandTrace) -> SyncResult:
            return sync_to_experiment(repo_root())

        result = await with_repo_lock("sync", body)
        if not result.files:
            summary = "nothing to sync"
        else:
            header = f"Synced {pluralize(len(result.files), 'file')} to experiment worktree:"
            summary = "\n".join([header, *(f"  {f}" for f in result.files)])
        trailer, _ = budget_snapshot(repo_root())
        write_and_flush(sys.stdout, summary + trailer + "\n")

    run_cli(run)
