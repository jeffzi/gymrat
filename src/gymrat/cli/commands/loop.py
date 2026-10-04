"""The optimization-loop subcommands: iterate, keep, discard, status.

Each command resolves configuration at the repository root and holds the
single-flight lock for the duration. ``discard`` prompts before taking the lock
so the repository is not held hostage to a reader who never answers; the session
id from prompt time guards the locked revert. ``iterate`` routes SIGINT/SIGTERM
into an abort event so an interrupted iteration abandons the current sample.
"""

from __future__ import annotations

import sys
import traceback
from typing import TYPE_CHECKING, Annotated

import typer
from rich.prompt import Confirm
from rich.text import Text

from gymrat import clock as _clock
from gymrat.cli.budget_report import budget_snapshot, write_budget_report
from gymrat.cli.console import (
    apply_command_flags,
    is_debug_mode,
    resolve_stream_color,
    stderr_console,
)
from gymrat.cli.exit import exit_with_error, run_cli, write_and_flush, write_stdout
from gymrat.cli.iterate.progress import IterateRenderer
from gymrat.cli.options import (
    AdapterOption,
    BenchOption,
    ColorOption,
    ConfigOption,
    DebugOption,
    FormatOption,
    OutputFormat,
    PrepareOption,
    SamplesOption,
    TimeoutOption,
)
from gymrat.cli.run_setup import resolve_render_mode, run_with_signal_abort
from gymrat.cli.supervised import guard_supervised_origin
from gymrat.command_run import CommandTrace, with_repo_lock
from gymrat.config import CliFlags, config_trace_args, resolve_benchless_config, resolve_config
from gymrat.errors import GATE_EXIT_CODE
from gymrat.loop.discard import DiscardResult, discard_session
from gymrat.loop.iterate.run import IterateOptions, IterateResult, LoopStopError, iterate_session
from gymrat.loop.keep import KeepOptions, KeepResult, keep_session
from gymrat.loop.status import status_data, status_session
from gymrat.report.json_doc import (
    render_discard_json,
    render_iterate_json,
    render_iterate_stop_json,
    render_keep_json,
    render_status_json,
)
from gymrat.session.paths import repo_root
from gymrat.session.progress_file import SidecarWriter, clear_progress
from gymrat.session.store import require_open_session
from gymrat.signals import install_termination_cleanup
from gymrat.utils import MS_PER_SECOND, fan_out, is_tty

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.utils import WarnSink

_VerboseOption = Annotated[
    bool, typer.Option("--verbose", "-v", help="keep the progress tree visible after the run")
]
_AllowUnimprovedOption = Annotated[
    bool,
    typer.Option(
        "--allow-unimproved",
        help="keep the edit even when the iteration was not improved",
    ),
]
ForceOption = Annotated[bool, typer.Option("--force", "-f", help="skip the confirmation prompt")]
"""--force/-f: skip the confirmation prompt."""

# ---------------------------------------------------------------------------
# Iterate
# ---------------------------------------------------------------------------


def _subscriber_failure_sink(warn: WarnSink) -> Callable[[Exception], None]:
    # fan_out hands over only the exception, never the subscriber that raised
    # it, so a failure recurring on every event is recognized by its text.
    reported: set[str] = set()

    def report(error: Exception) -> None:
        text = str(error)
        if text in reported:
            return
        reported.add(text)
        message = f"warning: {text}"
        if is_debug_mode():
            message += "\n" + "".join(traceback.format_exception(error)).rstrip()
        warn(message)

    return report


async def _iterate_body(
    trace: CommandTrace,
    flags: CliFlags,
    *,
    verbose: bool,
    resolved_color: bool,
) -> IterateResult:
    root = repo_root()
    guard_supervised_origin(root, "iterate")
    resolved = resolve_config(flags, root)
    required = require_open_session(root, "iterate")

    mode = resolve_render_mode()
    console = stderr_console()
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
        clock=lambda: _clock.monotonic_ms() / MS_PER_SECOND,
        checks_cmd=resolved.checks,
        has_before_hook=resolved.hooks is not None and resolved.hooks.before is not None,
        has_after_hook=resolved.hooks is not None and resolved.hooks.after is not None,
    )
    sidecar_writer = SidecarWriter(root)
    on_progress = fan_out(
        [renderer.report, sidecar_writer], _subscriber_failure_sink(renderer.warn)
    )
    uninstall_progress_cleanup = install_termination_cleanup(lambda: clear_progress(root))
    try:
        result = await run_with_signal_abort(
            lambda abort: iterate_session(
                root,
                resolved,
                IterateOptions(abort=abort, on_progress=on_progress, warn=renderer.warn),
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
    verbose: _VerboseOption = False,
    output_format: FormatOption = OutputFormat.text,
    debug: DebugOption = False,
) -> None:
    """Measure the session's experiment worktree against its baseline."""
    apply_command_flags(debug=debug, color=color)

    use_json = output_format == OutputFormat.json
    resolved_color = resolve_stream_color(None, sys.stdout)
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
                    trace, flags, verbose=verbose, resolved_color=resolved_color
                ),
                args=iterate_args,
            )
        except LoopStopError as error:
            trailer, summary = budget_snapshot(root)
            if use_json:
                write_stdout(render_iterate_stop_json(str(error), budget=summary) + "\n")
                raise typer.Exit(GATE_EXIT_CODE) from None
            if trailer:
                write_and_flush(sys.stderr, trailer.lstrip("\n") + "\n")
            exit_with_error(error, GATE_EXIT_CODE)
        write_budget_report(
            root,
            use_json=use_json,
            render_json=lambda summary: render_iterate_json(result, budget=summary),
            text_report=result.report,
        )

    run_cli(run)


# ---------------------------------------------------------------------------
# Keep
# ---------------------------------------------------------------------------


def keep(  # noqa: PLR0913 -- one parameter per CLI flag
    *,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    message: Annotated[
        str | None,
        typer.Option("--message", "-m", help="commit message for the kept edit"),
    ] = None,
    allow_unimproved: _AllowUnimprovedOption = False,
    output_format: FormatOption = OutputFormat.text,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Commit the session's measured edit once its checks pass."""
    apply_command_flags(debug=debug, color=color)

    use_json = output_format == OutputFormat.json
    resolved_color = resolve_stream_color(None, sys.stdout)
    flags = CliFlags(config=config, timeout=timeout)

    # A false flag stays out of the record, as an unset one does.
    keep_args = config_trace_args(flags, message=message, allow_unimproved=allow_unimproved or None)

    async def run() -> None:
        async def body(trace: CommandTrace) -> KeepResult:
            root = repo_root()
            keep_result = await keep_session(
                root,
                resolve_benchless_config(flags, root),
                KeepOptions(
                    message=message,
                    allow_unimproved=allow_unimproved,
                    warn_color=resolve_stream_color(None, sys.stderr),
                ),
                color=resolved_color,
            )
            trace.seq = keep_result.record.seq
            if keep_result.record.status == "blocked":
                trace.gate = True
                trace.reason = keep_result.record.reason
            return keep_result

        result = await with_repo_lock("keep", body, args=keep_args)
        root = repo_root()
        write_budget_report(
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


def _confirm_discard(worktree: str) -> bool:
    # A Text question skips markup and emoji parsing, so the path prints as typed.
    question = Text(
        f"discard will revert uncommitted changes in {worktree}.\nProceed?", style="prompt"
    )
    console = stderr_console()
    # Soft wrap hands line breaking to the terminal; rich would otherwise split or
    # crop a path longer than the console width.
    console.soft_wrap = True
    try:
        return Confirm.ask(question, console=console, default=False)
    except EOFError:
        return False


def discard(
    *,
    force: ForceOption = False,
    output_format: FormatOption = OutputFormat.text,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Revert the session's experiment worktree to its last commit."""
    apply_command_flags(debug=debug, color=color)

    use_json = output_format == OutputFormat.json

    async def run() -> None:
        root = repo_root()
        confirmed_session_id: str | None = None
        if is_tty(sys.stdin) and not force:
            required = require_open_session(root, "discard")
            confirmed_session_id = required.session.session_id
            # With stderr closed the question cannot be shown, so there is no
            # consent to act on — and no stream to report the decline on.
            if sys.stderr is None:
                raise typer.Exit(GATE_EXIT_CODE)
            if not _confirm_discard(required.session.worktrees.experiment):
                write_and_flush(sys.stderr, "discard cancelled\n")
                raise typer.Exit(GATE_EXIT_CODE)

        async def body(trace: CommandTrace) -> DiscardResult:
            discard_result = discard_session(root, confirmed_session_id)
            if discard_result.record is not None:
                trace.seq = discard_result.record.seq
            return discard_result

        result = await with_repo_lock("discard", body, args={"force": force})
        write_budget_report(
            root,
            use_json=use_json,
            render_json=lambda summary: render_discard_json(result, budget=summary),
            text_report=result.report,
        )

    run_cli(run)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def status(
    *,
    config: ConfigOption = None,
    output_format: FormatOption = OutputFormat.text,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Show this repository's session history, read from its log."""
    apply_command_flags(debug=debug, color=color)

    use_json = output_format == OutputFormat.json
    resolved_color = resolve_stream_color(None, sys.stdout)
    flags = CliFlags(config=config)

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
        write_stdout(report + "\n")

    run_cli(run)
