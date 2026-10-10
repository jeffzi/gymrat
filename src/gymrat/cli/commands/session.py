"""The session-lifecycle subcommands: start, stop, finalize, sync.

Each command resolves configuration at the repository root and holds the
single-flight lock for the duration.
"""

from __future__ import annotations

from typing import Annotated

import typer

from gymrat.cli.budget_report import write_budget_report
from gymrat.cli.console import apply_command_flags
from gymrat.cli.exit import run_cli
from gymrat.cli.options import (
    AdapterOption,
    BaselineOption,
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
from gymrat.cli.run_setup import SharedFlags
from gymrat.command_run import CommandTrace, with_repo_lock
from gymrat.config import config_trace_args, resolve_config
from gymrat.loop.finalize import FinalizeOptions, FinalizeResult, finalize_session
from gymrat.loop.start import StartResult, start_session
from gymrat.loop.stop import StopResult, stop_session
from gymrat.loop.sync import SyncResult, sync_to_experiment
from gymrat.report.json_doc import (
    render_finalize_json,
    render_start_json,
    render_stop_json,
    render_sync_json,
)
from gymrat.report.loop import format_start_summary
from gymrat.session.paths import repo_root
from gymrat.utils import pluralize

# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------


def start(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    *,
    baseline: BaselineOption = None,
    bench: BenchOption = None,
    prepare: PrepareOption = None,
    adapter: AdapterOption = None,
    samples: SamplesOption = None,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    output_format: FormatOption = OutputFormat.text,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Create or resume this repository's optimization session."""
    apply_command_flags(debug=debug, color=color)

    flags = SharedFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
        format=output_format,
    )

    start_args = config_trace_args(flags, baseline=baseline)

    async def run() -> None:
        root = repo_root()

        async def body(_trace: CommandTrace) -> tuple[StartResult, str | None]:
            resolved = resolve_config(flags, root)
            return start_session(root, baseline, resolved), resolved.runbook

        result, runbook = await with_repo_lock("start", body, args=start_args, root=root)
        write_budget_report(
            root,
            flags,
            render_json=lambda summary: render_start_json(result, runbook=runbook, budget=summary),
            text_report=format_start_summary(result, runbook),
        )

    run_cli(run)


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


_BranchOption = Annotated[
    str | None,
    typer.Option("--branch", help="branch to point at the squash commit (default: <branch>-final)"),
]


def finalize(
    *,
    message: Annotated[
        str | None,
        typer.Option("--message", "-m", help="message for the squash commit"),
    ] = None,
    branch: _BranchOption = None,
    output_format: FormatOption = OutputFormat.text,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Collapse the session's kept iterations into one commit and close it."""
    apply_command_flags(debug=debug, color=color)
    flags = SharedFlags(format=output_format)

    async def run() -> None:
        root = repo_root()

        async def body(_trace: CommandTrace) -> FinalizeResult:
            return finalize_session(root, FinalizeOptions(message=message, branch=branch))

        result = await with_repo_lock(
            "finalize",
            body,
            args=config_trace_args(flags, branch=branch, message=message),
            root=root,
        )
        write_budget_report(
            root,
            flags,
            render_json=lambda summary: render_finalize_json(result, budget=summary),
            text_report=result.report,
        )

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
    output_format: FormatOption = OutputFormat.text,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Record a stop in the session log without reverting or committing."""
    apply_command_flags(debug=debug, color=color)
    flags = SharedFlags(format=output_format)
    if not message.strip():
        msg = "message must not be empty"
        raise typer.BadParameter(msg)

    async def run() -> None:
        root = repo_root()

        async def body(_trace: CommandTrace) -> StopResult:
            return stop_session(root, message)

        result = await with_repo_lock("stop", body, args=config_trace_args(flags), root=root)
        write_budget_report(
            root,
            flags,
            render_json=lambda summary: render_stop_json(
                at=result.record.at, message=result.record.message, budget=summary
            ),
            text_report=result.report,
        )

    run_cli(run)


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def sync(
    *,
    output_format: FormatOption = OutputFormat.text,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Sync uncommitted main-tree changes into the experiment worktree."""
    apply_command_flags(debug=debug, color=color)
    flags = SharedFlags(format=output_format)

    async def run() -> None:
        root = repo_root()

        async def body(_trace: CommandTrace) -> SyncResult:
            return sync_to_experiment(root)

        result = await with_repo_lock("sync", body, args=config_trace_args(flags), root=root)
        if not result.files:
            text_report = "nothing to sync"
        else:
            header = f"Synced {pluralize(len(result.files), 'file')} to experiment worktree:"
            text_report = "\n".join([header, *(f"  {f}" for f in result.files)])
        write_budget_report(
            root,
            flags,
            render_json=lambda summary: render_sync_json(result, budget=summary),
            text_report=text_report,
        )

    run_cli(run)
