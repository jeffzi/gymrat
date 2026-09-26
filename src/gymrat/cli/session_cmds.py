"""The session-lifecycle subcommands: start, stop, finalize, sync.

Each command resolves configuration at the repository root and holds the
single-flight lock for the duration.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer

from gymrat.cli.lock import CommandTrace, config_trace_args, with_repo_lock
from gymrat.cli.shared import (
    AdapterOption,
    BaselineOption,
    BenchOption,
    BranchOption,
    ColorOption,
    ConfigOption,
    DebugOption,
    FormatOption,
    OutputFormat,
    PrepareOption,
    SamplesOption,
    TimeoutOption,
    apply_color_override,
    apply_debug,
    budget_snapshot,
    run_cli,
    write_and_flush,
    write_budget_report,
)
from gymrat.config import CliFlags, resolve_config
from gymrat.loop.finalize import FinalizeOptions, FinalizeResult, finalize_session
from gymrat.loop.start import StartResult, start_session
from gymrat.loop.stop import StopResult, stop_session
from gymrat.loop.sync import SyncResult, sync_to_experiment
from gymrat.plural import pluralize
from gymrat.report.json_doc import (
    render_finalize_json,
    render_start_json,
    render_stop_json,
    render_sync_json,
)
from gymrat.report.loop import format_start_summary
from gymrat.session.paths import repo_root

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
    apply_debug(debug)
    apply_color_override(color)

    flags = CliFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
    )

    start_args: dict[str, object] = config_trace_args(flags)
    if baseline is not None:
        start_args["baseline"] = baseline

    use_json = output_format == OutputFormat.json

    async def run() -> None:
        async def body(_trace: CommandTrace) -> tuple[StartResult, str | None]:
            root = repo_root()
            resolved = resolve_config(flags, root)
            return start_session(root, baseline, resolved), resolved.runbook

        result, runbook = await with_repo_lock("start", body, args=start_args)
        if use_json:
            _, summary = budget_snapshot(repo_root())
            write_and_flush(
                sys.stdout,
                render_start_json(result, runbook=runbook, budget=summary) + "\n",
            )
        else:
            write_and_flush(sys.stdout, format_start_summary(result, runbook) + "\n")

    run_cli(run)


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


def finalize(
    *,
    message: Annotated[
        str | None,
        typer.Option("--message", "-m", help="message for the squash commit"),
    ] = None,
    branch: BranchOption = None,
    output_format: FormatOption = OutputFormat.text,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Collapse the session's kept iterations into one commit and close it."""
    apply_debug(debug)
    apply_color_override(color)

    finalize_args: dict[str, object] = {}
    if branch is not None:
        finalize_args["branch"] = branch
    if message is not None:
        finalize_args["message"] = message

    use_json = output_format == OutputFormat.json

    async def run() -> None:
        async def body(_trace: CommandTrace) -> FinalizeResult:
            return finalize_session(repo_root(), FinalizeOptions(message=message, branch=branch))

        result = await with_repo_lock("finalize", body, args=finalize_args)
        if use_json:
            _, summary = budget_snapshot(repo_root())
            write_and_flush(
                sys.stdout,
                render_finalize_json(result, budget=summary) + "\n",
            )
        else:
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
    output_format: FormatOption = OutputFormat.text,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Record a stop in the session log without reverting or committing."""
    apply_debug(debug)
    apply_color_override(color)
    if not message.strip():
        msg = "message must not be empty"
        raise typer.BadParameter(msg)

    use_json = output_format == OutputFormat.json

    async def run() -> None:
        root = repo_root()

        async def body(_trace: CommandTrace) -> StopResult:
            return stop_session(root, message)

        result = await with_repo_lock("stop", body)
        write_budget_report(
            root,
            use_json=use_json,
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
    apply_debug(debug)
    apply_color_override(color)

    use_json = output_format == OutputFormat.json

    async def run() -> None:
        async def body(_trace: CommandTrace) -> SyncResult:
            return sync_to_experiment(repo_root())

        result = await with_repo_lock("sync", body)
        if use_json:
            write_budget_report(
                repo_root(),
                use_json=True,
                render_json=lambda summary: render_sync_json(result, budget=summary),
                text_report="",
            )
        else:
            if not result.files:
                summary = "nothing to sync"
            else:
                header = f"Synced {pluralize(len(result.files), 'file')} to experiment worktree:"
                summary = "\n".join([header, *(f"  {f}" for f in result.files)])
            trailer, _ = budget_snapshot(repo_root())
            write_and_flush(sys.stdout, summary + trailer + "\n")

    run_cli(run)
