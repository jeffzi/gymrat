"""The ``gymrat probe`` command: a spot check of the session's experiment worktree.

The action holds the repository lock for the length of the bench. Inside the
lock it refuses a shell-typed probe while a supervised run is live, then warns
about an over-budget run before the progress reporter starts. It stops the
progress reporter before any report or error text, and renders the probe to
stdout once the lock is released. It never gates the exit code.

The bench comes from ``gymrat.toml`` alone — a probe reads the session's own
configuration rather than taking a one-off bench, so there is no ``--bench``,
``--prepare``, ``--adapter``, or ``--timeout`` here.
"""

from __future__ import annotations

from typing import Annotated

import typer

from gymrat.cli.budget_report import budget_for_report, warn_duration_over_budget
from gymrat.cli.console import apply_color_override, apply_debug
from gymrat.cli.options import (
    ColorOption,
    ConfigOption,
    DebugOption,
    FormatOption,
    OutputFormat,
    SamplesOption,
)
from gymrat.cli.shared import (
    ReportRenderers,
    SharedFlags,
    begin_run,
    emit_report,
    run_cli,
    run_with_signal_abort,
)
from gymrat.cli.supervised import guard_supervised_origin
from gymrat.command_run import with_repo_lock
from gymrat.config.resolve import resolve_config
from gymrat.loop.probe import EXPERIMENT_LABEL, ProbeOptions, ProbeResult, probe_session
from gymrat.report.json_doc import render_probe_json
from gymrat.report.text.probe import render_probe_report
from gymrat.report.types import ReportOptions
from gymrat.session.paths import repo_root

_NamesArgument = Annotated[
    list[str] | None,
    typer.Argument(
        metavar="[NAMES]...",
        help="metric names to narrow the bench to, via the configured filter template",
    ),
]


async def _probe_body(flags: SharedFlags, names: list[str]) -> ProbeResult:
    root = repo_root()
    guard_supervised_origin(root, "probe")
    # Warn before the progress reporter starts, or the warning prints under a live display.
    warn_duration_over_budget(halve=True)
    progress = begin_run(flags, 1, command="probe", target_labels=[EXPERIMENT_LABEL])
    try:
        resolved = resolve_config(flags, root)
        # ProbeOptions takes no abort event, so the one handed out here goes
        # unused: the value of the wrapper is its cleanup, which kills the live
        # bench process group before the signal handler exits the process.
        return await run_with_signal_abort(
            lambda _abort: probe_session(
                root,
                resolved,
                ProbeOptions(
                    names=names,
                    samples=flags.samples,
                    on_progress=progress.report,
                    warn=progress.warn,
                ),
            )
        )
    finally:
        progress.stop()


def probe(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    names: _NamesArgument = None,
    *,
    samples: SamplesOption = None,
    config: ConfigOption = None,
    output_format: FormatOption = OutputFormat.text,
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Bench the session's experiment worktree against its newest recorded baseline."""
    apply_debug(debug)
    color_override = apply_color_override(color)
    probed = list(names or [])
    flags = SharedFlags(
        samples=samples,
        config=config,
        color=color,
        format=output_format.value,
    )

    async def run() -> None:
        result = await with_repo_lock(
            "probe",
            lambda _trace: _probe_body(flags, probed),
            args={"names": probed, "samples": samples},
        )
        budget_trailer, budget_summary = budget_for_report()
        emit_report(
            result,
            flags,
            ReportRenderers(text=render_probe_report, json=render_probe_json),
            ReportOptions(color=color_override),
            budget_trailer=budget_trailer,
            budget_summary=budget_summary,
        )

    run_cli(run)
