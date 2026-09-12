"""The ``gymrat probe`` command: a spot check of the session's experiment worktree.

The action warns about an over-budget run before it queues on the lock, holds the
repository lock for the length of the bench, stops the progress reporter before
any report or error text, and renders the probe to stdout once the lock is
released. It never gates the exit code.

The bench comes from ``gymrat.toml`` alone — a probe reads the session's own
configuration rather than taking a one-off bench, so there is no ``--bench``,
``--prepare``, ``--adapter``, or ``--timeout`` here.
"""

from __future__ import annotations

from typing import Annotated

import typer

from gymrat.cli.shared import (
    ColorOption,
    ConfigOption,
    DebugOption,
    FormatOption,
    OutputFormat,
    ReportRenderers,
    SamplesOption,
    SharedFlags,
    apply_color_override,
    apply_debug,
    begin_run,
    budget_for_report,
    emit_report,
    run_cli,
    run_with_signal_abort,
    warn_duration_over_budget,
    with_repo_lock,
)
from gymrat.config import resolve_config
from gymrat.loop.probe import ProbeOptions, ProbeResult, probe_session
from gymrat.report import render_probe_json, render_probe_report
from gymrat.report.types import ReportOptions
from gymrat.session.paths import repo_root

_NamesArgument = Annotated[
    list[str] | None,
    typer.Argument(
        metavar="[NAMES]...",
        help="metric names to narrow the bench to, via the configured filter template",
    ),
]

#: The display label of the worktree a probe benches, as the progress header names it.
_EXPERIMENT_LABEL = "experiment"


async def _probe_body(flags: SharedFlags, names: list[str]) -> ProbeResult:
    progress = begin_run(flags, 1, command="probe", target_labels=[_EXPERIMENT_LABEL])
    try:
        root = repo_root()
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
        warn_duration_over_budget(halve=True)
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
