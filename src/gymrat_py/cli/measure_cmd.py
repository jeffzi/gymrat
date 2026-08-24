"""The ``gymrat measure`` command: one revision or directory, on its own.

The action holds the repository lock for the length of the run, stops the
progress reporter before any report or error text, and renders the measurement
to stdout once the lock is released. It never gates the exit code. The
measurement engine is imported inside the action so assembling the CLI never
pulls the heavy statistics stack.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer

from gymrat_py.cli.shared import (
    AdapterOption,
    BenchOption,
    ConfigOption,
    DebugOption,
    FormatOption,
    MeasureFlags,
    NoColorOption,
    OutputFormat,
    PrepareOption,
    ReportRenderers,
    SamplesOption,
    TimeoutOption,
    begin_run,
    color_override_of,
    emit_report,
    exit_with_error,
    parse_positional,
    run_options_of,
    set_debug_mode,
    with_repo_lock,
)
from gymrat_py.config import resolve_config
from gymrat_py.report import render_measure_json, render_measure_report
from gymrat_py.report.types import MeasurementResult, ReportOptions
from gymrat_py.sampling import TargetSpec

_TargetArgument = Annotated[
    TargetSpec | None,
    typer.Argument(
        parser=parse_positional,
        metavar="[TARGET]",
        help="[label=]<ref|dir> to measure; defaults to the current directory",
    ),
]


def measure(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    target: _TargetArgument = None,
    *,
    bench: BenchOption = None,
    prepare: PrepareOption = None,
    adapter: AdapterOption = None,
    samples: SamplesOption = None,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    output_format: FormatOption = OutputFormat.text,
    no_color: NoColorOption = False,
    debug: DebugOption = False,
) -> None:
    """Measure one revision or directory on its own, with nothing to compare it to."""
    set_debug_mode(debug)
    resolved_target = target if target is not None else TargetSpec(label=None, target=".")
    flags = MeasureFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
        color=not no_color,
        format=output_format.value,
    )
    color_override = color_override_of(flags.color)

    async def run() -> None:
        async def body() -> MeasurementResult:
            progress = begin_run(flags, 1)
            try:
                config_resolved = resolve_config(flags)
                # Lazy: keep the heavy statistics stack out of CLI assembly and --help.
                from gymrat_py import measure as engine  # noqa: PLC0415

                run_opts = run_options_of(config_resolved, progress)
                options = engine.MeasureOptions(
                    target=resolved_target,
                    bench=run_opts.bench,
                    prepare=run_opts.prepare,
                    adapter=run_opts.adapter,
                    samples=run_opts.samples,
                    timeout_seconds=run_opts.timeout_seconds,
                    config_metrics=run_opts.config_metrics,
                    config_kinds=run_opts.config_kinds,
                    on_progress=run_opts.on_progress,
                    warn=run_opts.warn,
                )
                return await engine.measure(options)
            finally:
                progress.stop()

        result = await with_repo_lock("measure", body)

        emit_report(
            result,
            flags,
            ReportRenderers(text=render_measure_report, json=render_measure_json),
            ReportOptions(color=color_override),
        )

    try:
        asyncio.run(run())
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 -- CLI boundary: route any failure through the formatter
        exit_with_error(error)
