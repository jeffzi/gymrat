"""The ``gymrat doctor`` command: probe the project setup and report problems.

Doctor validates the project's configuration — environment, config file, workflow
keys, bench command, and adapter — without running any benchmarks. Any check
failure exits 1; an unexpected crash exits 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from gymrat.cli.shared import (
    GATE_EXIT_CODE,
    AdapterOption,
    BenchOption,
    ColorOption,
    ConfigOption,
    DebugOption,
    FormatOption,
    OutputFormat,
    PrepareOption,
    SamplesOption,
    SharedFlags,
    TimeoutOption,
    apply_color_override,
    apply_debug,
    resolve_stream_color,
    run_cli,
    wants_json,
    write_and_flush,
)
from gymrat.doctor.render import (
    render_doctor_json,
    render_doctor_report,
)
from gymrat.doctor.report import (
    build_doctor_report,
)


def doctor_command(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    *,
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
    """Check the project setup and report any problems."""
    apply_debug(debug)
    color_override = apply_color_override(color)
    flags = SharedFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
        color=color,
        format=output_format.value,
    )

    async def run() -> None:
        report = build_doctor_report(flags, cwd=str(Path.cwd()))

        if wants_json(flags):
            write_and_flush(sys.stdout, render_doctor_json(report) + "\n")
        else:
            resolved_color = resolve_stream_color(color_override, sys.stdout)
            output = render_doctor_report(report, color=resolved_color)
            write_and_flush(sys.stdout, output + "\n")

        if report.has_failures:
            raise typer.Exit(GATE_EXIT_CODE)

    run_cli(run)
