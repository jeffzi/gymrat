"""The ``gymrat doctor`` command: probe the project setup and report problems.

Doctor validates the project's configuration — environment, config file, workflow
keys, bench command, and adapter — without running any benchmarks. Any check
failure exits 1; an unexpected crash exits 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from gymrat.cli.budget_report import wants_json
from gymrat.cli.console import apply_command_flags, resolve_stream_color
from gymrat.cli.exit import run_guarded, write_stdout
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
from gymrat.cli.run_setup import SharedFlags
from gymrat.doctor import (
    build_doctor_report,
    render_doctor_json,
    render_doctor_report,
)
from gymrat.errors import GATE_EXIT_CODE


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
    """Run all doctor checks and exit ``GATE_EXIT_CODE`` if any check fails."""
    apply_command_flags(debug=debug, color=color)
    flags = SharedFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
        format=output_format.value,
    )

    def run() -> None:
        report = build_doctor_report(flags, cwd=str(Path.cwd()))

        if wants_json(flags):
            output = render_doctor_json(report)
        else:
            resolved_color = resolve_stream_color(None, sys.stdout)
            output = render_doctor_report(report, color=resolved_color)
        write_stdout(output + "\n")

        if report.has_failures:
            raise typer.Exit(GATE_EXIT_CODE)

    run_guarded(run)
