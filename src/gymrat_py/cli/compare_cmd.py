"""The ``gymrat compare`` command: one baseline against one or more candidates.

The action holds the repository lock for the length of the run, stops the
progress reporter before any report or error text, renders the result to stdout
once the lock is released, and gates the exit code on the ``--fail-on``
conditions. The comparison engine is imported inside the action so assembling the
CLI never pulls the heavy statistics stack.
"""

from __future__ import annotations

from typing import Annotated

import typer

from gymrat_py.cli.gating import should_fail_gate, warn_empty_geomean_gates
from gymrat_py.cli.shared import (
    GATE_EXIT_CODE,
    AdapterOption,
    BenchOption,
    CompareFlags,
    ConfigOption,
    DebugOption,
    FormatOption,
    NoColorOption,
    OutputFormat,
    PrepareOption,
    ReportRenderers,
    SamplesOption,
    TimeoutOption,
    begin_run,
    color_override_of,
    emit_report,
    parse_fail_on,
    parse_positional,
    run_cli,
    run_options_of,
    set_debug_mode,
    with_repo_lock,
)
from gymrat_py.config import resolve_config
from gymrat_py.report import render_json, render_report
from gymrat_py.report.types import ComparisonResult, FailOnCondition, ReportOptions
from gymrat_py.sampling import TargetSpec

_BaselineArgument = Annotated[
    TargetSpec,
    typer.Argument(
        parser=parse_positional,
        metavar="BASELINE",
        help="[label=]<ref|dir> to measure against",
    ),
]
_CandidatesArgument = Annotated[
    list[TargetSpec],
    typer.Argument(
        parser=parse_positional,
        metavar="CANDIDATES...",
        help="[label=]<ref|dir>, each judged against the baseline",
    ),
]
_VerboseOption = Annotated[
    bool, typer.Option("--verbose", help="name the statistical method behind each verdict")
]
_FailOnOption = Annotated[
    list[FailOnCondition] | None,
    typer.Option(
        "--fail-on",
        parser=parse_fail_on,
        help='exit 1 when a condition trips (repeatable: "regressed", "geomean:<pct>")',
    ),
]


def compare(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    baseline: _BaselineArgument,
    candidates: _CandidatesArgument,
    *,
    bench: BenchOption = None,
    prepare: PrepareOption = None,
    adapter: AdapterOption = None,
    samples: SamplesOption = None,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    output_format: FormatOption = OutputFormat.text,
    fail_on: _FailOnOption = None,
    verbose: _VerboseOption = False,
    no_color: NoColorOption = False,
    debug: DebugOption = False,
) -> None:
    """Run each candidate against the baseline and exit non-zero when --fail-on fires."""
    if debug:
        set_debug_mode(True)
    flags = CompareFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
        color=not no_color,
        format=output_format.value,
        verbose=verbose,
        fail_on=tuple(fail_on) if fail_on is not None else (),
    )
    color_override = color_override_of(flags.color)

    async def run() -> None:
        async def body() -> ComparisonResult:
            progress = begin_run(flags, 1 + len(candidates))
            try:
                config_resolved = resolve_config(flags)
                # Lazy: keep the heavy statistics stack out of CLI assembly and --help.
                from gymrat_py import compare as engine  # noqa: PLC0415

                run_opts = run_options_of(config_resolved, progress)
                options = engine.CompareOptions(
                    baseline=baseline,
                    candidates=candidates,
                    unstable_noise_pct=config_resolved.unstable_noise_pct,
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
                result = await engine.compare(options)
                progress.set_metric_count(len(result.metrics))
                return result
            finally:
                progress.stop()

        result = await with_repo_lock("compare", body)

        emit_report(
            result,
            flags,
            ReportRenderers(text=render_report, json=render_json),
            ReportOptions(verbose=flags.verbose, color=color_override, fail_on=flags.fail_on),
        )
        warn_empty_geomean_gates(flags.fail_on, result)
        if should_fail_gate(flags.fail_on, result):
            raise typer.Exit(GATE_EXIT_CODE)

    run_cli(run)
