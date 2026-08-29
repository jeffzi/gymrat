"""The ``gymrat measure`` command: one revision or directory, on its own.

The action holds the repository lock for the length of the run, stops the
progress reporter before any report or error text, and renders the measurement
to stdout once the lock is released. It never gates the exit code. The
measurement engine is imported inside the action so assembling the CLI never
pulls the heavy statistics stack.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Annotated

import typer

from gymrat.cli.shared import (
    AdapterOption,
    BenchOption,
    ConfigOption,
    DebugOption,
    FormatOption,
    MeasureFlags,
    NoColorOption,
    OutputFormat,
    PrepareOption,
    RecordOption,
    ReportRenderers,
    SamplesOption,
    TimeoutOption,
    begin_run,
    color_override_of,
    emit_report,
    parse_positional,
    run_cli,
    run_options_of,
    set_debug_mode,
    wants_json,
    with_repo_lock,
    write_and_flush,
)
from gymrat.config import resolve_config
from gymrat.report import render_measure_json, render_measure_report
from gymrat.report.types import MeasurementResult, ReportOptions
from gymrat.sampling import TargetSpec
from gymrat.session.clock import now_iso
from gymrat.session.paths import repo_root
from gymrat.session.records import BaselineRecord
from gymrat.session.store import RequiredSession, append_record, require_open_session

_TargetArgument = Annotated[
    TargetSpec | None,
    typer.Argument(
        parser=parse_positional,
        metavar="[TARGET]",
        help="[label=]<ref|dir> to measure; defaults to the current directory",
    ),
]


@dataclass(frozen=True, slots=True)
class _MeasureOutcome:
    """What the locked run produced: the measurement and the session it recorded to.

    ``recording`` is the open session ``--record`` wrote the baseline into, or
    ``None`` when recording was not asked for — carried out of the lock so the
    post-report note can name the session by id.
    """

    result: MeasurementResult
    recording: RequiredSession | None


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
    record: RecordOption = False,
    debug: DebugOption = False,
) -> None:
    """Measure one revision or directory on its own, with nothing to compare it to."""
    if debug:
        set_debug_mode(True)
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
        record=record,
    )
    color_override = color_override_of(flags.color)

    async def run() -> None:
        async def body() -> _MeasureOutcome:
            progress = begin_run(flags, 1)
            try:
                config_resolved = resolve_config(flags)
                # Recording needs somewhere to write, so the open-session check
                # comes before any benching: failing after a long run would throw
                # the samples away with nowhere to record them.
                recording = (
                    require_open_session(repo_root(), "recording a measurement")
                    if flags.record
                    else None
                )
                # Lazy: keep the heavy statistics stack out of CLI assembly and --help.
                from gymrat import measure as engine  # noqa: PLC0415

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
                result = await engine.measure(options)
            finally:
                progress.stop()

            # Still under the repo lock: a failed bench raised above and appended
            # nothing, so only a completed run reaches here.
            if recording is not None:
                append_record(
                    recording.jsonl_path,
                    BaselineRecord(
                        type="baseline",
                        at=now_iso(),
                        label=result.label,
                        samples=tuple(result.rounds),
                    ),
                )
            return _MeasureOutcome(result=result, recording=recording)

        outcome = await with_repo_lock("measure", body)

        emit_report(
            outcome.result,
            flags,
            ReportRenderers(text=render_measure_report, json=render_measure_json),
            ReportOptions(color=color_override),
        )

        recording = outcome.recording
        if recording is not None:
            note = (
                f'baseline "{outcome.result.label}" '
                f"recorded to session {recording.session.session_id}\n"
            )
            # A JSON run keeps stdout a clean document, so the note rides stderr.
            stream = sys.stderr if wants_json(flags) else sys.stdout
            write_and_flush(stream, note)

    run_cli(run)
