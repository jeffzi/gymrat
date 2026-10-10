"""The ``gymrat measure`` command: one revision or directory, on its own.

The action holds the repository lock for the length of the run, stops the
progress reporter before any report or error text, and renders the measurement
to stdout once the lock is released. It never gates the exit code.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Annotated

import typer

from gymrat.cli.budget_report import emit_report, wants_json, warn_duration_over_budget
from gymrat.cli.console import apply_command_flags
from gymrat.cli.exit import run_cli, write_stdout
from gymrat.cli.options import (
    AdapterOption,
    BenchOption,
    ColorOption,
    ConfigOption,
    DebugOption,
    FormatOption,
    OutputFormat,
    PositionalParamType,
    PrepareOption,
    SamplesOption,
    TimeoutOption,
)
from gymrat.cli.run_setup import SharedFlags, begin_run
from gymrat.command_run import with_repo_lock
from gymrat.config import config_trace_args, resolve_config
from gymrat.loop.baseline import measure_baseline
from gymrat.report.json_doc import render_measure_json
from gymrat.report.text.render import render_measure_report
from gymrat.report.types import DEFAULT_REPORT_OPTIONS, MeasurementResult
from gymrat.sampling import RunOptions, TargetSpec
from gymrat.session.paths import repo_root
from gymrat.session.store import RequiredSession, append_record, require_open_session
from gymrat.utils import write_and_flush

_TargetArgument = Annotated[
    TargetSpec | None,
    typer.Argument(
        click_type=PositionalParamType(),
        metavar="[TARGET]",
        help="[label=]<ref|dir> to measure; defaults to the current directory",
    ),
]
_RecordOption = Annotated[
    bool,
    typer.Option("--record", "-r", help="append the run to the session log as a baseline"),
]


@dataclass(frozen=True, slots=True)
class MeasureFlags(SharedFlags):
    """The measure command's flags: the shared set plus whether to record the run."""

    record: bool = False


@dataclass(frozen=True, slots=True)
class _MeasureOutcome:
    """What the locked run produced: the measurement and the session it recorded to.

    Attributes:
        result: The measurement the run produced.
        recording: The open session ``--record`` wrote the baseline into, or
            ``None`` when recording was not asked for — carried out of the lock
            so the post-report note can name the session by id.
    """

    result: MeasurementResult
    recording: RequiredSession | None


async def _measure_body(
    flags: MeasureFlags,
    resolved_target: TargetSpec,
) -> _MeasureOutcome:
    progress = begin_run(flags, 1, target_labels=[resolved_target.display_label])
    try:
        config_resolved = resolve_config(flags)
        # Session check before bench: failing after a long run would lose samples.
        recording = (
            require_open_session(repo_root(), "recording a measurement") if flags.record else None
        )
        run_opts = RunOptions.from_config(
            config_resolved, on_progress=progress.report, warn=progress.warn
        )
        result, record = await measure_baseline(resolved_target, run_opts)
    finally:
        progress.stop()

    # Still under the repo lock: only a completed run reaches here.
    if recording is not None:
        append_record(recording.jsonl_path, record)
    return _MeasureOutcome(result=result, recording=recording)


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
    color: ColorOption = None,
    record: _RecordOption = False,
    debug: DebugOption = False,
) -> None:
    """Measure one revision or directory on its own, with nothing to compare it to."""
    apply_command_flags(debug=debug, color=color)
    resolved_target = target if target is not None else TargetSpec(label=None, target=".")
    flags = MeasureFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
        format=output_format,
        record=record,
    )

    async def run() -> None:
        warn_duration_over_budget(halve=True)
        outcome = await with_repo_lock(
            "measure",
            lambda _trace: _measure_body(flags, resolved_target),
            args=config_trace_args(
                flags, target=resolved_target.display_label, record=record or None
            ),
        )
        emit_report(
            outcome.result,
            flags,
            DEFAULT_REPORT_OPTIONS,
            text=render_measure_report,
            json=render_measure_json,
        )
        if outcome.recording is not None:
            note = (
                f'baseline "{outcome.result.label}" '
                f"recorded to session {outcome.recording.session.session_id}\n"
            )
            if wants_json(flags):
                write_and_flush(sys.stderr, note)
            else:
                write_stdout(note)

    run_cli(run)
