"""The ``gymrat compare`` command: one baseline against one or more candidates.

The action holds the repository lock for the length of the run, stops the
progress reporter before any report or error text, renders the result to stdout
once the lock is released, and gates the exit code on the ``--fail-on``
conditions. The comparison engine is imported inside the action so assembling the
CLI never pulls the heavy statistics stack.

Only gating metrics participate in the fail-on gate — informational verdicts never
trip an exit-code gate. Conditions are OR-ed: any one that trips fails the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, assert_never

import typer

from gymrat.cli.budget_report import emit_report, warn_duration_over_budget
from gymrat.cli.console import apply_color_override, apply_debug
from gymrat.cli.exit import run_cli
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
    parse_fail_on,
)
from gymrat.cli.run_setup import SharedFlags, begin_run
from gymrat.command_run import CommandTrace, config_trace_args, with_repo_lock
from gymrat.config.resolve import resolve_config
from gymrat.errors import GATE_EXIT_CODE
from gymrat.report.json_doc import render_json
from gymrat.report.tally import count_verdicts
from gymrat.report.text.render import render_report
from gymrat.report.types import (
    CandidateComparison,
    ComparisonResult,
    FailOnCondition,
    GeomeanFailOn,
    MetricComparisons,
    RegressedFailOn,
    ReportOptions,
)
from gymrat.sampling import RunOptions, TargetSpec
from gymrat.warn import WarnSink, warn_to_stderr

if TYPE_CHECKING:
    from gymrat.model import GeomeanResult

_BaselineArgument = Annotated[
    TargetSpec,
    typer.Argument(
        click_type=PositionalParamType(),
        metavar="BASELINE",
        help="[label=]<ref|dir> to measure against",
    ),
]
_CandidatesArgument = Annotated[
    list[TargetSpec],
    typer.Argument(
        click_type=PositionalParamType(),
        metavar="CANDIDATES...",
        help="[label=]<ref|dir>, each judged against the baseline",
    ),
]
_VerboseOption = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="name the statistical method behind each verdict"),
]
_FailOnOption = Annotated[
    list[FailOnCondition] | None,
    typer.Option(
        "--fail-on",
        parser=parse_fail_on,
        metavar="<condition>",
        help='exit 1 when a condition trips (repeatable: "regressed", "geomean:<pct>")',
    ),
]


@dataclass(frozen=True, slots=True)
class CompareFlags(SharedFlags):
    """The compare command's flags: the shared set plus the two only a verdict can answer."""

    verbose: bool = False
    fail_on: tuple[FailOnCondition, ...] = ()


def _serialize_fail_on(conditions: tuple[FailOnCondition, ...]) -> str:
    """Serialize fail-on conditions to the CLI grammar for trace args."""
    parts: list[str] = []
    for condition in conditions:
        if isinstance(condition, RegressedFailOn):
            parts.append("regressed")
        elif isinstance(condition, GeomeanFailOn):
            parts.append(f"geomean:{condition.pct:g}")
        else:
            assert_never(condition)
    return ",".join(parts)


def _gating_metrics(metrics: MetricComparisons) -> MetricComparisons:
    """The gating subset of ``metrics`` — the only metrics a gate may judge."""
    return {name: metric for name, metric in metrics.items() if metric.meta.gating}


def _gated_geomeans_of(candidate: CandidateComparison) -> list[GeomeanResult]:
    """The gated geomean of every kind that gates, one entry per such kind."""
    return [kind.gated_geomean for kind in candidate.kinds if kind.gated_geomean is not None]


def should_fail_gate(conditions: tuple[FailOnCondition, ...], result: ComparisonResult) -> bool:
    """Return ``True`` when any condition trips — meaning the process should exit non-zero.

    Args:
        conditions: The fail-on conditions to evaluate (OR-ed).
        result: The comparison result to check the conditions against.

    Returns:
        ``True`` when any condition trips.
    """
    if not conditions:
        return False

    gating = _gating_metrics(result.metrics)

    for condition in conditions:
        match condition:
            case RegressedFailOn():
                if any(
                    count_verdicts(gating, index).regressed > 0
                    for index in range(len(result.candidates))
                ):
                    return True
            case GeomeanFailOn(pct=pct):
                if any(
                    geomean.n > 0 and geomean.value >= pct
                    for candidate in result.candidates
                    for geomean in _gated_geomeans_of(candidate)
                ):
                    return True
            case _ as unreachable:
                assert_never(unreachable)

    return False


def warn_empty_geomean_gates(
    conditions: tuple[FailOnCondition, ...],
    result: ComparisonResult,
    *,
    warn: WarnSink = warn_to_stderr,
) -> None:
    """Warn once per candidate whose geomean gate had nothing stable to judge.

    Runs only when a geomean condition is present; such a candidate never trips
    the gate, so the warning is how the user learns the gate was inert for it.

    Args:
        conditions: The fail-on conditions in effect.
        result: The comparison result to inspect for inert gates.
        warn: Where each warning goes.
    """
    if not any(isinstance(condition, GeomeanFailOn) for condition in conditions):
        return

    for candidate in result.candidates:
        if all(geomean.n == 0 for geomean in _gated_geomeans_of(candidate)):
            warn(
                f'warning: geomean gate for "{candidate.label}" '
                "had no stable gating metrics to measure"
            )


async def _compare_body(
    flags: CompareFlags,
    baseline: TargetSpec,
    candidates: list[TargetSpec],
) -> ComparisonResult:
    labels = [spec.display_label for spec in [baseline, *candidates]]
    progress = begin_run(
        flags,
        1 + len(candidates),
        target_labels=labels,
    )
    try:
        config_resolved = resolve_config(flags)
        from gymrat import (  # noqa: PLC0415 -- lazy import keeps CLI startup off the heavy comparison stack
            compare as engine,
        )

        options = engine.CompareOptions(
            run=RunOptions.from_config(
                config_resolved, on_progress=progress.report, warn=progress.warn
            ),
            baseline=baseline,
            candidates=candidates,
            unstable_noise_pct=config_resolved.unstable_noise_pct,
        )
        return await engine.compare(options)
    finally:
        progress.stop()


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
    color: ColorOption = None,
    debug: DebugOption = False,
) -> None:
    """Run each candidate against the baseline and exit non-zero when --fail-on fires."""
    apply_debug(debug)
    color_override = apply_color_override(color)
    flags = CompareFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
        color=color,
        format=output_format.value,
        verbose=verbose,
        fail_on=tuple(fail_on) if fail_on is not None else (),
    )

    async def run() -> None:
        warn_duration_over_budget(halve=False)
        trace_args: dict[str, object] = {
            "baseline": baseline.display_label,
            "candidates": [spec.display_label for spec in candidates],
            "fail_on": _serialize_fail_on(flags.fail_on),
            **config_trace_args(flags),
        }

        async def body(trace: CommandTrace) -> ComparisonResult:
            comparison = await _compare_body(flags, baseline, candidates)
            warn_empty_geomean_gates(flags.fail_on, comparison)
            if should_fail_gate(flags.fail_on, comparison):
                trace.gate = True
                trace.reason = "fail-on"
            return comparison

        result = await with_repo_lock("compare", body, args=trace_args)
        emit_report(
            result,
            flags,
            ReportOptions(verbose=flags.verbose, color=color_override, fail_on=flags.fail_on),
            text=render_report,
            json=render_json,
        )
        if should_fail_gate(flags.fail_on, result):
            raise typer.Exit(GATE_EXIT_CODE)

    run_cli(run)
