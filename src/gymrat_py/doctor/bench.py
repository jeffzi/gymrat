"""The doctor bench smoke-run section.

Runs the resolved bench command once and validates that the configured adapter
can parse its output, then cross-checks the parsed metric names against the
config's ``primary``, ``metrics``, and ``kinds`` expectations. The adapter name
is resolved first, so an unknown adapter fails even under ``--no-bench`` or a
failed config section; everything else short-circuits before a process runs.
"""

import asyncio
from dataclasses import dataclass

from gymrat_py.adapters import Adapter, AdapterError, get_adapter
from gymrat_py.adapters.defaults import DEFAULT_METRIC_KIND
from gymrat_py.config import GEOMEAN_PRIMARY, KindEntry, MetricEntry
from gymrat_py.doctor.checks import Check, CheckSection
from gymrat_py.errors import GymratError, hint_of
from gymrat_py.exec import ExecOptions, ExecTimeoutError
from gymrat_py.exec import exec as run_exec

_MAX_STDERR_EXCERPT_LINES = 5
_MAX_METRIC_NAMES_SHOWN = 5

_TIMEOUT_HINT = 'Raise the limit with --timeout or the "timeout_seconds" config key'
_NO_BENCH_HINT = 'Set the bench command with --bench or the "bench" config key'


@dataclass(frozen=True, slots=True)
class BenchSectionInput:
    """Everything the smoke run needs, resolved from flags and config.

    ``no_bench`` and ``config_failed`` short-circuit before any process runs.
    ``metrics`` and ``kinds`` drive the post-parse cross-checks, so they are
    absent when the config constrains neither. Setting ``abort`` kills the smoke
    run's process group, so a Ctrl-C leaves nothing behind.
    """

    bench: str | None
    adapter: str
    timeout_seconds: int
    primary: str
    repo_root: str
    abort: asyncio.Event
    metrics: dict[str, MetricEntry] | None = None
    kinds: dict[str, KindEntry] | None = None
    no_bench: bool = False
    config_failed: bool = False


def _bench_section(checks: list[Check]) -> CheckSection:
    return CheckSection(title="Bench", checks=checks)


def _skipped_bench_section(detail: str) -> CheckSection:
    """A section with a single OK "bench" check, used when the smoke run was skipped."""
    return _bench_section([Check(name="bench", status="ok", detail=detail)])


def _cross_check_metrics(
    input_: BenchSectionInput, metric_names: list[str], adapter: Adapter
) -> list[Check]:
    checks: list[Check] = []

    if input_.primary != GEOMEAN_PRIMARY and input_.primary not in metric_names:
        checks.append(
            Check(
                name="primary",
                status="fail",
                detail=f'primary "{input_.primary}" was not found in parsed metrics',
            )
        )

    if input_.metrics is not None:
        missing = [name for name in input_.metrics if name not in metric_names]
        if missing:
            checks.append(
                Check(
                    name="metrics",
                    status="warn",
                    detail=f"Config metrics not found in bench output: {', '.join(missing)}",
                )
            )

    if input_.kinds is not None:
        parsed_kinds = {adapter.defaults(name).kind or DEFAULT_METRIC_KIND for name in metric_names}
        missing = [kind for kind in input_.kinds if kind not in parsed_kinds]
        if missing:
            checks.append(
                Check(
                    name="kinds",
                    status="warn",
                    detail=f"Config kinds not matched by any parsed metric: {', '.join(missing)}",
                )
            )

    return checks


def _summarize_metric_names(metric_names: list[str]) -> str:
    if len(metric_names) <= _MAX_METRIC_NAMES_SHOWN:
        name_list = ", ".join(metric_names)
    else:
        shown = ", ".join(metric_names[:_MAX_METRIC_NAMES_SHOWN])
        name_list = f"{shown} … ({len(metric_names)} total)"
    return f"{len(metric_names)} metric(s): {name_list}"


def _exit_failure_check(exit_code: int, stderr: str) -> Check:
    excerpt = "\n".join(stderr.strip().split("\n")[:_MAX_STDERR_EXCERPT_LINES])
    failure = f"Bench command exited with code {exit_code}"
    detail = failure if excerpt == "" else f"{failure}: {excerpt}"
    return Check(name="bench run", status="fail", detail=detail)


async def _run_and_parse_bench(
    bench: str, adapter: Adapter, input_: BenchSectionInput
) -> list[Check]:
    result = await run_exec(
        bench,
        ExecOptions(
            cwd=input_.repo_root,
            timeout_ms=input_.timeout_seconds * 1000,
            abort=input_.abort,
        ),
    )

    if isinstance(result, ExecTimeoutError):
        return [
            Check(
                name="bench run",
                status="fail",
                detail=f"Bench command timed out after {input_.timeout_seconds}s",
                hint=_TIMEOUT_HINT,
            )
        ]

    if result.exit_code != 0:
        return [_exit_failure_check(result.exit_code, result.stderr)]

    warnings: list[str] = []
    try:
        parsed = adapter.parse(result.stdout, warnings.append)
    except Exception as error:  # noqa: BLE001 -- adapter.parse may raise anything; surface it as a check
        checks: list[Check] = []
        if warnings:
            checks.append(Check(name="adapter warnings", status="warn", detail="\n".join(warnings)))
        detail = (
            str(error) if isinstance(error, AdapterError) else f"{type(error).__name__}: {error}"
        )
        checks.append(Check(name="parse", status="fail", detail=detail))
        return checks

    metric_names = list(parsed)
    checks = [Check(name="bench run", status="ok", detail=_summarize_metric_names(metric_names))]
    # A dedicated check so the warnings move the report's warn count and reach a
    # ``--format json`` consumer reading statuses rather than prose.
    if warnings:
        checks.append(Check(name="adapter warnings", status="warn", detail="\n".join(warnings)))
    checks.extend(_cross_check_metrics(input_, metric_names, adapter))
    return checks


async def build_bench_section(input_: BenchSectionInput) -> CheckSection:
    """Build the "Bench" section by running the bench command once and parsing its output.

    The smoke run is skipped when ``--no-bench`` was passed or the config section
    already failed — a run against broken config proves nothing — but the adapter
    name is validated first regardless, since a typo there is worth reporting.
    """
    try:
        adapter = get_adapter(input_.adapter)
    except GymratError as error:
        return _bench_section(
            [Check(name="adapter", status="fail", detail=str(error), hint=hint_of(error))]
        )

    if input_.no_bench:
        return _skipped_bench_section("Bench smoke run skipped (--no-bench)")

    if input_.config_failed:
        return _skipped_bench_section("Bench smoke run skipped — fix config errors first")

    if input_.bench is None:
        return _bench_section(
            [
                Check(
                    name="bench",
                    status="fail",
                    detail="No bench command resolved",
                    hint=_NO_BENCH_HINT,
                )
            ]
        )

    return _bench_section(await _run_and_parse_bench(input_.bench, adapter, input_))
