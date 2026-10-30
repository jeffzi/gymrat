"""Shared inputs and expected messages for adapter tests.

:func:`build_stdout` wraps a list of raw benchmark entries in the top-level
``{"benchmarks": [...]}`` envelope the mitata adapter expects on stdout. Named
``build_stdout`` rather than ``stdout`` because callers commonly bind its
result to a local variable named ``stdout``, which would otherwise shadow the
import. :func:`build_tinybench_stdout` does the same for the
``{"tinybench": [...]}`` envelope the tinybench adapter expects.

:func:`benchmark` builds one mitata benchmark entry with a single run, and
:data:`VALID_BENCHMARK` is the well-formed sibling a test places next to a bad
entry to prove the adapter keeps parsing past it. :func:`task` and
:data:`VALID_TASK` are their tinybench counterparts.

:func:`unknown_adapter_error` and :func:`malformed_line_warning` ask the
adapters themselves for the error ``get_adapter`` raises on an unknown name and
the warning the metric-lines adapter gives for a line it cannot read. A test
that only checks such an error or warning reaches its caller compares against
them, so the wording stays pinned in the adapter tests alone.

This is test-support code shared across test modules, not a test module. It
carries no test functions of its own.
"""

import json
from typing import Any

from gymrat.adapters.metric_lines import metric_lines_adapter
from gymrat.adapters.registry import get_adapter
from gymrat.errors import GymratError


def unknown_adapter_error(name: str) -> GymratError:
    """Return the error ``get_adapter`` raises for an unregistered adapter name.

    Args:
        name: The adapter name to look up.

    Returns:
        The error ``get_adapter`` raised.

    Raises:
        AssertionError: When ``name`` is a registered adapter.
    """
    try:
        get_adapter(name)
    except GymratError as error:
        return error
    msg = f"{name!r} is a registered adapter"
    raise AssertionError(msg)


def malformed_line_warning(line: str) -> str:
    """Return the warning the metric-lines adapter gives for a line it cannot parse.

    Args:
        line: The offending stdout line.

    Returns:
        The single warning the adapter gives for ``line``.
    """
    warnings: list[str] = []
    metric_lines_adapter.parse(f"{line}\nMETRIC valid=1", warnings.append)
    (warning,) = warnings
    return warning


def build_stdout(benchmarks: list[Any]) -> str:
    """Serialize ``benchmarks`` into the mitata stdout JSON envelope.

    Args:
        benchmarks: The raw benchmark entries, well-formed or not.

    Returns:
        The JSON document the mitata adapter reads from stdout.
    """
    return json.dumps({"benchmarks": benchmarks})


def benchmark(
    alias: str,
    p50: float = 1,
    args: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a mitata benchmark entry holding one run.

    Args:
        alias: The benchmark alias, which may carry ``$placeholders``.
        p50: The run's median time; ignored when ``stats`` is given.
        args: The run's argument values; ``None`` means an empty object.
        stats: The run's whole stats object; ``None`` means ``{"p50": p50}``.

    Returns:
        The benchmark entry, ready for :func:`build_stdout`.
    """
    run_stats = {"p50": p50} if stats is None else stats
    return {"alias": alias, "runs": [{"args": args or {}, "stats": run_stats}]}


VALID_BENCHMARK = benchmark("valid")
"""A well-formed benchmark that parses to ``{"valid#time": 1}``."""


def build_tinybench_stdout(tasks: list[Any]) -> str:
    """Serialize ``tasks`` into the stdout JSON envelope the tinybench adapter reads.

    Args:
        tasks: The raw task entries, well-formed or not.

    Returns:
        The JSON document a bench script prints through the documented emission line.
    """
    return json.dumps({"tinybench": tasks})


def task(
    name: str,
    p50: float = 0.5,
    state: str = "completed",
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a tinybench task entry as the documented emission line writes it.

    Args:
        name: The task name.
        p50: The median latency in milliseconds; ignored when ``result`` is given.
        state: The result's state; ignored when ``result`` is given.
        result: The task's whole result object; ``None`` means one holding
            ``state`` and ``latency.p50``.

    Returns:
        The task entry, ready for :func:`build_tinybench_stdout`.
    """
    task_result = {"state": state, "latency": {"p50": p50}} if result is None else result
    return {"name": name, "result": task_result}


VALID_TASK = task("valid")
"""A well-formed completed task that parses to ``{"valid#time": 500_000}``."""
