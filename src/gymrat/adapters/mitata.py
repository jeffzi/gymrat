"""The ``mitata`` adapter, which reads the JSON ``mitata --json`` writes.

The payload is a ``benchmarks`` array whose entries carry an ``alias`` and a list
of ``runs``. Each run becomes ``<alias>#time`` from ``stats.p50`` and, when
mitata measured it, ``<alias>#memory`` from ``stats.heap.avg``. Parameterized
benchmarks carry ``$name`` placeholders in the alias that are substituted with
``name=value``, so one benchmark becomes one metric per argument combination.
"""

import json
import re
from typing import Annotated

from pydantic import BaseModel, Field, Strict, ValidationError

from gymrat.adapters.contract import MEMORY_SUFFIX, TIME_SUFFIX
from gymrat.adapters.json_payload import JsonArrayAdapter, NameWording, accepts_metric_prefix
from gymrat.utils import (
    Number,
    WarnSink,
    describe_error,
    first_problem,
    is_finite_or_warn,
    named_subject,
    record_metric,
    warn_skip,
)

_COLLISION_ADVICE = "give the benchmark aliases distinct $placeholders to separate the runs"


def _serialize_arg_value(value: object) -> str:
    """Serialize a run-argument value for inclusion in a metric name.

    Primitives keep a JavaScript ``String()`` form so booleans read ``true``/
    ``false`` and ``None`` reads ``null``; objects and arrays serialize via JSON
    with recursively sorted keys so two structurally equal objects always produce
    the same metric name.

    Args:
        value: The argument value as parsed from JSON.

    Returns:
        The serialized string representation suitable for metric names.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float) and value.is_integer():
        # JS String(5.0) is "5", not "5.0"; an int falls through to str() below.
        return str(int(value))
    return str(value)


def _build_metric_name_prefix(alias: str, args: dict[str, object]) -> str:
    r"""Substitute each ``$key`` in ``alias`` with ``key=value`` via a single regex pass.

    Keys are matched longest-first to prevent a shorter key from consuming a
    prefix of a longer one (``$ab`` matches key ``ab`` before ``a``).  The
    callable replacement returns a literal string, so ``re.sub`` never interprets
    backslash escapes (``\1``, ``\g<0>``, ``\n``) in argument values.
    Unmatched ``$`` tokens stay as-is because the alternation only covers keys
    present in ``args``.

    Args:
        alias: The benchmark alias, possibly carrying ``$key`` placeholders.
        args: The run's arguments, keyed by name.

    Returns:
        The alias with ``$key`` placeholders replaced by ``key=value`` pairs.
    """
    if not args:
        return alias
    pattern = re.compile(
        r"\$(" + "|".join(re.escape(k) for k in sorted(args, key=len, reverse=True)) + ")"
    )

    def _repl(m: re.Match[str]) -> str:
        return f"{m.group(1)}={_serialize_arg_value(args[m.group(1)])}"

    return pattern.sub(_repl, alias)


class _Heap(BaseModel):
    """The ``stats.heap`` object mitata writes when it measured allocations."""

    avg: Number | None = None


class _Stats(BaseModel):
    """A run's ``stats`` object.

    Attributes:
        p50: The run's median time, in nanoseconds.
        heap: The raw ``stats.heap`` value, left loose and validated against
            :class:`_Heap` on its own, so a malformed heap costs only the
            ``#memory`` metric, never the ``#time`` one.
    """

    p50: Number
    heap: object = None


class _Run(BaseModel):
    """One entry of a benchmark's ``runs`` array."""

    # A run that never varied its arguments can omit ``args`` entirely.
    args: dict[str, object] = Field(default_factory=dict)
    stats: _Stats


class _Benchmark(BaseModel):
    """One entry of the top-level ``benchmarks`` array."""

    alias: Annotated[str, Strict()]
    runs: list[object]


def _record_heap_metric(
    heap: object,
    prefix: str,
    quoted_alias: str,
    metrics: dict[str, float],
    warn: WarnSink,
) -> None:
    """Record ``<prefix>#memory`` from ``stats.heap.avg`` when mitata measured it."""
    if heap is None:
        return
    subject = f"memory metric of {quoted_alias}"
    try:
        avg = _Heap.model_validate(heap).avg
    except ValidationError as exc:
        warn_skip(warn, subject, first_problem(exc, ("stats", "heap")))
        return
    if avg is None:
        return
    if not is_finite_or_warn(avg, warn, subject, "stats.heap.avg"):
        return
    record_metric(metrics, f"{prefix}{MEMORY_SUFFIX}", avg, warn, _COLLISION_ADVICE)


def _extract_run_metrics(
    run: object, alias: str, metrics: dict[str, float], warn: WarnSink
) -> None:
    quoted_alias = json.dumps(alias)
    # An errored run carries no usable stats, so the error is the one problem
    # worth reporting — checked before the shape, which it would otherwise fail.
    if isinstance(run, dict) and run.get("error") is not None:
        error = describe_error(run["error"])
        warn_skip(warn, "run", f" with an error: {quoted_alias} ({error})")
        return

    subject = f"run of {quoted_alias}"
    try:
        parsed = _Run.model_validate(run)
    except ValidationError as exc:
        warn_skip(warn, subject, first_problem(exc))
        return

    p50 = parsed.stats.p50
    if not is_finite_or_warn(p50, warn, subject, "stats.p50"):
        return

    wording = NameWording(
        skipped="run",
        label="Metric prefix",
        origin=quoted_alias,
        hash_note=f" (alias: {quoted_alias})",
        terminator_note=" (the alias or one of its argument values carries one)",
    )
    prefix = _build_metric_name_prefix(alias, parsed.args)
    if not accepts_metric_prefix(prefix, warn, wording):
        return

    record_metric(metrics, f"{prefix}{TIME_SUFFIX}", p50, warn, _COLLISION_ADVICE)
    _record_heap_metric(parsed.stats.heap, prefix, quoted_alias, metrics, warn)


def _extract_benchmark_metrics(
    benchmark: object, metrics: dict[str, float], warn: WarnSink
) -> None:
    try:
        parsed = _Benchmark.model_validate(benchmark)
    except ValidationError as exc:
        warn_skip(warn, named_subject("benchmark", benchmark, "alias"), first_problem(exc))
        return

    for run in parsed.runs:
        _extract_run_metrics(run, parsed.alias, metrics, warn)


mitata_adapter = JsonArrayAdapter(
    name="mitata",
    key="benchmarks",
    last=False,
    extract=_extract_benchmark_metrics,
    empty_msg="No valid benchmark runs found",
)
"""Runs that errored, reported a non-finite ``p50``, resolved to a metric name
carrying a line terminator or an empty path segment, or carried a malformed
``args``/``stats`` shape are skipped rather than failing the parse — a single bad
argument combination should not discard the rest of the run — and likewise for a
benchmark whose ``alias``/``runs`` shape is malformed. Every skip warns rather
than vanishing silently, as does a collision between two runs landing on one
metric name; on a collision the last run still wins.

A substituted metric prefix containing ``#`` (reserved as the metric-type
separator) aborts the parse with :class:`~gymrat.adapters.contract.AdapterError`.
"""
