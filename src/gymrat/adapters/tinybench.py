"""The ``tinybench`` adapter, which reads the payload a tinybench 6 bench script prints.

tinybench prints nothing on its own, so a bench script prints a
``{"tinybench": [...]}`` object through one documented line. Each completed task
becomes ``<name>#time`` from ``result.latency.p50``, converted from tinybench's
milliseconds to nanoseconds. The last object carrying a ``tinybench`` list wins:
task functions print while the bench runs, before the payload, so an object a
task logs cannot shadow it.
"""

import json
from typing import Annotated, Final

from pydantic import BaseModel, Strict, ValidationError

from gymrat.adapters.contract import TIME_SUFFIX
from gymrat.adapters.json_payload import JsonArrayAdapter, NameWording, accepts_metric_prefix
from gymrat.utils import (
    NS_PER_MS,
    Number,
    WarnSink,
    describe_error,
    first_problem,
    is_finite_or_warn,
    named_subject,
    record_metric,
    warn_skip,
)

_KEY = "tinybench"

_COLLISION_ADVICE = "give the tasks distinct names"

_EMPTY_ERRORS: Final[tuple[object, ...]] = (None, "", {})
"""Error values that carry no detail.

``JSON.stringify`` writes an ``Error`` as ``{}``, so an errored task reads as
``{}`` unless the bench script put something readable in its place.
"""


class _Task(BaseModel):
    """One entry of the top-level ``tinybench`` array.

    Attributes:
        name: The task name, the prefix of its metric name.
        result: The task's raw ``result`` object, left loose: its shape depends
            on its ``state``, which is checked before the statistics a completed
            task carries.
    """

    name: Annotated[str, Strict()]
    result: dict[str, object]


class _Latency(BaseModel):
    """A completed result's ``latency`` statistics, in milliseconds."""

    p50: Number


class _CompletedResult(BaseModel):
    """The ``result`` of a task whose ``state`` is ``"completed"``."""

    latency: _Latency


def _completed_p50_ns(quoted_name: str, result: dict[str, object], warn: WarnSink) -> float | None:
    """Read a task's median latency in nanoseconds, warning when the task is skipped.

    The finite check runs on the nanosecond value, so a median that is finite in
    milliseconds but overflows on conversion is skipped too.

    Args:
        quoted_name: The JSON-escaped task name.
        result: The task's ``result`` object.
        warn: The sink that receives the warning about a task to skip.

    Returns:
        The finite ``latency.p50`` converted from tinybench's milliseconds to
        nanoseconds, or ``None`` when the task is skipped.
    """
    subject = f"task {quoted_name}"
    # tinybench 6 introduced ``state``; an earlier major's result has none, so its
    # absence means the bench runs an unsupported version.
    try:
        state = result["state"]
    except KeyError:
        tail = " with missing result.state (the tinybench adapter reads tinybench 6 or later)"
        warn_skip(warn, subject, tail)
        return None
    if state == "errored":
        error = result.get("error")
        detail = "" if error in _EMPTY_ERRORS else f" ({describe_error(error)})"
        warn_skip(warn, "task", f" with an error: {quoted_name}{detail}")
        return None
    if state != "completed":
        tail = f" with state {json.dumps(state)} (only a completed task yields a metric)"
        warn_skip(warn, subject, tail)
        return None
    try:
        p50 = _CompletedResult.model_validate(result).latency.p50
    except ValidationError as exc:
        warn_skip(warn, subject, first_problem(exc, ("result",)))
        return None
    # tinybench reports latency in milliseconds; a ``#time`` metric is in nanoseconds.
    p50_ns = p50 * NS_PER_MS
    if not is_finite_or_warn(p50_ns, warn, subject, "result.latency.p50"):
        return None
    return p50_ns


def _extract_task_metric(entry: object, metrics: dict[str, float], warn: WarnSink) -> None:
    try:
        task = _Task.model_validate(entry)
    except ValidationError as exc:
        warn_skip(warn, named_subject("task", entry, "name"), first_problem(exc))
        return

    quoted_name = json.dumps(task.name)
    p50_ns = _completed_p50_ns(quoted_name, task.result, warn)
    if p50_ns is None:
        return

    wording = NameWording(skipped="task", label="Task name", origin=quoted_name)
    prefix = task.name
    if not accepts_metric_prefix(prefix, warn, wording):
        return

    if p50_ns == 0:
        warn(
            f"Task {quoted_name} has a median of 0: its calls run faster than the "
            "timer can resolve, so a comparison cannot see a change in it (have the task "
            "batch its calls and return the time per call as overriddenDuration)"
        )
    record_metric(metrics, f"{prefix}{TIME_SUFFIX}", p50_ns, warn, _COLLISION_ADVICE)


tinybench_adapter = JsonArrayAdapter(
    name="tinybench",
    key=_KEY,
    last=True,
    extract=_extract_task_metric,
    empty_msg="No valid tinybench tasks found",
)
"""A median of 0 is kept, with a warning that the timer cannot resolve the task. A
task that errored, did not complete, comes from a tinybench release before 6,
carries a malformed ``name``/``result`` shape or a non-finite median, or whose
name holds a line terminator or an empty path segment is skipped rather than
failing the parse. On a collision between two tasks sharing a name, the last task
wins.

A task name containing ``#`` (reserved as the metric-type separator) aborts the
parse with :class:`~gymrat.adapters.contract.AdapterError`.
"""
