"""Shared inputs and expected messages for adapter tests.

:func:`build_stdout` wraps a list of raw benchmark entries in the top-level
``{"benchmarks": [...]}`` envelope the mitata adapter expects on stdout. Named
``build_stdout`` rather than ``stdout`` because callers commonly bind its
result to a local variable named ``stdout``, which would otherwise shadow the
import.

:func:`benchmark` builds one mitata benchmark entry with a single run, and
:data:`VALID_BENCHMARK` is the well-formed sibling a test places next to a bad
entry to prove the adapter keeps parsing past it.

:data:`LINE_BREAKS` lists every character ``str.splitlines`` breaks on, shared by
the adapter tests that check a metric name holding one is rejected.

:data:`VALID_ADAPTERS_HINT` and :func:`unknown_adapter_message` are the hint and
message ``get_adapter`` puts on an unknown-adapter error, shared by every test
that surfaces that error.

This is test-support code, not a test module: ``test_mitata``,
``test_adapters``, ``tests/config/test_config_load.py``,
``tests/test_doctor.py`` and ``tests/loop/iterate/run/test_run.py`` import
it. It carries no test functions of its own.
"""

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LineBreak:
    """One character that ``str.splitlines`` breaks on, with its JSON-escaped form and test id."""

    char: str
    """The line-break character itself."""
    escaped: str
    """The escape ``json.dumps`` writes for ``char``, as a warning shows it."""
    name: str
    """A readable test id."""


LINE_BREAKS = (
    LineBreak("\n", r"\n", "line-feed"),
    LineBreak("\r", r"\r", "carriage-return"),
    LineBreak("\v", r"\u000b", "vertical-tab"),
    LineBreak("\f", r"\f", "form-feed"),
    LineBreak("\x1c", r"\u001c", "file-separator"),
    LineBreak("\x1d", r"\u001d", "group-separator"),
    LineBreak("\x1e", r"\u001e", "record-separator"),
    LineBreak("\x85", r"\u0085", "next-line"),
    LineBreak("\u2028", r"\u2028", "line-separator-u2028"),
    LineBreak("\u2029", r"\u2029", "paragraph-separator-u2029"),
)
"""Every character ``str.splitlines`` breaks on.

``metric_name.LINE_TERMINATORS`` must match exactly this set.
"""

VALID_ADAPTERS_HINT = "valid adapters are: metric-lines, mitata"
"""The hint on an unknown-adapter error, listing every built-in adapter."""


def unknown_adapter_message(name: str) -> str:
    """Return the message ``get_adapter`` raises for an unregistered adapter name.

    Args:
        name: The adapter name that was looked up.

    Returns:
        The error message naming ``name``.
    """
    return f'Unknown adapter: "{name}".'


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
