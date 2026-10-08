"""Shared stdout-payload builder for mitata adapter tests.

:func:`build_stdout` wraps a list of raw benchmark entries in the top-level
``{"benchmarks": [...]}`` envelope the mitata adapter expects on stdout. Named
``build_stdout`` rather than ``stdout`` because callers commonly bind its
result to a local variable named ``stdout``, which would otherwise shadow the
import.

:data:`LINE_BREAKS` lists every character ``str.splitlines`` breaks on, shared by
the adapter tests that check a metric name holding one is rejected.

:data:`VALID_ADAPTERS_HINT` is the hint ``get_adapter`` attaches to an
unknown-adapter error, shared by every test that surfaces that error.

This is test-support code, not a test module: ``test_mitata``,
``test_adapters``, ``tests/config/test_config_load.py``,
``tests/test_doctor.py`` and ``tests/loop/iterate/run/test_run.py`` import
it. It carries no test functions of its own.
"""

import json
from typing import Any, NamedTuple


class LineBreak(NamedTuple):
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


def build_stdout(benchmarks: list[Any]) -> str:
    """Serialize ``benchmarks`` into the mitata stdout JSON envelope."""
    return json.dumps({"benchmarks": benchmarks})
