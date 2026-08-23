"""Shared stdout-payload builder for mitata adapter tests.

:func:`build_stdout` wraps a list of raw benchmark entries in the top-level
``{"benchmarks": [...]}`` envelope the mitata adapter expects on stdout. Named
``build_stdout`` rather than ``stdout`` because callers commonly bind its
result to a local variable named ``stdout``, which would otherwise shadow the
import.

This is test-support code, not a test module: ``test_mitata`` and
``test_mitata_errors`` import it. It carries no test functions of its own.
"""

import json
from typing import Any


def build_stdout(benchmarks: list[Any]) -> str:
    """Serialize ``benchmarks`` into the mitata stdout JSON envelope."""
    return json.dumps({"benchmarks": benchmarks})


__all__ = ["build_stdout"]
