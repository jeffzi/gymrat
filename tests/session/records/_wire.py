"""Wire-format session records shared by the record parsing tests.

Each constant is a record exactly as it is written to the session log, before
``parse_record`` validates it, and the helpers derive invalid or partial
variants from them.
"""

from tests.session.records._fixtures import (
    AT,
    BASELINE_SHA,
    SESSION_ID,
)

SESSION_RECORD: dict[str, object] = {
    "type": "session",
    "schema": 1,
    "session_id": SESSION_ID,
    "at": AT,
    "baseline": {"ref": "main", "sha": BASELINE_SHA},
    "branch": f"gymrat/{SESSION_ID}",
    "worktrees": {
        "experiment": "/repo/.gymrat/experiment",
        "baseline": "/repo/.gymrat/baseline",
    },
    "config": {
        "bench": "npm run bench",
        "adapter": "metric-lines",
        "samples": 10,
        "timeout_seconds": 1800,
        "primary": "geomean",
    },
}

BASELINE_RECORD: dict[str, object] = {
    "type": "baseline",
    "at": AT,
    "label": "main",
    "samples": [{"total_ms": 15200}, {"total_ms": 15184}],
}

METRIC_VERDICT: dict[str, object] = {
    "delta_pct": -7.2,
    "verdict": "improved",
    "method": "permutation",
    "p": 0.002,
    "noise_pct": 1.4,
    "gating": True,
    "confirmed": False,
}

ITERATION_RECORD: dict[str, object] = {
    "type": "iteration",
    "seq": 1,
    "at": AT,
    "samples": {
        "experiment": [{"total_ms": 14100}, {"total_ms": 14088}],
        "baseline": [{"total_ms": 15200}, {"total_ms": 15190}],
    },
    "metrics": {"total_ms": METRIC_VERDICT},
    "primary": {"kind": "geomean", "delta_pct": -7.2},
    "outcome": "improved",
    "target_reached": False,
}


DISCARD_RECORD: dict[str, object] = {"type": "discard", "seq": 3, "at": AT}


COMMAND_RECORD: dict[str, object] = {
    "type": "command",
    "at": AT,
    "name": "iterate",
    "args": {},
    "exit_code": 1,
    "reason": "budget-exceeded",
    "duration_ms": 1840,
    "origin": "cli",
    "seq": 3,
}

COMMAND_RECORD_SUCCESS: dict[str, object] = {
    "type": "command",
    "at": AT,
    "name": "keep",
    "args": {"message": "cache the regex"},
    "exit_code": 0,
    "duration_ms": 520,
    "origin": "cli",
    "seq": 1,
}


def omitting(record: dict[str, object], key: str) -> dict[str, object]:
    """Copy of ``record`` without ``key``."""
    clone = dict(record)
    del clone[key]
    return clone


def patching(record: dict[str, object], patch: dict[str, object]) -> dict[str, object]:
    """Copy of ``record`` with ``patch`` merged over it."""
    return record | patch


def nested_with(record: dict[str, object], field: str, **overrides: object) -> dict[str, object]:
    """Copy of ``record`` with ``overrides`` merged over the object it holds under ``field``."""
    return patching(record, {field: {**field_of(record, field), **overrides}})


def verdict_with(**overrides: object) -> dict[str, object]:
    """``ITERATION_RECORD`` whose ``total_ms`` verdict has ``overrides`` merged over it."""
    return patching(ITERATION_RECORD, {"metrics": {"total_ms": {**METRIC_VERDICT, **overrides}}})


def field_of(record: dict[str, object], key: str) -> dict[str, object]:
    """The nested object ``record`` holds under ``key``.

    Args:
        record: The wire record to read from.
        key: The field holding the nested object.

    Returns:
        The nested object.

    Raises:
        TypeError: When the value under ``key`` is not an object.
    """
    value = record[key]
    if not isinstance(value, dict):
        msg = f"{key!r} holds {type(value).__name__}, not an object"
        raise TypeError(msg)
    return value
