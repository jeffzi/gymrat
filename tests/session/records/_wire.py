"""Wire-format session records shared by the record parsing tests.

Each constant is a record exactly as it is written to the session log, before
``parse_record`` validates it, and the helpers derive invalid or partial
variants from them.
"""

import json

from tests.session.records._fixtures import (
    AT,
    BASELINE_SHA,
    COMMIT,
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

CONFIRM: dict[str, object] = {
    "ran": True,
    "filtered": ["total_ms"],
    "absent": ["rss_kb"],
    "samples": {
        "experiment": [{"total_ms": 14120}],
        "baseline": [{"total_ms": 15170}],
    },
}

COMMITTED_KEEP_RECORD: dict[str, object] = {
    "type": "keep",
    "seq": 1,
    "at": AT,
    "status": "committed",
    "commit": COMMIT,
    "message": "cache the regex",
    "checks": {"configured": True, "passed": True},
}

BLOCKED_KEEP_RECORD: dict[str, object] = {
    "type": "keep",
    "seq": 2,
    "at": AT,
    "status": "blocked",
    "reason": "checks-failed",
    "checks": {"configured": True, "passed": False},
}

DISCARD_RECORD: dict[str, object] = {"type": "discard", "seq": 3, "at": AT}

HOOK_RECORD: dict[str, object] = {
    "type": "hook",
    "at": AT,
    "stage": "before",
    "seq": 4,
    "exit_code": 0,
    "duration_ms": 120,
    "stdout_bytes": 80,
    "timed_out": False,
}

FINALIZE_RECORD: dict[str, object] = {
    "type": "finalize",
    "at": AT,
    "branch": f"gymrat/{SESSION_ID}-final",
    "commit": COMMIT,
    "message": "squash 3 kept iterations",
}

STOP_RECORD: dict[str, object] = {
    "type": "stop",
    "at": AT,
    "message": "user requested stop",
}

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

COMMAND_RECORD_WITH_TRACEPARENT: dict[str, object] = {
    "type": "command",
    "at": AT,
    "name": "iterate",
    "args": {},
    "exit_code": 2,
    "reason": "error",
    "duration_ms": 100,
    "origin": "cli",
    "seq": 5,
    "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
}


def omitting(record: dict[str, object], key: str) -> dict[str, object]:
    """Copy of ``record`` without ``key``."""
    clone = dict(record)
    del clone[key]
    return clone


def patching(record: dict[str, object], patch: dict[str, object]) -> dict[str, object]:
    """Copy of ``record`` with ``patch`` merged over it."""
    return record | patch


def confirm_with(**overrides: object) -> dict[str, object]:
    """Copy of ``CONFIRM`` with ``overrides`` merged over it."""
    return {**CONFIRM, **overrides}


def verdict_with(**overrides: object) -> dict[str, object]:
    """``ITERATION_RECORD`` whose ``total_ms`` verdict has ``overrides`` merged over it."""
    return patching(ITERATION_RECORD, {"metrics": {"total_ms": {**METRIC_VERDICT, **overrides}}})


def verdict_without(*keys: str) -> dict[str, object]:
    """``ITERATION_RECORD`` whose ``total_ms`` verdict lacks every key in ``keys``."""
    verdict = {key: value for key, value in METRIC_VERDICT.items() if key not in keys}
    return patching(ITERATION_RECORD, {"metrics": {"total_ms": verdict}})


_RAW_NUMBER = "raw-number-placeholder"


def with_raw_number(line: str, keys: tuple[str, ...], literal: str) -> str:
    """Rewrite a JSON ``line`` so the value at ``keys`` is the bare number text ``literal``.

    ``json.dumps`` cannot emit ``NaN``-style or overflowing literals as bare
    numbers, so a placeholder string stands in for the value and is replaced,
    quotes included, once the line is dumped.

    Args:
        line: A JSON object line.
        keys: The key path from the top-level object to the value to replace.
        literal: The number text to splice in unquoted, such as ``"NaN"`` or ``"1e999"``.

    Returns:
        The rewritten JSON line.
    """
    data = json.loads(line)
    node = data
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = _RAW_NUMBER
    return json.dumps(data).replace(f'"{_RAW_NUMBER}"', literal)


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


def config_with(**overrides: object) -> dict[str, object]:
    """The session record's config object with ``overrides`` merged over it."""
    return patching(field_of(SESSION_RECORD, "config"), overrides)
