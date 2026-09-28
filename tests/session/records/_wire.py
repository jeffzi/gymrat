"""Wire-format session records shared by the record parsing tests.

Each constant is a record exactly as it is written to the session log, before
``parse_record`` validates it, and the helpers derive invalid or partial
variants from them.
"""

import re

from tests.session.records._fixtures import AT, COMMIT, SESSION_ID

SHA = "a" * 40

SESSION_RECORD: dict[str, object] = {
    "type": "session",
    "schema": 1,
    "session_id": SESSION_ID,
    "at": AT,
    "baseline": {"ref": "main", "sha": SHA},
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
    return {**record, **patch}


def mentions(field: str) -> re.Pattern[str]:
    """Matches an error message that names ``field`` as the failing location."""
    return re.compile(rf"\b{re.escape(field)}\b")


def field_of(record: dict[str, object], key: str) -> dict[str, object]:
    """The nested object ``record`` holds under ``key``."""
    value = record[key]
    if not isinstance(value, dict):
        msg = f"{key!r} holds {type(value).__name__}, not an object"
        raise TypeError(msg)
    return value


def config_with(**overrides: object) -> dict[str, object]:
    """The session record's config object with ``overrides`` merged over it."""
    return patching(field_of(SESSION_RECORD, "config"), overrides)
