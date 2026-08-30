"""Shared CLI subprocess constants and helpers for out-of-process test modules."""

import json
import os
import sys
from pathlib import Path
from typing import Any

ENTRY = [sys.executable, "-m", "gymrat.cli.app"]
"""The command that launches the CLI the way a user's shell would."""


def no_color_env() -> dict[str, str]:
    """A child environment with color forced off for deterministic output."""
    env = dict(os.environ)
    env["NO_COLOR"] = "1"
    env.pop("FORCE_COLOR", None)
    return env


def try_read_report(report_path: Path) -> dict[str, Any] | None:
    """Load the JSON report if it exists and is complete, else ``None``.

    Wrapped in a sync helper so the blocking filesystem read stays out of the
    async test body, where it would trip the async-blocking-call lint.
    """
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
