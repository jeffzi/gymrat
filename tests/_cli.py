"""Shared CLI subprocess constants for out-of-process test modules."""

import os
import sys

ENTRY = [sys.executable, "-m", "gymrat.cli.app"]
"""The command that launches the CLI the way a user's shell would."""


def no_color_env() -> dict[str, str]:
    """A child environment with color forced off for deterministic output."""
    env = dict(os.environ)
    env["NO_COLOR"] = "1"
    env.pop("FORCE_COLOR", None)
    return env
