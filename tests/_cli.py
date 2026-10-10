"""Shared CLI constants and helpers: subprocess launches, and reading a run's output."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

from tests._ansi import strip_ansi

if TYPE_CHECKING:
    from pathlib import Path

    from typer.testing import Result

ENTRY = [sys.executable, "-m", "gymrat.cli.app"]
"""The command that launches the CLI the way a user's shell would."""


def no_color_env() -> dict[str, str]:
    """A child environment with color forced off for deterministic output."""
    env = dict(os.environ)
    env["NO_COLOR"] = "1"
    env.pop("FORCE_COLOR", None)
    return env


def run_module(
    module: str, *args: str, cwd: str | Path | None = None, check: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run ``python -m <module> <args>`` in a child process with color forced off.

    Args:
        module: The module to run as ``__main__``.
        *args: The command line after the module name.
        cwd: The directory the child runs in; ``None`` means the current one.
        check: Whether a non-zero exit raises ``CalledProcessError``.

    Returns:
        The finished child with text-decoded stdout and stderr.
    """
    return subprocess.run(  # noqa: S603 -- fixed interpreter plus test-chosen args
        [sys.executable, "-m", module, *args],
        cwd=cwd,
        env=no_color_env(),
        capture_output=True,
        text=True,
        check=check,
    )


def run_cli(
    args: list[str],
    cwd: str | Path,
    *,
    check: bool = True,
    timeout: float,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one gymrat command out of process in ``cwd``, blocking until it ends.

    Args:
        args: The command line after the program name.
        cwd: The directory the command runs in.
        check: Whether a non-zero exit fails the caller.
        timeout: Seconds the command may run before ``TimeoutExpired``.
        env: The child's environment; ``None`` means :func:`no_color_env`.

    Returns:
        The finished child with text-decoded stdout and stderr.

    Raises:
        AssertionError: When ``check`` holds and the command exits non-zero; the
            message carries the child's stderr.
    """
    try:
        return subprocess.run(  # noqa: S603 -- fixed interpreter plus test-chosen args
            [*ENTRY, *args],
            cwd=cwd,
            env=no_color_env() if env is None else env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )
    except subprocess.CalledProcessError as error:
        detail = f"gymrat {' '.join(args)} failed (exit {error.returncode}): {error.stderr}"
        raise AssertionError(detail) from error


def err_text(result: Result) -> str:
    """The combined stdout and stderr of a ``CliRunner`` run, stripped of color."""
    return strip_ansi((result.stdout or "") + (result.stderr or ""))
