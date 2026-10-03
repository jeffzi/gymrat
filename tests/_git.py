"""Shared git subprocess helper for test fixtures and test modules."""

import subprocess


def run_git(args: list[str], cwd: str) -> str:
    """Run git in ``cwd``, returning its stripped stdout and failing loudly on error."""
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
