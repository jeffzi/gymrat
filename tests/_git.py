"""Shared git subprocess helpers for test fixtures and test modules."""

import subprocess


def run_git(args: list[str], cwd: str) -> str:
    """Run git in ``cwd`` for fixture setup, returning stdout and failing loudly."""
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def git(repo: str, *args: str) -> None:
    """Run a git command in ``repo``, discarding output and failing loudly."""
    subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
