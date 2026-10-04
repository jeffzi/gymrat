"""Shared git helpers for test fixtures and test modules."""

import subprocess
from pathlib import Path

#: A bench script that reports one metric line and exits cleanly.
EMIT_ONE_BENCH = "#!/bin/sh\necho 'METRIC x=1'\n"


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


def head_of(worktree: str) -> str:
    """The commit ``worktree`` currently has checked out."""
    return run_git(["rev-parse", "HEAD"], worktree)


def write_committed_bench(repo: str, script: str, *, message: str = "add bench") -> None:
    """Drop ``script`` as ``bench.sh`` and commit it so every ref can run it."""
    (Path(repo) / "bench.sh").write_text(script, encoding="utf-8")
    run_git(["add", "bench.sh"], repo)
    run_git(["commit", "-m", message], repo)
