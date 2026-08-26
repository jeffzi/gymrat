"""Shared shell-bench helpers for the hardening test modules.

Every hardening test spawns the CLI out of process against a throwaway git
repo and a real ``sh`` bench script, then polls a pid or drains a pty. This
module is the shared plumbing so each hardening test module does not carry
its own copy of the same subprocess and pid-checking code.
"""

import os
import subprocess
from pathlib import Path


def env() -> dict[str, str]:
    """A child environment with color forced off, so output is deterministic."""
    result = dict(os.environ)
    result["NO_COLOR"] = "1"
    result.pop("FORCE_COLOR", None)
    return result


def git(repo: str, *args: str) -> None:
    subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def write_committed_bench(repo: str, script: str) -> None:
    """Drop ``script`` as ``bench.sh`` and commit it so every ref can run it."""
    (Path(repo) / "bench.sh").write_text(script, encoding="utf-8")
    git(repo, "add", "bench.sh")
    git(repo, "commit", "-m", "add bench")


def drain(fd: int, chunks: list[bytes]) -> None:
    """Read a pty master until the child closes the slave, collecting bytes."""
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            return
        if not chunk:
            return
        chunks.append(chunk)
