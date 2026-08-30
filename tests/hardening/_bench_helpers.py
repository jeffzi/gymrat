"""Shared shell-bench helpers for the hardening test modules.

Every hardening test spawns the CLI out of process against a throwaway git
repo and a real ``sh`` bench script, then polls a pid or drains a pty. This
module is the shared plumbing so each hardening test module does not carry
its own copy of the same subprocess and pid-checking code.
"""

import os
from pathlib import Path

from tests._git import git


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
