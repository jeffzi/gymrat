"""Skip markers for tests that need a POSIX host.

Each marker names the platform capability a test relies on, so tests that share
a reason share one marker instead of redeclaring the same ``skipif``.
"""

import sys

import pytest

needs_posix_shell = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell")

needs_symlinks = pytest.mark.skipif(
    sys.platform == "win32", reason="creating symlinks needs extra privileges on Windows"
)

needs_posix_worktrees = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only worktrees and gating"
)

needs_named_pipes = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only named pipes")

needs_posix_kill = pytest.mark.skipif(
    sys.platform == "win32", reason="post-checkout SIGKILL is POSIX-only"
)
