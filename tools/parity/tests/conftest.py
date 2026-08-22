"""Shared fixtures for the parity-harness tests.

These tests live outside the project's ``testpaths`` and run by hand via
``uv run pytest tools/parity/tests``. They import the harness as
``tools.parity.<module>``, so the repository root must be importable; ``uv run
pytest`` does not add it, so this conftest inserts it explicitly.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# tools/parity/tests/conftest.py -> repo root is three levels up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _node_available() -> bool:
    return shutil.which("node") is not None


def _ts_repo_available() -> bool:
    from tools.parity.oracle import PINNED_ORACLE_SHA, ts_repo_path

    try:
        repo = ts_repo_path()
    except Exception:  # noqa: BLE001 -- any resolution failure means "unavailable, skip"
        return False
    if not repo.is_dir():
        return False
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return False
    return head == PINNED_ORACLE_SHA


@pytest.fixture
def requires_oracle():
    """Skip cleanly when node or the pinned reference checkout is unavailable."""
    if not _node_available():
        pytest.skip("node is not available on PATH")
    if not _ts_repo_available():
        pytest.skip("reference checkout is unavailable or not at the pinned commit")
