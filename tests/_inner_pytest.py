"""Child pytest runs under the project conftest, for tests that check what one test leaves the next."""

import os
import subprocess
import sys
from pathlib import Path

#: Directory holding the ``tests`` package, which the child needs on ``PYTHONPATH``.
REPO_ROOT = Path(__file__).resolve().parent.parent

SINGLE_PROCESS_ARGS = ["-p", "no:xdist"]


def inherited_env() -> dict[str, str]:
    """The parent's environment with the repository root on ``PYTHONPATH``."""
    return {**os.environ, "PYTHONPATH": str(REPO_ROOT)}


def run_inner_pytest(
    tmp_path: Path, source: str, env: dict[str, str], extra_args: list[str]
) -> subprocess.CompletedProcess[str]:
    """Run ``source`` as one test file in a child pytest that loads the project conftest.

    The child ignores the project's pytest configuration and runs its tests in
    file order, so each one after the first sees what the one before it left
    behind.

    Args:
        tmp_path: Directory the test file is written to and the child runs in.
        source: Contents of the test file.
        env: The child's whole environment; it must put the repository root on
            ``PYTHONPATH`` for ``tests.conftest`` to import.
        extra_args: Further pytest arguments, such as :data:`SINGLE_PROCESS_ARGS`.

    Returns:
        The finished child, with its output captured as text.
    """
    test_file = tmp_path / "test_inner.py"
    test_file.write_text(source, encoding="utf-8")
    # With `-c os.devnull` the conftest cutoff is the null device's directory, so
    # collection would list every ancestor of tmp_path, the shared temp
    # directory included, and fail when another test deletes an entry mid-scan.
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        os.devnull,
        "--rootdir",
        str(tmp_path),
        "--confcutdir",
        str(tmp_path),
        "-p",
        "tests.conftest",
        "--import-mode=importlib",
        "-p",
        "no:randomly",
        *extra_args,
        "-q",
        str(test_file),
    ]
    return subprocess.run(  # noqa: S603 -- fixed interpreter plus test-chosen args
        command, cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
