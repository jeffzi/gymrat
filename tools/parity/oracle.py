"""Locate, build, and invoke the pinned reference gymrat CLI.

The parity harness diffs the shipped TypeScript CLI (the *oracle*) against the
Python port. This module owns the oracle side: it resolves the reference
checkout, asserts it sits on the pinned commit, builds its Node entry point when
needed, and exposes a uniform :class:`Runner` interface so the differ can drive
either side through the same call shape.
"""

import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gymrat_py.errors import CommandError, GymratError

# The reference CLI is verified against this exact commit; every parity run pins
# to it so oracle output stays reproducible across machines.
PINNED_ORACLE_SHA = "b55e31b0ced0bfbbeefbd14b836ff2dda73097b8"

# tools/parity/oracle.py -> repo root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Captured outcome of a subprocess invocation."""

    exit_code: int
    stdout: str
    stderr: str


def _run(cmd: Sequence[str], cwd: Path, env: dict[str, str] | None = None) -> RunResult:
    """Run a fixed-argv command and capture its streams.

    This is the single subprocess seam for the module: every external command
    routes through here so tests can substitute it wholesale.

    Args:
        cmd: The argv to execute; never interpreted by a shell.
        cwd: The working directory for the child process.
        env: The child environment, or ``None`` to inherit this process's.

    Returns:
        The captured exit code, stdout, and stderr.
    """
    completed = subprocess.run(  # noqa: S603
        list(cmd),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return RunResult(
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def ts_repo_path() -> Path:
    """Resolve the reference TypeScript repository.

    Reads ``$GYMRAT_TS_REPO`` when set, otherwise defaults to the sibling
    ``ts/gymrat`` checkout next to this project's repo root.

    Returns:
        The resolved reference repository directory.

    Raises:
        GymratError: When the resolved directory does not exist.
    """
    override = os.environ.get("GYMRAT_TS_REPO")
    repo = Path(override) if override is not None else _REPO_ROOT.parent / "ts" / "gymrat"
    if not repo.is_dir():
        msg = (
            f"reference repo not found at {repo}; "
            "set GYMRAT_TS_REPO to a valid gymrat TypeScript checkout"
        )
        raise GymratError(
            msg,
            hint="clone the gymrat TypeScript repo and point GYMRAT_TS_REPO at it",
        )
    return repo


def assert_pinned_sha(repo: Path) -> None:
    """Verify ``repo`` is checked out at :data:`PINNED_ORACLE_SHA`.

    Args:
        repo: The reference repository directory to inspect.

    Raises:
        CommandError: When ``git rev-parse HEAD`` fails.
        GymratError: When the repository HEAD differs from the pinned commit.
    """
    result = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    if result.exit_code != 0:
        msg = f"could not read HEAD of {repo}: {result.stderr.strip()}"
        raise CommandError(msg)
    head = result.stdout.strip()
    if head != PINNED_ORACLE_SHA:
        msg = f"reference repo {repo} is at {head}, expected pinned {PINNED_ORACLE_SHA}"
        raise GymratError(
            msg,
            hint=f"check out {PINNED_ORACLE_SHA} in the reference repo",
        )


def ensure_built(repo: Path, *, force: bool = False) -> Path:
    """Return the path to the built CLI entry point, building it if necessary.

    Args:
        repo: The reference repository directory.
        force: When ``True``, rebuild even if the entry point already exists.

    Returns:
        The path to ``repo/dist/cli.js``.

    Raises:
        CommandError: When a build command exits non-zero.
    """
    cli_js = repo / "dist" / "cli.js"
    if cli_js.exists() and not force:
        return cli_js
    for cmd in (["npm", "ci", "--ignore-scripts"], ["npm", "run", "build"]):
        result = _run(cmd, cwd=repo)
        if result.exit_code != 0:
            msg = f"{' '.join(cmd)} failed in {repo}: {result.stderr.strip()}"
            raise CommandError(msg)
    return cli_js


class Runner(Protocol):
    """A uniform interface for invoking either side of a parity comparison."""

    def run(self, args: Sequence[str], cwd: Path) -> RunResult:
        """Invoke the underlying CLI with ``args`` in ``cwd`` and capture its streams."""
        ...


class OracleRunner:
    """Run the reference CLI via its built Node entry point."""

    def __init__(self, cli_js: Path) -> None:
        self._cli_js = cli_js

    def run(self, args: Sequence[str], cwd: Path) -> RunResult:
        """Invoke ``node <cli.js> <args...>`` with color output disabled.

        The child environment forces ``NO_COLOR`` and strips ``FORCE_COLOR`` so
        the oracle emits plain, deterministic text regardless of the parent
        shell's terminal settings.

        Args:
            args: Arguments passed to the reference CLI.
            cwd: The working directory for the invocation.

        Returns:
            The captured exit code, stdout, and stderr.
        """
        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        env.pop("FORCE_COLOR", None)
        return _run(["node", str(self._cli_js), *args], cwd=cwd, env=env)


class PortRunner:
    """Placeholder runner for the port CLI, which ships in v0.8."""

    def run(self, args: Sequence[str], cwd: Path) -> RunResult:  # noqa: ARG002
        """Signal that the port CLI is not yet available.

        Args:
            args: Ignored; accepted to satisfy the :class:`Runner` interface.
            cwd: Ignored; accepted to satisfy the :class:`Runner` interface.

        Raises:
            GymratError: Always — the port CLI does not exist until v0.8.
        """
        msg = "the port CLI is not built yet; it ships in v0.8"
        raise GymratError(
            msg,
            hint="run the oracle side only until the v0.8 CLI lands",
        )
