"""Shared start barrier for the hardening tests that race real processes.

Each racing child parks on one named pipe after leaving a ``ready.<pid>``
marker beside it; the parent writes the go bytes only once every marker exists,
so every child starts its contended step within the same scheduling window.
The child side ships as source the child scripts embed, because the children
run out of process; the parent side is :func:`racing_children`.
"""

import contextlib
import os
import subprocess
from collections.abc import Callable, Generator, Sequence
from pathlib import Path

from tests._process_helpers import poll_until_blocking, reaped, spawn_child_script

#: Child-script source defining ``wait_at_barrier(barrier_path)``. A child script
#: starts with it and calls ``wait_at_barrier`` right before its contended step.
CHILD_BARRIER = """\
import os
from pathlib import Path


def wait_at_barrier(barrier_path):
    # Unbuffered read so each child consumes exactly one go byte; a buffered
    # reader would slurp the whole pipe and starve its siblings. The ready
    # marker tells the parent this child is parked on the pipe.
    barrier_fd = os.open(barrier_path, os.O_RDONLY)
    Path(barrier_path).with_name(f"ready.{os.getpid()}").touch()
    os.read(barrier_fd, 1)
    os.close(barrier_fd)

"""


def _create_barrier(directory: Path) -> Path:
    """Create the named pipe the racing children park on.

    Args:
        directory: Where the pipe and the children's ready markers live.

    Returns:
        The path of the new named pipe.
    """
    barrier = directory / "barrier.pipe"
    os.mkfifo(barrier)
    return barrier


def _release_together(
    barrier: Path,
    count: int,
    *,
    on_poll: Callable[[], None] | None = None,
    timeout_s: float = 30.0,
) -> None:
    """Wait for ``count`` children to park on ``barrier``, then release them at once.

    Args:
        barrier: The named pipe from :func:`_create_barrier`.
        count: How many children must leave a ready marker before the release.
        on_poll: Called on every wait tick, so a caller can fail fast on a
            child that died before reaching the barrier.
        timeout_s: Ceiling on the wait for every child to park.

    Raises:
        AssertionError: Fewer than ``count`` children parked within ``timeout_s``.
    """

    def all_parked() -> bool:
        if on_poll is not None:
            on_poll()
        return len(list(barrier.parent.glob("ready.*"))) >= count

    go_fd = os.open(str(barrier), os.O_RDWR)
    try:
        poll_until_blocking(
            all_parked,
            timeout_s,
            lambda: AssertionError(
                f"fewer than {count} children reached the barrier in {barrier.parent}"
            ),
        )
        os.write(go_fd, b"\x00" * count)
    finally:
        os.close(go_fd)


@contextlib.contextmanager
def racing_children(
    directory: Path,
    script: str,
    args_for: Callable[[Path, int], Sequence[str]],
    count: int,
    *,
    on_poll: Callable[[list[subprocess.Popen[str]]], None] | None = None,
    timeout_s: float = 30.0,
) -> Generator[list[subprocess.Popen[str]]]:
    """Start ``count`` children of ``script`` and release them through one barrier together.

    Every child is killed and reaped on the way out if it is still running, so
    none outlives the block.

    Args:
        directory: Where the barrier, the ready markers and the child scripts live.
        script: The child source; it starts with :data:`CHILD_BARRIER` and calls
            ``wait_at_barrier`` right before its contended step.
        args_for: Builds a child's command-line arguments from the barrier path
            and the child's index.
        count: How many children race.
        on_poll: Called with the children on every wait tick, so a caller can
            fail fast on a child that died before reaching the barrier.
        timeout_s: Ceiling on the wait for every child to park.

    Yields:
        The released children, in index order.

    Raises:
        AssertionError: Fewer than ``count`` children parked within ``timeout_s``.
    """
    barrier = _create_barrier(directory)
    with contextlib.ExitStack() as stack:
        children = [
            stack.enter_context(
                reaped(
                    spawn_child_script(
                        directory, f"race_child_{index}", script, *args_for(barrier, index)
                    )
                )
            )
            for index in range(count)
        ]
        _release_together(
            barrier,
            count,
            on_poll=None if on_poll is None else lambda: on_poll(children),
            timeout_s=timeout_s,
        )
        yield children
