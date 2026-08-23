"""Benchmark targets: the two things a run can compare against.

A target names *what* to benchmark. Either the working tree as it stands, or a
committed ref materialized into its own worktree. The variant class is the
discriminant: ``isinstance`` checks distinguish them, so no separate tag field is
carried.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InPlaceTarget:
    """The working tree exactly as it is on disk.

    Attributes:
        dir: The directory the benchmark runs in.
    """

    dir: str


@dataclass(frozen=True, slots=True)
class RefTarget:
    """A committed ref, benchmarked from a worktree checked out at its commit.

    Attributes:
        ref: The ref the user named (branch, tag, or revision expression).
        resolved_sha: The commit the ref resolved to when the run began.
    """

    ref: str
    resolved_sha: str


type Target = InPlaceTarget | RefTarget
"""Either the working tree in place or a committed ref in its own worktree."""


@dataclass(frozen=True, slots=True)
class WorktreeRemovalFailure:
    """A worktree cleanup could not remove, with the reason git gave.

    Attributes:
        dir: The worktree directory that could not be removed.
        error: The reason git reported for the failed removal.
    """

    dir: str
    error: str
