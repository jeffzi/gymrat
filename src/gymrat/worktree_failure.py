"""A worktree that cleanup could not remove, as every report of the cleanup names it."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorktreeRemovalFailure:
    """A worktree cleanup could not remove, with the reason git gave.

    Attributes:
        dir: The worktree directory that could not be removed.
        error: The reason git reported for the failed removal.
    """

    dir: str
    error: str
