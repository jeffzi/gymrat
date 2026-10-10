"""Supervised-run detection and the guard that refuses shell-typed commands during one.

A supervised run is live while its budget is live (see
:func:`~gymrat.session.budget.read_budget`). During that window the agent must
drive the loop through its tools, so a command whose origin is not ``tool`` is
refused with a hint pointing at the matching tool.
"""

from gymrat import clock
from gymrat.command_run import command_origin
from gymrat.errors import GymratError
from gymrat.session.budget import read_budget

__all__ = [
    "guard_supervised_origin",
    "is_supervised_run_live",
]


def is_supervised_run_live(root: str) -> bool:
    """Whether a supervised run with a live budget owns the repository at ``root``.

    Args:
        root: Repository root whose budget and supervise lock are read.

    Returns:
        ``True`` when a supervised run with a live budget holds the repository.

    Raises:
        GymratError: When the supervise lock file cannot be opened, so whether
            a supervised run holds it is unknown.
    """
    return read_budget(root, now_ms=clock.now_ms()) is not None


def guard_supervised_origin(root: str, command: str) -> None:
    """Refuse a command typed outside the supervisor while a supervised run is live.

    Args:
        root: Repository root whose budget decides whether a supervised run is live.
        command: Name of the command being run, echoed in the refusal.

    Raises:
        GymratError: When a supervised run is live and the command's origin is
            not ``tool``; carries the ``supervised-use-tool`` reason. Also when the
            supervise lock file cannot be opened, so whether a supervised run holds it is
            unknown.
    """
    if command_origin() == "tool" or not is_supervised_run_live(root):
        return
    message = f"a supervised run is live; use the {command} tool"
    raise GymratError(
        message,
        hint="Call the tool instead of the command.",
        reason="supervised-use-tool",
    )
