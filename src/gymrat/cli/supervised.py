"""Supervised-run detection and the guard that refuses shell-typed commands during one.

A supervised run is live while its budget is live (see
:func:`~gymrat.session.budget.read_budget`). During that window the agent must
drive the loop through its tools, so a command whose origin is not ``tool`` is
refused with a hint pointing at the matching tool.
"""

import os

from gymrat.errors import GymratError
from gymrat.session import clock
from gymrat.session.budget import read_budget
from gymrat.session.schema import CommandOrigin

__all__ = [
    "command_origin",
    "guard_supervised_origin",
    "is_supervised_run_live",
]


def command_origin() -> CommandOrigin:
    """The running command's origin: ``tool`` only when the supervisor says so, else ``cli``."""
    return "tool" if os.environ.get("GYMRAT_COMMAND_ORIGIN") == "tool" else "cli"


def is_supervised_run_live(root: str) -> bool:
    """Whether a supervised run with a live budget owns the repository at ``root``."""
    return read_budget(root, now_ms=clock.now_ms()) is not None


def guard_supervised_origin(root: str, command: str) -> None:
    """Refuse a command typed outside the supervisor while a supervised run is live.

    Args:
        root: Repository root whose budget decides whether a supervised run is live.
        command: Name of the command being run, echoed in the refusal.

    Raises:
        GymratError: When a supervised run is live and the command's origin is
            not ``tool``; carries the ``supervised-use-tool`` reason.
    """
    if command_origin() == "tool" or not is_supervised_run_live(root):
        return
    message = f"a supervised run is live; use the {command} tool"
    raise GymratError(
        message,
        hint="Call the tool instead of the command.",
        reason="supervised-use-tool",
    )
