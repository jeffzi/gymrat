"""Error hierarchy for gymrat.

Every error raised by gymrat extends :class:`GymratError` rather than a bare
``Exception``. A single base class lets the CLI boundary catch and route the
whole family in one place, and it gives every error an optional ``hint`` field
carrying a human-facing next step alongside the machine-facing message.

Exit-code routing contract (enforced by ``exit_with_error`` in ``gymrat.cli.exit``):

- An uncaught ``GymratError`` — including any subclass such as
  :class:`CommandError` — maps to exit code ``2`` (``TOOL_FAILURE_EXIT_CODE``).
- A gate trip maps to exit code ``1`` (``GATE_EXIT_CODE``).

Anything else escaping the boundary is an unexpected crash and is not covered by
this contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gymrat.session.schema import CommandReason

GATE_EXIT_CODE = 1
"""Exit status for a tripped gate, also the ``CommandRecord.exit_code`` recorded for it.

``with_repo_lock`` records this code when the command body returns with ``CommandTrace.gate``
set, raises ``LoopStopError``, or exits with ``typer.Exit(1)``. Recording does not set the
process status: the command must itself exit with this code for the two to agree.
"""

TOOL_FAILURE_EXIT_CODE = 2
"""Exit status for a tool failure, including any uncaught :class:`GymratError`."""


class GymratError(Exception):
    """Base class for every error gymrat raises.

    Follows the standard ``Exception`` calling convention — positional ``*args``
    become ``err.args`` and drive ``str(err)`` — extended with optional
    keyword-only ``hint`` and ``reason`` fields.

    Args:
        *args: Passed through to ``Exception``; a lone string message is the
            common case, and ``str(err)`` then returns exactly that message.
        hint: An optional human-facing suggestion for what to do next. ``None``
            when no hint applies.
        reason: A :data:`~gymrat.session.schema.CommandReason` tag classifying
            why the error was raised. ``None`` when no classification applies;
            subclasses may override the default.
    """

    def __init__(
        self,
        *args: object,
        hint: str | None = None,
        reason: CommandReason | None = None,
    ) -> None:
        super().__init__(*args)
        self.hint = hint
        self.reason = reason


class CommandError(GymratError):
    """A subprocess command invoked by gymrat failed."""
