"""Error hierarchy for gymrat.

Every error raised by gymrat extends :class:`GymratError` rather than a bare
``Exception``. A single base class lets the CLI boundary catch and route the
whole family in one place, and it gives every error an optional ``hint`` field
carrying a human-facing next step alongside the machine-facing message.

Exit-code routing contract (enforced at the v0.8 CLI boundary):

- An uncaught ``GymratError`` — including any subclass such as
  :class:`CommandError` — maps to exit code ``2``.
- A gate trip maps to exit code ``1``.

Anything else escaping the boundary is an unexpected crash and is not covered by
this contract.
"""


class GymratError(Exception):
    """Base class for every error gymrat raises.

    Follows the standard ``Exception`` calling convention — positional ``*args``
    become ``err.args`` and drive ``str(err)`` — extended with an optional
    keyword-only ``hint`` carrying a human-facing next step alongside the
    machine-facing message.

    Args:
        *args: Passed through to ``Exception``; a lone string message is the
            common case, and ``str(err)`` then returns exactly that message.
        hint: An optional human-facing suggestion for what to do next. ``None``
            when no hint applies.
    """

    def __init__(self, *args: object, hint: str | None = None) -> None:
        super().__init__(*args)
        self.hint = hint


class CommandError(GymratError):
    """A subprocess command invoked by gymrat failed."""
