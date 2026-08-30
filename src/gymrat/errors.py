"""Error hierarchy for gymrat.

Every error raised by gymrat extends :class:`GymratError` rather than a bare
``Exception``. A single base class lets the CLI boundary catch and route the
whole family in one place, and it gives every error an optional ``hint`` field
carrying a human-facing next step alongside the machine-facing message.

Exit-code routing contract (enforced by ``exit_with_error`` in ``gymrat.cli.shared``):

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


def hint_of(error: object) -> str | None:
    """Extract the ``hint`` from a :class:`GymratError`, else ``None``.

    Args:
        error: The caught value.

    Returns:
        The error's ``hint`` when it is a :class:`GymratError`, otherwise
        ``None`` — no other thrown value carries a hint.
    """
    return error.hint if isinstance(error, GymratError) else None


def _stream_text_of(error: Exception, stream_name: str) -> str | None:
    """The trimmed text of a captured child stream, or ``None`` when blank."""
    stream = getattr(error, stream_name, None)
    if isinstance(stream, bytes):
        # Captured streams are bytes unless the call ran in text mode; replace
        # undecodable sequences so diagnostics never raise here.
        stream = stream.decode("utf-8", errors="replace")
    if isinstance(stream, str) and stream.strip():
        return stream.strip()
    return None


def stderr_text_of(error: object) -> str:
    """The diagnostics a failed child process wrote to its output streams.

    ``subprocess.CalledProcessError`` and ``TimeoutExpired`` attach the child's
    captured output separately from the exception message, which carries
    ``Command '...' returned non-zero exit status`` noise. Preferring the raw
    stderr keeps git's own diagnostics — the text repository-lookup
    classification keys on — instead of that wrapper noise.

    Falls back to the captured stdout when stderr is blank: some tools explain
    a failure on stdout instead (git's "nothing to commit", a commit hook's
    rejection message), leaving stderr an empty or whitespace-only string. Only
    when neither stream carries text does the argv-shaped ``str(error)`` stand
    in.

    Args:
        error: The caught value.

    Returns:
        The trimmed stderr when it is present and non-blank, else the trimmed
        stdout on the same terms, else the message.
    """
    if isinstance(error, Exception):
        for stream_name in ("stderr", "stdout"):
            text = _stream_text_of(error, stream_name)
            if text is not None:
                return text

    return str(error)
