"""Interactive yes/no confirmation prompt for destructive actions.

The prompt is written to stderr so stdout stays reserved for machine-readable
data — a caller piping stdout still sees the question on their terminal. Only an
exact ``y`` or ``Y`` counts as consent; every other answer, including an empty
line or end-of-input, is a decline.
"""

import sys
from typing import TextIO


def confirm_action(message: str, stream: TextIO) -> bool:
    """Prompt on stderr and return whether the user consented.

    Writes ``f"{message} [y/N] "`` to stderr, then reads one line from
    ``stream``. Anything short of an explicit yes declines, so a destructive
    action never proceeds on an unseen or unanswered question.

    Args:
        message: The question to display before the ``[y/N]`` suffix.
        stream: The stream to read the answer line from.

    Returns:
        ``True`` only when the line is exactly ``y`` or ``Y`` (ignoring the
        trailing line terminator). ``False`` otherwise, including at
        end-of-input and when stderr cannot be written to (a closed or broken
        pipe).
    """
    try:
        sys.stderr.write(f"{message} [y/N] ")
        sys.stderr.flush()
    except OSError:
        # The user never saw the question, so there is no consent to act on:
        # decline rather than let the write error escape and abort the caller
        # with a status that claims the destructive action succeeded.
        return False
    answer = stream.readline().rstrip("\r\n")
    return answer in ("y", "Y")
