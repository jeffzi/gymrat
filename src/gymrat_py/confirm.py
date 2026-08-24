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
    ``stream``. Returns ``True`` only when that line is exactly ``y`` or ``Y``
    (ignoring the trailing line terminator). End-of-input reads as an empty line
    and declines.
    """
    sys.stderr.write(f"{message} [y/N] ")
    answer = stream.readline().rstrip("\r\n")
    return answer in ("y", "Y")
