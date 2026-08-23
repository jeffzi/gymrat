"""Shared warning sink for callers reporting output they could not read.

Defines the warning-sink type and its stderr default, kept adapter-agnostic so
any part of the codebase can route a complaint about unreadable output through
the same seam.
"""

import sys
from collections.abc import Callable

type WarnSink = Callable[[str], None]
"""Where a caller sends a complaint about output it could not read.

The caller owns the destination so a warning can be interleaved with whatever
else is on the terminal — the CLI's progress line, for one — instead of landing
on stderr wherever the cursor happens to be.
"""


def warn_to_stderr(message: str) -> None:
    """Default :data:`WarnSink` for callers called without an explicit one.

    Writes ``message`` followed by a newline to stderr.
    """
    sys.stderr.write(f"{message}\n")
