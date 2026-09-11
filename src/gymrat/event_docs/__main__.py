"""Entry point for ``python -m gymrat.event_docs``.

Regenerates all event-documentation artifacts under the repo root and prints
one line per written file.
"""

import sys

from gymrat.cli.shared import TOOL_FAILURE_EXIT_CODE
from gymrat.errors import GymratError, hint_of
from gymrat.event_docs import write_all
from gymrat.session.paths import repo_root


def main() -> None:
    """Write artifacts to the repo root and print each path."""
    try:
        root = repo_root()
    except GymratError as exc:
        print(exc, file=sys.stderr)  # noqa: T201 — CLI entry point
        hint = hint_of(exc)
        if hint is not None:
            print(hint, file=sys.stderr)  # noqa: T201 — CLI entry point
        sys.exit(TOOL_FAILURE_EXIT_CODE)
    paths = write_all(root)
    for path in paths:
        print(path)  # noqa: T201 — CLI entry point


if __name__ == "__main__":
    main()
