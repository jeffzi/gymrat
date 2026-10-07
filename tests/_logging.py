"""Shared logging test helpers."""

import contextlib
import logging
from collections.abc import Generator
from unittest.mock import patch


@contextlib.contextmanager
def unhandled_logging() -> Generator[None]:
    """Run the block with no logging handler configured, as a plain ``gymrat`` process does.

    A record nothing handles falls to Python's last-resort handler, which
    writes it to ``sys.stderr``. pytest attaches its capture handlers to the
    root logger anew for each test phase, so the handlers are lifted here, in
    the test body, rather than in a fixture.
    """
    no_handlers: list[logging.Handler] = []
    with patch.object(logging.getLogger(), "handlers", no_handlers):
        yield
