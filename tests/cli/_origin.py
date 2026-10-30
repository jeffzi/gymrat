"""Command-origin helpers for the CLI tests of the supervised-run refusal.

The origin says whether the shell or the agent's tool ran a command; while a
supervised run is live, the shell is refused. The budget-file builders that make
a run live are in :mod:`tests.session._budget`. This is test-support code, not
a test module: it carries no test functions or pytest fixtures of its own.
"""

import pytest


def set_origin(monkeypatch: pytest.MonkeyPatch, origin: str) -> None:
    """Set ``GYMRAT_COMMAND_ORIGIN`` to ``origin``."""
    monkeypatch.setenv("GYMRAT_COMMAND_ORIGIN", origin)
