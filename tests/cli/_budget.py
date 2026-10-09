"""Shared command-origin helpers for the CLI command test files.

Builders used by more than one CLI test module to exercise the
supervised-run refusal. The budget-file builders live in
:mod:`tests.session._budget`. This is test-support code, not a test module: it
carries no test functions or pytest fixtures of its own.
"""

import pytest

#: The hint every supervised-run refusal attaches, pointing the caller at the tool.
SUPERVISED_HINT = "Call the tool instead of the command."


def set_origin(monkeypatch: pytest.MonkeyPatch, origin: str | None) -> None:
    """Set ``GYMRAT_COMMAND_ORIGIN`` to ``origin``; None leaves it unset."""
    if origin is not None:
        monkeypatch.setenv("GYMRAT_COMMAND_ORIGIN", origin)
