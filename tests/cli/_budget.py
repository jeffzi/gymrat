"""Shared budget and command-origin helpers for the CLI command test files.

Builders used by more than one CLI test module to exercise the budget
time-left line, the JSON budget object, duration warnings, and the
supervised-run refusal. This is test-support code, not a test module: it
carries no test functions or pytest fixtures of its own.
"""

from pathlib import Path

import pytest

from gymrat.session.budget import Budget, write_budget

__all__ = [
    "LIVE_BUDGET",
    "SUPERVISED_HINT",
    "install_budget",
    "install_tight_budget",
    "mark_tool_origin",
    "set_origin",
]

#: A 30-minute budget whose deadline sits far in the future, so it never expires mid-test.
LIVE_BUDGET = Budget(started_at_ms=0.0, max_minutes=30, deadline_ms=9_999_999_999_999.0)

#: The hint every supervised-run refusal attaches, pointing the caller at the tool.
SUPERVISED_HINT = "Call the tool instead of the command."


def _install(repo: str, monkeypatch: pytest.MonkeyPatch, budget: Budget) -> None:
    """Write a budget file and patch the supervise lock so read_budget succeeds."""
    Path(repo, ".gymrat").mkdir(exist_ok=True)
    write_budget(repo, budget)
    # Target is a dotted string, so pyrefly can't check the lambda against
    # is_held's real signature.
    monkeypatch.setattr("gymrat.session.budget.is_held", lambda _path: True)  # pyrefly: ignore


def install_budget(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a live budget file and patch the supervise lock so read_budget succeeds."""
    _install(repo, monkeypatch, LIVE_BUDGET)


def install_tight_budget(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a budget with 5 minutes left and freeze the clock."""
    tight_budget = Budget(
        started_at_ms=0.0,
        max_minutes=30,
        deadline_ms=300_000.0,
    )
    _install(repo, monkeypatch, tight_budget)
    monkeypatch.setattr("gymrat.session.clock.now_ms", lambda: 0.0)


def mark_tool_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the command origin the tool host sets, so a live budget allows the command."""
    set_origin(monkeypatch, "tool")


def set_origin(monkeypatch: pytest.MonkeyPatch, origin: str | None) -> None:
    """Set ``GYMRAT_COMMAND_ORIGIN`` to ``origin``, or clear it when ``origin`` is None."""
    if origin is None:
        monkeypatch.delenv("GYMRAT_COMMAND_ORIGIN", raising=False)
    else:
        monkeypatch.setenv("GYMRAT_COMMAND_ORIGIN", origin)
