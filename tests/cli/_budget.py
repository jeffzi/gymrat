"""Shared budget fixtures for the CLI command test files.

Builders used by more than one ``test_*_cmd.py`` module to exercise the
budget time-left line, the JSON budget object, and duration warnings. This is
test-support code, not a test module: it carries no test functions or pytest
fixtures of its own.
"""

from pathlib import Path

import pytest

from gymrat.session.budget import Budget, write_budget

__all__ = [
    "install_budget",
    "install_tight_budget",
]

#: A 30-minute budget with a far-future deadline so the budget is always live.
BUDGET = Budget(started_at_ms=0.0, max_minutes=30, deadline_ms=9_999_999_999_999.0)


def _install(repo: str, monkeypatch: pytest.MonkeyPatch, budget: Budget) -> None:
    """Write a budget file and patch the supervise lock so read_budget succeeds."""
    Path(repo, ".gymrat").mkdir(exist_ok=True)
    write_budget(repo, budget)
    # Target is a dotted string, so pyrefly can't check the lambda against
    # is_held's real signature.
    monkeypatch.setattr("gymrat.session.budget.is_held", lambda _path: True)  # pyrefly: ignore


def install_budget(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a live budget file and patch the supervise lock so read_budget succeeds."""
    _install(repo, monkeypatch, BUDGET)


def install_tight_budget(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a budget with 5 minutes left and freeze the clock."""
    tight_budget = Budget(
        started_at_ms=0.0,
        max_minutes=30,
        deadline_ms=300_000.0,
    )
    _install(repo, monkeypatch, tight_budget)
    monkeypatch.setattr("gymrat.session.clock.now_ms", lambda: 0.0)
