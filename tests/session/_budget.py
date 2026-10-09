"""Shared budget-file helpers for the tests that run a command under a live budget.

Builders used by the CLI command tests and the iterate engine tests to lay a
real budget file down and pin the clock it is read against. This is
test-support code, not a test module: it carries no test functions or pytest
fixtures of its own.
"""

from pathlib import Path
from unittest.mock import create_autospec

import pytest

from gymrat.clock import now_ms
from gymrat.session.budget import Budget, write_budget
from gymrat.session.lock import is_held

#: A 30-minute budget whose deadline sits far in the future, so it never expires mid-test.
LIVE_BUDGET = Budget(max_minutes=30, deadline_ms=9_999_999_999_999.0)


def write_budget_file(repo: str, budget: Budget = LIVE_BUDGET) -> None:
    """Write ``budget`` into ``repo``'s state directory, creating the directory if needed.

    Args:
        repo: The repository whose budget file is written.
        budget: The budget to write; a live one by default.
    """
    Path(repo, ".gymrat").mkdir(exist_ok=True)
    write_budget(repo, budget)


def install_budget(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    deadline_ms: float = LIVE_BUDGET.deadline_ms,
    frozen_now_ms: int | None = None,
) -> None:
    """Write a 30-minute budget file and patch the supervise lock so ``read_budget`` succeeds.

    Args:
        repo: The repository whose budget file is written.
        monkeypatch: The fixture that patches the supervise-lock probe and the clock.
        deadline_ms: The budget's deadline; far in the future by default.
        frozen_now_ms: The wall-clock reading to freeze ``gymrat.clock.now_ms``
            at; None leaves the clock real.
    """
    write_budget_file(repo, Budget(max_minutes=LIVE_BUDGET.max_minutes, deadline_ms=deadline_ms))
    monkeypatch.setattr(
        "gymrat.session.budget.is_held", create_autospec(is_held, return_value=True)
    )
    if frozen_now_ms is not None:
        monkeypatch.setattr(
            "gymrat.clock.now_ms", create_autospec(now_ms, return_value=frozen_now_ms)
        )


def install_tight_budget(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a budget with 5 minutes left and freeze the clock.

    Args:
        repo: The repository whose budget file is written.
        monkeypatch: The fixture that patches the supervise-lock probe and the clock.
    """
    install_budget(repo, monkeypatch, deadline_ms=300_000.0, frozen_now_ms=0)


def install_budget_with_ten_minutes_left(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a live budget and freeze the clock ten minutes before its deadline.

    Args:
        repo: The repository whose budget file is written.
        monkeypatch: The fixture that patches the supervise-lock probe and the clock.
    """
    install_budget(repo, monkeypatch, frozen_now_ms=int(LIVE_BUDGET.deadline_ms) - 600_000)
