"""Tests for the supervised-run liveness predicate and the supervised-origin guard.

A supervised run is live when the repository holds a budget file whose deadline
is still ahead and the supervise lock is held. While one is live, a command typed
at the shell (any origin other than ``tool``) is refused so the agent calls the
matching tool instead.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from gymrat.cli.supervised import guard_supervised_origin, is_supervised_run_live
from gymrat.errors import GymratError
from gymrat.session.budget import Budget, write_budget
from gymrat.session.paths import budget_path
from tests._lock import held_supervise_lock, hold_supervise_lock, remove_lock_files
from tests.cli._budget import (
    LIVE_BUDGET,
    SUPERVISED_HINT,
    set_origin,
)

#: A budget whose deadline is already in the past, so every liveness check against it answers false.
EXPIRED_BUDGET = Budget(max_minutes=30, deadline_ms=1.0)


@pytest.fixture
def state_dir(tmp_path: Path) -> Iterator[str]:
    """A repository root with an empty ``.gymrat`` state directory, its lock files removed after."""
    (tmp_path / ".gymrat").mkdir()
    yield str(tmp_path)
    remove_lock_files(str(tmp_path))


@pytest.fixture
def supervise_lock(state_dir: str) -> Iterator[None]:
    """Hold the real supervise lock for ``state_dir`` for the duration of the test."""
    with held_supervise_lock(state_dir):
        yield


def _write_budget_directory(state_dir: str) -> None:
    """Put a directory where the budget file belongs so reading it fails."""
    Path(budget_path(state_dir)).mkdir()


def _write_garbage_budget(state_dir: str) -> None:
    """Write a budget file that is not valid JSON."""
    Path(budget_path(state_dir)).write_text("banana", encoding="utf-8")


@pytest.fixture(
    params=[
        pytest.param("no-budget", id="no-budget-file"),
        pytest.param("expired", id="deadline-passed"),
        pytest.param("released", id="supervise-lock-not-held"),
        pytest.param("garbage", id="budget-not-json"),
        pytest.param("directory", id="budget-unreadable"),
    ]
)
def not_live_repo(request: pytest.FixtureRequest, state_dir: str) -> Iterator[str]:
    """A repository root in one of the states where no supervised run is live."""
    case: str = request.param
    lock = None if case == "released" else hold_supervise_lock(state_dir)
    if case == "expired":
        write_budget(state_dir, EXPIRED_BUDGET)
    elif case == "released":
        write_budget(state_dir, LIVE_BUDGET)
    elif case == "garbage":
        _write_garbage_budget(state_dir)
    elif case == "directory":
        _write_budget_directory(state_dir)
    yield state_dir
    if lock is not None:
        lock.release()


# ---------------------------------------------------------------------------
# is_supervised_run_live
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("supervise_lock")
def test_is_supervised_run_live_when_budget_live_and_lock_held_does_answer_true(state_dir: str):
    write_budget(state_dir, LIVE_BUDGET)

    live = is_supervised_run_live(state_dir)

    assert live is True


def test_is_supervised_run_live_when_not_live_does_answer_false(not_live_repo: str):
    live = is_supervised_run_live(not_live_repo)

    assert live is False


# ---------------------------------------------------------------------------
# guard_supervised_origin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "origin",
    [
        pytest.param(None, id="origin-unset"),
        pytest.param("cli", id="origin-cli"),
        pytest.param("", id="origin-empty"),
        pytest.param("TOOL", id="origin-uppercase-tool"),
    ],
)
@pytest.mark.usefixtures("supervise_lock")
def test_guard_supervised_origin_when_live_and_origin_not_tool_does_refuse(
    state_dir: str, monkeypatch: pytest.MonkeyPatch, origin: str | None
):
    write_budget(state_dir, LIVE_BUDGET)
    set_origin(monkeypatch, origin)

    with pytest.raises(GymratError) as exc:
        guard_supervised_origin(state_dir, "iterate")

    assert str(exc.value) == "a supervised run is live; use the iterate tool"
    assert exc.value.hint == SUPERVISED_HINT
    assert exc.value.reason == "supervised-use-tool"


@pytest.mark.usefixtures("supervise_lock")
def test_guard_supervised_origin_when_live_and_origin_tool_does_allow(
    state_dir: str, monkeypatch: pytest.MonkeyPatch
):
    write_budget(state_dir, LIVE_BUDGET)
    set_origin(monkeypatch, "tool")

    result = guard_supervised_origin(state_dir, "iterate")

    assert result is None


def test_guard_supervised_origin_when_not_live_does_allow(
    state_dir: str, monkeypatch: pytest.MonkeyPatch
):
    set_origin(monkeypatch, "cli")

    result = guard_supervised_origin(state_dir, "keep")

    assert result is None
