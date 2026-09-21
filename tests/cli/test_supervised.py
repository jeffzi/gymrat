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
from gymrat.session.paths import budget_path, supervise_lockfile_path
from tests.cli._budget import (
    LIVE_BUDGET,
    SUPERVISED_HINT,
    mark_tool_origin,
    set_origin,
)
from tests.conftest import hold_lock

#: A budget whose deadline is already in the past, so every liveness check against it answers false.
EXPIRED_BUDGET = Budget(started_at_ms=0.0, max_minutes=30, deadline_ms=1.0)


@pytest.fixture
def repo(tmp_path: Path) -> str:
    """A repository root with an empty ``.gymrat`` state directory."""
    (tmp_path / ".gymrat").mkdir()
    return str(tmp_path)


def _write_budget_directory(repo: str) -> None:
    """Put a directory where the budget file belongs so reading it fails."""
    Path(budget_path(repo)).mkdir()


def _write_garbage_budget(repo: str) -> None:
    """Write a budget file that is not valid JSON."""
    Path(budget_path(repo)).write_text("banana", encoding="utf-8")


@pytest.fixture(
    params=[
        pytest.param("no-budget", id="no-budget-file"),
        pytest.param("expired", id="deadline-passed"),
        pytest.param("released", id="supervise-lock-not-held"),
        pytest.param("garbage", id="budget-not-json"),
        pytest.param("directory", id="budget-unreadable"),
    ]
)
def not_live_repo(request: pytest.FixtureRequest, repo: str) -> Iterator[str]:
    """A repository root in one of the states where no supervised run is live."""
    case: str = request.param
    lock = None if case == "released" else hold_lock(supervise_lockfile_path(repo), "supervise")
    if case == "expired":
        write_budget(repo, EXPIRED_BUDGET)
    elif case == "released":
        write_budget(repo, LIVE_BUDGET)
    elif case == "garbage":
        _write_garbage_budget(repo)
    elif case == "directory":
        _write_budget_directory(repo)
    yield repo
    if lock is not None:
        lock.release()


# ---------------------------------------------------------------------------
# is_supervised_run_live
# ---------------------------------------------------------------------------

#: Every command origin the guard treats differently: unset, the tool host, and the shell.
ORIGINS = [
    pytest.param(None, id="origin-unset"),
    pytest.param("tool", id="origin-tool"),
    pytest.param("cli", id="origin-cli"),
]
#: Two more non-tool spellings, added where a test needs every origin the guard refuses.
NON_TOOL_EXTRAS = [
    pytest.param("", id="origin-empty"),
    pytest.param("TOOL", id="origin-uppercase-tool"),
]


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.usefixtures("supervise_lock")
def test_is_supervised_run_live_when_budget_live_and_lock_held_does_answer_true(
    repo: str, monkeypatch: pytest.MonkeyPatch, origin: str | None
):
    write_budget(repo, LIVE_BUDGET)
    set_origin(monkeypatch, origin)

    live = is_supervised_run_live(repo)

    assert live is True


@pytest.mark.parametrize("origin", ORIGINS)
def test_is_supervised_run_live_when_not_live_does_answer_false(
    not_live_repo: str, monkeypatch: pytest.MonkeyPatch, origin: str | None
):
    set_origin(monkeypatch, origin)

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
        *NON_TOOL_EXTRAS,
    ],
)
@pytest.mark.usefixtures("supervise_lock")
def test_guard_supervised_origin_when_live_and_origin_not_tool_does_refuse(
    repo: str, monkeypatch: pytest.MonkeyPatch, origin: str | None
):
    write_budget(repo, LIVE_BUDGET)
    set_origin(monkeypatch, origin)

    with pytest.raises(GymratError) as exc:
        guard_supervised_origin(repo, "iterate")

    assert str(exc.value) == "a supervised run is live; use the iterate tool"
    assert exc.value.hint == SUPERVISED_HINT
    assert exc.value.reason == "supervised-use-tool"


@pytest.mark.usefixtures("supervise_lock")
def test_guard_supervised_origin_when_live_and_origin_tool_does_allow(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_budget(repo, LIVE_BUDGET)
    mark_tool_origin(monkeypatch)

    result = guard_supervised_origin(repo, "iterate")

    assert result is None


@pytest.mark.parametrize("origin", [*ORIGINS, *NON_TOOL_EXTRAS])
def test_guard_supervised_origin_when_not_live_does_allow(
    repo: str, monkeypatch: pytest.MonkeyPatch, origin: str | None
):
    set_origin(monkeypatch, origin)

    result = guard_supervised_origin(repo, "keep")

    assert result is None
