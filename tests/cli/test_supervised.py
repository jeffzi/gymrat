"""Tests for the supervised-origin guard.

A supervised run is live while the repository holds a parseable budget file and
the supervise lock is held, whether or not the budget's deadline has passed.
While one is live, a command typed at the shell (any origin other than ``tool``)
is refused so the agent calls the matching tool instead.
"""

import pytest

from gymrat.cli.supervised import guard_supervised_origin
from gymrat.errors import GymratError
from gymrat.session.budget import Budget, write_budget
from gymrat.session.paths import supervise_lockfile_path
from tests._lock import lock_probe_error
from tests.cli._origin import set_origin
from tests.session._budget import LIVE_BUDGET, block_supervise_lock

#: A budget whose deadline passed long ago, as one left running past its time.
_EXPIRED_BUDGET = Budget(max_minutes=30, deadline_ms=1_000.0)

_WHATEVER_THE_DEADLINE = pytest.mark.parametrize(
    "budget",
    [
        pytest.param(LIVE_BUDGET, id="deadline-ahead"),
        pytest.param(_EXPIRED_BUDGET, id="deadline-passed"),
    ],
)

# ---------------------------------------------------------------------------
# guard_supervised_origin
# ---------------------------------------------------------------------------


@_WHATEVER_THE_DEADLINE
@pytest.mark.usefixtures("supervise_lock")
def test_guard_supervised_origin_when_lock_held_and_origin_not_tool_does_refuse(
    root: str, monkeypatch: pytest.MonkeyPatch, budget: Budget
):
    write_budget(root, budget)
    set_origin(monkeypatch, "cli")

    with pytest.raises(GymratError) as exc:
        guard_supervised_origin(root, "iterate")

    assert "iterate" in str(exc.value)


@_WHATEVER_THE_DEADLINE
@pytest.mark.usefixtures("supervise_lock")
def test_guard_supervised_origin_when_lock_held_and_origin_tool_does_allow(
    root: str, monkeypatch: pytest.MonkeyPatch, budget: Budget
):
    write_budget(root, budget)
    set_origin(monkeypatch, "tool")

    guard_supervised_origin(root, "iterate")


def test_guard_supervised_origin_when_budget_left_behind_and_lock_free_does_allow(
    root: str, monkeypatch: pytest.MonkeyPatch
):
    write_budget(root, LIVE_BUDGET)
    set_origin(monkeypatch, "cli")

    guard_supervised_origin(root, "iterate")


def test_guard_supervised_origin_when_no_budget_file_does_allow_without_probing_the_lock(
    root: str, monkeypatch: pytest.MonkeyPatch
):
    set_origin(monkeypatch, "cli")

    with block_supervise_lock(root):
        guard_supervised_origin(root, "keep")


def test_guard_supervised_origin_when_budget_written_and_lock_cannot_be_probed_does_raise_the_lock_error(
    root: str, monkeypatch: pytest.MonkeyPatch
):
    write_budget(root, _EXPIRED_BUDGET)
    set_origin(monkeypatch, "cli")

    with block_supervise_lock(root):
        expected = lock_probe_error(supervise_lockfile_path(root))

        with pytest.raises(GymratError) as exc:
            guard_supervised_origin(root, "iterate")

    assert str(exc.value) == expected
