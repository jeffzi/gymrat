"""Tests for the supervised-origin guard.

A supervised run is live when the repository holds a budget file whose deadline
is still ahead and the supervise lock is held. While one is live, a command typed
at the shell (any origin other than ``tool``) is refused so the agent calls the
matching tool instead.
"""

import pytest

from gymrat.cli.supervised import guard_supervised_origin
from gymrat.errors import GymratError
from gymrat.session.budget import write_budget
from tests.cli._origin import set_origin
from tests.session._budget import LIVE_BUDGET

# ---------------------------------------------------------------------------
# guard_supervised_origin
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("supervise_lock")
def test_guard_supervised_origin_when_live_and_origin_not_tool_does_refuse(
    root: str, monkeypatch: pytest.MonkeyPatch
):
    write_budget(root, LIVE_BUDGET)
    set_origin(monkeypatch, "cli")

    with pytest.raises(GymratError) as exc:
        guard_supervised_origin(root, "iterate")

    assert "iterate" in str(exc.value)


@pytest.mark.usefixtures("supervise_lock")
def test_guard_supervised_origin_when_live_and_origin_tool_does_allow(
    root: str, monkeypatch: pytest.MonkeyPatch
):
    write_budget(root, LIVE_BUDGET)
    set_origin(monkeypatch, "tool")

    guard_supervised_origin(root, "iterate")


def test_guard_supervised_origin_when_not_live_does_allow(
    root: str, monkeypatch: pytest.MonkeyPatch
):
    set_origin(monkeypatch, "cli")

    guard_supervised_origin(root, "keep")
