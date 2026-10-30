"""Tests for the ``gymrat probe`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` against a real
scratch repository, a real session log, and the real ``probe_session`` engine.
The one seam replaced is the measurement engine — it shells out to the
consumer's bench script, which no test here can run — so the command trace is
exercised end to end. The engine's own refusals and scoping are pinned in
``tests/loop/test_probe.py``; the budget trailer and tight-budget warning in
``tests/cli/test_budget_report.py``; the shared session and supervised-run
guards in ``tests/session/test_store.py`` and ``tests/cli/test_supervised.py``.
"""

import pytest

from gymrat.cli.app import app
from tests.cli._command_stubs import (
    stub_probe_measure,
)
from tests.cli._runner import (
    runner,
)
from tests.cli._session import (
    open_probe_session,
)
from tests.loop._probe import (
    MeasureRecorder,
)
from tests.session.records._fixtures import (
    last_command_record,
)


@pytest.fixture
def measure(monkeypatch: pytest.MonkeyPatch) -> MeasureRecorder:
    """The stubbed measurement engine, answering a canned result."""
    return stub_probe_measure(monkeypatch)


@pytest.fixture
def probe_repo(repo: str) -> str:
    """A scratch repo with an open session and a recorded baseline."""
    open_probe_session(repo)
    return repo


# ---------------------------------------------------------------------------
# command trace
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("measure")
def test_probe_command_when_run_bare_does_trace_no_names(probe_repo: str):
    result = runner.invoke(app, ["probe"])

    assert result.exit_code == 0
    assert last_command_record(probe_repo).args == {"names": []}
