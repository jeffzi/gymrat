"""Command-level tests for stop and for session commands writing to a closed stdout.

Each command is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.

The closed-stdout table drives ``measure --record`` and ``supervise`` with the
stdout reader gone.
"""

import errno
from collections.abc import Callable

import pytest

from gymrat.cli.app import app
from tests.cli._command_stubs import (
    force_render_mode,
    stub_measure,
)
from tests.cli._runner import (
    FailingStdoutRunner,
    runner,
)
from tests.cli._session import (
    open_session,
)
from tests.cli.commands.supervise._seams import install_seams

# ---------------------------------------------------------------------------
# the stop command
# ---------------------------------------------------------------------------


def test_stop_command_when_message_blank_does_exit_two(stop_repo: str):
    result = runner.invoke(app, ["stop", "-m", "   "])

    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# a closed stdout reader
# ---------------------------------------------------------------------------


def _open_and_stub_measure(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session and replace the measurement engine and config resolution."""
    open_session(repo)
    stub_measure(monkeypatch)


def _supervise_live(_repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every supervise seam and force the live dashboard."""
    install_seams(monkeypatch)
    force_render_mode(monkeypatch, "supervise", "live")


_MEASURE_MAIN = ["measure", "main", "--bench", "sh bench.sh"]


@pytest.mark.parametrize(
    ("argv", "arrange"),
    [
        pytest.param(
            [*_MEASURE_MAIN, "--record"], _open_and_stub_measure, id="measure-record-note"
        ),
        pytest.param(
            ["supervise", "optimize it", "--max-minutes", "10"], _supervise_live, id="supervise"
        ),
    ],
)
def test_command_when_stdout_reader_closed_does_exit_without_stderr(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    arrange: Callable[[str, pytest.MonkeyPatch], None],
):
    arrange(repo, monkeypatch)

    result = FailingStdoutRunner(BrokenPipeError(errno.EPIPE, "Broken pipe")).invoke(app, argv)

    assert (result.exit_code, result.stderr) == (0, "")
