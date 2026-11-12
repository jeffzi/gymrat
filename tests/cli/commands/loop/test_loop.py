"""Command-level tests for keep, discard and status.

Each command is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
The seams mocked here mirror the engine suites' boundaries: for ``discard``'s
prompt the ``is_tty`` check as the ``commands.loop`` module imports it, with the
answer typed on the runner's stdin. The prompt's closed-stderr case runs the
CLI in a child process, since only a real file descriptor can be closed. Config
resolution runs for real.

The documents' shapes are pinned in ``tests/report/test_json_doc.py``;
iterate's tests live in ``test_iterate``. The budget key of a JSON document is
pinned on ``status`` and on iterate's stop document: every other command writes
its document through ``write_budget_report``, whose budget branches its own
tests cover. The budget
time-left line is pinned in ``tests/cli/test_budget_report.py``.
"""

import contextlib
import io
import json
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, override
from unittest.mock import create_autospec

import pytest
from typer.testing import CliRunner

from gymrat.cli.app import app
from gymrat.session.paths import experiment_worktree_dir
from gymrat.session.records import (
    KeepRecord,
)
from gymrat.utils import is_tty
from tests._ansi import SGR_RE, strip_ansi
from tests._cli import no_color_env
from tests._git import status_of
from tests._platform import needs_posix_shell, needs_posix_signals
from tests._process_helpers import (
    wait_for_pid_file_blocking,
    wait_until_dead_blocking,
)
from tests.cli._command_stubs import (
    never_tty,
)
from tests.cli._runner import (
    runner,
)
from tests.cli._session import (
    start_edited_session,
    write_settled_session,
)
from tests.cli._signalled_cli import (
    SETTLE_TIMEOUT_S,
    pid_recording_script,
    spawned_gymrat,
    stop_by_signal,
)
from tests.loop._settle import (
    CHECKS,
    checks_fail,
    checks_pass,
    settling_record_of,
    unimproved,
)
from tests.session._budget import install_budget_with_ten_minutes_left
from tests.session.records._fixtures import (
    last_command_record,
    log_records,
)


def _always_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its interactive path."""
    return True


# ---------------------------------------------------------------------------
# the discard command
# ---------------------------------------------------------------------------


_TTY_DISCARD = (
    "import runpy, sys\n"
    "from unittest.mock import create_autospec\n"
    "from gymrat.cli.commands import loop\n"
    "loop.is_tty = create_autospec(loop.is_tty, return_value=True)\n"
    "sys.argv = ['gymrat', 'discard']\n"
    "runpy.run_module('gymrat.cli.app', run_name='__main__')\n"
)
"""A child-process driver running ``gymrat discard`` as if stdin were a terminal."""


def _discard_state(repo: str) -> tuple[bool, bool]:
    """Whether the experiment edit is still present, and whether a discard was logged."""
    edit_present = status_of(experiment_worktree_dir(repo)) != ""
    discard_logged = any(record.type == "discard" for record in log_records(repo))
    return edit_present, discard_logged


@pytest.fixture
def edited_repo(repo: str) -> str:
    """A configured repository with an open session and an experiment edit to discard."""
    start_edited_session(repo)
    return repo


@pytest.fixture
def tty_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the discard command believe stdin is a terminal."""
    monkeypatch.setattr(
        "gymrat.cli.commands.loop.is_tty", create_autospec(is_tty, side_effect=_always_tty)
    )


@pytest.mark.usefixtures("tty_stdin")
def test_discard_command_when_tty_does_ask_on_stderr_naming_the_worktree(edited_repo: str):
    result = runner.invoke(app, ["discard"], input="y\n")

    assert experiment_worktree_dir(edited_repo) in strip_ansi(result.stderr)


@pytest.mark.usefixtures("tty_stdin")
def test_discard_command_when_tty_and_confirmed_does_discard(edited_repo: str):
    result = runner.invoke(app, ["discard"], input="y\n")

    assert result.exit_code == 0
    assert _discard_state(edited_repo) == (False, True)


@pytest.mark.usefixtures("tty_stdin")
def test_discard_command_when_tty_and_input_ends_does_cancel_with_exit_one(edited_repo: str):
    result = runner.invoke(app, ["discard"], input="")

    assert result.exit_code == 1
    assert _discard_state(edited_repo) == (True, False)


@pytest.mark.parametrize(
    ("args", "is_tty_stub", "traced"),
    [
        pytest.param(["--force"], _always_tty, {"force": True}, id="force-long"),
        pytest.param([], never_tty, {}, id="stdin-not-tty"),
    ],
)
def test_discard_command_when_force_or_stdin_not_tty_does_discard_without_prompting(
    *,
    edited_repo: str,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    is_tty_stub: Callable[[object], bool],
    traced: dict[str, object],
):
    monkeypatch.setattr(
        "gymrat.cli.commands.loop.is_tty", create_autospec(is_tty, side_effect=is_tty_stub)
    )

    result = runner.invoke(app, ["discard", *args], input="n\n")

    assert result.exit_code == 0
    assert _discard_state(edited_repo) == (False, True)
    cmd = last_command_record(edited_repo)
    assert (cmd.args, cmd.seq) == (traced, 1)


@needs_posix_shell
def test_discard_command_when_stderr_closed_does_decline_silently(edited_repo: str):
    result = subprocess.run(  # noqa: S603 -- fixed /bin/sh script relaying sys.executable and a fixed program
        ["/bin/sh", "-c", 'exec "$0" -c "$1" 2>&-', sys.executable, _TTY_DISCARD],
        cwd=edited_repo,
        input=b"y\n",
        stdout=subprocess.PIPE,
        env=no_color_env(),
        check=False,
        timeout=60,
    )

    assert (result.returncode, result.stdout) == (1, b"")
    assert _discard_state(edited_repo) == (True, False)


# ---------------------------------------------------------------------------
# the keep command
# ---------------------------------------------------------------------------


def test_keep_command_when_checks_pass_does_trace_its_message_and_seq(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_edited_session(repo, checks=CHECKS)
    checks_pass(monkeypatch)

    result = runner.invoke(app, ["keep", "-m", "cache the regex"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert (cmd.args, cmd.seq) == ({"message": "cache the regex"}, 1)


def test_keep_command_when_keep_blocked_does_exit_one_tracing_the_reason(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_edited_session(repo, checks=CHECKS)
    checks_fail(monkeypatch)

    result = runner.invoke(app, ["keep"])

    cmd = last_command_record(repo)
    assert (result.exit_code, cmd.seq, cmd.exit_code, cmd.reason) == (1, 1, 1, "checks-failed")


def test_keep_command_when_allow_unimproved_does_commit_with_the_flag_traced(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_edited_session(repo, (unimproved(1, "no-signal"),), checks=CHECKS)
    checks_pass(monkeypatch)

    result = runner.invoke(app, ["keep", "--allow-unimproved"])

    assert result.exit_code == 0
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "committed"
    cmd = last_command_record(repo)
    assert cmd.args["allow_unimproved"] is True


@needs_posix_signals
def test_keep_command_when_signalled_mid_checks_does_leave_no_checks_process(
    repo: str,
    tmp_path: Path,
    reap_groups: list[int],
):
    # The checks record their own pid and a background grandchild's, then block,
    # so ``keep`` is always mid-checks when signalled.
    script = tmp_path / "checks.sh"
    body = f"sleep 120 &\necho $! > {shlex.quote(str(tmp_path / 'grandchild.pid'))}\nwait\n"
    script.write_text(pid_recording_script(tmp_path / "checks.pid", body), encoding="utf-8")
    start_edited_session(repo, checks=f"sh {shlex.quote(str(script))}")

    with spawned_gymrat(["keep"], repo) as proc:
        checks_pid = wait_for_pid_file_blocking(tmp_path / "checks.pid", SETTLE_TIMEOUT_S)
        reap_groups.append(checks_pid)
        grandchild = wait_for_pid_file_blocking(tmp_path / "grandchild.pid", SETTLE_TIMEOUT_S)
        reap_groups.append(grandchild)

        stop_by_signal(proc, signal.SIGTERM)

    wait_until_dead_blocking(grandchild, timeout_s=SETTLE_TIMEOUT_S)
    # Polled rather than checked once: a killed leader stays visible as a zombie
    # until its parent reaps it.
    wait_until_dead_blocking(checks_pid, timeout_s=SETTLE_TIMEOUT_S)


# ---------------------------------------------------------------------------
# warning color when stderr is a terminal
# ---------------------------------------------------------------------------


class _TerminalTextIO(io.TextIOWrapper):
    """A text stream over a captured buffer that reports itself as a terminal."""

    @override
    def isatty(self) -> bool:
        return True


class _TerminalStderrRunner(CliRunner):
    """A ``CliRunner`` whose isolated ``sys.stderr`` is a terminal while stdout stays piped."""

    @override
    @contextlib.contextmanager
    def isolation(self, *args: Any, **kwargs: Any) -> Generator[Any]:
        with super().isolation(*args, **kwargs) as streams:
            captured_stderr = sys.stderr
            terminal = _TerminalTextIO(captured_stderr.buffer, encoding="utf-8", write_through=True)
            sys.stderr = terminal
            try:
                yield streams
            finally:
                terminal.flush()
                # Detach so collecting the wrapper never closes the runner's capture buffer.
                terminal.detach()
                sys.stderr = captured_stderr


def test_keep_command_when_no_checks_and_stderr_a_terminal_does_color_the_warning_hint(
    repo: str,
):
    start_edited_session(repo)

    result = _TerminalStderrRunner().invoke(app, ["keep"])

    assert result.exit_code == 0
    assert "no checks command is configured" in strip_ansi(result.stderr)
    hint = result.stderr.splitlines()[1]
    assert SGR_RE.search(hint)


# ---------------------------------------------------------------------------
# status --format json
# ---------------------------------------------------------------------------


def test_status_command_when_format_json_under_budget_does_include_the_budget_key(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_settled_session(repo)
    install_budget_with_ten_minutes_left(repo, monkeypatch)

    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["budget"] == {"cap_minutes": 30, "remaining_seconds": 600}
