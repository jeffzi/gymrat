"""Command-level tests for keep, discard, status, and subdirectory resolution.

Each command is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
The seams mocked here mirror the engine suites' boundaries: for ``discard``'s
prompt the ``is_tty`` check as the ``commands.loop`` module imports it, with the
answer typed on the runner's stdin. The prompt's broken-stderr cases run the
CLI in a child process, since only a real file descriptor can be closed or
broken. Config resolution is exercised for real where a test lays down a
``gymrat.toml`` and stubbed at the ``commands.loop`` seam in the subdirectory
case, where the test observes the directory a command resolves its config from.

Each command's ``--format json`` document is pinned beside its text tests;
iterate's tests live in ``test_iterate``. The budget key of a JSON document is
pinned once, on ``status``: every other command writes its document through
``write_budget_report``, whose budget branches its own tests cover. The budget
time-left line, ``status``'s own trailer included, is pinned with the other
commands' in ``test_session_cmds``.
"""

import contextlib
import io
import json
import re
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
from gymrat.clock import now_ns
from gymrat.session.paths import experiment_worktree_dir
from gymrat.session.records import (
    CommandRecord,
    FinalizeRecord,
    KeepRecord,
    SessionLogRecord,
    StopRecord,
)
from gymrat.utils import SHORT_SHA_LENGTH, is_tty
from tests._ansi import SGR_RE, strip_ansi
from tests._cli import no_color_env
from tests._git import head_of, status_of
from tests._process_helpers import (
    run_with_closed_reader,
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
    close_session_with_one_keep,
    leave_as_is,
    make_discard_repo,
    open_session_with_one_keep,
    open_unedited_session,
    start_edited_session,
    write_bench_config,
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
    measured_rounds,
    settling_record_of,
    start_with,
    unimproved,
)
from tests.session._budget import install_budget_with_ten_minutes_left
from tests.session.records._fixtures import (
    AT,
    COMMIT,
    SESSION_ID,
    iteration_record,
    last_command_record,
    log_records,
    records_of_type,
)

#: The per-metric medians of ``KEPT_ROUNDS`` as ``status`` renders them, in the
#: order the rounds report the metrics.
KEPT_MEDIANS_LINE = "total_ms 14200 · alloc_bytes 2048"


def _always_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its interactive path."""
    return True


# ---------------------------------------------------------------------------
# the status command
# ---------------------------------------------------------------------------


def _write_open_session_with_one_keep(root: str) -> str:
    """Log a configured open session with one kept iteration, and return its id."""
    write_settled_session(root)
    return SESSION_ID


@pytest.mark.parametrize(
    ("arrange", "args", "traced"),
    [
        pytest.param(_write_open_session_with_one_keep, [], {}, id="open"),
        pytest.param(close_session_with_one_keep, [], {}, id="finalized"),
        pytest.param(
            _write_open_session_with_one_keep,
            ["--config", "gymrat.toml"],
            {"config": "gymrat.toml"},
            id="config-file-named",
        ),
    ],
)
def test_status_command_when_run_does_render_the_session_with_its_trace(
    repo: str, arrange: Callable[[str], str], args: list[str], traced: dict[str, object]
):
    session_id = arrange(repo)
    write_bench_config(repo)

    result = runner.invoke(app, ["status", *args])

    assert result.exit_code == 0
    text = strip_ansi(result.stdout)
    assert f"session {session_id}" in text
    assert "1 kept" in text
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.args, cmd.exit_code, cmd.reason) == ("status", traced, 0, None)


def test_status_command_when_run_inside_the_experiment_worktree_does_render_the_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    header = open_session_with_one_keep(repo)
    write_bench_config(repo)
    monkeypatch.chdir(experiment_worktree_dir(repo))

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    text = strip_ansi(result.stdout)
    assert f"session {header.session_id}" in text
    assert "1 kept" in text


def test_status_command_when_iteration_kept_does_close_its_history_with_the_kept_baseline(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_edited_session(repo, (measured_rounds(1),), checks=CHECKS)
    checks_pass(monkeypatch)
    runner.invoke(app, ["keep"])
    short_sha = head_of(experiment_worktree_dir(repo))[:SHORT_SHA_LENGTH]

    status = runner.invoke(app, ["status"])

    history = [line for line in strip_ansi(status.stdout).splitlines() if line.strip()]
    assert history[-2:] == [
        f"baseline {short_sha} · {KEPT_MEDIANS_LINE}",
        "1 iteration · 1 kept · 0 discarded",
    ]


# ---------------------------------------------------------------------------
# the discard command
# ---------------------------------------------------------------------------


_PROMPT_CHOICES = "[y/n] (n): "
"""How the discard question ends: the two answers and the default."""

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
def markup_repo(create_scratch_repo: Callable[..., str], monkeypatch: pytest.MonkeyPatch) -> str:
    """A discard-ready repository whose path carries rich markup and emoji codes, chdir'd into."""
    # Windows forbids ":" in a file name, so the emoji codes are POSIX-only.
    prefix = "bench [fast] " if sys.platform == "win32" else "bench [fast] :x: :ok: "
    root = create_scratch_repo(prefix)
    monkeypatch.chdir(root)
    start_edited_session(root)
    return root


@pytest.fixture
def narrow_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Narrow the console well below the length of any experiment worktree path."""
    monkeypatch.setenv("COLUMNS", "40")


@pytest.fixture
def tty_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the discard command believe stdin is a terminal."""
    monkeypatch.setattr(
        "gymrat.cli.commands.loop.is_tty", create_autospec(is_tty, side_effect=_always_tty)
    )


@pytest.mark.usefixtures("narrow_terminal", "tty_stdin")
def test_discard_command_when_tty_does_ask_on_stderr_naming_the_worktree_literally(
    markup_repo: str,
):
    worktree = experiment_worktree_dir(markup_repo)

    result = runner.invoke(app, ["discard"], input="y\n")

    assert re.search(
        re.escape(worktree) + ".*" + re.escape(_PROMPT_CHOICES),
        strip_ansi(result.stderr),
        flags=re.DOTALL,
    )
    assert _PROMPT_CHOICES not in result.stdout


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("y\n", id="lowercase-y"),
        pytest.param("Y\n", id="uppercase-y"),
        pytest.param("  y  \n", id="padded-y"),
    ],
)
@pytest.mark.usefixtures("tty_stdin")
def test_discard_command_when_tty_and_confirmed_does_discard(edited_repo: str, answer: str):
    result = runner.invoke(app, ["discard"], input=answer)

    assert result.exit_code == 0
    assert _discard_state(edited_repo) == (False, True)
    assert re.search(r"discard", result.stdout, re.IGNORECASE)


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("n\n", id="lowercase-n"),
        pytest.param("N\n", id="uppercase-n"),
        pytest.param("\n", id="empty-line-takes-the-default"),
        pytest.param("", id="end-of-input"),
    ],
)
@pytest.mark.usefixtures("tty_stdin")
def test_discard_command_when_tty_and_declined_does_cancel_with_exit_one(
    edited_repo: str, answer: str
):
    result = runner.invoke(app, ["discard"], input=answer)

    assert result.exit_code == 1
    assert "discard cancelled" in result.stderr
    assert _discard_state(edited_repo) == (True, False)


@pytest.mark.parametrize(
    ("answers", "outcome"),
    [
        pytest.param("yes\ny\n", (0, False, (False, True)), id="yes-then-y-discards"),
        pytest.param("maybe\nN\n", (1, True, (True, False)), id="maybe-then-n-declines"),
        pytest.param("yes\n", (1, True, (True, False)), id="yes-then-end-of-input-declines"),
    ],
)
@pytest.mark.usefixtures("tty_stdin")
def test_discard_command_when_tty_and_answer_invalid_does_ask_again(
    edited_repo: str,
    answers: str,
    outcome: tuple[int, bool, tuple[bool, bool]],
):
    result = runner.invoke(app, ["discard"], input=answers)

    assert "Please enter Y or N" in result.stderr
    assert result.stderr.count(_PROMPT_CHOICES) == 2
    assert (
        result.exit_code,
        "discard cancelled" in result.stderr,
        _discard_state(edited_repo),
    ) == outcome


@pytest.mark.parametrize(
    ("args", "is_tty_stub", "traced"),
    [
        pytest.param(["--force"], _always_tty, {"force": True}, id="force-long"),
        pytest.param(["-f"], _always_tty, {"force": True}, id="force-short"),
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
    assert _PROMPT_CHOICES not in result.stderr
    assert re.search(r"discard", result.stdout, re.IGNORECASE)
    assert _discard_state(edited_repo) == (False, True)
    cmd = last_command_record(edited_repo)
    assert (cmd.name, cmd.args, cmd.seq, cmd.exit_code, cmd.reason) == (
        "discard",
        traced,
        1,
        0,
        None,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell")
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


def test_discard_command_when_stderr_pipe_breaks_does_exit_one_without_output(edited_repo: str):
    result = run_with_closed_reader(
        [sys.executable, "-c", _TTY_DISCARD],
        stream="stderr",
        cwd=edited_repo,
        input=b"y\n",
        env=no_color_env(),
        timeout=60,
    )

    assert (result.returncode, result.stdout) == (1, b"")
    assert _discard_state(edited_repo) == (True, False)


# ---------------------------------------------------------------------------
# the loop commands, run from a subdirectory of the repository
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["start", "--baseline", "main"], id="start"),
        pytest.param(["iterate"], id="iterate"),
        pytest.param(["keep"], id="keep"),
        pytest.param(["status"], id="status"),
    ],
)
def test_loop_command_when_run_from_subdirectory_does_read_the_config_at_repo_root(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch, args: list[str]
):
    root = create_scratch_repo()
    start_with(root, (iteration_record(seq=1),))
    # Only the root holds a config, and its invalid value is what the command reports.
    write_bench_config(root, samples=0)
    nested = Path(root) / "packages" / "core"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    result = runner.invoke(app, args)

    assert (result.exit_code, strip_ansi(result.stderr)) == (
        2,
        "Error: Invalid config value for samples: expected a number at or above 1, got 0\n",
    )


# ---------------------------------------------------------------------------
# the keep command
# ---------------------------------------------------------------------------


def test_keep_command_when_checks_pass_does_commit_reporting_the_short_commit_with_its_trace(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_edited_session(repo, checks=CHECKS)
    checks_pass(monkeypatch)

    result = runner.invoke(app, ["keep", "-m", "cache the regex"])

    assert result.exit_code == 0
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "committed"
    assert head_of(experiment_worktree_dir(repo))[:7] in result.stdout
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.args, cmd.seq, cmd.exit_code, cmd.reason) == (
        "keep",
        {"message": "cache the regex"},
        1,
        0,
        None,
    )


def test_keep_command_when_nothing_to_commit_does_exit_one_tracing_the_reason(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    open_unedited_session(repo)
    checks_pass(monkeypatch)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.seq, cmd.exit_code, cmd.reason) == ("keep", 1, 1, "nothing-to-commit")


def test_keep_command_when_checks_fail_does_exit_one_recording_the_block(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_edited_session(repo, checks=CHECKS)
    checks_fail(monkeypatch)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert (record.status, record.reason) == ("blocked", "checks-failed")
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.exit_code, cmd.reason) == ("keep", 1, "checks-failed")


def test_keep_command_when_outcome_not_improved_does_exit_one_tracing_the_reason_without_the_flag(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_edited_session(repo, (unimproved(1, "no-signal"),), checks=CHECKS)
    checks_pass(monkeypatch)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    cmd = last_command_record(repo)
    assert cmd.exit_code == 1
    assert cmd.reason == "not-improved"
    assert "allow_unimproved" not in cmd.args


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
    assert cmd.reason is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell and signals")
def test_keep_command_when_signalled_mid_checks_does_exit_by_signal_code_leaving_no_checks_process_or_command_record(
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

    assert proc.returncode == 128 + signal.SIGTERM
    wait_until_dead_blocking(grandchild, timeout_s=SETTLE_TIMEOUT_S)
    # Polled rather than checked once: a killed leader stays visible as a zombie
    # until its parent reaps it.
    wait_until_dead_blocking(checks_pid, timeout_s=SETTLE_TIMEOUT_S)
    assert records_of_type(repo, CommandRecord) == []


# ---------------------------------------------------------------------------
# --color / --no-color on keep
# ---------------------------------------------------------------------------


#: Environment that forces color on, which a --no-color flag must outrank.
_FORCE_COLOR = {"FORCE_COLOR": "1"}


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


@pytest.mark.parametrize(
    ("args", "env", "cli_runner", "expect_ansi"),
    [
        pytest.param(["keep", "--no-color"], _FORCE_COLOR, runner, False, id="no-color-flag"),
        pytest.param(["keep"], {}, _TerminalStderrRunner(), True, id="stderr-tty-stdout-piped"),
    ],
)
def test_keep_command_when_no_checks_does_color_the_warning_hint_per_flags_and_stderr(
    repo: str,
    args: list[str],
    env: dict[str, str],
    cli_runner: CliRunner,
    expect_ansi: bool,
):
    start_edited_session(repo)

    result = cli_runner.invoke(app, args, env=env)

    assert result.exit_code == 0
    assert "no checks command is configured" in strip_ansi(result.stderr)
    hint = result.stderr.splitlines()[1]
    assert bool(SGR_RE.search(hint)) is expect_ansi


# ---------------------------------------------------------------------------
# keep --format json
# ---------------------------------------------------------------------------


def test_keep_command_when_format_json_and_committed_does_emit_structured_json(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_edited_session(repo, checks=CHECKS)
    checks_pass(monkeypatch)

    result = runner.invoke(app, ["keep", "-m", "cache the regex", "--format", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "status": "committed",
        "reason": None,
        "checks": {
            "configured": True,
            "passed": True,
            "stdout_bytes": None,
            "stderr_bytes": None,
        },
        "commit": head_of(experiment_worktree_dir(repo)),
        "message": "cache the regex",
    }


def test_keep_command_when_format_json_and_iteration_unimproved_does_emit_the_blocked_document(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_edited_session(repo, (unimproved(1, "no-signal"),), checks=CHECKS)
    checks_pass(monkeypatch)

    result = runner.invoke(app, ["keep", "--format", "json"])

    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc.keys() == {"status", "reason", "checks", "commit", "message"}
    assert {key: doc[key] for key in ("status", "reason", "commit", "message")} == {
        "status": "blocked",
        "reason": "not-improved",
        "commit": None,
        "message": None,
    }


# ---------------------------------------------------------------------------
# discard --format json
# ---------------------------------------------------------------------------


def _unmeasured_edit(repo: str) -> None:
    """Open a session with an edit in the experiment worktree and nothing measured."""
    start_edited_session(repo, ())


@pytest.mark.parametrize(
    ("arrange", "expected_seq", "expected_measured"),
    [
        pytest.param(make_discard_repo, 1, True, id="measured"),
        pytest.param(_unmeasured_edit, None, False, id="unmeasured"),
    ],
)
def test_discard_command_when_format_json_does_emit_structured_json(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    arrange: Callable[[str], object],
    expected_seq: int | None,
    expected_measured: bool,
):
    arrange(repo)
    monkeypatch.setattr("gymrat.loop.discard.now_ns", create_autospec(now_ns, return_value=AT))

    result = runner.invoke(app, ["discard", "--force", "--format", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "seq": expected_seq,
        "at": AT,
        "measured": expected_measured,
    }


# ---------------------------------------------------------------------------
# status --format json
# ---------------------------------------------------------------------------


_FINALIZE = FinalizeRecord(
    type="finalize",
    at=AT,
    branch=f"gymrat/{SESSION_ID}-final",
    commit=COMMIT,
    message="squash 1 kept iteration",
)


#: The status document of a settled session with one kept iteration, minus its end-state flags.
_SETTLED_STATUS = {
    "session_id": SESSION_ID,
    "branch": f"gymrat/{SESSION_ID}",
    "baseline": {"ref": "main", "sha": "a" * 40},
    "iteration_count": 1,
    "keep_count": 1,
    "discard_count": 0,
    "unsettled": False,
}


@pytest.mark.parametrize(
    ("trailing", "arrange", "expected"),
    [
        pytest.param((), leave_as_is, {"finalized": False, "stopped": False}, id="open"),
        pytest.param(
            (_FINALIZE,), leave_as_is, {"finalized": True, "stopped": False}, id="finalized"
        ),
        pytest.param(
            (StopRecord(type="stop", at=AT, message="user requested stop"),),
            leave_as_is,
            {"finalized": False, "stopped": True},
            id="stopped",
        ),
        pytest.param(
            (),
            install_budget_with_ten_minutes_left,
            {
                "finalized": False,
                "stopped": False,
                "budget": {"cap_minutes": 30, "remaining_seconds": 600},
            },
            id="open-under-budget",
        ),
    ],
)
def test_status_command_when_format_json_does_emit_structured_json_on_stdout(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    trailing: tuple[SessionLogRecord, ...],
    arrange: Callable[[str, pytest.MonkeyPatch], None],
    expected: dict[str, object],
):
    write_settled_session(repo, *trailing)
    arrange(repo, monkeypatch)

    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {**_SETTLED_STATUS, **expected}
