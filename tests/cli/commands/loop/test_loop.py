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

Iterate and JSON-contract tests live in ``test_iterate`` and ``test_json``.

Budget-line tests verify that loop commands append a time-left line to their
text output when a live budget is present; ``status`` builds its own trailer, so
it also pins the line's absence without one.
"""

import contextlib
import io
import re
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, override

import pytest
from typer.testing import CliRunner

from gymrat.cli.app import app
from gymrat.git import SHORT_SHA_LENGTH
from gymrat.session.paths import experiment_worktree_dir
from gymrat.session.records import (
    CommandRecord,
    KeepRecord,
    SessionLogRecord,
)
from tests._ansi import SGR_RE, strip_ansi
from tests._cli import ENTRY, no_color_env
from tests._git import head_of, status_of
from tests._process_helpers import (
    run_with_closed_reader,
    wait_for_pid_file_blocking,
    wait_until_dead_blocking,
)
from tests.cli._budget import install_budget, set_origin
from tests.cli._session import (
    FailingStdoutRunner,
    close_session_with_one_keep,
    closed_stdout_error,
    last_command_record,
    never_tty,
    open_session_with_one_keep,
    runner,
    write_bench_config,
)
from tests.loop._settle import (
    CHECKS,
    checks_fail,
    checks_pass,
    edit_experiment,
    measured_rounds,
    settling_record_of,
    start_with,
    unimproved,
)
from tests.loop.iterate._fixtures import (
    baseline_rounds,
    improved_rounds,
    install_collect_samples,
    iterate_session_header,
    stub_samples,
)
from tests.session.records._fixtures import (
    SESSION_ID,
    committed_keep,
    iteration_record,
    log_records,
    records_of_type,
    session_record,
    write_session_log,
)

#: The per-metric medians of ``KEPT_ROUNDS`` as ``status`` renders them, in the
#: order the rounds report the metrics.
KEPT_MEDIANS_LINE = "total_ms 14200 · alloc_bytes 2048"


def _always_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its interactive path."""
    return True


def _start_edited_session(
    root: str,
    history: tuple[SessionLogRecord, ...] = (iteration_record(seq=1),),
    **config: object,
) -> None:
    """Open a session on ``history``, edit the experiment, and write the config.

    Args:
        root: The repository root.
        history: The records logged after the session header; one unsettled
            iteration by default.
        **config: Extra ``gymrat.toml`` entries beside the bench.
    """
    start_with(root, history)
    edit_experiment(root)
    write_bench_config(root, **config)


def _start_unedited_session(root: str) -> None:
    """Open a session with one unsettled iteration and configured checks, and edit nothing."""
    start_with(root, (iteration_record(seq=1),))
    write_bench_config(root, checks=CHECKS)


# ---------------------------------------------------------------------------
# the status command
# ---------------------------------------------------------------------------


def test_status_command_when_run_does_render_the_session_on_stdout_and_record_the_trace(
    repo: str,
):
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_bench_config(repo)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    text = strip_ansi(result.stdout)
    assert f"session {SESSION_ID}" in text
    assert "1 kept" in text
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.args, cmd.exit_code, cmd.reason) == ("status", {}, 0, None)


def test_status_command_when_finalized_does_record_command_trace_with_exit_zero(repo: str):
    close_session_with_one_keep(repo)
    write_bench_config(repo)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.name == "status"
    assert cmd.exit_code == 0


@pytest.fixture
def keep_ready_repo(repo: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """A repository with an experiment edit whose configured checks pass, ready for any loop command."""
    _start_edited_session(repo, checks=CHECKS)
    checks_pass(monkeypatch)
    return repo


@pytest.mark.usefixtures("keep_ready_repo")
@pytest.mark.parametrize(
    "command",
    [
        pytest.param(["status", "--format", "text"], id="status-text"),
        pytest.param(["status", "--format", "json"], id="status-json"),
        pytest.param(["keep"], id="keep"),
        pytest.param(["discard"], id="discard"),
    ],
)
def test_loop_command_when_stdout_reader_closed_does_exit_zero_without_stderr(command: list[str]):
    result = FailingStdoutRunner(closed_stdout_error()).invoke(app, command)

    assert (result.exit_code, result.stderr) == (0, "")


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


# ---------------------------------------------------------------------------
# the discard command
# ---------------------------------------------------------------------------


_PROMPT_CHOICES = "[y/n] (n): "
"""How the discard question ends: the two answers and the default."""

_TTY_DISCARD = (
    "import runpy, sys\n"
    "from gymrat.cli.commands import loop\n"
    "loop.is_tty = lambda _stream: True\n"
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
    _start_edited_session(repo)
    return repo


@pytest.fixture
def markup_repo(create_scratch_repo: Callable[..., str], monkeypatch: pytest.MonkeyPatch) -> str:
    """A discard-ready repository whose path carries rich markup and emoji codes, chdir'd into."""
    # Windows forbids ":" in a file name, so the emoji codes are POSIX-only.
    prefix = "bench [fast] " if sys.platform == "win32" else "bench [fast] :x: :ok: "
    root = create_scratch_repo(prefix)
    monkeypatch.chdir(root)
    _start_edited_session(root)
    return root


@pytest.fixture
def narrow_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Narrow the console well below the length of any experiment worktree path."""
    monkeypatch.setenv("COLUMNS", "40")


@pytest.fixture
def tty_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the discard command believe stdin is a terminal."""
    monkeypatch.setattr("gymrat.cli.commands.loop.is_tty", _always_tty)


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
    ("args", "is_tty_stub"),
    [
        pytest.param(["--force"], _always_tty, id="force-long"),
        pytest.param(["-f"], _always_tty, id="force-short"),
        pytest.param([], never_tty, id="stdin-not-tty"),
    ],
)
def test_discard_command_when_force_or_stdin_not_tty_does_skip_the_prompt(
    edited_repo: str,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    is_tty_stub: Callable[[object], bool],
):
    monkeypatch.setattr("gymrat.cli.commands.loop.is_tty", is_tty_stub)

    result = runner.invoke(app, ["discard", *args], input="n\n")

    assert result.exit_code == 0
    assert _PROMPT_CHOICES not in result.stderr
    assert _discard_state(edited_repo) == (False, True)


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


def test_keep_command_when_checks_pass_does_commit_print_the_short_commit_and_record_the_trace(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _start_edited_session(repo, checks=CHECKS)
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


def test_keep_command_when_committed_does_add_the_kept_baseline_to_the_status_history(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _start_edited_session(repo, (measured_rounds(1),), checks=CHECKS)
    checks_pass(monkeypatch)
    runner.invoke(app, ["keep"])
    short_sha = head_of(experiment_worktree_dir(repo))[:SHORT_SHA_LENGTH]

    result = runner.invoke(app, ["status"])

    history = [line for line in strip_ansi(result.stdout).splitlines() if line.strip()]
    assert history[-2] == f"baseline {short_sha} · {KEPT_MEDIANS_LINE}"
    assert history[-1] == "1 iteration · 1 kept · 0 discarded"


def test_keep_command_when_nothing_to_commit_does_exit_one_refusing_without_a_hint_label(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _start_unedited_session(repo)
    checks_pass(monkeypatch)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert (record.status, record.reason) == ("blocked", "nothing-to-commit")
    assert "Hint" not in result.stdout
    assert "run iterate again" in result.stdout
    assert "gymrat " not in result.stdout
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.seq, cmd.exit_code, cmd.reason) == ("keep", 1, 1, "nothing-to-commit")


def test_keep_command_when_checks_fail_does_exit_one_recording_the_block(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _start_edited_session(repo, checks=CHECKS)
    checks_fail(monkeypatch)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert (record.status, record.reason) == ("blocked", "checks-failed")
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.exit_code, cmd.reason) == ("keep", 1, "checks-failed")


def test_keep_command_when_outcome_not_improved_does_exit_one_refusing_with_both_ways_out(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _start_edited_session(repo, (unimproved(1, "no-signal"),), checks=CHECKS)
    checks_pass(monkeypatch)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert (record.status, record.reason) == ("blocked", "not-improved")
    assert "Keep refused: the iteration was no-signal, not improved." in strip_ansi(result.stdout)
    assert "pass --allow-unimproved to keep it anyway" in strip_ansi(result.stdout)
    cmd = last_command_record(repo)
    assert cmd.exit_code == 1
    assert cmd.reason == "not-improved"
    assert "allow_unimproved" not in cmd.args


def test_keep_command_when_allow_unimproved_does_commit_and_record_the_flag_in_args(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _start_edited_session(repo, (unimproved(1, "no-signal"),), checks=CHECKS)
    checks_pass(monkeypatch)

    result = runner.invoke(app, ["keep", "--allow-unimproved"])

    assert result.exit_code == 0
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "committed"
    cmd = last_command_record(repo)
    assert cmd.args["allow_unimproved"] is True
    assert cmd.reason is None


_SETTLE_TIMEOUT_S = 30.0
"""Budget for every pid-file and death-wait poll of the out-of-process keep test."""

_TRACKED_CHECKS = """#!/bin/sh
echo $$ > "{directory}/checks.pid"
sleep 120 &
echo $! > "{directory}/grandchild.pid"
wait
"""
"""A checks script that records its own pid and a background grandchild's, then blocks.

The checks never finish on their own, so ``keep`` is always mid-checks when signalled.
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell and signals")
@pytest.mark.parametrize(
    ("signal_name", "expected_code"),
    [
        pytest.param("SIGTERM", 143, id="sigterm"),
        pytest.param("SIGHUP", 129, id="sighup"),
        pytest.param("SIGINT", 130, id="sigint"),
    ],
)
def test_keep_command_when_signalled_mid_checks_does_exit_by_signal_code_leaving_no_checks_process_or_command_record(
    signal_name: str,
    expected_code: int,
    repo: str,
    tmp_path: Path,
    reap_groups: list[int],
):
    script = tmp_path / "checks.sh"
    script.write_text(_TRACKED_CHECKS.format(directory=tmp_path), encoding="utf-8")
    _start_edited_session(repo, checks=f"sh {shlex.quote(str(script))}")

    proc = subprocess.Popen(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [*ENTRY, "keep"],
        cwd=repo,
        env=no_color_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        checks_pid = wait_for_pid_file_blocking(tmp_path / "checks.pid", _SETTLE_TIMEOUT_S)
        reap_groups.append(checks_pid)
        grandchild = wait_for_pid_file_blocking(tmp_path / "grandchild.pid", _SETTLE_TIMEOUT_S)
        reap_groups.append(grandchild)
        proc.send_signal(signal.Signals[signal_name])
        proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == expected_code
    wait_until_dead_blocking(grandchild, timeout_s=_SETTLE_TIMEOUT_S)
    # Polled rather than checked once: a killed leader stays visible as a zombie
    # until its parent reaps it.
    wait_until_dead_blocking(checks_pid, timeout_s=_SETTLE_TIMEOUT_S)
    assert records_of_type(repo, CommandRecord) == []


@pytest.mark.parametrize(
    ("flags", "force"),
    [
        pytest.param([], False, id="plain"),
        pytest.param(["--force"], True, id="force"),
    ],
)
def test_discard_command_when_run_does_clean_the_worktree_and_record_the_discard_and_trace(
    repo: str, flags: list[str], force: bool
):
    _start_edited_session(repo)

    result = runner.invoke(app, ["discard", *flags])

    assert result.exit_code == 0
    assert status_of(experiment_worktree_dir(repo)) == ""
    assert settling_record_of(repo).type == "discard"
    assert re.search(r"discard", result.stdout, re.IGNORECASE)
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.args, cmd.seq, cmd.exit_code, cmd.reason) == (
        "discard",
        {"force": force},
        1,
        0,
        None,
    )


# ---------------------------------------------------------------------------
# budget time-left line — text output
# ---------------------------------------------------------------------------


def _settled_session(repo: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """Log a session with one kept iteration, for status."""
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_bench_config(repo)


def _keep_ready_session(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open an edited session whose checks pass, for keep."""
    _start_edited_session(repo, checks=CHECKS)
    checks_pass(monkeypatch)


def _discard_ready_session(repo: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """Open an edited session, for discard."""
    _start_edited_session(repo)


def _iterate_ready_session(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a fresh session whose stubbed bench improves, iterated from the tool."""
    write_session_log(repo, iterate_session_header(repo))
    stub_samples(install_collect_samples(monkeypatch), repo, improved_rounds(), baseline_rounds())
    set_origin(monkeypatch, "tool")


def _no_budget(_repo: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the session without a budget."""


@pytest.mark.parametrize(
    ("argv", "arrange", "budget", "trailers"),
    [
        pytest.param(["status"], _settled_session, install_budget, ["left of 30m"], id="status"),
        pytest.param(["status"], _settled_session, _no_budget, [], id="status-no-budget"),
        pytest.param(
            ["keep", "-m", "cache the regex"],
            _keep_ready_session,
            install_budget,
            ["left of 30m"],
            id="keep",
        ),
        pytest.param(
            ["discard"], _discard_ready_session, install_budget, ["left of 30m"], id="discard"
        ),
        pytest.param(
            ["iterate", "--bench", "npm run bench"],
            _iterate_ready_session,
            install_budget,
            ["left of 30m"],
            id="iterate",
        ),
    ],
)
def test_loop_command_when_run_does_end_text_with_a_time_left_line_only_under_a_budget(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    arrange: Callable[[str, pytest.MonkeyPatch], None],
    budget: Callable[[str, pytest.MonkeyPatch], None],
    trailers: list[str],
):
    arrange(repo, monkeypatch)
    budget(repo, monkeypatch)

    result = runner.invoke(app, argv)

    assert result.exit_code == 0
    assert re.findall(r"left of \d+m", strip_ansi(result.stdout)) == trailers


# ---------------------------------------------------------------------------
# --color / --no-color on keep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "env", "expect_ansi"),
    [
        pytest.param(["keep", "--no-color"], {"FORCE_COLOR": "1"}, False, id="no-color-flag"),
        pytest.param(["keep", "--color"], {}, True, id="color-flag"),
    ],
)
def test_keep_command_when_refusing_does_style_its_report_per_the_color_flags_and_env(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    env: dict[str, str],
    expect_ansi: bool,
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _start_unedited_session(repo)

    result = runner.invoke(app, argv)

    assert result.exit_code == 1
    assert bool(SGR_RE.search(result.stdout)) is expect_ansi


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
        pytest.param(["keep", "--color"], {}, runner, True, id="color-flag-without-tty"),
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
    _start_edited_session(repo)

    result = cli_runner.invoke(app, args, env=env)

    assert result.exit_code == 0
    assert "no checks command is configured" in strip_ansi(result.stderr)
    hint = result.stderr.splitlines()[1]
    assert bool(SGR_RE.search(hint)) is expect_ansi
