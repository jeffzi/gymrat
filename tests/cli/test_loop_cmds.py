"""Command-level tests for keep, discard, status, and subdirectory resolution.

Each command is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
The seams mocked here mirror the engine suites' boundaries: for ``discard``'s
prompt the ``is_tty`` check as the ``loop_cmds`` module imports it, with the
answer typed on the runner's stdin. The prompt's broken-stderr cases run the
CLI in a child process, since only a real file descriptor can be closed or
broken. Config resolution is exercised for real where a test lays down a
``gymrat.toml`` and stubbed at the ``loop_cmds`` seam in the subdirectory case,
where the test observes the directory a command resolves its config from.

Iterate and JSON-contract tests live in ``test_loop_cmds_iterate`` and
``test_loop_cmds_json``.

Budget-line tests verify that loop commands append a time-left line to their
text output when a live budget is present, and omit it otherwise.
"""

import contextlib
import io
import re
import subprocess
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, override

import pytest
from typer.testing import CliRunner

from gymrat.cli.app import app
from gymrat.git import SHORT_SHA_LENGTH
from gymrat.session.paths import experiment_worktree_dir, session_jsonl_path
from gymrat.session.records import KeepRecord
from gymrat.session.store import read_records
from tests._ansi import SGR_RE, strip_ansi
from tests._cli import no_color_env
from tests._process_helpers import run_with_closed_reader
from tests.cli._budget import install_budget
from tests.cli._help import help_output
from tests.cli._session import (
    FailingStdoutRunner,
    ResolverRecorder,
    always_tty,
    close_session_with_one_keep,
    closed_stdout_error,
    last_command_record,
    never_tty,
    open_session_with_one_keep,
    runner,
    write_config,
)
from tests.loop._settle import (
    CHECKS,
    KEPT_MEDIANS_LINE,
    checks_fail,
    checks_pass,
    edit_experiment,
    head_of,
    measured_rounds,
    settling_record_of,
    start_with,
    status_of,
    unimproved,
)
from tests.loop.iterate._fixtures import resolved_config
from tests.session.records._fixtures import (
    SESSION_ID,
    committed_keep,
    iteration_record,
    session_record,
    write_session_log,
)


def _start_edited_session(root: str, **config: object) -> None:
    """Open a session with one unsettled iteration, edit the experiment, and write the config."""
    start_with(root, (iteration_record(seq=1),))
    edit_experiment(root)
    write_config(root, **config)


# ---------------------------------------------------------------------------
# the status command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--bench", "x"),
        ("--prepare", "x"),
        ("--adapter", "x"),
        ("--samples", "3"),
        ("--timeout", "30"),
    ],
)
def test_status_command_when_given_a_bench_run_flag_does_exit_two_with_usage_error(
    flag: str, value: str
):
    result = runner.invoke(app, ["status", flag, value])

    assert result.exit_code == 2
    assert "No such option" in result.stderr


def test_status_command_when_run_does_render_the_session_on_stdout(repo: str):
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    text = strip_ansi(result.stdout)
    assert f"session {SESSION_ID}" in text
    assert "1 kept" in text


def test_status_command_when_run_does_record_command_trace_with_exit_zero(repo: str):
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.name == "status"
    assert cmd.args == {}
    assert cmd.exit_code == 0
    assert cmd.reason is None


def test_status_command_when_finalized_does_record_command_trace_with_exit_zero(repo: str):
    close_session_with_one_keep(repo)
    write_config(repo)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.name == "status"
    assert cmd.exit_code == 0


@pytest.mark.parametrize("has_config", [True, False])
def test_status_command_when_no_session_does_exit_two_with_a_start_hint(
    repo: str, has_config: bool
):
    if has_config:
        write_config(repo)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr


@pytest.mark.parametrize(
    ("args", "expect_ansi"),
    [
        pytest.param([], True, id="color-on-by-default"),
        pytest.param(["--no-color"], False, id="no-color-beats-force-color"),
    ],
)
def test_status_command_color(
    repo: str, monkeypatch: pytest.MonkeyPatch, args: list[str], expect_ansi: bool
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)

    result = runner.invoke(app, ["status", *args])

    assert result.exit_code == 0
    assert bool(SGR_RE.search(result.stdout)) is expect_ansi


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
    write_config(repo)
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
    "from gymrat.cli import loop_cmds\n"
    "loop_cmds.is_tty = lambda _stream: True\n"
    "sys.argv = ['gymrat', 'discard']\n"
    "runpy.run_module('gymrat.cli.app', run_name='__main__')\n"
)
"""A child-process driver running ``gymrat discard`` as if stdin were a terminal."""


def _discard_state(repo: str) -> tuple[bool, bool]:
    """Whether the experiment edit is still present, and whether a discard was logged."""
    edit_present = status_of(experiment_worktree_dir(repo)) != ""
    discard_logged = any(
        record.type == "discard" for record in read_records(session_jsonl_path(repo))
    )
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
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", always_tty)


def test_discard_command_when_help_requested_does_document_force():
    assert "--force" in help_output("discard")


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
        pytest.param(["--force"], always_tty, id="force-long"),
        pytest.param(["-f"], always_tty, id="force-short"),
        pytest.param([], never_tty, id="stdin-not-tty"),
    ],
)
def test_discard_command_when_force_or_stdin_not_tty_does_skip_the_prompt(
    edited_repo: str,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    is_tty_stub: Callable[[object], bool],
):
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", is_tty_stub)

    result = runner.invoke(app, ["discard", *args], input="n\n")

    assert result.exit_code == 0
    assert _PROMPT_CHOICES not in result.stderr
    assert _discard_state(edited_repo) == (False, True)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell")
def test_discard_command_when_stderr_closed_does_decline_silently(edited_repo: str):
    result = subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
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
    ("args", "resolver_name"),
    [
        pytest.param(["iterate"], "resolve_config", id="iterate"),
        pytest.param(["keep"], "resolve_benchless_config", id="keep"),
        pytest.param(["status"], "resolve_benchless_config", id="status"),
    ],
)
def test_loop_command_when_run_from_subdirectory_does_resolve_config_at_repo_root(
    create_scratch_repo: Callable[[], str],
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    resolver_name: str,
):
    root = create_scratch_repo()
    nested = Path(root) / "packages" / "core"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    recorder = ResolverRecorder(resolved_config())
    monkeypatch.setattr(f"gymrat.cli.loop_cmds.{resolver_name}", recorder)

    runner.invoke(app, args)

    assert recorder.calls
    base_dir = recorder.calls[0][1]
    assert base_dir is not None
    assert Path(base_dir) == Path(root)


# ---------------------------------------------------------------------------
# the keep command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--bench", "x"),
        ("--prepare", "x"),
        ("--adapter", "x"),
        ("--samples", "3"),
    ],
)
def test_keep_command_when_given_a_bench_run_flag_does_exit_two_with_usage_error(
    flag: str, value: str
):
    result = runner.invoke(app, ["keep", flag, value])

    assert result.exit_code == 2
    assert "No such option" in result.stderr


def test_keep_command_when_checks_pass_does_commit_and_print_the_short_commit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep", "-m", "cache the regex"])

    assert result.exit_code == 0
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "committed"
    assert head_of(experiment_worktree_dir(repo))[:7] in result.stdout


def test_keep_command_when_committed_does_add_the_kept_baseline_to_the_status_history(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (measured_rounds(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    write_config(repo, checks=CHECKS)
    assert runner.invoke(app, ["keep"]).exit_code == 0
    short_sha = head_of(experiment_worktree_dir(repo))[:SHORT_SHA_LENGTH]

    result = runner.invoke(app, ["status"])

    history = [line for line in strip_ansi(result.stdout).splitlines() if line.strip()]
    assert history[-2] == f"baseline {short_sha} · {KEPT_MEDIANS_LINE}"
    assert history[-1] == "1 iteration · 1 kept · 0 discarded"


def test_keep_command_when_committed_does_record_command_trace_with_seq_and_exit_zero(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep", "-m", "cache the regex"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.name == "keep"
    assert cmd.args == {"message": "cache the regex"}
    assert cmd.seq == 1
    assert cmd.exit_code == 0
    assert cmd.reason is None


def test_keep_command_when_blocked_does_record_command_trace_with_gate_and_reason(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    checks_pass(monkeypatch)
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    cmd = last_command_record(repo)
    assert cmd.name == "keep"
    assert cmd.seq == 1
    assert cmd.exit_code == 1
    assert cmd.reason == "nothing-to-commit"


def test_keep_command_when_checks_fail_does_record_command_trace_with_checks_failed_reason(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    checks_fail(monkeypatch)
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    cmd = last_command_record(repo)
    assert cmd.name == "keep"
    assert cmd.exit_code == 1
    assert cmd.reason == "checks-failed"


def test_keep_command_when_finalized_does_record_command_trace_with_exit_two(
    repo: str,
):
    close_session_with_one_keep(repo)
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 2
    cmd = last_command_record(repo)
    assert cmd.name == "keep"
    assert cmd.exit_code == 2
    assert cmd.reason == "finalized"


def test_keep_command_when_nothing_to_commit_does_exit_one_recording_the_block(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    checks_pass(monkeypatch)
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "blocked"
    assert record.reason == "nothing-to-commit"


def test_keep_command_when_refusing_does_print_a_report_carrying_no_hint_label(repo: str):
    start_with(repo, (iteration_record(seq=1),))
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    assert "Hint" not in result.stdout
    assert "run iterate again" in result.stdout
    assert "gymrat " not in result.stdout


@pytest.mark.parametrize(
    ("variable", "expect_ansi"),
    [
        pytest.param("FORCE_COLOR", True, id="force-color"),
        pytest.param("NO_COLOR", False, id="no-color"),
    ],
)
def test_keep_command_when_refusing_does_take_report_color_from_the_environment(
    repo: str, monkeypatch: pytest.MonkeyPatch, variable: str, expect_ansi: bool
):
    monkeypatch.setenv(variable, "1")
    start_with(repo, (iteration_record(seq=1),))
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert bool(SGR_RE.search(result.stdout)) is expect_ansi


def test_keep_command_when_help_requested_does_document_allow_unimproved():
    help_text = help_output("keep")

    assert "--allow-unimproved" in help_text
    assert "keep the edit even when the iteration was not improved" in help_text


def test_keep_command_when_help_does_describe_message_as_commit_message_for_kept_edit():
    help_text = help_output("keep")

    assert "commit message for the kept edit" in help_text


def test_keep_command_when_outcome_not_improved_does_exit_one_refusing_with_both_ways_out(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (unimproved(1, "no-signal"),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    write_config(repo, checks=CHECKS)

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
    start_with(repo, (unimproved(1, "no-signal"),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep", "--allow-unimproved"])

    assert result.exit_code == 0
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "committed"
    cmd = last_command_record(repo)
    assert cmd.args["allow_unimproved"] is True
    assert cmd.reason is None


def test_keep_command_when_checks_fail_does_exit_one_recording_the_block(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    checks_fail(monkeypatch)
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "blocked"
    assert record.reason == "checks-failed"


def test_discard_command_when_run_does_record_command_trace_with_seq_and_force(
    repo: str,
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    write_config(repo)

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.name == "discard"
    assert cmd.args == {"force": False}
    assert cmd.seq == 1
    assert cmd.exit_code == 0
    assert cmd.reason is None


def test_discard_command_when_force_does_record_force_true_in_args(
    repo: str,
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    write_config(repo)

    result = runner.invoke(app, ["discard", "--force"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.args == {"force": True}


def test_discard_command_when_finalized_does_record_command_trace_with_exit_two(
    repo: str,
):
    close_session_with_one_keep(repo)
    write_config(repo)

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 2
    cmd = last_command_record(repo)
    assert cmd.name == "discard"
    assert cmd.exit_code == 2
    assert cmd.reason == "finalized"


def test_discard_command_when_run_does_clean_the_worktree_and_record_the_discard(repo: str):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    write_config(repo)

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 0
    assert status_of(experiment_worktree_dir(repo)) == ""
    assert settling_record_of(repo).type == "discard"
    assert re.search(r"discard", result.stdout, re.IGNORECASE)


@pytest.mark.parametrize("command", ["keep", "discard"])
def test_settle_command_when_no_session_does_exit_two_with_a_start_hint(repo: str, command: str):
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, [command])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr


# ---------------------------------------------------------------------------
# budget time-left line — text output
# ---------------------------------------------------------------------------


def test_status_command_when_budget_active_does_end_text_with_time_left_line(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    text = strip_ansi(result.stdout)
    assert re.search(r"left of 30m", text)


def test_status_command_when_no_budget_does_omit_time_left_line(
    repo: str,
):
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "left of" not in result.stdout


def test_keep_command_when_budget_active_and_committed_does_end_text_with_time_left_line(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    write_config(repo, checks=CHECKS)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["keep", "-m", "cache the regex"])

    assert result.exit_code == 0
    text = strip_ansi(result.stdout)
    assert re.search(r"left of 30m", text)


def test_keep_command_when_budget_active_and_blocked_does_end_text_with_time_left_line(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    write_config(repo, checks=CHECKS)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    text = strip_ansi(result.stdout)
    assert re.search(r"left of 30m", text)


def test_discard_command_when_budget_active_does_end_text_with_time_left_line(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    write_config(repo)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 0
    text = strip_ansi(result.stdout)
    assert re.search(r"left of 30m", text)


# ---------------------------------------------------------------------------
# --color / --no-color on keep and discard
# ---------------------------------------------------------------------------


def test_keep_command_when_no_color_does_strip_ansi_from_stdout_report(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    start_with(repo, (iteration_record(seq=1),))
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep", "--no-color"])

    assert result.exit_code == 1
    assert not SGR_RE.search(result.stdout)


def test_keep_command_when_color_does_force_ansi_on_stdout_report(repo: str):
    start_with(repo, (iteration_record(seq=1),))
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep", "--color"])

    assert result.exit_code == 1
    assert SGR_RE.search(result.stdout)


#: Environment that forces color on, alone or under a --no-color flag that must outrank it.
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
        pytest.param(["--no-color", "keep"], _FORCE_COLOR, runner, False, id="root-no-color"),
        pytest.param(["keep"], {}, _TerminalStderrRunner(), True, id="stderr-tty-stdout-piped"),
        pytest.param(["keep"], _FORCE_COLOR, runner, True, id="force-color-without-tty"),
        pytest.param(
            ["keep"], {"NO_COLOR": "1"}, _TerminalStderrRunner(), False, id="no-color-on-tty"
        ),
        pytest.param(["--color", "keep", "--no-color"], {}, runner, False, id="command-beats-root"),
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


def test_discard_command_when_no_color_does_strip_ansi_from_stderr_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    start_with(repo, (iteration_record(seq=1),))
    edit_experiment(repo)
    write_config(repo)

    result = runner.invoke(app, ["discard", "--no-color"])

    assert result.exit_code == 0
    assert not SGR_RE.search(result.stdout)
