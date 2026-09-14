"""Command-level tests for iterate, keep, discard, status, and subdirectory resolution.

Each command is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
The seams mocked here mirror the engine suites' boundaries: for ``discard``'s
prompt the ``is_tty`` and ``confirm_action`` helpers as the ``loop_cmds`` module
imports them. Config resolution is exercised for real where a test lays down a
``gymrat.toml`` and stubbed at the ``loop_cmds`` seam where a test needs to pin
what a command reads (the runbook row) or observe where it looked (the
subdirectory case).

Iterate and JSON-contract tests live in ``test_loop_cmds_iterate`` and
``test_loop_cmds_json``.

Budget-line tests verify that loop commands append a time-left line to their
text output when a live budget is present, and omit it otherwise.
"""

import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from gymrat.cli import loop_cmds
from gymrat.cli.app import app
from gymrat.git import SHORT_SHA_LENGTH
from gymrat.loop.finalize import finalize_session
from gymrat.loop.start import start_session
from gymrat.session import (
    KeepRecord,
    SessionRecord,
    append_record,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from tests._ansi import SGR_RE
from tests.cli._budget import install_budget
from tests.cli._session import (
    always_tty,
    last_command_record,
    make_discard_repo,
    never_tty,
    runner,
    strip_ansi,
    write_config,
)
from tests.loop.iterate._fixtures import resolved_config
from tests.loop.settle._fixtures import (
    CHECKS,
    KEPT_MEDIANS_LINE,
    checks_fail,
    checks_pass,
    edit_experiment,
    git,
    head_of,
    iteration,
    measured_rounds,
    settling_record_of,
    start_with,
    status_of,
    unimproved,
)
from tests.session.records._fixtures import (
    SESSION_ID,
    committed_keep,
    iteration_record,
    session_record,
    write_session_log,
)


class _ConfirmRecorder:
    """A stand-in for ``confirm_action`` recording its calls and answering ``answer``."""

    def __init__(self, *, answer: bool) -> None:
        self.answer = answer
        self.calls: list[tuple[str, object]] = []

    def __call__(self, message: str, stream: object) -> bool:
        self.calls.append((message, stream))
        return self.answer


class _ResolverRecorder:
    """A stand-in for a config resolver recording ``(flags, base_dir)`` per call."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[object, str | Path | None]] = []

    def __call__(self, flags: object, base_dir: str | Path | None = None) -> object:
        self.calls.append((flags, base_dir))
        return self.result


def _record_lock_names(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch ``with_repo_lock`` to record every command it locks, forwarding through."""
    lock_names: list[str] = []
    original_with_repo_lock = loop_cmds.with_repo_lock

    async def recording_lock[T](
        command: str,
        body: Callable[..., Awaitable[T]],
        *,
        args: dict[str, object] | None = None,
    ) -> T:
        lock_names.append(command)
        return await original_with_repo_lock(command, body, args=args)

    monkeypatch.setattr(loop_cmds, "with_repo_lock", recording_lock)
    return lock_names


def _open_session_with_one_keep(root: str) -> SessionRecord:
    """Open a session, commit and log one kept iteration, and return the session header."""
    start_session(root, "main", resolved_config())
    worktree = experiment_worktree_dir(root)
    (Path(worktree) / "step.txt").write_text("cache the regex\n", encoding="utf-8")
    git(["add", "-A"], worktree)
    git(["commit", "-m", "cache the regex"], worktree)
    commit = head_of(worktree)
    append_record(session_jsonl_path(root), iteration_record(seq=1))
    append_record(session_jsonl_path(root), committed_keep(1, commit=commit))
    header = read_records(session_jsonl_path(root))[0]
    assert isinstance(header, SessionRecord)
    return header


def _close_session_with_one_keep(root: str) -> str:
    """Open a session with one kept commit, finalize it, and return its closed id."""
    header = _open_session_with_one_keep(root)
    finalize_session(root)
    return header.session_id


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
) -> None:
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
    _close_session_with_one_keep(repo)
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
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)

    result = runner.invoke(app, ["status", *args])

    assert result.exit_code == 0
    assert bool(SGR_RE.search(result.stdout)) is expect_ansi


def test_status_command_when_stdout_broken_pipe_does_exit_zero(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)

    def broken_write(_stream: object, _data: str) -> None:
        raise BrokenPipeError

    monkeypatch.setattr("gymrat.cli.loop_cmds.write_and_flush", broken_write)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0


def test_status_command_when_run_inside_the_experiment_worktree_does_render_the_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    header = _open_session_with_one_keep(repo)
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


@pytest.fixture
def discard_repo(repo: str) -> str:
    """A repository with an open session and one unsettled iteration to discard."""
    return make_discard_repo(repo)


def test_discard_command_documents_force_in_its_help():
    from tests.cli._help import help_output

    assert "--force" in help_output("discard")


def test_discard_command_when_tty_and_confirmed_does_prompt_and_proceed(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", always_tty)
    confirm = _ConfirmRecorder(answer=True)
    monkeypatch.setattr("gymrat.cli.loop_cmds.confirm_action", confirm)

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 0
    assert len(confirm.calls) == 1
    assert experiment_worktree_dir(discard_repo) in confirm.calls[0][0]
    assert re.search(r"discard", result.stdout, re.IGNORECASE)


def test_discard_command_when_tty_and_declined_does_cancel_with_exit_one(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", always_tty)
    monkeypatch.setattr("gymrat.cli.loop_cmds.confirm_action", _ConfirmRecorder(answer=False))

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 1
    assert "discard cancelled" in result.stderr


@pytest.mark.parametrize("flag", ["--force", "-f"])
def test_discard_command_when_force_does_skip_the_prompt(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch, flag: str
):
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", always_tty)
    confirm = _ConfirmRecorder(answer=True)
    monkeypatch.setattr("gymrat.cli.loop_cmds.confirm_action", confirm)

    result = runner.invoke(app, ["discard", flag])

    assert result.exit_code == 0
    assert confirm.calls == []
    assert re.search(r"discard", result.stdout, re.IGNORECASE)


def test_discard_command_when_stdin_not_tty_does_skip_the_prompt(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", never_tty)
    confirm = _ConfirmRecorder(answer=True)
    monkeypatch.setattr("gymrat.cli.loop_cmds.confirm_action", confirm)

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 0
    assert confirm.calls == []
    assert re.search(r"discard", result.stdout, re.IGNORECASE)


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
    recorder = _ResolverRecorder(resolved_config())
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
) -> None:
    result = runner.invoke(app, ["keep", flag, value])

    assert result.exit_code == 2
    assert "No such option" in result.stderr


def test_keep_command_when_checks_pass_does_commit_and_print_the_short_commit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
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
    start_with(repo, (iteration(1),))
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
    start_with(repo, (iteration(1),))
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
    start_with(repo, (iteration(1),))
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
    _close_session_with_one_keep(repo)
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
    start_with(repo, (iteration(1),))
    checks_pass(monkeypatch)
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    record = settling_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "blocked"
    assert record.reason == "nothing-to-commit"


def test_keep_command_when_refusing_does_print_a_report_carrying_no_hint_label(repo: str):
    start_with(repo, (iteration(1),))
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
    for name in ("FORCE_COLOR", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "1")
    start_with(repo, (iteration(1),))
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert bool(SGR_RE.search(result.stdout)) is expect_ansi


def test_keep_command_when_help_requested_does_document_allow_unimproved():
    from tests.cli._help import help_output

    help_text = help_output("keep")

    assert "--allow-unimproved" in help_text
    assert "keep the edit even when the iteration was not improved" in help_text


def test_keep_command_when_help_does_describe_message_as_commit_message_for_kept_edit():
    from tests.cli._help import help_output

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
    start_with(repo, (iteration(1),))
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
    start_with(repo, (iteration(1),))
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
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    write_config(repo)

    result = runner.invoke(app, ["discard", "--force"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.args == {"force": True}


def test_discard_command_when_finalized_does_record_command_trace_with_exit_two(
    repo: str,
):
    _close_session_with_one_keep(repo)
    write_config(repo)

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 2
    cmd = last_command_record(repo)
    assert cmd.name == "discard"
    assert cmd.exit_code == 2
    assert cmd.reason == "finalized"


def test_discard_command_when_run_does_clean_the_worktree_and_record_the_discard(repo: str):
    start_with(repo, (iteration(1),))
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
    start_with(repo, (iteration(1),))
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
    start_with(repo, (iteration(1),))
    write_config(repo, checks=CHECKS)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    text = strip_ansi(result.stdout)
    assert re.search(r"left of 30m", text)


def test_discard_command_when_budget_active_does_end_text_with_time_left_line(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
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
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    start_with(repo, (iteration(1),))
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep", "--no-color"])

    assert result.exit_code == 1
    assert not SGR_RE.search(result.stdout)


def test_keep_command_when_color_does_force_ansi_on_stdout_report(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    start_with(repo, (iteration(1),))
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep", "--color"])

    assert result.exit_code == 1
    assert SGR_RE.search(result.stdout)


def test_discard_command_when_no_color_does_strip_ansi_from_stderr_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    write_config(repo)

    result = runner.invoke(app, ["discard", "--no-color"])

    assert result.exit_code == 0
    assert not SGR_RE.search(result.stdout)
