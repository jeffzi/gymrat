"""Command-level tests for start, status, keep, discard, finalize, sync, and subdirectory resolution.

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
from gymrat.loop.finalize import finalize_session
from gymrat.loop.start import start_session
from gymrat.session import (
    FinalizeRecord,
    KeepRecord,
    SessionRecord,
    append_record,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from tests._ansi import SGR_RE
from tests.cli._budget import install_budget
from tests.cli._help import help_output
from tests.cli._loop_cmds import (
    always_tty,
    make_discard_repo,
    never_tty,
    runner,
    strip_ansi,
    write_config,
)
from tests.loop.iterate._fixtures import resolved_config
from tests.loop.settle._fixtures import (
    CHECKS,
    checks_fail,
    checks_pass,
    edit_experiment,
    git,
    head_of,
    iteration,
    last_record_of,
    start_with,
    status_of,
)
from tests.session.records._fixtures import (
    SESSION_ID,
    committed_keep,
    iteration_record,
    session_record,
    write_session_log,
)

#: The runbook path a session's config points an agent at, when it has one.
_RUNBOOK_PATH = ".claude/skills/ecstatic-bench/SKILL.md"


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


# ---------------------------------------------------------------------------
# the start command
# ---------------------------------------------------------------------------


def _stub_resolve_config(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> object:
    """Pin what ``start`` reads by replacing its ``resolve_config`` with a fixed config."""
    config = resolved_config(**overrides)

    def fake(*_a: object, **_k: object) -> object:
        return config

    monkeypatch.setattr("gymrat.cli.loop_cmds.resolve_config", fake)
    return config


def _close_session_with_one_keep(root: str) -> str:
    """Open a session with one kept commit, finalize it, and return its closed id."""
    start_session(root, "main", resolved_config())
    worktree = experiment_worktree_dir(root)
    (Path(worktree) / "README.md").write_text("# cache the regex\n", encoding="utf-8")
    git(["add", "README.md"], worktree)
    git(["commit", "-m", "cache the regex"], worktree)
    commit = head_of(worktree)
    append_record(session_jsonl_path(root), iteration_record(seq=1))
    append_record(session_jsonl_path(root), committed_keep(1, commit=commit))
    finalize_session(root)
    header = read_records(session_jsonl_path(root))[0]
    assert isinstance(header, SessionRecord)
    return header.session_id


def test_start_command_when_run_does_create_a_session_and_report_its_branch(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "main"])

    assert result.exit_code == 0
    header = read_records(session_jsonl_path(repo))[0]
    assert isinstance(header, SessionRecord)
    assert header.branch in result.stdout


def test_start_command_when_reopening_after_finalize_does_name_the_archived_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    closed_id = _close_session_with_one_keep(repo)
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "main"])

    assert result.exit_code == 0
    assert re.search(r"archived", result.stdout, re.IGNORECASE)
    assert closed_id in result.stdout


@pytest.mark.parametrize("resumed", [False, True])
def test_start_command_when_runbook_configured_does_include_a_runbook_row(
    repo: str, monkeypatch: pytest.MonkeyPatch, resumed: bool
):
    if resumed:
        start_session(repo, "main", resolved_config())
    _stub_resolve_config(monkeypatch, runbook=_RUNBOOK_PATH)

    result = runner.invoke(app, ["start", "main"])

    assert result.exit_code == 0
    assert f"runbook: {_RUNBOOK_PATH} — read it before your first edit" in result.stdout


def test_start_command_when_runbook_absent_does_omit_the_runbook_row(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "main"])

    assert result.exit_code == 0
    assert "runbook" not in result.stdout


@pytest.mark.parametrize("resumed", [False, True])
def test_start_command_when_run_does_include_edit_here_line_with_sync_hint(
    repo: str, monkeypatch: pytest.MonkeyPatch, resumed: bool
):
    if resumed:
        start_session(repo, "main", resolved_config())
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "main"])

    assert result.exit_code == 0
    exp_dir = experiment_worktree_dir(repo)
    assert f"edit in {exp_dir}" in result.stdout
    assert "gymrat sync" in result.stdout


# ---------------------------------------------------------------------------
# the status command
# ---------------------------------------------------------------------------


def test_status_command_when_run_does_render_the_session_on_stdout(repo: str):
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    text = strip_ansi(result.stdout)
    assert f"session {SESSION_ID}" in text
    assert "1 kept" in text


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


# ---------------------------------------------------------------------------
# the discard command
# ---------------------------------------------------------------------------


@pytest.fixture
def discard_repo(repo: str) -> str:
    """A repository with an open session and one unsettled iteration to discard."""
    return make_discard_repo(repo)


def test_discard_command_documents_force_in_its_help():
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
# the finalize command
# ---------------------------------------------------------------------------


def _session_with_one_keep(root: str) -> str:
    """Open a session with one kept commit on it, and return its branch."""
    start_session(root, "main", resolved_config())
    header = read_records(session_jsonl_path(root))[0]
    assert isinstance(header, SessionRecord)
    branch = header.branch
    worktree = experiment_worktree_dir(root)
    (Path(worktree) / "step.txt").write_text("cache the regex\n", encoding="utf-8")
    git(["add", "-A"], worktree)
    git(["commit", "-m", "cache the regex"], worktree)
    commit = head_of(worktree)
    append_record(session_jsonl_path(root), iteration_record(seq=1))
    append_record(session_jsonl_path(root), committed_keep(1, commit=commit))
    return branch


def test_finalize_command_documents_its_flags_and_default_branch_in_help():
    out = help_output("finalize")

    assert "--message" in out
    assert "--branch" in out
    assert "-final" in out


@pytest.mark.parametrize(
    ("args", "named"),
    [
        pytest.param(
            ["--branch", "perf/regex-cache"], "perf/regex-cache", id="branch-the-caller-named"
        ),
        pytest.param([], None, id="session-branch-final-by-default"),
    ],
)
def test_finalize_command_records_and_reports_the_branch(
    repo: str, args: list[str], named: str | None
):
    branch = _session_with_one_keep(repo)
    final_branch = named if named is not None else f"{branch}-final"

    result = runner.invoke(app, ["finalize", *args])

    assert result.exit_code == 0
    record = last_record_of(repo)
    assert isinstance(record, FinalizeRecord)
    assert record.branch == final_branch
    assert final_branch in result.stdout


def test_finalize_command_commits_the_message_it_was_given(repo: str):
    _session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "-m", "squash the tuning session"])

    assert result.exit_code == 0
    record = last_record_of(repo)
    assert isinstance(record, FinalizeRecord)
    assert record.message == "squash the tuning session"


def test_finalize_command_when_no_session_does_exit_two_with_a_start_hint(repo: str):
    result = runner.invoke(app, ["finalize"])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr


# ---------------------------------------------------------------------------
# the loop commands, run from a subdirectory of the repository
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "resolver_name"),
    [
        pytest.param(["start", "main"], "resolve_config", id="start"),
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

    # Whether the command then finds the session it needs is beside the point;
    # where it looked the configuration up is what is under test.
    runner.invoke(app, args)

    assert recorder.calls
    base_dir = recorder.calls[0][1]
    assert base_dir is not None
    assert Path(base_dir) == Path(root)


# ---------------------------------------------------------------------------
# the sync command
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_repo(repo: str) -> str:
    """A repository with an open session, ready for sync tests."""
    start_session(repo, "main", resolved_config())
    return repo


def test_sync_command_when_registered_does_appear_in_the_app_commands():
    assert "sync" in help_output("sync").lower()


def test_sync_command_when_changes_exist_does_print_synced_file_count_and_names(
    sync_repo: str,
):
    root = Path(sync_repo)
    (root / "extra.py").write_text("# new\n", encoding="utf-8")
    (root / "README.md").write_text("# updated\n", encoding="utf-8")

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "2 files" in result.stdout
    assert "extra.py" in result.stdout
    assert "README.md" in result.stdout


def test_sync_command_when_nothing_to_sync_does_print_nothing_to_sync(
    sync_repo: str,
):
    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "nothing to sync" in result.stdout


def test_sync_command_when_run_does_take_the_repo_lock(
    sync_repo: str, monkeypatch: pytest.MonkeyPatch
):
    lock_names: list[str] = []
    original_with_repo_lock = loop_cmds.with_repo_lock

    async def recording_lock[T](command: str, body: Callable[[], Awaitable[T]]) -> T:
        lock_names.append(command)
        return await original_with_repo_lock(command, body)

    monkeypatch.setattr(loop_cmds, "with_repo_lock", recording_lock)

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "sync" in lock_names


def test_sync_command_when_no_session_does_exit_two_with_a_start_hint(
    repo: str,
):
    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr


# ---------------------------------------------------------------------------
# the keep command
# ---------------------------------------------------------------------------


def test_keep_command_when_checks_pass_does_commit_and_print_the_short_commit(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_pass(monkeypatch)
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep", "-m", "cache the regex"])

    assert result.exit_code == 0
    record = last_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "committed"
    assert head_of(experiment_worktree_dir(repo))[:7] in result.stdout


def test_keep_command_when_nothing_to_commit_does_exit_one_recording_the_block(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    checks_pass(monkeypatch)
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    record = last_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "blocked"
    assert record.reason == "nothing-to-commit"


def test_keep_command_when_refusing_does_print_a_report_carrying_no_hint_label(repo: str):
    start_with(repo, (iteration(1),))
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    assert "Hint" not in result.stdout
    assert "gymrat iterate" in result.stdout


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


def test_keep_command_when_checks_fail_does_exit_one_recording_the_block(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    checks_fail(monkeypatch)
    write_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["keep"])

    assert result.exit_code == 1
    record = last_record_of(repo)
    assert isinstance(record, KeepRecord)
    assert record.status == "blocked"
    assert record.reason == "checks-failed"


def test_discard_command_when_run_does_clean_the_worktree_and_record_the_discard(repo: str):
    start_with(repo, (iteration(1),))
    edit_experiment(repo)
    write_config(repo)

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 0
    assert status_of(experiment_worktree_dir(repo)) == ""
    assert last_record_of(repo).type == "discard"
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


def test_sync_command_when_budget_active_does_end_text_with_time_left_line(
    sync_repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_budget(sync_repo, monkeypatch)

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    text = strip_ansi(result.stdout)
    assert re.search(r"left of 30m", text)


def test_sync_command_when_no_budget_does_omit_time_left_line(
    sync_repo: str,
):
    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "left of" not in result.stdout


def test_start_command_when_budget_active_does_not_include_time_left_line(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)
    # start creates the session dir itself, but write_budget needs it
    Path(repo, ".gymrat").mkdir(exist_ok=True)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["start", "main"])

    assert result.exit_code == 0
    assert "left of" not in result.stdout


def test_finalize_command_when_budget_active_does_not_include_time_left_line(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _session_with_one_keep(repo)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["finalize"])

    assert result.exit_code == 0
    assert "left of" not in result.stdout
