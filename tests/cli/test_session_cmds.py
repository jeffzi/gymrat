"""Command-level tests for start, finalize, stop, sync, and subdirectory resolution.

Each command is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.

Budget-line tests verify that loop commands append a time-left line to their
text output when a live budget is present, and omit it otherwise.
"""

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli import session_cmds
from gymrat.cli.app import app
from gymrat.loop.finalize import finalize_session
from gymrat.loop.start import start_session
from gymrat.session import (
    FinalizeRecord,
    SessionRecord,
    StopRecord,
    append_record,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from tests._ansi import SGR_RE
from tests.cli._budget import install_budget
from tests.cli._session import (
    last_command_record,
    make_stop_repo,
    runner,
    strip_ansi,
    write_config,
)
from tests.loop.iterate._fixtures import resolved_config
from tests.loop.settle._fixtures import (
    git,
    head_of,
    settling_record_of,
)
from tests.session.records._fixtures import (
    committed_keep,
    iteration_record,
)


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

    monkeypatch.setattr("gymrat.cli.session_cmds.resolve_config", fake)
    return config


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


def test_start_command_when_run_does_create_a_session_and_report_its_branch(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0
    header = read_records(session_jsonl_path(repo))[0]
    assert isinstance(header, SessionRecord)
    assert header.branch in result.stdout


def test_start_command_when_reopening_after_finalize_does_name_the_archived_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    closed_id = _close_session_with_one_keep(repo)
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0
    assert re.search(r"archived", result.stdout, re.IGNORECASE)
    assert closed_id in result.stdout


@pytest.mark.parametrize("resumed", [False, True])
def test_start_command_when_runbook_configured_does_include_a_runbook_row(
    repo: str, monkeypatch: pytest.MonkeyPatch, resumed: bool
):
    if resumed:
        start_session(repo, "main", resolved_config())
    _stub_resolve_config(monkeypatch, runbook=".claude/skills/ecstatic-bench/SKILL.md")

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0
    assert (
        "runbook: .claude/skills/ecstatic-bench/SKILL.md — read it before your first edit"
        in result.stdout
    )


def test_start_command_when_runbook_absent_does_omit_the_runbook_row(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0
    assert "runbook" not in result.stdout


def test_start_command_when_run_does_record_command_trace_with_baseline_and_exit_zero(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.name == "start"
    assert cmd.args == {"baseline": "main"}
    assert cmd.exit_code == 0
    assert cmd.reason is None


def test_start_command_when_config_overrides_given_does_record_them_in_args(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(
        app, ["start", "--baseline", "main", "--bench", "sh run.sh", "--samples", "5"]
    )

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.name == "start"
    assert cmd.args["baseline"] == "main"
    assert cmd.args["bench"] == "sh run.sh"
    assert cmd.args["samples"] == 5


@pytest.mark.parametrize("resumed", [False, True])
def test_start_command_when_run_does_include_edit_here_line_with_sync_hint(
    repo: str, monkeypatch: pytest.MonkeyPatch, resumed: bool
):
    if resumed:
        start_session(repo, "main", resolved_config())
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0
    exp_dir = experiment_worktree_dir(repo)
    assert f"edit in {exp_dir}" in result.stdout
    assert "gymrat sync" in result.stdout


def test_start_command_when_no_baseline_does_default_to_head(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start"])

    assert result.exit_code == 0
    header = read_records(session_jsonl_path(repo))[0]
    assert isinstance(header, SessionRecord)


def test_start_command_when_positional_ref_given_does_exit_two_with_usage_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "main"])

    assert result.exit_code == 2
    combined = result.stdout + result.stderr
    assert "unexpected extra argument" in combined.lower() or "no such option" in combined.lower()


def test_start_command_help_does_show_baseline_option_with_help_string():
    from tests.cli._help import help_output

    out = help_output("start")

    assert "--baseline" in out
    assert "git ref that pins a freshly opened session" in out


# ---------------------------------------------------------------------------
# the finalize command
# ---------------------------------------------------------------------------


def _session_with_one_keep(root: str) -> str:
    """Open a session with one kept commit on it, and return its branch."""
    return _open_session_with_one_keep(root).branch


def test_finalize_command_documents_its_flags_and_default_branch_in_help():
    from tests.cli._help import help_output

    out = help_output("finalize")

    assert "--message" in out
    assert "--branch" in out
    assert "-final" in out


def test_finalize_command_when_help_does_describe_message_as_squash_commit():
    from tests.cli._help import help_output

    out = help_output("finalize")

    assert "message for the squash commit" in out


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
    record = settling_record_of(repo)
    assert isinstance(record, FinalizeRecord)
    assert record.branch == final_branch
    assert final_branch in result.stdout


def test_finalize_command_commits_the_message_it_was_given(repo: str):
    _session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "-m", "squash the tuning session"])

    assert result.exit_code == 0
    record = settling_record_of(repo)
    assert isinstance(record, FinalizeRecord)
    assert record.message == "squash the tuning session"


def test_finalize_command_when_run_does_record_command_trace_with_branch_and_message(
    repo: str,
):
    _session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "--branch", "perf/regex", "-m", "squash the session"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.name == "finalize"
    assert cmd.args == {"branch": "perf/regex", "message": "squash the session"}
    assert cmd.exit_code == 0
    assert cmd.reason is None


def test_finalize_command_when_finalized_does_record_command_trace_with_exit_two(
    repo: str,
):
    _close_session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize"])

    assert result.exit_code == 2
    cmd = last_command_record(repo)
    assert cmd.name == "finalize"
    assert cmd.exit_code == 2
    assert cmd.reason == "finalized"


def test_finalize_command_when_no_session_does_exit_two_with_a_start_hint(repo: str):
    result = runner.invoke(app, ["finalize"])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr


# ---------------------------------------------------------------------------
# the loop commands, run from a subdirectory of the repository
# ---------------------------------------------------------------------------


def test_start_command_when_run_from_subdirectory_does_resolve_config_at_repo_root(
    create_scratch_repo: Callable[[], str],
    monkeypatch: pytest.MonkeyPatch,
):
    root = create_scratch_repo()
    nested = Path(root) / "packages" / "core"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    recorder = _ResolverRecorder(resolved_config())
    monkeypatch.setattr("gymrat.cli.session_cmds.resolve_config", recorder)

    runner.invoke(app, ["start", "--baseline", "main"])

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
    from tests.cli._help import help_output

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
    from collections.abc import Awaitable

    lock_names: list[str] = []
    original_with_repo_lock = session_cmds.with_repo_lock

    async def recording_lock[T](
        command: str,
        body: Callable[..., Awaitable[T]],
        *,
        args: dict[str, object] | None = None,
    ) -> T:
        lock_names.append(command)
        return await original_with_repo_lock(command, body, args=args)

    monkeypatch.setattr(session_cmds, "with_repo_lock", recording_lock)

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "sync" in lock_names


def test_sync_command_when_finalized_does_record_command_trace_with_exit_two(
    repo: str,
):
    _close_session_with_one_keep(repo)

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 2
    cmd = last_command_record(repo)
    assert cmd.name == "sync"
    assert cmd.exit_code == 2
    assert cmd.reason == "finalized"


def test_sync_command_when_no_session_does_exit_two_with_a_start_hint(
    repo: str,
):
    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr


# ---------------------------------------------------------------------------
# budget time-left line — text output
# ---------------------------------------------------------------------------


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
    Path(repo, ".gymrat").mkdir(exist_ok=True)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main"])

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


# ---------------------------------------------------------------------------
# the stop command
# ---------------------------------------------------------------------------


@pytest.fixture
def stop_repo(repo: str) -> str:
    """A repository with a settled, configured session ready for the stop command."""
    return make_stop_repo(repo)


def test_stop_command_when_message_given_does_print_stopped_and_append_a_stop_record(
    stop_repo: str,
):
    result = runner.invoke(app, ["stop", "-m", "switched to a different approach"])

    assert result.exit_code == 0
    text = strip_ansi(result.stdout)
    assert "Stopped" in text
    assert "switched to a different approach" in text
    records = read_records(session_jsonl_path(stop_repo))
    stop_records = [r for r in records if isinstance(r, StopRecord)]
    assert len(stop_records) == 1
    assert stop_records[0].message == "switched to a different approach"


def test_stop_command_when_message_flag_does_accept_both_forms(stop_repo: str):
    result = runner.invoke(app, ["stop", "--message", "done"])

    assert result.exit_code == 0


def test_stop_command_when_no_message_does_exit_two_naming_the_option(
    stop_repo: str,
):
    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 2
    assert "message" in (result.stderr + result.stdout).lower()


def test_stop_command_when_blank_message_does_exit_two_naming_the_option(
    stop_repo: str,
):
    result = runner.invoke(app, ["stop", "-m", "   "])

    assert result.exit_code == 2
    assert "message" in (result.stderr + result.stdout).lower()


def test_stop_command_when_finalized_does_record_command_trace_with_exit_two(
    repo: str,
):
    _close_session_with_one_keep(repo)
    write_config(repo)

    result = runner.invoke(app, ["stop", "-m", "done"])

    assert result.exit_code == 2
    cmd = last_command_record(repo)
    assert cmd.name == "stop"
    assert cmd.exit_code == 2
    assert cmd.reason == "finalized"


def test_stop_command_when_no_session_does_exit_two_with_a_start_hint(repo: str):
    write_config(repo)

    result = runner.invoke(app, ["stop", "-m", "done"])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr


def test_stop_command_when_run_does_take_the_repo_lock(
    stop_repo: str, monkeypatch: pytest.MonkeyPatch
):
    from collections.abc import Awaitable

    lock_names: list[str] = []
    original_with_repo_lock = session_cmds.with_repo_lock

    async def recording_lock[T](
        command: str,
        body: Callable[..., Awaitable[T]],
        *,
        args: dict[str, object] | None = None,
    ) -> T:
        lock_names.append(command)
        return await original_with_repo_lock(command, body, args=args)

    monkeypatch.setattr(session_cmds, "with_repo_lock", recording_lock)

    result = runner.invoke(app, ["stop", "-m", "done"])

    assert result.exit_code == 0
    assert "stop" in lock_names


# ---------------------------------------------------------------------------
# budget time-left line — stop text output
# ---------------------------------------------------------------------------


def test_stop_command_when_budget_active_does_end_text_with_time_left_line(
    stop_repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_budget(stop_repo, monkeypatch)

    result = runner.invoke(app, ["stop", "-m", "done"])

    assert result.exit_code == 0
    text = strip_ansi(result.stdout)
    assert re.search(r"left of 30m", text)


def test_stop_command_when_no_budget_does_omit_time_left_line(
    stop_repo: str,
):
    result = runner.invoke(app, ["stop", "-m", "done"])

    assert result.exit_code == 0
    assert "left of" not in result.stdout


# ---------------------------------------------------------------------------
# --color / --no-color on session commands
# ---------------------------------------------------------------------------


def test_start_command_when_no_color_does_strip_ansi_from_stderr_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    result = runner.invoke(app, ["start", "--no-color", "--baseline", "banana"])

    assert result.exit_code == 2
    assert "No such option" not in result.stderr
    assert not SGR_RE.search(result.stderr)


def test_stop_command_when_no_color_does_strip_ansi_from_stderr_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    write_config(repo)

    result = runner.invoke(app, ["stop", "--no-color", "-m", "done"])

    assert result.exit_code == 2
    assert "No such option" not in result.stderr
    assert not SGR_RE.search(result.stderr)


def test_finalize_command_when_no_color_does_strip_ansi_from_stderr_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    result = runner.invoke(app, ["finalize", "--no-color"])

    assert result.exit_code == 2
    assert "No such option" not in result.stderr
    assert not SGR_RE.search(result.stderr)


def test_sync_command_when_no_color_does_strip_ansi_from_stderr_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    result = runner.invoke(app, ["sync", "--no-color"])

    assert result.exit_code == 2
    assert "No such option" not in result.stderr
    assert not SGR_RE.search(result.stderr)
