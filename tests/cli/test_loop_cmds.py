"""Command-level tests for the loop subcommands: start, iterate, discard, finalize, status.

Each command is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
The seams mocked here mirror the engine suites' boundaries: sampling for
``iterate`` (``gymrat_py.loop.iterate.collect_samples``), and for ``discard``'s
prompt the ``is_tty`` and ``confirm_action`` helpers as the ``loop_cmds`` module
imports them. Config resolution is exercised for real where a test lays down a
``gymrat.toml`` and stubbed at the ``loop_cmds`` seam where a test needs to pin
what a command reads (the runbook row) or observe where it looked (the
subdirectory case).
"""

import re
from collections.abc import Callable
from pathlib import Path

import pytest
import tomli_w
from typer.testing import CliRunner

from gymrat_py.cli.app import app
from gymrat_py.loop.finalize import finalize_session
from gymrat_py.loop.start import start_session
from gymrat_py.session import (
    FinalizeRecord,
    SessionRecord,
    append_record,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from tests.loop._iterate import (
    baseline_rounds,
    improved_rounds,
    install_collect_samples,
    resolved_config,
    stub_samples,
)
from tests.loop._iterate import session_record as iterate_session_header
from tests.loop._settle import git, head_of, last_record_of
from tests.session._records import (
    SESSION_ID,
    committed_keep,
    iteration_record,
    session_record,
    write_session_log,
)

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

#: The runbook path a session's config points an agent at, when it has one.
_RUNBOOK_PATH = ".claude/skills/ecstatic-bench/SKILL.md"


def _plain_lines(text: str) -> list[str]:
    """The non-blank lines of ``text``, stripped of color and surrounding space."""
    return [_ANSI_RE.sub("", line).strip() for line in text.split("\n") if line.strip()]


def _always_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its interactive path."""
    return True


def _never_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its non-interactive path."""
    return False


def _write_config(root: str, **extra: object) -> None:
    """Write the implicit ``gymrat.toml`` at the repository root."""
    payload: dict[str, object] = {"bench": "npm run bench", **extra}
    (Path(root) / "gymrat.toml").write_text(tomli_w.dumps(payload), encoding="utf-8")


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


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch) -> str:
    """A fresh scratch repository, chdir'd into so the command runs there."""
    root = create_scratch_repo()
    monkeypatch.chdir(root)
    return root


# ---------------------------------------------------------------------------
# the start command
# ---------------------------------------------------------------------------


def _stub_resolve_config(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> object:
    """Pin what ``start`` reads by replacing its ``resolve_config`` with a fixed config."""
    config = resolved_config(**overrides)

    def fake(*_a: object, **_k: object) -> object:
        return config

    monkeypatch.setattr("gymrat_py.cli.loop_cmds.resolve_config", fake)
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


# ---------------------------------------------------------------------------
# the iterate command
# ---------------------------------------------------------------------------


def test_iterate_command_when_run_does_measure_the_repo_and_report_on_stdout(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    mock = install_collect_samples(monkeypatch)
    stub_samples(mock, repo, improved_rounds(), baseline_rounds())

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 0
    lines = _plain_lines(result.stdout)
    assert lines[0] == "iteration 1 · experiment vs baseline · 10 paired samples"
    assert lines[-1] == "Hint: gymrat keep"
    assert len(read_records(session_jsonl_path(repo))) == 2


def test_iterate_command_when_stop_condition_met_does_exit_one_without_measuring(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(
        repo, iterate_session_header(repo), (iteration_record(seq=1), committed_keep(1))
    )
    mock = install_collect_samples(monkeypatch)
    _write_config(repo, stop={"max_iterations": 1})

    result = runner.invoke(app, ["iterate"])

    assert result.exit_code == 1
    assert "max iterations" in result.stderr
    assert mock.call_count == 0


def test_iterate_command_when_no_session_does_exit_two_with_a_start_hint(repo: str):
    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr


# ---------------------------------------------------------------------------
# the status command
# ---------------------------------------------------------------------------


def test_status_command_when_run_does_render_the_session_on_stdout(repo: str):
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    _write_config(repo)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    text = _ANSI_RE.sub("", result.stdout)
    assert f"session {SESSION_ID}" in text
    assert "1 kept" in text


@pytest.mark.parametrize("has_config", [True, False])
def test_status_command_when_no_session_does_exit_two_with_a_start_hint(
    repo: str, has_config: bool
):
    if has_config:
        _write_config(repo)

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
    _write_config(repo)

    result = runner.invoke(app, ["status", *args])

    assert result.exit_code == 0
    assert bool(_ANSI_RE.search(result.stdout)) is expect_ansi


# ---------------------------------------------------------------------------
# the discard command
# ---------------------------------------------------------------------------


@pytest.fixture
def discard_repo(repo: str) -> str:
    """A repository with an open session and one unsettled iteration to discard."""
    start_session(repo, "main", resolved_config())
    append_record(session_jsonl_path(repo), iteration_record(seq=1))
    return repo


def test_discard_command_documents_force_in_its_help():
    result = runner.invoke(app, ["discard", "--help"])

    assert result.exit_code == 0
    assert "--force" in result.output


def test_discard_command_when_tty_and_confirmed_does_prompt_and_proceed(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("gymrat_py.cli.loop_cmds.is_tty", _always_tty)
    confirm = _ConfirmRecorder(answer=True)
    monkeypatch.setattr("gymrat_py.cli.loop_cmds.confirm_action", confirm)

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 0
    assert len(confirm.calls) == 1
    assert experiment_worktree_dir(discard_repo) in confirm.calls[0][0]
    assert re.search(r"discard", result.stdout, re.IGNORECASE)


def test_discard_command_when_tty_and_declined_does_cancel_with_exit_one(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("gymrat_py.cli.loop_cmds.is_tty", _always_tty)
    monkeypatch.setattr("gymrat_py.cli.loop_cmds.confirm_action", _ConfirmRecorder(answer=False))

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 1
    assert "discard cancelled" in result.stderr


@pytest.mark.parametrize("flag", ["--force", "-f"])
def test_discard_command_when_force_does_skip_the_prompt(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch, flag: str
):
    monkeypatch.setattr("gymrat_py.cli.loop_cmds.is_tty", _always_tty)
    confirm = _ConfirmRecorder(answer=True)
    monkeypatch.setattr("gymrat_py.cli.loop_cmds.confirm_action", confirm)

    result = runner.invoke(app, ["discard", flag])

    assert result.exit_code == 0
    assert confirm.calls == []
    assert re.search(r"discard", result.stdout, re.IGNORECASE)


def test_discard_command_when_stdin_not_tty_does_skip_the_prompt(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("gymrat_py.cli.loop_cmds.is_tty", _never_tty)
    confirm = _ConfirmRecorder(answer=True)
    monkeypatch.setattr("gymrat_py.cli.loop_cmds.confirm_action", confirm)

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
    result = runner.invoke(app, ["finalize", "--help"])

    assert result.exit_code == 0
    out = result.output
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
    monkeypatch.setattr(f"gymrat_py.cli.loop_cmds.{resolver_name}", recorder)

    # Whether the command then finds the session it needs is beside the point;
    # where it looked the configuration up is what is under test.
    runner.invoke(app, args)

    assert recorder.calls
    base_dir = recorder.calls[0][1]
    assert base_dir is not None
    assert Path(base_dir) == Path(root)
