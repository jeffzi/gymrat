"""Command-level tests for start, finalize, stop, sync, and subdirectory resolution.

Each command is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.

The shared guards — a finalized session, no session, ``--no-color`` on a stderr
error, the budget time-left line, and the tight-budget duration warning — are
pinned once each, in one table across every command they apply to.
"""

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.loop.start import start_session
from gymrat.report.types import ComparisonResult
from gymrat.session.paths import experiment_worktree_dir
from gymrat.session.records import FinalizeRecord, IterationRecord, StopRecord
from tests._ansi import SGR_RE, strip_ansi
from tests._config import resolved_config
from tests._git import head_of
from tests.cli._budget import install_budget, install_tight_budget, set_origin
from tests.cli._session import (
    FailingStdoutRunner,
    close_session_with_one_keep,
    closed_stdout_error,
    last_command_record,
    open_session_with_one_keep,
    runner,
    stub_measure,
    stub_resolve_config,
    write_bench_config,
)
from tests.loop._probe import BASELINE_SAMPLES, install_measure, measurement
from tests.loop._settle import (
    CHECKS,
    keep_iteration,
    settling_record_of,
    start_with,
)
from tests.report._comparisons import create_comparison_result
from tests.session.records._fixtures import (
    baseline_record,
    iteration_record,
    records_of_type,
    session_header_of,
    session_record,
    write_session_log,
)

# ---------------------------------------------------------------------------
# the start command
# ---------------------------------------------------------------------------


def test_start_command_when_run_does_create_a_session_and_report_its_branch(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0
    assert session_header_of(repo).branch in result.stdout


def test_start_command_when_reopening_after_finalize_does_name_the_archived_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    closed_id = close_session_with_one_keep(repo)
    stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0
    assert re.search(r"archived", result.stdout, re.IGNORECASE)
    assert closed_id in result.stdout


_RUNBOOK = ".claude/skills/ecstatic-bench/SKILL.md"


@pytest.mark.parametrize(
    ("runbook", "resumed", "rows"),
    [
        pytest.param(
            _RUNBOOK,
            False,
            [f"runbook: {_RUNBOOK} — read it before your first edit"],
            id="configured-fresh",
        ),
        pytest.param(
            _RUNBOOK,
            True,
            [f"runbook: {_RUNBOOK} — read it before your first edit"],
            id="configured-resumed",
        ),
        pytest.param(None, False, [], id="absent-omits-the-row"),
    ],
)
def test_start_command_when_run_does_show_a_runbook_row_only_when_configured(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    runbook: str | None,
    resumed: bool,
    rows: list[str],
):
    if resumed:
        start_session(repo, "main", resolved_config())
    stub_resolve_config(monkeypatch, runbook=runbook)

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0
    assert [
        line.strip() for line in strip_ansi(result.stdout).splitlines() if "runbook" in line
    ] == rows


@pytest.mark.parametrize(
    ("overrides", "args"),
    [
        pytest.param([], [("baseline", "main")], id="baseline-only"),
        pytest.param(
            ["--bench", "sh run.sh", "--samples", "5"],
            [("bench", "sh run.sh"), ("samples", 5), ("baseline", "main")],
            id="config-overrides",
        ),
    ],
)
def test_start_command_when_run_does_record_command_trace_with_its_args_and_exit_zero(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    overrides: list[str],
    args: list[tuple[str, object]],
):
    stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main", *overrides])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.exit_code, cmd.reason) == ("start", 0, None)
    assert list(cmd.args.items()) == args


@pytest.mark.parametrize("resumed", [False, True])
def test_start_command_when_run_does_include_edit_here_line_with_sync_hint(
    repo: str, monkeypatch: pytest.MonkeyPatch, resumed: bool
):
    if resumed:
        start_session(repo, "main", resolved_config())
    stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0
    exp_dir = experiment_worktree_dir(repo)
    assert f"edit in {exp_dir}" in result.stdout
    assert "gymrat sync" in result.stdout


def test_start_command_when_no_baseline_does_default_to_head(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start"])

    assert result.exit_code == 0
    assert session_header_of(repo).baseline.sha == head_of(repo)


# ---------------------------------------------------------------------------
# the finalize command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "named"),
    [
        pytest.param(
            ["--branch", "perf/regex-cache"], "perf/regex-cache", id="branch-the-caller-named"
        ),
        pytest.param([], None, id="session-branch-final-by-default"),
    ],
)
def test_finalize_command_when_run_does_finalize_onto_the_final_branch(
    repo: str, args: list[str], named: str | None
):
    branch = open_session_with_one_keep(repo).branch
    final_branch = named if named is not None else f"{branch}-final"

    result = runner.invoke(app, ["finalize", *args])

    assert result.exit_code == 0
    record = settling_record_of(repo)
    assert isinstance(record, FinalizeRecord)
    assert record.branch == final_branch
    assert final_branch in result.stdout


def test_finalize_command_when_message_given_does_commit_with_it(repo: str):
    open_session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "-m", "squash the tuning session"])

    assert result.exit_code == 0
    record = settling_record_of(repo)
    assert isinstance(record, FinalizeRecord)
    assert record.message == "squash the tuning session"


def test_finalize_command_when_run_does_record_command_trace_with_branch_and_message(
    repo: str,
):
    open_session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "--branch", "perf/regex", "-m", "squash the session"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.name == "finalize"
    assert cmd.args == {"branch": "perf/regex", "message": "squash the session"}
    assert cmd.exit_code == 0
    assert cmd.reason is None


# ---------------------------------------------------------------------------
# the loop commands, run from a subdirectory of the repository
# ---------------------------------------------------------------------------


def test_start_command_when_run_from_subdirectory_does_open_the_session_with_the_root_config(
    create_scratch_repo: Callable[[], str],
    monkeypatch: pytest.MonkeyPatch,
):
    root = create_scratch_repo()
    write_bench_config(root, bench="sh root-bench.sh")
    nested = Path(root) / "packages" / "core"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0, result.stderr
    assert session_header_of(root).config.bench == "sh root-bench.sh"


# ---------------------------------------------------------------------------
# the sync command
# ---------------------------------------------------------------------------


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


def test_sync_command_when_experiment_has_uncommitted_change_to_a_synced_path_does_exit_two(
    sync_repo: str,
):
    Path(sync_repo, "README.md").write_text("# from main\n", encoding="utf-8")
    experiment_copy = Path(experiment_worktree_dir(sync_repo), "README.md")
    experiment_copy.write_text("# from the agent\n", encoding="utf-8")

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 2
    assert "README.md" in result.stderr
    assert last_command_record(sync_repo).reason == "dirty-worktree"
    assert experiment_copy.read_text(encoding="utf-8") == "# from the agent\n"


# ---------------------------------------------------------------------------
# the stop command
# ---------------------------------------------------------------------------


def test_stop_command_when_message_given_does_stop_the_session_with_it(
    stop_repo: str,
):
    result = runner.invoke(app, ["stop", "--message", "switched to a different approach"])

    assert result.exit_code == 0
    text = strip_ansi(result.stdout)
    assert "Stopped" in text
    assert "switched to a different approach" in text
    stop_records = records_of_type(stop_repo, StopRecord)
    assert len(stop_records) == 1
    assert stop_records[0].message == "switched to a different approach"


@pytest.fixture
def kept_repo(repo: str) -> str:
    """A configured repository whose open session has one kept commit, ready for any session command."""
    open_session_with_one_keep(repo)
    write_bench_config(repo)
    return repo


@pytest.mark.usefixtures("kept_repo")
@pytest.mark.parametrize(
    "command",
    [
        pytest.param(["start", "--format", "text"], id="start-text"),
        pytest.param(["start", "--format", "json"], id="start-json"),
        pytest.param(["stop", "-m", "done"], id="stop"),
        pytest.param(["finalize", "--format", "text"], id="finalize-text"),
        pytest.param(["finalize", "--format", "json"], id="finalize-json"),
        pytest.param(["sync", "--format", "text"], id="sync-text"),
        pytest.param(["sync", "--format", "json"], id="sync-json"),
    ],
)
def test_session_command_when_stdout_reader_closed_does_exit_zero_without_stderr(
    command: list[str],
):
    result = FailingStdoutRunner(closed_stdout_error()).invoke(app, command)

    assert (result.exit_code, result.stderr) == (0, "")


@pytest.mark.parametrize(
    "message",
    [
        pytest.param([], id="no-message"),
        pytest.param(["-m", "   "], id="blank-message"),
    ],
)
def test_stop_command_when_message_missing_or_blank_does_exit_two_naming_the_option(
    stop_repo: str, message: list[str]
):
    result = runner.invoke(app, ["stop", *message])

    assert result.exit_code == 2
    assert "message" in (result.stderr + result.stdout).lower()


# ---------------------------------------------------------------------------
# guards every session command shares
# ---------------------------------------------------------------------------


def _configure(repo: str) -> None:
    """Write a config with a bench and checks, so only the session decides the outcome."""
    write_bench_config(repo, checks=CHECKS)


def _write_no_config(repo: str) -> None:
    """Write no config at all."""


@pytest.mark.parametrize(
    ("argv", "name"),
    [
        pytest.param(["finalize"], "finalize", id="finalize"),
        pytest.param(["sync"], "sync", id="sync"),
        pytest.param(["stop", "-m", "done"], "stop", id="stop"),
        pytest.param(["keep"], "keep", id="keep"),
        pytest.param(["discard"], "discard", id="discard"),
        pytest.param(["iterate", "--bench", "npm run bench"], "iterate", id="iterate"),
        pytest.param(
            ["measure", "main", "--bench", "sh bench.sh", "--record"], "measure", id="measure"
        ),
    ],
)
def test_session_command_when_finalized_does_refuse_as_finalized(
    repo: str, argv: list[str], name: str
):
    close_session_with_one_keep(repo)
    _configure(repo)

    result = runner.invoke(app, argv)

    assert result.exit_code == 2
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.exit_code, cmd.reason) == (name, 2, "finalized")


@pytest.mark.parametrize(
    ("argv", "arrange"),
    [
        pytest.param(["finalize"], _write_no_config, id="finalize"),
        pytest.param(["sync"], _write_no_config, id="sync"),
        pytest.param(["stop", "-m", "done"], _configure, id="stop"),
        pytest.param(["status"], _configure, id="status-configured"),
        pytest.param(["status"], _write_no_config, id="status-without-config"),
        pytest.param(["keep"], _configure, id="keep"),
        pytest.param(["discard"], _configure, id="discard"),
        pytest.param(["iterate", "--bench", "npm run bench"], _write_no_config, id="iterate"),
    ],
)
def test_session_command_when_no_session_does_exit_two_with_a_start_hint(
    repo: str, argv: list[str], arrange: Callable[[str], None]
):
    arrange(repo)

    result = runner.invoke(app, argv)

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["start", "--no-color", "--baseline", "banana"], id="start"),
        pytest.param(["stop", "--no-color", "-m", "done"], id="stop"),
        pytest.param(["finalize", "--no-color"], id="finalize"),
        pytest.param(["sync", "--no-color"], id="sync"),
        pytest.param(["discard", "--no-color"], id="discard"),
        pytest.param(["export", "--no-color", "missing/session.jsonl"], id="export"),
    ],
)
def test_session_command_when_no_color_does_strip_ansi_from_stderr_error(
    repo: str, monkeypatch: pytest.MonkeyPatch, argv: list[str]
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    write_bench_config(repo)

    result = runner.invoke(app, argv)

    assert result.exit_code == 2
    assert result.stderr.startswith("Error: ")
    assert not SGR_RE.search(result.stderr)


def _open_for_start(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepare a fresh start: config resolution stubbed and the state directory present."""
    stub_resolve_config(monkeypatch)
    Path(repo, ".gymrat").mkdir(exist_ok=True)


def _open_for_sync(repo: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session for sync."""
    start_session(repo, "main", resolved_config())


def _open_for_finalize(repo: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session with one kept commit for finalize."""
    open_session_with_one_keep(repo)


def _open_for_stop(repo: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a configured, settled session for stop."""
    start_with(repo)
    keep_iteration(repo, 1)
    write_bench_config(repo)


def _stub_compare(_repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the comparison engine with one returning a comparison with no regressions."""

    async def fake_compare(_options: object) -> ComparisonResult:
        return create_comparison_result()

    monkeypatch.setattr("gymrat.compare.compare", fake_compare)


def _stub_measure(_repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the measurement engine and config resolution with canned stand-ins."""
    stub_measure(monkeypatch)


def _open_for_probe(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session with a baseline and stub the engine, run as the supervised tool."""
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    write_bench_config(repo)
    install_measure(monkeypatch, measurement(adapter="metric-lines"))
    set_origin(monkeypatch, "tool")


_COMPARE_ARGV = ["compare", "main", "cand", "--bench", "sh bench.sh"]
_MEASURE_ARGV = ["measure", "--bench", "sh bench.sh"]


@pytest.mark.parametrize(
    ("argv", "arrange"),
    [
        pytest.param(["start", "--baseline", "main"], _open_for_start, id="start"),
        pytest.param(["sync"], _open_for_sync, id="sync"),
        pytest.param(["finalize"], _open_for_finalize, id="finalize"),
        pytest.param(["stop", "-m", "done"], _open_for_stop, id="stop"),
        pytest.param(_COMPARE_ARGV, _stub_compare, id="compare"),
        pytest.param(_MEASURE_ARGV, _stub_measure, id="measure"),
        pytest.param(["probe"], _open_for_probe, id="probe"),
    ],
)
def test_session_command_when_budget_active_does_end_text_with_time_left_line(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    arrange: Callable[[str, pytest.MonkeyPatch], None],
):
    arrange(repo, monkeypatch)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, argv)

    assert result.exit_code == 0
    assert re.search(r"left of 30m\n$", strip_ansi(result.stdout))


@pytest.mark.parametrize(
    ("argv", "arrange"),
    [
        pytest.param(_COMPARE_ARGV, _stub_compare, id="compare"),
        pytest.param(_MEASURE_ARGV, _stub_measure, id="measure"),
    ],
)
@pytest.mark.parametrize(
    ("records", "warns"),
    [
        pytest.param((iteration_record(duration_ms=720_000),), True, id="estimate-known-warns"),
        pytest.param((), False, id="estimate-unknown-stays-silent"),
    ],
)
def test_session_command_when_budget_tight_does_warn_on_stderr_only_with_a_known_estimate(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    arrange: Callable[[str, pytest.MonkeyPatch], None],
    records: tuple[IterationRecord, ...],
    warns: bool,
):
    arrange(repo, monkeypatch)
    install_tight_budget(repo, monkeypatch)
    write_session_log(repo, session_record(), records)

    result = runner.invoke(app, argv)

    assert result.exit_code == 0
    assert ("warning" in result.stderr.lower()) is warns
