"""Command-level tests for start, finalize, stop, and sync.

Each command is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.

The shared guards — a finalized session, no session, ``--no-color`` on a stderr
error, a closed stdout, the budget time-left line, and the tight-budget duration
warning — are pinned once each here: in one table across every command they
apply to, or, where the commands share one code path, once per distinct path
(``write_budget_report``, ``emit_report``, and ``status``'s own trailer for the
time-left line). The closed-stdout table also covers ``doctor`` and ``init``.
"""

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.loop.start import start_session
from gymrat.session.paths import experiment_worktree_dir
from gymrat.session.records import FinalizeRecord, IterationRecord, StopRecord
from tests._ansi import SGR_RE, strip_ansi
from tests._config import resolved_config
from tests._git import head_of
from tests.cli._budget import install_budget, install_tight_budget, set_origin
from tests.cli._doctor_seams import patch_doctor
from tests.cli._session import (
    FailingStdoutRunner,
    close_session_with_one_keep,
    closed_stdout_error,
    last_command_record,
    open_session,
    open_session_with_one_keep,
    open_stop_ready_session,
    runner,
    stub_compare,
    stub_config,
    stub_measure,
    write_bench_config,
    write_settled_session,
)
from tests.loop._probe import BASELINE_SAMPLES, install_measure, measurement
from tests.loop._settle import (
    CHECKS,
    settling_record_of,
    start_with,
)
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


def test_start_command_when_reopening_after_finalize_does_name_the_archived_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    closed_id = close_session_with_one_keep(repo)
    stub_config(monkeypatch, "session", resolved_config())

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
    stub_config(monkeypatch, "session", resolved_config(runbook=runbook))

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0
    assert [
        line.strip() for line in strip_ansi(result.stdout).splitlines() if "runbook" in line
    ] == rows


def test_start_command_when_config_overrides_given_does_record_them_in_its_trace(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    stub_config(monkeypatch, "session", resolved_config())

    result = runner.invoke(
        app, ["start", "--baseline", "main", "--bench", "sh run.sh", "--samples", "5"]
    )

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.exit_code, cmd.reason) == ("start", 0, None)
    assert list(cmd.args.items()) == [("bench", "sh run.sh"), ("samples", 5), ("baseline", "main")]


@pytest.mark.parametrize("resumed", [False, True])
def test_start_command_when_run_does_print_the_session_summary_with_a_sync_hint(
    repo: str, monkeypatch: pytest.MonkeyPatch, resumed: bool
):
    if resumed:
        start_session(repo, "main", resolved_config())
    stub_config(monkeypatch, "session", resolved_config())

    result = runner.invoke(app, ["start", "--baseline", "main"])

    assert result.exit_code == 0
    assert session_header_of(repo).branch in result.stdout
    exp_dir = experiment_worktree_dir(repo)
    assert f"edit in {exp_dir}" in result.stdout
    assert "gymrat sync" in result.stdout
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.args, cmd.exit_code, cmd.reason) == (
        "start",
        {"baseline": "main"},
        0,
        None,
    )


def test_start_command_when_no_baseline_does_default_to_head(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    stub_config(monkeypatch, "session", resolved_config())

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


def test_finalize_command_when_branch_and_message_given_does_carry_them_into_its_records(
    repo: str,
):
    open_session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "--branch", "perf/regex", "-m", "squash the session"])

    assert result.exit_code == 0
    record = settling_record_of(repo)
    assert isinstance(record, FinalizeRecord)
    assert (record.branch, record.message) == ("perf/regex", "squash the session")
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.args, cmd.exit_code, cmd.reason) == (
        "finalize",
        {"branch": "perf/regex", "message": "squash the session"},
        0,
        None,
    )


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


def test_session_command_when_no_color_does_strip_ansi_from_stderr_error(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    write_bench_config(repo)

    result = runner.invoke(app, ["start", "--no-color", "--baseline", "banana"])

    assert result.exit_code == 2
    assert result.stderr.startswith("Error: ")
    assert not SGR_RE.search(result.stderr)


def _settled_session(repo: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """Log a configured session with one kept iteration, for status."""
    write_settled_session(repo)


def _open_for_stop(repo: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a configured, settled session for stop."""
    open_stop_ready_session(repo)


def _stub_compare(_repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the comparison engine with one returning a comparison with no regressions."""
    stub_compare(monkeypatch)


def _stub_measure(_repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the measurement engine and config resolution with canned stand-ins."""
    stub_measure(monkeypatch)


def _open_for_probe(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session with a baseline and stub the engine, run as the supervised tool."""
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    write_bench_config(repo)
    install_measure(monkeypatch, measurement(adapter="metric-lines"))
    set_origin(monkeypatch, "tool")


def _no_budget(_repo: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the session without a budget."""


def _open_and_stub_measure(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session and replace the measurement engine and config resolution."""
    open_session(repo)
    stub_measure(monkeypatch)


def _stub_doctor(_repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every doctor seam with a canned stand-in."""
    patch_doctor(monkeypatch)


def _nothing_to_arrange(_repo: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the repository as the fixture made it."""


_MEASURE_MAIN = ["measure", "main", "--bench", "sh bench.sh"]


@pytest.mark.parametrize(
    ("argv", "arrange"),
    [
        pytest.param(["stop", "-m", "done"], _open_for_stop, id="stop"),
        pytest.param(["status"], _settled_session, id="status"),
        pytest.param(
            [*_MEASURE_MAIN, "--format", "text"], _open_and_stub_measure, id="measure-text"
        ),
        pytest.param(
            [*_MEASURE_MAIN, "--format", "json"], _open_and_stub_measure, id="measure-json"
        ),
        pytest.param(
            [*_MEASURE_MAIN, "--record"], _open_and_stub_measure, id="measure-record-note"
        ),
        pytest.param(["doctor"], _stub_doctor, id="doctor"),
        pytest.param(["init", "--bench", "npm run bench"], _nothing_to_arrange, id="init"),
    ],
)
def test_command_when_stdout_reader_closed_does_exit_zero_without_stderr(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    arrange: Callable[[str, pytest.MonkeyPatch], None],
):
    arrange(repo, monkeypatch)

    result = FailingStdoutRunner(closed_stdout_error()).invoke(app, argv)

    assert (result.exit_code, result.stderr) == (0, "")


_COMPARE_ARGV = ["compare", "main", "cand", "--bench", "sh bench.sh"]
_MEASURE_ARGV = ["measure", "--bench", "sh bench.sh"]


@pytest.mark.parametrize(
    ("argv", "arrange", "budget", "trailers"),
    [
        pytest.param(
            ["stop", "-m", "done"], _open_for_stop, install_budget, ["left of 30m"], id="stop"
        ),
        pytest.param(["probe"], _open_for_probe, install_budget, ["left of 30m"], id="probe"),
        pytest.param(["status"], _settled_session, install_budget, ["left of 30m"], id="status"),
        pytest.param(["status"], _settled_session, _no_budget, [], id="status-no-budget"),
    ],
)
def test_session_command_when_run_does_end_text_with_a_time_left_line_only_under_a_budget(
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
    assert re.findall(r"left of \d+m(?=\n\Z)", strip_ansi(result.stdout)) == trailers


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
