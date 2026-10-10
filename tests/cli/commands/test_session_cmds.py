"""Command-level tests for start, finalize, stop, and sync, text and ``--format json``.

Each command is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.

The shared guards — a finalized session, no session, ``--no-color`` on a stderr
error, a closed stdout, the budget time-left line, the tight-budget duration
warning, and the refusal of a shell-typed command during a live supervised run — are pinned once each here: in one table across every command they
apply to, or, where the commands share one code path, once per distinct path
(``write_budget_report``, ``emit_report``, and ``status``'s own trailer for the
time-left line). The closed-stdout table also covers ``doctor``, ``init``,
``supervise``, and ``--version``.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.loop.start import start_session
from gymrat.session.paths import experiment_worktree_dir, progress_path, session_jsonl_path
from gymrat.session.records import CommandRecord, FinalizeRecord, IterationRecord, StopRecord
from tests._ansi import SGR_RE, strip_ansi, stripped_lines
from tests._cli import err_text
from tests._config import resolved_config
from tests._git import head_of
from tests.cli._command_stubs import (
    force_render_mode,
    stub_compare,
    stub_config,
    stub_measure,
    stub_probe_measure,
)
from tests.cli._doctor_seams import patch_doctor
from tests.cli._origin import SUPERVISED_HINT, set_origin
from tests.cli._runner import (
    FailingStdoutRunner,
    closed_stdout_error,
    runner,
)
from tests.cli._session import (
    close_session_with_one_keep,
    leave_as_is,
    open_capped_session_under_budget,
    open_hooked_iterate_session,
    open_outlasted_session,
    open_probe_session,
    open_session,
    open_session_with_one_keep,
    open_stop_ready_session,
    open_stubbed_probe_session,
    open_unsettled_session_under_budget,
    write_bench_config,
    write_settled_session,
)
from tests.cli.commands.supervise._seams import install_seams
from tests.loop._probe import MeasureRecorder
from tests.loop._settle import (
    CHECKS,
    settling_record_of,
)
from tests.loop.iterate._fixtures import CollectSamplesRecorder, install_improved_samples
from tests.session._budget import install_budget, install_tight_budget
from tests.session.records._fixtures import (
    append_records,
    committed_keep,
    iteration_record,
    last_command_record,
    records_of_type,
    session_header_of,
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
def test_start_command_when_run_does_print_the_session_summary_with_a_runbook_row_if_configured(
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
    assert session_header_of(repo).branch in result.stdout
    assert f"edit in {experiment_worktree_dir(repo)}" in result.stdout
    assert "gymrat sync" in result.stdout
    assert [
        line.strip() for line in strip_ansi(result.stdout).splitlines() if "runbook" in line
    ] == rows


@pytest.mark.parametrize(
    ("overrides", "traced_args"),
    [
        pytest.param(
            ["--bench", "sh run.sh", "--samples", "5"],
            [("bench", "sh run.sh"), ("samples", 5), ("baseline", "main")],
            id="config-overrides",
        ),
        pytest.param([], [("baseline", "main")], id="no-overrides"),
    ],
)
def test_start_command_when_run_does_trace_only_the_options_given(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    overrides: list[str],
    traced_args: list[tuple[str, object]],
):
    stub_config(monkeypatch, "session", resolved_config())

    result = runner.invoke(app, ["start", "--baseline", "main", *overrides])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.exit_code, cmd.reason) == ("start", 0, None)
    assert list(cmd.args.items()) == traced_args


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


def test_finalize_command_when_run_does_finalize_onto_the_session_final_branch(repo: str):
    final_branch = f"{open_session_with_one_keep(repo).branch}-final"

    result = runner.invoke(app, ["finalize"])

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
    assert "perf/regex" in result.stdout
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
    session_repo: str,
):
    root = Path(session_repo)
    (root / "extra.py").write_text("# new\n", encoding="utf-8")
    (root / "README.md").write_text("# updated\n", encoding="utf-8")

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "2 files" in result.stdout
    assert "extra.py" in result.stdout
    assert "README.md" in result.stdout


def test_sync_command_when_nothing_to_sync_does_print_nothing_to_sync(
    session_repo: str,
):
    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "nothing to sync" in result.stdout


def test_sync_command_when_experiment_has_uncommitted_change_to_a_synced_path_does_exit_two(
    session_repo: str,
):
    Path(session_repo, "README.md").write_text("# from main\n", encoding="utf-8")
    experiment_copy = Path(experiment_worktree_dir(session_repo), "README.md")
    experiment_copy.write_text("# from the agent\n", encoding="utf-8")

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 2
    assert "README.md" in result.stderr
    assert last_command_record(session_repo).reason == "dirty-worktree"
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
    assert "message" in err_text(result).lower()


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
        pytest.param(["probe"], "probe", id="probe"),
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
        pytest.param(["probe"], _configure, id="probe"),
    ],
)
def test_session_command_when_no_session_does_exit_two_with_a_start_hint(
    repo: str, argv: list[str], arrange: Callable[[str], None]
):
    arrange(repo)

    result = runner.invoke(app, argv)

    assert (result.exit_code, result.stdout) == (2, "")
    assert "gymrat start" in result.stderr
    assert not Path(session_jsonl_path(repo)).exists()


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


def _stub_measure(_repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the measurement engine and config resolution with canned stand-ins."""
    stub_measure(monkeypatch)


def _open_for_probe(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session with a baseline and stub the engine, run as the supervised tool."""
    open_stubbed_probe_session(repo, monkeypatch)
    set_origin(monkeypatch, "tool")


def _open_and_stub_measure(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session and replace the measurement engine and config resolution."""
    open_session(repo)
    stub_measure(monkeypatch)


def _open_and_stub_compare(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session and replace the comparison engine."""
    open_session(repo)
    stub_compare(monkeypatch)


def _stub_doctor(_repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every doctor seam with a canned stand-in."""
    patch_doctor(monkeypatch)


def _supervise_live(_repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every supervise seam and force the live dashboard."""
    install_seams(monkeypatch)
    force_render_mode(monkeypatch, "supervise", "live")


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
        pytest.param(["init", "--bench", "npm run bench"], leave_as_is, id="init"),
        pytest.param(
            ["supervise", "optimize it", "--max-minutes", "10"], _supervise_live, id="supervise"
        ),
        pytest.param(["--version"], leave_as_is, id="version"),
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


def _probe_under_tight_budget(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe-ready session under a tight live budget its last iteration outlasts.

    A probe that got past the guard would warn that the halved estimate outlasts
    the budget, so the refusal test also proves the guard runs before that warning.
    """
    open_probe_session(repo)
    install_tight_budget(repo, monkeypatch)
    append_records(repo, iteration_record(duration_ms=720_000))


def _probe_engine(_repo: str, monkeypatch: pytest.MonkeyPatch) -> MeasureRecorder:
    """The stubbed measurement engine a probe would bench through."""
    return stub_probe_measure(monkeypatch)


def _iterate_engine(repo: str, monkeypatch: pytest.MonkeyPatch) -> CollectSamplesRecorder:
    """The stubbed sampling an iterate would bench through."""
    return install_improved_samples(monkeypatch, repo)


def _hooked_iterate_session(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An open session whose before hook can run, under a live budget.

    A hook that ran would log a hook record, so the refusal test also proves the
    guard stops the command before its before hook.
    """
    open_hooked_iterate_session(repo, monkeypatch)


@pytest.mark.parametrize(
    ("command", "arrange", "engine"),
    [
        pytest.param("probe", _probe_under_tight_budget, _probe_engine, id="probe"),
        pytest.param("iterate", _hooked_iterate_session, _iterate_engine, id="iterate"),
        pytest.param(
            "iterate",
            open_unsettled_session_under_budget,
            _iterate_engine,
            id="iterate-unsettled",
        ),
        pytest.param(
            "iterate",
            open_capped_session_under_budget,
            _iterate_engine,
            id="iterate-stop-condition",
        ),
        pytest.param(
            "iterate", open_outlasted_session, _iterate_engine, id="iterate-budget-exceeded"
        ),
    ],
)
def test_session_command_when_supervised_run_live_does_refuse_as_supervised_use_tool(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    arrange: Callable[[str, pytest.MonkeyPatch], None],
    engine: Callable[[str, pytest.MonkeyPatch], MeasureRecorder | CollectSamplesRecorder],
):
    arrange(repo, monkeypatch)
    bench = engine(repo, monkeypatch)
    before = records_of_type(repo, CommandRecord, matching=False)

    result = runner.invoke(app, [command])

    assert (result.exit_code, result.stdout) == (2, "")
    stderr = " ".join(stripped_lines(result.stderr, keep_blank=False))
    assert f"a supervised run is live; use the {command} tool" in stderr
    assert SUPERVISED_HINT in stderr
    assert "warning" not in stderr.lower()
    assert bench.calls == []
    assert records_of_type(repo, CommandRecord, matching=False) == before
    assert not Path(progress_path(repo)).exists()
    commands = records_of_type(repo, CommandRecord)
    assert [(cmd.name, cmd.exit_code, cmd.reason, cmd.origin, cmd.seq) for cmd in commands] == [
        (command, 2, "supervised-use-tool", "cli", None)
    ]


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
        pytest.param(["status"], _settled_session, leave_as_is, [], id="status-no-budget"),
        pytest.param(_MEASURE_ARGV, _stub_measure, leave_as_is, [], id="measure-no-budget"),
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


#: An iteration of 12 minutes, which outlasts the 5 minutes a tight budget leaves.
_OUTLASTING = (iteration_record(duration_ms=720_000),)


@pytest.mark.parametrize(
    ("argv", "arrange", "records", "warns"),
    [
        pytest.param(
            _COMPARE_ARGV, _open_and_stub_compare, _OUTLASTING, True, id="compare-estimate-warns"
        ),
        pytest.param(_COMPARE_ARGV, _open_and_stub_compare, (), False, id="compare-no-estimate"),
        pytest.param(
            _MEASURE_ARGV, _open_and_stub_measure, _OUTLASTING, True, id="measure-estimate-warns"
        ),
        pytest.param(_MEASURE_ARGV, _open_and_stub_measure, (), False, id="measure-no-estimate"),
        pytest.param(["probe"], _open_for_probe, _OUTLASTING, True, id="probe-half-outlasts"),
        pytest.param(
            ["probe"],
            _open_for_probe,
            (iteration_record(duration_ms=400_000),),
            False,
            id="probe-half-still-fits",
        ),
    ],
)
def test_session_command_when_budget_tight_does_warn_on_stderr_only_when_the_estimate_outlasts_it(
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
    append_records(repo, *records)

    result = runner.invoke(app, argv)

    assert result.exit_code == 0
    assert ("warning" in result.stderr.lower()) is warns


# ---------------------------------------------------------------------------
# stop --format json
# ---------------------------------------------------------------------------


def test_stop_command_when_format_json_does_emit_structured_json_with_at_and_message(
    stop_repo: str,
):
    result = runner.invoke(app, ["stop", "-m", "user requested stop", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["message"] == "user requested stop"
    assert "at" in doc
    assert isinstance(doc["at"], int)


# ---------------------------------------------------------------------------
# start --format json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runbook",
    [
        pytest.param(None, id="no-runbook"),
        pytest.param(".claude/skills/ecstatic-bench/SKILL.md", id="runbook"),
    ],
)
def test_start_command_when_format_json_and_fresh_does_emit_structured_json(
    repo: str, monkeypatch: pytest.MonkeyPatch, runbook: str | None
):
    stub_config(monkeypatch, "session", resolved_config(runbook=runbook))

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "session_id" in doc
    assert doc["branch"].startswith("gymrat/")
    assert doc["baseline"]["ref"] == "main"
    assert isinstance(doc["baseline"]["sha"], str)
    assert isinstance(doc["worktrees"], dict)
    assert doc["resumed"] is False
    assert doc["iteration_count"] == 0
    assert doc["keep_count"] == 0
    assert doc["runbook"] == runbook
    assert doc["archived"] is None


def test_start_command_when_format_json_and_resumed_does_set_resumed_true_with_counts(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_session(repo, "main", resolved_config())
    append_records(repo, iteration_record(seq=1))
    append_records(repo, committed_keep(1))
    stub_config(monkeypatch, "session", resolved_config())

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["resumed"] is True
    assert doc["iteration_count"] == 1
    assert doc["keep_count"] == 1


def test_start_command_when_format_json_and_archived_does_include_archived_session_id(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    closed_id = close_session_with_one_keep(repo)
    stub_config(monkeypatch, "session", resolved_config())

    result = runner.invoke(app, ["start", "--baseline", "main", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["archived"]["session_id"] == closed_id
    assert isinstance(doc["archived"]["path"], str)
    assert doc["resumed"] is False


# ---------------------------------------------------------------------------
# finalize --format json
# ---------------------------------------------------------------------------


def test_finalize_command_when_format_json_does_emit_structured_json(
    repo: str,
):
    open_session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "-m", "squash the tuning session", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["branch"].endswith("-final")
    assert isinstance(doc["commit"], str)
    assert len(doc["commit"]) == 40
    assert doc["message"] == "squash the tuning session"
    assert isinstance(doc["at"], int)


# ---------------------------------------------------------------------------
# sync --format json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "files",
    [
        pytest.param(["extra.py"], id="files-synced"),
        pytest.param([], id="nothing-to-sync"),
    ],
)
def test_sync_command_when_format_json_does_emit_the_synced_files(
    session_repo: str, files: list[str]
):
    for name in files:
        Path(session_repo, name).write_text("# new\n", encoding="utf-8")

    result = runner.invoke(app, ["sync", "--format", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["files"] == files
