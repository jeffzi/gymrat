"""Tests for the CLI budget report helpers.

``write_budget_report`` appends the live budget to a command's report: a
trailer line under the text report, a ``budget`` object in the JSON document.

``emit_report`` and ``warn_duration_over_budget`` sit between the report
writers and the session store. They read the live budget and the session log,
swallow the expected failures — no git repository, a corrupt session log, any
other ``OSError`` — and let anything else propagate so a programming error is
never mistaken for a missing budget.

``warn_duration_over_budget`` also owns the wording of the over-budget warning,
which differs between ``measure`` (half an iterate, so per-side) and ``compare``
(a whole iterate, so the full cost with the per-side figure in parentheses).
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

import pytest

from gymrat.cli import budget_report
from gymrat.cli.options import OutputFormat
from gymrat.cli.run_setup import SharedFlags
from gymrat.errors import GymratError
from gymrat.report.json_doc import BudgetSummary
from gymrat.report.types import ReportOptions
from gymrat.session.budget import Budget
from gymrat.session.paths import repo_root, session_jsonl_path, supervise_lockfile_path
from gymrat.session.records import IterationRecord
from gymrat.utils import format_time_left
from tests._clock import freeze_now_ms
from tests._lock import held_supervise_lock, lock_probe_error
from tests.session._budget import LIVE_BUDGET, block_supervise_lock, write_budget_file
from tests.session.records._fixtures import (
    append_raw,
    iteration_record,
    session_record,
    write_session_log,
)

#: The last full measurement took 48 minutes, so 24 minutes per side.
ITERATE_MS = 2_880_000.0

#: 12 minutes left: too little for one side, so ``measure`` warns.
MEASURE_REMAINING_MS = 720_000.0

#: 30 minutes left: enough for one side but not for the pair, so ``compare`` warns.
COMPARE_REMAINING_MS = 1_800_000.0


#: One timed iteration: the estimate every over-budget check measures against.
_TIMED_ITERATION = (iteration_record(duration_ms=ITERATE_MS),)


@pytest.fixture
def session_root(repo: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A repository under a live supervised run, with the clock frozen at 0."""
    freeze_now_ms(monkeypatch, 0)
    with held_supervise_lock(repo):
        yield repo


def _seed_session(
    root: str,
    *,
    budget: Budget | None,
    records: tuple[IterationRecord, ...] = _TIMED_ITERATION,
) -> None:
    """Write a session log holding ``records`` and, when one is given, the budget file.

    Args:
        root: The repository whose session state is written.
        budget: The budget to write, or None to leave no budget file.
        records: The iteration records appended after the session header.
    """
    write_session_log(root, session_record(), records)
    if budget is not None:
        write_budget_file(root, budget)


def _budget_left(remaining_ms: float) -> Budget:
    """A 60-minute budget with ``remaining_ms`` left at the frozen clock's instant."""
    return Budget(max_minutes=60, deadline_ms=remaining_ms)


# ---------------------------------------------------------------------------
# emit_report
# ---------------------------------------------------------------------------


def _render_text(result: str, _options: ReportOptions) -> str:
    """A text renderer that prints the result as it is."""
    return result


def _render_result_json(result: str, /, *, budget: BudgetSummary | None = None) -> str:
    """A JSON renderer that carries the result and the budget summary it was handed."""
    return json.dumps({
        "result": result,
        "budget": None if budget is None else [budget.cap_minutes, budget.remaining_seconds],
    })


def _emit(output_format: Literal["text", "json"]) -> None:
    """Emit the ``report`` result in ``output_format`` through the stub renderers."""
    budget_report.emit_report(
        "report",
        SharedFlags(format=OutputFormat(output_format)),
        ReportOptions(color=False),
        text=_render_text,
        json=_render_result_json,
    )


def _outside_a_repository(request: pytest.FixtureRequest) -> None:
    """Run from the shared non-repository directory, so the root lookup fails."""
    request.getfixturevalue("_in_non_repo")


def _fail_getcwd(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Make reading the process working directory raise ``error``."""

    def getcwd() -> str:
        raise error

    monkeypatch.setattr("os.getcwd", getcwd)


def _repo_root_io_error(request: pytest.FixtureRequest) -> None:
    """Make the repository root lookup fail with an I/O error reading the working directory."""
    _fail_getcwd(request.getfixturevalue("monkeypatch"), OSError("input/output error"))


@pytest.fixture
def failed_root_lookup(request: pytest.FixtureRequest) -> None:
    """Make the repository root lookup fail the way the parametrized arrangement does."""
    request.param(request)


@pytest.mark.parametrize(
    "failed_root_lookup",
    [
        pytest.param(_outside_a_repository, id="not-a-repository"),
        pytest.param(_repo_root_io_error, id="os-error"),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("failed_root_lookup")
def test_emit_report_when_repo_root_fails_expectedly_does_write_the_report_without_a_budget(
    capsys: pytest.CaptureFixture[str],
):
    _emit("text")

    assert capsys.readouterr().out == "report\n"


@pytest.mark.parametrize(
    ("output_format", "expected"),
    [
        pytest.param("text", f"report\n{format_time_left(720_999.0, 60)}\n", id="text-trailer"),
        pytest.param(
            "json", '{"result": "report", "budget": [60.0, 720]}\n', id="json-whole-seconds-left"
        ),
    ],
)
def test_emit_report_when_budget_active_does_write_the_report_with_the_budget(
    output_format: Literal["text", "json"],
    expected: str,
    session_root: str,
    capsys: pytest.CaptureFixture[str],
):
    _seed_session(session_root, budget=Budget(max_minutes=60, deadline_ms=720_999.0))

    _emit(output_format)

    assert capsys.readouterr().out == expected


# ---------------------------------------------------------------------------
# warn_duration_over_budget: exception handling
# ---------------------------------------------------------------------------


def _corrupt_session_log(_request: pytest.FixtureRequest) -> None:
    """Append a line that is not JSON to the session log, so reading it fails."""
    append_raw(session_jsonl_path(repo_root()), b"not json\n")


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_outside_a_repository, id="not-a-repository"),
        pytest.param(_corrupt_session_log, id="corrupt-session-log"),
    ],
)
def test_warn_duration_over_budget_when_a_lookup_fails_expectedly_does_stay_silent(
    arrange: Callable[[pytest.FixtureRequest], None],
    session_root: str,
    request: pytest.FixtureRequest,
    capsys: pytest.CaptureFixture[str],
):
    _seed_session(session_root, budget=_budget_left(MEASURE_REMAINING_MS))
    arrange(request)

    budget_report.warn_duration_over_budget(halve=True)

    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# emit_report and warn_duration_over_budget: unexpected failures propagate
# ---------------------------------------------------------------------------


def _warn() -> None:
    """Run the ``measure`` over-budget check."""
    budget_report.warn_duration_over_budget(halve=True)


def test_emit_report_when_repo_root_fails_unexpectedly_does_propagate(
    session_root: str, monkeypatch: pytest.MonkeyPatch
):
    _seed_session(session_root, budget=_budget_left(MEASURE_REMAINING_MS))
    _fail_getcwd(monkeypatch, RuntimeError("patched wrong"))

    with pytest.raises(RuntimeError, match="patched wrong"):
        _emit("text")


def test_warn_duration_over_budget_when_reading_records_fails_unexpectedly_does_propagate(
    session_root: str, monkeypatch: pytest.MonkeyPatch
):
    _seed_session(session_root, budget=_budget_left(MEASURE_REMAINING_MS))
    log = Path(session_jsonl_path(session_root))
    real_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path == log:
            message = "patched wrong"
            raise RuntimeError(message)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    with pytest.raises(RuntimeError, match="patched wrong"):
        _warn()


def test_emit_report_when_supervise_lock_cannot_be_probed_does_raise_the_lock_error(repo: str):
    _seed_session(repo, budget=LIVE_BUDGET)

    with block_supervise_lock(repo):
        expected = lock_probe_error(supervise_lockfile_path(repo))
        with pytest.raises(GymratError) as exc:
            _emit("text")

    assert str(exc.value) == expected


def test_warn_duration_over_budget_when_supervise_lock_cannot_be_probed_does_raise_the_lock_error(
    repo: str,
):
    _seed_session(repo, budget=LIVE_BUDGET)

    with block_supervise_lock(repo):
        expected = lock_probe_error(supervise_lockfile_path(repo))
        with pytest.raises(GymratError) as exc:
            _warn()

    assert str(exc.value) == expected


# ---------------------------------------------------------------------------
# over-budget warning: when and how it warns
# ---------------------------------------------------------------------------


def test_warn_duration_over_budget_when_estimate_fits_the_budget_does_stay_silent(
    session_root: str, capsys: pytest.CaptureFixture[str]
):
    _seed_session(session_root, budget=_budget_left(ITERATE_MS))

    budget_report.warn_duration_over_budget(halve=False)

    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("halve", "remaining_ms", "shown", "hidden"),
    [
        pytest.param(
            True,
            MEASURE_REMAINING_MS,
            ("12m 0s", "24m 0s per side"),
            ("(",),
            id="halving-names-the-per-side-cost",
        ),
        pytest.param(
            False,
            COMPARE_REMAINING_MS,
            ("30m 0s", "48m 0s (24m 0s per side)"),
            (),
            id="whole-names-the-full-cost-with-the-per-side-one",
        ),
    ],
)
def test_warn_duration_over_budget_when_over_budget_does_name_the_cost_the_command_pays(
    *,
    halve: bool,
    remaining_ms: float,
    shown: tuple[str, ...],
    hidden: tuple[str, ...],
    session_root: str,
    capsys: pytest.CaptureFixture[str],
):
    _seed_session(session_root, budget=_budget_left(remaining_ms))

    budget_report.warn_duration_over_budget(halve=halve)

    err = capsys.readouterr().err
    assert [part for part in shown if part not in err] == []
    assert [part for part in hidden if part in err] == []


# ---------------------------------------------------------------------------
# write_budget_report
# ---------------------------------------------------------------------------


def _render_json(_summary: BudgetSummary | None) -> str:
    """Render a JSON document; the text format never calls it."""
    return json.dumps({"metric": "ops/s"})


def test_write_budget_report_when_no_budget_is_live_does_write_the_report_alone(
    session_root: str, capsys: pytest.CaptureFixture[str]
):
    _seed_session(session_root, budget=None, records=())

    budget_report.write_budget_report(
        session_root,
        SharedFlags(format=OutputFormat.text),
        render_json=_render_json,
        text_report="benchmark results here",
    )

    assert capsys.readouterr().out == "benchmark results here\n"
