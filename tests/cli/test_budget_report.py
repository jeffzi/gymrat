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
from unittest.mock import create_autospec

import pytest

from gymrat.cli import budget_report
from gymrat.cli.console import apply_command_flags
from gymrat.cli.options import OutputFormat
from gymrat.cli.run_setup import SharedFlags
from gymrat.clock import now_ms
from gymrat.report.json_doc import BudgetSummary
from gymrat.report.types import DEFAULT_REPORT_OPTIONS, ReportOptions
from gymrat.session.budget import Budget
from gymrat.session.paths import repo_root, session_jsonl_path
from gymrat.session.records import IterationRecord
from gymrat.session.store import read_records
from tests._lock import held_supervise_lock
from tests.session._budget import write_budget_file
from tests.session.records._fixtures import iteration_record, session_record, write_session_log

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
    monkeypatch.setattr("gymrat.clock.now_ms", create_autospec(now_ms, return_value=0))
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


def test_emit_report_when_command_color_flag_installed_does_hand_it_to_the_text_renderer(
    session_root: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")
    apply_command_flags(debug=False, color=True)
    rendered_with: list[bool | None] = []

    def render(result: str, options: ReportOptions) -> str:
        rendered_with.append(options.color)
        return result

    budget_report.emit_report(
        "report", SharedFlags(), DEFAULT_REPORT_OPTIONS, text=render, json=_render_result_json
    )

    assert rendered_with == [True]


def _outside_a_repository(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run from a directory that is not a git repository, so the root lookup fails."""
    monkeypatch.chdir(tmp_path)


def _repo_root_io_error(monkeypatch: pytest.MonkeyPatch, _tmp_path: Path) -> None:
    """Make the repository root lookup fail with an I/O error."""
    monkeypatch.setattr(
        "gymrat.cli.budget_report.repo_root",
        create_autospec(repo_root, side_effect=OSError("input/output error")),
    )


@pytest.fixture
def failed_root_lookup(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Make the repository root lookup fail the way the parametrized arrangement does."""
    request.param(monkeypatch, tmp_path)


@pytest.mark.parametrize(
    ("failed_root_lookup", "output_format", "expected"),
    [
        pytest.param(_outside_a_repository, "text", "report\n", id="not-a-repository"),
        pytest.param(_repo_root_io_error, "text", "report\n", id="os-error"),
        pytest.param(
            _repo_root_io_error,
            "json",
            '{"result": "report", "budget": null}\n',
            id="json-carries-no-budget",
        ),
    ],
    indirect=["failed_root_lookup"],
)
@pytest.mark.usefixtures("failed_root_lookup")
def test_emit_report_when_repo_root_fails_expectedly_does_write_the_report_without_a_budget(
    output_format: Literal["text", "json"],
    expected: str,
    capsys: pytest.CaptureFixture[str],
):
    _emit(output_format)

    assert capsys.readouterr().out == expected


@pytest.mark.parametrize(
    ("output_format", "expected"),
    [
        pytest.param("text", "report\n12m 0s left of 60m\n", id="text-trailer"),
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

#: The session lookups ``warn_duration_over_budget`` reads, by their name in the module.
_LOOKUPS: dict[str, Callable[..., object]] = {"repo_root": repo_root, "read_records": read_records}


def _corrupt_session_log(_monkeypatch: pytest.MonkeyPatch, _tmp_path: Path) -> None:
    """Append a line that is not JSON to the session log, so reading it fails."""
    with Path(session_jsonl_path(repo_root())).open("a", encoding="utf-8") as log:
        log.write("not json\n")


def _read_records_io_error(monkeypatch: pytest.MonkeyPatch, _tmp_path: Path) -> None:
    """Make reading the session log fail with an I/O error."""
    monkeypatch.setattr(
        "gymrat.cli.budget_report.read_records",
        create_autospec(read_records, side_effect=OSError("input/output error")),
    )


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_outside_a_repository, id="not-a-repository"),
        pytest.param(_corrupt_session_log, id="corrupt-session-log"),
        pytest.param(_read_records_io_error, id="read-records-os-error"),
    ],
)
def test_warn_duration_over_budget_when_a_lookup_fails_expectedly_does_stay_silent(
    arrange: Callable[[pytest.MonkeyPatch, Path], None],
    session_root: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    _seed_session(session_root, budget=_budget_left(MEASURE_REMAINING_MS))
    arrange(monkeypatch, tmp_path)

    budget_report.warn_duration_over_budget(halve=True)

    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# emit_report and warn_duration_over_budget: unexpected failures propagate
# ---------------------------------------------------------------------------


def _warn() -> None:
    """Run the ``measure`` over-budget check."""
    budget_report.warn_duration_over_budget(halve=True)


@pytest.mark.parametrize(
    ("helper", "lookup"),
    [
        pytest.param(lambda: _emit("text"), "repo_root", id="emit-report-repo-root"),
        pytest.param(_warn, "repo_root", id="warn-repo-root"),
        pytest.param(_warn, "read_records", id="warn-read-records"),
    ],
)
def test_budget_helper_when_a_lookup_fails_unexpectedly_does_propagate(
    helper: Callable[[], None], lookup: str, session_root: str, monkeypatch: pytest.MonkeyPatch
):
    _seed_session(session_root, budget=_budget_left(MEASURE_REMAINING_MS))
    monkeypatch.setattr(
        f"gymrat.cli.budget_report.{lookup}",
        create_autospec(_LOOKUPS[lookup], side_effect=RuntimeError("patched wrong")),
    )

    with pytest.raises(RuntimeError, match="patched wrong"):
        helper()


@pytest.mark.parametrize(
    ("budget", "records"),
    [
        pytest.param(_budget_left(ITERATE_MS), _TIMED_ITERATION, id="fits-the-budget"),
        pytest.param(None, _TIMED_ITERATION, id="no-budget"),
        pytest.param(_budget_left(MEASURE_REMAINING_MS), (), id="no-estimate"),
    ],
)
def test_warn_duration_over_budget_when_nothing_to_warn_about_does_stay_silent(
    budget: Budget | None,
    records: tuple[IterationRecord, ...],
    session_root: str,
    capsys: pytest.CaptureFixture[str],
):
    _seed_session(session_root, budget=budget, records=records)

    budget_report.warn_duration_over_budget(halve=False)

    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# over-budget warning wording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("halve", "remaining_ms", "warning"),
    [
        pytest.param(
            True,
            MEASURE_REMAINING_MS,
            "warning: 12m 0s left; the last full measurement took at most 24m 0s per side\n",
            id="halving-names-the-per-side-cost",
        ),
        pytest.param(
            False,
            COMPARE_REMAINING_MS,
            "warning: 30m 0s left; the last full measurement took at most 48m 0s"
            " (24m 0s per side)\n",
            id="whole-names-the-full-cost-with-the-per-side-one",
        ),
    ],
)
def test_warn_duration_over_budget_when_over_budget_does_name_the_cost_the_command_pays(
    halve: bool,
    remaining_ms: float,
    warning: str,
    session_root: str,
    capsys: pytest.CaptureFixture[str],
):
    _seed_session(session_root, budget=_budget_left(remaining_ms))

    budget_report.warn_duration_over_budget(halve=halve)

    assert capsys.readouterr().err == warning


# ---------------------------------------------------------------------------
# write_budget_report
# ---------------------------------------------------------------------------


def _render_json(summary: BudgetSummary | None) -> str:
    """Render a JSON document that carries a ``budget`` object only when *summary* is set."""
    doc: dict[str, object] = {"metric": "ops/s"}
    if summary is not None:
        doc["budget"] = {
            "cap_minutes": summary.cap_minutes,
            "remaining_seconds": summary.remaining_seconds,
        }
    return json.dumps(doc)


#: A live budget 29 minutes from its 30-minute deadline at the frozen clock's instant.
_ACTIVE_BUDGET = Budget(max_minutes=30, deadline_ms=1_740_000.0)


@pytest.mark.parametrize(
    ("budget", "use_json", "expected"),
    [
        pytest.param(
            _ACTIVE_BUDGET,
            True,
            '{"metric": "ops/s", "budget": {"cap_minutes": 30.0, "remaining_seconds": 1740}}\n',
            id="json-budget-active",
        ),
        pytest.param(None, True, '{"metric": "ops/s"}\n', id="json-no-budget"),
        pytest.param(
            _ACTIVE_BUDGET,
            False,
            "benchmark results here\n29m 0s left of 30m\n",
            id="text-budget-active",
        ),
        pytest.param(None, False, "benchmark results here\n", id="text-no-budget"),
    ],
)
def test_write_budget_report_when_budget_active_or_not_does_append_it_only_if_active(
    budget: Budget | None,
    use_json: bool,
    expected: str,
    session_root: str,
    capsys: pytest.CaptureFixture[str],
):
    _seed_session(session_root, budget=budget, records=())

    budget_report.write_budget_report(
        session_root,
        SharedFlags(format=OutputFormat.json if use_json else OutputFormat.text),
        render_json=_render_json,
        text_report="benchmark results here",
    )

    assert capsys.readouterr().out == expected
