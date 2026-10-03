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
from collections.abc import Callable
from typing import Literal

import pytest

from gymrat.cli import budget_report
from gymrat.cli.run_setup import SharedFlags
from gymrat.errors import GymratError
from gymrat.git import NotAGitRepositoryError
from gymrat.report.json_doc import BudgetSummary
from gymrat.report.types import ReportOptions
from gymrat.session.budget import Budget
from gymrat.session.records import IterationRecord
from tests.session.records._fixtures import iteration_record

#: The last full measurement took 48 minutes, so 24 minutes per side.
ITERATE_MS = 2_880_000.0

#: 12 minutes left: too little for one side, so ``measure`` warns.
MEASURE_REMAINING_MS = 720_000.0

#: 30 minutes left: enough for one side but not for the pair, so ``compare`` warns.
COMPARE_REMAINING_MS = 1_800_000.0


def _raise(error: Exception) -> Callable[..., object]:
    """A stand-in for a patched lookup that always fails with *error*."""

    def raiser(*_args: object, **_kwargs: object) -> object:
        raise error

    return raiser


def _install_over_budget_session(monkeypatch: pytest.MonkeyPatch, *, remaining_ms: float) -> None:
    """Patch the budget report lookups onto a live budget plus one timed iteration record."""
    records = [iteration_record(duration_ms=ITERATE_MS)]
    budget = Budget(started_at_ms=0.0, max_minutes=60, deadline_ms=remaining_ms)

    def repo_root(_cwd: str | None = None) -> str:
        return "/repo"

    def read_budget(_root: str, **_kwargs: object) -> Budget:
        return budget

    def read_records(_jsonl_path: str) -> list[IterationRecord]:
        return records

    monkeypatch.setattr("gymrat.clock.now_ms", lambda: 0.0)
    monkeypatch.setattr("gymrat.cli.budget_report.repo_root", repo_root)
    monkeypatch.setattr("gymrat.cli.budget_report.read_budget", read_budget)
    monkeypatch.setattr("gymrat.cli.budget_report.read_records", read_records)


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
        SharedFlags(format=output_format),
        ReportOptions(color=False),
        text=_render_text,
        json=_render_result_json,
    )


@pytest.mark.parametrize(
    ("output_format", "expected"),
    [
        pytest.param("text", "report\n", id="text"),
        pytest.param("json", '{"result": "report", "budget": null}\n', id="json"),
    ],
)
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(NotAGitRepositoryError("not a git repository"), id="not-a-git-repository"),
        pytest.param(GymratError("detected dubious ownership"), id="gymrat-error"),
        pytest.param(OSError("input/output error"), id="os-error"),
    ],
)
def test_emit_report_when_repo_root_fails_expectedly_does_write_the_report_without_a_budget(
    error: Exception,
    output_format: Literal["text", "json"],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr("gymrat.cli.budget_report.repo_root", _raise(error))

    _emit(output_format)

    assert capsys.readouterr().out == expected


def test_emit_report_when_repo_root_fails_unexpectedly_does_propagate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("gymrat.cli.budget_report.repo_root", _raise(RuntimeError("patched wrong")))

    with pytest.raises(RuntimeError, match="patched wrong"):
        _emit("text")


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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    budget = Budget(started_at_ms=0.0, max_minutes=60, deadline_ms=720_999.0)

    def read_budget(_root: str, **_kwargs: object) -> Budget:
        return budget

    monkeypatch.setattr("gymrat.clock.now_ms", lambda: 0.0)
    monkeypatch.setattr("gymrat.cli.budget_report.repo_root", lambda: "/repo")
    monkeypatch.setattr("gymrat.cli.budget_report.read_budget", read_budget)

    _emit(output_format)

    assert capsys.readouterr().out == expected


# ---------------------------------------------------------------------------
# warn_duration_over_budget: exception handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lookup", ["repo_root", "read_records"])
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(GymratError("session log is corrupt"), id="gymrat-error"),
        pytest.param(OSError("input/output error"), id="os-error"),
    ],
)
def test_warn_duration_over_budget_when_a_lookup_fails_expectedly_does_stay_silent(
    lookup: str,
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _install_over_budget_session(monkeypatch, remaining_ms=MEASURE_REMAINING_MS)
    monkeypatch.setattr(f"gymrat.cli.budget_report.{lookup}", _raise(error))

    budget_report.warn_duration_over_budget(halve=True)

    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("lookup", ["repo_root", "read_records"])
def test_warn_duration_over_budget_when_a_lookup_fails_unexpectedly_does_propagate(
    lookup: str, monkeypatch: pytest.MonkeyPatch
):
    _install_over_budget_session(monkeypatch, remaining_ms=MEASURE_REMAINING_MS)
    monkeypatch.setattr(f"gymrat.cli.budget_report.{lookup}", _raise(RuntimeError("patched wrong")))

    with pytest.raises(RuntimeError, match="patched wrong"):
        budget_report.warn_duration_over_budget(halve=True)


def _no_budget(_root: str, **_kwargs: object) -> None:
    """A budget lookup for a session with no active budget."""


def _no_records(_jsonl_path: str) -> list[IterationRecord]:
    """A session log lookup with no timed iteration to estimate from."""
    return []


@pytest.mark.parametrize(
    ("remaining_ms", "lookup", "stub"),
    [
        pytest.param(ITERATE_MS, None, None, id="fits-the-budget"),
        pytest.param(MEASURE_REMAINING_MS, "read_budget", _no_budget, id="no-budget"),
        pytest.param(MEASURE_REMAINING_MS, "read_records", _no_records, id="no-estimate"),
    ],
)
def test_warn_duration_over_budget_when_nothing_to_warn_about_does_stay_silent(
    remaining_ms: float,
    lookup: str | None,
    stub: Callable[..., object] | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _install_over_budget_session(monkeypatch, remaining_ms=remaining_ms)
    if lookup is not None:
        monkeypatch.setattr(f"gymrat.cli.budget_report.{lookup}", stub)

    budget_report.warn_duration_over_budget(halve=False)

    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# over-budget warning wording
# ---------------------------------------------------------------------------


def test_warn_duration_over_budget_when_halving_does_name_the_per_side_cost(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _install_over_budget_session(monkeypatch, remaining_ms=MEASURE_REMAINING_MS)

    budget_report.warn_duration_over_budget(halve=True)

    assert capsys.readouterr().err == (
        "warning: 12m 0s left; the last full measurement took at most 24m 0s per side\n"
    )


def test_warn_duration_over_budget_when_not_halving_does_name_the_full_cost_with_the_per_side_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _install_over_budget_session(monkeypatch, remaining_ms=COMPARE_REMAINING_MS)

    budget_report.warn_duration_over_budget(halve=False)

    assert capsys.readouterr().err == (
        "warning: 30m 0s left; the last full measurement took at most 48m 0s (24m 0s per side)\n"
    )


# ---------------------------------------------------------------------------
# write_budget_report
# ---------------------------------------------------------------------------


def _budget_active(root: str) -> tuple[str, BudgetSummary]:
    """Stub returning an active budget snapshot."""
    return "\n⏱ 29m left of 30m", BudgetSummary(cap_minutes=30, remaining_seconds=1740)


def _budget_inactive(root: str) -> tuple[str, None]:
    """Stub returning no budget."""
    return "", None


def _render_json(summary: BudgetSummary | None) -> str:
    """Render a JSON document that carries a ``budget`` object only when *summary* is set."""
    doc: dict[str, object] = {"metric": "ops/s"}
    if summary is not None:
        doc["budget"] = {
            "cap_minutes": summary.cap_minutes,
            "remaining_seconds": summary.remaining_seconds,
        }
    return json.dumps(doc)


def test_write_budget_report_when_json_and_budget_active_does_write_json_with_budget(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(budget_report, "budget_snapshot", _budget_active)

    budget_report.write_budget_report(
        "/fake/root",
        use_json=True,
        render_json=_render_json,
        text_report="ignored text",
    )

    out = json.loads(capsys.readouterr().out)
    assert out["budget"] == {"cap_minutes": 30, "remaining_seconds": 1740}


def test_write_budget_report_when_json_and_no_budget_does_write_json_without_budget(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(budget_report, "budget_snapshot", _budget_inactive)

    budget_report.write_budget_report(
        "/fake/root",
        use_json=True,
        render_json=_render_json,
        text_report="ignored text",
    )

    out = json.loads(capsys.readouterr().out)
    assert "budget" not in out


def _noop_json(_s: BudgetSummary | None) -> str:
    """A no-op JSON renderer for text-mode tests."""
    return ""


def test_write_budget_report_when_text_and_budget_active_does_write_report_with_trailer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(budget_report, "budget_snapshot", _budget_active)

    budget_report.write_budget_report(
        "/fake/root",
        use_json=False,
        render_json=_noop_json,
        text_report="benchmark results here",
    )

    out = capsys.readouterr().out
    assert out == "benchmark results here\n⏱ 29m left of 30m\n"


def test_write_budget_report_when_text_and_no_budget_does_write_report_without_trailer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(budget_report, "budget_snapshot", _budget_inactive)

    budget_report.write_budget_report(
        "/fake/root",
        use_json=False,
        render_json=_noop_json,
        text_report="benchmark results here",
    )

    out = capsys.readouterr().out
    assert out == "benchmark results here\n"
