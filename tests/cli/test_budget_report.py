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
from unittest.mock import create_autospec

import pytest

from gymrat.cli import budget_report
from gymrat.cli.console import apply_command_flags
from gymrat.cli.run_setup import SharedFlags
from gymrat.errors import GymratError
from gymrat.git import NotAGitRepositoryError
from gymrat.report.json_doc import BudgetSummary
from gymrat.report.types import DEFAULT_REPORT_OPTIONS, ReportOptions
from gymrat.session.budget import Budget, read_budget
from gymrat.session.paths import repo_root
from gymrat.session.records import IterationRecord
from gymrat.session.store import read_records
from tests.session.records._fixtures import iteration_record

#: The last full measurement took 48 minutes, so 24 minutes per side.
ITERATE_MS = 2_880_000.0

#: 12 minutes left: too little for one side, so ``measure`` warns.
MEASURE_REMAINING_MS = 720_000.0

#: 30 minutes left: enough for one side but not for the pair, so ``compare`` warns.
COMPARE_REMAINING_MS = 1_800_000.0


def _install_over_budget_session(monkeypatch: pytest.MonkeyPatch, *, remaining_ms: float) -> None:
    """Patch the budget report lookups onto a live budget plus one timed iteration record."""
    records = [iteration_record(duration_ms=ITERATE_MS)]
    budget = Budget(max_minutes=60, deadline_ms=remaining_ms)

    def fake_repo_root(_cwd: str | None = None) -> str:
        return "/repo"

    def fake_read_budget(_root: str, **_kwargs: object) -> Budget:
        return budget

    def fake_read_records(_jsonl_path: str) -> list[IterationRecord]:
        return records

    monkeypatch.setattr("gymrat.clock.now_ms", lambda: 0.0)
    monkeypatch.setattr("gymrat.cli.budget_report.repo_root", fake_repo_root)
    monkeypatch.setattr("gymrat.cli.budget_report.read_budget", fake_read_budget)
    monkeypatch.setattr("gymrat.cli.budget_report.read_records", fake_read_records)


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
    ("color", "env", "expected"),
    [
        pytest.param(True, "NO_COLOR", True, id="color-flag-outranks-no-color-env"),
        pytest.param(False, "FORCE_COLOR", False, id="no-color-flag-outranks-force-color-env"),
    ],
)
def test_emit_report_when_command_color_flag_installed_does_hand_it_to_the_text_renderer(
    color: bool, env: str, expected: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(env, "1")
    not_a_repo = create_autospec(
        repo_root, side_effect=NotAGitRepositoryError("not a git repository")
    )
    monkeypatch.setattr("gymrat.cli.budget_report.repo_root", not_a_repo)
    apply_command_flags(debug=False, color=color)
    rendered_with: list[bool | None] = []

    def render(result: str, options: ReportOptions) -> str:
        rendered_with.append(options.color)
        return result

    budget_report.emit_report(
        "report", SharedFlags(), DEFAULT_REPORT_OPTIONS, text=render, json=_render_result_json
    )

    assert rendered_with == [expected]


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
    monkeypatch.setattr(
        "gymrat.cli.budget_report.repo_root", create_autospec(repo_root, side_effect=error)
    )

    _emit(output_format)

    assert capsys.readouterr().out == expected


def test_emit_report_when_repo_root_fails_unexpectedly_does_propagate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "gymrat.cli.budget_report.repo_root",
        create_autospec(repo_root, side_effect=RuntimeError("patched wrong")),
    )

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
    budget = Budget(max_minutes=60, deadline_ms=720_999.0)

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

#: The session lookups ``warn_duration_over_budget`` reads, by their name in the module.
_LOOKUPS: dict[str, Callable[..., object]] = {"repo_root": repo_root, "read_records": read_records}


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
    monkeypatch.setattr(
        f"gymrat.cli.budget_report.{lookup}", create_autospec(_LOOKUPS[lookup], side_effect=error)
    )

    budget_report.warn_duration_over_budget(halve=True)

    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("lookup", ["repo_root", "read_records"])
def test_warn_duration_over_budget_when_a_lookup_fails_unexpectedly_does_propagate(
    lookup: str, monkeypatch: pytest.MonkeyPatch
):
    _install_over_budget_session(monkeypatch, remaining_ms=MEASURE_REMAINING_MS)
    monkeypatch.setattr(
        f"gymrat.cli.budget_report.{lookup}",
        create_autospec(_LOOKUPS[lookup], side_effect=RuntimeError("patched wrong")),
    )

    with pytest.raises(RuntimeError, match="patched wrong"):
        budget_report.warn_duration_over_budget(halve=True)


@pytest.mark.parametrize(
    ("remaining_ms", "lookup", "stub"),
    [
        pytest.param(ITERATE_MS, None, None, id="fits-the-budget"),
        pytest.param(
            MEASURE_REMAINING_MS,
            "read_budget",
            create_autospec(read_budget, return_value=None),
            id="no-budget",
        ),
        pytest.param(
            MEASURE_REMAINING_MS,
            "read_records",
            create_autospec(read_records, return_value=[]),
            id="no-estimate",
        ),
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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _install_over_budget_session(monkeypatch, remaining_ms=remaining_ms)

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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr("gymrat.clock.now_ms", lambda: 0.0)
    monkeypatch.setattr(
        "gymrat.cli.budget_report.read_budget", create_autospec(read_budget, return_value=budget)
    )

    budget_report.write_budget_report(
        "/fake/root",
        use_json=use_json,
        render_json=_render_json,
        text_report="benchmark results here",
    )

    assert capsys.readouterr().out == expected
