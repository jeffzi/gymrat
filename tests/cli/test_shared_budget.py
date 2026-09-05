"""Tests for the budget helpers in the CLI shared module.

``budget_for_report`` and ``warn_duration_over_budget`` sit between the report
writers and the session store. They read the live budget and the session log,
swallow the expected failures — no git repository, a corrupt session log, any
other ``OSError`` — and let anything else propagate so a programming error is
never mistaken for a missing budget.

``warn_duration_over_budget`` also owns the wording of the over-budget warning,
which differs between ``measure`` (half an iterate, so per-side) and ``compare``
(a whole iterate, so the full cost with the per-side figure in parentheses).
"""

from collections.abc import Callable

import pytest

from gymrat.cli import shared
from gymrat.errors import GymratError
from gymrat.git import NotAGitRepositoryError
from gymrat.session import IterationRecord
from gymrat.session.budget import Budget
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
    """Patch the shared lookups onto a live budget plus one timed iteration record."""
    records = [iteration_record(duration_ms=ITERATE_MS)]
    budget = Budget(started_at_ms=0.0, max_minutes=60, deadline_ms=remaining_ms)

    def repo_root(_cwd: str | None = None) -> str:
        return "/repo"

    def read_budget(_root: str, **_kwargs: object) -> Budget:
        return budget

    def read_records(_jsonl_path: str) -> list[IterationRecord]:
        return records

    monkeypatch.setattr("gymrat.session.clock.now_ms", lambda: 0.0)
    monkeypatch.setattr("gymrat.cli.shared.repo_root", repo_root)
    monkeypatch.setattr("gymrat.cli.shared.read_budget", read_budget)
    monkeypatch.setattr("gymrat.cli.shared.read_records", read_records)


# ---------------------------------------------------------------------------
# budget_for_report
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(NotAGitRepositoryError("not a git repository"), id="not-a-git-repository"),
        pytest.param(GymratError("detected dubious ownership"), id="gymrat-error"),
        pytest.param(OSError("input/output error"), id="os-error"),
    ],
)
def test_budget_for_report_when_repo_root_fails_expectedly_does_return_an_empty_snapshot(
    error: Exception, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("gymrat.cli.shared.repo_root", _raise(error))

    result = shared.budget_for_report()

    assert result == ("", None)


def test_budget_for_report_when_repo_root_fails_unexpectedly_does_propagate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("gymrat.cli.shared.repo_root", _raise(RuntimeError("patched wrong")))

    with pytest.raises(RuntimeError, match="patched wrong"):
        shared.budget_for_report()


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
    monkeypatch.setattr(f"gymrat.cli.shared.{lookup}", _raise(error))

    shared.warn_duration_over_budget(halve=True)

    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("lookup", ["repo_root", "read_records"])
def test_warn_duration_over_budget_when_a_lookup_fails_unexpectedly_does_propagate(
    lookup: str, monkeypatch: pytest.MonkeyPatch
):
    _install_over_budget_session(monkeypatch, remaining_ms=MEASURE_REMAINING_MS)
    monkeypatch.setattr(f"gymrat.cli.shared.{lookup}", _raise(RuntimeError("patched wrong")))

    with pytest.raises(RuntimeError, match="patched wrong"):
        shared.warn_duration_over_budget(halve=True)


# ---------------------------------------------------------------------------
# over-budget warning wording
# ---------------------------------------------------------------------------


def test_warn_duration_over_budget_when_halving_does_name_the_per_side_cost(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _install_over_budget_session(monkeypatch, remaining_ms=MEASURE_REMAINING_MS)

    shared.warn_duration_over_budget(halve=True)

    assert capsys.readouterr().err == (
        "warning: 12m 0s left; the last full measurement took at most 24m 0s per side\n"
    )


def test_warn_duration_over_budget_when_not_halving_does_name_the_full_cost_with_the_per_side_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _install_over_budget_session(monkeypatch, remaining_ms=COMPARE_REMAINING_MS)

    shared.warn_duration_over_budget(halve=False)

    assert capsys.readouterr().err == (
        "warning: 30m 0s left; the last full measurement took at most 48m 0s (24m 0s per side)\n"
    )
