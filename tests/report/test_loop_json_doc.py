"""Tests for the JSON documents of the loop commands.

These cover the start, keep, finalize, and sync documents
(``render_start_json``, ``render_keep_json``, ``render_finalize_json``,
``render_sync_json``): their schema shapes, the resumed and archived start
variants, the runbook and budget fields, and the keep record's checks object.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from gymrat.loop.finalize import FinalizeResult
from gymrat.loop.settle import KeepResult
from gymrat.loop.start import StartResult
from gymrat.loop.sync import SyncResult
from gymrat.report.json_doc import (
    BudgetSummary,
    render_finalize_json,
    render_keep_json,
    render_start_json,
    render_sync_json,
)
from gymrat.session import KeepChecks
from tests.session.records._fixtures import (
    committed_keep,
    empty_session_state,
    finalize_record,
    session_record,
    session_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# helpers — start / finalize / sync
# ---------------------------------------------------------------------------


def _fresh_start() -> StartResult:
    """A brand-new session (not resumed, nothing archived)."""
    return StartResult(
        session=session_record(),
        state=empty_session_state(),
        resumed=False,
    )


def _resumed_start(*, iteration_count: int = 3, keep_count: int = 2) -> StartResult:
    """A resumed session with prior iteration and keep counts."""
    return StartResult(
        session=session_record(),
        state=session_state(iteration_count=iteration_count, keep_count=keep_count),
        resumed=True,
    )


def _archived_start() -> StartResult:
    """A session that archived a finalized predecessor."""
    return StartResult(
        session=session_record(),
        state=empty_session_state(),
        resumed=False,
        archived="20260701-120000-beef",
        archived_path="/repo/.gymrat/archive/20260701-120000-beef",
    )


# ---------------------------------------------------------------------------
# render_start_json — fresh, resumed, and runbook
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "runbook", "expected"),
    [
        pytest.param(
            _fresh_start(),
            None,
            {"resumed": False, "iteration_count": 0, "keep_count": 0, "runbook": None},
            id="fresh",
        ),
        pytest.param(
            _resumed_start(iteration_count=5, keep_count=3),
            None,
            {"resumed": True, "iteration_count": 5, "keep_count": 3, "runbook": None},
            id="resumed",
        ),
        pytest.param(
            _fresh_start(),
            "my-runbook.yml",
            {"resumed": False, "iteration_count": 0, "keep_count": 0, "runbook": "my-runbook.yml"},
            id="runbook",
        ),
    ],
)
def test_render_start_json_when_start_varies_does_reflect_state_and_runbook(
    result: StartResult, runbook: str | None, expected: dict[str, object]
):
    doc = json.loads(render_start_json(result, runbook=runbook))

    assert doc["session_id"] == result.session.session_id
    assert doc["branch"] == result.session.branch
    assert doc["baseline"] == {
        "ref": result.session.baseline.ref,
        "sha": result.session.baseline.sha,
    }
    assert doc["worktrees"] == {
        "experiment": result.session.worktrees.experiment,
        "baseline": result.session.worktrees.baseline,
    }
    assert doc["resumed"] == expected["resumed"]
    assert doc["iteration_count"] == expected["iteration_count"]
    assert doc["keep_count"] == expected["keep_count"]
    assert doc["runbook"] == expected["runbook"]


# ---------------------------------------------------------------------------
# render_start_json — archived start
# ---------------------------------------------------------------------------


def test_render_start_json_when_archived_does_include_archived_object():
    result = _archived_start()

    doc = json.loads(render_start_json(result))

    assert doc["archived"] == {
        "session_id": "20260701-120000-beef",
        "path": "/repo/.gymrat/archive/20260701-120000-beef",
    }


# ---------------------------------------------------------------------------
# render_keep_json — checks object
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        pytest.param(
            KeepChecks(configured=True, passed=False, stdout_bytes=120, stderr_bytes=45),
            [("configured", True), ("passed", False), ("stdout_bytes", 120), ("stderr_bytes", 45)],
            id="checks-ran",
        ),
        pytest.param(
            KeepChecks(configured=False),
            [
                ("configured", False),
                ("passed", None),
                ("stdout_bytes", None),
                ("stderr_bytes", None),
            ],
            id="no-checks-configured",
        ),
    ],
)
def test_render_keep_json_when_checks_recorded_does_serialize_every_checks_field_in_order(
    checks: KeepChecks, expected: list[tuple[str, object]]
):
    result = KeepResult(record=committed_keep(1, checks=checks), report="keep report")

    doc = json.loads(render_keep_json(result))

    assert list(doc["checks"].items()) == expected


# ---------------------------------------------------------------------------
# render_finalize_json — schema shape
# ---------------------------------------------------------------------------


def test_render_finalize_json_when_rendered_does_produce_expected_keys():
    record = finalize_record()
    result = FinalizeResult(record=record, report="final report text")

    doc = json.loads(render_finalize_json(result))

    assert doc["branch"] == record.branch
    assert doc["commit"] == record.commit
    assert doc["message"] == record.message
    assert doc["at"] == record.at


# ---------------------------------------------------------------------------
# render_sync_json — schema shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param(
            ("src/main.py", "src/lib.py"), ["src/main.py", "src/lib.py"], id="files-synced"
        ),
        pytest.param((), [], id="no-files"),
    ],
)
def test_render_sync_json_when_files_vary_does_list_them(
    files: tuple[str, ...], expected: list[str]
):
    result = SyncResult(files=files)

    doc = json.loads(render_sync_json(result))

    assert doc["files"] == expected


# ---------------------------------------------------------------------------
# render_*_json — budget key
# ---------------------------------------------------------------------------


_ABSENT = object()


@pytest.mark.parametrize(
    ("render", "expected"),
    [
        pytest.param(
            lambda: render_start_json(
                _fresh_start(), budget=BudgetSummary(cap_minutes=30, remaining_seconds=900)
            ),
            {"cap_minutes": 30, "remaining_seconds": 900},
            id="budget-given",
        ),
        pytest.param(
            lambda: render_finalize_json(
                FinalizeResult(record=finalize_record(), report="report"), budget=None
            ),
            _ABSENT,
            id="no-budget",
        ),
    ],
)
def test_render_json_when_budget_varies_does_reflect_the_budget_key(
    render: Callable[[], str], expected: object
):
    doc = json.loads(render())

    assert doc.get("budget", _ABSENT) == expected
