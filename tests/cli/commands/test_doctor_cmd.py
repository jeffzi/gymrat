"""Tests for the ``gymrat doctor`` command wiring.

These drive the assembled app through :class:`typer.testing.CliRunner` with the
report builder replaced by a one-check report; both renderers run for real. They
cover the exit-code contract and the JSON path. Color precedence is pinned on
``status`` in ``test_app`` and in ``tests/cli/test_console.py``.
"""

import json
from unittest.mock import create_autospec

import pytest

from gymrat.cli.app import app
from gymrat.doctor import Check, build_doctor_report
from tests._doctor_fixtures import single_check_report
from tests.cli._runner import (
    runner,
)


def _patch_doctor(monkeypatch: pytest.MonkeyPatch, *, bench_fail: bool = False) -> None:
    """Replace the report builder with a report holding one bench check."""
    bench_check = (
        Check("bench", "fail", "bench crashed")
        if bench_fail
        else Check("bench", "ok", "1 metric found")
    )
    monkeypatch.setattr(
        "gymrat.cli.commands.doctor.build_doctor_report",
        create_autospec(
            build_doctor_report, return_value=single_check_report(bench_check, title="Bench")
        ),
    )


# ---------------------------------------------------------------------------
# exit-code contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bench_fail", "exit_code"),
    [
        pytest.param(False, 0, id="no-failures-exit-zero"),
        pytest.param(True, 1, id="failures-exit-one"),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_doctor_when_report_written_does_exit_on_its_failures(
    bench_fail: bool, exit_code: int, monkeypatch: pytest.MonkeyPatch
):
    _patch_doctor(monkeypatch, bench_fail=bench_fail)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == exit_code


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def test_doctor_when_format_json_does_write_the_json_document_once(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_doctor(monkeypatch)

    result = runner.invoke(app, ["doctor", "--format", "json"])

    assert result.exit_code == 0
    assert isinstance(json.loads(result.stdout), dict)
