"""The stand-ins the ``gymrat doctor`` command tests install over the command's seams."""

from unittest.mock import create_autospec

import pytest

from gymrat.doctor import (
    Check,
    CheckSection,
    build_doctor_report,
    render_doctor_json,
    render_doctor_report,
)
from tests._doctor_fixtures import doctor_report


def patch_doctor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bench_fail: bool = False,
    report_error: Exception | None = None,
    real_report: bool = False,
    stub_text: bool = True,
    stub_json: bool = True,
) -> None:
    """Replace the report builder and the renderers the doctor command calls.

    Args:
        monkeypatch: The fixture that installs the stand-ins.
        bench_fail: Whether the canned report's bench check failed.
        report_error: An error the report builder raises instead of returning.
        real_report: ``True`` leaves the real report builder in place.
        stub_text: ``False`` leaves the real text renderer in place.
        stub_json: ``False`` leaves the real JSON renderer in place.
    """
    if not real_report:
        bench_check = (
            Check("bench", "fail", "bench crashed")
            if bench_fail
            else Check("bench", "ok", "1 metric found")
        )
        report = doctor_report([
            CheckSection(title="Environment", checks=[Check("git", "ok", "available")]),
            CheckSection(
                title="Configuration", checks=[Check("config", "ok", "/project/gymrat.json")]
            ),
            CheckSection(title="Workflow", checks=[Check("skill file", "ok", "found")]),
            CheckSection(title="Bench", checks=[bench_check]),
        ])
        monkeypatch.setattr(
            "gymrat.cli.commands.doctor.build_doctor_report",
            create_autospec(build_doctor_report, return_value=report, side_effect=report_error),
        )
    if stub_text:
        monkeypatch.setattr(
            "gymrat.cli.commands.doctor.render_doctor_report",
            create_autospec(render_doctor_report, return_value="doctor text report"),
        )
    if stub_json:
        monkeypatch.setattr(
            "gymrat.cli.commands.doctor.render_doctor_json",
            create_autospec(render_doctor_json, return_value='{"doctor": true}'),
        )
