"""The stand-ins the ``gymrat doctor`` command tests install over every doctor seam."""

from unittest.mock import create_autospec

import pytest

from gymrat.doctor import (
    Check,
    CheckSection,
    build_environment_section,
    render_doctor_json,
    render_doctor_report,
)
from tests._doctor_fixtures import patch_common_seams


def patch_doctor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bench_fail: bool = False,
    env_error: Exception | None = None,
    stub_text: bool = True,
    stub_json: bool = True,
) -> None:
    """Replace every doctor seam.

    Args:
        monkeypatch: The fixture that installs the stand-ins.
        bench_fail: Whether the bench section reports a failed check.
        env_error: An error the environment section raises instead of returning.
        stub_text: ``False`` leaves the real text renderer in place.
        stub_json: ``False`` leaves the real JSON renderer in place.
    """
    patch_common_seams(monkeypatch, config_failure=False, bench_fail=bench_fail, problems=[])

    env_section = CheckSection(title="Environment", checks=[Check("git", "ok", "available")])
    monkeypatch.setattr(
        "gymrat.doctor.build_environment_section",
        create_autospec(build_environment_section, return_value=env_section, side_effect=env_error),
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
