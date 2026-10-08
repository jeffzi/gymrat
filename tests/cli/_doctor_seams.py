"""The stand-ins the ``gymrat doctor`` command tests install over every doctor seam."""

import pytest

from gymrat.doctor import Check, CheckSection
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

    def env_section(*_a: object, **_k: object) -> CheckSection:
        if env_error is not None:
            raise env_error
        return CheckSection(title="Environment", checks=[Check("git", "ok", "available")])

    monkeypatch.setattr("gymrat.doctor.build_environment_section", env_section)

    def fake_text(_report: object, **_kwargs: object) -> str:
        return "doctor text report"

    def fake_json(_report: object) -> str:
        return '{"doctor": true}'

    if stub_text:
        monkeypatch.setattr("gymrat.cli.commands.doctor.render_doctor_report", fake_text)
    if stub_json:
        monkeypatch.setattr("gymrat.cli.commands.doctor.render_doctor_json", fake_json)
