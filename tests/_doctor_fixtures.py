"""Shared inputs and seam patches for ``gymrat.doctor`` tests.

This is test-support code, not a test module: it carries no test functions of
its own. :func:`environment_info` builds the version and platform context a
doctor report opens with, and :func:`doctor_report` and
:func:`single_check_report` build whole reports on it, for
``tests/test_doctor.py``, ``tests/cli/supervise/test_preflight.py`` and
``tests/hardening/test_rendering_matrix.py``. :func:`patch_common_seams`
installs fixed config-inspection, config, workflow and bench sections over the
matching seams on ``gymrat.doctor``.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import create_autospec

if TYPE_CHECKING:
    import pytest

from gymrat.config import ConfigInspection, inspect_config
from gymrat.doctor import (
    Check,
    CheckSection,
    DoctorReport,
    EnvironmentInfo,
    build_bench_section,
    build_config_section,
    build_workflow_section,
    create_doctor_report,
)
from tests._config import benchless_config

_MODULE = "gymrat.doctor"


def environment_info(**overrides: Any) -> EnvironmentInfo:
    """The version and platform context a doctor report opens with, any field overridable."""
    default = EnvironmentInfo(gymrat_version="0.5.0", python_version="3.13.0", platform="darwin")
    return replace(default, **overrides)


def doctor_report(sections: list[CheckSection], **env_overrides: Any) -> DoctorReport:
    """A doctor report over ``sections``, opened by :func:`environment_info`.

    Args:
        sections: The report's check sections, in order.
        **env_overrides: Fields of the opening environment context to override.

    Returns:
        The assembled report, its counts aggregated from ``sections``.
    """
    return create_doctor_report(environment_info(**env_overrides), sections)


def single_check_report(check: Check, title: str = "Environment") -> DoctorReport:
    """A doctor report holding ``check`` alone, in one section titled ``title``.

    Args:
        check: The report's only check.
        title: The title of the section holding it.

    Returns:
        The assembled report.
    """
    return doctor_report([CheckSection(title=title, checks=[check])])


def patch_common_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_failure: bool,
    bench_fail: bool,
    problems: list[str],
) -> SimpleNamespace:
    """Patch the config-inspection, config, workflow, and bench seams with fixed sections.

    Args:
        monkeypatch: The fixture the seams are patched through.
        config_failure: Whether the config inspection finds no file and the
            Configuration section reports a failing check.
        bench_fail: Whether the Bench section reports a crashed bench.
        problems: The problems the config inspection reports.

    Returns:
        A namespace whose ``bench_calls`` lists the keyword arguments of every
        Bench section build.
    """
    inspection = ConfigInspection(
        config_path="/missing/gymrat.json" if config_failure else "/project/gymrat.json",
        problems=problems,
        config=None if config_failure else benchless_config(),
        bench="node bench.js",
    )

    monkeypatch.setattr(
        f"{_MODULE}.inspect_config", create_autospec(inspect_config, return_value=inspection)
    )

    config_checks = (
        [Check("config", "fail", "not found", hint="create gymrat.json")]
        if config_failure
        else [Check("config", "ok", "/project/gymrat.json")]
    )
    monkeypatch.setattr(
        f"{_MODULE}.build_config_section",
        create_autospec(
            build_config_section,
            return_value=CheckSection(title="Configuration", checks=config_checks),
        ),
    )

    monkeypatch.setattr(
        f"{_MODULE}.build_workflow_section",
        create_autospec(
            build_workflow_section,
            return_value=CheckSection(
                title="Workflow", checks=[Check("skill file", "ok", "found")]
            ),
        ),
    )

    bench_calls: list[dict[str, object]] = []
    bench_check = (
        Check("bench", "fail", "bench crashed")
        if bench_fail
        else Check("bench", "ok", "1 metric found")
    )

    def bench_section(*, bench: object, adapter: object, **kwargs: object) -> CheckSection:
        bench_calls.append({"bench": bench, "adapter": adapter, **kwargs})
        return CheckSection(title="Bench", checks=[bench_check])

    monkeypatch.setattr(
        f"{_MODULE}.build_bench_section",
        create_autospec(build_bench_section, side_effect=bench_section),
    )

    return SimpleNamespace(bench_calls=bench_calls)
