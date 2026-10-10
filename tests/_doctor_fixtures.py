"""Shared inputs and seam patches for ``gymrat.doctor`` tests.

This is test-support code, not a test module: it carries no test functions of
its own. :func:`environment_info` builds the version and platform context a
doctor report opens with, and :func:`doctor_report` and
:func:`single_check_report` build whole reports on it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from gymrat.doctor import (
    Check,
    CheckSection,
    DoctorReport,
    EnvironmentInfo,
    create_doctor_report,
)


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
