"""Shared report inputs for ``gymrat.doctor`` tests.

This is test-support code, not a test module: it carries no test functions of
its own. :func:`environment_info` builds the version and platform context a
doctor report opens with, and :func:`doctor_report` and
:func:`single_check_report` build whole reports on it.
"""

from __future__ import annotations

from gymrat.doctor import (
    Check,
    CheckSection,
    DoctorReport,
    EnvironmentInfo,
    create_doctor_report,
)


def environment_info() -> EnvironmentInfo:
    """The version and platform context a doctor report opens with."""
    return EnvironmentInfo(gymrat_version="0.5.0", python_version="3.13.0", platform="darwin")


def doctor_report(sections: list[CheckSection]) -> DoctorReport:
    """A doctor report over ``sections``, opened by :func:`environment_info`.

    Args:
        sections: The report's check sections, in order.

    Returns:
        The assembled report, its counts aggregated from ``sections``.
    """
    return create_doctor_report(environment_info(), sections)


def single_check_report(check: Check, title: str = "Environment") -> DoctorReport:
    """A doctor report holding ``check`` alone, in one section titled ``title``.

    Args:
        check: The report's only check.
        title: The title of the section holding it.

    Returns:
        The assembled report.
    """
    return doctor_report([CheckSection(title=title, checks=[check])])
