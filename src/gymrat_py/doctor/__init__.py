"""The ``doctor`` diagnostic: report model, section builders, and renderers.

The report model and pure section builders live in :mod:`gymrat_py.doctor.checks`,
the filesystem-touching bench smoke run in :mod:`gymrat_py.doctor.bench`, and the
text and JSON renderers in :mod:`gymrat_py.doctor.render`.
"""

from gymrat_py.doctor.bench import BenchSectionInput, build_bench_section
from gymrat_py.doctor.checks import (
    Check,
    CheckSection,
    CheckStatus,
    DoctorReport,
    EnvironmentInfo,
    build_config_section,
    build_environment_section,
    build_workflow_section,
    create_doctor_report,
)
from gymrat_py.doctor.render import render_doctor_json, render_doctor_report

__all__ = [
    "BenchSectionInput",
    "Check",
    "CheckSection",
    "CheckStatus",
    "DoctorReport",
    "EnvironmentInfo",
    "build_bench_section",
    "build_config_section",
    "build_environment_section",
    "build_workflow_section",
    "create_doctor_report",
    "render_doctor_json",
    "render_doctor_report",
]
