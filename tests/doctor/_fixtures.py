"""Shared seam patches for ``gymrat.doctor.report`` tests.

This is test-support code, not a test module: it carries no test functions of
its own. Both ``tests/doctor/test_report.py`` (assembly-level) and
``tests/cli/test_doctor_cmd.py`` (CLI-level) patch the config-inspection,
config-section, workflow-section, and bench-section seams on
``gymrat.doctor.report`` the same way; :func:`patch_common_seams` holds that
shared body. Each call site still owns its own environment/git seams and its
own ``problems`` wording, since those diverge between the two test files.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

from gymrat.config import BenchlessConfig
from gymrat.config.inspect import ConfigInspection
from gymrat.doctor.checks import Check, CheckSection

_MODULE = "gymrat.doctor.report"


def sample_config() -> BenchlessConfig:
    return BenchlessConfig(
        adapter="metric-lines",
        samples=10,
        timeout_seconds=1800,
        unstable_noise_pct=200,
        primary="geomean",
    )


def fixed_section(title: str, checks: list[Check]) -> Callable[..., CheckSection]:
    """Build a section-builder stand-in that always returns the same section."""

    def build(*_a: object, **_k: object) -> CheckSection:
        return CheckSection(title=title, checks=checks)

    return build


def patch_common_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_failure: bool,
    bench_fail: bool,
    problems: list[str],
) -> SimpleNamespace:
    """Patch the config-inspection, config, workflow, and bench seams shared by both test files."""
    inspection = ConfigInspection(
        config_path="/missing/gymrat.json" if config_failure else "/project/gymrat.json",
        problems=problems,
        config=None if config_failure else sample_config(),
        bench="node bench.js",
    )

    def fake_inspect(*_a: object, **_k: object) -> ConfigInspection:
        return inspection

    monkeypatch.setattr(f"{_MODULE}.inspect_config", fake_inspect)

    config_checks = (
        [Check("config", "fail", "not found", hint="create gymrat.json")]
        if config_failure
        else [Check("config", "ok", "/project/gymrat.json")]
    )
    monkeypatch.setattr(
        f"{_MODULE}.build_config_section", fixed_section("Configuration", config_checks)
    )

    monkeypatch.setattr(
        f"{_MODULE}.build_workflow_section",
        fixed_section("Workflow", [Check("skill file", "ok", "found")]),
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

    monkeypatch.setattr(f"{_MODULE}.build_bench_section", bench_section)

    return SimpleNamespace(bench_calls=bench_calls)
