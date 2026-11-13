"""Stand-ins for the seams a CLI command composes over.

They replace config resolution, the render mode, the tty check, and the
``measure``, ``probe`` and ``compare`` engines.

This is test-support code, not a test module: it carries no test functions.
"""

from typing import Literal
from unittest.mock import create_autospec

import pytest

from gymrat.cli.run_setup import resolve_render_mode
from gymrat.compare import compare
from gymrat.config import ResolvedConfig, resolve_config
from gymrat.measure import MeasureOptions
from gymrat.report.types import ComparisonResult, MeasurementResult
from tests._config import resolved_config
from tests.loop._probe import install_measure
from tests.report._comparisons import create_comparison_result
from tests.report._measurements import create_measurement_result

#: The config the stubbed ``measure`` command resolves; its fake engine never benches against it.
MEASURE_CONFIG = resolved_config(
    bench="sh bench.sh",
    samples=5,
    timeout_seconds=30,
    unstable_noise_pct=2.0,
    primary="time",
)


#: The config the stubbed ``compare`` command resolves; its fake engine never benches against it.
COMPARE_CONFIG = resolved_config(bench="sh bench.sh")


def stub_config(monkeypatch: pytest.MonkeyPatch, command: str, config: ResolvedConfig) -> None:
    """Replace a command's config resolution with one that hands back ``config``.

    The stand-in keeps the real resolver's signature, so a call the real
    ``resolve_config`` would reject fails the test.

    Args:
        monkeypatch: The fixture that installs the stand-in.
        command: The module under ``gymrat.cli.commands`` whose resolver is replaced.
        config: What every resolution hands back.
    """
    monkeypatch.setattr(
        f"gymrat.cli.commands.{command}.resolve_config",
        create_autospec(resolve_config, return_value=config),
    )


def force_render_mode(
    monkeypatch: pytest.MonkeyPatch, command: str, mode: Literal["live", "plain"]
) -> None:
    """Make a command resolve ``mode`` as its render mode, whatever the terminal says.

    Args:
        monkeypatch: The fixture that installs the stand-in.
        command: The module under ``gymrat.cli.commands`` whose render mode is forced.
        mode: The render mode every resolution hands back.
    """
    monkeypatch.setattr(
        f"gymrat.cli.commands.{command}.resolve_render_mode",
        create_autospec(resolve_render_mode, return_value=mode),
    )


def stub_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the ``measure`` command's config resolution with ``MEASURE_CONFIG``."""
    stub_config(monkeypatch, "measure", MEASURE_CONFIG)


def capture_measure(
    monkeypatch: pytest.MonkeyPatch, result: MeasurementResult | None = None
) -> list[MeasureOptions]:
    """Stub the ``measure`` seam and capture the options of each call.

    The fake lets a test pin the label and raw rounds a recording is built from.

    Args:
        monkeypatch: The fixture that installs the fake.
        result: What the fake hands back; a default clean run when ``None``.

    Returns:
        The options of every call, in order; empty if the seam was never reached.
    """
    handed_back = create_measurement_result() if result is None else result
    return install_measure(monkeypatch, handed_back).calls


def stub_measure(monkeypatch: pytest.MonkeyPatch) -> list[MeasureOptions]:
    """Stub both config resolution and the ``measure`` seam with a default clean run.

    Args:
        monkeypatch: The fixture that installs the fakes.

    Returns:
        The options each call to the fake ``measure`` received, in call order.
    """
    stub_resolve(monkeypatch)
    return capture_measure(monkeypatch)


def stub_compare_command(
    monkeypatch: pytest.MonkeyPatch, result: ComparisonResult | None = None
) -> None:
    """Stub both config resolution and the ``compare`` seam, so invoking ``compare`` succeeds.

    Args:
        monkeypatch: The fixture that installs the fakes.
        result: What the fake ``compare`` hands back; a comparison with no
            regressions when ``None``.
    """
    stub_config(monkeypatch, "compare", COMPARE_CONFIG)
    handed_back = create_comparison_result() if result is None else result
    monkeypatch.setattr(
        "gymrat.compare.compare", create_autospec(compare, return_value=handed_back)
    )
