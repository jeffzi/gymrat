"""Tests for ``python -m gymrat``: the ``__main__`` module entry point.

The module entry must behave identically to ``python -m gymrat.cli.app`` for
``--help`` and ``--version``, so the two subprocesses are compared directly.
"""

import importlib.metadata
import os
import subprocess
import sys

import pytest

from tests._ansi import strip_ansi


def _child_env() -> dict[str, str]:
    """Child environment with color forced off for deterministic output."""
    env = dict(os.environ)
    env["NO_COLOR"] = "1"
    env.pop("FORCE_COLOR", None)
    return env


def _normalize(text: str) -> str:
    """Collapse whitespace after stripping ANSI codes so reflowed blocks match."""
    return " ".join(strip_ansi(text).split())


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m <module> <args>`` in a child process with color forced off."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        check=False,
        env=_child_env(),
    )


def test_main_module_when_help_does_show_same_description_and_epilogue_as_cli_app():
    module_result = _run_module("gymrat", "--help")
    app_result = _run_module("gymrat.cli.app", "--help")

    assert module_result.returncode == 0, module_result.stderr
    module_text = _normalize(module_result.stdout)
    app_text = _normalize(app_result.stdout)
    assert "Performance comparison tool for benchmarks" in module_text
    assert 'gymrat compare main my-branch --bench "npm run bench"' in module_text
    assert module_text == app_text


def test_main_module_when_version_flag_does_print_package_version():
    result = _run_module("gymrat", "--version")

    assert result.returncode == 0, result.stderr
    assert importlib.metadata.version("gymrat") in result.stdout


# ---------------------------------------------------------------------------
# program name in the usage line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["gymrat", "gymrat.cli.app"])
def test_module_entry_when_help_does_print_usage_with_gymrat_program_name(module: str):
    result = _run_module(module, "--help")

    assert result.returncode == 0, result.stderr
    assert "Usage: gymrat [" in _normalize(result.stdout)


@pytest.mark.parametrize("module", ["gymrat", "gymrat.cli.app"])
def test_module_entry_when_usage_error_does_print_usage_with_gymrat_program_name(module: str):
    result = _run_module(module, "compare", "main")

    assert result.returncode == 2
    assert "Usage: gymrat compare [" in _normalize(result.stderr)
