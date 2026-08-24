"""Shared fixtures for the report tests.

Every report test renders against a stable, colorless default so a stray
``FORCE_COLOR`` or ``NO_COLOR`` in the developer's shell cannot bleed ANSI into
an assertion. The color-specific tests opt back in by patching the environment
themselves.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
