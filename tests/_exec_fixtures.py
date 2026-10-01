"""The live-group isolation both ``exec`` test modules install as an autouse fixture."""

import pytest

from gymrat import exec as exec_mod


@pytest.fixture(autouse=True)
def isolate_live_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the module-level live-group registry from bleeding across tests."""
    monkeypatch.setattr(exec_mod, "_live_process_groups", set())
