"""Tests for the supervisor kickoff composition.

``compose_kickoff`` reads the bundled skill, validates the configured runbook,
and returns the system-prompt append and the kickoff message the supervisor
hands to the driven session.
"""

from pathlib import Path

import pytest

from gymrat.config import BenchlessConfig
from gymrat.errors import GymratError
from gymrat.supervisor import compose_kickoff

# The heading the packaged SKILL.md opens its body with; proves the real
# bundled skill text made it into the append.
SKILL_MARKER = "# Driving a gymrat optimization session"

RUNBOOK_CONTENT = "# My Runbook\n\nStep 1: run benchmarks.\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(runbook: str | None) -> BenchlessConfig:
    """A benchless run configuration pointing at ``runbook`` (or none)."""
    return BenchlessConfig(
        adapter="mitata",
        samples=30,
        timeout_seconds=60,
        unstable_noise_pct=5.0,
        primary="geomean",
        runbook=runbook,
    )


def _write_runbook(directory: Path, content: str = RUNBOOK_CONTENT) -> str:
    """Write a runbook file under ``directory`` and return its path string."""
    runbook_path = directory / "runbook.md"
    runbook_path.write_text(content, encoding="utf-8")
    return str(runbook_path)


# ---------------------------------------------------------------------------
# compose_kickoff — bundled skill
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_bundled_skill_missing_does_raise_before_runbook_check(
    monkeypatch: pytest.MonkeyPatch,
):
    def _raise() -> str:
        message = "bundled skill unavailable"
        raise GymratError(message)

    monkeypatch.setattr("gymrat.supervisor.kickoff.read_bundled_skill", _raise)
    config = _make_config(runbook=None)

    with pytest.raises(GymratError, match="bundled skill unavailable"):
        compose_kickoff(config)


# ---------------------------------------------------------------------------
# compose_kickoff — runbook validation
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_no_runbook_configured_does_raise_naming_gymrat_toml():
    config = _make_config(runbook=None)

    with pytest.raises(GymratError) as excinfo:
        compose_kickoff(config)

    message = str(excinfo.value)
    assert "runbook" in message.lower()
    assert "gymrat.toml" in message
    assert excinfo.value.hint


def test_compose_kickoff_when_runbook_path_missing_does_raise_not_found_with_cause(
    tmp_path: Path,
):
    missing = str(tmp_path / "absent-runbook.md")
    config = _make_config(runbook=missing)

    with pytest.raises(GymratError) as excinfo:
        compose_kickoff(config)

    assert str(excinfo.value) == f"Runbook not found at {missing}."
    assert excinfo.value.hint
    assert excinfo.value.__cause__ is not None


# ---------------------------------------------------------------------------
# compose_kickoff — happy path composition
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_skill_and_runbook_present_does_order_skill_before_runbook(
    tmp_path: Path,
):
    config = _make_config(runbook=_write_runbook(tmp_path))

    result = compose_kickoff(config)

    append = result.system_prompt_append
    assert SKILL_MARKER in append
    assert RUNBOOK_CONTENT in append
    assert f"## Runbook: {config.runbook}" in append
    assert append.index(SKILL_MARKER) < append.index("## Runbook:")


# ---------------------------------------------------------------------------
# compose_kickoff — kickoff message
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_no_prompt_given_does_return_default_mentioning_optimization(
    tmp_path: Path,
):
    config = _make_config(runbook=_write_runbook(tmp_path))

    result = compose_kickoff(config)

    assert "optimization" in result.kickoff


def test_compose_kickoff_when_prompt_given_does_return_it_verbatim(tmp_path: Path):
    config = _make_config(runbook=_write_runbook(tmp_path))

    result = compose_kickoff(config, "optimize the decoder loop")

    assert result.kickoff == "optimize the decoder loop"


# ---------------------------------------------------------------------------
# B33 — non-UTF-8 runbook
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_runbook_not_utf8_does_raise_gymrat_error_naming_path(
    tmp_path: Path,
):
    runbook_path = tmp_path / "runbook.md"
    runbook_path.write_bytes(b"\x80\x81\x82 invalid utf-8")
    config = _make_config(runbook=str(runbook_path))

    with pytest.raises(GymratError) as excinfo:
        compose_kickoff(config)

    assert str(runbook_path) in str(excinfo.value)
    assert excinfo.value.hint
