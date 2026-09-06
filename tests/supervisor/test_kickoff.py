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
from gymrat.supervisor.kickoff import KickoffResult

# The heading the packaged SKILL.md opens its body with; proves the real
# bundled skill text made it into the append.
SKILL_MARKER = "# Driving a gymrat optimization session"

RUNBOOK_CONTENT = "# My Runbook\n\nStep 1: run benchmarks.\n"

_EXPERIMENT_WORKTREE = "/tmp/experiment"

# A minimal skill body used where the clock-rule/cap-omission checks below
# only need some text, not the bundled skill's actual content.
_GENERIC_SKILL_TEXT = "# Skill Title\n\nSome guidance.\n"


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
    runbook_path = directory / "runbook.md"
    runbook_path.write_text(content, encoding="utf-8")
    return str(runbook_path)


def _compose_with_skill_text(
    skill_text: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    experiment_worktree: str = _EXPERIMENT_WORKTREE,
) -> KickoffResult:
    monkeypatch.setattr("gymrat.supervisor.kickoff.read_bundled_skill", lambda: skill_text)
    config = _make_config(runbook=_write_runbook(tmp_path))
    return compose_kickoff(config, experiment_worktree=experiment_worktree)


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
        compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)


# ---------------------------------------------------------------------------
# compose_kickoff — runbook validation
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_no_runbook_configured_does_raise_naming_gymrat_toml():
    config = _make_config(runbook=None)

    with pytest.raises(GymratError) as excinfo:
        compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

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
        compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

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

    result = compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

    append = result.system_prompt_append
    assert SKILL_MARKER in append
    assert RUNBOOK_CONTENT in append
    assert f"## Runbook: {config.runbook}" in append
    assert append.index(SKILL_MARKER) < append.index("## Runbook:")


def test_compose_kickoff_when_bundled_skill_has_frontmatter_does_omit_it_from_append(
    tmp_path: Path,
):
    config = _make_config(runbook=_write_runbook(tmp_path))

    result = compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

    prelude = result.system_prompt_append.partition(SKILL_MARKER)[0]
    assert SKILL_MARKER in result.system_prompt_append
    assert "---" not in prelude
    assert "name: gymrat" not in prelude
    assert "description:" not in prelude
    assert "when_to_use:" not in prelude


def test_compose_kickoff_when_skill_has_no_frontmatter_does_keep_text_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    skill_text = "# Plain skill\n\nGuidance.\n\n---\n\nMore after a horizontal rule.\n"

    result = _compose_with_skill_text(skill_text, tmp_path, monkeypatch)

    assert skill_text in result.system_prompt_append


def test_compose_kickoff_when_frontmatter_values_span_lines_does_drop_whole_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    body = "# Folded skill\n\nBody survives.\n"
    skill_text = (
        "---\n"
        "name: folded\n"
        "description: >-\n"
        "  first folded line\n"
        "  second folded line\n"
        "when_to_use: >-\n"
        "  another folded line\n"
        "---\n"
        "\n" + body
    )

    result = _compose_with_skill_text(skill_text, tmp_path, monkeypatch)

    prelude = result.system_prompt_append.partition("# Folded skill")[0]
    assert body in result.system_prompt_append
    assert "---" not in prelude
    assert "name: folded" not in prelude
    assert "folded line" not in prelude


# ---------------------------------------------------------------------------
# compose_kickoff — kickoff message
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_no_prompt_given_does_return_default_mentioning_optimization(
    tmp_path: Path,
):
    config = _make_config(runbook=_write_runbook(tmp_path))

    result = compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

    assert "optimization" in result.kickoff


def test_compose_kickoff_when_prompt_given_does_start_with_it_verbatim(tmp_path: Path):
    config = _make_config(runbook=_write_runbook(tmp_path))

    result = compose_kickoff(
        config, "optimize the decoder loop", experiment_worktree=_EXPERIMENT_WORKTREE
    )

    assert result.kickoff.startswith("optimize the decoder loop")


# ---------------------------------------------------------------------------
# compose_kickoff — clock rule in system-prompt append
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_happy_path_does_include_clock_rule_in_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    result = _compose_with_skill_text(_GENERIC_SKILL_TEXT, tmp_path, monkeypatch)

    append = result.system_prompt_append.lower()
    assert "wall-clock" in append
    assert "time left" in append
    assert "never estimate" in append
    assert "records nothing" in append
    records_nothing_pos = append.index("records nothing")
    runbook_pos = append.index("## runbook:")
    assert records_nothing_pos < runbook_pos


@pytest.mark.parametrize("field", ["system_prompt_append", "kickoff"])
def test_compose_kickoff_when_happy_path_does_omit_cap_numbers_and_spend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
):
    result = _compose_with_skill_text(_GENERIC_SKILL_TEXT, tmp_path, monkeypatch)

    text = getattr(result, field).lower()
    assert "30 minute" not in text
    assert "max_minutes" not in text
    assert "spend" not in text


# ---------------------------------------------------------------------------
# non-UTF-8 runbook
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_runbook_not_utf8_does_raise_gymrat_error_naming_path(
    tmp_path: Path,
):
    runbook_path = tmp_path / "runbook.md"
    runbook_path.write_bytes(b"\x80\x81\x82 invalid utf-8")
    config = _make_config(runbook=str(runbook_path))

    with pytest.raises(GymratError) as excinfo:
        compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

    assert str(runbook_path) in str(excinfo.value)
    assert excinfo.value.hint


# ---------------------------------------------------------------------------
# compose_kickoff — pre-flight-done paragraph in kickoff message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param(None, id="default-prompt"),
        pytest.param("optimize the decoder loop", id="prompt-given"),
    ],
)
def test_compose_kickoff_when_prompt_is_default_or_given_does_end_with_preflight_paragraph(
    tmp_path: Path,
    prompt: str | None,
):
    experiment_path = str(tmp_path / "experiment-worktree")
    config = _make_config(runbook=_write_runbook(tmp_path))

    result = compose_kickoff(config, prompt, experiment_worktree=experiment_path)

    trailing = result.kickoff.split("\n\n")[-1]
    assert "session" in trailing.lower()
    assert "baseline" in trailing.lower()
    assert experiment_path in trailing
    assert "step" in trailing.lower()
    assert "runbook" in trailing.lower()
