"""Tests for the supervisor kickoff composition.

``compose_kickoff`` reads the bundled skill, validates the configured runbook,
and returns the system-prompt append and the kickoff message the supervisor
hands to the driven session.
"""

from pathlib import Path
from unittest.mock import create_autospec

import pytest

from gymrat.bundled_skill import read_bundled_skill
from gymrat.errors import GymratError
from gymrat.supervisor.kickoff import DEFAULT_KICKOFF, KickoffResult, compose_kickoff
from tests._config import benchless_config

RUNBOOK_CONTENT = "# My Runbook\n\nStep 1: run benchmarks.\n"

_EXPERIMENT_WORKTREE = "/tmp/experiment"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bundled_skill_heading() -> str:
    # The heading the packaged SKILL.md opens its body with; finding it in the append proves the
    # real bundled skill text made it in.
    return next(line for line in read_bundled_skill().splitlines() if line.startswith("# "))


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
    monkeypatch.setattr(
        "gymrat.supervisor.kickoff.read_bundled_skill",
        create_autospec(read_bundled_skill, return_value=skill_text),
    )
    config = benchless_config(runbook=_write_runbook(tmp_path))
    return compose_kickoff(config, experiment_worktree=experiment_worktree)


# ---------------------------------------------------------------------------
# compose_kickoff — runbook validation
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_no_runbook_configured_does_raise():
    config = benchless_config(runbook=None)

    with pytest.raises(GymratError):
        compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)


def test_compose_kickoff_when_runbook_path_missing_does_raise_not_found_with_cause(
    tmp_path: Path,
):
    missing = str(tmp_path / "absent-runbook.md")
    config = benchless_config(runbook=missing)

    with pytest.raises(GymratError) as excinfo:
        compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

    assert missing in str(excinfo.value)
    assert excinfo.value.__cause__ is not None


# ---------------------------------------------------------------------------
# compose_kickoff — skill and runbook in the system-prompt append
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_bundled_skill_meets_a_runbook_does_append_the_skill(
    tmp_path: Path,
):
    config = benchless_config(runbook=_write_runbook(tmp_path))
    skill_heading = _bundled_skill_heading()

    result = compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

    assert skill_heading in result.system_prompt_append


def test_compose_kickoff_when_runbook_configured_does_append_its_content_and_path(
    tmp_path: Path,
):
    runbook = _write_runbook(tmp_path)
    config = benchless_config(runbook=runbook)

    result = compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

    append = result.system_prompt_append
    assert RUNBOOK_CONTENT in append
    assert runbook in append


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


@pytest.mark.parametrize(
    ("prompt", "opening"),
    [
        pytest.param(None, DEFAULT_KICKOFF, id="default-prompt"),
        pytest.param("optimize the decoder loop", "optimize the decoder loop", id="prompt-given"),
    ],
)
def test_compose_kickoff_when_prompt_is_default_or_given_does_lead_the_kickoff_ahead_of_preflight(
    tmp_path: Path,
    prompt: str | None,
    opening: str,
):
    experiment_path = str(tmp_path / "experiment-worktree")
    config = benchless_config(runbook=_write_runbook(tmp_path))

    result = compose_kickoff(config, prompt, experiment_worktree=experiment_path)

    trailing = result.kickoff.split("\n\n")[-1]
    assert result.kickoff.startswith(opening)
    assert experiment_path in trailing


# ---------------------------------------------------------------------------
# non-UTF-8 runbook
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_runbook_not_utf8_does_raise_gymrat_error_naming_path(
    tmp_path: Path,
):
    runbook_path = tmp_path / "runbook.md"
    runbook_path.write_bytes(b"\x80\x81\x82 invalid utf-8")
    config = benchless_config(runbook=str(runbook_path))

    with pytest.raises(GymratError) as excinfo:
        compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

    assert str(runbook_path) in str(excinfo.value)
