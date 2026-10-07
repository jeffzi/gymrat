"""Tests for the supervisor kickoff composition.

``compose_kickoff`` reads the bundled skill, validates the configured runbook,
and returns the system-prompt append and the kickoff message the supervisor
hands to the driven session.
"""

from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.supervisor.kickoff import KickoffResult, compose_kickoff
from tests._config import benchless_config

# The heading the packaged SKILL.md opens its body with; proves the real
# bundled skill text made it into the append.
SKILL_MARKER = "# Driving a gymrat optimization session"

RUNBOOK_CONTENT = "# My Runbook\n\nStep 1: run benchmarks.\n"

_EXPERIMENT_WORKTREE = "/tmp/experiment"

# The supervised-mode contract an unattended session runs under, word for word.
_CONTRACT_PARAGRAPH = (
    "No human reads the turns of this session. Ask-first rules resolve to deciding "
    "from the runbook — the runbook is the authority. When the work is done, run "
    '`gymrat stop -m "<report>"` and only then end the turn. The supervisor replies '
    "after every turn and the session continues, so ending a turn never waits for "
    "anything. Never run a gymrat command in the background."
)

# A minimal skill body used where the paragraph checks below only need some
# text, not the bundled skill's actual content.
_GENERIC_SKILL_TEXT = "# Skill Title\n\nSome guidance.\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    config = benchless_config(runbook=_write_runbook(tmp_path))
    return compose_kickoff(config, experiment_worktree=experiment_worktree)


@pytest.fixture
def generic_kickoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KickoffResult:
    """The kickoff composed around a minimal skill body and the default runbook."""
    return _compose_with_skill_text(_GENERIC_SKILL_TEXT, tmp_path, monkeypatch)


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
    config = benchless_config(runbook=None)

    with pytest.raises(GymratError, match="bundled skill unavailable"):
        compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)


# ---------------------------------------------------------------------------
# compose_kickoff — runbook validation
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_no_runbook_configured_does_raise_naming_gymrat_toml():
    config = benchless_config(runbook=None)

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
    config = benchless_config(runbook=missing)

    with pytest.raises(GymratError) as excinfo:
        compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

    assert str(excinfo.value) == f"Runbook not found at {missing}."
    assert excinfo.value.hint
    assert excinfo.value.__cause__ is not None


# ---------------------------------------------------------------------------
# compose_kickoff — happy path composition
# ---------------------------------------------------------------------------


def test_compose_kickoff_when_skill_and_runbook_present_does_put_the_skill_body_before_runbook(
    tmp_path: Path,
):
    config = benchless_config(runbook=_write_runbook(tmp_path))

    result = compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

    append = result.system_prompt_append
    prelude = append.partition(SKILL_MARKER)[0]
    assert SKILL_MARKER in append
    assert RUNBOOK_CONTENT in append
    assert f"## Runbook: {config.runbook}" in append
    assert append.index(SKILL_MARKER) < append.index("## Runbook:")
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
    config = benchless_config(runbook=_write_runbook(tmp_path))

    result = compose_kickoff(config, experiment_worktree=_EXPERIMENT_WORKTREE)

    assert "optimization" in result.kickoff


def test_compose_kickoff_when_prompt_given_does_start_with_it_verbatim(tmp_path: Path):
    config = benchless_config(runbook=_write_runbook(tmp_path))

    result = compose_kickoff(
        config, "optimize the decoder loop", experiment_worktree=_EXPERIMENT_WORKTREE
    )

    assert result.kickoff.startswith("optimize the decoder loop")


# ---------------------------------------------------------------------------
# compose_kickoff — clock rule in system-prompt append
# ---------------------------------------------------------------------------


def _clock_rule_paragraph(append: str) -> str:
    """The append paragraph carrying the wall-clock reading rule."""
    return next(p for p in append.split("\n\n") if "never estimate" in p.lower())


@pytest.mark.parametrize(
    "phrase",
    [
        pytest.param("bash", id="command-form"),
        pytest.param("time-left", id="command-prints-time-left"),
        pytest.param("`iterate`", id="iterate-tool"),
        pytest.param("`probe`", id="probe-tool"),
        pytest.param("budget.remaining_seconds", id="tool-json-field"),
        pytest.param("json", id="tool-form"),
        pytest.param("wall-clock", id="wall-clock-cap"),
        pytest.param("time left", id="read-time-left"),
        pytest.param("never estimate", id="never-estimate"),
        pytest.param("records nothing", id="killed-measurement-records-nothing"),
    ],
)
def test_compose_kickoff_when_happy_path_does_state_phrase_in_the_clock_rule(
    generic_kickoff: KickoffResult,
    phrase: str,
):
    result = generic_kickoff

    clock_rule = _clock_rule_paragraph(result.system_prompt_append).lower()

    assert phrase in clock_rule


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
    config = benchless_config(runbook=_write_runbook(tmp_path))

    result = compose_kickoff(config, prompt, experiment_worktree=experiment_path)

    trailing = result.kickoff.split("\n\n")[-1]
    assert "session" in trailing.lower()
    assert "baseline" in trailing.lower()
    assert experiment_path in trailing
    assert "step" in trailing.lower()
    assert "runbook" in trailing.lower()


# ---------------------------------------------------------------------------
# compose_kickoff — no cap or spend language in code-authored text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["system_prompt_append", "kickoff"])
@pytest.mark.parametrize("forbidden", ["usd", "spend", "$", "30 minute", "max_minutes"])
def test_compose_kickoff_when_skill_mentions_spend_does_keep_cap_and_spend_out_of_authored_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    forbidden: str,
):
    skill_text = "# Skill Title\n\nSome guidance with --max-usd 10 and spend and $ dollar.\n"

    result = _compose_with_skill_text(skill_text, tmp_path, monkeypatch)

    # The skill text may legitimately mention spend; only code-authored text is checked.
    authored = getattr(result, field).replace(skill_text, "").lower()
    assert forbidden not in authored


# ---------------------------------------------------------------------------
# compose_kickoff — tools paragraph in system-prompt append
# ---------------------------------------------------------------------------


def _tools_paragraph_index(paragraphs: list[str]) -> int:
    """Index of the append paragraph that introduces the ``probe`` tool."""
    return next(i for i, paragraph in enumerate(paragraphs) if "`probe`" in paragraph)


@pytest.mark.parametrize(
    "phrase",
    [
        "`probe`",
        "`iterate`",
        "bash",
        "`measure`",
        "`compare`",
        "`keep`",
        "`discard`",
        "`status`",
        "`stop`",
        "foreground",
        "json document",
    ],
)
def test_compose_kickoff_when_happy_path_does_state_phrase_in_tools_paragraph(
    generic_kickoff: KickoffResult,
    phrase: str,
):
    result = generic_kickoff

    paragraphs = result.system_prompt_append.split("\n\n")
    tools_paragraph = paragraphs[_tools_paragraph_index(paragraphs)]
    assert phrase in tools_paragraph.lower()


def test_compose_kickoff_when_happy_path_does_order_the_authored_paragraphs_ahead_of_the_runbook(
    generic_kickoff: KickoffResult,
):
    result = generic_kickoff

    paragraphs = result.system_prompt_append.split("\n\n")
    contract_index = paragraphs.index(_CONTRACT_PARAGRAPH)
    clock_rule_index = paragraphs.index(_clock_rule_paragraph(result.system_prompt_append))
    runbook_index = next(i for i, p in enumerate(paragraphs) if p.startswith("## Runbook:"))
    assert contract_index < _tools_paragraph_index(paragraphs) < clock_rule_index < runbook_index


def test_compose_kickoff_when_happy_path_does_not_mention_tools_in_kickoff(
    generic_kickoff: KickoffResult,
):
    result = generic_kickoff

    kickoff_lower = result.kickoff.lower()
    assert "tool" not in kickoff_lower
    assert "`probe`" not in kickoff_lower
