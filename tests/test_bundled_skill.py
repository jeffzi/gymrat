import re
import zipfile
from collections.abc import Callable
from importlib import resources
from importlib.resources.abc import Traversable
from typing import NoReturn
from unittest.mock import create_autospec

import pytest

from gymrat.bundled_skill import read_bundled_skill
from gymrat.errors import GymratError

SKILL_HEADING = "# Driving a gymrat optimization session"


def _assert_reinstall_error(error: GymratError, cause_type: type[BaseException]) -> None:
    """Assert the failure's SKILL.md message, reinstall hint, and cause."""
    assert "SKILL.md" in str(error)
    assert error.hint is not None
    assert re.search("reinstall", error.hint, re.IGNORECASE)
    assert isinstance(error.__cause__, cause_type)


# ---------------------------------------------------------------------------
# read_bundled_skill
# ---------------------------------------------------------------------------


def test_read_bundled_skill_when_packaged_file_present_does_return_its_text():
    result = read_bundled_skill()

    assert SKILL_HEADING in result


def test_read_bundled_skill_when_file_unreadable_does_raise_gymrat_error(
    monkeypatch: pytest.MonkeyPatch,
):
    missing = resources.files("gymrat") / "skills" / "gymrat" / "does-not-exist.md"
    monkeypatch.setattr("gymrat.bundled_skill._skill_resource", lambda: missing)

    with pytest.raises(GymratError) as caught:
        read_bundled_skill()

    _assert_reinstall_error(caught.value, FileNotFoundError)
    assert str(missing) in str(caught.value)


@pytest.mark.parametrize(
    ("side_effect", "cause_type"),
    [
        pytest.param(
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
            UnicodeDecodeError,
            id="bad-encoding",
        ),
        pytest.param(
            zipfile.BadZipFile("Bad magic number"),
            zipfile.BadZipFile,
            id="corrupt-archive",
        ),
    ],
)
def test_read_bundled_skill_when_read_text_fails_does_raise_gymrat_error(
    side_effect: Exception,
    cause_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
):
    mock_resource = create_autospec(Traversable, instance=True)
    mock_resource.read_text.side_effect = side_effect
    monkeypatch.setattr("gymrat.bundled_skill._skill_resource", lambda: mock_resource)

    with pytest.raises(GymratError) as caught:
        read_bundled_skill()

    _assert_reinstall_error(caught.value, cause_type)


def test_read_bundled_skill_when_resource_lookup_fails_does_raise_gymrat_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def unresolvable_resource() -> NoReturn:
        message = "Bad magic number"
        raise zipfile.BadZipFile(message)

    monkeypatch.setattr("gymrat.bundled_skill._skill_resource", unresolvable_resource)

    with pytest.raises(GymratError) as caught:
        read_bundled_skill()

    _assert_reinstall_error(caught.value, zipfile.BadZipFile)


# ---------------------------------------------------------------------------
# Loop discipline — stop condition rule
# ---------------------------------------------------------------------------


def _never_stop_early_rule() -> str:
    """The 'Loop discipline' list item forbidding a stop before a stop condition fires."""
    section = _section("## Loop discipline")
    return next(item for item in section.split("\n\n") if "never stop before" in item.lower())


@pytest.mark.parametrize(
    "phrase",
    [
        pytest.param("`iterate`", id="names-the-action"),
        pytest.param("exits 1", id="command-exit-code"),
        pytest.param("tool", id="tool-form"),
        pytest.param("stopped: true", id="tool-stopped-field"),
        pytest.param("reason", id="tool-reason-field"),
    ],
)
def test_read_bundled_skill_when_stop_condition_rule_read_does_state_both_command_and_tool_forms(
    phrase: str,
):
    rule = _never_stop_early_rule()

    assert phrase in rule


# ---------------------------------------------------------------------------
# Supervised mode — refused command forms
# ---------------------------------------------------------------------------


def _section(heading: str) -> str:
    """The skill text under ``heading``, up to the next top-level section."""
    return read_bundled_skill().partition(heading)[2].partition("\n## ")[0]


def _paragraph(section: str, marker: str) -> str:
    """The paragraph (or list item) of ``section`` containing ``marker``, unwrapped to one line."""
    paragraphs = (" ".join(item.split()) for item in section.split("\n\n"))
    return next(item for item in paragraphs if marker in item)


def _supervised_mode_paragraph(marker: str) -> str:
    return _paragraph(_section("### Supervised mode"), marker)


def _nested_supervise_paragraph() -> str:
    return _supervised_mode_paragraph("Never run `gymrat supervise` yourself.")


def _concurrent_sessions_rule() -> str:
    return _paragraph(_section("## Loop discipline"), "Never run concurrent sessions.")


NESTED_SUPERVISE_PASSAGES = [
    pytest.param(_nested_supervise_paragraph, id="supervised-mode"),
    pytest.param(_concurrent_sessions_rule, id="concurrent-sessions-rule"),
]


@pytest.mark.parametrize(
    "phrase",
    [
        pytest.param("`gymrat iterate`", id="names-iterate-command"),
        pytest.param("`gymrat probe`", id="names-probe-command"),
        pytest.param("through Bash", id="names-shell-form"),
        pytest.param("refused", id="command-form-refused"),
        pytest.param("only form", id="tools-only-form"),
    ],
)
def test_read_bundled_skill_when_tools_paragraph_read_does_state_command_form_refused(
    phrase: str,
):
    paragraph = _supervised_mode_paragraph("Use the `probe` and `iterate` tools.")

    assert phrase in paragraph


@pytest.mark.parametrize("passage", NESTED_SUPERVISE_PASSAGES)
def test_read_bundled_skill_when_nested_supervise_passage_read_does_state_launch_refused(
    passage: Callable[[], str],
):
    text = passage()

    assert "nested" in text
    assert "refused" in text


@pytest.mark.parametrize("passage", NESTED_SUPERVISE_PASSAGES)
def test_read_bundled_skill_when_read_does_not_say_lock_lets_nested_supervise_through(
    passage: Callable[[], str],
):
    text = passage()

    assert not re.search(r"does not (stop|block)|\blets?\b.*\bthrough\b", text)


@pytest.mark.parametrize(
    ("heading", "phrase"),
    [
        pytest.param(
            "### 3. The iteration cycle",
            "Never pass `--bench` or `--samples` to `iterate`.",
            id="iteration-cycle-rule",
        ),
        pytest.param(
            "## Red flags",
            "About to pass `--bench` or `--samples` to `iterate`.",
            id="red-flag",
        ),
    ],
)
def test_read_bundled_skill_when_read_does_keep_iterate_bench_and_samples_rule(
    heading: str,
    phrase: str,
):
    section = _section(heading)

    assert phrase in section


# ---------------------------------------------------------------------------
# Supervised mode — refused edits and background commands
# ---------------------------------------------------------------------------

OUTSIDE_EDIT_MARKER = "**A file edit outside the experiment worktree is refused.**"
BACKGROUND_MARKER = "**Never run a gymrat command in the background.**"


@pytest.mark.parametrize(
    ("marker", "phrase"),
    [
        pytest.param(
            OUTSIDE_EDIT_MARKER, "Edit, Write, MultiEdit, or NotebookEdit", id="edit-tools"
        ),
        pytest.param(
            OUTSIDE_EDIT_MARKER, "outside the experiment worktree", id="edit-outside-worktree"
        ),
        pytest.param(OUTSIDE_EDIT_MARKER, "temporary directories", id="edit-temp-directories"),
        pytest.param(OUTSIDE_EDIT_MARKER, "names its rule", id="refusal-names-rule"),
        pytest.param(OUTSIDE_EDIT_MARKER, "change the call", id="fix-changes-call"),
        pytest.param(OUTSIDE_EDIT_MARKER, "not to retry it", id="fix-is-not-retry"),
        pytest.param(
            BACKGROUND_MARKER, "contains `gymrat` is refused", id="background-command-refused"
        ),
    ],
)
def test_read_bundled_skill_when_supervised_mode_read_does_state_edit_and_background_refusals(
    marker: str,
    phrase: str,
):
    paragraph = _supervised_mode_paragraph(marker)

    assert phrase in paragraph


@pytest.mark.parametrize(
    "phrase",
    [
        pytest.param("Under supervise", id="names-supervised-mode"),
        pytest.param("refused", id="main-tree-edit-refused"),
        pytest.param("make the change in the experiment worktree", id="edit-in-worktree"),
        pytest.param("a person's main-tree edits", id="sync-for-person-edits"),
    ],
)
def test_read_bundled_skill_when_sync_section_read_does_state_main_tree_edit_refused(
    phrase: str,
):
    paragraph = _paragraph(_section("## Syncing main-tree edits"), "Under supervise")

    assert phrase in paragraph


def test_read_bundled_skill_when_iteration_cycle_read_does_keep_worktree_rule():
    section = _section("### 3. The iteration cycle")

    assert "Edit code **only in the experiment worktree**" in section
