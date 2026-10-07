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


def _install_package_root(monkeypatch: pytest.MonkeyPatch, resource: Traversable) -> None:
    """Make the package-data lookup resolve every path to ``resource``."""
    root = create_autospec(Traversable, instance=True)
    root.joinpath.return_value = resource

    def package_root(_package: str) -> Traversable:
        return root

    monkeypatch.setattr(resources, "files", package_root)


# ---------------------------------------------------------------------------
# read_bundled_skill failures
# ---------------------------------------------------------------------------


def _missing_file() -> Traversable:
    """A package-data path that resolves but holds no file."""
    return resources.files("gymrat") / "skills" / "gymrat" / "does-not-exist.md"


def _failing_read(error: Exception) -> Callable[[], Traversable]:
    """Build a factory for a resolved resource whose read raises ``error``."""

    def build() -> Traversable:
        resource = create_autospec(Traversable, instance=True)
        resource.read_text.side_effect = error
        return resource

    return build


@pytest.mark.parametrize(
    ("build_resource", "cause_type"),
    [
        pytest.param(_missing_file, FileNotFoundError, id="missing-file"),
        pytest.param(
            _failing_read(UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")),
            UnicodeDecodeError,
            id="bad-encoding",
        ),
        pytest.param(
            _failing_read(zipfile.BadZipFile("Bad magic number")),
            zipfile.BadZipFile,
            id="corrupt-archive",
        ),
    ],
)
def test_read_bundled_skill_when_read_fails_does_raise_naming_the_resolved_location(
    build_resource: Callable[[], Traversable],
    cause_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
):
    resource = build_resource()
    _install_package_root(monkeypatch, resource)

    with pytest.raises(GymratError) as caught:
        read_bundled_skill()

    _assert_reinstall_error(caught.value, cause_type)
    assert f"(resolved to {resource})" in str(caught.value)


def test_read_bundled_skill_when_resource_lookup_fails_does_raise_without_a_location(
    monkeypatch: pytest.MonkeyPatch,
):
    def unresolvable_package(_package: str) -> NoReturn:
        message = "Bad magic number"
        raise zipfile.BadZipFile(message)

    monkeypatch.setattr(resources, "files", unresolvable_package)

    with pytest.raises(GymratError) as caught:
        read_bundled_skill()

    _assert_reinstall_error(caught.value, zipfile.BadZipFile)
    assert "resolved to" not in str(caught.value)


# ---------------------------------------------------------------------------
# read_bundled_skill content
# ---------------------------------------------------------------------------

OUTSIDE_EDIT_MARKER = "**A file edit outside the experiment worktree is refused.**"
BACKGROUND_MARKER = "**Never run a gymrat command in the background.**"


def _section(heading: str) -> str:
    """The skill text under ``heading``, up to the next top-level section."""
    return read_bundled_skill().partition(heading)[2].partition("\n## ")[0]


def _paragraph(section: str, marker: str) -> str:
    """The paragraph (or list item) of ``section`` containing ``marker``, unwrapped to one line."""
    paragraphs = (" ".join(item.split()) for item in section.split("\n\n"))
    return next(item for item in paragraphs if marker in item)


def _unwrapped_section(heading: str) -> str:
    """The skill text under ``heading`` with every line break and run of spaces collapsed."""
    return " ".join(_section(heading).split())


def _supervised_mode_paragraph(marker: str) -> str:
    """The supervised-mode paragraph containing ``marker``, unwrapped to one line."""
    return _paragraph(_section("### Supervised mode"), marker)


def _stop_condition_rule() -> str:
    """The loop-discipline item forbidding a stop before a stop condition fires."""
    return _paragraph(_section("## Loop discipline"), "Never stop before")


def _tools_paragraph() -> str:
    """The supervised-mode paragraph naming the probe and iterate tools."""
    return _supervised_mode_paragraph("Use the `probe` and `iterate` tools.")


def _nested_supervise_paragraph() -> str:
    """The supervised-mode paragraph forbidding a nested ``gymrat supervise``."""
    return _supervised_mode_paragraph("Never run `gymrat supervise` yourself.")


def _concurrent_sessions_rule() -> str:
    """The loop-discipline item forbidding concurrent sessions."""
    return _paragraph(_section("## Loop discipline"), "Never run concurrent sessions.")


def _iteration_cycle() -> str:
    """The iteration-cycle section, unwrapped."""
    return _unwrapped_section("### 3. The iteration cycle")


def _red_flags() -> str:
    """The red-flags section, unwrapped."""
    return _unwrapped_section("## Red flags")


def _outside_edit_paragraph() -> str:
    """The supervised-mode paragraph refusing edits outside the experiment worktree."""
    return _supervised_mode_paragraph(OUTSIDE_EDIT_MARKER)


def _background_paragraph() -> str:
    """The supervised-mode paragraph refusing background gymrat commands."""
    return _supervised_mode_paragraph(BACKGROUND_MARKER)


def _sync_paragraph() -> str:
    """The sync-section paragraph refusing main-tree edits under supervise."""
    return _paragraph(_section("## Syncing main-tree edits"), "Under supervise")


@pytest.mark.parametrize(
    ("passage", "phrase"),
    [
        pytest.param(read_bundled_skill, SKILL_HEADING, id="whole-file-heading"),
        pytest.param(_stop_condition_rule, "`iterate`", id="stop-rule-names-the-action"),
        pytest.param(_stop_condition_rule, "exits 1", id="stop-rule-command-exit-code"),
        pytest.param(_stop_condition_rule, "tool", id="stop-rule-tool-form"),
        pytest.param(_stop_condition_rule, "stopped: true", id="stop-rule-tool-stopped-field"),
        pytest.param(_stop_condition_rule, "reason", id="stop-rule-tool-reason-field"),
        pytest.param(_tools_paragraph, "`gymrat iterate`", id="tools-names-iterate-command"),
        pytest.param(_tools_paragraph, "`gymrat probe`", id="tools-names-probe-command"),
        pytest.param(_tools_paragraph, "through Bash", id="tools-names-shell-form"),
        pytest.param(_tools_paragraph, "refused", id="tools-command-form-refused"),
        pytest.param(_tools_paragraph, "only form", id="tools-only-form"),
        pytest.param(_nested_supervise_paragraph, "nested", id="supervised-mode-names-nesting"),
        pytest.param(_nested_supervise_paragraph, "refused", id="supervised-mode-nesting-refused"),
        pytest.param(_concurrent_sessions_rule, "nested", id="concurrent-rule-names-nesting"),
        pytest.param(_concurrent_sessions_rule, "refused", id="concurrent-rule-nesting-refused"),
        pytest.param(
            _iteration_cycle,
            "Never pass `--bench` or `--samples` to `iterate`.",
            id="iteration-cycle-bench-rule",
        ),
        pytest.param(
            _red_flags,
            "About to pass `--bench` or `--samples` to `iterate`.",
            id="red-flag-bench-rule",
        ),
        pytest.param(
            _iteration_cycle,
            "Edit code **only in the experiment worktree**",
            id="iteration-cycle-worktree-rule",
        ),
        pytest.param(
            _outside_edit_paragraph, "Edit, Write, MultiEdit, or NotebookEdit", id="edit-tools"
        ),
        pytest.param(
            _outside_edit_paragraph, "outside the experiment worktree", id="edit-outside-worktree"
        ),
        pytest.param(_outside_edit_paragraph, "temporary directories", id="edit-temp-directories"),
        pytest.param(_outside_edit_paragraph, "names its rule", id="edit-refusal-names-rule"),
        pytest.param(_outside_edit_paragraph, "change the call", id="edit-fix-changes-call"),
        pytest.param(_outside_edit_paragraph, "not to retry it", id="edit-fix-is-not-retry"),
        pytest.param(
            _background_paragraph, "contains `gymrat` is refused", id="background-command-refused"
        ),
        pytest.param(_sync_paragraph, "Under supervise", id="sync-names-supervised-mode"),
        pytest.param(_sync_paragraph, "refused", id="sync-main-tree-edit-refused"),
        pytest.param(
            _sync_paragraph,
            "make the change in the experiment worktree",
            id="sync-edit-in-worktree",
        ),
        pytest.param(_sync_paragraph, "a person's main-tree edits", id="sync-for-person-edits"),
    ],
)
def test_read_bundled_skill_when_passage_read_does_state_its_rule(
    passage: Callable[[], str],
    phrase: str,
):
    text = passage()

    assert phrase in text
