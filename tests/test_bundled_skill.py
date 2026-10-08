import re
import zipfile
from collections.abc import Callable
from functools import partial
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

LOOP_DISCIPLINE = "## Loop discipline"
SUPERVISED_MODE = "### Supervised mode"


def _section(heading: str) -> str:
    """The skill text under ``heading``, up to the next top-level section."""
    return read_bundled_skill().partition(heading)[2].partition("\n## ")[0]


def _paragraph(heading: str, marker: str) -> str:
    """The paragraph (or list item) under ``heading`` containing ``marker``, unwrapped to one line."""
    paragraphs = (" ".join(item.split()) for item in _section(heading).split("\n\n"))
    return next(item for item in paragraphs if marker in item)


def _unwrapped_section(heading: str) -> str:
    """The skill text under ``heading`` with every line break and run of spaces collapsed."""
    return " ".join(_section(heading).split())


@pytest.mark.parametrize(
    ("passage", "phrases"),
    [
        pytest.param(read_bundled_skill, (SKILL_HEADING,), id="whole-file"),
        pytest.param(
            partial(_paragraph, LOOP_DISCIPLINE, "Never stop before"),
            ("`iterate`", "exits 1", "tool", "stopped: true", "reason"),
            id="stop-rule",
        ),
        pytest.param(
            partial(_paragraph, SUPERVISED_MODE, "Use the `probe` and `iterate` tools."),
            ("`gymrat iterate`", "`gymrat probe`", "through Bash", "refused", "only form"),
            id="tools",
        ),
        pytest.param(
            partial(_paragraph, SUPERVISED_MODE, "Never run `gymrat supervise` yourself."),
            ("nested", "refused"),
            id="nested-supervise",
        ),
        pytest.param(
            partial(_paragraph, LOOP_DISCIPLINE, "Never run concurrent sessions."),
            ("nested", "refused"),
            id="concurrent-rule",
        ),
        pytest.param(
            partial(_unwrapped_section, "### 3. The iteration cycle"),
            (
                "Never pass `--bench` or `--samples` to `iterate`.",
                "Edit code **only in the experiment worktree**",
            ),
            id="iteration-cycle",
        ),
        pytest.param(
            partial(_unwrapped_section, "## Red flags"),
            ("About to pass `--bench` or `--samples` to `iterate`.",),
            id="red-flags",
        ),
        pytest.param(
            partial(
                _paragraph,
                SUPERVISED_MODE,
                "**A file edit outside the experiment worktree is refused.**",
            ),
            (
                "Edit, Write, MultiEdit, or NotebookEdit",
                "outside the experiment worktree",
                "temporary directories",
                "names its rule",
                "change the call",
                "not to retry it",
            ),
            id="outside-edit",
        ),
        pytest.param(
            partial(
                _paragraph, SUPERVISED_MODE, "**Never run a gymrat command in the background.**"
            ),
            ("contains `gymrat` is refused",),
            id="background",
        ),
        pytest.param(
            partial(_paragraph, "## Syncing main-tree edits", "Under supervise"),
            (
                "Under supervise",
                "refused",
                "make the change in the experiment worktree",
                "a person's main-tree edits",
            ),
            id="sync",
        ),
    ],
)
def test_read_bundled_skill_when_passage_read_does_state_its_rule(
    passage: Callable[[], str],
    phrases: tuple[str, ...],
):
    text = passage()

    missing = [phrase for phrase in phrases if phrase not in text]
    assert missing == []
