import re
import zipfile
from collections.abc import Callable
from functools import partial
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import NoReturn
from unittest.mock import create_autospec

import pytest

from gymrat.bundled_skill import read_bundled_skill
from gymrat.errors import GymratError
from tests._ansi import normalize

SKILL_HEADING = "# Driving a gymrat optimization session"


def _assert_reinstall_error(error: GymratError, cause_type: type[BaseException]) -> None:
    """Assert the failure's SKILL.md message, reinstall hint, and cause."""
    assert "SKILL.md" in str(error)
    assert error.hint is not None
    assert re.search("reinstall", error.hint, re.IGNORECASE)
    assert isinstance(error.__cause__, cause_type)


def _install_package_root(monkeypatch: pytest.MonkeyPatch, root: Traversable) -> None:
    """Make the package-data lookup answer with ``root`` as the package root."""

    def package_root(_package: str) -> Traversable:
        return root

    monkeypatch.setattr(resources, "files", package_root)


# ---------------------------------------------------------------------------
# read_bundled_skill failures
# ---------------------------------------------------------------------------


def _skill_path(package_root: Path) -> Path:
    """Where the skill file sits under a package root on disk."""
    return package_root / "skills" / "gymrat" / "SKILL.md"


def _missing_file(package_root: Path) -> tuple[Traversable, Traversable]:
    """Build a real package root that holds no skill file.

    Args:
        package_root: Empty directory standing in for the installed package.

    Returns:
        The package root and the skill file location under it.
    """
    return package_root, _skill_path(package_root)


def _undecodable_file(package_root: Path) -> tuple[Traversable, Traversable]:
    """Build a real package root whose skill file is not valid UTF-8.

    Args:
        package_root: Empty directory standing in for the installed package.

    Returns:
        The package root and the skill file location under it.
    """
    skill = _skill_path(package_root)
    skill.parent.mkdir(parents=True)
    skill.write_bytes(b"\xff")
    return package_root, skill


def _corrupt_archive(_package_root: Path) -> tuple[Traversable, Traversable]:
    """Build a package root inside an archive that fails on read.

    Every traversal from the root, by ``joinpath`` or by ``/``, lands on the one resource.

    Args:
        _package_root: Unused; an archive has no directory on disk.

    Returns:
        The same resource twice: it is its own root and its own resolved location.
    """
    resource = create_autospec(Traversable, instance=True)
    resource.joinpath.return_value = resource
    resource.__truediv__.return_value = resource
    resource.read_text.side_effect = zipfile.BadZipFile("Bad magic number")
    return resource, resource


@pytest.mark.parametrize(
    ("build_package", "cause_type"),
    [
        pytest.param(_missing_file, FileNotFoundError, id="missing-file"),
        pytest.param(_undecodable_file, UnicodeDecodeError, id="bad-encoding"),
        pytest.param(_corrupt_archive, zipfile.BadZipFile, id="corrupt-archive"),
    ],
)
def test_read_bundled_skill_when_read_fails_does_raise_naming_the_resolved_location(
    build_package: Callable[[Path], tuple[Traversable, Traversable]],
    cause_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root, resource = build_package(tmp_path)
    _install_package_root(monkeypatch, root)

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
    paragraphs = (normalize(item) for item in _section(heading).split("\n\n"))
    return next(item for item in paragraphs if marker in item)


def _unwrapped_section(heading: str) -> str:
    """The skill text under ``heading`` with every line break and run of spaces collapsed."""
    return normalize(_section(heading))


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
def test_read_bundled_skill_when_section_inspected_does_state_its_rule(
    passage: Callable[[], str],
    phrases: tuple[str, ...],
):
    text = passage()

    missing = [phrase for phrase in phrases if phrase not in text]
    assert missing == []
