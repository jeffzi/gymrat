import zipfile
from collections.abc import Callable
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import NoReturn
from unittest.mock import create_autospec

import pytest

from gymrat.bundled_skill import read_bundled_skill
from gymrat.errors import GymratError


def _install_package_root(monkeypatch: pytest.MonkeyPatch, root: Traversable) -> None:
    """Make the package-data lookup answer with ``root`` as the package root."""

    def package_root(_package: str) -> Traversable:
        return root

    monkeypatch.setattr(
        resources, "files", create_autospec(resources.files, side_effect=package_root)
    )


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

    assert isinstance(caught.value.__cause__, cause_type)
    assert str(resource) in str(caught.value)


def test_read_bundled_skill_when_resource_lookup_fails_does_raise_without_a_location(
    monkeypatch: pytest.MonkeyPatch,
):
    def unresolvable_package(_package: str) -> NoReturn:
        message = "Bad magic number"
        raise zipfile.BadZipFile(message)

    monkeypatch.setattr(
        resources, "files", create_autospec(resources.files, side_effect=unresolvable_package)
    )

    with pytest.raises(GymratError) as caught:
        read_bundled_skill()

    assert isinstance(caught.value.__cause__, zipfile.BadZipFile)
    assert "resolved to" not in str(caught.value)
