import re
import zipfile
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


def test_read_bundled_skill_when_file_has_bad_encoding_does_raise_gymrat_error(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_resource = create_autospec(Traversable, instance=True)
    mock_resource.read_text.side_effect = UnicodeDecodeError(
        "utf-8", b"\xff", 0, 1, "invalid start byte"
    )
    monkeypatch.setattr("gymrat.bundled_skill._skill_resource", lambda: mock_resource)

    with pytest.raises(GymratError) as caught:
        read_bundled_skill()

    _assert_reinstall_error(caught.value, UnicodeDecodeError)


def test_read_bundled_skill_when_archive_corrupt_does_raise_gymrat_error(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_resource = create_autospec(Traversable, instance=True)
    mock_resource.read_text.side_effect = zipfile.BadZipFile("Bad magic number")
    monkeypatch.setattr("gymrat.bundled_skill._skill_resource", lambda: mock_resource)

    with pytest.raises(GymratError) as caught:
        read_bundled_skill()

    _assert_reinstall_error(caught.value, zipfile.BadZipFile)


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
