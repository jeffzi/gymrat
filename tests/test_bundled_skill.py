import re
from importlib import resources

import pytest

from gymrat_py.bundled_skill import read_bundled_skill
from gymrat_py.errors import GymratError

SKILL_HEADING = "# Driving a gymrat optimization session"


# ---------------------------------------------------------------------------
# read_bundled_skill
# ---------------------------------------------------------------------------


def test_read_bundled_skill_when_packaged_file_present_does_return_its_text():
    result = read_bundled_skill()

    assert SKILL_HEADING in result


def test_read_bundled_skill_when_file_unreadable_does_raise_gymrat_error(
    monkeypatch: pytest.MonkeyPatch,
):
    missing = resources.files("gymrat_py") / "skills" / "gymrat" / "does-not-exist.md"
    monkeypatch.setattr("gymrat_py.bundled_skill._skill_resource", lambda: missing)

    with pytest.raises(GymratError) as caught:
        read_bundled_skill()

    error = caught.value
    assert "SKILL.md" in str(error)
    assert str(missing) in str(error)
    assert error.hint is not None
    assert re.search("reinstall", error.hint, re.IGNORECASE)
    assert isinstance(error.__cause__, FileNotFoundError)
