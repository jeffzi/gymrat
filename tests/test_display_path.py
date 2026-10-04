"""Tests for :mod:`gymrat.display_path`, which abbreviates the home directory."""

from pathlib import Path, PureWindowsPath

import pytest

from gymrat.display_path import abbreviate_home


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        pytest.param(".", "~", id="home-itself"),
        pytest.param("work/repo", "~/work/repo", id="under-home"),
    ],
)
def test_abbreviate_home_when_path_under_home_does_use_a_tilde(
    relative: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert abbreviate_home(str(tmp_path / relative)) == expected


def test_abbreviate_home_when_path_outside_home_does_return_it_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    outside = str(tmp_path / "elsewhere" / "repo")

    assert abbreviate_home(outside) == outside


def test_abbreviate_home_when_home_cannot_be_found_does_return_path_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    def no_home() -> Path:
        msg = "Could not determine home directory."
        raise RuntimeError(msg)

    monkeypatch.setattr(Path, "home", no_home)

    assert abbreviate_home("/srv/repo") == "/srv/repo"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param("work/repo", "work/repo", id="relative"),
        pytest.param("{home}/../elsewhere", "~/../elsewhere", id="dot-dot-kept-lexically"),
    ],
)
def test_abbreviate_home_when_path_is_not_normalized_does_compare_it_as_written(
    path: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    assert abbreviate_home(path.format(home=tmp_path)) == expected


class _WindowsPath(PureWindowsPath):
    """A Windows-flavoured path whose home is a fixed user directory."""

    @classmethod
    def home(cls) -> "_WindowsPath":
        """The fixed home directory every host sees."""
        return cls(r"C:\Users\ada")


def test_abbreviate_home_when_windows_path_under_home_does_use_forward_slashes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("gymrat.display_path.Path", _WindowsPath)

    assert abbreviate_home(r"C:\Users\ada\work\repo") == "~/work/repo"
