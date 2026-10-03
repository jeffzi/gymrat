"""Behavioral tests for the generic helpers in :mod:`gymrat.utils`.

``write_text_atomic`` writes UTF-8 text to a temporary sibling, flushes it and
syncs it to disk, then renames it over the target, so a reader sees either the
previous content or the full new content — never a partial file. On failure
the target is untouched, the temporary file is removed, and the ``OSError``
propagates.
"""

import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path, PureWindowsPath

import pytest

from gymrat.utils import abbreviate_home, fan_out, pluralize, warn_to_stderr, write_text_atomic

OLD_TEXT = "previous content\n"
NEW_TEXT = "café ✓ new content\n"


# ---------------------------------------------------------------------------
# abbreviate_home
# ---------------------------------------------------------------------------


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
    monkeypatch.setattr("gymrat.utils.Path", _WindowsPath)

    assert abbreviate_home(r"C:\Users\ada\work\repo") == "~/work/repo"


# ---------------------------------------------------------------------------
# pluralize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "noun",
    [
        "file",
        "metric",
        "iteration",
        "keep",
        "sample",
        "worktree",
        "edit",
        "warning",
        "failure",
        "kept iteration",
        "uncommitted file",
    ],
)
def test_pluralize_when_count_is_plural_does_append_s(noun: str):
    assert pluralize(2, noun) == f"2 {noun}s"


@pytest.mark.parametrize(
    ("noun", "expected"),
    [
        pytest.param("pass", "2 passes", id="ends-in-s"),
        pytest.param("box", "2 boxes", id="ends-in-x"),
        pytest.param("buzz", "2 buzzes", id="ends-in-z"),
        pytest.param("branch", "2 branches", id="ends-in-ch"),
        pytest.param("dish", "2 dishes", id="ends-in-sh"),
        pytest.param("query", "2 queries", id="consonant-then-y"),
        pytest.param("key", "2 keys", id="vowel-then-y"),
    ],
)
def test_pluralize_when_count_is_plural_does_apply_english_suffix_rules(noun: str, expected: str):
    assert pluralize(2, noun) == expected


@pytest.mark.parametrize("noun", ["pass", "query", "box", "metric", "kept iteration"])
def test_pluralize_when_count_is_one_does_leave_noun_unchanged(noun: str):
    assert pluralize(1, noun) == f"1 {noun}"


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        pytest.param(0, "0 passes", id="zero"),
        pytest.param(2, "2 passes", id="many"),
        pytest.param(-1, "-1 passes", id="negative"),
    ],
)
def test_pluralize_when_count_is_not_one_does_use_the_plural_form(count: int, expected: str):
    assert pluralize(count, "pass") == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        pytest.param(1, "1 index", id="singular-keeps-noun"),
        pytest.param(2, "2 indices", id="plural-takes-override"),
        pytest.param(0, "0 indices", id="zero-takes-override"),
    ],
)
def test_pluralize_when_plural_given_does_override_the_suffix_rules(count: int, expected: str):
    assert pluralize(count, "index", "indices") == expected


# ---------------------------------------------------------------------------
# warn_to_stderr
# ---------------------------------------------------------------------------


def test_warn_to_stderr_when_called_does_write_message_with_newline_to_stderr(
    capsys: pytest.CaptureFixture[str],
):
    warn_to_stderr("hello")

    captured = capsys.readouterr()
    assert captured.err == "hello\n"
    assert captured.out == ""


# ---------------------------------------------------------------------------
# fan_out
# ---------------------------------------------------------------------------


class _Event:
    """A distinct object whose identity the subscribers can check."""


def _raise_boom(_: _Event) -> None:
    msg = "boom"
    raise RuntimeError(msg)


def test_fan_out_when_called_does_dispatch_identical_event_to_subscribers_in_order():
    calls: list[tuple[str, _Event]] = []
    errors: list[Exception] = []

    def first(event: _Event) -> None:
        calls.append(("first", event))

    def second(event: _Event) -> None:
        calls.append(("second", event))

    dispatch = fan_out([first, second], errors.append)
    event = _Event()

    dispatch(event)

    assert [name for name, _ in calls] == ["first", "second"]
    assert all(received is event for _, received in calls)
    assert errors == []


def test_fan_out_when_subscriber_raises_does_report_error_and_call_remaining():
    received: list[_Event] = []
    errors: list[Exception] = []
    dispatch = fan_out([_raise_boom, received.append], errors.append)
    event = _Event()

    dispatch(event)

    assert received == [event]
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "boom"


def test_fan_out_when_no_subscribers_does_not_report_errors():
    errors: list[Exception] = []
    dispatch = fan_out([], errors.append)

    dispatch(_Event())

    assert errors == []


# ---------------------------------------------------------------------------
# write_text_atomic: successful write
# ---------------------------------------------------------------------------


@pytest.fixture
def target(tmp_path: Path) -> Iterator[Path]:
    """A target file holding ``OLD_TEXT``; its directory is made writable again on teardown."""
    path = tmp_path / "state.json"
    path.write_text(OLD_TEXT, encoding="utf-8")
    yield path
    tmp_path.chmod(0o755)


def _entries(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


@pytest.mark.parametrize(
    "previous",
    [
        pytest.param(None, id="target-absent"),
        pytest.param(OLD_TEXT, id="target-exists"),
    ],
)
def test_write_text_atomic_when_called_does_write_utf8_text_without_leftover_temp(
    tmp_path: Path, previous: str | None
):
    path = tmp_path / "state.json"
    if previous is not None:
        path.write_text(previous, encoding="utf-8")

    write_text_atomic(path, NEW_TEXT)

    assert path.read_bytes() == NEW_TEXT.encode("utf-8")
    assert _entries(tmp_path) == ["state.json"]


def test_write_text_atomic_when_renaming_does_swap_in_a_fully_synced_file_over_untouched_target(
    target: Path, monkeypatch: pytest.MonkeyPatch
):
    real_fsync = os.fsync
    real_replace = os.replace
    synced_sizes: dict[int, int] = {}
    at_rename: dict[str, object] = {}

    def recording_fsync(fd: int) -> None:
        real_fsync(fd)
        stat = os.fstat(fd)
        synced_sizes[stat.st_ino] = stat.st_size

    def observing_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        at_rename["size_when_synced"] = synced_sizes.get(Path(src).stat().st_ino)
        at_rename["source_text"] = Path(src).read_text(encoding="utf-8")
        at_rename["target_text"] = Path(dst).read_text(encoding="utf-8")
        real_replace(src, dst)

    monkeypatch.setattr("os.fsync", recording_fsync)
    monkeypatch.setattr("os.replace", observing_replace)

    write_text_atomic(target, NEW_TEXT)

    assert at_rename == {
        "size_when_synced": len(NEW_TEXT.encode("utf-8")),
        "source_text": NEW_TEXT,
        "target_text": OLD_TEXT,
    }


# ---------------------------------------------------------------------------
# write_text_atomic: failed write
# ---------------------------------------------------------------------------


def _make_directory_read_only(directory: Path, _monkeypatch: pytest.MonkeyPatch) -> str:
    directory.chmod(0o555)
    return "Permission denied"


def _fail_fsync(_directory: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    def exploding_fsync(fd: int) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr("os.fsync", exploding_fsync)
    return "disk full"


def _fail_replace(_directory: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    def exploding_replace(src: object, dst: object) -> None:
        msg = "Read-only file system"
        raise OSError(msg)

    monkeypatch.setattr("os.replace", exploding_replace)
    return "Read-only file system"


@pytest.mark.parametrize(
    "break_write",
    [
        pytest.param(
            _make_directory_read_only,
            id="read-only-directory",
            marks=pytest.mark.skipif(
                sys.platform == "win32" or os.geteuid() == 0,
                reason="POSIX file modes, not enforced for root",
            ),
        ),
        pytest.param(_fail_fsync, id="fsync-fails"),
        pytest.param(_fail_replace, id="rename-fails"),
    ],
)
def test_write_text_atomic_when_write_fails_does_raise_and_leave_target_untouched(
    target: Path,
    monkeypatch: pytest.MonkeyPatch,
    break_write: Callable[[Path, pytest.MonkeyPatch], str],
):
    reason = break_write(target.parent, monkeypatch)

    with pytest.raises(OSError, match=reason):
        write_text_atomic(target, NEW_TEXT)

    assert target.read_text(encoding="utf-8") == OLD_TEXT
    assert _entries(target.parent) == ["state.json"]
