"""Behavioral tests for the shared atomic text writer.

``write_text_atomic`` writes UTF-8 text to a temporary sibling, flushes it and
syncs it to disk, then renames it over the target, so a reader sees either the
previous content or the full new content — never a partial file. On failure
the target is untouched, the temporary file is removed, and the ``OSError``
propagates.
"""

import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from gymrat.atomic_write import write_text_atomic

OLD_TEXT = "previous content\n"
NEW_TEXT = "café ✓ new content\n"


@pytest.fixture
def target(tmp_path: Path) -> Iterator[Path]:
    """A target file holding ``OLD_TEXT``; its directory is made writable again on teardown."""
    path = tmp_path / "state.json"
    path.write_text(OLD_TEXT, encoding="utf-8")
    yield path
    tmp_path.chmod(0o755)


def _entries(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


# ---------------------------------------------------------------------------
# successful write
# ---------------------------------------------------------------------------


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
# failed write
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
