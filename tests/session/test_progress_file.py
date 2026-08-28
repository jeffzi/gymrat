"""Behavioral tests for the progress sidecar file (write / read / clear).

The sidecar carries a JSON snapshot that a dashboard or supervisor polls via
``read_progress``.  ``write_progress`` writes atomically so readers never see a
partial file.  ``clear_progress`` removes the sidecar when the iteration exits.
"""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from gymrat_py.session.paths import progress_path, session_dir
from gymrat_py.session.progress_file import (
    STALENESS_BOUND_SECONDS,
    ProgressSnapshot,
    clear_progress,
    read_progress,
    write_progress,
)


@pytest.fixture
def root(tmp_path: Path) -> str:
    """A fake repo root with the .gymrat session directory pre-created."""
    session = tmp_path / ".gymrat"
    session.mkdir()
    return str(tmp_path)


# ---------------------------------------------------------------------------
# progress_path
# ---------------------------------------------------------------------------


def test_progress_path_when_given_root_does_place_file_under_session_dir(
    root: str,
):
    result = progress_path(root)

    expected = str(Path(session_dir(root)) / "progress.json")
    assert result == expected


def test_progress_path_when_given_root_does_not_place_file_under_worktrees(
    root: str,
):
    result = progress_path(root)

    assert "worktrees" not in result


# ---------------------------------------------------------------------------
# write_progress
# ---------------------------------------------------------------------------


def _make_snapshot(**overrides: object) -> ProgressSnapshot:
    """Build a ProgressSnapshot with sensible defaults, overridable per-field."""
    defaults: dict[str, object] = {
        "seq": 1,
        "phase": "measure",
        "passes_completed": 3,
        "passes_total": 10,
        "current_side": "experiment",
        "current_round": 1,
        "last_pass_duration_ms": 1234.5,
        "started_at": 1700000000.0,
    }
    defaults.update(overrides)
    return ProgressSnapshot(**defaults)  # type: ignore[arg-type]


def _progress_file(root: str) -> Path:
    """The sidecar path under *root* as a ``Path``."""
    return Path(progress_path(root))


def _read_json(root: str) -> dict[str, object]:
    """Read and parse the raw sidecar JSON under *root*."""
    return json.loads(_progress_file(root).read_text(encoding="utf-8"))


def test_write_progress_when_called_does_create_readable_json_file(root: str):
    snapshot = _make_snapshot()

    write_progress(root, snapshot)

    assert _read_json(root) == asdict(snapshot)


def test_write_progress_when_called_twice_does_overwrite_previous_snapshot(
    root: str,
):
    write_progress(root, _make_snapshot(seq=1))
    write_progress(root, _make_snapshot(seq=2))

    assert _read_json(root)["seq"] == 2


def test_write_progress_when_snapshot_has_none_side_does_serialize_null(
    root: str,
):
    write_progress(root, _make_snapshot(current_side=None))

    assert _read_json(root)["current_side"] is None


# ---------------------------------------------------------------------------
# read_progress
# ---------------------------------------------------------------------------


def test_read_progress_when_file_exists_does_return_snapshot(root: str):
    original = _make_snapshot()
    write_progress(root, original)

    result = read_progress(root)

    assert result == original


def test_read_progress_when_file_absent_does_return_none(root: str):
    result = read_progress(root)

    assert result is None


def test_read_progress_when_file_contains_invalid_json_does_return_none(
    root: str,
):
    _progress_file(root).write_text("not valid json{{{", encoding="utf-8")

    result = read_progress(root)

    assert result is None


def test_read_progress_when_file_contains_wrong_schema_does_return_none(
    root: str,
):
    _progress_file(root).write_text(json.dumps({"unexpected_field": 42}), encoding="utf-8")

    result = read_progress(root)

    assert result is None


def test_read_progress_when_file_is_stale_does_return_none(root: str):
    write_progress(root, _make_snapshot())
    path = _progress_file(root)
    stale_time = time.time() - STALENESS_BOUND_SECONDS - 60
    os.utime(path, (stale_time, stale_time))

    result = read_progress(root)

    assert result is None


# ---------------------------------------------------------------------------
# clear_progress
# ---------------------------------------------------------------------------


def test_clear_progress_when_file_exists_does_remove_it(root: str):
    write_progress(root, _make_snapshot())
    assert _progress_file(root).exists()

    clear_progress(root)

    assert not _progress_file(root).exists()


def test_clear_progress_when_file_absent_does_not_raise(root: str):
    assert not _progress_file(root).exists()

    clear_progress(root)


# ---------------------------------------------------------------------------
# ProgressSnapshot
# ---------------------------------------------------------------------------


def test_progress_snapshot_when_constructed_does_be_frozen():
    snapshot = _make_snapshot()

    with pytest.raises(AttributeError):
        snapshot.seq = 99  # type: ignore[misc]
