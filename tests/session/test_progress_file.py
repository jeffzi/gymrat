"""Behavioral tests for the progress sidecar file (write / read / clear).

The sidecar carries a JSON snapshot that a dashboard or supervisor polls via
``read_progress``.  ``write_progress`` writes atomically so readers never see a
partial file.  ``clear_progress`` removes the sidecar when the iteration exits.
``create_sidecar_writer`` returns a callback that translates ``PassStarted`` /
``PassFinished`` events into sidecar writes.
"""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from gymrat.progress_events import (
    HookStarted,
    PassFinished,
    PassStarted,
    PrepareStarted,
)
from gymrat.session.paths import progress_path, session_dir
from gymrat.session.progress_file import (
    STALENESS_BOUND_SECONDS,
    ProgressSnapshot,
    clear_progress,
    create_sidecar_writer,
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
        "passes_completed": 3,
        "passes_total": 10,
        "last_pass_duration_ms": 1234.5,
    }
    defaults.update(overrides)
    return ProgressSnapshot(**defaults)  # type: ignore[arg-type]


def _progress_file(root: str) -> Path:
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
    write_progress(root, _make_snapshot(passes_completed=1))
    write_progress(root, _make_snapshot(passes_completed=2))

    assert _read_json(root)["passes_completed"] == 2


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


@pytest.mark.parametrize(
    "exception",
    [
        pytest.param(
            FileNotFoundError(2, "No such file", "progress.json"),
            id="file-vanishes-mid-read",
        ),
        pytest.param(
            PermissionError(13, "Permission denied", "progress.json"),
            id="permission-denied",
        ),
    ],
)
def test_read_progress_when_read_text_raises_os_error_does_return_none(
    root: str, monkeypatch: pytest.MonkeyPatch, exception: OSError
):
    write_progress(root, _make_snapshot())
    original_read_text = Path.read_text

    def failing_read(self: Path, *args: object, **kwargs: object) -> str:
        if str(self) == str(_progress_file(root)):
            raise exception
        return original_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", failing_read)

    result = read_progress(root)

    assert result is None


def test_read_progress_when_file_contains_non_utf8_bytes_does_return_none(root: str):
    _progress_file(root).write_bytes(b"\x80\x81\x82")

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
        snapshot.passes_completed = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# create_sidecar_writer
# ---------------------------------------------------------------------------


def test_create_sidecar_writer_when_pass_started_does_write_snapshot_with_zero_completed(
    root: str,
):
    writer = create_sidecar_writer(root)
    event = PassStarted(
        round=1,
        total_rounds=5,
        target_count=2,
        label="baseline",
        at_ms=100.0,
        phase="measure",
    )

    writer(event)

    snapshot = read_progress(root)
    assert snapshot is not None
    assert snapshot.passes_completed == 0
    assert snapshot.passes_total == 10


def test_create_sidecar_writer_when_pass_finished_does_increment_completed_and_record_duration(
    root: str,
):
    writer = create_sidecar_writer(root)
    writer(
        PassStarted(
            round=1,
            total_rounds=3,
            target_count=2,
            label="experiment",
            at_ms=100.0,
        )
    )

    writer(
        PassFinished(
            round=1,
            total_rounds=3,
            target_count=2,
            label="experiment",
            at_ms=350.0,
        )
    )

    snapshot = read_progress(root)
    assert snapshot is not None
    assert snapshot.passes_completed == 1
    assert snapshot.last_pass_duration_ms == 250.0


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            PrepareStarted(label="baseline", at_ms=100.0),
            id="prepare-started",
        ),
        pytest.param(
            HookStarted(stage="before", at_ms=100.0),
            id="hook-started",
        ),
    ],
)
def test_create_sidecar_writer_when_non_pass_event_does_not_write(
    root: str,
    event: object,
):
    writer = create_sidecar_writer(root)

    writer(event)  # type: ignore[arg-type]

    assert read_progress(root) is None


def test_create_sidecar_writer_when_confirm_follows_measure_does_reset_passes_completed(
    root: str,
):
    writer = create_sidecar_writer(root)
    # Complete all measure passes: 2 rounds * 1 target = 2 total
    writer(
        PassStarted(
            round=1, total_rounds=2, target_count=1, label="x", at_ms=100.0, phase="measure"
        )
    )
    writer(
        PassFinished(
            round=1, total_rounds=2, target_count=1, label="x", at_ms=150.0, phase="measure"
        )
    )
    writer(
        PassStarted(
            round=2, total_rounds=2, target_count=1, label="x", at_ms=200.0, phase="measure"
        )
    )
    writer(
        PassFinished(
            round=2, total_rounds=2, target_count=1, label="x", at_ms=250.0, phase="measure"
        )
    )

    # Confirm phase starts: passes_completed must reset to 0, not carry over
    writer(
        PassStarted(
            round=1, total_rounds=2, target_count=1, label="x", at_ms=300.0, phase="confirm"
        )
    )

    snapshot = read_progress(root)
    assert snapshot is not None
    assert snapshot.passes_completed == 0
    assert snapshot.passes_total == 2
