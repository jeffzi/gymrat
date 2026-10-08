"""Behavioral tests for the progress sidecar file (write / read / clear).

The sidecar carries a JSON snapshot that a dashboard or supervisor polls via
``read_progress``.  ``write_progress`` writes atomically so readers never see a
partial file.  ``clear_progress`` removes the sidecar when the iteration exits.
``SidecarWriter`` is a callback that translates ``PassStarted`` /
``PassFinished`` events into sidecar writes.
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gymrat.progress_events import (
    HookStarted,
    PassFinished,
    PassStarted,
    PrepareStarted,
    ProgressEvent,
)
from gymrat.session.paths import progress_path, session_dir
from gymrat.session.progress_file import (
    ProgressSnapshot,
    SidecarWriter,
    clear_progress,
    read_progress,
    write_progress,
)
from tests._mode_bits import needs_mode_bits

# ---------------------------------------------------------------------------
# write_progress
# ---------------------------------------------------------------------------


def _make_snapshot(
    *, passes_completed: int = 3, passes_total: int = 10, last_pass_duration_ms: float = 1234.5
) -> ProgressSnapshot:
    """Build a ProgressSnapshot with sensible defaults, overridable per-field."""
    return ProgressSnapshot(
        passes_completed=passes_completed,
        passes_total=passes_total,
        last_pass_duration_ms=last_pass_duration_ms,
    )


_VALID_FIELDS: dict[str, object] = _make_snapshot().model_dump()


def _progress_file(root: str) -> Path:
    return Path(progress_path(root))


def test_write_progress_when_called_does_create_readable_json_file(root: str):
    snapshot = _make_snapshot()

    write_progress(root, snapshot)

    assert _progress_file(root).read_text(encoding="utf-8") == (
        '{"passes_completed":3,"passes_total":10,"last_pass_duration_ms":1234.5}'
    )


# ---------------------------------------------------------------------------
# read_progress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "original",
    [
        pytest.param(_make_snapshot(), id="typical"),
        pytest.param(
            _make_snapshot(passes_completed=0, passes_total=0, last_pass_duration_ms=0.0),
            id="all-zero",
        ),
        pytest.param(_make_snapshot(last_pass_duration_ms=250.0), id="whole-number-duration"),
    ],
)
def test_read_progress_when_file_written_does_round_trip_every_field(
    root: str, original: ProgressSnapshot
):
    write_progress(root, original)

    result = read_progress(root)

    assert result == original


def _json(payload: object) -> bytes:
    return json.dumps(payload).encode()


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param(None, id="file-absent"),
        pytest.param(_json({**_VALID_FIELDS, "unexpected_field": 42}), id="extra-unknown-key"),
        pytest.param(_json({"passes_completed": 3, "passes_total": 10}), id="missing-key"),
        pytest.param(_json({**_VALID_FIELDS, "passes_completed": "x"}), id="string-for-int"),
        pytest.param(_json({**_VALID_FIELDS, "passes_total": True}), id="bool-for-int"),
        pytest.param(_json({**_VALID_FIELDS, "passes_completed": 3.0}), id="float-for-int"),
        pytest.param(
            _json({**_VALID_FIELDS, "last_pass_duration_ms": "fast"}), id="string-for-float"
        ),
        pytest.param(_json({**_VALID_FIELDS, "last_pass_duration_ms": False}), id="bool-for-float"),
    ],
)
def test_read_progress_when_file_absent_or_not_a_snapshot_does_return_none(
    root: str, contents: bytes | None
):
    if contents is not None:
        _progress_file(root).write_bytes(contents)

    result = read_progress(root)

    assert result is None


@pytest.mark.parametrize(
    ("age_seconds", "survives"),
    [
        pytest.param(540, True, id="inside-the-ten-minute-bound"),
        pytest.param(660, False, id="past-the-ten-minute-bound"),
    ],
)
def test_read_progress_when_clock_advances_does_discard_only_past_the_bound(
    root: str, monkeypatch: pytest.MonkeyPatch, age_seconds: int, survives: bool
):
    snapshot = _make_snapshot()
    write_progress(root, snapshot)
    written_ms = _progress_file(root).stat().st_mtime * 1000
    monkeypatch.setattr("gymrat.clock.now_ms", lambda: written_ms + age_seconds * 1000)

    result = read_progress(root)

    assert result == (snapshot if survives else None)


@pytest.fixture
def unsearchable_sidecar(root: str) -> Iterator[str]:
    """A written sidecar whose session directory denies search, so the file cannot be stat'd."""
    write_progress(root, _make_snapshot())
    session = Path(session_dir(root))
    session.chmod(0o600)
    yield root
    session.chmod(0o700)


@needs_mode_bits
def test_read_progress_when_sidecar_cannot_be_stat_does_return_none(unsearchable_sidecar: str):
    result = read_progress(unsearchable_sidecar)

    assert result is None


# ---------------------------------------------------------------------------
# clear_progress
# ---------------------------------------------------------------------------


def test_clear_progress_when_file_exists_does_remove_it(root: str):
    write_progress(root, _make_snapshot())

    clear_progress(root)

    assert not _progress_file(root).exists()


def test_clear_progress_when_file_absent_does_not_warn(root: str):
    warnings: list[str] = []

    clear_progress(root, warn=warnings.append)

    assert warnings == []


@pytest.fixture
def held_open_sidecar(root: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """A written sidecar whose removal fails the way a win32 sharing violation does."""
    write_progress(root, _make_snapshot())
    sidecar = progress_path(root)
    original_unlink = os.unlink

    def failing_unlink(path: str | os.PathLike[str], *args: Any, **kwargs: Any) -> None:
        if str(path) == sidecar:
            raise PermissionError(13, "The process cannot access the file", sidecar)
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", failing_unlink)
    return root


def test_clear_progress_when_unlink_raises_os_error_does_warn_instead_of_raising(
    held_open_sidecar: str,
):
    warnings: list[str] = []

    clear_progress(held_open_sidecar, warn=warnings.append)

    assert len(warnings) == 1
    assert progress_path(held_open_sidecar) in warnings[0]


def test_clear_progress_when_unlink_raises_and_no_sink_given_does_warn_on_stderr(
    held_open_sidecar: str, capsys: pytest.CaptureFixture[str]
):
    clear_progress(held_open_sidecar)

    assert progress_path(held_open_sidecar) in capsys.readouterr().err


# ---------------------------------------------------------------------------
# SidecarWriter
# ---------------------------------------------------------------------------


def test_sidecar_writer_when_pass_started_does_write_snapshot_with_zero_completed(
    root: str,
):
    writer = SidecarWriter(root)
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


def test_sidecar_writer_when_pass_finished_does_increment_completed_and_record_duration(
    root: str,
):
    writer = SidecarWriter(root)
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
def test_sidecar_writer_when_non_pass_event_does_not_write(
    root: str,
    event: ProgressEvent,
):
    writer = SidecarWriter(root)

    writer(event)

    assert read_progress(root) is None


def test_sidecar_writer_when_confirm_follows_measure_does_reset_passes_completed(
    root: str,
):
    writer = SidecarWriter(root)
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
