"""Behavioral tests for the progress sidecar file (write / read).

The sidecar carries a JSON snapshot that a dashboard or supervisor polls via
``read_progress``.  ``write_progress`` writes atomically so readers never see a
partial file.  ``SidecarWriter`` is a callback that translates ``PassStarted`` /
``PassFinished`` events into sidecar writes.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from gymrat.progress_events import (
    PassFinished,
    PassStarted,
    PrepareStarted,
)
from gymrat.session.paths import progress_path, session_dir
from gymrat.session.progress_file import (
    ProgressSnapshot,
    SidecarWriter,
    read_progress,
    write_progress,
)
from tests._clock import freeze_now_ms
from tests._mode_bits import needs_mode_bits


def _make_snapshot(
    *, passes_completed: int = 3, passes_total: int = 10, last_pass_duration_ms: float = 1234.5
) -> ProgressSnapshot:
    """Build a ProgressSnapshot with sensible defaults, overridable per-field."""
    return ProgressSnapshot(
        passes_completed=passes_completed,
        passes_total=passes_total,
        last_pass_duration_ms=last_pass_duration_ms,
    )


def _progress_file(root: str) -> Path:
    return Path(progress_path(root))


# ---------------------------------------------------------------------------
# read_progress
# ---------------------------------------------------------------------------


def test_read_progress_when_file_absent_does_return_none(root: str):
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
    freeze_now_ms(monkeypatch, written_ms + age_seconds * 1000)

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


def _start_and_finish_one_pass(writer: SidecarWriter) -> None:
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


def test_sidecar_writer_when_pass_finished_does_increment_passes_completed(root: str):
    writer = SidecarWriter(root)

    _start_and_finish_one_pass(writer)

    snapshot = read_progress(root)
    assert snapshot is not None
    assert snapshot.passes_completed == 1


def test_sidecar_writer_when_pass_finished_does_record_the_pass_duration(root: str):
    writer = SidecarWriter(root)

    _start_and_finish_one_pass(writer)

    snapshot = read_progress(root)
    assert snapshot is not None
    assert snapshot.last_pass_duration_ms == 250.0


def test_sidecar_writer_when_non_pass_event_does_not_write(root: str):
    writer = SidecarWriter(root)

    writer(PrepareStarted(label="baseline", at_ms=100.0))

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

    writer(
        PassStarted(
            round=1, total_rounds=2, target_count=1, label="x", at_ms=300.0, phase="confirm"
        )
    )

    snapshot = read_progress(root)
    assert snapshot is not None
    assert snapshot.passes_completed == 0
