"""Behavioral tests for duration/ETA formatting and EtaTracker removal."""

import importlib

import pytest

from gymrat.eta import format_clock, format_duration, format_eta, format_timestamp

# ---------------------------------------------------------------------------
# EtaTracker removal
# ---------------------------------------------------------------------------


def test_eta_tracker_when_imported_does_raise_import_error() -> None:
    with pytest.raises(ImportError):
        from gymrat.eta import EtaTracker  # type: ignore[missing-module-attribute]  # noqa: F401


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("PassStarted", id="PassStarted"),
        pytest.param("PrepareStarted", id="PrepareStarted"),
        pytest.param("ProgressEvent", id="ProgressEvent"),
        pytest.param("default_clock", id="default_clock"),
    ],
)
def test_eta_module_when_inspected_does_not_expose_progress_event_symbol(name: str) -> None:
    eta = importlib.import_module("gymrat.eta")

    assert not hasattr(eta, name)


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        (0, "0s"),
        (999, "0s"),
        (1000, "1s"),
        (45_000, "45s"),
        (59_999, "59s"),
        (60_000, "1m 0s"),
        (90_000, "1m 30s"),
        (723_000, "12m 3s"),
        (3_599_999, "59m 59s"),
        (3_600_000, "1h 00m"),
        (3_900_000, "1h 05m"),
        (5_400_000, "1h 30m"),
        (7_200_000, "2h 00m"),
        pytest.param(36_000_000, "10h 00m", id="multi-digit-hours"),
    ],
)
def test_format_duration_when_given_milliseconds_does_render_expected_duration(
    ms: float, expected: str
) -> None:
    assert format_duration(ms) == expected


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        (-1, "0s"),
        (-1000, "0s"),
        (-999_999, "0s"),
    ],
)
def test_format_duration_when_negative_input_does_render_zero(ms: float, expected: str) -> None:
    assert format_duration(ms) == expected


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("at_ms", "run_start_ms"),
    [
        pytest.param(1_000, 1_000, id="zero-elapsed"),
        pytest.param(999, 1_000, id="negative-elapsed-clamps-to-zero"),
        pytest.param(0, 90_000, id="large-negative-elapsed-clamps-to-zero"),
    ],
)
def test_format_timestamp_when_elapsed_not_positive_does_render_zero_timestamp(
    at_ms: float, run_start_ms: float
) -> None:
    assert format_timestamp(at_ms, run_start_ms) == "[00:00:00]"


@pytest.mark.parametrize(
    ("at_ms", "run_start_ms", "expected"),
    [
        pytest.param(91_000, 1_000, "[00:01:30]", id="ninety-seconds-elapsed"),
        pytest.param(3_601_000, 1_000, "[01:00:00]", id="one-hour-elapsed"),
        pytest.param(36_001_000, 1_000, "[10:00:00]", id="multi-digit-hours-elapsed"),
    ],
)
def test_format_timestamp_when_positive_elapsed_does_render_elapsed_clock(
    at_ms: float, run_start_ms: float, expected: str
) -> None:
    assert format_timestamp(at_ms, run_start_ms) == expected


# ---------------------------------------------------------------------------
# format_clock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        pytest.param(0, "00:00", id="zero"),
        pytest.param(9_000, "00:09", id="sub-minute"),
        pytest.param(9_999, "00:09", id="floors-partial-second"),
        pytest.param(59_999, "00:59", id="last-second-before-minute-tier"),
        pytest.param(465_000, "07:45", id="minutes"),
        pytest.param(3_599_000, "59:59", id="last-second-before-hour-tier"),
        pytest.param(3_600_000, "1:00:00", id="hour-tier-starts"),
        pytest.param(4_065_000, "1:07:45", id="hour-tier"),
        pytest.param(36_000_000, "10:00:00", id="multi-digit-hours"),
        pytest.param(-1, "00:00", id="negative-clamps-to-zero"),
        pytest.param(-90_000, "00:00", id="large-negative-clamps-to-zero"),
    ],
)
def test_format_clock_when_given_milliseconds_does_render_expected_clock(
    ms: float, expected: str
) -> None:
    assert format_clock(ms) == expected


# ---------------------------------------------------------------------------
# format_eta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        (0, "~1s left"),
        (500, "~1s left"),
        (999, "~1s left"),
        (1000, "~1s left"),
        (48200, "~48s left"),
        (59999, "~1m left"),
        (60000, "~1m left"),
        (130000, "~2m 10s left"),
        (120000, "~2m left"),
        (3599999, "~1h left"),
        (3600000, "~1h left"),
        (3900000, "~1h 05m left"),
        (7200000, "~2h left"),
        (7260000, "~2h 01m left"),
    ],
)
def test_format_eta_when_given_milliseconds_does_render_expected_eta(
    ms: float, expected: str
) -> None:
    assert format_eta(ms) == expected
