"""Behavioral tests for duration/ETA formatting and EtaTracker removal."""

import importlib

import pytest

from gymrat_py.eta import format_duration, format_eta

# ---------------------------------------------------------------------------
# EtaTracker removal
# ---------------------------------------------------------------------------


def test_eta_tracker_when_imported_does_raise_import_error() -> None:
    with pytest.raises(ImportError):
        from gymrat_py.eta import EtaTracker  # type: ignore[missing-module-attribute]  # noqa: F401


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
    eta = importlib.import_module("gymrat_py.eta")

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
