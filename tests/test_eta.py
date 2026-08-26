"""Behavioral tests for the ETA tracker and duration/ETA formatting."""

from collections.abc import Callable

import pytest

from gymrat_py.eta import EtaTracker, format_duration, format_eta
from gymrat_py.sampling import (
    PrepareProgressStep,
    ProgressStep,
    SampleProgressStep,
)


def prepare(label: str) -> ProgressStep:
    """Build a prepare progress step."""
    return PrepareProgressStep(label=label)


def sample(index: int, total: int, label: str) -> ProgressStep:
    """Build a sample progress step."""
    return SampleProgressStep(index=index, total=total, label=label)


def clock_sequence(*times: float) -> Callable[[], float]:
    """Return a clock that yields each value in turn, raising when exhausted."""
    it = iter(times)

    def clock() -> float:
        try:
            return next(it)
        except StopIteration:
            message = "Clock sequence exhausted"
            raise RuntimeError(message) from None

    return clock


# ---------------------------------------------------------------------------
# EtaTracker.record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "step",
    [
        pytest.param(prepare("A"), id="prepare-step"),
        pytest.param(sample(1, 3, "A"), id="first-sample-step-no-gap"),
    ],
)
def test_record_when_no_gap_known_does_return_none(step: ProgressStep) -> None:
    tracker = EtaTracker(1, clock_sequence(0))

    result = tracker.record(step)

    assert result is None


@pytest.mark.parametrize(
    ("targets", "total", "expected"),
    [
        pytest.param(2, 3, 500, id="two-targets"),
        pytest.param(3, 4, 1100, id="target-count-from-constructor"),
    ],
)
def test_record_when_one_gap_known_does_return_estimate(
    targets: int, total: int, expected: float
) -> None:
    tracker = EtaTracker(targets, clock_sequence(0, 100))

    tracker.record(sample(1, total, "A"))
    result = tracker.record(sample(1, total, "B"))

    assert result == expected


def test_record_when_gaps_from_different_targets_does_pool_into_shared_mean() -> None:
    tracker = EtaTracker(2, clock_sequence(0, 100, 300))

    tracker.record(sample(1, 3, "A"))
    tracker.record(sample(1, 3, "B"))
    result = tracker.record(sample(2, 3, "A"))

    assert result == 600


def test_record_when_gap_follows_prepare_does_exclude_from_mean() -> None:
    tracker = EtaTracker(1, clock_sequence(0, 1000, 1100))

    tracker.record(prepare("A"))
    tracker.record(sample(1, 3, "A"))
    result = tracker.record(sample(2, 3, "A"))

    assert result == 200


def test_record_when_prepare_appears_mid_run_does_exclude_its_gap() -> None:
    tracker = EtaTracker(2, clock_sequence(0, 100, 1000, 1100))

    tracker.record(sample(1, 3, "A"))
    tracker.record(prepare("B"))
    tracker.record(sample(1, 3, "B"))
    result = tracker.record(sample(2, 3, "A"))

    assert result == 400


def test_record_when_clock_moves_backwards_does_discard_negative_gap() -> None:
    tracker = EtaTracker(1, clock_sequence(0, 100, 50))

    tracker.record(sample(1, 3, "A"))
    tracker.record(sample(2, 3, "A"))
    result = tracker.record(sample(3, 3, "A"))

    assert result == 100


def test_record_when_no_clock_injected_does_measure_gaps_with_perf_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.perf_counter", clock_sequence(0.0, 0.1))
    tracker = EtaTracker(1)

    tracker.record(sample(1, 3, "A"))
    result = tracker.record(sample(2, 3, "A"))

    assert result == 200


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
        (3_600_000, "1h 0m"),
        (5_400_000, "1h 30m"),
        (7_200_000, "2h 0m"),
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
        (3900000, "~1h 5m left"),
        (7200000, "~2h left"),
        (7260000, "~2h 1m left"),
    ],
)
def test_format_eta_when_given_milliseconds_does_render_expected_eta(
    ms: float, expected: str
) -> None:
    assert format_eta(ms) == expected
