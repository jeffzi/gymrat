"""Metric metadata, sample builders and verdict accessors shared by the verdict-engine tests."""

from gymrat.model import (
    DEFAULT_UNSTABLE_NOISE_PCT,
    BandVerdict,
    Direction,
    MetricMeta,
    MetricUnit,
    MetricVerdict,
    PermutationVerdict,
)
from gymrat.utils import WarnSink
from gymrat.verdict import compute_verdicts


def meta_of(
    direction: Direction,
    *,
    exact: bool = False,
    unit: MetricUnit | None = None,
    resolution: float | None = None,
) -> MetricMeta:
    """Build a gating ``MetricMeta``; by default approximate, unitless, and without a step."""
    return MetricMeta(
        direction=direction, gating=True, exact=exact, unit=unit, resolution=resolution
    )


METRIC_EXACT_LOWER = {"metric": meta_of("lower", exact=True)}
METRIC_EXACT_HIGHER = {"metric": meta_of("higher", exact=True)}
METRIC_APPROX_LOWER = {"metric": meta_of("lower")}
METRIC_APPROX_HIGHER = {"metric": meta_of("higher")}
METRIC_BYTES_LOWER = {"metric": meta_of("lower", unit="bytes", resolution=1.0)}
METRIC_BYTES_HIGHER = {"metric": meta_of("higher", unit="bytes", resolution=1.0)}
METRIC_NS_LOWER = {"metric": meta_of("lower", unit="ns")}
METRIC_BYTES_NO_RESOLUTION_LOWER = {"metric": meta_of("lower", unit="bytes")}
METRIC_EXACT_BYTES_LOWER = {"metric": meta_of("lower", exact=True, unit="bytes", resolution=1.0)}

# A 24 MHz timer read in whole nanoseconds: one tick is 41.67 ns, so readings land
# 41 or 42 ns apart.
TIMER_TICK_NS = 41.67
METRIC_TIMER_LOWER = {"metric": meta_of("lower", unit="ns", resolution=TIMER_TICK_NS)}
METRIC_EXACT_TIMER_LOWER = {
    "metric": meta_of("lower", exact=True, unit="ns", resolution=TIMER_TICK_NS)
}


def noop_warn(_message: str) -> None:
    """Swallow divergence warnings so these cases stay silent on stderr."""


def samples(*values: float) -> list[dict[str, float]]:
    """Build round dicts keying each value under the default ``"metric"`` name."""
    return [{"metric": value} for value in values]


def create_samples(n: int, value: float) -> list[dict[str, float]]:
    """Build *n* single-metric sample rounds, each recording *value* under ``"metric"``."""
    return samples(*[value] * n)


def run(
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    meta: dict[str, MetricMeta],
    *,
    unstable_noise_pct: float = DEFAULT_UNSTABLE_NOISE_PCT,
    warn: WarnSink | None = None,
) -> dict[str, MetricVerdict]:
    """Pair two round lists and compute verdicts, discarding warnings unless a sink is given."""
    sink = noop_warn if warn is None else warn
    return compute_verdicts(
        samples_a, samples_b, meta, unstable_noise_pct=unstable_noise_pct, warn=sink
    )


def get_permutation(result: dict[str, MetricVerdict], key: str = "metric") -> PermutationVerdict:
    """Narrow the verdict at *key* to ``PermutationVerdict``, failing if the method differs."""
    verdict = result[key]
    assert isinstance(verdict, PermutationVerdict), f"expected permutation, got {verdict.method}"
    return verdict


def get_band(result: dict[str, MetricVerdict], key: str = "metric") -> BandVerdict:
    """Narrow the verdict at *key* to ``BandVerdict``, failing if the method differs."""
    verdict = result[key]
    assert isinstance(verdict, BandVerdict), f"expected band, got {verdict.method}"
    return verdict


# Six paired windows noisy enough to reach noise_pct = 30 while staying on the
# permutation path: all six diffs are non-zero and negative, so every pair shows
# the decrease and the sign-flip null makes it significant (p = 2 / 2**6).
NOISY_PERMUTATION_A = samples(80.0, 90.0, 100.0, 100.0, 110.0, 120.0)
NOISY_PERMUTATION_B = samples(40.0, 45.0, 50.0, 50.0, 55.0, 60.0)

# Enough identical rounds to stay on the permutation path with p well below 0.05.
FLAT_PERMUTATION_ROUNDS = 10

# Two paired windows noisy enough to reach noise_pct = 30 on the band path.
NOISY_BAND_A = samples(80.0, 120.0)
NOISY_BAND_B = samples(8.0, 12.0)

# Six windows whose medians are 4 vs 3 bytes — a one-byte move (delta -25%) —
# that every pair shows, so the sign-flip null makes it significant
# (p = 0.03125). The whole-byte floor, not the p-value, is what withholds the
# signal on a byte metric.
BYTE_ONE_MOVE_A = samples(3.4, 3.7, 4.0, 4.0, 4.3, 4.6)
BYTE_ONE_MOVE_B = samples(2.4, 2.7, 3.0, 3.0, 3.3, 3.6)

# Two six-window groups where every pair moves the same way: the sign-flip null
# makes the shift significant (p = 0.03125) in either pairing direction.
SEPARATED_HIGH = samples(180.0, 190.0, 200.0, 200.0, 210.0, 220.0)
SEPARATED_LOW = samples(90.0, 95.0, 100.0, 100.0, 105.0, 110.0)
