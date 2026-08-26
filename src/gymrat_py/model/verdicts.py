"""Verdict literals and the tagged per-method verdict record union."""

from dataclasses import dataclass
from typing import Literal

from gymrat_py.model.effect import Effect

Verdict = Literal["improved", "regressed", "no-signal"]
"""Outcome of a comparison that cannot be flagged unstable."""

ApproximateVerdict = Literal["improved", "regressed", "no-signal", "unstable"]
"""Outcome of an approximate comparison, which may additionally be ``"unstable"``."""


@dataclass(frozen=True, slots=True)
class PermutationVerdict:
    """Verdict from the sign-flip permutation method.

    Attributes:
        method: Discriminant tag, always ``"permutation"``.
        verdict: The approximate outcome.
        p: The sign-flip permutation test p-value.
        noise_pct: Estimated noise as a percentage.
        noise_abs: Estimated noise in absolute units.
        delta: The unit-tagged effect size of the comparison.
        n: Number of paired samples.
    """

    method: Literal["permutation"]
    verdict: ApproximateVerdict
    p: float
    noise_pct: float
    noise_abs: float
    delta: Effect
    n: int


@dataclass(frozen=True, slots=True)
class BandVerdict:
    """Verdict from the band method.

    Attributes:
        method: Discriminant tag, always ``"band"``.
        verdict: The approximate outcome.
        usable_n: Number of usable samples.
        noise_pct: Estimated noise as a percentage.
        noise_abs: Estimated noise in absolute units.
        delta: The unit-tagged effect size of the comparison.
        n: Number of paired samples.
    """

    method: Literal["band"]
    verdict: ApproximateVerdict
    usable_n: int
    noise_pct: float
    noise_abs: float
    delta: Effect
    n: int


@dataclass(frozen=True, slots=True)
class ExactVerdict:
    """Verdict from the exact method, which is never unstable and carries no noise.

    Attributes:
        method: Discriminant tag, always ``"exact"``.
        verdict: The (non-approximate) outcome.
        delta: The unit-tagged effect size of the comparison.
        n: Number of samples.
    """

    method: Literal["exact"]
    verdict: Verdict
    delta: Effect
    n: int


MetricVerdict = PermutationVerdict | BandVerdict | ExactVerdict
"""Union of per-method verdict records, discriminated on ``method``."""
