"""Verdict literals and the tagged per-method verdict record union."""

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["improved", "regressed", "no-signal"]
"""Outcome of a comparison that cannot be flagged unstable."""

ApproximateVerdict = Literal["improved", "regressed", "no-signal", "unstable"]
"""Outcome of an approximate comparison, which may additionally be ``"unstable"``."""


@dataclass(frozen=True, slots=True)
class SignedRankVerdict:
    """Verdict from the signed-rank method.

    Attributes:
        method: Discriminant tag, always ``"signed-rank"``.
        verdict: The approximate outcome.
        p: The signed-rank test p-value.
        noise_pct: Estimated noise as a percentage.
        noise_abs: Estimated noise in absolute units.
    """

    method: Literal["signed-rank"]
    verdict: ApproximateVerdict
    p: float
    noise_pct: float
    noise_abs: float


@dataclass(frozen=True, slots=True)
class BandVerdict:
    """Verdict from the band method.

    Attributes:
        method: Discriminant tag, always ``"band"``.
        verdict: The approximate outcome.
        usable_n: Number of usable samples.
        noise_pct: Estimated noise as a percentage.
        noise_abs: Estimated noise in absolute units.
    """

    method: Literal["band"]
    verdict: ApproximateVerdict
    usable_n: int
    noise_pct: float
    noise_abs: float


@dataclass(frozen=True, slots=True)
class ExactVerdict:
    """Verdict from the exact method, which is never unstable and carries no noise.

    Attributes:
        method: Discriminant tag, always ``"exact"``.
        verdict: The (non-approximate) outcome.
        delta: The exact difference between compared values.
        n: Number of samples.
    """

    method: Literal["exact"]
    verdict: Verdict
    delta: float
    n: int


MetricVerdict = SignedRankVerdict | BandVerdict | ExactVerdict
"""Union of per-method verdict records, discriminated on ``method``."""
