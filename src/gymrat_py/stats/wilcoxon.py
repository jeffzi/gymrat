"""Wilcoxon signed-rank test wrapper over index-paired samples."""

from collections.abc import Sequence

from gymrat_py.stats.results import SignificanceResult, count_nonzero_pairs

# The signed-rank test needs at least two paired observations to be meaningful;
# a single pair carries no rank spread and is short-circuited to non-significant.
_MIN_PAIRS = 2


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> SignificanceResult:
    """Run the two-sided Wilcoxon signed-rank test on index-paired samples.

    The inputs are paired positionally over the shorter of the two: ``x[i]`` is
    compared with ``y[i]`` for ``i < min(len(x), len(y))``, and any trailing
    entries of the longer input are ignored. Pairs whose difference is zero
    carry no rank information and are dropped, so the reported ``n`` counts only
    the pairs with a non-zero difference.

    Degenerate inputs cannot support the test and are short-circuited to a
    non-significant ``p = 1.0`` without invoking the underlying test: empty
    input, input whose paired differences are all zero, and input with fewer
    than two entries in ``x``.

    Args:
        x: The first sample.
        y: The second sample, paired positionally with ``x``.

    Returns:
        A :class:`SignificanceResult` whose ``p`` is the two-sided p-value
        clamped to at most ``1.0`` and whose ``n`` is the number of non-zero
        paired differences.
    """
    m, n = count_nonzero_pairs(x, y)
    if n == 0 or len(x) < _MIN_PAIRS:
        return SignificanceResult(p=1.0, n=n)

    # Import lazily so importing ``gymrat_py.stats`` never pulls in scipy; see
    # tests/test_import_latency.py, which guards the package's startup cost.
    from scipy.stats import wilcoxon  # noqa: PLC0415

    result = wilcoxon(x[:m], y[:m], zero_method="wilcox", method="auto")
    return SignificanceResult(p=min(float(result.pvalue), 1.0), n=n)
