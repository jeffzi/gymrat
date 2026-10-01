"""Tests for the layout and public surface of the ``gymrat.stats`` module."""

from pathlib import Path

import gymrat.stats


def test_stats_when_imported_does_load_from_single_flat_module_file():
    module_file = Path(gymrat.stats.__file__)

    assert (module_file.parent.name, module_file.name) == ("gymrat", "stats.py")


_PUBLIC_NAMES = frozenset({
    "PERMUTATION_SEED",
    "RESAMPLE_BUDGET",
    "GeomeanCombination",
    "RatioExclusion",
    "RatioOutcome",
    "SignificanceResult",
    "combine_geomean",
    "compute_half_range",
    "count_nonzero_pairs",
    "normalize_ratio",
    "percent_delta",
    "sign_flip_permutation_test",
})


def test_stats_when_public_names_resolved_does_export_every_helper():
    exported = {name: getattr(gymrat.stats, name) for name in gymrat.stats.__all__}

    assert exported.keys() == _PUBLIC_NAMES
