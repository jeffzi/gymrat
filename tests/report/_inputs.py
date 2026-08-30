"""Shared comparison/verdict builders for the report formatting tests.

Re-export shim: the implementations live in ``_verdicts``, ``_comparisons``,
``_measurements``, and ``_assertions``; this module re-exports them all so
existing importers keep working unchanged.
"""

from gymrat.report.types import MetricComparison, MetricComparisons
from tests.report._assertions import (
    DIMMED_LINE,
    cells_of,
    delta_cell,
    highlight_lines,
    last_table_row,
    line_containing,
    line_starting_with,
    offsets_of,
    separator_offsets,
    separator_styles,
    strip_ansi,
    styles_at,
    table_region,
    table_rows,
    table_shape,
)
from tests.report._comparisons import (
    NWayCandidate,
    create_candidate,
    create_comparison_result,
    exact_metric,
    grouped_comparison,
    kind_metric,
    memory_kind,
    metric_meta,
    multi_candidate_result,
    n_way_kind_metric,
    n_way_metric,
    other_kind,
    permutation_metric,
    single_sample_result,
    time_kind,
    two_kind_metrics,
    two_kind_result,
    without_gated_geomean,
)
from tests.report._measurements import (
    create_measurement_result,
    measured_metric,
    two_kind_measurement,
)
from tests.report._verdicts import (
    CandidateSpec,
    approximate_metric,
    band_metric,
    band_verdict,
    exact_verdict,
    geomean_of,
    metric_for,
    one_sided_metric,
    permutation_verdict,
)

# The name-keyed map of every metric compared in a run.
Metrics = MetricComparisons
# One metric's comparison data: baseline, per-candidate results, and meta.
MetricEntry = MetricComparison

__all__ = [
    "DIMMED_LINE",
    "CandidateSpec",
    "MetricEntry",
    "Metrics",
    "NWayCandidate",
    "approximate_metric",
    "band_metric",
    "band_verdict",
    "cells_of",
    "create_candidate",
    "create_comparison_result",
    "create_measurement_result",
    "delta_cell",
    "exact_metric",
    "exact_verdict",
    "geomean_of",
    "grouped_comparison",
    "highlight_lines",
    "kind_metric",
    "last_table_row",
    "line_containing",
    "line_starting_with",
    "measured_metric",
    "memory_kind",
    "metric_for",
    "metric_meta",
    "multi_candidate_result",
    "n_way_kind_metric",
    "n_way_metric",
    "offsets_of",
    "one_sided_metric",
    "other_kind",
    "permutation_metric",
    "permutation_verdict",
    "separator_offsets",
    "separator_styles",
    "single_sample_result",
    "strip_ansi",
    "styles_at",
    "table_region",
    "table_rows",
    "table_shape",
    "time_kind",
    "two_kind_measurement",
    "two_kind_metrics",
    "two_kind_result",
    "without_gated_geomean",
]
