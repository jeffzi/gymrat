"""Tests for grouped table layout in single-section (flat) comparison reports.

A single-kind comparison uses the flat (single-section) layout path.  These
tests verify that the flat body groups metrics under group headers, uses case
names for member rows, places ungrouped rows after groups, preserves
first-appearance ordering, and renders the full path prefix for deeper groups.

The verdict-cell alignment section tests that the styled (colored) and plain
verdict cells produce the same visible width for every field combination,
including the NaN-delta case where the delta field is empty but the column
carries a non-zero width from sibling rows.
"""

from __future__ import annotations

from gymrat.report.style import render_lines
from gymrat.report.table import (
    VerdictParts,
    VerdictWidths,
    join_verdict_cell,
    style_verdict_cell,
)
from gymrat.report.text import render_report
from gymrat.verdict import GroupAggregate, KindAggregate
from tests.report._inputs import (
    create_candidate,
    create_comparison_result,
    geomean_of,
    kind_metric,
    table_region,
)


def _grouped_flat_result():
    """Single ``time`` kind: ``entity`` group (2 members) + ungrouped ``warmup``."""
    geomean = geomean_of(-3.2, 3)
    return create_comparison_result(
        metrics={
            "entity/alive_check#time": kind_metric(
                kind="time",
                short_name="entity.alive_check",
                verdict="improved",
                delta=-10,
            ),
            "entity/spawn#time": kind_metric(
                kind="time",
                short_name="entity.spawn",
                verdict="regressed",
                delta=4,
            ),
            "warmup#time": kind_metric(
                kind="time",
                short_name="warmup",
                verdict="no-signal",
                delta=0.3,
            ),
        },
        candidates=[
            create_candidate(
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean,
                        groups=(
                            GroupAggregate(
                                group="entity",
                                geomean=geomean_of(-3.1, 2),
                            ),
                        ),
                        gated_geomean=geomean,
                    )
                ]
            )
        ],
    )


# ---------------------------------------------------------------------------
# group headers and case names
# ---------------------------------------------------------------------------


def test_table_region_when_flat_body_with_groups_does_emit_group_headers_and_case_names():
    """The flat body emits a group header line and uses case names for members.

    Currently ``_plan_flat_body`` iterates ``layout.ordered`` as flat
    ``MetricLine`` entries using the full metric name (e.g.
    ``entity/alive_check#time``).  With grouping, a ``GroupLine`` header for
    ``entity`` must appear and member rows must show their case name
    (``alive_check``, ``spawn``), not the full metric name.
    """
    region = table_region(render_report(_grouped_flat_result()))

    assert "alive_check" in region
    assert "spawn" in region
    assert "entity/alive_check#time" not in region
    assert "entity/spawn#time" not in region

    group_headers = [entry for entry in region if entry.startswith("entity") and "/" not in entry]
    assert group_headers


# ---------------------------------------------------------------------------
# ungrouped trailing rows
# ---------------------------------------------------------------------------


def test_table_region_when_flat_body_has_ungrouped_rows_does_trail_after_groups():
    """Single-segment names (no ``/`` in the path) render after grouped rows."""
    region = table_region(render_report(_grouped_flat_result()))

    group_member_indices = [
        i for i, entry in enumerate(region) if entry in ("alive_check", "spawn")
    ]
    warmup_indices = [
        i for i, entry in enumerate(region) if "warmup" in entry and "geomean" not in entry.lower()
    ]

    assert group_member_indices
    assert warmup_indices
    assert max(group_member_indices) < min(warmup_indices)


# ---------------------------------------------------------------------------
# first-appearance order
# ---------------------------------------------------------------------------


def test_table_region_when_flat_body_with_multiple_groups_does_preserve_first_appearance_order():
    """Group headers appear in the order of their first member's emission."""
    geomean = geomean_of(-2, 4)
    result = create_comparison_result(
        metrics={
            "node/get#time": kind_metric(
                kind="time", short_name="node.get", verdict="improved", delta=-5
            ),
            "entity/spawn#time": kind_metric(
                kind="time",
                short_name="entity.spawn",
                verdict="regressed",
                delta=4,
            ),
            "node/set#time": kind_metric(
                kind="time",
                short_name="node.set",
                verdict="no-signal",
                delta=0.1,
            ),
            "entity/check#time": kind_metric(
                kind="time",
                short_name="entity.check",
                verdict="improved",
                delta=-3,
            ),
        },
        candidates=[
            create_candidate(
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean,
                        groups=(
                            GroupAggregate(group="node", geomean=geomean_of(-2.5, 2)),
                            GroupAggregate(group="entity", geomean=geomean_of(0.5, 2)),
                        ),
                        gated_geomean=geomean,
                    )
                ]
            )
        ],
    )

    region = table_region(render_report(result))

    node_headers = [
        i for i, entry in enumerate(region) if entry.startswith("node") and "/" not in entry
    ]
    entity_headers = [
        i for i, entry in enumerate(region) if entry.startswith("entity") and "/" not in entry
    ]
    assert node_headers
    assert entity_headers
    assert node_headers[0] < entity_headers[0]


# ---------------------------------------------------------------------------
# deeper paths
# ---------------------------------------------------------------------------


def test_table_region_when_flat_body_with_deeper_path_does_use_full_prefix_as_group():
    """A metric name ``node/access/get_1field#time`` groups under ``node/access``."""
    geomean = geomean_of(-3.2, 2)
    result = create_comparison_result(
        metrics={
            "node/access/get_1field#time": kind_metric(
                kind="time",
                short_name="node_access.get_1field",
                verdict="improved",
                delta=-5,
            ),
            "node/access/get_2field#time": kind_metric(
                kind="time",
                short_name="node_access.get_2field",
                verdict="no-signal",
                delta=0.1,
            ),
        },
        candidates=[
            create_candidate(
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean,
                        groups=(GroupAggregate(group="node/access", geomean=geomean),),
                        gated_geomean=geomean,
                    )
                ]
            )
        ],
    )

    region = table_region(render_report(result))

    assert "get_1field" in region
    assert "get_2field" in region
    assert "node/access/get_1field#time" not in region
    assert "node/access/get_2field#time" not in region


# ---------------------------------------------------------------------------
# styled verdict cell alignment with plain verdict cell
# ---------------------------------------------------------------------------


def _plain_of_styled(markup: str) -> str:
    """Render markup through rich to get visible text (ANSI stripped)."""
    return render_lines(markup, color=False, width=200)


def test_verdict_cell_when_delta_empty_and_band_present_does_match_plain_and_styled_width():
    """NaN delta with a band column must pad equally in plain and styled paths.

    The plain path pads delta to ``widths.delta`` spaces; the styled path
    must not drop the padding.
    """
    parts = VerdictParts(glyph="~", delta="", word="", band="±2.5%", pairs="")
    widths = VerdictWidths(delta=7, band=5)

    plain = join_verdict_cell(parts, widths)
    styled = style_verdict_cell(
        parts, widths, glyph_style="dim", delta_style=None, band_style="dim"
    )

    assert len(plain) == len(_plain_of_styled(styled))


def test_verdict_cell_when_all_fields_present_does_match_plain_and_styled_width():
    """A fully populated verdict cell must render to the same visible width in both paths."""
    parts = VerdictParts(glyph="✓", delta="-10.0%", word="", band="±2.5%", pairs="")
    widths = VerdictWidths(delta=7, band=5)

    plain = join_verdict_cell(parts, widths)
    styled = style_verdict_cell(
        parts, widths, glyph_style="green", delta_style="green", band_style="dim"
    )

    assert len(plain) == len(_plain_of_styled(styled))


def test_verdict_cell_when_word_empty_and_delta_present_does_match_plain_and_styled_width():
    """An empty word with a present delta must produce matching widths."""
    parts = VerdictParts(glyph="~", delta="+4.0%", word="", band="", pairs="")
    widths = VerdictWidths(delta=6, band=0)

    plain = join_verdict_cell(parts, widths)
    styled = style_verdict_cell(
        parts, widths, glyph_style="dim", delta_style="dim", band_style=None
    )

    assert len(plain) == len(_plain_of_styled(styled))
