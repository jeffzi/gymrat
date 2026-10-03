"""Tests for a table's verdict cells, its geomean labels, and its section planning.

A verdict cell's plain text is the column's width source, and its styles sit on
the glyph, the delta (or the word standing in for it) and the band, never on
padding.

The ``plan_sections`` function groups metrics into kind sections.  Group
membership derives from ``gymrat.metric_name.parse`` applied to the full metric
name (the dict key), not from the ``short_name`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from rich.text import Text

from gymrat.model import Exclusion, GeomeanResult, ResolvedMetricMeta
from gymrat.report.table.markup import (
    NO_AGGREGATE,
    GroupBlock,
    VerdictParts,
    VerdictWidths,
    geomean_value_style,
    group_geomean_of,
    kind_geomean_of,
    plan_sections,
    scoped_geomean_label,
    verdict_cell,
)
from tests.report._comparisons import create_candidate, memory_kind, time_kind
from tests.report._verdicts import geomean_of

if TYPE_CHECKING:
    from gymrat.report.table.markup import SectionLayout

# ---------------------------------------------------------------------------
# styled verdict cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parts", "widths", "delta_style", "band_style", "expected"),
    [
        pytest.param(
            VerdictParts(glyph="~", delta="", word="", band="2.5%", pairs=""),
            VerdictWidths(delta=7, band=4),
            None,
            "dim",
            ("~           ±2.5%", [("~", "green"), ("±2.5%", "dim")]),
            id="delta-empty-band-present",
        ),
        pytest.param(
            VerdictParts(glyph="✓", delta="-10.0%", word="", band="2.5%", pairs="n=8"),
            VerdictWidths(delta=7, band=4),
            "red",
            "dim",
            ("✓   -10.0%  ±2.5%  n=8", [("✓", "green"), ("-10.0%", "red"), ("±2.5%", "dim")]),
            id="all-fields-present",
        ),
        pytest.param(
            VerdictParts(glyph="✓", delta="-10.0%", word="", band="", pairs="n=8"),
            VerdictWidths(delta=7, band=4),
            "red",
            "dim",
            ("✓   -10.0%         n=8", [("✓", "green"), ("-10.0%", "red")]),
            id="band-absent-with-pairs-reserves-band-slot",
        ),
        pytest.param(
            VerdictParts(glyph="≈", delta="", word="unstable", band="", pairs=""),
            VerdictWidths(delta=6, band=0),
            "red",
            None,
            ("≈  unstable", [("≈", "green"), ("unstable", "red")]),
            id="word-stands-in-for-delta",
        ),
        pytest.param(
            VerdictParts(glyph="~", delta="+4.0%", word="", band="", pairs=""),
            VerdictWidths(delta=6, band=0),
            None,
            None,
            ("~   +4.0%", [("~", "green")]),
            id="delta-unstyled-band-absent",
        ),
    ],
)
def test_verdict_cell_when_fields_padded_to_widths_does_style_only_field_text(
    parts: VerdictParts,
    widths: VerdictWidths,
    delta_style: str | None,
    band_style: str | None,
    expected: tuple[str, list[tuple[str, str]]],
):
    plain, styled = expected

    cell = verdict_cell(
        parts, widths, glyph_style="green", delta_style=delta_style, band_style=band_style
    )

    assert isinstance(cell, Text)
    assert cell.plain == plain
    assert [(cell.plain[span.start : span.end], span.style) for span in cell.spans] == styled


# ---------------------------------------------------------------------------
# scoped_geomean_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scope", "geomean", "expected"),
    [
        pytest.param("entity", geomean_of(n=1), "geomean · entity (1)", id="nothing-excluded"),
        pytest.param(
            "entity",
            geomean_of(n=1, excluded=[Exclusion(metric="entity.spawn/time", reason="unstable")]),
            "geomean · entity (1/2)",
            id="one-exclusion",
        ),
        pytest.param(
            "memory",
            geomean_of(
                n=13,
                excluded=[
                    Exclusion(metric="a/heap", reason="unstable"),
                    Exclusion(metric="b/heap", reason="undefined-ratio"),
                ],
            ),
            "geomean · memory (13/15)",
            id="several-exclusions",
        ),
    ],
)
def test_scoped_geomean_label_when_subset_given_does_count_the_subset(
    scope: str, geomean: GeomeanResult, expected: str
):
    assert scoped_geomean_label(scope, geomean) == expected


# ---------------------------------------------------------------------------
# geomean_value_style
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("geomean", "expected"),
    [
        pytest.param(geomean_of(value=-6, band=5), "bold green", id="improvement-past-band"),
        pytest.param(geomean_of(value=6, band=5), "bold red", id="regression-past-band"),
        pytest.param(geomean_of(value=-4, band=5), "bold", id="improvement-inside-band"),
        pytest.param(geomean_of(value=4, band=5), "bold", id="regression-inside-band"),
        pytest.param(geomean_of(value=-5, band=5), "bold", id="level-with-band"),
        pytest.param(geomean_of(value=-0.2, band=0), "bold green", id="improvement-no-band"),
        pytest.param(geomean_of(value=float("nan"), n=0), "bold", id="no-stable-metrics"),
    ],
)
def test_geomean_value_style_when_value_given_does_style_against_band(
    geomean: GeomeanResult, expected: str
):
    assert geomean_value_style(geomean, []) == expected


# ---------------------------------------------------------------------------
# plan_sections — contract-derived groups from metric name
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Row:
    """Minimal row the measure callback produces, carrying enough to assert on."""

    name: str
    group: str | None


@dataclass(frozen=True, slots=True)
class _FakeMetric:
    """Satisfies the ``SectionedMetric`` protocol: one ``.meta`` attribute."""

    meta: ResolvedMetricMeta


def _measure(name: str, group: str | None, metric: _FakeMetric) -> _Row:
    return _Row(name=name, group=group)


def _metric(*, kind: str = "time", short_name: str = "x") -> _FakeMetric:
    return _FakeMetric(
        meta=ResolvedMetricMeta(
            direction="lower",
            gating=True,
            exact=False,
            unit=None,
            kind=kind,
            short_name=short_name,
        ),
    )


def _group_blocks(layout: SectionLayout[_Row]) -> list[GroupBlock[_Row]]:
    """The ``GroupBlock``s in the layout's first section."""
    section = layout.sections[0]
    return [block for block in section.blocks if isinstance(block, GroupBlock)]


def test_plan_sections_when_deeper_path_does_use_full_prefix_as_group():
    layout = plan_sections(
        {
            "node/access/get_1field#time": _metric(short_name="get_1field"),
            "node/access/get_2field#time": _metric(short_name="get_2field"),
        },
        _measure,
    )

    groups = _group_blocks(layout)
    assert [g.group for g in groups] == ["node/access"]
    assert len(groups[0].metrics) == 2


def test_plan_sections_when_single_segment_name_does_produce_no_group():
    layout = plan_sections(
        {
            "fib#time": _metric(short_name="fib"),
            "warmup#time": _metric(short_name="warmup"),
        },
        _measure,
    )

    groups = _group_blocks(layout)
    assert groups == []


def test_plan_sections_when_measure_callback_does_receive_contract_derived_group():
    # The group derives from the metric name key ("entity/alive_check#time" →
    # "entity"), not from short_name ("alive_check" → None).
    layout = plan_sections(
        {
            "entity/alive_check#time": _metric(short_name="alive_check"),
            "fib#time": _metric(short_name="fib"),
        },
        _measure,
    )

    rows_by_name = {row.name: row for row in layout.ordered}
    assert rows_by_name["entity/alive_check#time"].group == "entity"
    assert rows_by_name["fib#time"].group is None


def test_plan_sections_when_group_members_interleave_does_gather_them_in_the_first_block():
    # The group derives from the metric name key, not short_name:
    # "entity/spawn#time" → group "entity".
    layout = plan_sections(
        {
            "entity/spawn#time": _metric(),
            "render/frame#time": _metric(),
            "entity/remove#time": _metric(),
        },
        _measure,
    )

    (section,) = layout.sections
    assert section.blocks == [
        GroupBlock(
            group="entity",
            metrics=[
                _Row(name="entity/spawn#time", group="entity"),
                _Row(name="entity/remove#time", group="entity"),
            ],
        ),
        GroupBlock(group="render", metrics=[_Row(name="render/frame#time", group="render")]),
    ]


def test_plan_sections_when_group_spans_kinds_does_open_one_block_per_section():
    layout = plan_sections(
        {
            "entity/spawn#time": _metric(kind="time"),
            "entity/spawn#memory": _metric(kind="memory"),
        },
        _measure,
    )

    assert [section.blocks for section in layout.sections] == [
        [GroupBlock(group="entity", metrics=[_Row(name="entity/spawn#time", group="entity")])],
        [GroupBlock(group="entity", metrics=[_Row(name="entity/spawn#memory", group="entity")])],
    ]


# ---------------------------------------------------------------------------
# kind_geomean_of / group_geomean_of — reading a candidate's aggregate back out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        pytest.param("memory", memory_kind().geomean, id="reported-kind"),
        pytest.param("cpu", NO_AGGREGATE, id="unreported-kind"),
    ],
)
def test_kind_geomean_of_when_looking_up_a_kind_does_return_its_geomean_or_no_aggregate(
    kind: str, expected: GeomeanResult
):
    candidate = create_candidate(kinds=[time_kind(), memory_kind()])

    geomean = kind_geomean_of(candidate, kind)

    assert geomean == expected


@pytest.mark.parametrize(
    ("kind", "group", "expected"),
    [
        pytest.param("time", "entity", time_kind().groups[0].geomean, id="reported-group"),
        pytest.param("time", "render", NO_AGGREGATE, id="unreported-group"),
        pytest.param("cpu", "entity", NO_AGGREGATE, id="unreported-kind"),
    ],
)
def test_group_geomean_of_when_looking_up_a_group_does_return_its_geomean_or_no_aggregate(
    kind: str, group: str, expected: GeomeanResult
):
    candidate = create_candidate(kinds=[time_kind(), memory_kind()])

    geomean = group_geomean_of(candidate, kind, group)

    assert geomean == expected
