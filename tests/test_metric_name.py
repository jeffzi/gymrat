from __future__ import annotations

import sys

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from gymrat.metric_name import (
    LINE_TERMINATORS,
    EmptyKindError,
    EmptyPathSegmentError,
    MetricNameError,
    MultipleHashesError,
    format_inline,
    parse,
)
from gymrat.report.style import render_lines


@pytest.mark.parametrize(
    ("raw_name", "path", "kind", "group", "case"),
    [
        pytest.param(
            "node/access.get_1field#time",
            ("node", "access.get_1field"),
            "time",
            "node",
            "access.get_1field",
            id="two-segments-with-kind",
        ),
        pytest.param(
            "node/access/get_1field#time",
            ("node", "access", "get_1field"),
            "time",
            "node/access",
            "get_1field",
            id="three-segments-with-kind",
        ),
        pytest.param("fib/total", ("fib", "total"), None, "fib", "total", id="no-kind"),
        pytest.param("fib", ("fib",), None, None, "fib", id="one-segment-no-kind"),
    ],
)
def test_parse_when_name_well_formed_does_split_path_kind_group_and_case(
    raw_name: str, path: tuple[str, ...], kind: str | None, group: str | None, case: str
):
    result = parse(raw_name)

    assert (result.path, result.kind, result.group, result.case) == (path, kind, group, case)


@pytest.mark.parametrize(
    ("raw_name", "rule_error"),
    [
        pytest.param("a#b#c", MultipleHashesError, id="multiple-hashes"),
        pytest.param("a//b", EmptyPathSegmentError, id="empty-inner-segment"),
        pytest.param("a/", EmptyPathSegmentError, id="empty-trailing-segment"),
        pytest.param("/x", EmptyPathSegmentError, id="empty-leading-segment"),
        pytest.param("", EmptyPathSegmentError, id="empty-name"),
        pytest.param("#time", EmptyPathSegmentError, id="empty-path-before-kind"),
        pytest.param("a/#time", EmptyPathSegmentError, id="empty-trailing-segment-before-kind"),
        pytest.param("a#", EmptyKindError, id="empty-kind"),
    ],
)
def test_parse_when_name_breaks_the_grammar_does_raise_the_rule_error_naming_it(
    raw_name: str, rule_error: type[MetricNameError]
):
    with pytest.raises(rule_error) as excinfo:
        parse(raw_name)

    assert raw_name in str(excinfo.value)


def test_format_inline_when_name_has_group_and_kind_does_dim_the_group_and_kind():
    name = parse("node/access.get_1field#time")

    result = format_inline(name)

    assert result == "[dim]node/[/dim]access.get_1field[dim]#time[/dim]"


_SEGMENT = st.text(alphabet="ab.=@[]\\", min_size=1, max_size=8)


@st.composite
def _metric_names(draw: st.DrawFn) -> str:
    """Draw a well-formed metric name whose segments and kind are dense in markup characters."""
    path = "/".join(draw(st.lists(_SEGMENT, min_size=1, max_size=3)))
    kind = draw(st.none() | _SEGMENT)
    return path if kind is None else f"{path}#{kind}"


@given(raw_name=_metric_names())
@example("parse[js]#time").via("square-brackets-in-case")
@example("codec[h264]/decode#time").via("square-brackets-in-group")
@example("render#fps[avg]").via("square-brackets-in-kind")
@example("parse[/html]#time").via("closing-tag-shaped-bracket")
@example("dir\\/case#time").via("group-ends-in-backslash")
@example("top/dir\\/case").via("nested-group-ends-in-backslash")
@example("dir[x]\\/case#time").via("bracketed-group-ends-in-backslash")
@example("dir/case\\").via("case-ends-in-backslash-without-kind")
@example("dir/case\\#time").via("case-ends-in-backslash-before-kind")
@example("dir/case\\\\#time").via("case-ends-in-two-backslashes-before-kind")
@example("dir\\\\/case#time").via("group-ends-in-two-backslashes")
@example("di\\r/ca\\se#time").via("inner-backslashes")
@example("dir\\[x]/case#time").via("backslash-before-bracket-in-group")
@example("a\\[1]").via("backslash-before-plain-bracket-in-case")
@example("a\\\\[1]").via("two-backslashes-before-plain-bracket-in-case")
@example("dir\\[1]/case").via("backslash-before-plain-bracket-in-group")
@example("case#k\\[1]").via("backslash-before-plain-bracket-in-kind")
@example("x#time\\").via("kind-ends-in-backslash")
@example("case\\").via("single-segment-ends-in-backslash")
def test_format_inline_when_name_well_formed_does_render_it_literally(
    raw_name: str,
):
    name = parse(raw_name)

    rendered = render_lines(format_inline(name), color=False)

    assert rendered == raw_name


def test_line_terminators_when_scanning_every_code_point_does_match_exactly_splitlines_breaks():
    code_points = [chr(code) for code in range(sys.maxunicode + 1)]

    matched = {char for char in code_points if LINE_TERMINATORS.search(char)}

    assert matched == {char for char in code_points if len(f"a{char}b".splitlines()) == 2}
