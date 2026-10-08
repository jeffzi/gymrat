from __future__ import annotations

import sys

import pytest

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


def test_format_inline_when_called_does_return_rich_markup():
    name = parse("node/access.get_1field#time")

    result = format_inline(name)

    assert result == "[dim]node/[/dim]access.get_1field[dim]#time[/dim]"


@pytest.mark.parametrize(
    "raw_name",
    [
        pytest.param("parse[js]#time", id="square-brackets-in-case"),
        pytest.param("codec[h264]/decode#time", id="square-brackets-in-group"),
        pytest.param("render#fps[avg]", id="square-brackets-in-kind"),
        pytest.param("parse[/html]#time", id="closing-tag-shaped-bracket"),
        pytest.param("dir\\/case#time", id="group-ends-in-backslash"),
        pytest.param("top/dir\\/case", id="nested-group-ends-in-backslash"),
        pytest.param("dir[x]\\/case#time", id="bracketed-group-ends-in-backslash"),
        pytest.param("dir/case\\", id="case-ends-in-backslash-without-kind"),
        pytest.param("dir/case\\#time", id="case-ends-in-backslash-before-kind"),
        pytest.param("dir/case\\\\#time", id="case-ends-in-two-backslashes-before-kind"),
        pytest.param("dir\\\\/case#time", id="group-ends-in-two-backslashes"),
        pytest.param("di\\r/ca\\se#time", id="inner-backslashes"),
        pytest.param("dir\\[x]/case#time", id="backslash-before-bracket-in-group"),
        pytest.param("a\\[1]", id="backslash-before-plain-bracket-in-case"),
        pytest.param("a\\\\[1]", id="two-backslashes-before-plain-bracket-in-case"),
        pytest.param("dir\\[1]/case", id="backslash-before-plain-bracket-in-group"),
        pytest.param("case#k\\[1]", id="backslash-before-plain-bracket-in-kind"),
        pytest.param("x#time\\", id="kind-ends-in-backslash"),
        pytest.param("case\\", id="single-segment-ends-in-backslash"),
    ],
)
def test_format_inline_when_name_holds_brackets_or_backslashes_does_render_it_literally(
    raw_name: str,
):
    name = parse(raw_name)

    rendered = render_lines(format_inline(name), color=False)

    assert rendered == raw_name


def test_line_terminators_when_scanning_every_code_point_does_match_exactly_splitlines_breaks():
    code_points = [chr(code) for code in range(sys.maxunicode + 1)]

    matched = {char for char in code_points if LINE_TERMINATORS.search(char)}

    assert matched == {char for char in code_points if len(f"a{char}b".splitlines()) == 2}
