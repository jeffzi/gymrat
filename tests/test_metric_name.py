from __future__ import annotations

import sys

import pytest

from gymrat.errors import GymratError
from gymrat.metric_name import LINE_TERMINATORS, format_inline, parse
from gymrat.report.style import render_lines


def test_parse_when_name_has_kind_does_split_path_and_kind():
    result = parse("node/access.get_1field#time")

    assert result.path == ("node", "access.get_1field")
    assert result.kind == "time"


def test_parse_when_name_has_no_kind_does_return_none_kind():
    result = parse("fib/total")

    assert result.path == ("fib", "total")
    assert result.kind is None


def test_parse_when_name_has_multiple_hashes_does_raise_gymrat_error():
    with pytest.raises(GymratError, match="a#b#c"):
        parse("a#b#c")


@pytest.mark.parametrize(
    "raw_name",
    [
        pytest.param("a//b", id="empty-inner-segment"),
        pytest.param("a/", id="empty-trailing-segment"),
        pytest.param("/x", id="empty-leading-segment"),
        pytest.param("", id="empty-name"),
        pytest.param("#time", id="empty-path-before-kind"),
        pytest.param("a/#time", id="empty-trailing-segment-before-kind"),
        pytest.param("a#", id="empty-kind"),
    ],
)
def test_parse_when_name_has_empty_segment_does_raise_gymrat_error(raw_name: str):
    with pytest.raises(GymratError) as excinfo:
        parse(raw_name)

    assert raw_name in str(excinfo.value)


@pytest.mark.parametrize(
    "raw_name",
    [
        pytest.param("a#b#c", id="multiple-hashes"),
        pytest.param("a#", id="empty-kind"),
        pytest.param("a//b", id="empty-path-segment"),
    ],
)
def test_parse_when_name_breaks_grammar_does_raise_error_that_rebuilds_from_message_and_hint(
    raw_name: str,
):
    with pytest.raises(GymratError) as excinfo:
        parse(raw_name)

    error = excinfo.value
    rebuilt = type(error)(str(error), hint=error.hint)
    assert type(rebuilt) is type(error)
    assert str(rebuilt) == str(error)
    assert rebuilt.hint == error.hint


@pytest.mark.parametrize(
    ("name", "expected_group", "expected_case"),
    [
        pytest.param(
            "node/access.get_1field#time",
            "node",
            "access.get_1field",
            id="two-segments-with-kind",
        ),
        pytest.param(
            "node/access/get_1field#time",
            "node/access",
            "get_1field",
            id="three-segments-with-kind",
        ),
        pytest.param("fib", None, "fib", id="one-segment-no-kind"),
    ],
)
def test_parse_when_varying_depth_does_expose_correct_group_and_case(
    name: str,
    expected_group: str | None,
    expected_case: str,
):
    result = parse(name)

    assert result.group == expected_group
    assert result.case == expected_case


def test_format_inline_when_called_does_return_rich_markup():
    name = parse("node/access.get_1field#time")

    result = format_inline(name)

    assert result == "[dim]node/[/dim]access.get_1field[dim]#time[/dim]"


def test_metric_name_when_mutated_does_raise():
    name = parse("fib/total")

    with pytest.raises(AttributeError):
        name.kind = "time"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("raw_name", "expected_plain"),
    [
        pytest.param(
            "parse[js]#time",
            "parse[js]#time",
            id="square-brackets-in-case",
        ),
        pytest.param(
            "codec[h264]/decode#time",
            "codec[h264]/decode#time",
            id="square-brackets-in-group",
        ),
        pytest.param(
            "render#fps[avg]",
            "render#fps[avg]",
            id="square-brackets-in-kind",
        ),
        pytest.param(
            "parse[/html]#time",
            "parse[/html]#time",
            id="closing-tag-shaped-bracket",
        ),
    ],
)
def test_format_inline_when_color_on_and_brackets_in_segments_does_render_literally(
    raw_name: str,
    expected_plain: str,
):
    name = parse(raw_name)

    markup = format_inline(name)
    rendered = render_lines(markup, color=False)

    assert rendered == expected_plain


@pytest.mark.parametrize(
    "raw_name",
    [
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
def test_format_inline_when_name_contains_backslash_does_render_name_literally(raw_name: str):
    name = parse(raw_name)

    rendered = render_lines(format_inline(name), color=False)

    assert rendered == raw_name


def test_line_terminators_when_scanning_every_code_point_does_match_exactly_splitlines_breaks():
    code_points = [chr(code) for code in range(sys.maxunicode + 1)]

    matched = {char for char in code_points if LINE_TERMINATORS.search(char)}

    assert matched == {char for char in code_points if len(f"a{char}b".splitlines()) == 2}
