from __future__ import annotations

import pytest

from gymrat.errors import GymratError
from gymrat.metric_name import format_inline, parse
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
    rendered = render_lines(markup, color=False, width=200)

    assert rendered == expected_plain
