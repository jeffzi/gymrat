from __future__ import annotations

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from rich.text import Text

from gymrat.metric_name import (
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
        pytest.param("a#", EmptyKindError, id="empty-kind"),
    ],
)
def test_parse_when_name_breaks_the_grammar_does_raise_the_rule_error_naming_it(
    raw_name: str, rule_error: type[MetricNameError]
):
    with pytest.raises(rule_error) as excinfo:
        parse(raw_name)

    assert str(excinfo.value).endswith(f": {raw_name}")


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
@example("dir\\/case#time").via("group-ends-in-backslash")
@example("dir[x]\\/case#time").via("bracketed-group-ends-in-backslash")
@example("dir/case\\").via("case-ends-in-backslash-without-kind")
@example("dir/case\\#time").via("case-ends-in-backslash-before-kind")
@example("dir/case\\\\#time").via("case-ends-in-two-backslashes-before-kind")
@example("dir\\\\/case#time").via("group-ends-in-two-backslashes")
@example("di\\r/ca\\se#time").via("inner-backslashes")
@example("dir\\[x]/case#time").via("backslash-before-bracket-in-group")
@example("a\\[1]").via("backslash-before-plain-bracket-in-case")
@example("a\\\\[1]").via("two-backslashes-before-plain-bracket-in-case")
@example("x#time\\").via("kind-ends-in-backslash")
def test_format_inline_when_name_well_formed_does_render_it_literally(
    raw_name: str,
):
    name = parse(raw_name)

    rendered = render_lines(format_inline(name), color=False)

    assert rendered == raw_name


def test_format_inline_when_group_and_kind_present_does_dim_them_around_the_case():
    markup = format_inline(parse("node/access#time"))

    text = Text.from_markup(markup)

    runs = [(text.plain[span.start : span.end], str(span.style)) for span in text.spans]
    assert (text.plain, runs) == ("node/access#time", [("node/", "dim"), ("#time", "dim")])
