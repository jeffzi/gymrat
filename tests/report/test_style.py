"""Tests for the report style/color primitives.

These cover the style/color primitives (label clipping by terminal cells and
grapheme clusters) plus the color-resolution and capture-rendering tests.
"""

import os

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from rich.cells import cell_len, split_graphemes
from rich.markup import escape

from gymrat.report.style import (
    RENDER_WIDTH,
    format_hint,
    highlight_inline_code,
    make_capture_console,
    render_lines,
    shorten_label,
    truncate_labels,
)
from tests._ansi import strip_ansi
from tests.report._assertions import render_colored, render_plain, styles_at

# ---------------------------------------------------------------------------
# shorten_label
# ---------------------------------------------------------------------------

_TEXT = "rain on the lake"


@pytest.mark.parametrize(
    "max_width",
    [
        pytest.param(20, id="wider-than-text"),
        pytest.param(len(_TEXT), id="exactly-text-width"),
    ],
)
def test_shorten_label_when_text_already_fits_does_return_verbatim(max_width: int):
    assert shorten_label(_TEXT, max_width) == _TEXT


@pytest.mark.parametrize(
    ("max_width", "expected"),
    [
        pytest.param(9, "rain…lake", id="odd-width-splits-evenly"),
        pytest.param(8, "rain…ake", id="even-width-favors-head"),
        pytest.param(2, "r…", id="tail-squeezed-out"),
        pytest.param(1, "…", id="ellipsis-alone"),
    ],
)
def test_shorten_label_when_text_overflows_does_middle_ellipsis(max_width: int, expected: str):
    assert shorten_label(_TEXT, max_width) == expected


@pytest.mark.parametrize(
    "max_width",
    [pytest.param(0, id="zero"), pytest.param(-5, id="negative")],
)
def test_shorten_label_when_width_leaves_no_room_does_return_empty(max_width: int):
    assert shorten_label(_TEXT, max_width) == ""


_E_ACUTE = "e\N{COMBINING ACUTE ACCENT}"
_HEART = "\N{HEAVY BLACK HEART}\N{VARIATION SELECTOR-16}"
_FAMILY = "\N{MAN}\N{ZERO WIDTH JOINER}\N{WOMAN}\N{ZERO WIDTH JOINER}\N{GIRL}"


@pytest.mark.parametrize(
    ("text", "max_width", "expected"),
    [
        pytest.param("一二三", 6, "一二三", id="fits-by-cells-verbatim"),
        pytest.param("一二三", 4, "一…", id="overflows-by-cells-truncates"),
        pytest.param("一二三四五六", 9, "一二…五六", id="wide-middle-ellipsis"),
        pytest.param("世界世界世界", 3, "…", id="wide-clusters-wider-than-both-shares"),
        pytest.param("世界世界世界", 5, "世…界", id="wide-clusters-filling-both-shares"),
        pytest.param("世abc", 3, "…c", id="wide-first-cluster-wider-than-head-share"),
        pytest.param(
            f"abc{_E_ACUTE}thunderbird", 9, f"abc{_E_ACUTE}…bird", id="combining-mark-in-head"
        ),
        pytest.param(f"abcdefghij{_E_ACUTE}xyz", 7, "abc…xyz", id="combining-mark-at-tail-edge"),
        pytest.param(f"ab{_HEART}understatement", 7, "ab…ent", id="emoji-selector-at-head-edge"),
        pytest.param(
            f"abcdefghijklmnop{_FAMILY}xy", 9, f"abcd…{_FAMILY}xy", id="zwj-sequence-in-tail"
        ),
    ],
)
def test_shorten_label_when_text_has_wide_or_combined_clusters_does_clip_on_whole_clusters(
    text: str, max_width: int, expected: str
):
    assert shorten_label(text, max_width) == expected


_CLUSTER_TEXT = st.lists(st.sampled_from(["a", "世", _E_ACUTE, _HEART, _FAMILY]), max_size=12).map(
    "".join
)


@given(text=_CLUSTER_TEXT, max_width=st.integers(min_value=0, max_value=20))
def test_shorten_label_when_given_any_text_does_stay_within_budget(text: str, max_width: int):
    assert cell_len(shorten_label(text, max_width)) <= max_width


@given(text=_CLUSTER_TEXT, max_width=st.integers(min_value=1, max_value=20))
def test_shorten_label_when_clipping_does_keep_whole_clusters_from_both_ends(
    text: str, max_width: int
):
    assume(cell_len(text) > max_width)
    spans, _ = split_graphemes(text)
    boundaries = {0, *(end for _start, end, _width in spans)}

    head, ellipsis, tail = shorten_label(text, max_width).partition("…")

    assert ellipsis == "…"
    assert text.startswith(head)
    assert text.endswith(tail)
    assert {len(head), len(text) - len(tail)} <= boundaries


# ---------------------------------------------------------------------------
# truncate_labels
# ---------------------------------------------------------------------------


def test_truncate_labels_when_every_label_fits_does_return_verbatim():
    labels = ["main", "feature/short-branch"]

    assert truncate_labels(labels) == ["main", "feature/short-branch"]


def test_truncate_labels_when_a_label_overflows_does_join_head_and_tail():
    result = truncate_labels(["feature/entity-spawn-fastpath"])

    assert result == ["feature/en…-fastpath"]
    assert len(result[0]) == 20


def test_truncate_labels_when_widening_past_a_fitting_label_does_not_lengthen_it():
    short_enough = "release/candidate-2.1"

    result = truncate_labels([
        "feature/experiment-one-fastpath",
        "feature/exploration-two-fastpath",
        short_enough,
    ])

    assert result == [
        "feature/ex…e-fastpath",
        "feature/ex…o-fastpath",
        short_enough,
    ]


# ---------------------------------------------------------------------------
# highlight_inline_code
# ---------------------------------------------------------------------------


def test_highlight_inline_code_when_span_in_prose_does_paint_the_span_blue_without_backticks():
    styled = highlight_inline_code("Run `gymrat doctor` to verify.")

    assert render_plain(styled) == "Run gymrat doctor to verify."
    assert styles_at(render_colored(styled), "gymrat doctor") == ["34"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "Use `gymrat compare` or `gymrat measure`.",
            "Use gymrat compare or gymrat measure.",
            id="multiple-spans",
        ),
        pytest.param("No inline code here.", "No inline code here.", id="no-backticks"),
        pytest.param("Metric `[i]` counts.", "Metric [i] counts.", id="markup-metacharacters"),
    ],
)
def test_highlight_inline_code_when_rendered_plain_does_yield_content(text: str, expected: str):
    assert render_plain(highlight_inline_code(text)) == expected


# ---------------------------------------------------------------------------
# format_hint
# ---------------------------------------------------------------------------

_HINT = "run `gymrat doctor` first"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("`gymrat keep` settles it", "gymrat keep settles it", id="code-first"),
        pytest.param("then run `gymrat keep`", "then run gymrat keep", id="code-last"),
        pytest.param("`up``on`", "upon", id="adjacent-code"),
        pytest.param("counts [i] rounds", "counts [i] rounds", id="brackets-in-prose"),
        pytest.param("counts `[i]` rounds", "counts [i] rounds", id="brackets-in-code"),
    ],
)
def test_format_hint_when_given_text_does_render_code_spans_as_literal_text(
    text: str, expected: str
):
    assert render_plain(format_hint(text)) == expected


def test_format_hint_when_code_spans_bracket_prose_does_paint_only_the_spans_blue():
    line = render_colored(format_hint("`a` or `b`"))

    assert (styles_at(line, "a"), styles_at(line, " or "), styles_at(line, "b", last=True)) == (
        ["2", "34"],
        ["2"],
        ["2", "34"],
    )


# ---------------------------------------------------------------------------
# render_lines — color resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "color", "has_ansi"),
    [
        pytest.param({"NO_COLOR": "1"}, True, True, id="color-on-beats-no-color"),
        pytest.param({"FORCE_COLOR": "1"}, False, False, id="color-off-beats-force-color"),
        pytest.param({"NO_COLOR": "1"}, None, False, id="unset-follows-no-color"),
        pytest.param({"FORCE_COLOR": "1"}, None, True, id="unset-follows-force-color"),
        pytest.param({}, None, False, id="unset-without-env-captures-plain"),
        pytest.param({"TERM": "dumb"}, True, True, id="color-on-beats-dumb-terminal"),
    ],
)
def test_render_lines_when_color_and_env_vary_does_resolve_ansi(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], color: bool | None, has_ansi: bool
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    result = render_lines("[red]hi[/red]", color=color)

    assert ("\x1b[" in result, strip_ansi(result)) == (has_ansi, "hi")


def test_render_lines_when_invoked_does_not_mutate_os_environ(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NO_COLOR", "1")

    render_lines("[red]hi[/red]", color=True)

    assert os.environ.get("NO_COLOR") == "1"
    assert "FORCE_COLOR" not in os.environ


# ---------------------------------------------------------------------------
# render_lines — layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        pytest.param(
            "x" * (RENDER_WIDTH + 100), "x" * (RENDER_WIDTH + 100), id="wider-than-width-unwrapped"
        ),
        pytest.param("hi", "hi", id="shorter-than-width-no-trailing-space"),
        pytest.param(escape("[i]"), "[i]", id="escaped-markup-metacharacters"),
        pytest.param("lat:100:p99", "lat:100:p99", id="colon-word-not-emoji"),
    ],
)
def test_render_lines_when_plain_does_render_text_verbatim(markup: str, expected: str):
    result = render_lines(markup, color=False)

    assert result == expected


def test_render_lines_when_given_multiple_renderables_does_join_with_newlines():
    result = render_lines("line1", "line2", color=False)

    assert result == "line1\nline2"


# ---------------------------------------------------------------------------
# make_capture_console
# ---------------------------------------------------------------------------


def test_make_capture_console_when_term_dumb_does_keep_the_render_width(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TERM", "dumb")

    console = make_capture_console(color=True)

    assert console.width == RENDER_WIDTH
