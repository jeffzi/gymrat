"""Tests for the report style/color primitives.

These port the surviving cases from the TypeScript ``format-style`` test suite
(code-point splitting only — the grapheme-cluster/ZWJ-emoji cases are out of
scope) plus the Python-specific color-resolution and capture-rendering tests.
"""

import io
import os

import pytest
from rich.markup import escape

from gymrat_py.report.format import DisplayClass
from gymrat_py.report.style import (
    AGGREGATE_LABEL_STYLE,
    GROUP_LABEL_STYLE,
    LABEL_DISPLAY_WIDTH,
    VARIANT_NAME_STYLE,
    VERDICT_STYLES,
    format_hint_label,
    highlight_inline_code,
    make_capture_console,
    render_lines,
    shorten_label,
    truncate_labels,
)


def _sgr_params(text: str) -> str:
    """The SGR parameter list of the last ANSI escape in ``text`` (e.g. ``"4;33"``).

    Used to assert which attributes a styled span carries without pinning the
    exact escape bytes rich emits (which differ from Node's styleText).
    """
    start = text.rindex("\x1b[")
    end = text.index("m", start)
    return text[start + 2 : end]


# ---------------------------------------------------------------------------
# shorten_label
# ---------------------------------------------------------------------------

_TEXT = "abcdefghijklmnop"


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
        pytest.param(9, "abcd…mnop", id="odd-width-splits-evenly"),  # cspell:disable-line
        pytest.param(8, "abcd…nop", id="even-width-favors-head"),
        pytest.param(2, "a…", id="tail-squeezed-out"),
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


@pytest.mark.parametrize(
    ("text", "max_width", "expected"),
    [
        pytest.param("一二三", 6, "一二三", id="fits-by-cells-verbatim"),
        pytest.param("一二三", 4, "一…", id="overflows-by-cells-truncates"),
        pytest.param("一二三四五六", 9, "一二…五六", id="wide-middle-ellipsis"),
    ],
)
def test_shorten_label_when_measuring_wide_chars_does_use_terminal_cells(
    text: str, max_width: int, expected: str
):
    assert shorten_label(text, max_width) == expected


# ---------------------------------------------------------------------------
# truncate_labels
# ---------------------------------------------------------------------------


def test_label_display_width_is_twenty():
    assert LABEL_DISPLAY_WIDTH == 20


def test_truncate_labels_when_every_label_fits_does_return_verbatim():
    labels = ["main", "feature/short-branch"]

    assert truncate_labels(labels) == ["main", "feature/short-branch"]


def test_truncate_labels_when_a_label_overflows_does_join_head_and_tail():
    result = truncate_labels(["feature/entity-spawn-fastpath"])

    assert result == ["feature/en…-fastpath"]
    assert len(result[0]) == 20


def test_truncate_labels_when_two_collide_does_widen_until_distinct():
    result = truncate_labels(
        [
            "feature/experiment-one-fastpath",
            "feature/exploration-two-fastpath",
        ]
    )

    assert result == ["feature/ex…e-fastpath", "feature/ex…o-fastpath"]


def test_truncate_labels_when_widening_past_a_fitting_label_does_not_lengthen_it():
    short_enough = "release/candidate-2.1"

    result = truncate_labels(
        [
            "feature/experiment-one-fastpath",
            "feature/exploration-two-fastpath",
            short_enough,
        ]
    )

    assert result == [
        "feature/ex…e-fastpath",
        "feature/ex…o-fastpath",
        short_enough,
    ]


# ---------------------------------------------------------------------------
# style constants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("display_class", "expected"),
    [
        pytest.param("improved", "green", id="improved"),
        pytest.param("regressed", "red", id="regressed"),
        pytest.param("unstable", "yellow", id="unstable"),
        pytest.param("identical", "cyan", id="identical"),
        pytest.param("within-noise", "dim", id="within-noise"),
        pytest.param("inconclusive", "dim", id="inconclusive"),
    ],
)
def test_verdict_styles_maps_display_class_to_style(display_class: DisplayClass, expected: str):
    assert VERDICT_STYLES[display_class] == expected


def test_variant_name_style_is_bold_underline():
    assert VARIANT_NAME_STYLE == "bold underline"


def test_group_label_style_is_blue():
    assert GROUP_LABEL_STYLE == "blue"


def test_aggregate_label_style_is_bold():
    assert AGGREGATE_LABEL_STYLE == "bold"


# ---------------------------------------------------------------------------
# format_hint_label
# ---------------------------------------------------------------------------


def test_format_hint_label_when_rendered_plain_does_read_hint_colon():
    assert render_lines(format_hint_label(), color=False, width=80) == "Hint:"


def test_format_hint_label_when_colored_does_underline_hint_but_not_colon():
    styled = render_lines(format_hint_label(), color=True, width=80)

    # Assert the *intent* (underline reaches the word but stops before the
    # colon), not literal Node styleText bytes: rich renders each styled span as
    # a combined SGR (e.g. "\x1b[4;33m") followed by a full reset, and never
    # emits the incremental underline-off "\x1b[24m" that Node's styleText does.
    assert "\x1b[" in styled
    assert "4" in _sgr_params(styled[: styled.index("Hint")])  # underline on the word
    assert "4" not in _sgr_params(styled[: styled.index(":")])  # colon colored, not underlined


# ---------------------------------------------------------------------------
# highlight_inline_code
# ---------------------------------------------------------------------------


def test_highlight_inline_code_when_span_present_does_strip_backticks_and_keep_content():
    result = highlight_inline_code("Run `gymrat doctor` to verify.")

    assert "`" not in result
    assert "gymrat doctor" in result


def test_highlight_inline_code_when_colored_does_emit_ansi_around_content():
    styled = render_lines(
        highlight_inline_code("Run `gymrat doctor` to verify."),
        color=True,
        width=80,
    )

    assert "\x1b[" in styled
    assert "gymrat doctor" in styled
    assert "`" not in styled


def test_highlight_inline_code_when_suppressed_does_render_bare_content():
    plain = render_lines(
        highlight_inline_code("Run `gymrat doctor` to verify."),
        color=False,
        width=80,
    )

    assert plain == "Run gymrat doctor to verify."


def test_highlight_inline_code_when_multiple_spans_does_render_each_bare():
    plain = render_lines(
        highlight_inline_code("Use `gymrat compare` or `gymrat measure`."),
        color=False,
        width=80,
    )

    assert plain == "Use gymrat compare or gymrat measure."


def test_highlight_inline_code_when_no_backticks_does_return_unchanged():
    assert highlight_inline_code("No inline code here.") == "No inline code here."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("`gymrat doctor`", "gymrat doctor", id="command-with-spaces"),
        pytest.param("`--bench`", "--bench", id="flag"),
        pytest.param("`gymrat.json`", "gymrat.json", id="path"),
        pytest.param("`runbook`", "runbook", id="single-word"),
    ],
)
def test_highlight_inline_code_when_rendered_plain_does_yield_content(text: str, expected: str):
    assert render_lines(highlight_inline_code(text), color=False, width=80) == expected


def test_highlight_inline_code_when_content_has_markup_metacharacters_render_literally():
    plain = render_lines(
        highlight_inline_code("Metric `[i]` counts."),
        color=False,
        width=80,
    )

    assert plain == "Metric [i] counts."


# ---------------------------------------------------------------------------
# render_lines — color resolution
# ---------------------------------------------------------------------------


def test_render_lines_when_color_true_does_emit_ansi_despite_no_color_env(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")

    result = render_lines("[red]hi[/red]", color=True, width=80)

    assert "\x1b[" in result
    assert "hi" in result


def test_render_lines_when_color_false_does_suppress_ansi_despite_force_color_env(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    result = render_lines("[red]hi[/red]", color=False, width=80)

    assert "\x1b[" not in result
    assert result == "hi"


def test_render_lines_when_color_none_and_no_color_env_does_render_plain(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

    result = render_lines("[red]hi[/red]", color=None, width=80)

    assert "\x1b[" not in result


def test_render_lines_when_color_none_and_force_color_env_does_emit_ansi(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    result = render_lines("[red]hi[/red]", color=None, width=80)

    assert "\x1b[" in result


def test_render_lines_when_color_none_and_both_env_set_does_let_force_color_win(
    monkeypatch: pytest.MonkeyPatch,
):
    # Parity with the oracle: FORCE_COLOR beats NO_COLOR when both are set.
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    result = render_lines("[red]hi[/red]", color=None, width=80)

    assert "\x1b[" in result


def test_render_lines_when_color_none_and_no_env_and_capture_does_render_plain(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)

    result = render_lines("[red]hi[/red]", color=None, width=80)

    assert "\x1b[" not in result


def test_render_lines_does_not_mutate_os_environ(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    render_lines("[red]hi[/red]", color=True, width=80)

    assert os.environ.get("NO_COLOR") == "1"
    assert "FORCE_COLOR" not in os.environ


# ---------------------------------------------------------------------------
# render_lines — layout
# ---------------------------------------------------------------------------


def test_render_lines_when_content_exceeds_width_does_not_soft_wrap():
    long_line = "x" * 100

    result = render_lines(long_line, color=False, width=10)

    assert result == long_line


def test_render_lines_when_given_multiple_renderables_does_join_with_newlines():
    result = render_lines("line1", "line2", color=False, width=80)

    assert result == "line1\nline2"


def test_render_lines_does_not_emit_trailing_whitespace():
    result = render_lines("hi", color=False, width=80)

    assert result == "hi"


def test_render_lines_when_text_escaped_does_render_markup_metacharacters_literally():
    result = render_lines(escape("[i]"), color=False, width=80)

    assert result == "[i]"


# ---------------------------------------------------------------------------
# make_capture_console
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("color", "has_ansi"),
    [
        pytest.param(True, True, id="color-on-emits-ansi"),
        pytest.param(False, False, id="color-off-plain"),
    ],
)
def test_make_capture_console_honors_color_and_captures_output(color: bool, has_ansi: bool):
    console = make_capture_console(color=color, width=80)

    console.print("[red]hi[/red]")

    assert isinstance(console.file, io.StringIO)
    captured = console.file.getvalue()
    assert "hi" in captured
    assert ("\x1b[" in captured) is has_ansi


# ---------------------------------------------------------------------------
# report package re-exports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "shorten_label",
        "truncate_labels",
        "render_lines",
        "make_capture_console",
        "highlight_inline_code",
        "format_hint_label",
        "VERDICT_STYLES",
        "VARIANT_NAME_STYLE",
        "GROUP_LABEL_STYLE",
        "AGGREGATE_LABEL_STYLE",
        "LABEL_DISPLAY_WIDTH",
    ],
)
def test_report_package_reexports_public_style_name(name: str):
    from gymrat_py import report
    from gymrat_py.report import style

    assert getattr(report, name) is getattr(style, name)
