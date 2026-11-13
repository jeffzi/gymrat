"""Tests for the report style primitives: inline-code highlighting, hints and rendering."""

import pytest

from gymrat.report.style import format_hint, highlight_inline_code, render_lines
from tests._ansi import strip_ansi
from tests.report._assertions import render_colored, render_plain, styles_at

# ---------------------------------------------------------------------------
# highlight_inline_code
# ---------------------------------------------------------------------------


def test_highlight_inline_code_when_span_in_prose_does_style_the_span():
    styled = highlight_inline_code("Run `gymrat doctor` to verify.")

    assert styles_at(render_colored(styled), "gymrat doctor") != []


def test_highlight_inline_code_when_rendered_plain_does_drop_the_backticks_of_every_span():
    styled = highlight_inline_code("Use `gymrat compare` or `gymrat measure`.")

    assert render_plain(styled) == "Use gymrat compare or gymrat measure."


def test_highlight_inline_code_when_span_holds_markup_metacharacters_does_keep_them_literal():
    styled = highlight_inline_code("Metric `[i]` counts.")

    assert render_plain(styled) == "Metric [i] counts."


# ---------------------------------------------------------------------------
# format_hint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("`up``on`", "upon", id="adjacent-code"),
        pytest.param("counts [i] rounds", "counts [i] rounds", id="brackets-in-prose"),
        pytest.param("counts `[i]` rounds", "counts [i] rounds", id="brackets-in-code"),
    ],
)
def test_format_hint_when_given_text_does_render_code_spans_as_literal_text(
    text: str, expected: str
):
    assert render_plain(format_hint(text)) == expected


def test_format_hint_when_code_spans_bracket_prose_does_style_the_spans_apart_from_the_prose():
    line = render_colored(format_hint("`a` or `b`"))

    assert styles_at(line, "a") == styles_at(line, "b", last=True) != styles_at(line, " or ")


# ---------------------------------------------------------------------------
# render_lines — color resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "color", "has_ansi"),
    [
        pytest.param({"NO_COLOR": "1"}, True, True, id="color-on-beats-no-color"),
        pytest.param({"FORCE_COLOR": "1"}, False, False, id="color-off-beats-force-color"),
        pytest.param({}, None, False, id="unset-without-env-captures-plain"),
    ],
)
def test_render_lines_when_color_and_env_vary_does_resolve_ansi(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], color: bool | None, has_ansi: bool
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    result = render_lines("[red]hi[/red]", color=color)

    assert ("\x1b[" in result, strip_ansi(result)) == (has_ansi, "hi")


# ---------------------------------------------------------------------------
# render_lines — layout
# ---------------------------------------------------------------------------


def test_render_lines_when_given_multiple_renderables_does_join_with_newlines():
    result = render_lines("line1", "line2", color=False)

    assert result == "line1\nline2"
