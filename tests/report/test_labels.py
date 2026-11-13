"""Tests for label clipping by terminal cells and grapheme clusters."""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from rich.cells import cell_len, split_graphemes

from gymrat.report.labels import shorten_label, truncate_labels

# ---------------------------------------------------------------------------
# shorten_label
# ---------------------------------------------------------------------------

_TEXT = "rain on the lake"


def test_shorten_label_when_text_already_fits_does_return_verbatim():
    assert shorten_label(_TEXT, len(_TEXT)) == _TEXT


@pytest.mark.parametrize(
    ("max_width", "expected"),
    [
        pytest.param(9, "rain…lake", id="odd-width-splits-evenly"),
        pytest.param(8, "rain…ake", id="even-width-favors-head"),
        pytest.param(2, "r…", id="tail-squeezed-out"),
    ],
)
def test_shorten_label_when_text_overflows_does_middle_ellipsis(max_width: int, expected: str):
    assert shorten_label(_TEXT, max_width) == expected


_E_ACUTE = "e\N{COMBINING ACUTE ACCENT}"
_HEART = "\N{HEAVY BLACK HEART}\N{VARIATION SELECTOR-16}"
_FAMILY = "\N{MAN}\N{ZERO WIDTH JOINER}\N{WOMAN}\N{ZERO WIDTH JOINER}\N{GIRL}"


@pytest.mark.parametrize(
    ("text", "max_width", "expected"),
    [
        pytest.param("一二三", 4, "一…", id="overflows-by-cells-truncates"),
        pytest.param("世abc", 3, "…c", id="wide-first-cluster-wider-than-head-share"),
        pytest.param(f"abcdefghij{_E_ACUTE}xyz", 7, "abc…xyz", id="combining-mark-at-tail-edge"),
        pytest.param(f"ab{_HEART}understatement", 7, "ab…ent", id="emoji-selector-at-head-edge"),
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
