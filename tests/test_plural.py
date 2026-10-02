"""Tests for counting a noun with its English plural form."""

import pytest

from gymrat.plural import pluralize

# ---------------------------------------------------------------------------
# pluralize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "noun",
    [
        "file",
        "metric",
        "iteration",
        "keep",
        "sample",
        "worktree",
        "edit",
        "warning",
        "failure",
        "kept iteration",
        "uncommitted file",
    ],
)
def test_pluralize_when_count_is_plural_does_append_s(noun: str):
    assert pluralize(2, noun) == f"2 {noun}s"


@pytest.mark.parametrize(
    ("noun", "expected"),
    [
        pytest.param("pass", "2 passes", id="ends-in-s"),
        pytest.param("box", "2 boxes", id="ends-in-x"),
        pytest.param("buzz", "2 buzzes", id="ends-in-z"),
        pytest.param("branch", "2 branches", id="ends-in-ch"),
        pytest.param("dish", "2 dishes", id="ends-in-sh"),
        pytest.param("query", "2 queries", id="consonant-then-y"),
        pytest.param("key", "2 keys", id="vowel-then-y"),
    ],
)
def test_pluralize_when_count_is_plural_does_apply_english_suffix_rules(noun: str, expected: str):
    assert pluralize(2, noun) == expected


@pytest.mark.parametrize("noun", ["pass", "query", "box", "metric", "kept iteration"])
def test_pluralize_when_count_is_one_does_leave_noun_unchanged(noun: str):
    assert pluralize(1, noun) == f"1 {noun}"


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        pytest.param(0, "0 passes", id="zero"),
        pytest.param(2, "2 passes", id="many"),
        pytest.param(-1, "-1 passes", id="negative"),
    ],
)
def test_pluralize_when_count_is_not_one_does_use_the_plural_form(count: int, expected: str):
    assert pluralize(count, "pass") == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        pytest.param(1, "1 index", id="singular-keeps-noun"),
        pytest.param(2, "2 indices", id="plural-takes-override"),
        pytest.param(0, "0 indices", id="zero-takes-override"),
    ],
)
def test_pluralize_when_plural_given_does_override_the_suffix_rules(count: int, expected: str):
    assert pluralize(count, "index", "indices") == expected
