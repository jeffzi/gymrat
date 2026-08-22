"""Tests for the parsed-JSON parity comparator."""

import pytest

from tools.parity.differ import MISSING, DiffEntry, diff_json

# ---------------------------------------------------------------------------
# equal structures
# ---------------------------------------------------------------------------


def test_diff_json_when_structures_deeply_equal_does_report_green():
    left: dict[str, object] = {
        "samples": 10,
        "metrics": {"lat/time": {"baseline": {"median": 1.5}, "candidates": [1, 2]}},
        "empty": None,
        "nested": {"a": [{"b": None}]},
    }
    right: dict[str, object] = {
        "samples": 10,
        "metrics": {"lat/time": {"baseline": {"median": 1.5}, "candidates": [1, 2]}},
        "empty": None,
        "nested": {"a": [{"b": None}]},
    }

    report = diff_json(left, right)

    assert report.differences == ()
    assert report.p_notes == ()
    assert report.is_green is True


# ---------------------------------------------------------------------------
# scalar differences
# ---------------------------------------------------------------------------


def test_diff_json_when_top_level_scalar_differs_does_report_dotted_path():
    report = diff_json({"samples": 10}, {"samples": 12})

    assert report.differences == (DiffEntry(path="samples", left=10, right=12),)
    assert report.is_green is False


def test_diff_json_when_nested_scalar_differs_does_report_full_dotted_path():
    left = {"metrics": {"lat/time": {"baseline": {"median": 1.5}}}}
    right = {"metrics": {"lat/time": {"baseline": {"median": 2.0}}}}

    report = diff_json(left, right)

    assert report.differences == (
        DiffEntry(path="metrics.lat/time.baseline.median", left=1.5, right=2.0),
    )
    assert report.is_green is False


# ---------------------------------------------------------------------------
# lists
# ---------------------------------------------------------------------------


def test_diff_json_when_list_element_differs_does_report_index_segment():
    report = diff_json({"candidates": [1, 2, 3]}, {"candidates": [1, 9, 3]})

    assert report.differences == (DiffEntry(path="candidates[1]", left=2, right=9),)
    assert report.is_green is False


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        pytest.param(
            {"xs": [1, 2, 3]},
            {"xs": [1, 2]},
            DiffEntry(path="xs[2]", left=3, right=MISSING),
            id="left-longer",
        ),
        pytest.param(
            {"xs": [1, 2]},
            {"xs": [1, 2, 3]},
            DiffEntry(path="xs[2]", left=MISSING, right=3),
            id="right-longer",
        ),
    ],
)
def test_diff_json_when_list_lengths_differ_does_report_trailing_indices(
    left: dict[str, object], right: dict[str, object], expected: DiffEntry
):
    report = diff_json(left, right)

    assert report.differences == (expected,)
    assert report.is_green is False


# ---------------------------------------------------------------------------
# missing keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        pytest.param(
            {"a": 1, "b": 2},
            {"a": 1},
            DiffEntry(path="b", left=2, right=MISSING),
            id="missing-on-right",
        ),
        pytest.param(
            {"a": 1},
            {"a": 1, "b": 2},
            DiffEntry(path="b", left=MISSING, right=2),
            id="missing-on-left",
        ),
    ],
)
def test_diff_json_when_key_absent_on_one_side_does_report_missing_sentinel(
    left: dict[str, object], right: dict[str, object], expected: DiffEntry
):
    report = diff_json(left, right)

    assert report.differences == (expected,)
    assert report.is_green is False


def test_diff_json_when_key_none_on_both_sides_does_report_green():
    report = diff_json({"a": None}, {"a": None})

    assert report.differences == ()
    assert report.is_green is True


# ---------------------------------------------------------------------------
# type mismatches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        pytest.param(
            {"node": {"x": 1}},
            {"node": [1, 2]},
            DiffEntry(path="node", left={"x": 1}, right=[1, 2]),
            id="dict-vs-list",
        ),
        pytest.param(
            {"node": {"x": 1}},
            {"node": 5},
            DiffEntry(path="node", left={"x": 1}, right=5),
            id="dict-vs-scalar",
        ),
    ],
)
def test_diff_json_when_dict_versus_other_type_does_report_single_difference(
    left: dict[str, object], right: dict[str, object], expected: DiffEntry
):
    report = diff_json(left, right)

    assert report.differences == (expected,)
    assert report.is_green is False


# ---------------------------------------------------------------------------
# volatile worktree paths (always ignored)
# ---------------------------------------------------------------------------


def test_diff_json_when_only_volatile_worktree_fields_differ_does_report_green():
    left = {
        "worktrees": {
            "leftBehind": [{"path": "/tmp/a"}, {"path": "/tmp/b"}],
            "pruneError": "boom",
        }
    }
    right = {
        "worktrees": {
            "leftBehind": [{"path": "/var/x"}, {"path": "/var/y"}],
            "pruneError": None,
        }
    }

    report = diff_json(left, right)

    assert report.differences == ()
    assert report.p_notes == ()
    assert report.is_green is True


def test_diff_json_when_worktrees_removed_differs_does_report_difference():
    left = {"worktrees": {"removed": 1, "pruneError": "boom"}}
    right = {"worktrees": {"removed": 2, "pruneError": None}}

    report = diff_json(left, right)

    assert report.differences == (DiffEntry(path="worktrees.removed", left=1, right=2),)
    assert report.is_green is False


# ---------------------------------------------------------------------------
# caller-supplied ignore_paths
# ---------------------------------------------------------------------------


def test_diff_json_when_ignore_path_matches_does_suppress_difference():
    left = {"a": {"b": 1}}
    right = {"a": {"b": 2}}

    report = diff_json(left, right, ignore_paths=("a.b",))

    assert report.differences == ()
    assert report.is_green is True


def test_diff_json_when_ignore_pattern_uses_segment_wildcard_does_suppress_all_metrics():
    left = {
        "metrics": {
            "lat/time": {"baseline": {"spreadPct": 1.0, "median": 5.0}},
            "mem=peak": {"baseline": {"spreadPct": 3.0, "median": 5.0}},
        }
    }
    right = {
        "metrics": {
            "lat/time": {"baseline": {"spreadPct": 2.0, "median": 5.0}},
            "mem=peak": {"baseline": {"spreadPct": 9.0, "median": 5.0}},
        }
    }

    report = diff_json(left, right, ignore_paths=("metrics.*.baseline.spreadPct",))

    assert report.differences == ()
    assert report.is_green is True


def test_diff_json_when_ignore_pattern_uses_index_wildcard_does_suppress_all_elements():
    left = {"xs": [{"v": 1}, {"v": 2}]}
    right = {"xs": [{"v": 8}, {"v": 9}]}

    report = diff_json(left, right, ignore_paths=("xs[*].v",))

    assert report.differences == ()
    assert report.is_green is True


# ---------------------------------------------------------------------------
# p-value paths are informational
# ---------------------------------------------------------------------------


def test_diff_json_when_only_p_value_paths_differ_does_report_green_with_notes():
    left = {"metrics": {"lat/time": {"candidates": [{"p": 0.01}, {"p": 0.5}]}}}
    right = {"metrics": {"lat/time": {"candidates": [{"p": 0.99}, {"p": 0.5}]}}}

    report = diff_json(left, right)

    assert report.differences == ()
    assert report.p_notes == (
        DiffEntry(path="metrics.lat/time.candidates[0].p", left=0.01, right=0.99),
    )
    assert report.is_green is True


def test_diff_json_when_p_value_and_other_path_differ_does_split_and_report_red():
    left = {"metrics": {"lat/time": {"candidates": [{"p": 0.01, "median": 5.0}]}}}
    right = {"metrics": {"lat/time": {"candidates": [{"p": 0.99, "median": 6.0}]}}}

    report = diff_json(left, right)

    assert report.differences == (
        DiffEntry(path="metrics.lat/time.candidates[0].median", left=5.0, right=6.0),
    )
    assert report.p_notes == (
        DiffEntry(path="metrics.lat/time.candidates[0].p", left=0.01, right=0.99),
    )
    assert report.is_green is False


def test_diff_json_when_p_path_also_in_ignore_paths_does_ignore_not_note():
    left = {"metrics": {"lat/time": {"candidates": [{"p": 0.01}]}}}
    right = {"metrics": {"lat/time": {"candidates": [{"p": 0.99}]}}}

    report = diff_json(left, right, ignore_paths=("metrics.*.candidates[*].p",))

    assert report.differences == ()
    assert report.p_notes == ()
    assert report.is_green is True
