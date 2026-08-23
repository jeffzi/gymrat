"""Behavioral tests for the benchmark target union."""

import dataclasses

import pytest

from gymrat_py.targets import InPlaceTarget, RefTarget


def test_ref_target_when_constructed_does_carry_ref_and_resolved_sha():
    target = RefTarget(ref="feature", resolved_sha="deadbeef")

    assert target.ref == "feature"
    assert target.resolved_sha == "deadbeef"


def test_in_place_target_when_constructed_does_carry_dir():
    target = InPlaceTarget(dir="/work")

    assert target.dir == "/work"


def test_ref_target_when_field_assigned_does_raise_frozen():
    target = RefTarget(ref="feature", resolved_sha="deadbeef")

    with pytest.raises(dataclasses.FrozenInstanceError):
        target.ref = "other"  # type: ignore[misc]


def test_in_place_target_when_field_assigned_does_raise_frozen():
    target = InPlaceTarget(dir="/work")

    with pytest.raises(dataclasses.FrozenInstanceError):
        target.dir = "/elsewhere"  # type: ignore[misc]


def test_targets_when_discriminated_by_type_does_distinguish_variants():
    variants = [InPlaceTarget(dir="/work"), RefTarget(ref="main", resolved_sha="abc123")]

    kinds = [isinstance(target, RefTarget) for target in variants]

    assert kinds == [False, True]
