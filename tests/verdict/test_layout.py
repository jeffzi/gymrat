"""Tests for the layout and public surface of the ``gymrat.verdict`` module."""

from pathlib import Path

import gymrat.verdict


def test_verdict_when_imported_does_load_from_single_flat_module_file():
    module_file = Path(gymrat.verdict.__file__)

    assert (module_file.parent.name, module_file.name) == ("gymrat", "verdict.py")


_PUBLIC_NAMES = frozenset({
    "GroupAggregate",
    "KindAggregate",
    "compute_geomean",
    "compute_kind_aggregates",
    "compute_verdicts",
    "infer_group",
})


def test_verdict_when_public_names_resolved_does_export_every_verdict_entry_point():
    exported = {name: getattr(gymrat.verdict, name) for name in gymrat.verdict.__all__}

    assert exported.keys() == _PUBLIC_NAMES
