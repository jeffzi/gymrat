"""Primary-direction tests for the ``gymrat supervise`` command.

Shares its seam-installation harness with :mod:`tests.cli.commands.supervise.test_supervise`,
which owns the ``CliRunner`` wiring these tests reuse.
"""

from dataclasses import replace
from unittest.mock import patch

import pytest

from gymrat.adapters import MetricDefaults
from gymrat.config import KindEntry, MetricEntry
from tests.cli.commands.supervise.test_supervise import _config, _install_seams, _run
from tests.report._comparisons import metric_meta
from tests.sampling._adapters import make_adapter

# ---------------------------------------------------------------------------
# reporter direction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("primary", "metrics", "expected"),
    [
        pytest.param("geomean", None, "lower", id="geomean-ignores-the-adapter"),
        pytest.param(
            "throughput",
            {"throughput": MetricEntry(direction="lower")},
            "lower",
            id="metric-entry-direction",
        ),
        pytest.param("throughput", None, "higher", id="adapter-default"),
        pytest.param(
            "throughput",
            {"throughput": MetricEntry(gating=False)},
            "higher",
            id="metric-entry-without-direction",
        ),
    ],
)
def test_supervise_when_adapter_defaults_to_higher_does_build_reporter_with_the_primary_direction(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    primary: str,
    metrics: dict[str, MetricEntry] | None,
    expected: str,
):
    config = replace(_config(), primary=primary, metrics=metrics)
    adapters = {config.adapter: make_adapter(lambda _name: MetricDefaults(direction="higher"))}
    monkeypatch.setattr("gymrat.cli.commands.supervise.get_adapter", adapters.__getitem__)
    seams = _install_seams(monkeypatch, config=config)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.reporter_calls[0]["primary_direction"] == expected


def test_supervise_when_engine_resolves_a_named_primary_does_build_reporter_with_that_direction(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    entry = MetricEntry(direction="lower")
    kinds = {"memory": KindEntry(gating=False)}
    config = replace(_config(), primary="throughput", metrics={"throughput": entry}, kinds=kinds)
    adapter = make_adapter()
    adapters = {config.adapter: adapter}
    monkeypatch.setattr("gymrat.cli.commands.supervise.get_adapter", adapters.__getitem__)
    seams = _install_seams(monkeypatch, config=config)

    with patch(
        "gymrat.cli.commands.supervise.resolve_metric_meta",
        autospec=True,
        return_value=metric_meta("throughput", direction="higher"),
    ) as resolver:
        result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.reporter_calls[0]["primary_direction"] == "higher"
    resolver.assert_called_once_with("throughput", entry, adapter, kinds)
