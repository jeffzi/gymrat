"""Tests for the deterministic parity-harness fixture builders and matrix."""

import json
import subprocess
from pathlib import Path

import pytest

from tools.parity.fixtures import (
    Fixture,
    create_scratch_repo,
    create_two_ref_repo,
    fixture_matrix,
    write_config,
    write_counter_emitter,
    write_metric_lines_emitter,
    write_mitata_emitter,
)
from tools.parity.oracle import OracleRunner, ensure_built, ts_repo_path

_EXPECTED_FIXTURE_NAMES = {
    "metric_lines_compare_band",
    "metric_lines_measure",
    "mitata_compare",
    "signed_rank_compare",
    "exact_compare",
    "multi_candidate_compare",
    "zero_diff_compare",
    "nan_delta_compare",
    "one_sided_metric_compare",
    "ref_compare",
    "ref_measure",
    "measure_record",
    "measure_record_missing_session",
}

# measure_record_missing_session drives measure --record with no open session, so
# the reference binary is meant to reject it (exit 2). It is a compare-only
# fixture — both sides fail identically and compare's exit-code agreement passes
# it — so the success-path checks below run every other fixture.
_ORACLE_SUCCESS_FIXTURES = tuple(
    fixture for fixture in fixture_matrix() if fixture.name != "measure_record_missing_session"
)


def _git(cwd: Path, *args: str) -> str:
    """Run ``git`` in ``cwd`` and return stripped stdout."""
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _run_sh(cwd: Path, script: str) -> str:
    """Run ``sh <script>`` in ``cwd`` and return its stdout."""
    return subprocess.run(  # noqa: S603
        ["sh", script],  # noqa: S607
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _parse_metric_lines(stdout: str) -> dict[str, float]:
    """Parse ``METRIC <name>=<value>`` lines into a name->value mapping."""
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        if not line.startswith("METRIC "):
            continue
        body = line[len("METRIC ") :]
        name, _, value = body.rpartition("=")
        metrics[name] = float(value)
    return metrics


# ---------------------------------------------------------------------------
# create_scratch_repo
# ---------------------------------------------------------------------------


def test_create_scratch_repo_when_run_does_reproduce_the_pinned_recipe(tmp_path: Path):
    create_scratch_repo(tmp_path)

    assert _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert _git(tmp_path, "config", "--get", "user.name") == "Test User"
    assert _git(tmp_path, "config", "--get", "user.email") == "test@example.com"
    assert _git(tmp_path, "config", "--get", "commit.gpgsign") == "false"
    assert _git(tmp_path, "config", "--get", "core.autocrlf") == "false"
    assert _git(tmp_path, "rev-list", "--count", "HEAD") == "1"
    assert _git(tmp_path, "ls-files") == "README.md"
    assert (tmp_path / "README.md").read_text() == "# Test Repo\n"


# ---------------------------------------------------------------------------
# create_two_ref_repo
# ---------------------------------------------------------------------------


def test_create_two_ref_repo_when_built_does_bench_each_ref_with_its_own_metrics(tmp_path: Path):
    create_two_ref_repo(
        tmp_path,
        main_metrics={"latency/time": 100.0},
        feature_metrics={"latency/time": 80.0},
    )

    assert _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert _git(tmp_path, "rev-parse", "main") != _git(tmp_path, "rev-parse", "feature")
    feature_tree = tmp_path.parent / f"{tmp_path.name}-feature"
    _git(tmp_path, "worktree", "add", "-q", str(feature_tree), "feature")
    assert _parse_metric_lines(_run_sh(tmp_path, "bench.sh")) == {"latency/time": 100.0}
    assert _parse_metric_lines(_run_sh(feature_tree, "bench.sh")) == {"latency/time": 80.0}


# ---------------------------------------------------------------------------
# write_metric_lines_emitter
# ---------------------------------------------------------------------------


def test_write_metric_lines_emitter_when_run_does_print_expected_metric_lines(tmp_path: Path):
    metrics: dict[str, float] = {"latency/time": 100, "mem/heap": 200, "count": 5}
    write_metric_lines_emitter(tmp_path, metrics)

    stdout = _run_sh(tmp_path, "bench.sh")

    assert _parse_metric_lines(stdout) == {name: float(v) for name, v in metrics.items()}


# ---------------------------------------------------------------------------
# write_counter_emitter
# ---------------------------------------------------------------------------


def test_write_counter_emitter_when_run_after_prepare_does_yield_varying_sequence(tmp_path: Path):
    write_counter_emitter(tmp_path, "op/time", base=100, step=10)

    _run_sh(tmp_path, "prepare.sh")
    sequence = [_parse_metric_lines(_run_sh(tmp_path, "bench.sh"))["op/time"] for _ in range(3)]
    _run_sh(tmp_path, "prepare.sh")
    after_reset = _parse_metric_lines(_run_sh(tmp_path, "bench.sh"))["op/time"]

    assert sequence == [100.0, 110.0, 120.0]
    assert after_reset == 100.0


# ---------------------------------------------------------------------------
# write_mitata_emitter
# ---------------------------------------------------------------------------


def test_write_mitata_emitter_when_run_does_emit_parseable_mitata_payload(tmp_path: Path):
    benchmarks: dict[str, dict[str, float]] = {
        "encode": {"time": 42, "heap": 1024},
        "decode": {"time": 100},
    }
    write_mitata_emitter(tmp_path, benchmarks)

    document = json.loads(_run_sh(tmp_path, "bench.sh"))

    by_alias = {entry["alias"]: entry for entry in document["benchmarks"]}
    assert set(by_alias) == {"encode", "decode"}
    assert by_alias["encode"]["runs"][0]["stats"]["p50"] == 42
    assert by_alias["encode"]["runs"][0]["stats"]["heap"]["avg"] == 1024
    assert by_alias["decode"]["runs"][0]["stats"]["p50"] == 100
    assert "heap" not in by_alias["decode"]["runs"][0]["stats"]


# ---------------------------------------------------------------------------
# write_config
# ---------------------------------------------------------------------------


def test_write_config_when_written_does_round_trip_to_given_dict(tmp_path: Path):
    config: dict[str, object] = {"metrics": {"count": {"exact": True}}}
    write_config(tmp_path, config)

    assert json.loads((tmp_path / "gymrat.json").read_text()) == config


# ---------------------------------------------------------------------------
# fixture_matrix
# ---------------------------------------------------------------------------


def test_fixture_matrix_when_called_does_return_all_named_entries():
    matrix = fixture_matrix()

    assert isinstance(matrix, tuple)
    assert all(isinstance(fixture, Fixture) for fixture in matrix)
    assert {fixture.name for fixture in matrix} == _EXPECTED_FIXTURE_NAMES


# ---------------------------------------------------------------------------
# integration: every fixture drives the oracle to a valid JSON document
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", _ORACLE_SUCCESS_FIXTURES, ids=lambda f: f.name)
def test_fixture_matrix_entry_when_run_through_oracle_does_return_expected_schema(
    fixture: Fixture, requires_oracle: None, tmp_path: Path
):
    fixture.build(tmp_path)
    runner = OracleRunner(ensure_built(ts_repo_path()))

    result = runner.run(list(fixture.argv), cwd=tmp_path)

    assert result.exit_code == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["schemaVersion"] == fixture.schema_version
