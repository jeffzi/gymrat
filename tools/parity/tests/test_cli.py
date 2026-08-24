"""Tests for the parity-harness command-line entry point.

Behaviors 2-6 are hermetic: they substitute a scripted fake ``Runner`` for the
node-backed oracle via ``_build_oracle_runner`` (or, for ``compare``, lean on
``PortRunner`` raising before it would spawn node). Behavior 1 is the phase exit
criterion and runs the real reference binary on both sides of the whole matrix;
it is guarded by ``requires_oracle`` so it skips cleanly without node.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gymrat_py.errors import GymratError
from tools.parity import cli
from tools.parity.cli import app
from tools.parity.fixtures import fixture_matrix
from tools.parity.oracle import RunResult

_runner = CliRunner()


class _FakeRunner:
    """Scripted ``Runner`` that returns payloads by call index and records calls.

    The payload for call ``i`` is ``payloads[min(i, len(payloads) - 1)]`` so a
    single-element list yields a constant output (identical both sides ⇒ green)
    while a two-element list scripts a first-vs-second difference for one fixture.
    """

    def __init__(self, payloads: Sequence[str]) -> None:
        self._payloads = list(payloads)
        self.calls: list[tuple[list[str], Path]] = []

    def run(self, args: Sequence[str], cwd: Path) -> RunResult:
        index = min(len(self.calls), len(self._payloads) - 1)
        self.calls.append((list(args), cwd))
        return RunResult(exit_code=0, stdout=self._payloads[index], stderr="")


# ---------------------------------------------------------------------------
# self-diff — real reference binary on both sides (phase exit criterion)
# ---------------------------------------------------------------------------


def test_self_diff_when_run_over_full_matrix_does_report_green(requires_oracle: None):
    result = _runner.invoke(app, ["self-diff"])

    assert result.exit_code == 0, result.stdout
    for fixture in fixture_matrix():
        assert fixture.name in result.stdout


# ---------------------------------------------------------------------------
# self-diff — hermetic green / red / p-notes paths
# ---------------------------------------------------------------------------


def test_self_diff_when_both_sides_identical_does_exit_zero_and_name_each_fixture(
    monkeypatch: pytest.MonkeyPatch,
):
    names = [fixture.name for fixture in fixture_matrix()[:2]]
    payload = json.dumps({"schemaVersion": 1, "samples": 10})
    fake = _FakeRunner([payload])
    monkeypatch.setattr(cli, "_build_oracle_runner", lambda: fake)

    result = _runner.invoke(app, ["self-diff", *names])

    assert result.exit_code == 0, result.stdout
    for name in names:
        assert name in result.stdout


def test_self_diff_when_sides_differ_does_exit_one_and_show_path_and_both_values(
    monkeypatch: pytest.MonkeyPatch,
):
    name = fixture_matrix()[0].name
    fake = _FakeRunner([json.dumps({"samples": 10}), json.dumps({"samples": 12})])
    monkeypatch.setattr(cli, "_build_oracle_runner", lambda: fake)

    result = _runner.invoke(app, ["self-diff", name])

    assert result.exit_code == 1
    assert "samples" in result.stdout
    assert "10" in result.stdout
    assert "12" in result.stdout


def test_self_diff_when_only_p_values_differ_does_exit_zero_and_surface_note(
    monkeypatch: pytest.MonkeyPatch,
):
    name = fixture_matrix()[0].name
    left = json.dumps({"metrics": {"lat/time": {"candidates": [{"p": 0.01}]}}})
    right = json.dumps({"metrics": {"lat/time": {"candidates": [{"p": 0.99}]}}})
    fake = _FakeRunner([left, right])
    monkeypatch.setattr(cli, "_build_oracle_runner", lambda: fake)

    result = _runner.invoke(app, ["self-diff", name])

    assert result.exit_code == 0, result.stdout
    assert "metrics.lat/time.candidates[0].p" in result.stdout


def test_self_diff_when_fixture_name_unknown_does_error_before_running_runner(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeRunner([json.dumps({})])
    monkeypatch.setattr(cli, "_build_oracle_runner", lambda: fake)

    result = _runner.invoke(app, ["self-diff", "definitely_not_a_fixture"])

    assert result.exit_code != 0
    assert "definitely_not_a_fixture" in result.stdout
    assert fake.calls == []


# ---------------------------------------------------------------------------
# _build_oracle_runner — enforce the pinned SHA before building
# ---------------------------------------------------------------------------


def test_build_oracle_runner_when_repo_not_at_pinned_sha_does_raise_before_building(
    monkeypatch: pytest.MonkeyPatch,
):
    repo = Path("/fake/ts-repo")
    calls: list[str] = []

    def fake_assert(path: Path) -> None:
        assert path == repo
        calls.append("assert")
        msg = "reference checkout is not at the pinned commit"
        raise GymratError(msg)

    def fake_build(path: Path, *, force: bool = False) -> Path:
        calls.append("build")
        return Path("/fake/ts-repo/dist/cli.js")

    monkeypatch.setattr(cli, "ts_repo_path", lambda: repo)
    monkeypatch.setattr(cli, "assert_pinned_sha", fake_assert)
    monkeypatch.setattr(cli, "ensure_built", fake_build)

    with pytest.raises(GymratError):
        cli._build_oracle_runner()

    assert calls == ["assert"]


def test_build_oracle_runner_when_repo_at_pinned_sha_does_build_and_return_runner(
    monkeypatch: pytest.MonkeyPatch,
):
    repo = Path("/fake/ts-repo")
    dist = Path("/fake/ts-repo/dist/cli.js")
    calls: list[tuple[str, Path]] = []

    def fake_assert(path: Path) -> None:
        calls.append(("assert", path))

    def fake_build(path: Path, *, force: bool = False) -> Path:
        calls.append(("build", path))
        return dist

    monkeypatch.setattr(cli, "ts_repo_path", lambda: repo)
    monkeypatch.setattr(cli, "assert_pinned_sha", fake_assert)
    monkeypatch.setattr(cli, "ensure_built", fake_build)

    result = cli._build_oracle_runner()

    assert isinstance(result, cli.OracleRunner)
    assert calls == [("assert", repo), ("build", repo)]


# ---------------------------------------------------------------------------
# compare — real reference binary vs the port over the whole matrix (the gate)
# ---------------------------------------------------------------------------


def test_compare_when_run_over_full_matrix_does_report_green(requires_oracle: None):
    result = _runner.invoke(app, ["compare"])

    assert result.exit_code == 0, result.stdout
    for fixture in fixture_matrix():
        assert fixture.name in result.stdout


# ---------------------------------------------------------------------------
# compare — hermetic oracle-vs-port document and exit-code diffing
# ---------------------------------------------------------------------------


class _StubRunner:
    """Constant-payload ``Runner`` returning a fixed exit code and JSON document."""

    def __init__(self, payload: str, exit_code: int = 0) -> None:
        self._payload = payload
        self._exit_code = exit_code
        self.calls: list[tuple[list[str], Path]] = []

    def run(self, args: Sequence[str], cwd: Path) -> RunResult:
        self.calls.append((list(args), cwd))
        return RunResult(exit_code=self._exit_code, stdout=self._payload, stderr="")


def test_compare_when_both_sides_identical_does_exit_zero_and_name_each_fixture(
    monkeypatch: pytest.MonkeyPatch,
):
    names = [fixture.name for fixture in fixture_matrix()[:2]]
    payload = json.dumps({"schemaVersion": 1, "samples": 10})
    monkeypatch.setattr(cli, "_build_oracle_runner", lambda: _StubRunner(payload))
    monkeypatch.setattr(cli, "_build_port_runner", lambda: _StubRunner(payload))

    result = _runner.invoke(app, ["compare", *names])

    assert result.exit_code == 0, result.stdout
    for name in names:
        assert name in result.stdout


def test_compare_when_documents_differ_does_exit_one_and_show_path_and_both_values(
    monkeypatch: pytest.MonkeyPatch,
):
    name = fixture_matrix()[0].name
    monkeypatch.setattr(
        cli, "_build_oracle_runner", lambda: _StubRunner(json.dumps({"samples": 10}))
    )
    monkeypatch.setattr(cli, "_build_port_runner", lambda: _StubRunner(json.dumps({"samples": 12})))

    result = _runner.invoke(app, ["compare", name])

    assert result.exit_code == 1
    assert "samples" in result.stdout
    assert "10" in result.stdout
    assert "12" in result.stdout


def test_compare_when_exit_codes_differ_does_exit_one_and_flag_the_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    name = fixture_matrix()[0].name
    payload = json.dumps({"schemaVersion": 1, "samples": 10})
    monkeypatch.setattr(cli, "_build_oracle_runner", lambda: _StubRunner(payload, exit_code=0))
    monkeypatch.setattr(cli, "_build_port_runner", lambda: _StubRunner(payload, exit_code=1))

    result = _runner.invoke(app, ["compare", name])

    assert result.exit_code == 1
    assert "exit" in result.stdout.lower()
