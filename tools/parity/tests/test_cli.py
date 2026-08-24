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
from gymrat_py.session.paths import session_jsonl_path
from gymrat_py.session.records import BaselineRecord
from gymrat_py.session.store import append_record
from tools.parity import cli
from tools.parity.cli import app
from tools.parity.differ import DiffEntry
from tools.parity.fixtures import Fixture, fixture_matrix
from tools.parity.oracle import RunResult
from tools.parity.tests.conftest import MISSING_SESSION_FIXTURE

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
# _self_diff_exit_only — distinguish nondeterminism from off-contract exits
# ---------------------------------------------------------------------------


class _ExitStubRunner:
    """Runner returning a scripted exit code per call; stdout/stderr unused.

    The exit code for call ``i`` is ``exit_codes[min(i, len - 1)]`` so a
    single-element list yields a constant exit (identical both runs) while a
    two-element list scripts a first-vs-second difference.
    """

    def __init__(self, exit_codes: Sequence[int]) -> None:
        self._exit_codes = list(exit_codes)
        self.calls: list[tuple[list[str], Path]] = []

    def run(self, args: Sequence[str], cwd: Path) -> RunResult:
        index = min(len(self.calls), len(self._exit_codes) - 1)
        self.calls.append((list(args), cwd))
        return RunResult(exit_code=self._exit_codes[index], stdout="", stderr="")


def _exit_fixture(oracle_exit: int) -> Fixture:
    return Fixture(
        name="exit_probe",
        build=lambda _root: None,
        argv=("measure",),
        schema_version=1,
        oracle_exit=oracle_exit,
    )


def test_self_diff_exit_only_when_stable_but_off_contract_does_flag_expected_path(
    tmp_path: Path,
):
    runner = _ExitStubRunner([1, 1])
    fixture = _exit_fixture(oracle_exit=2)

    report = cli._self_diff_exit_only(runner, fixture, tmp_path)

    assert report.differences == (DiffEntry(path="exit_code.expected", left=2, right=1),)
    assert report.p_notes == ()


def test_self_diff_exit_only_when_runs_disagree_does_flag_exit_code_path(
    tmp_path: Path,
):
    runner = _ExitStubRunner([2, 1])
    fixture = _exit_fixture(oracle_exit=0)

    report = cli._self_diff_exit_only(runner, fixture, tmp_path)

    assert report.differences == (DiffEntry(path="exit_code", left=2, right=1),)
    assert report.p_notes == ()


def test_self_diff_exit_only_when_both_runs_match_oracle_exit_does_report_no_diffs(
    tmp_path: Path,
):
    runner = _ExitStubRunner([1, 1])
    fixture = _exit_fixture(oracle_exit=1)

    report = cli._self_diff_exit_only(runner, fixture, tmp_path)

    assert report.differences == ()
    assert report.p_notes == ()


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


# ---------------------------------------------------------------------------
# record fixtures — matrix registration and cross-implementation log round-trip
# ---------------------------------------------------------------------------


_MEASURE_DOC = json.dumps({"schemaVersion": 1, "label": "target", "samples": 3})
_MATCHING_SAMPLES: tuple[dict[str, float | int], ...] = (
    {"latency/time": 100.0},
    {"latency/time": 101.0},
)


class _RecordingRunner:
    """Runner that appends a baseline record on each ``--record`` run.

    ``exit_codes`` is indexed by call number and its last element repeats, so a
    test can make the oracle's later invocation (the Python->TS reread) fail while
    its first run succeeds. A record is appended only on a successful ``--record``
    run, into ``session_jsonl_path(cwd)`` so oracle and port write to the shared
    root the harness set up.
    """

    def __init__(
        self,
        payload: str,
        samples: tuple[dict[str, float | int], ...],
        exit_codes: Sequence[int] = (0,),
    ) -> None:
        self._payload = payload
        self._samples = samples
        self._exit_codes = list(exit_codes)
        self.calls: list[tuple[list[str], Path]] = []

    def run(self, args: Sequence[str], cwd: Path) -> RunResult:
        index = len(self.calls)
        self.calls.append((list(args), cwd))
        exit_code = self._exit_codes[min(index, len(self._exit_codes) - 1)]
        if exit_code == 0 and "--record" in args:
            record = BaselineRecord(
                type="baseline",
                at="2026-08-08T14:15:31.000Z",
                label="target",
                samples=self._samples,
            )
            append_record(session_jsonl_path(str(cwd)), record)
        return RunResult(exit_code=exit_code, stdout=self._payload, stderr="")


def test_compare_when_record_logs_match_does_exit_zero_and_name_the_fixture(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        cli, "_build_oracle_runner", lambda: _RecordingRunner(_MEASURE_DOC, _MATCHING_SAMPLES)
    )
    monkeypatch.setattr(
        cli, "_build_port_runner", lambda: _RecordingRunner(_MEASURE_DOC, _MATCHING_SAMPLES)
    )

    result = _runner.invoke(app, ["compare", "measure_record"])

    assert result.exit_code == 0, result.stdout
    assert "measure_record" in result.stdout


def test_compare_when_record_samples_differ_does_exit_one_and_name_record_path(
    monkeypatch: pytest.MonkeyPatch,
):
    oracle_samples: tuple[dict[str, float | int], ...] = (
        {"latency/time": 100.0},
        {"latency/time": 101.0},
    )
    port_samples: tuple[dict[str, float | int], ...] = (
        {"latency/time": 100.0},
        {"latency/time": 999.0},
    )
    monkeypatch.setattr(
        cli, "_build_oracle_runner", lambda: _RecordingRunner(_MEASURE_DOC, oracle_samples)
    )
    monkeypatch.setattr(
        cli, "_build_port_runner", lambda: _RecordingRunner(_MEASURE_DOC, port_samples)
    )

    result = _runner.invoke(app, ["compare", "measure_record"])

    assert result.exit_code == 1
    assert "session_log.record." in result.stdout


def test_compare_when_oracle_reread_fails_does_exit_one_and_name_reread_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        cli,
        "_build_oracle_runner",
        lambda: _RecordingRunner(_MEASURE_DOC, _MATCHING_SAMPLES, exit_codes=(0, 1)),
    )
    monkeypatch.setattr(
        cli, "_build_port_runner", lambda: _RecordingRunner(_MEASURE_DOC, _MATCHING_SAMPLES)
    )

    result = _runner.invoke(app, ["compare", "measure_record"])

    assert result.exit_code == 1
    assert "session_log.oracle_reread_exit" in result.stdout


def test_compare_when_missing_session_fixture_fails_both_sides_does_exit_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = json.dumps({"error": "no open session"})
    monkeypatch.setattr(cli, "_build_oracle_runner", lambda: _StubRunner(payload, exit_code=2))
    monkeypatch.setattr(cli, "_build_port_runner", lambda: _StubRunner(payload, exit_code=2))

    result = _runner.invoke(app, ["compare", MISSING_SESSION_FIXTURE])

    assert result.exit_code == 0, result.stdout
    assert MISSING_SESSION_FIXTURE in result.stdout


def test_self_diff_when_run_over_record_fixture_does_stay_green(monkeypatch: pytest.MonkeyPatch):
    fake = _RecordingRunner(_MEASURE_DOC, _MATCHING_SAMPLES)
    monkeypatch.setattr(cli, "_build_oracle_runner", lambda: fake)

    result = _runner.invoke(app, ["self-diff", "measure_record"])

    assert result.exit_code == 0, result.stdout
    assert "measure_record" in result.stdout
