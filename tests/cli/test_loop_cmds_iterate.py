"""Iterate command tests: basic execution, progress renderer wiring, format flags, and budget.

Budget tests verify that a live budget appends a time-left line to text output
and inserts a ``budget`` key in JSON output, including on stop-condition exits.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from gymrat import signals
from gymrat.cli.app import app
from gymrat.loop.iterate import IterateOptions, IterateResult, LoopStopError
from gymrat.session import (
    CommandRecord,
    Confirm,
    PairedSamples,
    read_records,
    session_jsonl_path,
)
from gymrat.session.paths import progress_path
from gymrat.session.progress_file import ProgressSnapshot, write_progress
from tests.cli._budget import (
    SUPERVISED_HINT,
    install_budget,
    install_tight_budget,
    mark_tool_origin,
    set_origin,
)
from tests.cli._session import last_command_record, plain_lines, records_of, runner, write_config
from tests.loop.iterate._fixtures import (
    CollectSamplesRecorder,
    baseline_rounds,
    improved_rounds,
    install_collect_samples,
    stub_samples,
)
from tests.loop.iterate._fixtures import session_record as iterate_session_header
from tests.session.records._fixtures import (
    SESSION_ID,
    committed_keep,
    finalize_record,
    iteration_record,
    write_session_log,
)

# ---------------------------------------------------------------------------
# the iterate command
# ---------------------------------------------------------------------------


def test_iterate_command_when_run_does_measure_the_repo_and_report_on_stdout(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    mock = install_collect_samples(monkeypatch)
    stub_samples(mock, repo, improved_rounds(), baseline_rounds())

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 0
    lines = plain_lines(result.stdout)
    assert lines[0] == "iteration 1 · experiment vs baseline · 10 paired samples"
    assert lines[-1] == "gymrat keep"

    non_command = [
        r for r in read_records(session_jsonl_path(repo)) if not isinstance(r, CommandRecord)
    ]
    assert len(non_command) == 2


def test_iterate_command_when_stop_condition_met_does_exit_one_without_measuring(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(
        repo, iterate_session_header(repo), (iteration_record(seq=1), committed_keep(1))
    )
    mock = install_collect_samples(monkeypatch)
    write_config(repo, stop={"max_iterations": 1})

    result = runner.invoke(app, ["iterate"])

    assert result.exit_code == 1
    assert "max iterations" in result.stderr
    assert mock.call_count == 0


def test_iterate_command_when_no_session_does_exit_two_with_a_start_hint(repo: str):
    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr


# ---------------------------------------------------------------------------
# the iterate command — command trace annotations
# ---------------------------------------------------------------------------


def test_iterate_command_when_run_does_record_command_trace_with_seq_and_args(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    mock = install_collect_samples(monkeypatch)
    stub_samples(mock, repo, improved_rounds(), baseline_rounds())

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--samples", "10"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.name == "iterate"
    assert cmd.args["bench"] == "npm run bench"
    assert cmd.args["samples"] == 10
    assert cmd.seq == 1
    assert cmd.exit_code == 0
    assert cmd.reason is None


def test_iterate_command_when_unsettled_does_record_command_trace_with_exit_two_and_seq(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo), (iteration_record(seq=1),))
    install_collect_samples(monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 2
    cmd = last_command_record(repo)
    assert cmd.name == "iterate"
    assert cmd.exit_code == 2
    assert cmd.reason == "unsettled"
    assert cmd.seq == 1


def test_iterate_command_when_finalized_does_record_command_trace_with_exit_two(
    repo: str,
):
    write_session_log(
        repo,
        iterate_session_header(repo),
        (iteration_record(seq=1), committed_keep(1), finalize_record()),
    )

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 2
    cmd = last_command_record(repo)
    assert cmd.name == "iterate"
    assert cmd.exit_code == 2
    assert cmd.reason == "finalized"


# ---------------------------------------------------------------------------
# the iterate command — progress renderer wiring
# ---------------------------------------------------------------------------


@dataclass
class _RendererRecord:
    """Captures the arguments ``IterateRenderer`` was called with."""

    mode: object
    console: object
    seq: int
    session_id: str
    sample_count: int
    metric_count: int
    primary_metric: str
    verbose: bool
    clock: object
    checks_cmd: str | None = None
    has_before_hook: bool = False
    has_after_hook: bool = False


class _FakeRenderer:
    """A stand-in for ``IterateRenderer`` recording ``report`` and ``stop`` calls."""

    def __init__(self) -> None:
        self.report_calls: list[object] = []
        self.stop_called = False

    def report(self, event: object) -> None:
        self.report_calls.append(event)

    def stop(self) -> None:
        self.stop_called = True


class _RendererFactory:
    """A stand-in for ``IterateRenderer`` recording what it was handed."""

    def __init__(self) -> None:
        self.renderer = _FakeRenderer()
        self.calls: list[_RendererRecord] = []

    def __call__(  # noqa: PLR0917 -- mirrors IterateRenderer signature
        self,
        mode: object,
        console: object,
        seq: int,
        session_id: str,
        sample_count: int,
        metric_count: int,
        primary_metric: str,
        *,
        verbose: bool,
        clock: object = None,
        checks_cmd: str | None = None,
        has_before_hook: bool = False,
        has_after_hook: bool = False,
    ) -> _FakeRenderer:
        self.calls.append(
            _RendererRecord(
                mode=mode,
                console=console,
                seq=seq,
                session_id=session_id,
                sample_count=sample_count,
                metric_count=metric_count,
                primary_metric=primary_metric,
                verbose=verbose,
                clock=clock,
                checks_cmd=checks_cmd,
                has_before_hook=has_before_hook,
                has_after_hook=has_after_hook,
            )
        )
        return self.renderer


class _IterateSessionRecorder:
    """A stand-in for ``iterate_session`` that captures its ``IterateOptions``."""

    def __init__(self, result: IterateResult) -> None:
        self._result = result
        self.captured_options: IterateOptions | None = None

    async def __call__(
        self,
        root: str,
        config: object,
        options: IterateOptions | None = None,
        *,
        color: bool | None = None,
    ) -> IterateResult:
        self.captured_options = options
        return self._result


class _IterateSessionRaiser:
    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.captured_options: IterateOptions | None = None

    async def __call__(
        self,
        root: str,
        config: object,
        options: IterateOptions | None = None,
        *,
        color: bool | None = None,
    ) -> IterateResult:
        self.captured_options = options
        raise self._error


def _install_renderer_factory(monkeypatch: pytest.MonkeyPatch) -> _RendererFactory:
    """Replace ``IterateRenderer`` in the loop_cmds module with a recorder."""
    factory = _RendererFactory()
    monkeypatch.setattr("gymrat.cli.loop_cmds.IterateRenderer", factory)
    return factory


def _install_iterate_session(
    monkeypatch: pytest.MonkeyPatch, recorder: _IterateSessionRecorder | _IterateSessionRaiser
) -> None:
    """Replace ``iterate_session`` in the loop_cmds module with a recorder or raiser."""
    monkeypatch.setattr("gymrat.cli.loop_cmds.iterate_session", recorder)


def _make_iterate_result() -> IterateResult:
    """A minimal ``IterateResult`` with a dummy report and record."""
    return IterateResult(
        record=iteration_record(seq=1),
        report="iteration 1 · experiment vs baseline · 10 paired samples\ngymrat keep",
    )


def _wire_successful_iterate(
    repo: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[_RendererFactory, _IterateSessionRecorder]:
    """Wire ``iterate`` with a renderer factory and a recording session stub."""
    write_session_log(repo, iterate_session_header(repo))
    factory = _install_renderer_factory(monkeypatch)
    recorder = _IterateSessionRecorder(_make_iterate_result())
    _install_iterate_session(monkeypatch, recorder)
    return factory, recorder


def test_iterate_command_when_run_does_wire_on_progress_into_iterate_options(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    factory, recorder = _wire_successful_iterate(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert recorder.captured_options is not None
    assert recorder.captured_options.on_progress is not None
    assert factory.renderer.stop_called


def test_iterate_command_when_error_does_still_call_renderer_stop(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    factory = _install_renderer_factory(monkeypatch)
    raiser = _IterateSessionRaiser(RuntimeError("bench exploded"))
    _install_iterate_session(monkeypatch, raiser)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code != 0
    assert factory.renderer.stop_called


@pytest.mark.parametrize("verbose_flag", [True, False])
def test_iterate_command_when_verbose_flag_does_forward_verbose_to_renderer(
    repo: str, monkeypatch: pytest.MonkeyPatch, verbose_flag: bool
):
    factory, _recorder = _wire_successful_iterate(repo, monkeypatch)
    args = ["iterate", "--bench", "npm run bench"]
    if verbose_flag:
        args.append("--verbose")

    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert len(factory.calls) == 1
    assert factory.calls[0].verbose is verbose_flag


def test_iterate_command_when_run_does_pass_session_metadata_to_renderer(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    factory, _recorder = _wire_successful_iterate(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert len(factory.calls) == 1
    call = factory.calls[0]
    # seq = last_seq + 1 = 0 + 1 = 1 for a fresh session
    assert call.seq == 1
    assert call.session_id == SESSION_ID
    # The default resolved_config has samples=10
    assert call.sample_count == 10
    # The default resolved_config has primary="geomean"
    assert call.primary_metric == "geomean"


def test_iterate_command_when_run_does_register_progress_cleanup_for_termination(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    _install_renderer_factory(monkeypatch)

    captured_cleanups: list[Callable[[], None]] = []
    real_install = signals.install_termination_cleanup

    def capturing_install(cleanup: Callable[[], None]) -> Callable[[], None]:
        captured_cleanups.append(cleanup)
        return real_install(cleanup)

    monkeypatch.setattr("gymrat.cli.loop_cmds.install_termination_cleanup", capturing_install)

    progress_cleared_mid_run = False

    async def check_cleanup(
        root: str,
        config: object,
        options: IterateOptions | None = None,
        *,
        color: bool | None = None,
    ) -> IterateResult:
        nonlocal progress_cleared_mid_run
        write_progress(
            root,
            ProgressSnapshot(
                passes_completed=1,
                passes_total=2,
                last_pass_duration_ms=100.0,
            ),
        )
        progress = Path(progress_path(root))
        assert progress.exists()  # noqa: ASYNC240 -- sync check in async test
        # Termination signals must clear the sidecar even when finally is skipped.
        for cleanup in captured_cleanups:
            cleanup()
        progress_cleared_mid_run = not progress.exists()  # noqa: ASYNC240 -- sync check in async test
        return _make_iterate_result()

    monkeypatch.setattr("gymrat.cli.loop_cmds.iterate_session", check_cleanup)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert progress_cleared_mid_run


# ---------------------------------------------------------------------------
# iterate --format json
# ---------------------------------------------------------------------------


def test_iterate_command_when_format_json_does_emit_structured_json_on_stdout(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _factory, _recorder = _wire_successful_iterate(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["seq"] == 1
    assert doc["outcome"] == "improved"
    assert doc["primary"]["kind"] == "geomean"
    assert doc["primary"]["delta_pct"] == pytest.approx(-7.2)
    assert "metrics" in doc
    assert doc["confirm"] is None


def test_iterate_command_when_format_json_does_include_confirm_when_rerun_happened(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    confirm = Confirm(
        ran=True,
        filtered=("total_ms",),
        samples=PairedSamples(
            experiment=({"total_ms": 14100},),
            baseline=({"total_ms": 15200},),
        ),
        absent=None,
    )
    record = iteration_record(seq=1, confirm=confirm)
    iterate_result = IterateResult(record=record, report="confirmed iteration report")
    write_session_log(repo, iterate_session_header(repo))
    _install_renderer_factory(monkeypatch)
    recorder = _IterateSessionRecorder(iterate_result)
    _install_iterate_session(monkeypatch, recorder)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["confirm"] is not None
    assert doc["confirm"]["ran"] is True
    assert doc["confirm"]["filtered"] == ["total_ms"]


def test_iterate_command_when_format_json_and_stop_condition_does_emit_stop_document(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    _install_renderer_factory(monkeypatch)
    raiser = _IterateSessionRaiser(LoopStopError("max iterations (3) reached"))
    _install_iterate_session(monkeypatch, raiser)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc["stopped"] is True
    assert "max iterations" in doc["reason"]


@pytest.mark.parametrize(
    "extra_args",
    [
        pytest.param(["--format", "text"], id="explicit-text"),
        pytest.param([], id="default-format"),
    ],
)
def test_iterate_command_when_format_text_does_produce_plain_report(
    repo: str, monkeypatch: pytest.MonkeyPatch, extra_args: list[str]
):
    _factory, _recorder = _wire_successful_iterate(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", *extra_args])

    assert result.exit_code == 0
    lines = plain_lines(result.stdout)
    assert lines[0] == "iteration 1 · experiment vs baseline · 10 paired samples"
    assert lines[-1] == "gymrat keep"


# ---------------------------------------------------------------------------
# the iterate command — budget line and JSON key
# ---------------------------------------------------------------------------


def test_iterate_command_when_budget_active_does_end_text_with_time_left_line(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _factory, _recorder = _wire_successful_iterate(repo, monkeypatch)
    install_budget(repo, monkeypatch)
    mark_tool_origin(monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 0
    lines = plain_lines(result.stdout)
    assert re.search(r"left of 30m", lines[-1])


def test_iterate_command_when_no_budget_does_omit_time_left_line(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _factory, _recorder = _wire_successful_iterate(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 0
    assert "left of" not in result.stdout


def test_iterate_command_when_format_json_and_budget_active_does_include_budget_object(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _factory, _recorder = _wire_successful_iterate(repo, monkeypatch)
    install_budget(repo, monkeypatch)
    mark_tool_origin(monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


def test_iterate_command_when_format_json_and_no_budget_does_omit_budget_key(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _factory, _recorder = _wire_successful_iterate(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" not in doc


def test_iterate_command_when_stop_condition_and_budget_active_does_include_time_left_in_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    _install_renderer_factory(monkeypatch)
    raiser = _IterateSessionRaiser(LoopStopError("max iterations (3) reached"))
    _install_iterate_session(monkeypatch, raiser)
    install_budget(repo, monkeypatch)
    mark_tool_origin(monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 1
    assert re.search(r"left of 30m", result.stderr)


def test_iterate_command_when_stop_and_format_json_and_budget_active_does_include_budget_key(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    _install_renderer_factory(monkeypatch)
    raiser = _IterateSessionRaiser(LoopStopError("max iterations (3) reached"))
    _install_iterate_session(monkeypatch, raiser)
    install_budget(repo, monkeypatch)
    mark_tool_origin(monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


def test_iterate_command_when_error_and_budget_active_does_not_include_budget(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    _install_renderer_factory(monkeypatch)
    raiser = _IterateSessionRaiser(RuntimeError("bench exploded"))
    _install_iterate_session(monkeypatch, raiser)
    install_budget(repo, monkeypatch)
    mark_tool_origin(monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code != 0
    assert "left of" not in result.stdout
    assert "left of" not in result.stderr


# ---------------------------------------------------------------------------
# the iterate command — refused while a supervised run is live
# ---------------------------------------------------------------------------


SUPERVISED_MESSAGE = "a supervised run is live; use the iterate tool"


@pytest.fixture
def samples_mock(repo: str, monkeypatch: pytest.MonkeyPatch) -> CollectSamplesRecorder:
    """The stubbed bench, answering every sampling call with an improved run."""
    mock = install_collect_samples(monkeypatch)
    stub_samples(mock, repo, improved_rounds(), baseline_rounds())
    return mock


@pytest.fixture
def supervised_repo(
    repo: str, monkeypatch: pytest.MonkeyPatch, samples_mock: CollectSamplesRecorder
) -> str:
    """An open session with a before hook under a live budget, run from the shell."""
    header = iterate_session_header(repo)
    write_session_log(repo, header)
    # The before hook runs in the experiment worktree, so it must exist for the hook to fire.
    Path(header.worktrees.experiment).mkdir()
    marker = Path(repo, "hook-ran")
    write_config(repo, hooks={"before": f"touch '{marker}'"})
    install_budget(repo, monkeypatch)
    set_origin(monkeypatch, None)
    return repo


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_iterate_command_when_supervised_run_live_does_refuse_without_running_and_record_it(
    supervised_repo: str, output_format: str, samples_mock: CollectSamplesRecorder
):
    before = records_of(supervised_repo, commands=False)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", output_format])

    assert result.exit_code == 2
    assert result.stdout == ""
    stderr = " ".join(plain_lines(result.stderr))
    assert SUPERVISED_MESSAGE in stderr
    assert SUPERVISED_HINT in stderr
    assert samples_mock.call_count == 0
    assert not Path(supervised_repo, "hook-ran").exists()
    assert records_of(supervised_repo, commands=False) == before
    assert not Path(progress_path(supervised_repo)).exists()
    commands = records_of(supervised_repo, commands=True)
    assert len(commands) == 1
    cmd = last_command_record(supervised_repo)
    assert cmd.name == "iterate"
    assert cmd.exit_code == 2
    assert cmd.reason == "supervised-use-tool"
    assert cmd.origin == "cli"
    assert cmd.seq is None


def test_iterate_command_when_tool_hosted_under_live_budget_does_run_the_before_hook(
    supervised_repo: str, monkeypatch: pytest.MonkeyPatch
):
    mark_tool_origin(monkeypatch)

    runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert Path(supervised_repo, "hook-ran").exists()


def _unsettled(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An open session whose last iteration was never kept or discarded."""
    write_session_log(repo, iterate_session_header(repo), (iteration_record(seq=1),))
    write_config(repo)
    install_budget(repo, monkeypatch)


def _stop_condition_met(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A settled session that already reached its configured iteration cap."""
    write_session_log(
        repo, iterate_session_header(repo), (iteration_record(seq=1), committed_keep(1))
    )
    write_config(repo, stop={"max_iterations": 1})
    install_budget(repo, monkeypatch)


def _budget_exceeded(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A settled session whose last iteration outlasts the 5 minutes the budget has left."""
    write_session_log(
        repo,
        iterate_session_header(repo),
        (iteration_record(seq=1, duration_ms=840_000), committed_keep(1)),
    )
    write_config(repo)
    install_tight_budget(repo, monkeypatch)


#: Every readiness setup, keyed by its parametrize id, and the ``(exit_code, reason)`` it expects.
READINESS_SETUPS = {
    "unsettled": _unsettled,
    "stop-condition": _stop_condition_met,
    "budget-exceeded": _budget_exceeded,
}
READINESS_EXPECTATIONS = {
    "unsettled": (2, "unsettled"),
    "stop-condition": (1, "stop-condition"),
    "budget-exceeded": (1, "budget-exceeded"),
}


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        pytest.param(setup, READINESS_EXPECTATIONS[setup_id], id=setup_id)
        for setup_id, setup in READINESS_SETUPS.items()
    ],
)
def test_iterate_command_when_tool_hosted_in_unready_state_does_refuse_with_its_own_reason(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable[[str, pytest.MonkeyPatch], None],
    expected: tuple[int, str],
    samples_mock: CollectSamplesRecorder,
):
    setup(repo, monkeypatch)
    mark_tool_origin(monkeypatch)

    result = runner.invoke(app, ["iterate"])

    assert (result.exit_code, last_command_record(repo).reason) == expected
    assert samples_mock.call_count == 0


@pytest.mark.parametrize(
    "setup",
    [pytest.param(setup, id=setup_id) for setup_id, setup in READINESS_SETUPS.items()],
)
def test_iterate_command_when_supervised_run_live_does_refuse_before_every_readiness_check(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable[[str, pytest.MonkeyPatch], None],
    samples_mock: CollectSamplesRecorder,
):
    setup(repo, monkeypatch)
    set_origin(monkeypatch, None)

    result = runner.invoke(app, ["iterate"])

    assert result.exit_code == 2
    assert SUPERVISED_MESSAGE in " ".join(plain_lines(result.stderr))
    assert last_command_record(repo).reason == "supervised-use-tool"
    assert samples_mock.call_count == 0
