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
    Confirm,
    PairedSamples,
    read_records,
    session_jsonl_path,
)
from gymrat.session.paths import progress_path
from gymrat.session.progress_file import ProgressSnapshot, write_progress
from tests.cli._budget import install_budget
from tests.cli._loop_cmds import plain_lines, runner, write_config
from tests.loop.iterate._fixtures import (
    baseline_rounds,
    improved_rounds,
    install_collect_samples,
    stub_samples,
)
from tests.loop.iterate._fixtures import session_record as iterate_session_header
from tests.session.records._fixtures import (
    SESSION_ID,
    committed_keep,
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
    assert len(read_records(session_jsonl_path(repo))) == 2


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
    """A stand-in for ``iterate_session`` that raises on call."""

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
    """Termination signals must clear the progress sidecar even when finally is skipped."""
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
    assert doc["primary"]["deltaPct"] == pytest.approx(-7.2)
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

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["capMinutes"] == 30
    assert isinstance(doc["budget"]["remainingSeconds"], int)


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

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["capMinutes"] == 30
    assert isinstance(doc["budget"]["remainingSeconds"], int)


def test_iterate_command_when_error_and_budget_active_does_not_include_budget(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    write_session_log(repo, iterate_session_header(repo))
    _install_renderer_factory(monkeypatch)
    raiser = _IterateSessionRaiser(RuntimeError("bench exploded"))
    _install_iterate_session(monkeypatch, raiser)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code != 0
    assert "left of" not in result.stdout
    assert "left of" not in result.stderr
