"""Command-level tests for the loop subcommands: start, iterate, discard, finalize, status.

Each command is driven through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
The seams mocked here mirror the engine suites' boundaries: sampling for
``iterate`` (``gymrat.loop.iterate.collect_samples``), and for ``discard``'s
prompt the ``is_tty`` and ``confirm_action`` helpers as the ``loop_cmds`` module
imports them. Config resolution is exercised for real where a test lays down a
``gymrat.toml`` and stubbed at the ``loop_cmds`` seam where a test needs to pin
what a command reads (the runbook row) or observe where it looked (the
subdirectory case).
"""

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import tomli_w
from typer.testing import CliRunner

from gymrat.cli import loop_cmds
from gymrat.cli.app import app
from gymrat.loop.finalize import finalize_session
from gymrat.loop.iterate import IterateOptions, IterateResult, LoopStopError
from gymrat.loop.settle import DiscardResult, KeepResult
from gymrat.loop.start import start_session
from gymrat.session import (
    Confirm,
    DiscardRecord,
    FinalizeRecord,
    KeepChecks,
    KeepRecord,
    PairedSamples,
    SessionRecord,
    append_record,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from tests.loop._iterate import (
    baseline_rounds,
    improved_rounds,
    install_collect_samples,
    resolved_config,
    stub_samples,
)
from tests.loop._iterate import session_record as iterate_session_header
from tests.loop._settle import git, head_of, last_record_of
from tests.session._records import (
    AT,
    COMMIT,
    SESSION_ID,
    committed_keep,
    iteration_record,
    session_record,
    write_session_log,
)

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

#: The runbook path a session's config points an agent at, when it has one.
_RUNBOOK_PATH = ".claude/skills/ecstatic-bench/SKILL.md"


def _plain_lines(text: str) -> list[str]:
    """The non-blank lines of ``text``, stripped of color and surrounding space."""
    return [_ANSI_RE.sub("", line).strip() for line in text.split("\n") if line.strip()]


def _always_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its interactive path."""
    return True


def _never_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its non-interactive path."""
    return False


def _write_config(root: str, **extra: object) -> None:
    """Write the implicit ``gymrat.toml`` at the repository root."""
    payload: dict[str, object] = {"bench": "npm run bench", **extra}
    (Path(root) / "gymrat.toml").write_text(tomli_w.dumps(payload), encoding="utf-8")


class _ConfirmRecorder:
    """A stand-in for ``confirm_action`` recording its calls and answering ``answer``."""

    def __init__(self, *, answer: bool) -> None:
        self.answer = answer
        self.calls: list[tuple[str, object]] = []

    def __call__(self, message: str, stream: object) -> bool:
        self.calls.append((message, stream))
        return self.answer


class _ResolverRecorder:
    """A stand-in for a config resolver recording ``(flags, base_dir)`` per call."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[object, str | Path | None]] = []

    def __call__(self, flags: object, base_dir: str | Path | None = None) -> object:
        self.calls.append((flags, base_dir))
        return self.result


# ---------------------------------------------------------------------------
# the start command
# ---------------------------------------------------------------------------


def _stub_resolve_config(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> object:
    """Pin what ``start`` reads by replacing its ``resolve_config`` with a fixed config."""
    config = resolved_config(**overrides)

    def fake(*_a: object, **_k: object) -> object:
        return config

    monkeypatch.setattr("gymrat.cli.loop_cmds.resolve_config", fake)
    return config


def _close_session_with_one_keep(root: str) -> str:
    """Open a session with one kept commit, finalize it, and return its closed id."""
    start_session(root, "main", resolved_config())
    worktree = experiment_worktree_dir(root)
    (Path(worktree) / "README.md").write_text("# cache the regex\n", encoding="utf-8")
    git(["add", "README.md"], worktree)
    git(["commit", "-m", "cache the regex"], worktree)
    commit = head_of(worktree)
    append_record(session_jsonl_path(root), iteration_record(seq=1))
    append_record(session_jsonl_path(root), committed_keep(1, commit=commit))
    finalize_session(root)
    header = read_records(session_jsonl_path(root))[0]
    assert isinstance(header, SessionRecord)
    return header.session_id


def test_start_command_when_run_does_create_a_session_and_report_its_branch(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "main"])

    assert result.exit_code == 0
    header = read_records(session_jsonl_path(repo))[0]
    assert isinstance(header, SessionRecord)
    assert header.branch in result.stdout


def test_start_command_when_reopening_after_finalize_does_name_the_archived_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    closed_id = _close_session_with_one_keep(repo)
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "main"])

    assert result.exit_code == 0
    assert re.search(r"archived", result.stdout, re.IGNORECASE)
    assert closed_id in result.stdout


@pytest.mark.parametrize("resumed", [False, True])
def test_start_command_when_runbook_configured_does_include_a_runbook_row(
    repo: str, monkeypatch: pytest.MonkeyPatch, resumed: bool
):
    if resumed:
        start_session(repo, "main", resolved_config())
    _stub_resolve_config(monkeypatch, runbook=_RUNBOOK_PATH)

    result = runner.invoke(app, ["start", "main"])

    assert result.exit_code == 0
    assert f"runbook: {_RUNBOOK_PATH} — read it before your first edit" in result.stdout


def test_start_command_when_runbook_absent_does_omit_the_runbook_row(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "main"])

    assert result.exit_code == 0
    assert "runbook" not in result.stdout


@pytest.mark.parametrize("resumed", [False, True])
def test_start_command_when_run_does_include_edit_here_line_with_sync_hint(
    repo: str, monkeypatch: pytest.MonkeyPatch, resumed: bool
):
    if resumed:
        start_session(repo, "main", resolved_config())
    _stub_resolve_config(monkeypatch)

    result = runner.invoke(app, ["start", "main"])

    assert result.exit_code == 0
    exp_dir = experiment_worktree_dir(repo)
    assert f"edit in {exp_dir}" in result.stdout
    assert "gymrat sync" in result.stdout


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
    lines = _plain_lines(result.stdout)
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
    _write_config(repo, stop={"max_iterations": 1})

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
    """Captures the arguments ``create_iterate_renderer`` was called with."""

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
    """A stand-in for ``create_iterate_renderer`` recording what it was handed."""

    def __init__(self) -> None:
        self.renderer = _FakeRenderer()
        self.calls: list[_RendererRecord] = []

    def __call__(  # noqa: PLR0917 -- mirrors create_iterate_renderer signature
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
    """Replace ``create_iterate_renderer`` in the loop_cmds module with a recorder."""
    factory = _RendererFactory()
    monkeypatch.setattr("gymrat.cli.loop_cmds.create_iterate_renderer", factory)
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


def test_iterate_command_when_format_json_does_include_stable_key_names(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _factory, _recorder = _wire_successful_iterate(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert {"seq", "outcome", "primary", "metrics", "confirm"} <= doc.keys()
    assert {"kind", "deltaPct"} <= doc["primary"].keys()


def test_iterate_command_when_format_text_does_produce_plain_report(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _factory, _recorder = _wire_successful_iterate(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench", "--format", "text"])

    assert result.exit_code == 0
    lines = _plain_lines(result.stdout)
    assert lines[0] == "iteration 1 · experiment vs baseline · 10 paired samples"
    assert lines[-1] == "gymrat keep"


def test_iterate_command_when_format_absent_does_produce_plain_report(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _factory, _recorder = _wire_successful_iterate(repo, monkeypatch)

    result = runner.invoke(app, ["iterate", "--bench", "npm run bench"])

    assert result.exit_code == 0
    lines = _plain_lines(result.stdout)
    assert lines[-1] == "gymrat keep"


# ---------------------------------------------------------------------------
# keep --format json
# ---------------------------------------------------------------------------


class _KeepSessionRecorder:
    """A stand-in for ``keep_session`` that returns a fixed ``KeepResult``."""

    def __init__(self, result: KeepResult) -> None:
        self._result = result

    async def __call__(self, *_args: object, **_kwargs: object) -> KeepResult:
        return self._result


def _make_committed_keep_result() -> KeepResult:
    """A ``KeepResult`` for a committed keep with checks passing."""
    record = KeepRecord(
        type="keep",
        seq=1,
        at=AT,
        status="committed",
        checks=KeepChecks(configured=True, passed=True, stdout_bytes=80, stderr_bytes=0),
        commit=COMMIT,
        message="cache the regex",
    )
    return KeepResult(record=record, report="committed keep report")


def _make_blocked_keep_result() -> KeepResult:
    """A ``KeepResult`` for a blocked keep where checks failed."""
    record = KeepRecord(
        type="keep",
        seq=1,
        at=AT,
        status="blocked",
        checks=KeepChecks(configured=True, passed=False, stdout_bytes=120, stderr_bytes=45),
        reason="checks-failed",
    )
    return KeepResult(record=record, report="blocked keep report")


def _wire_keep(repo: str, monkeypatch: pytest.MonkeyPatch, keep_result: KeepResult) -> None:
    """Wire ``keep`` with a config resolver and a recording keep_session stub."""
    start_session(repo, "main", resolved_config())
    append_record(session_jsonl_path(repo), iteration_record(seq=1))
    monkeypatch.setattr("gymrat.cli.loop_cmds.keep_session", _KeepSessionRecorder(keep_result))


def test_keep_command_when_format_json_and_committed_does_emit_structured_json(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_keep(repo, monkeypatch, _make_committed_keep_result())

    result = runner.invoke(app, ["keep", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["status"] == "committed"
    assert doc["commit"] == COMMIT
    assert doc["message"] == "cache the regex"
    assert doc["reason"] is None
    assert doc["checks"]["configured"] is True
    assert doc["checks"]["passed"] is True
    assert doc["checks"]["stdoutBytes"] == 80
    assert doc["checks"]["stderrBytes"] == 0


def test_keep_command_when_format_json_and_blocked_does_emit_blocked_json_with_reason(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_keep(repo, monkeypatch, _make_blocked_keep_result())

    result = runner.invoke(app, ["keep", "--format", "json"])

    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc["status"] == "blocked"
    assert doc["reason"] == "checks-failed"
    assert doc["commit"] is None
    assert doc["message"] is None


def test_keep_command_when_format_json_does_include_stable_key_names(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_keep(repo, monkeypatch, _make_committed_keep_result())

    result = runner.invoke(app, ["keep", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert {"status", "reason", "checks", "commit", "message"} <= doc.keys()
    assert {"configured", "passed", "stdoutBytes", "stderrBytes"} <= doc["checks"].keys()


def test_keep_command_when_format_text_does_produce_plain_report(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_keep(repo, monkeypatch, _make_committed_keep_result())

    result = runner.invoke(app, ["keep", "--format", "text"])

    assert result.exit_code == 0
    assert "committed keep report" in result.stdout


# ---------------------------------------------------------------------------
# discard --format json
# ---------------------------------------------------------------------------


class _DiscardSessionRecorder:
    """A stand-in for ``discard_session`` that returns a fixed ``DiscardResult``."""

    def __init__(self, result: DiscardResult) -> None:
        self._result = result

    def __call__(self, *_args: object, **_kwargs: object) -> DiscardResult:
        return self._result


def _make_discard_result() -> DiscardResult:
    """A ``DiscardResult`` for a discarded iteration 1."""
    return DiscardResult(
        record=DiscardRecord(type="discard", seq=1, at=AT),
        report="discarded iteration 1",
    )


def _wire_discard(monkeypatch: pytest.MonkeyPatch, discard_result: DiscardResult) -> None:
    """Wire ``discard`` with a recording discard_session stub and skip its TTY prompt."""
    monkeypatch.setattr(
        "gymrat.cli.loop_cmds.discard_session", _DiscardSessionRecorder(discard_result)
    )
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", _never_tty)


def test_discard_command_when_format_json_does_emit_structured_json(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_discard(monkeypatch, _make_discard_result())

    result = runner.invoke(app, ["discard", "--force", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["seq"] == 1
    assert doc["at"] == AT


def test_discard_command_when_format_json_does_include_stable_key_names(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    _wire_discard(monkeypatch, _make_discard_result())

    result = runner.invoke(app, ["discard", "--force", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert {"seq", "at"} <= doc.keys()


def test_discard_command_when_format_text_does_produce_plain_report(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", _never_tty)

    result = runner.invoke(app, ["discard", "--force", "--format", "text"])

    assert result.exit_code == 0
    assert "discard" in result.stdout.lower()


# ---------------------------------------------------------------------------
# the status command
# ---------------------------------------------------------------------------


def test_status_command_when_run_does_render_the_session_on_stdout(repo: str):
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    _write_config(repo)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    text = _ANSI_RE.sub("", result.stdout)
    assert f"session {SESSION_ID}" in text
    assert "1 kept" in text


@pytest.mark.parametrize("has_config", [True, False])
def test_status_command_when_no_session_does_exit_two_with_a_start_hint(
    repo: str, has_config: bool
):
    if has_config:
        _write_config(repo)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr


@pytest.mark.parametrize(
    ("args", "expect_ansi"),
    [
        pytest.param([], True, id="color-on-by-default"),
        pytest.param(["--no-color"], False, id="no-color-beats-force-color"),
    ],
)
def test_status_command_color(
    repo: str, monkeypatch: pytest.MonkeyPatch, args: list[str], expect_ansi: bool
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    _write_config(repo)

    result = runner.invoke(app, ["status", *args])

    assert result.exit_code == 0
    assert bool(_ANSI_RE.search(result.stdout)) is expect_ansi


# ---------------------------------------------------------------------------
# status --format json
# ---------------------------------------------------------------------------


@pytest.fixture
def status_repo(repo: str) -> str:
    """A repository with a configured session and one kept iteration."""
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    _write_config(repo)
    return repo


def test_status_command_when_format_json_does_emit_structured_json_on_stdout(status_repo: str):
    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["sessionId"] == SESSION_ID
    assert doc["branch"] == f"gymrat/{SESSION_ID}"
    assert doc["baseline"]["ref"] == "main"
    assert doc["baseline"]["sha"] == "a" * 40
    assert doc["iterationCount"] == 1
    assert doc["keepCount"] == 1
    assert doc["discardCount"] == 0
    assert doc["unsettled"] is False
    assert doc["finalized"] is False


def test_status_command_when_format_json_and_finalized_does_set_finalized_true(repo: str):
    write_session_log(
        repo,
        session_record(),
        (
            iteration_record(seq=1),
            committed_keep(1),
            FinalizeRecord(
                type="finalize",
                at=AT,
                branch=f"gymrat/{SESSION_ID}-final",
                commit=COMMIT,
                message="squash 1 kept iteration",
            ),
        ),
    )
    _write_config(repo)

    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["finalized"] is True


def test_status_command_when_format_json_does_include_stable_key_names(status_repo: str):
    result = runner.invoke(app, ["status", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert {
        "sessionId",
        "branch",
        "baseline",
        "iterationCount",
        "keepCount",
        "discardCount",
        "unsettled",
        "finalized",
    } <= doc.keys()
    assert {"ref", "sha"} <= doc["baseline"].keys()


def test_status_command_when_format_text_does_produce_identical_output(status_repo: str):
    result_text = runner.invoke(app, ["status", "--format", "text"])
    result_default = runner.invoke(app, ["status"])

    assert result_text.exit_code == 0
    assert result_default.exit_code == 0
    assert result_text.stdout == result_default.stdout


# ---------------------------------------------------------------------------
# the discard command
# ---------------------------------------------------------------------------


@pytest.fixture
def discard_repo(repo: str) -> str:
    """A repository with an open session and one unsettled iteration to discard."""
    start_session(repo, "main", resolved_config())
    append_record(session_jsonl_path(repo), iteration_record(seq=1))
    return repo


def test_discard_command_documents_force_in_its_help():
    result = runner.invoke(app, ["discard", "--help"])

    assert result.exit_code == 0
    assert "--force" in _ANSI_RE.sub("", result.output)


def test_discard_command_when_tty_and_confirmed_does_prompt_and_proceed(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", _always_tty)
    confirm = _ConfirmRecorder(answer=True)
    monkeypatch.setattr("gymrat.cli.loop_cmds.confirm_action", confirm)

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 0
    assert len(confirm.calls) == 1
    assert experiment_worktree_dir(discard_repo) in confirm.calls[0][0]
    assert re.search(r"discard", result.stdout, re.IGNORECASE)


def test_discard_command_when_tty_and_declined_does_cancel_with_exit_one(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", _always_tty)
    monkeypatch.setattr("gymrat.cli.loop_cmds.confirm_action", _ConfirmRecorder(answer=False))

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 1
    assert "discard cancelled" in result.stderr


@pytest.mark.parametrize("flag", ["--force", "-f"])
def test_discard_command_when_force_does_skip_the_prompt(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch, flag: str
):
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", _always_tty)
    confirm = _ConfirmRecorder(answer=True)
    monkeypatch.setattr("gymrat.cli.loop_cmds.confirm_action", confirm)

    result = runner.invoke(app, ["discard", flag])

    assert result.exit_code == 0
    assert confirm.calls == []
    assert re.search(r"discard", result.stdout, re.IGNORECASE)


def test_discard_command_when_stdin_not_tty_does_skip_the_prompt(
    discard_repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("gymrat.cli.loop_cmds.is_tty", _never_tty)
    confirm = _ConfirmRecorder(answer=True)
    monkeypatch.setattr("gymrat.cli.loop_cmds.confirm_action", confirm)

    result = runner.invoke(app, ["discard"])

    assert result.exit_code == 0
    assert confirm.calls == []
    assert re.search(r"discard", result.stdout, re.IGNORECASE)


# ---------------------------------------------------------------------------
# the finalize command
# ---------------------------------------------------------------------------


def _session_with_one_keep(root: str) -> str:
    """Open a session with one kept commit on it, and return its branch."""
    start_session(root, "main", resolved_config())
    header = read_records(session_jsonl_path(root))[0]
    assert isinstance(header, SessionRecord)
    branch = header.branch
    worktree = experiment_worktree_dir(root)
    (Path(worktree) / "step.txt").write_text("cache the regex\n", encoding="utf-8")
    git(["add", "-A"], worktree)
    git(["commit", "-m", "cache the regex"], worktree)
    commit = head_of(worktree)
    append_record(session_jsonl_path(root), iteration_record(seq=1))
    append_record(session_jsonl_path(root), committed_keep(1, commit=commit))
    return branch


def test_finalize_command_documents_its_flags_and_default_branch_in_help():
    result = runner.invoke(app, ["finalize", "--help"])

    assert result.exit_code == 0
    out = result.output
    assert "--message" in out
    assert "--branch" in out
    assert "-final" in out


@pytest.mark.parametrize(
    ("args", "named"),
    [
        pytest.param(
            ["--branch", "perf/regex-cache"], "perf/regex-cache", id="branch-the-caller-named"
        ),
        pytest.param([], None, id="session-branch-final-by-default"),
    ],
)
def test_finalize_command_records_and_reports_the_branch(
    repo: str, args: list[str], named: str | None
):
    branch = _session_with_one_keep(repo)
    final_branch = named if named is not None else f"{branch}-final"

    result = runner.invoke(app, ["finalize", *args])

    assert result.exit_code == 0
    record = last_record_of(repo)
    assert isinstance(record, FinalizeRecord)
    assert record.branch == final_branch
    assert final_branch in result.stdout


def test_finalize_command_commits_the_message_it_was_given(repo: str):
    _session_with_one_keep(repo)

    result = runner.invoke(app, ["finalize", "-m", "squash the tuning session"])

    assert result.exit_code == 0
    record = last_record_of(repo)
    assert isinstance(record, FinalizeRecord)
    assert record.message == "squash the tuning session"


def test_finalize_command_when_no_session_does_exit_two_with_a_start_hint(repo: str):
    result = runner.invoke(app, ["finalize"])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr


# ---------------------------------------------------------------------------
# the loop commands, run from a subdirectory of the repository
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "resolver_name"),
    [
        pytest.param(["start", "main"], "resolve_config", id="start"),
        pytest.param(["iterate"], "resolve_config", id="iterate"),
        pytest.param(["keep"], "resolve_benchless_config", id="keep"),
        pytest.param(["status"], "resolve_benchless_config", id="status"),
    ],
)
def test_loop_command_when_run_from_subdirectory_does_resolve_config_at_repo_root(
    create_scratch_repo: Callable[[], str],
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    resolver_name: str,
):
    root = create_scratch_repo()
    nested = Path(root) / "packages" / "core"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    recorder = _ResolverRecorder(resolved_config())
    monkeypatch.setattr(f"gymrat.cli.loop_cmds.{resolver_name}", recorder)

    # Whether the command then finds the session it needs is beside the point;
    # where it looked the configuration up is what is under test.
    runner.invoke(app, args)

    assert recorder.calls
    base_dir = recorder.calls[0][1]
    assert base_dir is not None
    assert Path(base_dir) == Path(root)


# ---------------------------------------------------------------------------
# the sync command
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_repo(repo: str) -> str:
    """A repository with an open session, ready for sync tests."""
    start_session(repo, "main", resolved_config())
    return repo


def test_sync_command_when_registered_does_appear_in_the_app_commands():
    result = runner.invoke(app, ["sync", "--help"])

    assert result.exit_code == 0
    assert "sync" in result.output.lower()


def test_sync_command_when_changes_exist_does_print_synced_file_count_and_names(
    sync_repo: str,
):
    root = Path(sync_repo)
    (root / "extra.py").write_text("# new\n", encoding="utf-8")
    (root / "README.md").write_text("# updated\n", encoding="utf-8")

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "2 files" in result.stdout
    assert "extra.py" in result.stdout
    assert "README.md" in result.stdout


def test_sync_command_when_nothing_to_sync_does_print_nothing_to_sync(
    sync_repo: str,
):
    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "nothing to sync" in result.stdout


def test_sync_command_when_run_does_take_the_repo_lock(
    sync_repo: str, monkeypatch: pytest.MonkeyPatch
):
    lock_names: list[str] = []
    original_with_repo_lock = loop_cmds.with_repo_lock

    async def recording_lock[T](command: str, body: Callable[[], Awaitable[T]]) -> T:
        lock_names.append(command)
        return await original_with_repo_lock(command, body)

    monkeypatch.setattr(loop_cmds, "with_repo_lock", recording_lock)

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "sync" in lock_names


def test_sync_command_when_no_session_does_exit_two_with_a_start_hint(
    repo: str,
):
    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 2
    assert "gymrat start" in result.stderr
