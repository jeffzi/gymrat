"""Tests for the CLI shared infrastructure: error rendering, render modes, output.

These cover the CLI shared surface — the stream helpers, the render-mode
resolution, the error formatter and exit path — plus the import-latency guard.
Flag parser tests live in ``test_options.py``; lock and trace tests live in
``test_lock.py``.
"""

import asyncio
import errno
import io
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import override

import pytest
import typer

from gymrat.adapters.types import AdapterError
from gymrat.cli import shared
from gymrat.cli.app import app
from gymrat.cli.lock import GATE_EXIT_CODE, TOOL_FAILURE_EXIT_CODE
from gymrat.cli.progress import ProgressReporter
from gymrat.cli.shared import (
    BUGS_URL,
    CompareFlags,
    MeasureFlags,
    SharedFlags,
    begin_run,
    exit_with_error,
    format_cli_error,
    is_broken_pipe,
    is_tty,
    point_stream_at_devnull,
    resolve_render_mode,
    run_with_signal_abort,
    set_color_override,
    set_debug_mode,
    write_and_flush,
    write_budget_report,
    write_stdout,
)
from gymrat.errors import GymratError
from gymrat.report.json_doc import BudgetSummary
from gymrat.report.types import RegressedFailOn
from tests._process_helpers import run_with_closed_reader, run_with_failing_stdout
from tests._rich import unwrap_panel
from tests._streams import FakeStream as _FakeStream
from tests._streams import RaisingStream
from tests.cli._help import help_output
from tests.cli._session import CLOSED_STDOUT_ERRORS, runner, stub_measure


class _StubReporter:
    """A minimal double satisfying the ProgressReporter report/stop contract."""

    def report(self, event: object) -> None: ...
    def stop(self) -> None: ...


def _capturing_progress_reporter(
    captured: dict[str, object],
) -> Callable[..., _StubReporter]:
    """A ``ProgressReporter`` stub that records the constructor's mode and counts."""

    def fake_init(
        mode: str,
        console: object,
        target_count: int,
        sample_count: int | None = None,
        *,
        clock: object = None,
        command: str | None = None,
        target_labels: list[str] | None = None,
    ) -> _StubReporter:
        captured["mode"] = mode
        captured["target_count"] = target_count
        captured["sample_count"] = sample_count
        return _StubReporter()

    return fake_init


def _force_color(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set FORCE_COLOR so color resolution is forced on."""
    monkeypatch.setenv("FORCE_COLOR", "1")


def _force_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set NO_COLOR so color resolution is forced off."""
    monkeypatch.setenv("NO_COLOR", "1")


@pytest.fixture(autouse=True)
def _reset_color_override():
    """Reset the module-level color override between tests so xdist workers don't leak state."""
    set_color_override(None)
    yield
    set_color_override(None)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


def test_constants_when_checked_does_match_the_shipped_contract():
    assert GATE_EXIT_CODE == 1
    assert TOOL_FAILURE_EXIT_CODE == 2
    assert BUGS_URL == "https://github.com/jeffzi/gymrat/issues"


# ---------------------------------------------------------------------------
# stream helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        pytest.param(_FakeStream(tty=True), True, id="tty"),
        pytest.param(_FakeStream(tty=False), False, id="non-tty"),
        pytest.param(object(), False, id="no-isatty"),
    ],
)
def test_is_tty_when_called_does_reflect_the_streams_isatty(stream: object, expected: bool):
    assert is_tty(stream) is expected


def test_write_and_flush_when_called_does_write_then_flush():
    class Recorder:
        def __init__(self):
            self.data = ""
            self.flushed = False

        def write(self, data: str) -> None:
            self.data += data

        def flush(self) -> None:
            self.flushed = True

    recorder = Recorder()

    write_and_flush(recorder, "hello")

    assert recorder.data == "hello"
    assert recorder.flushed is True


@pytest.mark.parametrize(
    ("error", "platform", "expected"),
    [
        pytest.param(BrokenPipeError(), "linux", True, id="posix-broken-pipe"),
        pytest.param(OSError(errno.EINVAL, "Invalid argument"), "win32", True, id="windows-einval"),
        pytest.param(OSError(errno.EINVAL, "Invalid argument"), "linux", False, id="posix-einval"),
        pytest.param(OSError(errno.EACCES, "Access denied"), "win32", False, id="windows-eacces"),
        pytest.param(ValueError("banana"), "win32", False, id="not-an-os-error"),
    ],
)
def test_is_broken_pipe_when_called_does_recognize_each_platforms_closed_pipe_error(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, platform: str, expected: bool
):
    monkeypatch.setattr("sys.platform", platform)

    assert is_broken_pipe(error) is expected


class _BadDescriptorStream(io.StringIO):
    """A stream whose descriptor is invalid, so redirecting it fails."""

    @override
    def fileno(self) -> int:
        return -1


def _open_descriptors() -> set[str]:
    """List the descriptors this process holds open."""
    return {entry.name for entry in Path("/dev/fd").iterdir()}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptor semantics")
def test_point_stream_at_devnull_when_redirected_does_point_descriptor_at_devnull(tmp_path: Path):
    with (tmp_path / "out.txt").open("w") as stream:
        point_stream_at_devnull(stream)

        assert os.path.samestat(os.fstat(stream.fileno()), Path(os.devnull).stat())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptor semantics")
def test_point_stream_at_devnull_when_redirected_does_close_devnull(tmp_path: Path):
    with (tmp_path / "out.txt").open("w") as stream:
        before = _open_descriptors()

        point_stream_at_devnull(stream)

        assert _open_descriptors() == before


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptor semantics")
def test_point_stream_at_devnull_when_redirect_fails_does_close_devnull_and_raise():
    before = _open_descriptors()

    with pytest.raises(OSError, match="Bad file descriptor"):
        point_stream_at_devnull(_BadDescriptorStream())

    assert _open_descriptors() == before


#: Lines each flood probe prints: 16384 lines of 64 bytes overflow any pipe buffer.
_FLOOD_LINES = 1 << 14

#: Characters per flood line; with the trailing newline, each line is 64 bytes.
_FLOOD_LINE_CHARS = 63

_FLOOD_THROUGH_RUN_CLI = f"""
import typer
from gymrat.cli.shared import run_cli, write_stdout

app = typer.Typer()

@app.command()
def flood() -> None:
    async def run() -> None:
        for _ in range({_FLOOD_LINES}):
            write_stdout("x" * {_FLOOD_LINE_CHARS} + "\\n")

    run_cli(run)

app()
"""

_FLOOD_OUTSIDE_RUN_CLI = f"""
import typer
from gymrat.cli.shared import write_stdout

app = typer.Typer()

@app.command()
def flood() -> None:
    for _ in range({_FLOOD_LINES}):
        write_stdout("x" * {_FLOOD_LINE_CHARS} + "\\n")

app()
"""


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(_FLOOD_THROUGH_RUN_CLI, id="run_cli"),
        pytest.param(_FLOOD_OUTSIDE_RUN_CLI, id="bare-command"),
    ],
)
def test_write_stdout_when_reader_closed_does_exit_zero_without_stderr(probe: str):
    result = run_with_closed_reader(
        [sys.executable, "-c", probe], stream="stdout", text=True, timeout=60
    )

    assert (result.returncode, result.stderr) == (0, "")


@pytest.mark.parametrize(("error", "platform"), CLOSED_STDOUT_ERRORS)
def test_write_stdout_when_pipe_closed_does_return_without_raising(
    monkeypatch: pytest.MonkeyPatch, error: OSError, platform: str
):
    monkeypatch.setattr("sys.platform", platform)
    monkeypatch.setattr("sys.stdout", RaisingStream(error))

    write_stdout("first line\n")


@pytest.mark.usefixtures("repo")
def test_run_cli_when_body_raises_broken_pipe_does_exit_two_with_error(
    monkeypatch: pytest.MonkeyPatch,
):
    stub_measure(monkeypatch)

    async def _explode(*_args: object, **_kwargs: object) -> None:
        raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    monkeypatch.setattr("gymrat.measure.measure", _explode)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh"])

    assert result.exit_code == TOOL_FAILURE_EXIT_CODE
    assert "Broken pipe" in unwrap_panel(result.stderr)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no file-size limit")
@pytest.mark.parametrize("fmt", ["text", "json"])
def test_gymrat_when_stdout_write_fails_otherwise_does_exit_two_without_shutdown_traceback(
    tmp_path: Path, fmt: str
):
    result = run_with_failing_stdout(
        [sys.executable, "-m", "gymrat", "doctor", "--format", fmt], cwd=tmp_path, timeout=60
    )

    assert result.returncode == TOOL_FAILURE_EXIT_CODE
    assert os.strerror(errno.EFBIG) in unwrap_panel(result.stderr)
    assert "Exception ignored" not in result.stderr


# ---------------------------------------------------------------------------
# color control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(("compare",), id="compare"),
        pytest.param(("measure",), id="measure"),
        pytest.param(("doctor",), id="doctor"),
        pytest.param(("init",), id="init"),
        pytest.param(("iterate",), id="iterate"),
        pytest.param(("status",), id="status"),
        pytest.param(("supervise",), id="supervise"),
    ],
)
def test_color_flag_when_help_does_show_color_no_color_pair(command: tuple[str, ...]):
    out = help_output(*command)

    tokens = out.split()
    assert "--color" in tokens
    assert "--no-color" in tokens


def test_format_cli_error_when_stderr_color_override_false_does_strip_all_sgr(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=True))
    monkeypatch.setenv("TERM", "xterm-256color")

    set_color_override(False)

    result = format_cli_error(ValueError("boom"))

    assert "\x1b[" not in result


# ---------------------------------------------------------------------------
# resolve_render_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tty", "expected"),
    [
        pytest.param(False, "plain", id="non-tty-plain"),
        pytest.param(True, "live", id="tty-live"),
    ],
)
def test_resolve_render_mode_when_called_does_map_tty_to_strategy(
    tty: bool,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=tty))

    assert resolve_render_mode() == expected


def test_resolve_render_mode_when_no_color_set_does_still_use_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=True))
    monkeypatch.setenv("NO_COLOR", "1")

    assert resolve_render_mode() == "live"


# ---------------------------------------------------------------------------
# begin_run
# ---------------------------------------------------------------------------


def test_begin_run_when_tty_does_create_progress_reporter_with_live_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=True))

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "gymrat.cli.shared.ProgressReporter",
        _capturing_progress_reporter(captured),
    )

    flags = SharedFlags(bench="b", samples=7)

    begin_run(flags, target_count=3)

    assert captured["mode"] == "live"
    assert captured["target_count"] == 3
    assert captured["sample_count"] == 7


def test_begin_run_when_non_tty_does_create_progress_reporter_with_plain_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=False))

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "gymrat.cli.shared.ProgressReporter",
        _capturing_progress_reporter(captured),
    )

    flags = SharedFlags(bench="b", samples=5)

    begin_run(flags, target_count=1)

    assert captured["mode"] == "plain"


def test_begin_run_does_return_progress_reporter(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=False))

    result = begin_run(SharedFlags(bench="b", samples=1), target_count=1)

    assert isinstance(result, ProgressReporter)


# ---------------------------------------------------------------------------
# run_with_signal_abort
# ---------------------------------------------------------------------------


async def test_run_with_signal_abort_when_cleanup_invoked_kills_groups_before_setting_abort(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_cleanup: list[Callable[[], None]] = []
    captured_abort: list[asyncio.Event] = []

    def _install(cleanup: Callable[[], None]) -> Callable[[], None]:
        captured_cleanup.append(cleanup)
        return lambda: None

    monkeypatch.setattr(shared, "install_termination_cleanup", _install)

    observed: dict[str, bool] = {}

    def _kill() -> None:
        observed["kill_ran"] = True
        observed["abort_set_at_kill"] = captured_abort[0].is_set()

    monkeypatch.setattr(shared, "kill_live_process_groups", _kill)

    async def execute(abort: asyncio.Event) -> str:
        captured_abort.append(abort)
        captured_cleanup[0]()
        observed["abort_after_cleanup"] = abort.is_set()
        return "done"

    result = await run_with_signal_abort(execute)

    assert result == "done"
    assert observed["kill_ran"] is True
    assert observed["abort_set_at_kill"] is False
    assert observed["abort_after_cleanup"] is True


# ---------------------------------------------------------------------------
# flag dataclasses
# ---------------------------------------------------------------------------


def test_shared_flags_when_built_does_carry_config_set_plus_defaults():
    flags = SharedFlags(bench="my-bench", samples=5)

    assert flags.bench == "my-bench"
    assert flags.samples == 5
    assert flags.color is None
    assert flags.format == "text"


def test_compare_flags_when_built_does_add_verbose_and_fail_on():
    flags = CompareFlags(verbose=True, fail_on=(RegressedFailOn(),))

    assert flags.verbose is True
    assert flags.fail_on == (RegressedFailOn(),)
    assert flags.color is None


def test_measure_flags_when_built_does_subclass_shared_flags():
    flags = MeasureFlags(adapter="mitata")

    assert flags.adapter == "mitata"
    assert isinstance(flags, SharedFlags)


# ---------------------------------------------------------------------------
# import-latency guard
# ---------------------------------------------------------------------------


def test_importing_cli_modules_does_not_pull_the_heavy_stack_or_command_bodies():
    probe = """
import sys
import gymrat.cli.shared
import gymrat.cli.options
import gymrat.cli.progress
import gymrat.cli.gating
heavy = sorted(
    name
    for name in sys.modules
    if name in {'scipy', 'numpy'} or name.startswith(('scipy.', 'numpy.'))
)
bodies = [name for name in ('gymrat.compare', 'gymrat.measure') if name in sys.modules]
assert not heavy, f'cli import pulled heavy modules: {heavy}'
assert not bodies, f'cli import pulled command bodies: {bodies}'
"""

    result = subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# format_cli_error
# ---------------------------------------------------------------------------


def test_format_cli_error_when_colored_paints_the_error_label_red(
    monkeypatch: pytest.MonkeyPatch,
):
    _force_color(monkeypatch)

    output = format_cli_error(ValueError("boom"))

    assert "\x1b[31m" in output
    assert "Error" in output


def test_format_cli_error_when_no_color_renders_plain_label_and_message(
    monkeypatch: pytest.MonkeyPatch,
):
    _force_no_color(monkeypatch)

    output = format_cli_error(ValueError("boom"))

    assert "Error: boom" in output
    assert "\x1b[" not in output


def test_format_cli_error_when_adapter_error_keeps_its_class_name_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    _force_no_color(monkeypatch)

    output = format_cli_error(AdapterError("parse failed"))

    assert "AdapterError: parse failed" in output


def test_format_cli_error_includes_stack_only_under_debug():
    message = "boom"
    try:
        raise ValueError(message)
    except ValueError as error:
        with_stack = format_cli_error(error, debug=True)
        without_stack = format_cli_error(error, debug=False)

    assert "Traceback" in with_stack
    assert "Traceback" not in without_stack


def test_format_cli_error_when_gymrat_error_carries_hint_appends_unlabeled_line_without_footer(
    monkeypatch: pytest.MonkeyPatch,
):
    _force_no_color(monkeypatch)

    output = format_cli_error(GymratError("boom", hint="run gymrat doctor"))

    assert output.splitlines()[-1] == "run gymrat doctor"
    assert "Hint" not in output
    assert BUGS_URL not in output


def test_format_cli_error_when_hint_colored_does_dim_the_line_and_paint_inline_code_blue(
    monkeypatch: pytest.MonkeyPatch,
):
    _force_color(monkeypatch)

    output = format_cli_error(GymratError("boom", hint="run `gymrat doctor` first"))

    hint_line = output.splitlines()[-1]
    assert hint_line.startswith("\x1b[2mrun ")  # cspell:disable-line
    assert "\x1b[2;34mgymrat doctor" in hint_line  # cspell:disable-line


def test_format_cli_error_when_not_gymrat_error_appends_bug_footer():
    output = format_cli_error(ValueError("boom"))

    assert BUGS_URL in output


def test_format_cli_error_when_value_is_not_an_exception_still_renders(
    monkeypatch: pytest.MonkeyPatch,
):
    _force_no_color(monkeypatch)

    output = format_cli_error("plain failure")

    assert "Error: plain failure" in output
    assert BUGS_URL in output


# ---------------------------------------------------------------------------
# exit_with_error
# ---------------------------------------------------------------------------


def test_exit_with_error_writes_to_stderr_and_exits_on_the_given_code(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = io.StringIO()
    monkeypatch.setattr("sys.stderr", captured)

    with pytest.raises(typer.Exit) as exc:
        exit_with_error(ValueError("boom"), code=TOOL_FAILURE_EXIT_CODE)

    assert exc.value.exit_code == TOOL_FAILURE_EXIT_CODE
    assert "Error: boom" in captured.getvalue()


def test_exit_with_error_when_stderr_write_fails_keeps_the_exit_code(
    monkeypatch: pytest.MonkeyPatch,
):
    class BrokenStderr:
        def write(self, _data: str) -> int:
            raise OSError

        def flush(self) -> None:
            raise OSError

        def isatty(self) -> bool:
            return False

    monkeypatch.setattr("sys.stderr", BrokenStderr())

    with pytest.raises(typer.Exit) as exc:
        exit_with_error(ValueError("boom"), code=TOOL_FAILURE_EXIT_CODE)

    assert exc.value.exit_code == TOOL_FAILURE_EXIT_CODE


def test_exit_with_error_honors_debug_mode_for_the_stack(monkeypatch: pytest.MonkeyPatch):
    captured = io.StringIO()
    monkeypatch.setattr("sys.stderr", captured)
    set_debug_mode(True)
    message = "boom"

    try:
        raise ValueError(message)
    except ValueError as error:
        with pytest.raises(typer.Exit):
            exit_with_error(error, code=TOOL_FAILURE_EXIT_CODE)

    set_debug_mode(False)
    assert "Traceback" in captured.getvalue()


# ---------------------------------------------------------------------------
# write_budget_report
# ---------------------------------------------------------------------------


def _budget_active(root: str) -> tuple[str, BudgetSummary]:
    """Stub returning an active budget snapshot."""
    return "\n⏱ 29m left of 30m", BudgetSummary(cap_minutes=30, remaining_seconds=1740)


def _budget_inactive(root: str) -> tuple[str, None]:
    """Stub returning no budget."""
    return "", None


def test_write_budget_report_when_json_and_budget_active_does_write_json_with_budget(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(shared, "budget_snapshot", _budget_active)

    def render_json(s: BudgetSummary | None) -> str:
        import json

        doc: dict[str, object] = {"metric": "ops/s"}
        if s is not None:
            doc["budget"] = {
                "cap_minutes": s.cap_minutes,
                "remaining_seconds": s.remaining_seconds,
            }
        return json.dumps(doc)

    write_budget_report(
        "/fake/root",
        use_json=True,
        render_json=render_json,
        text_report="ignored text",
    )

    import json

    out = json.loads(capsys.readouterr().out)
    assert out["budget"] == {"cap_minutes": 30, "remaining_seconds": 1740}


def test_write_budget_report_when_json_and_no_budget_does_write_json_without_budget(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(shared, "budget_snapshot", _budget_inactive)

    def render_json(s: BudgetSummary | None) -> str:
        import json

        doc: dict[str, object] = {"metric": "ops/s"}
        if s is not None:
            doc["budget"] = {
                "cap_minutes": s.cap_minutes,
                "remaining_seconds": s.remaining_seconds,
            }
        return json.dumps(doc)

    write_budget_report(
        "/fake/root",
        use_json=True,
        render_json=render_json,
        text_report="ignored text",
    )

    import json

    out = json.loads(capsys.readouterr().out)
    assert "budget" not in out


def _noop_json(_s: BudgetSummary | None) -> str:
    """A no-op JSON renderer for text-mode tests."""
    return ""


def test_write_budget_report_when_text_and_budget_active_does_write_report_with_trailer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(shared, "budget_snapshot", _budget_active)

    write_budget_report(
        "/fake/root",
        use_json=False,
        render_json=_noop_json,
        text_report="benchmark results here",
    )

    out = capsys.readouterr().out
    assert out == "benchmark results here\n⏱ 29m left of 30m\n"


def test_write_budget_report_when_text_and_no_budget_does_write_report_without_trailer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(shared, "budget_snapshot", _budget_inactive)

    write_budget_report(
        "/fake/root",
        use_json=False,
        render_json=_noop_json,
        text_report="benchmark results here",
    )

    out = capsys.readouterr().out
    assert out == "benchmark results here\n"
