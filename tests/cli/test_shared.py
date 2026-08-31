"""Tests for the CLI shared infrastructure: parsing, locking, render modes.

These cover the CLI shared surface — the positional grammar, the numeric flag
coercers, the render-mode resolution, and the repository lock wrapper — plus the
import-latency guard.
"""

import asyncio
import io
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import override

import pytest
import typer

from gymrat.adapters.types import AdapterError
from gymrat.cli import shared
from gymrat.cli.shared import (
    BUGS_URL,
    GATE_EXIT_CODE,
    TOOL_FAILURE_EXIT_CODE,
    CompareFlags,
    MeasureFlags,
    SharedFlags,
    begin_run,
    color_override_of,
    exit_with_error,
    format_cli_error,
    is_tty,
    parse_fail_on,
    parse_max_minutes,
    parse_positional,
    parse_positive_integer_up_to,
    parse_positive_number,
    resolve_render_mode,
    run_with_signal_abort,
    set_debug_mode,
    set_stderr_color_override,
    with_repo_lock,
    write_and_flush,
)
from gymrat.errors import GymratError
from gymrat.git import NotAGitRepositoryError
from gymrat.report.types import GeomeanFailOn, RegressedFailOn
from gymrat.sampling import TargetSpec
from gymrat.session import append_record, read_records, session_jsonl_path
from gymrat.session.paths import lockfile_path, repo_root
from tests.session.records._fixtures import (
    iteration_record,
    session_record,
    tear_final_line,
    write_session_log,
)


class _FakeStream(io.StringIO):
    """A stderr stand-in whose TTY status the test controls."""

    def __init__(self, *, tty: bool):
        super().__init__()
        self._tty = tty

    @override
    def isatty(self) -> bool:
        return self._tty


class _StubReporter:
    """A minimal double satisfying the ProgressReporter report/stop contract."""

    def report(self, event: object) -> None: ...
    def stop(self) -> None: ...


def _capturing_create_progress_reporter(
    captured: dict[str, object],
) -> Callable[..., _StubReporter]:
    """A ``create_progress_reporter`` stub that records the call's mode and counts."""

    def fake_create(
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

    return fake_create


def _path_exists(path: Path) -> bool:
    """Filesystem probe kept out of the async body so it is not flagged as blocking I/O."""
    return path.exists()


def _read_bytes(path: Path) -> bytes:
    """Filesystem read kept out of the async body so it is not flagged as blocking I/O."""
    return path.read_bytes()


def _clear_color_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete FORCE_COLOR/NO_COLOR so a stray value doesn't leak into color resolution."""
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)


@pytest.fixture(autouse=True)
def _reset_stderr_color_override():
    """Reset the module-level color override between tests so xdist workers don't leak state."""
    set_stderr_color_override(None)
    yield
    set_stderr_color_override(None)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


def test_exit_code_and_url_constants_match_the_shipped_contract():
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
def test_is_tty_reflects_the_streams_isatty(stream: object, expected: bool):
    assert is_tty(stream) is expected


def test_write_and_flush_writes_then_flushes():
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


def test_run_cli_when_broken_pipe_does_exit_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from gymrat.cli.shared import run_cli

    async def boom() -> None:
        raise BrokenPipeError

    with pytest.raises(typer.Exit) as exc:
        run_cli(boom)

    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert BUGS_URL not in captured.err


# ---------------------------------------------------------------------------
# color control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        pytest.param(True, None, id="color-on-defers"),
        pytest.param(False, False, id="color-off-vetoes"),
    ],
)
def test_color_override_of_maps_flag_to_renderer_override(color: bool, expected: object):
    assert color_override_of(color) is expected


def test_format_cli_error_when_stderr_color_override_false_does_strip_all_sgr(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=True))
    monkeypatch.setenv("TERM", "xterm-256color")
    _clear_color_env(monkeypatch)

    set_stderr_color_override(False)
    result = format_cli_error(ValueError("boom"))
    set_stderr_color_override(None)

    assert "\x1b[" not in result


# ---------------------------------------------------------------------------
# parse_positional / collect_positional
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("positional", "expected"),
    [
        pytest.param("main=HEAD", TargetSpec(label="main", target="HEAD"), id="label-and-target"),
        pytest.param("a=b=c", TargetSpec(label="a", target="b=c"), id="first-equals-splits"),
        pytest.param("HEAD", TargetSpec(label=None, target="HEAD"), id="no-equals"),
    ],
)
def test_parse_positional_splits_on_the_first_equals(positional: str, expected: TargetSpec):
    assert parse_positional(positional) == expected


def test_parse_positional_when_label_empty_raises_dedicated_message():
    with pytest.raises(typer.BadParameter) as exc:
        parse_positional("=HEAD")

    assert (
        exc.value.message
        == 'the label before "=" is empty; write the positional as "label=<ref|dir>" or drop the "=".'
    )


@pytest.mark.parametrize(
    "positional",
    [pytest.param("main=", id="trailing-equals"), pytest.param("", id="empty-string")],
)
def test_parse_positional_when_target_empty_raises_dedicated_message(positional: str):
    with pytest.raises(typer.BadParameter) as exc:
        parse_positional(positional)

    assert exc.value.message == 'the target is empty; write the positional as "[label=]<ref|dir>".'


# ---------------------------------------------------------------------------
# parse_positive_integer_up_to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "5", "100"])
def test_parse_positive_integer_accepts_positive_integers(value: str):
    parse = parse_positive_integer_up_to(2_147_483)

    assert parse(value) == int(value)


@pytest.mark.parametrize(
    "value",
    ["0", "-1", "1.5", "abc", "", " 5", "5 "],
)
def test_parse_positive_integer_rejects_non_positive_integers(value: str):
    parse = parse_positive_integer_up_to(2_147_483)

    with pytest.raises(typer.BadParameter) as exc:
        parse(value)

    assert exc.value.message == "must be a positive integer."


def test_parse_positive_integer_rejects_values_over_the_maximum():
    parse = parse_positive_integer_up_to(10)

    with pytest.raises(typer.BadParameter) as exc:
        parse("11")

    assert exc.value.message == "must be a positive integer no greater than 10."


# ---------------------------------------------------------------------------
# parse_positive_number / parse_max_minutes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "expected"), [("1", 1.0), ("2.5", 2.5)])
def test_parse_positive_number_accepts_positive_decimals(value: str, expected: float):
    assert parse_positive_number(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "abc", "", "1."])
def test_parse_positive_number_rejects_non_positive_or_malformed(value: str):
    with pytest.raises(typer.BadParameter) as exc:
        parse_positive_number(value)

    assert exc.value.message == "must be a positive number."


def test_parse_positive_number_when_value_overflows_to_infinity_does_reject():
    huge = "1" + "0" * 310

    with pytest.raises(typer.BadParameter) as exc:
        parse_positive_number(huge)

    assert exc.value.message == "must be a positive number."


def test_parse_max_minutes_accepts_within_the_timer_ceiling():
    assert parse_max_minutes("10") == 10.0


def test_parse_max_minutes_rejects_over_the_timer_ceiling():
    with pytest.raises(typer.BadParameter) as exc:
        parse_max_minutes("35792")

    assert exc.value.message == "must be at most 35791 minutes."


def test_parse_max_minutes_rejects_non_positive_before_bounding():
    with pytest.raises(typer.BadParameter) as exc:
        parse_max_minutes("0")

    assert exc.value.message == "must be a positive number."


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
def test_resolve_render_mode_maps_tty_to_strategy(
    tty: bool,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=tty))

    assert resolve_render_mode() == expected


def test_resolve_render_mode_ignores_color_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=True))
    monkeypatch.setenv("NO_COLOR", "1")

    assert resolve_render_mode() == "live"


# ---------------------------------------------------------------------------
# begin_run
# ---------------------------------------------------------------------------


def test_begin_run_when_tty_does_create_progress_reporter_with_live_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_color_env(monkeypatch)
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=True))

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "gymrat.cli.shared.create_progress_reporter",
        _capturing_create_progress_reporter(captured),
    )

    flags = SharedFlags(bench="b", samples=7)

    begin_run(flags, target_count=3)

    assert captured["mode"] == "live"
    assert captured["target_count"] == 3
    assert captured["sample_count"] == 7


def test_begin_run_when_non_tty_does_create_progress_reporter_with_plain_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_color_env(monkeypatch)
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=False))

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "gymrat.cli.shared.create_progress_reporter",
        _capturing_create_progress_reporter(captured),
    )

    flags = SharedFlags(bench="b", samples=5)

    begin_run(flags, target_count=1)

    assert captured["mode"] == "plain"


def test_begin_run_does_return_progress_reporter(
    monkeypatch: pytest.MonkeyPatch,
):
    from gymrat.cli.progress import ProgressReporter

    _clear_color_env(monkeypatch)
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=False))

    result = begin_run(SharedFlags(bench="b", samples=1), target_count=1)

    assert isinstance(result, ProgressReporter)


# ---------------------------------------------------------------------------
# with_repo_lock
# ---------------------------------------------------------------------------


async def test_with_repo_lock_when_inside_repo_holds_lock_during_body_and_releases_before_return(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    repo = create_scratch_repo()
    monkeypatch.chdir(repo)
    lock_path = Path(lockfile_path(repo_root()))
    held: dict[str, bool] = {}

    async def body() -> str:
        held["during"] = _path_exists(lock_path)
        return "measured"

    result = await with_repo_lock("compare", body)

    assert result == "measured"
    assert held["during"] is True
    assert not _path_exists(lock_path)


async def test_with_repo_lock_when_outside_repo_runs_body_without_a_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    def not_a_repo(*_args: object, **_kwargs: object) -> str:
        message = "nope"
        raise NotAGitRepositoryError(message)

    acquired: list[object] = []

    def spy_acquire(*args: object, **_kwargs: object):
        acquired.append(args)
        return lambda: None

    monkeypatch.setattr("gymrat.cli.shared.repo_root", not_a_repo)
    monkeypatch.setattr("gymrat.cli.shared.acquire_lock", spy_acquire)

    async def body() -> str:
        return "ran"

    result = await with_repo_lock("compare", body)

    assert result == "ran"
    assert acquired == []


async def test_with_repo_lock_when_session_log_torn_does_repair_it_before_running_the_body(
    repo: str,
):
    header = session_record()
    write_session_log(repo, header)
    jsonl_path = Path(session_jsonl_path(repo_root()))
    intact_log = _read_bytes(jsonl_path)
    tear_final_line(jsonl_path)
    iteration = iteration_record()
    seen: dict[str, bytes] = {}

    async def body() -> str:
        seen["log"] = _read_bytes(jsonl_path)
        append_record(str(jsonl_path), iteration)
        return "ran"

    result = await with_repo_lock("compare", body)

    assert result == "ran"
    assert seen["log"] == intact_log
    assert read_records(str(jsonl_path)) == [header, iteration]


async def test_with_repo_lock_when_git_fails_otherwise_exits_two_without_running_body(
    monkeypatch: pytest.MonkeyPatch,
):
    def broken_git(*_args: object, **_kwargs: object) -> str:
        message = "detected dubious ownership"
        raise GymratError(message)

    monkeypatch.setattr("gymrat.cli.shared.repo_root", broken_git)
    called: list[bool] = []

    async def body() -> str:
        called.append(True)
        return "should-not-run"

    with pytest.raises(typer.Exit) as exc:
        await with_repo_lock("compare", body)

    assert exc.value.exit_code == TOOL_FAILURE_EXIT_CODE
    assert called == []


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


def test_shared_flags_carry_the_config_set_plus_color_and_format_defaults():
    flags = SharedFlags(bench="my-bench", samples=5)

    assert flags.bench == "my-bench"
    assert flags.samples == 5
    assert flags.color is True
    assert flags.format == "text"


def test_compare_flags_add_verbose_and_fail_on_to_the_shared_set():
    flags = CompareFlags(verbose=True, fail_on=(RegressedFailOn(),))

    assert flags.verbose is True
    assert flags.fail_on == (RegressedFailOn(),)
    assert flags.color is True


def test_measure_flags_are_a_shared_flags_subclass():
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
# parse_fail_on
# ---------------------------------------------------------------------------


def test_parse_fail_on_accepts_regressed():
    assert parse_fail_on("regressed") == RegressedFailOn()


@pytest.mark.parametrize(
    ("value", "expected_pct"),
    [
        pytest.param("geomean:2", 2.0, id="integer"),
        pytest.param("geomean:-1.5", -1.5, id="negative-decimal"),
    ],
)
def test_parse_fail_on_accepts_geomean_percentage(value: str, expected_pct: float):
    condition = parse_fail_on(value)

    assert condition == GeomeanFailOn(pct=expected_pct)


@pytest.mark.parametrize(
    "value",
    ["geomean:", "geomean:0x10", "unknown", "", " geomean:2"],
)
def test_parse_fail_on_rejects_everything_else(value: str):
    with pytest.raises(typer.BadParameter) as exc:
        parse_fail_on(value)

    assert (
        exc.value.message
        == 'allowed values are "regressed" or "geomean:<number>" (e.g. geomean:2).'
    )


# ---------------------------------------------------------------------------
# format_cli_error
# ---------------------------------------------------------------------------


def test_format_cli_error_when_colored_paints_the_error_label_red(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    output = format_cli_error(ValueError("boom"))

    assert "\x1b[31m" in output
    assert "Error" in output


def test_format_cli_error_when_no_color_renders_plain_label_and_message(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

    output = format_cli_error(ValueError("boom"))

    assert "Error: boom" in output
    assert "\x1b[" not in output


def test_format_cli_error_when_adapter_error_keeps_its_class_name_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)

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
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    output = format_cli_error(GymratError("boom", hint="run gymrat doctor"))

    assert output.splitlines()[-1] == "run gymrat doctor"
    assert "Hint" not in output
    assert BUGS_URL not in output


def test_format_cli_error_when_hint_colored_does_dim_the_line_and_paint_inline_code_blue(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

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
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)

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
