"""Tests for the ``gymrat export`` command wiring.

These drive the assembled app through :class:`typer.testing.CliRunner` with the
telemetry replay, tracing provider, and session store replaced. They cover
registration and help, the argument/option contract, error exits, and the
success path with its printed summary.
"""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner, Result

from gymrat.cli.app import app
from gymrat.session import session_jsonl_path
from gymrat.session.records import record_to_wire
from gymrat.supervisor.events import LaunchEvent, TurnEndEvent, to_json_line
from tests._ansi import SGR_RE
from tests.cli._help import help_output
from tests.session.records._fixtures import AT, SESSION_ID, command_record, session_record

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_tracing_provider() -> Iterator[None]:
    """Reset the telemetry provider singleton between tests."""
    yield
    from gymrat.telemetry.provider import _reset_for_tests

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _reset_for_tests()


def _output(result: Result) -> str:
    """Combine stdout and stderr the way CliRunner splits them across streams."""
    return result.stdout + result.stderr


_HEAD_SHA = "a" * 40
_ONE_SECOND_NS = 1_000_000_000
_T0 = AT
_T1 = _T0 + _ONE_SECOND_NS
_T2 = _T1 + _ONE_SECOND_NS
_T3 = _T2 + _ONE_SECOND_NS

_ENDPOINT = "http://localhost:4318"


def _write_jsonl(path: str, lines: Iterable[str]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        fh.writelines(line + "\n" for line in lines)


def _write_session_log(path: str, records: list[Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path, (json.dumps(record_to_wire(rec)) for rec in records))


def _write_supervisor_log(path: str, events: list[Any]) -> None:
    _write_jsonl(path, (to_json_line(ev) for ev in events))


def _launch_event(
    session_id: str = SESSION_ID,
    at: int = _T0,
) -> LaunchEvent:
    return LaunchEvent(
        at=at,
        schema_version=1,
        session_id=session_id,
        head_sha=_HEAD_SHA,
        dirty=False,
        max_minutes=60.0,
        runbook_path="/dev/null",
        kickoff_summary="test",
    )


def _turn_end(at: int = _T3) -> TurnEndEvent:
    return TurnEndEvent(
        at=at,
        text="done",
        cost_usd=0.42,
        origin="agent",
        budget_exhausted=False,
    )


def _populate_session_dir(
    root: str,
    *,
    session_id: str = SESSION_ID,
    extra_supervisor_session_id: str | None = None,
) -> str:
    """Write a session log and a matching supervisor log under ``root``.

    Returns the session log path. When ``extra_supervisor_session_id`` is given,
    an additional supervisor log with that session id is written.
    """
    session_log = session_jsonl_path(root)
    header = session_record(session_id=session_id, at=_T0)
    cmd = command_record(
        name="measure", at=_T2, duration_ms=500, exit_code=0, reason=None, seq=None
    )
    _write_session_log(session_log, [header, cmd])

    sup_dir = str(Path(session_log).parent)
    sup_log = str(Path(sup_dir) / "supervisor-001.jsonl")
    _write_supervisor_log(
        sup_log, [_launch_event(session_id=session_id, at=_T1), _turn_end(at=_T3)]
    )

    if extra_supervisor_session_id is not None:
        other_log = str(Path(sup_dir) / "supervisor-002.jsonl")
        _write_supervisor_log(
            other_log,
            [_launch_event(session_id=extra_supervisor_session_id, at=_T1), _turn_end(at=_T3)],
        )

    return session_log


# ---------------------------------------------------------------------------
# lazy imports
# ---------------------------------------------------------------------------


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONOPTIMIZE", None)  # cspell:disable-line
    return env


def test_export_when_app_imported_does_not_import_telemetry_provider_or_replay():
    probe = """
import sys
from gymrat.cli.app import app
leaked = sorted(
    name
    for name in sys.modules
    if name in {'gymrat.telemetry.provider', 'gymrat.telemetry.replay'}
)
if leaked:
    print(f'app import pulled telemetry modules: {leaked}', file=sys.stderr)
    sys.exit(1)
"""

    result = subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        env=_child_env(),
    )

    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# registration and help
# ---------------------------------------------------------------------------


def test_export_when_root_help_does_list_export():
    assert "export" in help_output()


def test_export_when_help_does_document_endpoint_and_debug():
    out = help_output("export")

    assert "--endpoint" in out
    assert "--debug" in out


def test_export_when_help_does_show_session_log_positional_metavar():
    out = help_output("export")

    assert "[SESSION_LOG]" in out


# ---------------------------------------------------------------------------
# missing SDK
# ---------------------------------------------------------------------------


def _invoke_export_with_missing_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Result:
    """Invoke ``export`` with ``configure_tracing`` raising a missing-SDK ``ImportError``."""
    session_log = _populate_session_dir(str(tmp_path))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)

    monkeypatch.setattr(
        "gymrat.telemetry.provider.configure_tracing",
        lambda *_a, **_kw: (_ for _ in ()).throw(  # pyrefly: ignore[implicit-any-lambda]
            ImportError("No module named 'opentelemetry.sdk'")
        ),
    )

    return runner.invoke(app, ["export", session_log])


def test_export_when_sdk_not_importable_does_exit_two_naming_otel_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    result = _invoke_export_with_missing_sdk(monkeypatch, tmp_path)

    assert result.exit_code == 2
    assert "gymrat[otel]" in _output(result)


def test_export_when_sdk_not_importable_does_quote_extras_specifier_in_install_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    result = _invoke_export_with_missing_sdk(monkeypatch, tmp_path)

    assert "'gymrat[otel]'" in _output(result)


# ---------------------------------------------------------------------------
# missing endpoint
# ---------------------------------------------------------------------------


def test_export_when_no_endpoint_flag_and_no_env_does_exit_two_naming_env_var(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session_log = _populate_session_dir(str(tmp_path))
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    result = runner.invoke(app, ["export", session_log])

    assert result.exit_code == 2
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in _output(result)


# ---------------------------------------------------------------------------
# missing or corrupt session log
# ---------------------------------------------------------------------------


def test_export_when_session_log_missing_does_exit_two_with_path_in_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)
    missing = str(tmp_path / ".gymrat" / "session.jsonl")

    result = runner.invoke(app, ["export", missing])

    assert result.exit_code == 2
    assert f"No session found in {missing}" in _output(result)


def test_export_when_session_log_blank_does_exit_two_reporting_no_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)
    session_log = session_jsonl_path(str(tmp_path))
    Path(session_log).parent.mkdir(parents=True, exist_ok=True)
    Path(session_log).write_text("\n", encoding="utf-8")

    result = runner.invoke(app, ["export", session_log])

    assert result.exit_code == 2
    assert f"No session found in {session_log}" in _output(result)


@pytest.mark.parametrize(
    ("first_line", "expected_fragments"),
    [
        pytest.param(
            b"not json at all\n",
            ("Invalid JSON at {path}:1", "Line 1 is not a JSON object."),
            id="malformed-json",
        ),
        pytest.param(
            json.dumps(record_to_wire(command_record(name="measure", at=_T0))).encode("utf-8")
            + b"\n",
            (
                "Expected session header at {path}:1, got a command record",
                "Line 1 is not a session header. The session log is corrupt; start a new session.",
            ),
            id="non-session-record",
        ),
        pytest.param(
            b"\xff\xff\n",
            ("Corrupt session log at {path}:1", "Line 1 contains invalid UTF-8 bytes."),
            id="undecodable-bytes",
        ),
    ],
)
def test_export_when_first_line_corrupt_does_exit_two_naming_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_line: bytes,
    expected_fragments: tuple[str, ...],
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)
    session_log = session_jsonl_path(str(tmp_path))
    Path(session_log).parent.mkdir(parents=True, exist_ok=True)
    Path(session_log).write_bytes(first_line)

    result = runner.invoke(app, ["export", session_log])

    output = _output(result)
    assert result.exit_code == 2
    expected = [fragment.format(path=session_log) for fragment in expected_fragments]
    assert [fragment for fragment in expected if fragment not in output] == []


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="requires Unix file permissions and non-root user",
)
def test_export_when_session_log_unreadable_does_exit_two_naming_path_and_os_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_log = _populate_session_dir(str(tmp_path))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)
    Path(session_log).chmod(0o000)

    result = runner.invoke(app, ["export", session_log])

    output = _output(result)
    assert result.exit_code == 2
    assert session_log in output
    assert "Permission denied" in output
    assert "No session found" not in output


def _always_true(*_args: object, **_kwargs: object) -> bool:
    return True


def _return_one(*_args: object, **_kwargs: object) -> int:
    return 1


def _stub_tracing(
    monkeypatch: pytest.MonkeyPatch,
    *,
    configure_tracing: Callable[..., bool] = _always_true,
    replay_session: Callable[..., int] = _return_one,
) -> None:
    """Stub the tracing provider and replay calls the export command wires together."""
    monkeypatch.setattr("gymrat.telemetry.provider.configure_tracing", configure_tracing)
    monkeypatch.setattr("gymrat.telemetry.replay.replay_session", replay_session)
    monkeypatch.setattr("gymrat.telemetry.provider.flush_tracing", lambda: None)


# ---------------------------------------------------------------------------
# success path
# ---------------------------------------------------------------------------


def test_export_when_valid_session_does_exit_zero_and_print_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session_log = _populate_session_dir(str(tmp_path))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)

    span_count = 3
    _stub_tracing(
        monkeypatch,
        replay_session=lambda *_a, **_kw: span_count,  # pyrefly: ignore[implicit-any-lambda]
    )

    result = runner.invoke(app, ["export", session_log])

    assert result.exit_code == 0
    output = _output(result)
    assert "3" in output
    assert SESSION_ID in output
    assert _ENDPOINT in output


def test_export_when_endpoint_flag_given_does_use_flag_over_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session_log = _populate_session_dir(str(tmp_path))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://wrong:4318")
    custom_endpoint = "http://custom:4318"

    configure_calls: list[dict[str, object]] = []

    def fake_configure(*_a: object, **_kw: object) -> bool:
        configure_calls.append(dict(_kw))
        return True

    _stub_tracing(monkeypatch, configure_tracing=fake_configure)

    result = runner.invoke(app, ["export", session_log, "--endpoint", custom_endpoint])

    assert result.exit_code == 0
    assert custom_endpoint in _output(result)


# ---------------------------------------------------------------------------
# default session log
# ---------------------------------------------------------------------------


def test_export_when_no_session_log_argument_does_use_repo_session_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _populate_session_dir(str(tmp_path))
    monkeypatch.setattr("gymrat.cli.export_cmd.repo_root", lambda: str(tmp_path))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)
    _stub_tracing(monkeypatch)

    result = runner.invoke(app, ["export"])

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# supervisor log filtering
# ---------------------------------------------------------------------------


def test_export_when_supervisor_log_session_differs_does_skip_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session_log = _populate_session_dir(
        str(tmp_path),
        extra_supervisor_session_id="other-session-id",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)

    replay_calls: list[tuple[object, ...]] = []

    def fake_replay(session_path: str, sup_logs: list[str]) -> int:
        replay_calls.append((session_path, sup_logs))
        return 1

    _stub_tracing(monkeypatch, replay_session=fake_replay)

    result = runner.invoke(app, ["export", session_log])

    assert result.exit_code == 0
    assert len(replay_calls) == 1
    _, sup_logs = replay_calls[0]
    assert len(sup_logs) == 1  # pyrefly: ignore[bad-argument-type]
    assert "supervisor-001" in sup_logs[0]  # pyrefly: ignore[bad-index]


def _first_line_writer(first_line: bytes) -> Callable[[Path], None]:
    """Build a writer that creates a supervisor log holding only ``first_line``."""

    def write(path: Path) -> None:
        path.write_bytes(first_line)

    return write


def _dangling_symlink(path: Path) -> None:
    """Create an entry the listing sees but whose target is gone by the time it is read."""
    path.symlink_to(path.with_name("vanished.jsonl"))


@pytest.mark.parametrize(
    "make_entry",
    [
        pytest.param(_first_line_writer(b"{not json\n"), id="a-first-line-that-is-not-json"),
        pytest.param(_first_line_writer(b"\xff\xff\n"), id="a-first-line-that-is-not-utf8"),
        pytest.param(_first_line_writer(b"[1, 2]\n"), id="a-first-line-that-is-not-an-object"),
        pytest.param(
            _dangling_symlink,
            id="a-log-that-vanished-after-listing",
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="creating a symlink needs extra rights on Windows"
            ),
        ),
    ],
)
def test_export_when_supervisor_log_first_line_not_a_json_object_does_skip_it_silently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_entry: Callable[[Path], None],
):
    session_log = _populate_session_dir(str(tmp_path))
    make_entry(Path(session_log).parent / "supervisor-002.jsonl")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)
    replay_calls: list[tuple[str, list[str]]] = []

    def fake_replay(session_path: str, sup_logs: list[str]) -> int:
        replay_calls.append((session_path, sup_logs))
        return 1

    _stub_tracing(monkeypatch, replay_session=fake_replay)

    result = runner.invoke(app, ["export", session_log])

    assert result.exit_code == 0, _output(result)
    assert [[Path(log).name for log in logs] for _, logs in replay_calls] == [
        ["supervisor-001.jsonl"]
    ]
    assert "warning: " not in result.stderr


def _deny_read(path: Path) -> None:
    """Write a supervisor log for the session, then remove every permission on it."""
    _write_supervisor_log(str(path), [_launch_event(), _turn_end()])
    path.chmod(0o000)


def _make_directory(path: Path) -> None:
    """Create a directory whose name matches the supervisor log pattern."""
    path.mkdir()


@pytest.mark.parametrize(
    ("make_unreadable", "reason"),
    [
        pytest.param(
            _deny_read,
            os.strerror(errno.EACCES),
            id="read-denied",
            marks=pytest.mark.skipif(
                not hasattr(os, "geteuid") or os.geteuid() == 0,
                reason="requires Unix file permissions and non-root user",
            ),
        ),
        pytest.param(
            _make_directory,
            os.strerror(errno.EISDIR),
            id="a-directory",
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="Windows reports a directory as access denied"
            ),
        ),
    ],
)
def test_export_when_supervisor_log_unreadable_does_warn_and_export_the_rest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_unreadable: Callable[[Path], None],
    reason: str,
):
    session_log = _populate_session_dir(str(tmp_path))
    unreadable = Path(session_log).parent / "supervisor-002.jsonl"
    make_unreadable(unreadable)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)
    replay_calls: list[list[str]] = []

    def fake_replay(_session_path: str, sup_logs: list[str]) -> int:
        replay_calls.append(sup_logs)
        return 1

    _stub_tracing(monkeypatch, replay_session=fake_replay)

    result = runner.invoke(app, ["export", session_log])

    warning_lines = [line for line in result.stderr.splitlines() if line.startswith("warning: ")]
    assert result.exit_code == 0, _output(result)
    assert [[Path(log).name for log in logs] for logs in replay_calls] == [["supervisor-001.jsonl"]]
    assert len(warning_lines) == 1
    assert str(unreadable) in warning_lines[0]
    assert reason in warning_lines[0]
    assert f"exported 1 spans for session {SESSION_ID}" in result.stderr


# ---------------------------------------------------------------------------
# --color / --no-color on export
# ---------------------------------------------------------------------------


def test_export_command_when_no_color_does_strip_ansi_from_stderr_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    missing = str(tmp_path / ".gymrat" / "session.jsonl")

    result = runner.invoke(app, ["export", "--no-color", missing])

    assert result.exit_code == 2
    assert "No such option" not in result.stderr
    assert not SGR_RE.search(result.stderr)
