"""Scripted subprocess double for the stdio-driver tests.

Invoked as ``[sys.executable, <this file>, <config-json>]``. It speaks the
line-delimited JSON protocol the stdio driver expects: it reads the ``start``
command line from stdin, then behaves per the config's ``mode``. It never
imports the gymrat package -- every protocol line is hand-written JSON -- so it
runs as a bare child process.

Config keys (all optional unless noted):

``mode`` (required)
    ``"script"`` -- emit ``lines`` then optional ``outcome`` and exit;
    ``"await_interrupt"`` -- emit ``lines`` then block until an ``interrupt``
    command arrives, optionally emitting ``emit_outcome_on_interrupt`` first;
    ``"sleep_forever"`` -- emit ``lines``, spawn a grandchild sleeper, then
    sleep so an external abort can tree-kill the group.
``lines``
    List of ``{"json": <obj>}`` (serialized) or ``{"text": <str>}`` (verbatim)
    items written to stdout, one per line.
``outcome``
    A terminal outcome object emitted after ``lines`` in ``script`` mode.
``emit_outcome_on_interrupt``
    An outcome object emitted just before exiting on an ``interrupt`` command.
``stderr``
    A string written to stderr (never part of the protocol) to prove stderr is
    not relayed as events.
``report_path``
    A file the double writes a small JSON report to (start line + cwd, or the
    process/grandchild PID values in ``sleep_forever`` mode).
``line_delay_ms``
    Milliseconds to wait before each emitted line.
``exit_code``
    Process exit status (default ``0``).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

_GRANDCHILD_SLEEP_SECONDS = 30


def _writeln(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _emit_lines(config: dict[str, object]) -> None:
    lines = cast("list[dict[str, object]]", config.get("lines", []))
    delay_ms = cast("float", config.get("line_delay_ms", 0))
    for item in lines:
        if delay_ms:
            time.sleep(delay_ms / 1000)
        if "text" in item:
            _writeln(cast("str", item["text"]))
        else:
            _writeln(json.dumps(item["json"]))


def _write_report(config: dict[str, object], payload: dict[str, object]) -> None:
    report_path = config.get("report_path")
    if report_path is not None:
        Path(cast("str", report_path)).write_text(json.dumps(payload), encoding="utf-8")


def _write_stderr(config: dict[str, object]) -> None:
    text = config.get("stderr")
    if text is not None:
        sys.stderr.write(cast("str", text) + "\n")
        sys.stderr.flush()


def _exit_code(config: dict[str, object]) -> int:
    return cast("int", config.get("exit_code", 0))


def _base_report(start_line: str) -> dict[str, object]:
    return {"start_line": start_line.rstrip("\n"), "cwd": str(Path.cwd())}


def _wait_for_command(command_type: str) -> dict[str, object] | None:
    """Read stdin lines until one parses as JSON with a matching "type", or EOF."""
    while True:
        command = sys.stdin.readline()
        if not command:
            return None
        try:
            parsed = json.loads(command)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("type") == command_type:
            return parsed


def _run_script(config: dict[str, object], start_line: str) -> int:
    _write_report(config, _base_report(start_line))
    _write_stderr(config)
    _emit_lines(config)
    outcome = config.get("outcome")
    if outcome is not None:
        _writeln(json.dumps(outcome))
    return _exit_code(config)


def _run_await_interrupt(config: dict[str, object], start_line: str) -> int:
    _write_report(config, _base_report(start_line))
    _emit_lines(config)
    command = _wait_for_command("interrupt")
    if command is not None:
        outcome = config.get("emit_outcome_on_interrupt")
        if outcome is not None:
            _writeln(json.dumps(outcome))
    return _exit_code(config)


def _run_await_message(config: dict[str, object], start_line: str) -> int:
    """Mode that exercises the send/end turn protocol through stdin commands."""
    _write_report(config, _base_report(start_line))
    _emit_lines(config)
    turn_end = config.get("turn_end")
    if turn_end is not None:
        _writeln(json.dumps(turn_end))

    message_text: str | None = None
    message = _wait_for_command("message")
    if message is not None:
        message_text = cast("str", message.get("text", ""))

    _write_report(config, {**_base_report(start_line), "message_text": message_text})

    if turn_end is not None:
        _writeln(json.dumps(turn_end))

    _wait_for_command("end")

    outcome = config.get("outcome")
    if outcome is not None:
        _writeln(json.dumps(outcome))
    return _exit_code(config)


def _run_sleep_forever(config: dict[str, object]) -> int:
    _emit_lines(config)
    grandchild = subprocess.Popen(  # noqa: S603 - fixed argv spawning a local sleeper for the tree-kill test
        [sys.executable, "-c", f"import time; time.sleep({_GRANDCHILD_SLEEP_SECONDS})"]
    )
    _write_report(config, {"pid": os.getpid(), "grandchild": grandchild.pid})
    time.sleep(_GRANDCHILD_SLEEP_SECONDS)
    return 0


def main() -> int:
    """Dispatch to the mode handler named by config['mode']."""
    config = json.loads(sys.argv[1])
    mode = config["mode"]
    if mode == "sleep_forever":
        return _run_sleep_forever(config)
    start_line = sys.stdin.readline()
    if mode == "script":
        return _run_script(config, start_line)
    if mode == "await_interrupt":
        return _run_await_interrupt(config, start_line)
    if mode == "await_message":
        return _run_await_message(config, start_line)
    message = f"unknown mode: {mode}"
    raise SystemExit(message)


if __name__ == "__main__":
    sys.exit(main())
