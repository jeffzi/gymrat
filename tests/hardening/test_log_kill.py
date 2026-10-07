"""Session-log hardening tests over real subprocesses killed mid-write.

Where :mod:`tests.session.test_store` drives the store through hand-written
partial lines in one process, these tests pin the guarantees that only surface
when a real process dies while the log is being appended:

- a process hard-killed while appending leaves a log the next run reads
  cleanly: every record appended before the kill is intact, and the record
  being written is either dropped or whole — never a parse failure,
- a record is durable the instant ``append_record`` returns, so a process that
  exits abruptly right after the call — the path a signal handler's
  ``os._exit`` takes — never loses it,
- separate processes appending to one log never interleave their bytes within a
  line, so every line the log holds afterwards parses.

Each child runs out of process against a throwaway log under ``tmp_path`` and
builds its records from the same canonical builders the store tests use, so the
parent can compare against exact expected records. The module is POSIX-only: it
relies on real ``SIGKILL`` delivery and a named pipe to start a race together.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import read_records
from tests.hardening._barrier import CHILD_BARRIER, create_barrier, release_together
from tests.session.records._fixtures import (
    append_records,
    committed_keep,
    log_records,
    session_record,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only signals and named pipes"
)

# The repository root, so a child process can import the ``tests`` package for
# the shared record builders alongside the installed ``gymrat``.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _child_env() -> dict[str, str]:
    """A child environment that can import the ``tests`` package for its builders."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    return env


def _spawn(tmp_path: Path, name: str, source: str, *args: str) -> subprocess.Popen[str]:
    """Write ``source`` to a script under ``tmp_path`` and launch it as a child."""
    script = tmp_path / f"{name}.py"
    script.write_text(source, encoding="utf-8")
    return subprocess.Popen(  # noqa: S603 -- argv is a fixed list, not shell-injected
        [sys.executable, str(script), *args],
        cwd=str(REPO_ROOT),
        env=_child_env(),
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_file(path: Path, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() > deadline:
            message = f"file never appeared at {path}"
            raise AssertionError(message)
        time.sleep(0.01)


# A child that appends a session header and ``clean_count`` small keeps, signals
# it is ready, then appends one final keep whose message is huge. The huge
# record takes many write bursts to reach disk, so a kill sent on the ready
# signal usually lands mid-append; where the kernel finishes the write first
# (macOS), the final record lands whole instead.
_TORN_TAIL_CHILD = """\
import sys
from pathlib import Path

from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import append_record
from tests.session.records._fixtures import committed_keep, session_record

root, clean_count_raw, ready_flag, huge_chars_raw = sys.argv[1:5]
clean_count = int(clean_count_raw)
huge_chars = int(huge_chars_raw)
path = session_jsonl_path(root)

append_record(path, session_record())
for seq in range(clean_count):
    append_record(path, committed_keep(seq=seq))
Path(ready_flag).write_text("ready", encoding="utf-8")
append_record(path, committed_keep(seq=clean_count, message="x" * huge_chars))
"""

# A child that appends one record, then exits through ``os._exit`` — the abrupt
# path a signal handler takes, which skips interpreter-level buffer flushing.
_HARD_EXIT_CHILD = """\
import os
import sys

from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import append_record
from tests.session.records._fixtures import session_record

root = sys.argv[1]
path = session_jsonl_path(root)
append_record(path, session_record())
os._exit(0)
"""

# A child that blocks on a shared named pipe, then appends ``count`` keeps whose
# sequence numbers start at ``base``, so several children race to append to one
# log the instant the parent opens the pipe.
_RACE_CHILD = (
    CHILD_BARRIER
    + """\
import sys

from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import append_record
from tests.session.records._fixtures import committed_keep

root, barrier_path, count_raw, base_raw = sys.argv[1:5]
count = int(count_raw)
base = int(base_raw)
path = session_jsonl_path(root)

wait_at_barrier(barrier_path)

for offset in range(count):
    append_record(path, committed_keep(seq=base + offset))
"""
)


# ---------------------------------------------------------------------------
# a process hard-killed mid-append leaves a log the next run reads cleanly
# ---------------------------------------------------------------------------


# The final record's message length: large enough that writing it takes many
# write bursts, so the kill usually catches the append in flight.
_HUGE_CHARS = 64_000_000


def test_append_record_when_hard_killed_during_a_large_append_does_leave_the_earlier_records_readable(
    tmp_path: Path,
):
    root = str(tmp_path)
    clean_count = 5
    ready_flag = tmp_path / "ready.flag"
    child = _spawn(
        tmp_path,
        "torn_tail_child",
        _TORN_TAIL_CHILD,
        root,
        str(clean_count),
        str(ready_flag),
        str(_HUGE_CHARS),
    )
    try:
        _wait_for_file(ready_flag)
        child.kill()
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        if child.stderr is not None:
            child.stderr.close()

    records = read_records(session_jsonl_path(root))

    expected_prefix = [session_record(), *(committed_keep(seq=seq) for seq in range(clean_count))]
    assert records[: len(expected_prefix)] == expected_prefix
    # The kill may land before, during, or after the final append: a torn or
    # unwritten record is dropped, and one the kernel finished reads back whole.
    assert records[len(expected_prefix) :] in (
        [],
        [committed_keep(seq=clean_count, message="x" * _HUGE_CHARS)],
    )


# ---------------------------------------------------------------------------
# a record is durable the instant append_record returns
# ---------------------------------------------------------------------------


def test_append_record_when_process_hard_exits_right_after_return_does_keep_the_record(
    tmp_path: Path,
):
    root = str(tmp_path)

    child = _spawn(tmp_path, "hard_exit_child", _HARD_EXIT_CHILD, root)
    try:
        _, stderr = child.communicate(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()

    assert child.returncode == 0, stderr
    assert log_records(root) == [session_record()]


# ---------------------------------------------------------------------------
# separate processes appending together never interleave bytes within a line
# ---------------------------------------------------------------------------


def test_append_record_when_processes_append_together_does_never_interleave_bytes_in_a_line(
    tmp_path: Path,
):
    root = str(tmp_path)
    path = Path(session_jsonl_path(root))
    append_records(root, session_record())
    process_count = 4
    per_process = 150
    barrier = create_barrier(tmp_path)

    children = [
        _spawn(
            tmp_path,
            f"race_child_{index}",
            _RACE_CHILD,
            root,
            str(barrier),
            str(per_process),
            str(index * per_process),
        )
        for index in range(process_count)
    ]
    try:
        release_together(barrier, process_count)
        outcomes = [
            (child.wait(timeout=60), child.stderr.read() if child.stderr else "")
            for child in children
        ]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
            if child.stderr is not None:
                child.stderr.close()

    assert [code for code, _ in outcomes] == [0] * process_count, [err for _, err in outcomes]
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line]
    for line in lines:
        json.loads(line)
    assert len(lines) == 1 + process_count * per_process
    assert len(read_records(str(path))) == 1 + process_count * per_process
