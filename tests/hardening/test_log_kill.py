"""Session-log hardening tests over real subprocesses killed mid-write.

Where :mod:`tests.session.test_store` drives the store through hand-written
partial lines in one process, these tests pin the guarantees that only surface
when a real process dies while the log is being appended:

- a process hard-killed while appending leaves a log the next run reads
  cleanly: the torn final line is dropped and every record appended before the
  kill is intact,
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

from gymrat_py.session import session_jsonl_path
from gymrat_py.session.store import append_record, read_records
from tests.session._records import committed_keep, session_record

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only signals and named pipes"
)

# The repository root, so a child process can import the ``tests`` package for
# the shared record builders alongside the installed ``gymrat_py``.
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
    return subprocess.Popen(  # noqa: S603
        [sys.executable, str(script), *args],
        cwd=str(REPO_ROOT),
        env=_child_env(),
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_file(path: Path, timeout_s: float = 30.0) -> None:
    """Poll until ``path`` exists."""
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() > deadline:
            message = f"file never appeared at {path}"
            raise AssertionError(message)
        time.sleep(0.01)


def _wait_until_grew_or_dead(
    path: Path, baseline: int, child: subprocess.Popen[str], timeout_s: float = 30.0
) -> bool:
    """Poll until ``path`` grows past ``baseline`` bytes, or the child exits first.

    Growth past the clean prefix means the child has begun flushing its final,
    deliberately huge record to disk, so a kill now lands mid-append. If the
    child finishes the whole record first it exits, and the caller kills a
    corpse — the log then holds a complete final record rather than a torn one.

    Returns ``True`` when the log grew (kill should tear the record), ``False``
    when the child finished first (record is complete).
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if path.stat().st_size > baseline:
                return True
        except FileNotFoundError:
            pass
        if child.poll() is not None:
            return False
        if time.monotonic() > deadline:
            message = f"log at {path} never grew past {baseline} bytes"
            raise AssertionError(message)
        time.sleep(0.001)


# A child that appends a session header and ``clean_count`` small keeps, signals
# it is ready, then appends one final keep whose message is huge. The huge
# record takes many write bursts to reach disk, so a kill after the log starts
# growing lands mid-append and tears the final line.
_TORN_TAIL_CHILD = """\
import sys
from pathlib import Path

from gymrat_py.session import session_jsonl_path
from gymrat_py.session.store import append_record
from tests.session._records import committed_keep, session_record

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

from gymrat_py.session import session_jsonl_path
from gymrat_py.session.store import append_record
from tests.session._records import session_record

root = sys.argv[1]
path = session_jsonl_path(root)
append_record(path, session_record())
os._exit(0)
"""

# A child that blocks on a shared named pipe, then appends ``count`` keeps whose
# sequence numbers start at ``base``, so several children race to append to one
# log the instant the parent opens the pipe.
_RACE_CHILD = """\
import os
import sys

from gymrat_py.session import session_jsonl_path
from gymrat_py.session.store import append_record
from tests.session._records import committed_keep

root, barrier_path, count_raw, base_raw = sys.argv[1:5]
count = int(count_raw)
base = int(base_raw)
path = session_jsonl_path(root)

barrier_fd = os.open(barrier_path, os.O_RDONLY)
os.read(barrier_fd, 1)
os.close(barrier_fd)

for offset in range(count):
    append_record(path, committed_keep(seq=base + offset))
"""


# ---------------------------------------------------------------------------
# a process hard-killed mid-append leaves a log the next run reads cleanly
# ---------------------------------------------------------------------------


def test_append_record_when_hard_killed_mid_append_does_leave_a_log_the_next_run_reads_clean(
    tmp_path: Path,
):
    root = str(tmp_path)
    path = Path(session_jsonl_path(root))
    clean_count = 5
    ready_flag = tmp_path / "ready.flag"
    child = _spawn(
        tmp_path,
        "torn_tail_child",
        _TORN_TAIL_CHILD,
        root,
        str(clean_count),
        str(ready_flag),
        str(4_000_000),
    )
    try:
        _wait_for_file(ready_flag)
        clean_size = path.stat().st_size
        _wait_until_grew_or_dead(path, clean_size, child)
        child.kill()
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        if child.stderr is not None:
            child.stderr.close()

    records = read_records(str(path))
    expected_prefix = [session_record(), *(committed_keep(seq=seq) for seq in range(clean_count))]
    assert records[: len(expected_prefix)] == expected_prefix

    if len(records) == len(expected_prefix) + 1:
        pytest.skip("child flushed the full record before the kill tore it — no torn tail to test")

    assert len(records) == len(expected_prefix)


# ---------------------------------------------------------------------------
# a record is durable the instant append_record returns
# ---------------------------------------------------------------------------


def test_append_record_when_process_hard_exits_right_after_return_does_keep_the_record(
    tmp_path: Path,
):
    root = str(tmp_path)

    child = _spawn(tmp_path, "hard_exit_child", _HARD_EXIT_CHILD, root)
    _, stderr = child.communicate(timeout=30)

    assert child.returncode == 0, stderr
    assert read_records(session_jsonl_path(root)) == [session_record()]


# ---------------------------------------------------------------------------
# separate processes appending together never interleave bytes within a line
# ---------------------------------------------------------------------------


def test_append_record_when_processes_append_together_does_never_interleave_bytes_in_a_line(
    tmp_path: Path,
):
    root = str(tmp_path)
    path = Path(session_jsonl_path(root))
    append_record(str(path), session_record())
    process_count = 4
    per_process = 150
    barrier = tmp_path / "barrier.pipe"
    os.mkfifo(barrier)

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
    go_fd = os.open(str(barrier), os.O_RDWR)
    try:
        os.write(go_fd, b"\x00" * process_count)
        outcomes = [
            (child.wait(timeout=60), child.stderr.read() if child.stderr else "")
            for child in children
        ]
    finally:
        os.close(go_fd)
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
