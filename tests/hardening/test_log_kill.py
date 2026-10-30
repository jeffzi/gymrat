"""Session-log hardening tests over real subprocesses killed mid-write.

Where :mod:`tests.session.test_store` drives the store through hand-written
partial lines in one process, these tests pin the guarantees that only surface
when a real process dies while the log is being appended:

- a process hard-killed while appending leaves a log the next run reads
  cleanly: every record appended before the kill is intact, and the half-written
  record is dropped — never a parse failure,
- separate processes appending to one log never interleave their bytes within a
  line, so every line the log holds afterwards parses.

Each child runs out of process against a throwaway log under ``tmp_path`` and
builds its records from the same canonical builders the store tests use, so the
parent can compare against exact expected records. The module is POSIX-only: it
relies on real ``SIGKILL`` delivery and a named pipe to start a race together.
"""

from pathlib import Path

from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import read_records
from tests._platform import needs_named_pipes
from tests._process_helpers import reaped, spawn_child_script, wait_for_file_blocking
from tests.hardening._barrier import CHILD_BARRIER, racing_children
from tests.session.records._fixtures import (
    append_records,
    committed_keep,
    log_records,
    session_record,
)

pytestmark = needs_named_pipes

# A child that appends a session header and ``clean_count`` small keeps, then
# appends one final keep through ``append_record`` with the log's append handle
# swapped for one that writes the first half of the line, signals ready and
# stalls. The kill sent on that signal always lands mid-line.
_TORN_TAIL_CHILD = """\
import sys
import time
from pathlib import Path

from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import append_record
from tests.session.records._fixtures import committed_keep, session_record

root, clean_count_raw, ready_flag = sys.argv[1:4]
clean_count = int(clean_count_raw)
path = session_jsonl_path(root)

append_record(path, session_record())
for seq in range(clean_count):
    append_record(path, committed_keep(seq=seq))


class HalfWriter:
    def __init__(self, handle):
        self._handle = handle

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._handle.close()

    def write(self, payload):
        self._handle.write(payload[: len(payload) // 2])
        self._handle.flush()
        Path(ready_flag).write_text("ready", encoding="utf-8")
        time.sleep(60)


real_open = Path.open


def stalling_open(self, *args, **kwargs):
    handle = real_open(self, *args, **kwargs)
    return HalfWriter(handle) if str(self) == path else handle


Path.open = stalling_open
append_record(path, committed_keep(seq=clean_count))
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


def test_append_record_when_hard_killed_mid_line_does_leave_the_earlier_records_readable(
    tmp_path: Path,
):
    root = str(tmp_path)
    clean_count = 5
    ready_flag = tmp_path / "ready.flag"
    child = spawn_child_script(
        tmp_path, "torn_tail_child", _TORN_TAIL_CHILD, root, str(clean_count), str(ready_flag)
    )
    with reaped(child):
        wait_for_file_blocking(ready_flag, timeout_s=30.0)
        child.kill()
        child.wait(timeout=30)

    torn = Path(session_jsonl_path(root)).read_bytes()
    records = read_records(session_jsonl_path(root))

    assert not torn.endswith(b"\n")
    assert records == [session_record(), *(committed_keep(seq=seq) for seq in range(clean_count))]


# ---------------------------------------------------------------------------
# separate processes appending together never interleave bytes within a line
# ---------------------------------------------------------------------------


def test_append_record_when_processes_append_together_does_never_interleave_bytes_in_a_line(
    tmp_path: Path,
):
    root = str(tmp_path)
    append_records(root, session_record())
    process_count = 4
    per_process = 150
    expected_keeps = [committed_keep(seq=seq) for seq in range(process_count * per_process)]

    with racing_children(
        tmp_path,
        _RACE_CHILD,
        lambda barrier, index: (root, str(barrier), str(per_process), str(index * per_process)),
        process_count,
    ) as children:
        outcomes = [
            (child.wait(timeout=60), child.stderr.read() if child.stderr else "")
            for child in children
        ]

    assert [code for code, _ in outcomes] == [0] * process_count, [err for _, err in outcomes]
    header, *keeps = log_records(root)
    assert (header, sorted(keeps, key=repr)) == (session_record(), sorted(expected_keeps, key=repr))
