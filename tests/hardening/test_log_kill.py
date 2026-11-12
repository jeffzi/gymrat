"""Session-log hardening tests for appends cut short or run concurrently.

These tests pin two guarantees of the session log:

- a log whose final record was cut off mid-line, as a process killed while
  appending leaves it, reads back cleanly: every record before it is intact,
  and the half-written record is dropped — never a parse failure,
- separate processes appending to one log never interleave their bytes within a
  line, so every line the log holds afterwards parses.

The racing children run out of process against a throwaway log under
``tmp_path`` and build their records from the same canonical builders the store
tests use, so the parent can compare against exact expected records. The module
is POSIX-only: it relies on a named pipe to start the race together.
"""

from pathlib import Path

from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import read_records
from tests._platform import needs_named_pipes
from tests.hardening._barrier import CHILD_BARRIER, racing_children
from tests.session.records._fixtures import (
    append_raw,
    append_records,
    committed_keep,
    log_records,
    session_record,
)

pytestmark = needs_named_pipes

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
# a log whose final line was cut short reads back every complete record
# ---------------------------------------------------------------------------


def test_read_records_when_final_line_cut_short_does_return_the_earlier_records(
    tmp_path: Path,
):
    root = str(tmp_path)
    clean_count = 5
    append_records(root, session_record(), *(committed_keep(seq=seq) for seq in range(clean_count)))
    line = committed_keep(seq=clean_count).model_dump_json(exclude_none=True).encode()
    append_raw(session_jsonl_path(root), line[: len(line) // 2])

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
