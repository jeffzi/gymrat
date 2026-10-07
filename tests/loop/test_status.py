"""Behavioral tests for ``status_session``: rendering an open session's history.

Every test lays a session log down on disk with the real record builders in a
throwaway temp root, then drives the real ``status_session`` over it — nothing
is mocked, since the module under test is a read of the log plus a pure render.
The plain assertions strip color so a stray ``FORCE_COLOR`` in the environment
cannot bleed ANSI into a line comparison.

The settle-fold cases are the heart of the suite: a nothing-measured keep that
took a later iteration's number, a trailing blocked keep with no iteration after
it, a gating block superseded by a discard, and a checks-failed keep later
resettled all exercise the positional fold that decides which record settles
which iteration. The stop footer and runbook line come from the *live* config,
never the session snapshot, so those cases pass a config the snapshot never
carried.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.config import StopConfig
from gymrat.loop.status import status_session
from gymrat.session.records import (
    KeepChecks,
    SessionLogRecord,
    SessionRecord,
)
from gymrat.session.workspace import BaselineRef
from tests._ansi import (
    strip_sgr,
)
from tests._config import benchless_config as _config
from tests.session._store_records import (
    BASELINE,
    HOOK,
)
from tests.session.records._fixtures import (
    SESSION_ID,
    blocked_keep,
    command_record,
    committed_keep,
    discard_record,
    finalize_record,
    make_iteration,
    session_record,
    stop_record,
    worktrees_at,
    write_session_log,
)

if TYPE_CHECKING:
    from pathlib import Path

    from gymrat.config import BenchlessConfig

# A 40-hex baseline sha whose first seven characters are recognizable on their own.
_BASELINE_SHA = "a1b2c3d" + "e" * 33
# A 40-hex keep-commit sha whose first seven characters are recognizable on their own.
_KEEP_COMMIT = "b1b2b3b" + "c" * 33
# The runbook path a session's config points an agent at, when it has one.
_RUNBOOK_PATH = "docs/runbook.md"

# The four lines every report opens on: the session, its branch, and its worktrees.
_HEADER_LINE_COUNT = 4


def _report_lines(report: str) -> list[str]:
    """The report's lines, stripped of color, with trailing blanks dropped."""
    lines = [strip_sgr(line) for line in report.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _body_lines(report: str) -> list[str]:
    """The report below its header: one line per rendered record, then the totals."""
    return _report_lines(report)[_HEADER_LINE_COUNT:]


def _session(root: str) -> SessionRecord:
    """The session header ``start`` writes for ``root``."""
    return session_record(
        baseline=BaselineRef(ref="main", sha=_BASELINE_SHA),
        worktrees=worktrees_at(root),
    )


def four_iterations() -> tuple[SessionLogRecord, ...]:
    """Four measured iterations: one kept, one discarded, one blocked, one unsettled.

    The first iteration is kept, the second discarded, the third's keep is refused
    by the checks gate, and the fourth is still waiting to be settled.
    """
    return (
        BASELINE,
        HOOK,
        make_iteration(-7.2, "improved"),
        committed_keep(1, commit=_KEEP_COMMIT),
        make_iteration(9.4, "regressed", seq=2),
        discard_record(2),
        make_iteration(-3.1, "improved", seq=3),
        blocked_keep(3),
        make_iteration(0.1, "no-signal", seq=4),
    )


#: The body ``four_iterations`` renders under the header: one line per record, then the totals.
_FOUR_ITERATIONS_BODY = [
    "baseline main · total_ms 15192",
    "iteration 1 · ✓ -7.2% · kept b1b2b3b",
    "iteration 2 · ✗ +9.4% · discarded",
    "iteration 3 · ✓ -3.1% · keep-blocked (checks-failed)",
    "iteration 4 · ~ +0.1% · unsettled",
    "4 iterations · 1 kept · 1 discarded",
]


def _header(root: str) -> list[str]:
    """The four header lines a report opens on for the session ``_session`` writes at ``root``."""
    return [
        f"session {SESSION_ID} · baseline main@a1b2c3d · adapter metric-lines",
        f"branch gymrat/{SESSION_ID}",
        f"experiment worktree {worktrees_at(root).experiment}",
        f"baseline worktree {worktrees_at(root).baseline}",
    ]


# ---------------------------------------------------------------------------
# rendering a whole history
# ---------------------------------------------------------------------------


def test_status_session_when_log_holds_a_whole_history_does_render_header_records_and_totals(
    tmp_path: Path,
):
    root = str(tmp_path)
    write_session_log(root, _session(root), four_iterations())

    report = status_session(root, _config())

    assert _report_lines(report) == [*_header(root), *_FOUR_ITERATIONS_BODY]


# ---------------------------------------------------------------------------
# the positional settle fold and closing records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("history", "body"),
    [
        pytest.param(
            (
                make_iteration(-7.2, "improved"),
                committed_keep(1, commit=_KEEP_COMMIT),
                finalize_record(),
            ),
            [
                "iteration 1 · ✓ -7.2% · kept b1b2b3b",
                "1 iteration · 1 kept · 0 discarded",
                f"finalized · branch gymrat/{SESSION_ID}-final · commit ccccccc",
            ],
            id="finalized-closes-under-the-totals",
        ),
        pytest.param(
            (
                make_iteration(-7.2, "improved"),
                committed_keep(1, commit=_KEEP_COMMIT),
                blocked_keep(2, reason="nothing-measured", checks=KeepChecks(configured=True)),
                make_iteration(-3.1, "improved", seq=2),
            ),
            [
                "iteration 1 · ✓ -7.2% · kept b1b2b3b",
                "keep-blocked (nothing-measured)",
                "iteration 2 · ✓ -3.1% · unsettled",
                "2 iterations · 1 kept · 0 discarded",
            ],
            id="nothing-measured-keep-before-a-later-iteration",
        ),
        pytest.param(
            (
                make_iteration(-7.2, "improved"),
                committed_keep(1, commit=_KEEP_COMMIT),
                blocked_keep(2, reason="nothing-measured", checks=KeepChecks(configured=True)),
            ),
            [
                "iteration 1 · ✓ -7.2% · kept b1b2b3b",
                "keep-blocked (nothing-measured)",
                "1 iteration · 1 kept · 0 discarded",
            ],
            id="nothing-measured-keep-with-no-iteration-after",
        ),
        pytest.param(
            (
                make_iteration(9.4, "regressed"),
                blocked_keep(1, reason="gating-regression", checks=KeepChecks(configured=True)),
                discard_record(2),
            ),
            [
                "iteration 1 · ✗ +9.4% · discarded",
                "keep-blocked (gating-regression)",
                "1 iteration · 0 kept · 1 discarded",
            ],
            id="gating-block-superseded-by-a-discard",
        ),
        pytest.param(
            (make_iteration(0.1, "no-signal"), committed_keep(1, commit=_KEEP_COMMIT)),
            [
                "iteration 1 · ~ +0.1% · kept b1b2b3b (no-signal)",
                "1 iteration · 1 kept · 0 discarded",
            ],
            id="no-signal-iteration-kept",
        ),
        pytest.param(
            (make_iteration(9.4, "regressed"), committed_keep(1, commit=_KEEP_COMMIT)),
            [
                "iteration 1 · ✗ +9.4% · kept b1b2b3b (regressed)",
                "1 iteration · 1 kept · 0 discarded",
            ],
            id="regressed-iteration-kept",
        ),
        pytest.param(
            (
                make_iteration(0.1, "no-signal"),
                blocked_keep(1, reason="not-improved", checks=KeepChecks(configured=True)),
                committed_keep(1, commit=_KEEP_COMMIT),
            ),
            [
                "iteration 1 · ~ +0.1% · kept b1b2b3b (no-signal)",
                "keep-blocked (not-improved)",
                "1 iteration · 1 kept · 0 discarded",
            ],
            id="not-improved-keep-later-resettled",
        ),
        pytest.param(
            (
                make_iteration(-7.2, "improved"),
                blocked_keep(1, reason="checks-failed"),
                committed_keep(1, commit=_KEEP_COMMIT),
            ),
            [
                "iteration 1 · ✓ -7.2% · kept b1b2b3b",
                "keep-blocked (checks-failed)",
                "1 iteration · 1 kept · 0 discarded",
            ],
            id="checks-failed-keep-later-resettled",
        ),
        pytest.param(
            (
                make_iteration(-7.2, "improved"),
                committed_keep(1, commit=_KEEP_COMMIT),
                stop_record(message="target reached\ncleaning up"),
            ),
            [
                "iteration 1 · ✓ -7.2% · kept b1b2b3b",
                "stopped · target reached",
                "1 iteration · 1 kept · 0 discarded",
            ],
            id="stop-record-in-file-order",
        ),
    ],
)
def test_status_session_when_history_settles_records_does_render_each_line_in_file_order(
    tmp_path: Path, history: tuple[SessionLogRecord, ...], body: list[str]
):
    root = str(tmp_path)
    write_session_log(root, _session(root), history)

    report = status_session(root, _config())

    assert _body_lines(report) == body


# ---------------------------------------------------------------------------
# live-config footer and runbook
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "lines_above", "line"),
    [
        pytest.param(
            _config(stop=StopConfig(max_iterations=30)),
            10,
            "stop: 4 of 30 iterations",
            id="stop-line-closes-the-footer",
        ),
        pytest.param(
            _config(runbook=_RUNBOOK_PATH),
            _HEADER_LINE_COUNT,
            f"runbook {_RUNBOOK_PATH}",
            id="runbook-line-under-the-header",
        ),
    ],
)
def test_status_session_when_live_config_sets_a_field_does_render_its_line_in_place(
    tmp_path: Path, config: BenchlessConfig, lines_above: int, line: str
):
    root = str(tmp_path)
    write_session_log(root, _session(root), four_iterations())
    plain = [*_header(root), *_FOUR_ITERATIONS_BODY]

    report = status_session(root, config)

    assert _report_lines(report) == [*plain[:lines_above], line, *plain[lines_above:]]


# ---------------------------------------------------------------------------
# CommandRecord is invisible in the rendered history
# ---------------------------------------------------------------------------


def test_status_session_when_command_records_interleaved_does_render_same_lines(
    tmp_path: Path,
):
    root = str(tmp_path)
    history = four_iterations()
    history_with_commands = (
        command_record(seq=0),
        *history[:3],
        command_record(seq=1),
        *history[3:],
        command_record(seq=5),
    )
    write_session_log(root, _session(root), history_with_commands)

    report = status_session(root, _config())

    assert _body_lines(report) == _FOUR_ITERATIONS_BODY


# ---------------------------------------------------------------------------
# color parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("color", "emitted"),
    [
        pytest.param(False, False, id="false-suppresses-despite-force-color"),
        pytest.param(True, True, id="true-forces"),
        pytest.param(None, True, id="none-defers-to-force-color"),
    ],
)
def test_status_session_when_color_given_does_follow_it_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, color: bool | None, emitted: bool
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    root = str(tmp_path)
    write_session_log(root, _session(root), four_iterations())

    report = status_session(root, _config(), color=color)

    assert ("\x1b[" in report) is emitted
