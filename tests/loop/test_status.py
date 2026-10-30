"""Behavioral tests for ``status_session``: rendering an open session's history.

Every test lays a session log down on disk with the real record builders in a
throwaway temp root, then drives the real ``status_session`` over it — nothing
is mocked, since the module under test is a read of the log plus a pure render.
The plain assertions strip color so a stray ``FORCE_COLOR`` in the environment
cannot bleed ANSI into a line comparison.

The settle-fold cases are the heart of the suite: a nothing-measured keep that
took a later iteration's number, a gating block superseded by a discard, and a
not-improved keep later resettled all exercise the positional fold that decides
which record settles which iteration. The stop footer and runbook line come from
the *live* config, never the session snapshot, so those cases pass a config the
snapshot never carried.

Expected lines are built with the ``format_status_*`` helpers the report module
owns, so these tests pin which line the fold produces and in what order, not the
wording of each line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.loop.status import status_data, status_session
from gymrat.report.loop import (
    SettleDiscarded,
    SettleKeepBlocked,
    SettleKept,
    SettleState,
    SettleUnsettled,
    StatusIteration,
    StatusSummary,
    format_status_baseline,
    format_status_finalized,
    format_status_footer,
    format_status_header,
    format_status_iteration,
    format_status_settle,
    format_status_stop,
)
from gymrat.session.workspace import BaselineRef
from tests._ansi import (
    strip_sgr,
)
from tests._config import benchless_config as _config
from tests.report._assertions import render_plain
from tests.session._store_records import (
    BASELINE,
    HOOK,
)
from tests.session.records._fixtures import (
    RECOGNIZABLE_BASELINE_SHA,
    RECOGNIZABLE_KEEP_COMMIT,
    blocked_keep,
    committed_keep,
    discard_record,
    finalize_record,
    gate_block,
    make_iteration,
    session_record,
    stop_record,
    worktrees_at,
    write_session_log,
)

if TYPE_CHECKING:
    from pathlib import Path

    from gymrat.session.records import (
        Outcome,
        SessionLogRecord,
        SessionRecord,
    )

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
        baseline=BaselineRef(ref="main", sha=RECOGNIZABLE_BASELINE_SHA),
        worktrees=worktrees_at(root),
    )


def four_iterations() -> tuple[SessionLogRecord, ...]:
    """Build a session history of four measured iterations, each settled differently.

    The first iteration is kept, the second discarded, the third's keep is refused
    by the checks gate, and the fourth is still waiting to be settled.

    Returns:
        The records in log order, opening on a baseline and a hook record.
    """
    return (
        BASELINE,
        HOOK,
        make_iteration(-7.2, "improved"),
        committed_keep(1, commit=RECOGNIZABLE_KEEP_COMMIT),
        make_iteration(9.4, "regressed", seq=2),
        discard_record(2),
        make_iteration(-3.1, "improved", seq=3),
        blocked_keep(3),
        make_iteration(0.1, "no-signal", seq=4),
    )


def _plain(*lines: str) -> list[str]:
    """The ``format_status_*`` lines as a report shows them, stripped of color."""
    return render_plain(*lines).split("\n")


def _iteration(seq: int, delta_pct: float, outcome: Outcome, settle: SettleState) -> str:
    """The line ``status_session`` renders for an iteration settled the way ``settle`` says."""
    return format_status_iteration(StatusIteration(seq, delta_pct, outcome, settle))


def _totals(iterations: int, kept: int, discarded: int) -> list[str]:
    """The totals footer for a session with no stop condition and no target reached."""
    return format_status_footer(StatusSummary(iterations, kept, discarded, target_reached=False))


#: The settle a commit of :data:`RECOGNIZABLE_KEEP_COMMIT` gives an improved iteration.
_KEPT = SettleKept(commit=RECOGNIZABLE_KEEP_COMMIT)

#: The body ``four_iterations`` renders under the header: one line per record, then the totals.
_FOUR_ITERATIONS_BODY = _plain(
    format_status_baseline(BASELINE),
    _iteration(1, -7.2, "improved", _KEPT),
    _iteration(2, 9.4, "regressed", SettleDiscarded()),
    _iteration(3, -3.1, "improved", SettleKeepBlocked(reason="checks-failed")),
    _iteration(4, 0.1, "no-signal", SettleUnsettled()),
    *_totals(4, 1, 1),
)


def _header(root: str) -> list[str]:
    """The header lines a report opens on for the session ``_session`` writes at ``root``."""
    return _plain(*format_status_header(_session(root)))


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
                committed_keep(1, commit=RECOGNIZABLE_KEEP_COMMIT),
                finalize_record(),
            ),
            [
                _iteration(1, -7.2, "improved", _KEPT),
                *_totals(1, 1, 0),
                format_status_finalized(finalize_record()),
            ],
            id="finalized-closes-under-the-totals",
        ),
        pytest.param(
            (
                make_iteration(-7.2, "improved"),
                committed_keep(1, commit=RECOGNIZABLE_KEEP_COMMIT),
                gate_block(2, "nothing-measured"),
                make_iteration(-3.1, "improved", seq=2),
            ),
            [
                _iteration(1, -7.2, "improved", _KEPT),
                format_status_settle(SettleKeepBlocked(reason="nothing-measured")),
                _iteration(2, -3.1, "improved", SettleUnsettled()),
                *_totals(2, 1, 0),
            ],
            id="nothing-measured-keep-before-a-later-iteration",
        ),
        pytest.param(
            (
                make_iteration(9.4, "regressed"),
                gate_block(1, "gating-regression"),
                discard_record(2),
            ),
            [
                _iteration(1, 9.4, "regressed", SettleDiscarded()),
                format_status_settle(SettleKeepBlocked(reason="gating-regression")),
                *_totals(1, 0, 1),
            ],
            id="gating-block-superseded-by-a-discard",
        ),
        pytest.param(
            (
                make_iteration(0.1, "no-signal"),
                gate_block(1, "not-improved"),
                committed_keep(1, commit=RECOGNIZABLE_KEEP_COMMIT),
            ),
            [
                _iteration(
                    1,
                    0.1,
                    "no-signal",
                    SettleKept(commit=RECOGNIZABLE_KEEP_COMMIT, outcome="no-signal"),
                ),
                format_status_settle(SettleKeepBlocked(reason="not-improved")),
                *_totals(1, 1, 0),
            ],
            id="not-improved-keep-later-resettled",
        ),
        pytest.param(
            (
                make_iteration(-7.2, "improved"),
                committed_keep(1, commit=RECOGNIZABLE_KEEP_COMMIT),
                stop_record(message="target reached"),
            ),
            [
                _iteration(1, -7.2, "improved", _KEPT),
                format_status_stop("target reached"),
                *_totals(1, 1, 0),
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

    assert _body_lines(report) == _plain(*body)


# ---------------------------------------------------------------------------
# live-config footer and runbook
# ---------------------------------------------------------------------------


def test_status_session_when_live_config_sets_a_runbook_does_render_its_line_under_the_header(
    tmp_path: Path,
):
    root = str(tmp_path)
    write_session_log(root, _session(root), four_iterations())
    plain = [*_header(root), *_FOUR_ITERATIONS_BODY]

    report = status_session(root, _config(runbook=_RUNBOOK_PATH))

    lines = _report_lines(report)
    assert _RUNBOOK_PATH in lines[_HEADER_LINE_COUNT]
    assert [*lines[:_HEADER_LINE_COUNT], *lines[_HEADER_LINE_COUNT + 1 :]] == plain


# ---------------------------------------------------------------------------
# status_data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("closing", "flags"),
    [
        pytest.param((finalize_record(),), (True, False), id="finalized"),
        pytest.param((stop_record(message="target reached"),), (False, True), id="stopped"),
        pytest.param((), (False, False), id="open"),
    ],
)
def test_status_data_when_session_closes_or_not_does_flag_finalized_and_stopped(
    tmp_path: Path, closing: tuple[SessionLogRecord, ...], flags: tuple[bool, bool]
):
    root = str(tmp_path)
    history = (make_iteration(-7.2, "improved"), committed_keep(1), *closing)
    write_session_log(root, _session(root), history)

    data = status_data(root)

    assert (data.finalized, data.stopped) == flags
