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

import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from gymrat.config import BenchlessConfig, StopConfig
from gymrat.errors import GymratError, hint_of
from gymrat.loop.status import status_session
from gymrat.session import (
    BaselineRecord,
    BaselineRef,
    IterationPrimary,
    IterationRecord,
    KeepChecks,
    KeepRecord,
    SessionLogRecord,
    SessionRecord,
    Worktrees,
    session_jsonl_path,
)
from tests.session._records import (
    AT,
    SESSION_ID,
    blocked_keep,
    committed_keep,
    discard_record,
    finalize_record,
    hook_record,
    iteration_record,
    session_record,
    write_session_log,
)

if TYPE_CHECKING:
    from gymrat.session.schema import Outcome

# A 40-hex baseline sha whose first seven characters are recognizable on their own.
_BASELINE_SHA = "a1b2c3d" + "e" * 33
# A 40-hex keep-commit sha whose first seven characters are recognizable on their own.
_KEEP_COMMIT = "b1b2b3b" + "c" * 33
# The runbook path a session's config points an agent at, when it has one.
_RUNBOOK_PATH = "docs/runbook.md"

# The four lines every report opens on: the session, its branch, and its worktrees.
_HEADER_LINE_COUNT = 4

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _report_lines(report: str) -> list[str]:
    """The report's lines, stripped of color, with trailing blanks dropped."""
    lines = [_ANSI_RE.sub("", line) for line in report.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _body_lines(report: str) -> list[str]:
    """The report below its header: one line per rendered record, then the totals."""
    return _report_lines(report)[_HEADER_LINE_COUNT:]


def _worktrees(root: str) -> Worktrees:
    """The worktree paths a session under ``root`` records."""
    base = Path(root) / ".gymrat" / "worktrees"
    return Worktrees(experiment=str(base / "experiment"), baseline=str(base / "baseline"))


def _session(root: str) -> SessionRecord:
    """The session header ``start`` writes for ``root``."""
    return session_record(
        baseline=BaselineRef(ref="main", sha=_BASELINE_SHA),
        worktrees=_worktrees(root),
    )


def _config(**overrides: Any) -> BenchlessConfig:
    """A benchless run configuration; ``status`` never benches, so it needs no bench command."""
    default = BenchlessConfig(
        adapter="metric-lines",
        samples=10,
        timeout_seconds=1800,
        unstable_noise_pct=200.0,
        primary="geomean",
    )
    return replace(default, **overrides) if overrides else default


def _iteration(seq: int, delta_pct: float, outcome: Outcome) -> IterationRecord:
    """A measured iteration numbered ``seq``, reading as ``outcome`` on a ``delta_pct`` primary."""
    return iteration_record(
        seq=seq,
        primary=IterationPrimary(kind="geomean", delta_pct=delta_pct),
        outcome=outcome,
    )


def _nothing_measured_keep(seq: int) -> KeepRecord:
    """A keep refused for want of a measurement, numbered ``seq``.

    ``keep`` writes one of these when nothing has been measured since the last
    settle, numbering it past every iteration on file — so the number it carries
    belongs to an iteration that does not exist yet, and may never.
    """
    return blocked_keep(seq, reason="nothing-measured", checks=KeepChecks(configured=True))


# A recorded baseline measurement of ``main``.
_BASELINE = BaselineRecord(
    type="baseline",
    at=AT,
    label="main",
    samples=({"total_ms": 15200}, {"total_ms": 15184}),
)

# A hook run around the first iteration — history ``status`` has no line for.
_HOOK = hook_record()


def four_iterations() -> tuple[SessionLogRecord, ...]:
    """Four measured iterations: one kept, one discarded, one blocked, one unsettled.

    The first iteration is kept, the second discarded, the third's keep is refused
    by the checks gate, and the fourth is still waiting to be settled.
    """
    return (
        _BASELINE,
        _HOOK,
        _iteration(1, -7.2, "improved"),
        committed_keep(1, commit=_KEEP_COMMIT),
        _iteration(2, 9.4, "regressed"),
        discard_record(2),
        _iteration(3, -3.1, "improved"),
        blocked_keep(3),
        _iteration(4, 0.1, "no-signal"),
    )


# ---------------------------------------------------------------------------
# refusing to render
# ---------------------------------------------------------------------------


def test_status_session_when_no_session_does_refuse_pointing_at_start(tmp_path: Path):
    with pytest.raises(GymratError) as exc:
        status_session(str(tmp_path), _config())

    assert "gymrat start" in (hint_of(exc.value) or "")


def test_status_session_when_a_log_line_is_not_json_does_surface_the_store_error_with_path_and_line(
    tmp_path: Path,
):
    root = str(tmp_path)
    write_session_log(root, _session(root))
    with Path(session_jsonl_path(root)).open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")  # cspell:disable-line

    with pytest.raises(GymratError) as exc:
        status_session(root, _config())

    assert f"{session_jsonl_path(root)}:2" in str(exc.value)


# ---------------------------------------------------------------------------
# rendering a whole history
# ---------------------------------------------------------------------------


def test_status_session_when_log_holds_a_whole_history_does_render_header_records_and_totals(
    tmp_path: Path,
):
    root = str(tmp_path)
    write_session_log(root, _session(root), four_iterations())

    report = status_session(root, _config())

    assert _report_lines(report) == [
        f"session {SESSION_ID} · baseline main@a1b2c3d · adapter metric-lines",
        f"branch gymrat/{SESSION_ID}",
        f"experiment worktree {_worktrees(root).experiment}",
        f"baseline worktree {_worktrees(root).baseline}",
        "baseline main · total_ms 15192",
        "iteration 1 · ✓ -7.2% · kept b1b2b3b",
        "iteration 2 · ✗ +9.4% · discarded",
        "iteration 3 · ✓ -3.1% · keep-blocked (checks-failed)",
        "iteration 4 · ~ +0.1% · unsettled",
        "4 iterations · 1 kept · 1 discarded",
    ]


def test_status_session_when_finalized_does_close_the_report_under_the_totals(tmp_path: Path):
    root = str(tmp_path)
    write_session_log(
        root,
        _session(root),
        (
            _iteration(1, -7.2, "improved"),
            committed_keep(1, commit=_KEEP_COMMIT),
            finalize_record(),
        ),
    )

    report = status_session(root, _config())

    assert _body_lines(report) == [
        "iteration 1 · ✓ -7.2% · kept b1b2b3b",
        "1 iteration · 1 kept · 0 discarded",
        f"finalized · branch gymrat/{SESSION_ID}-final · commit ccccccc",
    ]


def test_status_session_when_final_line_torn_does_render_from_the_complete_records(tmp_path: Path):
    root = str(tmp_path)
    write_session_log(
        root,
        _session(root),
        (_iteration(1, -7.2, "improved"), committed_keep(1, commit=_KEEP_COMMIT)),
    )
    with Path(session_jsonl_path(root)).open("a", encoding="utf-8") as handle:
        handle.write('{"type":"itera')  # cspell:disable-line

    report = status_session(root, _config())

    assert _body_lines(report) == [
        "iteration 1 · ✓ -7.2% · kept b1b2b3b",
        "1 iteration · 1 kept · 0 discarded",
    ]


# ---------------------------------------------------------------------------
# the positional settle fold
# ---------------------------------------------------------------------------


def test_status_session_when_nothing_measured_keep_took_a_later_number_does_read_it_unsettled(
    tmp_path: Path,
):
    root = str(tmp_path)
    write_session_log(
        root,
        _session(root),
        (
            _iteration(1, -7.2, "improved"),
            committed_keep(1, commit=_KEEP_COMMIT),
            _nothing_measured_keep(2),
            _iteration(2, -3.1, "improved"),
        ),
    )

    report = status_session(root, _config())

    assert _body_lines(report) == [
        "iteration 1 · ✓ -7.2% · kept b1b2b3b",
        "keep-blocked (nothing-measured)",
        "iteration 2 · ✓ -3.1% · unsettled",
        "2 iterations · 1 kept · 0 discarded",
    ]


def test_status_session_when_no_iteration_followed_a_nothing_measured_keep_does_render_it_anyway(
    tmp_path: Path,
):
    root = str(tmp_path)
    write_session_log(
        root,
        _session(root),
        (
            _iteration(1, -7.2, "improved"),
            committed_keep(1, commit=_KEEP_COMMIT),
            _nothing_measured_keep(2),
        ),
    )

    report = status_session(root, _config())

    assert _body_lines(report) == [
        "iteration 1 · ✓ -7.2% · kept b1b2b3b",
        "keep-blocked (nothing-measured)",
        "1 iteration · 1 kept · 0 discarded",
    ]


def test_status_session_when_a_gating_block_was_superseded_by_a_discard_does_render_both(
    tmp_path: Path,
):
    root = str(tmp_path)
    write_session_log(
        root,
        _session(root),
        (
            _iteration(1, 9.4, "regressed"),
            blocked_keep(1, reason="gating-regression", checks=KeepChecks(configured=True)),
            discard_record(2),
        ),
    )

    report = status_session(root, _config())

    assert _body_lines(report) == [
        "iteration 1 · ✗ +9.4% · discarded",
        "keep-blocked (gating-regression)",
        "1 iteration · 0 kept · 1 discarded",
    ]


def test_status_session_when_a_checks_failed_keep_was_later_resettled_does_render_both(
    tmp_path: Path,
):
    root = str(tmp_path)
    write_session_log(
        root,
        _session(root),
        (
            _iteration(1, -7.2, "improved"),
            blocked_keep(1, reason="checks-failed"),
            committed_keep(1, commit=_KEEP_COMMIT),
        ),
    )

    report = status_session(root, _config())

    assert _body_lines(report) == [
        "iteration 1 · ✓ -7.2% · kept b1b2b3b",
        "keep-blocked (checks-failed)",
        "1 iteration · 1 kept · 0 discarded",
    ]


# ---------------------------------------------------------------------------
# live-config footer and runbook
# ---------------------------------------------------------------------------


def test_status_session_when_stop_configured_does_forward_it_to_the_footer(tmp_path: Path):
    root = str(tmp_path)
    write_session_log(root, _session(root), four_iterations())

    report = status_session(root, _config(stop=StopConfig(max_iterations=30)))

    assert "stop: 4 of 30 iterations" in _report_lines(report)


def test_status_session_when_runbook_configured_does_include_a_runbook_line(tmp_path: Path):
    root = str(tmp_path)
    write_session_log(root, _session(root), four_iterations())

    report = status_session(root, _config(runbook=_RUNBOOK_PATH))

    assert f"runbook {_RUNBOOK_PATH}" in _report_lines(report)


def test_status_session_when_runbook_not_configured_does_omit_the_runbook_line(tmp_path: Path):
    root = str(tmp_path)
    write_session_log(root, _session(root), four_iterations())

    report = status_session(root, _config())

    assert not any("runbook" in line for line in _report_lines(report))


# ---------------------------------------------------------------------------
# color parameter
# ---------------------------------------------------------------------------


def test_status_session_when_color_false_does_suppress_ansi(tmp_path: Path):
    root = str(tmp_path)
    write_session_log(root, _session(root), four_iterations())

    report = status_session(root, _config(), color=False)

    assert "\x1b[" not in report


def test_status_session_when_color_true_does_emit_ansi(tmp_path: Path):
    root = str(tmp_path)
    write_session_log(root, _session(root), four_iterations())

    report = status_session(root, _config(), color=True)

    assert "\x1b[" in report
