"""Rendering-matrix hardening for the shared color precedence.

The suite pins the guarantee that keeps color rendering honest no matter where
gymrat's output lands: every color surface (the report on stdout, the doctor
report on stdout, the progress reporter on stderr, and the error text on stderr)
routes through the one shared precedence rule, whose full ladder is pinned in
``tests/test_utils.py``.

The real-terminal, redirect, and bench-environment cases run out of process in
``tests/hardening/test_rendering_end_to_end.py``; these cases exercise the
public rendering surfaces in process and run everywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import create_autospec

import pytest

from gymrat.cli.budget_report import emit_report
from gymrat.cli.commands.doctor import doctor_command
from gymrat.cli.exit import format_cli_error
from gymrat.cli.run_setup import SharedFlags, begin_run
from gymrat.doctor import Check, CheckSection, build_doctor_report
from gymrat.git import NotAGitRepositoryError
from gymrat.report.style import render_lines
from gymrat.report.types import DEFAULT_REPORT_OPTIONS
from gymrat.session.paths import repo_root
from tests._ansi import SGR_RE
from tests._doctor_fixtures import doctor_report
from tests._streams import FakeStream

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.report.json_doc import BudgetSummary
    from gymrat.report.types import ReportOptions


def _render_probe_text(result: str, options: ReportOptions) -> str:
    """A text renderer that paints ``result`` red only when handed a resolved ``True``."""
    # A deferred ``None`` renders plain rather than falling back on the capture
    # console's own detection, so a report surface that stops resolving color
    # against stdout fails the force-on row.
    return render_lines(f"[red]{result}[/red]", color=options.color is True)


def _render_probe_json(result: str, /, *, budget: BudgetSummary | None = None) -> str:
    """A JSON renderer the text-format probe never reaches."""
    del budget
    return result


def _report_is_colored(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Whether ``emit_report`` styles a text report written to a terminal stdout.

    Args:
        monkeypatch: Replaces stdout with a terminal and keeps the budget read
            out of any surrounding repository.

    Returns:
        Whether the report written to stdout carries ANSI.
    """
    stdout = FakeStream(tty=True)
    monkeypatch.setattr("sys.stdout", stdout)
    outside_a_repo = create_autospec(
        repo_root, side_effect=NotAGitRepositoryError("not a git repository")
    )
    monkeypatch.setattr("gymrat.cli.budget_report.repo_root", outside_a_repo)

    emit_report(
        "probe",
        SharedFlags(),
        DEFAULT_REPORT_OPTIONS,
        text=_render_probe_text,
        json=_render_probe_json,
    )

    return "\x1b[" in stdout.getvalue()


def _doctor_report_is_colored(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Whether the doctor command styles its report written to a terminal stdout.

    Args:
        monkeypatch: Replaces stdout with a terminal and the setup probes with
            a fixed passing report.

    Returns:
        Whether the report written to stdout carries ANSI.
    """
    stdout = FakeStream(tty=True)
    monkeypatch.setattr("sys.stdout", stdout)
    report = doctor_report([CheckSection(title="T", checks=[Check("a", "ok", "x")])])
    monkeypatch.setattr(
        "gymrat.cli.commands.doctor.build_doctor_report",
        create_autospec(build_doctor_report, return_value=report),
    )

    doctor_command()

    return "\x1b[" in stdout.getvalue()


def _progress_is_colored(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Whether the run's progress reporter paints color on a terminal stderr.

    Builds the reporter the way a run does, through ``begin_run``, and stops it
    so it prints its closing timing line. A reporter that stops building its
    console through the shared stderr factory fails this probe. The live
    display's cursor control is not color, so only an SGR sequence counts.

    Args:
        monkeypatch: Replaces stderr with a terminal the probe can read back.

    Returns:
        Whether the progress output on stderr carries an SGR sequence.
    """
    stderr = FakeStream(tty=True)
    monkeypatch.setattr("sys.stderr", stderr)

    reporter = begin_run(SharedFlags(), 1)
    reporter.stop()

    return SGR_RE.search(stderr.getvalue()) is not None


def _error_is_colored(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Whether the stderr error surface would paint its label for the environment.

    Args:
        monkeypatch: Unused; taken so every surface probe shares one signature.

    Returns:
        Whether the formatted error carries ANSI.
    """
    del monkeypatch
    return "\x1b[" in format_cli_error(ValueError("boom"))


# ---------------------------------------------------------------------------
# one precedence rule across the report, progress, and error surfaces
# ---------------------------------------------------------------------------


# Environment states where the variables alone decide the outcome, so terminal
# detection never enters into it and every surface must give the same answer.
@pytest.mark.parametrize(
    ("force_and_no_color", "expected"),
    [
        pytest.param(("1", None), True, id="force-on"),
        pytest.param((None, "1"), False, id="no-color-suppresses"),
    ],
)
@pytest.mark.parametrize(
    "is_colored",
    [
        pytest.param(_report_is_colored, id="report"),
        pytest.param(_doctor_report_is_colored, id="doctor-report"),
        pytest.param(_progress_is_colored, id="progress"),
        pytest.param(_error_is_colored, id="error"),
    ],
)
def test_color_surface_when_env_decides_does_follow_the_shared_precedence(
    monkeypatch: pytest.MonkeyPatch,
    color_env: Callable[[str | None, str | None], None],
    is_colored: Callable[[pytest.MonkeyPatch], bool],
    force_and_no_color: tuple[str | None, str | None],
    expected: bool,
):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=True))
    # Pin TERM so the console factory's own terminal detection is capable of
    # color, leaving FORCE_COLOR/NO_COLOR as the only deciders under test.
    monkeypatch.setenv("TERM", "xterm-256color")
    color_env(*force_and_no_color)

    colored = is_colored(monkeypatch)

    assert colored is expected
