"""Rendering-matrix hardening across TTY, ``NO_COLOR``, and redirect.

The suite pins the guarantees that keep color rendering honest no matter where
gymrat's output lands:

- a report printed to a real terminal is styled, while the same report piped
  into a file or another process is plain,
- one precedence rule — explicit flag beats ``FORCE_COLOR`` beats ``NO_COLOR``
  beats terminal detection — governs all three color surfaces (the report on
  stdout, the progress line on stderr, and the error text on stderr), with
  ``FORCE_COLOR`` empty/``0``/``false`` not forcing and ``NO_COLOR`` empty
  treated the same everywhere,
- the ``--no-color`` flag never leaks into the environment a spawned bench
  command inherits,
- a terminal that reports zero width neither crashes nor spills a garbled
  status line.

The end-to-end cases drive a real pty and a real bench out of process, so they
are POSIX-only; the precedence cases exercise the public rendering surfaces
directly and run everywhere.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, override

if sys.platform != "win32":
    import pty

import pytest

from gymrat_py.cli.shared import (
    color_override_of,
    format_cli_error,
    resolve_render_mode,
    resolve_stream_color,
)
from gymrat_py.cli.status_line import create_status_line
from gymrat_py.doctor.checks import Check, CheckSection, EnvironmentInfo, create_doctor_report
from gymrat_py.doctor.render import render_doctor_report
from gymrat_py.report.style import shorten_label
from tests.hardening._bench_helpers import drain as _drain
from tests.hardening._bench_helpers import git as _git
from tests.hardening._bench_helpers import write_committed_bench as _write_committed_bench

if TYPE_CHECKING:
    from collections.abc import Callable

from tests._ansi import strip_ansi
from tests._cli import ENTRY as _ENTRY

_CLEAR_LINE = "\r\x1b[K"

_METRIC_BENCH = "#!/bin/sh\necho 'METRIC x=1'\n"

# A bench that records the ``NO_COLOR`` its own environment carries. The parent
# starts with ``NO_COLOR`` unset, so a leak would show up here as ``[1]``.
_ENV_PROBE_BENCH = """#!/bin/sh
printf 'NO_COLOR=[%s]' "${NO_COLOR-<unset>}" > "$GYMRAT_TEST_PROBE"
echo 'METRIC x=1'
"""

_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only pty and shell bench")


def _neutral_env() -> dict[str, str]:
    """A child environment with the color variables cleared and a real TERM.

    Neither ``NO_COLOR`` nor ``FORCE_COLOR`` is set, so color is decided purely
    by terminal detection. ``TERM`` is pinned so a bare CI image still presents
    a color-capable terminal on the pty.
    """
    env = dict(os.environ)
    env.pop("NO_COLOR", None)
    env.pop("FORCE_COLOR", None)
    env["TERM"] = "xterm-256color"
    return env


def _run_report_on_pty(args: list[str], repo: str) -> str:
    """Run the CLI with stdout attached to a real pty and return what it drew.

    The report is written to stdout, so stdout is the pty slave; stderr (the
    progress line) is sent elsewhere so the returned text is the report alone.
    """
    master, slave = pty.openpty()
    proc = subprocess.Popen(  # noqa: S603
        [*_ENTRY, *args],
        cwd=repo,
        env=_neutral_env(),
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave)
    chunks: list[bytes] = []
    reader = threading.Thread(target=_drain, args=(master, chunks))
    reader.start()
    try:
        proc.wait(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        reader.join(timeout=10)
        os.close(master)
        if proc.stderr is not None:
            proc.stderr.close()
    return b"".join(chunks).decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# a real terminal renders styled; a redirect renders plain
# ---------------------------------------------------------------------------


@_posix_only
def test_measure_report_when_stdout_is_a_real_tty_does_render_styled(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _METRIC_BENCH)

    output = _run_report_on_pty(["measure", "--bench", "sh bench.sh", "--samples", "1"], repo)

    assert "gymrat measure" in strip_ansi(output)
    assert "\x1b[" in output


@_posix_only
def test_compare_report_when_stdout_is_a_real_tty_does_render_styled(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _METRIC_BENCH)
    _git(repo, "switch", "-c", "candidate")
    _git(repo, "switch", "main")

    output = _run_report_on_pty(
        ["compare", "main", "candidate", "--bench", "sh bench.sh", "--samples", "1"],
        repo,
    )

    assert "gymrat compare" in strip_ansi(output)
    assert "\x1b[" in output


@_posix_only
def test_measure_report_when_stdout_is_redirected_does_render_plain(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _METRIC_BENCH)

    result = subprocess.run(  # noqa: S603
        [*_ENTRY, "measure", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=_neutral_env(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert "gymrat measure" in result.stdout, result.stderr
    assert "\x1b[" not in result.stdout


# ---------------------------------------------------------------------------
# --no-color does not leak into a spawned bench's environment
# ---------------------------------------------------------------------------


@_posix_only
def test_measure_when_no_color_flag_does_not_leak_no_color_into_the_bench_env(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _ENV_PROBE_BENCH)
    probe = Path(repo) / "env_probe.txt"
    env = _neutral_env()
    env["GYMRAT_TEST_PROBE"] = str(probe)

    result = subprocess.run(  # noqa: S603
        [*_ENTRY, "measure", "--no-color", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert probe.read_text(encoding="utf-8") == "NO_COLOR=[<unset>]"


# ---------------------------------------------------------------------------
# one precedence rule across the report, progress, and error surfaces
# ---------------------------------------------------------------------------


class _FakeStream(io.StringIO):
    """A stdout/stderr stand-in whose TTY status the test controls."""

    def __init__(self, *, tty: bool):
        super().__init__()
        self._tty = tty

    @override
    def isatty(self) -> bool:
        return self._tty


def _report_is_colored() -> bool:
    """Whether the report surface would emit color for the ambient environment.

    Drives the exact decision ``emit_report`` makes for the report — it resolves
    a deferred color with ``resolve_stream_color(render_opts.color, sys.stdout)``,
    so the probe asks the same helper for a terminal stdout. Routing through the
    shared helper (rather than ``render_lines``' own capture-console logic) means
    a change to the report surface's precedence fails this test.
    """
    return resolve_stream_color(None, _FakeStream(tty=True))


def _doctor_report_is_colored() -> bool:
    """Whether the doctor report surface renders ANSI for the ambient environment.

    Builds a minimal doctor report, resolves color the same way the doctor
    command does (``resolve_stream_color`` with no override on a TTY stream),
    and checks whether the rendered text carries ANSI. A change to either the
    doctor renderer's color parameter or the shared precedence fails this probe.
    """
    env = EnvironmentInfo(gymrat_version="0.1.0", python_version="3.13.0", platform="test")
    report = create_doctor_report(
        env, [CheckSection(title="T", checks=[Check("a", "ok", "x"), Check("b", "fail", "y")])]
    )
    color = resolve_stream_color(None, _FakeStream(tty=True))
    return "\x1b[" in render_doctor_report(report, color=color)


def _progress_is_colored() -> bool:
    """Whether the progress surface would paint color for the environment.

    Color is resolved by the console factory, through the same shared
    ``resolve_stream_color`` chain every other surface uses.
    """
    return resolve_stream_color(color_override_of(color=True), sys.stderr)


def _error_is_colored() -> bool:
    """Whether the stderr error surface would paint its label for the environment."""
    return "\x1b[" in format_cli_error(ValueError("boom"))


def _apply_color_env(
    monkeypatch: pytest.MonkeyPatch, force_color: str | None, no_color: str | None
) -> None:
    for name, value in (("FORCE_COLOR", force_color), ("NO_COLOR", no_color)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


# Environment states where the variables alone decide the outcome, so terminal
# detection never enters into it and all three surfaces must agree.
@pytest.mark.parametrize(
    ("force_color", "no_color", "expected"),
    [
        pytest.param("1", None, True, id="force-on"),
        pytest.param("1", "1", True, id="force-beats-no-color"),
        pytest.param("0", "1", False, id="force-zero-does-not-beat-no-color"),
        pytest.param("false", "1", False, id="force-false-does-not-beat-no-color"),
        pytest.param("", "1", False, id="force-empty-does-not-beat-no-color"),
        pytest.param(None, "1", False, id="no-color-suppresses"),
    ],
)
def test_color_precedence_is_consistent_across_report_progress_and_error_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    force_color: str | None,
    no_color: str | None,
    expected: bool,
):
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=True))
    _apply_color_env(monkeypatch, force_color, no_color)

    assert _report_is_colored() is expected
    assert _doctor_report_is_colored() is expected
    assert _progress_is_colored() is expected
    assert _error_is_colored() is expected


@pytest.mark.parametrize(
    "force_color",
    [pytest.param("0", id="zero"), pytest.param("false", id="false"), pytest.param("", id="empty")],
)
def test_error_surface_when_force_color_is_falsy_off_a_tty_does_render_plain(
    monkeypatch: pytest.MonkeyPatch, force_color: str
):
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=False))
    _apply_color_env(monkeypatch, force_color, None)

    assert "\x1b[" not in format_cli_error(ValueError("boom"))


def test_error_surface_when_force_color_is_truthy_off_a_tty_does_render_colored(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=False))
    _apply_color_env(monkeypatch, "1", None)

    assert "\x1b[" in format_cli_error(ValueError("boom"))


def test_progress_surface_when_force_color_is_truthy_off_a_tty_does_not_animate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", _FakeStream(tty=False))
    _apply_color_env(monkeypatch, "1", None)

    assert resolve_render_mode() == "plain"


# ---------------------------------------------------------------------------
# a zero-width terminal degrades gracefully
# ---------------------------------------------------------------------------


def test_shorten_label_when_width_is_zero_does_return_empty_without_garbage():
    assert shorten_label("abcdefghijklmnop", 0) == ""


def _zero_width_terminal(*_args: object, **_kwargs: object) -> os.terminal_size:
    """A ``get_terminal_size`` stand-in for a terminal that reports zero columns."""
    return os.terminal_size((0, 0))


def test_status_line_when_terminal_reports_zero_width_does_not_crash_or_spill(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.setattr("shutil.get_terminal_size", _zero_width_terminal)
    fake = _FakeStream(tty=True)
    monkeypatch.setattr("sys.stderr", fake)
    line = create_status_line("overwrite")

    line.write("abcdefghijklmnop")

    drawn = fake.getvalue()
    assert drawn.startswith(_CLEAR_LINE)
    assert strip_ansi(drawn).strip() == ""


@pytest.mark.parametrize(
    "columns",
    [pytest.param("0", id="zero"), pytest.param("", id="empty"), pytest.param("-5", id="negative")],
)
def test_status_line_when_columns_env_is_non_positive_does_not_spill(
    monkeypatch: pytest.MonkeyPatch, columns: str
):
    monkeypatch.setenv("COLUMNS", columns)
    fake = _FakeStream(tty=True)
    monkeypatch.setattr("sys.stderr", fake)
    line = create_status_line("overwrite")

    line.write("abcdefghijklmnop")

    drawn = fake.getvalue()
    assert drawn.startswith(_CLEAR_LINE)
    assert strip_ansi(drawn).strip() == ""


def test_status_line_when_columns_env_is_positive_does_truncate_to_fit(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("COLUMNS", "10")
    fake = _FakeStream(tty=True)
    monkeypatch.setattr("sys.stderr", fake)
    line = create_status_line("overwrite")

    label = "abcdefghijklmnopqrstuvwxyz0123456789"
    line.write(label)

    visible = strip_ansi(fake.getvalue()).strip()
    assert visible
    assert visible != label
    assert len(visible) <= 10
