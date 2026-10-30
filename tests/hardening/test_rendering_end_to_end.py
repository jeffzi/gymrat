"""End-to-end rendering hardening across a real TTY and a redirect.

The suite pins the rendering guarantee only a real out-of-process run can show:
a report printed to a real terminal is styled, while the same report piped into
a file or another process is plain.

Every case drives the CLI and a real ``sh`` bench out of process, and the TTY
case attaches stdout to a real pty, so the module is POSIX-only.
"""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

from tests._ansi import SGR_RE, strip_ansi
from tests._cli import run_cli
from tests._git import BENCH_ONCE_FLAGS, EMIT_ONE_BENCH
from tests._git import write_committed_bench as _write_committed_bench
from tests.cli._signalled_cli import spawned_gymrat
from tests.hardening._pty import PTY_SIZE, on_pty_kwargs, pty_capture, pty_env

if TYPE_CHECKING:
    from collections.abc import Callable


def _neutral_env() -> dict[str, str]:
    """A pty child environment with neither ``NO_COLOR`` nor ``FORCE_COLOR`` set.

    Color is then decided purely by terminal detection.

    Returns:
        A copy of the current environment with the terminal variables adjusted.
    """
    env = {
        name: value for name, value in os.environ.items() if name not in ("NO_COLOR", "FORCE_COLOR")
    }
    return pty_env(env)


def _run_report_on_pty(args: list[str], repo: str) -> tuple[int, str, str]:
    """Run the CLI with stdout attached to a real pty.

    The report is written to stdout, so stdout is the pty slave; stderr (the
    progress line) is piped apart so the drawn text is the report alone.

    Args:
        args: The CLI arguments after the entry point.
        repo: The repository the CLI runs in.

    Returns:
        The exit code, the captured stderr, and the text drawn on the pty.
    """
    with (
        pty_capture(size=PTY_SIZE) as terminal,
        spawned_gymrat(
            args,
            repo,
            env=_neutral_env(),
            stdin=subprocess.DEVNULL,
            **on_pty_kwargs(terminal, stdin_on_pty=False, stderr_on_pty=False),
        ) as proc,
    ):
        _, stderr = proc.communicate(timeout=120)
    return proc.returncode, stderr.decode("utf-8", "replace"), terminal.output


# ---------------------------------------------------------------------------
# a real terminal renders styled; a redirect renders plain
# ---------------------------------------------------------------------------


def test_measure_report_when_stdout_is_a_real_tty_does_render_styled(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, EMIT_ONE_BENCH)

    returncode, stderr, output = _run_report_on_pty(["measure", *BENCH_ONCE_FLAGS], repo)

    assert returncode == 0, stderr
    assert "gymrat measure" in strip_ansi(output)
    assert SGR_RE.search(output)


def test_measure_report_when_stdout_is_redirected_does_render_plain(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, EMIT_ONE_BENCH)

    result = run_cli(
        ["measure", *BENCH_ONCE_FLAGS],
        repo,
        check=False,
        timeout=120,
        env=_neutral_env(),
    )

    assert result.returncode == 0, result.stderr
    assert "gymrat measure" in result.stdout
    assert "\x1b[" not in result.stdout
