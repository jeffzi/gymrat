"""End-to-end rendering hardening across a real TTY, a redirect, and the bench env.

The suite pins the rendering guarantees only a real out-of-process run can show:

- a report printed to a real terminal is styled, while the same report piped
  into a file or another process is plain,
- the ``--no-color`` flag never leaks into the environment a spawned bench
  command inherits.

Every case drives the CLI and a real ``sh`` bench out of process, and the TTY
case attaches stdout to a real pty, so the module is POSIX-only.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests._ansi import strip_ansi
from tests._cli import ENTRY as _ENTRY
from tests._git import EMIT_ONE_BENCH
from tests._git import run_git as _git
from tests._git import write_committed_bench as _write_committed_bench
from tests.hardening._pty import drain as _drain

if TYPE_CHECKING:
    from collections.abc import Callable

pty = pytest.importorskip("pty", reason="POSIX-only pty and shell bench")

# A bench that records the ``NO_COLOR`` its own environment carries. The parent
# starts with ``NO_COLOR`` unset, so a leak would show up here as ``[1]``.
_ENV_PROBE_BENCH = """#!/bin/sh
printf 'NO_COLOR=[%s]' "${NO_COLOR-<unset>}" > "$GYMRAT_TEST_PROBE"
echo 'METRIC x=1'
"""


def _neutral_env() -> dict[str, str]:
    """A child environment with the color variables cleared and a real TERM.

    Neither ``NO_COLOR`` nor ``FORCE_COLOR`` is set, so color is decided purely
    by terminal detection. ``TERM`` is pinned so a bare CI image still presents
    a color-capable terminal on the pty.

    Returns:
        A copy of the current environment with the color variables adjusted.
    """
    env = dict(os.environ)
    env.pop("NO_COLOR", None)
    env.pop("FORCE_COLOR", None)
    env["TERM"] = "xterm-256color"
    return env


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
    master, slave = pty.openpty()
    proc = subprocess.Popen(  # noqa: S603 -- argv is a fixed list, not shell-injected
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
        _, stderr = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        reader.join(timeout=10)
        os.close(master)
    output = b"".join(chunks).decode("utf-8", "replace")
    return proc.returncode, stderr.decode("utf-8", "replace"), output


# ---------------------------------------------------------------------------
# a real terminal renders styled; a redirect renders plain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "title"),
    [
        pytest.param(
            ["measure", "--bench", "sh bench.sh", "--samples", "1"], "gymrat measure", id="measure"
        ),
        pytest.param(
            ["compare", "main", "candidate", "--bench", "sh bench.sh", "--samples", "1"],
            "gymrat compare",
            id="compare",
        ),
    ],
)
def test_report_when_stdout_is_a_real_tty_does_render_styled(
    create_scratch_repo: Callable[[], str], args: list[str], title: str
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, EMIT_ONE_BENCH)
    _git(["switch", "-c", "candidate"], repo)
    _git(["switch", "main"], repo)

    returncode, stderr, output = _run_report_on_pty(args, repo)

    assert returncode == 0, stderr
    assert title in strip_ansi(output)
    assert "\x1b[" in output


def test_measure_report_when_stdout_is_redirected_does_render_plain(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, EMIT_ONE_BENCH)

    result = subprocess.run(  # noqa: S603 -- argv is a fixed list, not shell-injected
        [*_ENTRY, "measure", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=_neutral_env(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "gymrat measure" in result.stdout
    assert "\x1b[" not in result.stdout


# ---------------------------------------------------------------------------
# --no-color does not leak into a spawned bench's environment
# ---------------------------------------------------------------------------


def test_measure_when_no_color_flag_does_not_leak_no_color_into_the_bench_env(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _ENV_PROBE_BENCH)
    probe = Path(repo) / "env_probe.txt"
    env = _neutral_env()
    env["GYMRAT_TEST_PROBE"] = str(probe)

    result = subprocess.run(  # noqa: S603 -- argv is a fixed list, not shell-injected
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
