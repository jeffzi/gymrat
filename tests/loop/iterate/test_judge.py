"""How ``gymrat.loop.iterate.judge`` quotes metric names into the confirmation rerun.

``shell_quote_name`` quotes each regressed metric name for the platform's shell,
and the rerun's bench string hands those names to that shell through the filter
command. The POSIX case runs the rerun's command through a real ``sh``; the
Windows case pins the cmd-style quoting it chooses instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat.loop.iterate.judge import shell_quote_name
from gymrat.loop.iterate.run import iterate_session
from tests._config import resolved_config
from tests.loop.iterate._fixtures import BASELINE_MS, PairedRun, scaled, stub_runs

if TYPE_CHECKING:
    from tests.loop.iterate._fixtures import CollectSamplesRecorder

# ---------------------------------------------------------------------------
# shell_quote_name
# ---------------------------------------------------------------------------

# The filter command reaches a POSIX shell, which is what decides where one
# argument ends and the next begins.
_ARGS_SCRIPT = '#!/bin/sh\nfor arg in "$@"; do\n  echo "$arg"\ndone\n'
_ARGS_FILTER = "sh args.sh {names}"


def _paired_with(name: str, values: list[float]) -> list[dict[str, float]]:
    """One round per entry, reporting ``name`` beside a plainly named metric."""
    return [{name: value, "total_ms": value} for value in values]


def _shell_args(directory: str, command: str) -> list[str]:
    """The arguments a POSIX shell hands the stand-in bench when it runs ``command``.

    The rerun's bench string is handed to a shell verbatim, so running it through
    one is the only assertion that speaks to what the bench is really given — a
    string comparison would pass for a command the shell refuses outright.

    Args:
        directory: The directory the shell runs in.
        command: The bench command string to hand the shell.

    Returns:
        Each non-empty line the stand-in bench printed, one per argument.
    """
    printed = subprocess.run(  # noqa: S603 -- argv is a fixed shell plus a test-built command
        ["sh", "-c", command],  # noqa: S607 -- the POSIX shell is resolved from PATH on purpose
        cwd=directory,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in printed.split("\n") if line]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell argument splitting only")
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("sort(n=1000)/time", id="parentheses-and-equals"),
        pytest.param("decode large payload", id="space"),
        pytest.param("it's/time", id="single-quote"),
    ],
)
async def test_iterate_session_when_metric_name_needs_quoting_does_pass_it_as_one_argument(
    open_repo: str, samples_mock: CollectSamplesRecorder, name: str
):
    (Path(open_repo) / "args.sh").write_text(_ARGS_SCRIPT, encoding="utf-8")
    regressed = PairedRun(
        experiment=_paired_with(name, scaled(BASELINE_MS, 1.1)),
        baseline=_paired_with(name, BASELINE_MS),
    )
    stub_runs(samples_mock, open_repo, [regressed, regressed])

    await iterate_session(open_repo, resolved_config(filter=_ARGS_FILTER))

    assert _shell_args(open_repo, samples_mock.calls[1].options.bench) == [name, "total_ms"]


def test_shell_quote_name_when_win32_does_double_quote_a_name_with_spaces(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sys, "platform", "win32")

    result = shell_quote_name("decode large payload")

    assert result == '"decode large payload"'
