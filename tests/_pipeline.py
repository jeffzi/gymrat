"""Shared sampling-pipeline stand-ins and run options for the orchestrator tests.

:func:`install_pipeline` fakes only the process boundary under the sampling
pipeline, so the compare and measure orchestrator tests pin result assembly
through real target resolution, collection and cleanup without spawning
processes or creating worktrees. :func:`run_options` builds the run settings
those tests hand the orchestrators. :func:`is_quantized_warning` recognizes the text sampling warns with
when time readings look quantized.
"""

import re
from collections import Counter
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from gymrat import sampling
from gymrat.exec import ExecOptions, ExecResult
from gymrat.exec import exec as run_exec
from gymrat.sampling import CleanupResult, RunOptions, SamplingOptions
from gymrat.utils import warn_to_stderr
from tests._exec_fixtures import expected_result
from tests._process_helpers import fake_install

CLEAN_RESULT = CleanupResult(removed=0, failures=(), prune_error=None)

QUANTIZED_STEP_NS = 42.0
"""The step, in nanoseconds, between quantized readings that the warning names."""

QUANTIZED_GAP = f"{QUANTIZED_STEP_NS:g}ns"

# A duration the warning would insert as the step, such as ``42ns`` or ``1.5 µs``.
_STEP_VALUE = re.compile(r"\d(?:\.\d+)?\s*[nµm]?s\b")
""":data:`QUANTIZED_STEP_NS` as the quantized-readings warning names it."""


def is_quantized_warning(text: str, name: str, gap: str | None) -> bool:
    """Whether ``text`` is the quantized-readings warning about ``name``.

    Only the parts sampling computes are checked: the metric the warning opens
    on, and the step it names, or no step at all.

    Args:
        text: The warning to check.
        name: The time metric the warning must be about.
        gap: The step the warning must name, such as ``"42ns"``, or ``None``
            when it must name no step.

    Returns:
        ``True`` when ``text`` opens on ``name`` and names exactly that step.
    """
    names_step = _STEP_VALUE.search(text) is None if gap is None else gap in text
    return text.startswith(f"{name}:") and names_step


def run_options(
    *,
    samples: int,
    bench: str = "run",
    prepare: str | None = "prep",
    timeout_seconds: float = 1.0,
) -> RunOptions:
    """Build ``metric-lines`` run options; the defaults suit a stubbed pipeline.

    Args:
        samples: The number of rounds.
        bench: The command run once per target per round.
        prepare: The command run once per target before sampling, or ``None``.
        timeout_seconds: The per-command wall-clock budget.

    Returns:
        Run options for the ``metric-lines`` adapter.
    """
    return RunOptions(
        sampling=SamplingOptions(
            bench=bench,
            prepare=prepare,
            samples=samples,
            timeout_seconds=timeout_seconds,
            on_progress=None,
            warn=warn_to_stderr,
        ),
        adapter="metric-lines",
        config_metrics=None,
        config_kinds=None,
    )


def install_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    rounds: dict[str, list[dict[str, float]]],
    *,
    bench: str = "run",
) -> None:
    """Bench in-place directory targets whose bench run reports canned metrics.

    Each key of ``rounds`` becomes a directory under ``root``, which becomes the
    working directory, so target resolution reads every key as an in-place
    target. Only the process boundary is faked: a stand-in for the ``exec``
    sampling calls answers the n-th bench run in a target's directory with that
    target's n-th round as metric lines, and any other command (the prepare
    step) with no output.

    Args:
        monkeypatch: The fixture the stand-ins and working directory go through.
        root: The directory the target directories are created under.
        rounds: Each target's per-round metrics, keyed by the target name the
            options name.
        bench: The bench command the run options carry.
    """
    for name in rounds:
        (root / name).mkdir()
    monkeypatch.chdir(root)
    benched: Counter[str] = Counter()

    async def run(command: str, options: ExecOptions) -> ExecResult:
        if command != bench:
            return expected_result()
        target = Path(options.cwd).name
        metrics = rounds[target][benched[target]]
        benched[target] += 1
        return expected_result("".join(f"METRIC {key}={value}\n" for key, value in metrics.items()))

    monkeypatch.setattr(sampling, "exec", create_autospec(run_exec, side_effect=run))
    monkeypatch.setattr(sampling, "install_termination_cleanup", fake_install([]))
