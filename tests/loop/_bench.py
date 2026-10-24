"""A runnable ``metric-lines`` bench harness for the loop integration tests.

The bench a worktree runs is a real Python script that reports whatever
``tuning.txt`` holds, defaulting to the untuned latency when the checkout has no
tuning file. :func:`commit_project` commits the script, its config, and a
gitignore into a scratch repository so every worktree the loop checks out carries
a runnable bench.

With a gate file the bench blocks until that file appears — the only way to hold
a run open long enough for a second command to collide with it without betting on
a sleep outlasting the first run. The gate is a plain poll loop.

The ``_`` prefix marks a shared helper rather than a test module; it is
imported as ``tests.loop._bench``.
"""

import json
import sys
from pathlib import Path

import tomli_w

from gymrat.session.paths import experiment_worktree_dir
from tests._git import run_git

#: Generous budget for a CLI run that creates real worktrees and spawns real benches.
LONG_RUN_TIMEOUT = 180

#: The bench script every worktree runs.
BENCH_FILE = "bench.py"

#: The file an iteration's edit tunes; the bench reports its contents as latency.
TUNING_FILE = "tuning.txt"

#: The latency the bench reports from a checkout with no tuning file.
BASELINE_LATENCY = 100


def bench_script(gate_file: str | None = None) -> str:
    """Python source for a ``metric-lines`` bench that reports ``tuning.txt``.

    The script reads ``tuning.txt`` relative to its working directory — the
    worktree the bench runs in — and prints ``METRIC latency=<contents>``,
    falling back to :data:`BASELINE_LATENCY` when the file is absent.

    Args:
        gate_file: When given, the script first spin-waits until that file
            exists (polling every 25ms, up to a 60s deadline) before it
            measures anything, so a caller can hold the run open by
            withholding the file.

    Returns:
        The bench script's Python source.
    """
    lines = ["import sys"]
    if gate_file is not None:
        lines.append("import time")
    lines.append("from pathlib import Path")
    if gate_file is not None:
        lines += [
            f"gate = Path({json.dumps(gate_file)})",
            "deadline = time.monotonic() + 60",
            "while not gate.exists() and time.monotonic() < deadline:",
            "    time.sleep(0.025)",
        ]
    lines += [
        "try:",
        f'    tuned = Path({json.dumps(TUNING_FILE)}).read_text(encoding="utf-8").strip()',
        "except FileNotFoundError:",
        f'    tuned = "{BASELINE_LATENCY}"',
        'sys.stdout.write("METRIC latency=" + tuned + "\\n")',
    ]
    return "\n".join(lines) + "\n"


def tune_experiment(repo: str, latency: int) -> None:
    """Tune the experiment worktree to ``latency``, the edit an agent would make."""
    (Path(experiment_worktree_dir(repo)) / TUNING_FILE).write_text(f"{latency}\n", encoding="utf-8")


#: A rerun template that scopes the bench to the metric names it is given.
#: The bench script ignores the extra arguments, so a scoped run measures the
#: same metric a whole run does.
FILTER_TEMPLATE = f"{sys.executable} {BENCH_FILE} --filter {{names}}"


def config_text(*, samples: int = 5, filter_template: str | None = None) -> str:
    """Render the ``gymrat.toml`` a bench project runs under.

    Args:
        samples: Paired samples per target the config asks for.
        filter_template: Rerun template that scopes the bench to named metrics,
            or None to leave the project without one — a probe given names then
            has no way to narrow the bench.

    Returns:
        The rendered TOML text.
    """
    config: dict[str, object] = {
        "bench": f"{sys.executable} {BENCH_FILE}",
        "adapter": "metric-lines",
        "samples": samples,
        "timeout_seconds": 120,
    }
    if filter_template is not None:
        config["filter"] = filter_template
    return tomli_w.dumps(config)


def commit_project(
    repo_dir: str,
    *,
    samples: int = 5,
    gate_file: str | None = None,
    filter_template: str | None = None,
) -> None:
    """Commit the bench script, config, and gitignore into ``repo_dir``.

    Every worktree the loop later checks out inherits the commit, so each one
    carries a runnable bench. The config names the bench with the current
    interpreter's absolute path, so it runs from any worktree's working
    directory.

    Args:
        repo_dir: Repository the project is committed into.
        samples: Paired samples per target the committed config asks for.
        gate_file: Path the bench waits for before measuring, or None to
            measure straight away.
        filter_template: Rerun template the committed config carries, or None
            to commit a project without one.
    """
    files = {
        ".gitignore": ".gymrat/\n",
        BENCH_FILE: bench_script(gate_file),
        "gymrat.toml": config_text(samples=samples, filter_template=filter_template),
    }
    for name, content in files.items():
        (Path(repo_dir) / name).write_text(content, encoding="utf-8")
    run_git(["add", *files], repo_dir)
    run_git(["commit", "-m", "bench harness"], repo_dir)
