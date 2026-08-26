"""A runnable ``metric-lines`` bench harness for the loop integration tests.

The bench a worktree runs is a real Python script that reports whatever
``tuning.txt`` holds, defaulting to the untuned latency when the checkout has no
tuning file. :func:`commit_project` commits the script, its config, and a
gitignore into a scratch repository so every worktree the loop checks out carries
a runnable bench.

With a gate file the bench blocks until that file appears — the only way to hold
a run open long enough for a second command to collide with it without betting on
a sleep outlasting the first run. Python has no ``Atomics.wait``, so the gate is
a plain poll loop.

The module is name-prefixed with ``_`` so pytest never collects it: it is a
helper imported as ``tests.loop._bench``.
"""

import json
import subprocess
import sys
from pathlib import Path

import tomli_w

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

    When ``gate_file`` is given, the script first spin-waits until that file
    exists (polling every 25ms, up to a 60s deadline) before it measures
    anything, so a caller can hold the run open by withholding the file.
    """
    lines = ["import os", "import sys"]
    if gate_file is not None:
        lines += [
            "import time",
            f"gate = {json.dumps(gate_file)}",
            "deadline = time.monotonic() + 60",
            "while not os.path.exists(gate) and time.monotonic() < deadline:",
            "    time.sleep(0.025)",
        ]
    lines += [
        f"tuning = {json.dumps(TUNING_FILE)}",
        "if os.path.exists(tuning):",
        '    with open(tuning, encoding="utf-8") as handle:',
        "        tuned = handle.read().strip()",
        "else:",
        f'    tuned = "{BASELINE_LATENCY}"',
        'sys.stdout.write("METRIC latency=" + tuned + "\\n")',
    ]
    return "\n".join(lines) + "\n"


def commit_project(
    repo_dir: str,
    *,
    samples: int = 5,
    gate_file: str | None = None,
) -> None:
    """Commit the bench script, config, and gitignore into ``repo_dir``.

    Every worktree the loop later checks out inherits the commit, so each one
    carries a runnable bench. The config names the bench with the current
    interpreter's absolute path, so it runs from any worktree's working
    directory.
    """
    files = {
        ".gitignore": ".gymrat/\n",
        BENCH_FILE: bench_script(gate_file),
        "gymrat.toml": tomli_w.dumps(
            {
                "bench": f"{sys.executable} {BENCH_FILE}",
                "adapter": "metric-lines",
                "samples": samples,
                "timeout_seconds": 120,
            }
        ),
    }
    for name, content in files.items():
        (Path(repo_dir) / name).write_text(content, encoding="utf-8")
    subprocess.run(  # noqa: S603
        ["git", "add", *files],  # noqa: S607
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "bench harness"],  # noqa: S607
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
