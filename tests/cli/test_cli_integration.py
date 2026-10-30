"""End-to-end CLI tests over real subprocesses, repos, and shell bench scripts.

These exercise paths :class:`typer.testing.CliRunner` cannot reach: running the
installed entry module out of process so the real lock and worktree lifecycle
run. ``python -m gymrat.cli.app`` stands in for the ``gymrat`` console script.
Signal-driven cleanup is pinned in :mod:`tests.hardening.test_signal_cleanup`.
"""

import json
from pathlib import Path

from tests._cli import run_cli
from tests._git import EMIT_ONE_BENCH
from tests._platform import needs_posix_signals

pytestmark = needs_posix_signals

# ---------------------------------------------------------------------------
# lock-free run outside a repo
# ---------------------------------------------------------------------------


def test_cli_when_outside_repo_does_measure_lock_free(tmp_path: Path):
    (tmp_path / "bench.sh").write_text(EMIT_ONE_BENCH, encoding="utf-8")

    result = run_cli(
        ["measure", "--bench", "sh bench.sh", "--samples", "2", "--format", "json"],
        tmp_path,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["metrics"]["x"]["median"] == 1.0
