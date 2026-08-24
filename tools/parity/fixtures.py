"""Deterministic fixture builders and matrix for the parity harness.

The parity harness drives the reference benchmarking CLI against controlled
inputs and diffs the JSON it emits. This module builds those inputs so every run
is reproducible: a one-commit scratch git repo, directory targets carrying shell
"bench" emitters, and the config files that steer verdict selection.

Most targets are plain directories, which the reference CLI resolves "in place"
without a worktree, keeping those runs free of any worktree lifecycle. The
two-ref fixtures instead name git refs (``main`` and ``feature``), which the CLI
checks out into throwaway worktrees; each ref carries its own committed emitter
so a worktree checkout benches the right revision. Either way each emitter prints
a fixed or per-round-varying set of metrics, so every run stays deterministic.

Shell emitters begin with ``#!/bin/sh`` and are invoked as ``sh <name>`` by the
harness, so they never need an executable bit.
"""

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


def _git(cwd: Path, *args: str) -> None:
    """Run a fixed-argv ``git`` command in ``cwd``, raising on failure.

    Args:
        cwd: The working directory for the git invocation.
        args: Arguments passed to ``git``; never interpreted by a shell.

    Raises:
        subprocess.CalledProcessError: When git exits non-zero.
    """
    subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _format_number(value: float) -> str:
    """Render ``value`` as an integer literal when it is integral, else a float.

    Integer rendering keeps the shell arithmetic in the counter emitter simple
    (``$(( ))`` is integer-only) and avoids trailing ``.0`` noise in metric
    lines.

    Args:
        value: The number to render.

    Returns:
        The shortest faithful decimal string for ``value``.
    """
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def create_scratch_repo(path: Path) -> None:
    """Initialize a deterministic one-commit git repo in ``path``.

    The directory must already exist. The repo is created on branch ``main``
    with fixed identity and signing/line-ending settings so its HEAD is stable
    across machines, then seeded with a single ``README.md`` commit.

    Args:
        path: An existing directory to turn into the scratch repo.

    Raises:
        subprocess.CalledProcessError: When any git command fails.
    """
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "commit.gpgsign", "false")
    _git(path, "config", "core.autocrlf", "false")
    (path / "README.md").write_text("# Test Repo\n")
    _git(path, "add", "README.md")
    # -c commit.gpgsign=false defends against a host whose global config forces
    # commit signing, which would otherwise block this non-interactive commit.
    _git(path, "-c", "commit.gpgsign=false", "commit", "-m", "Initial commit")


def create_two_ref_repo(
    path: Path,
    *,
    main_metrics: dict[str, float],
    feature_metrics: dict[str, float],
    feature_ref: str = "feature",
) -> None:
    """Build a scratch repo with ``main`` and a second committed ref.

    Starts from :func:`create_scratch_repo` (branch ``main``, one README commit),
    then commits a metric-lines emitter carrying ``main_metrics`` on ``main``,
    branches ``feature_ref`` off it, and commits an emitter carrying the differing
    ``feature_metrics`` there. HEAD is left on ``main`` with its own ``bench.sh``,
    so ``main`` and ``feature_ref`` each bench a distinct revision — enough for a
    ``compare main feature`` (or ``measure feature``) to exercise the real
    ref-to-worktree lifecycle.

    Args:
        path: An existing directory to turn into the scratch repo.
        main_metrics: Metrics emitted by ``main``'s committed ``bench.sh``.
        feature_metrics: Metrics emitted by ``feature_ref``'s committed ``bench.sh``.
        feature_ref: The name of the second branch to create.

    Raises:
        subprocess.CalledProcessError: When any git command fails.
    """
    create_scratch_repo(path)
    _commit_emitter(path, main_metrics, message="Add bench on main")
    # A committed working tree lets git switch move refs without discarding work;
    # each branch keeps its own bench.sh so a worktree checkout benches that ref.
    _git(path, "switch", "-q", "-c", feature_ref)
    _commit_emitter(path, feature_metrics, message="Vary bench on feature")
    _git(path, "switch", "-q", "main")


def _commit_emitter(path: Path, metrics: dict[str, float], *, message: str) -> None:
    """Write ``bench.sh`` with ``metrics`` at ``path`` and commit it."""
    write_metric_lines_emitter(path, metrics)
    _git(path, "add", "bench.sh")
    _git(path, "-c", "commit.gpgsign=false", "commit", "-m", message)


def write_config(dir: Path, config: dict[str, object], *, name: str = "gymrat.json") -> None:  # noqa: A002
    """Write ``config`` as pretty-printed JSON into ``dir``.

    ``dir`` matches the harness fixture-builder API and is intentionally kept
    despite shadowing the builtin.

    Args:
        dir: The directory to write the config file into.
        config: The configuration document to serialize.
        name: The config file name.
    """
    (dir / name).write_text(json.dumps(config, indent=2))


def write_metric_lines_emitter(
    dir: Path,  # noqa: A002
    metrics: dict[str, float],
    *,
    name: str = "bench.sh",
) -> None:
    """Write a ``metric-lines`` emitter that prints a fixed set of metrics.

    The generated script prints one ``METRIC <name>=<value>`` line per entry in
    ``metrics``. Metric-name suffixes drive metadata in the adapter: ``.../time``
    is a time metric, ``.../heap`` a memory metric, and any other name is an
    "other" metric.

    Args:
        dir: The target directory to write the emitter into.
        metrics: Mapping of metric name to emitted value.
        name: The emitter script file name.
    """
    lines = ["#!/bin/sh"]
    lines.extend(
        f'echo "METRIC {metric_name}={_format_number(value)}"'
        for metric_name, value in metrics.items()
    )
    (dir / name).write_text("\n".join(lines) + "\n")


def write_counter_emitter(  # noqa: PLR0913 -- fixed harness fixture-builder API shape
    dir: Path,  # noqa: A002
    metric: str,
    *,
    base: float,
    step: float,
    bench_name: str = "bench.sh",
    prepare_name: str = "prepare.sh",
) -> None:
    """Write a per-round-varying ``metric-lines`` emitter plus its reset script.

    The bench script reads a ``.round`` counter (defaulting to 0), emits a single
    metric with value ``base + step * round``, then increments and persists the
    counter. The prepare script resets ``.round`` to 0 so a harness ``--prepare``
    invocation starts each target's sequence from ``base``.

    Values use integer shell arithmetic, so ``base`` and ``step`` should be
    integral.

    Args:
        dir: The target directory to write both scripts into.
        metric: The metric name to emit.
        base: The round-zero value.
        step: The per-round increment.
        bench_name: The bench script file name.
        prepare_name: The prepare (reset) script file name.
    """
    base_literal = _format_number(base)
    step_literal = _format_number(step)
    bench = (
        "#!/bin/sh\n"
        "n=$(cat .round 2>/dev/null || echo 0)\n"
        f"value=$(( {base_literal} + {step_literal} * n ))\n"
        f'echo "METRIC {metric}=$value"\n'
        'echo "$(( n + 1 ))" > .round\n'
    )
    (dir / bench_name).write_text(bench)
    (dir / prepare_name).write_text("#!/bin/sh\necho 0 > .round\n")


def write_mitata_emitter(
    dir: Path,  # noqa: A002
    benchmarks: dict[str, dict[str, float]],
    *,
    name: str = "bench.sh",
) -> None:
    """Write a ``mitata`` emitter that prints a mitata-shaped JSON payload.

    Each ``benchmarks`` entry maps an alias to a stats dict with a required
    ``"time"`` (emitted as ``stats.p50`` → ``<alias>/time``) and an optional
    ``"heap"`` (emitted as ``stats.heap.avg`` → ``<alias>/heap``).

    Args:
        dir: The target directory to write the emitter into.
        benchmarks: Mapping of alias to ``{"time": p50, "heap": avg?}``.
        name: The emitter script file name.
    """
    entries: list[dict[str, object]] = []
    for alias, values in benchmarks.items():
        stats: dict[str, object] = {"p50": values["time"]}
        if "heap" in values:
            stats["heap"] = {"avg": values["heap"]}
        run_args: dict[str, object] = {}
        run: dict[str, object] = {"name": alias, "args": run_args, "stats": stats}
        entries.append({"alias": alias, "runs": [run]})
    payload = json.dumps({"benchmarks": entries})
    # A quoted heredoc emits the JSON verbatim, so no shell metacharacter in the
    # payload needs escaping.
    (dir / name).write_text(f"#!/bin/sh\ncat <<'EOF'\n{payload}\nEOF\n")


@dataclass(frozen=True, slots=True)
class Fixture:
    """A single parity-harness scenario.

    Attributes:
        name: Unique identifier for the fixture.
        build: Populates an existing root directory (scratch repo, targets,
            emitters, config) for this scenario.
        argv: Reference-CLI arguments; the run's cwd is the root directory.
        schema_version: Expected ``schemaVersion`` of the returned JSON document
            (2 for ``compare``, 1 for ``measure``).
        records: When true, the fixture drives ``measure --record`` against an
            open session, and the harness additionally verifies the session log
            each side wrote (a cross-implementation round-trip). Plain fixtures
            leave it false and are compared on stdout and exit code alone.
        oracle_exit: The reference CLI's expected exit code. Zero for every
            ordinary fixture; a compare-only fixture sets this to the exit code
            it deliberately provokes (e.g. a validation error), which tells
            self-diff to compare the two runs' exit codes rather than parse
            their stdout as JSON.
    """

    name: str
    build: Callable[[Path], None]
    argv: Sequence[str]
    schema_version: int
    records: bool = False
    oracle_exit: int = 0


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _metric_target(root: Path, name: str, metrics: dict[str, float]) -> None:
    """Create a target subdirectory and populate it with a metric-lines emitter.

    Args:
        root: The fixture's scratch-repo root.
        name: The target subdirectory name, created under ``root``.
        metrics: Mapping of metric name to emitted value, forwarded verbatim.
    """
    (root / name).mkdir()
    write_metric_lines_emitter(root / name, metrics)


def _build_metric_lines_compare_band(root: Path) -> None:
    create_scratch_repo(root)
    _metric_target(root, "base", {"latency/time": 100, "mem/heap": 200, "count": 5})
    _metric_target(root, "cand", {"latency/time": 110, "mem/heap": 190, "count": 6})


def _build_metric_lines_measure(root: Path) -> None:
    create_scratch_repo(root)
    _metric_target(root, "target", {"latency/time": 100, "mem/heap": 200, "count": 5})


# A minimal, valid open-session header (camelCase, TS session schema). The record
# fixtures seed it by hand rather than through the port's writer so the fixture
# stays independent of the code under test; both the reference binary and the
# port then append baseline records to this open log.
_SESSION_HEADER: dict[str, object] = {
    "type": "session",
    "schemaVersion": 1,
    "sessionId": "20260808-141530-a3f2",
    "createdAt": "2026-08-08T14:15:30.000Z",
    "baseline": {"ref": "main", "sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    "branch": "gymrat/20260808-141530-a3f2",
    "worktrees": {
        "experiment": ".gymrat/worktrees/experiment",
        "baseline": ".gymrat/worktrees/baseline",
    },
    "config": {
        "bench": "sh bench.sh",
        "adapter": "metric-lines",
        "samples": 3,
        "timeoutSeconds": 1800,
        "primary": "geomean",
    },
}


def _seed_open_session(root: Path) -> None:
    """Write a single session-header line into ``root``'s fresh session log."""
    session_dir = root / ".gymrat"
    session_dir.mkdir()
    (session_dir / "session.jsonl").write_text(json.dumps(_SESSION_HEADER) + "\n")


def _build_measure_record(root: Path) -> None:
    _build_metric_lines_measure(root)
    _seed_open_session(root)


def _build_mitata_compare(root: Path) -> None:
    create_scratch_repo(root)
    (root / "base").mkdir()
    write_mitata_emitter(
        root / "base", {"encode": {"time": 42, "heap": 1024}, "decode": {"time": 100}}
    )
    (root / "cand").mkdir()
    write_mitata_emitter(
        root / "cand", {"encode": {"time": 45, "heap": 1050}, "decode": {"time": 95}}
    )


def _build_signed_rank_compare(root: Path) -> None:
    create_scratch_repo(root)
    (root / "base").mkdir()
    write_counter_emitter(root / "base", "op/time", base=100, step=10)
    (root / "cand").mkdir()
    write_counter_emitter(root / "cand", "op/time", base=100, step=12)


def _build_exact_compare(root: Path) -> None:
    create_scratch_repo(root)
    write_config(root, {"metrics": {"count": {"exact": True}}})
    _metric_target(root, "base", {"count": 5})
    _metric_target(root, "cand", {"count": 6})


def _build_multi_candidate_compare(root: Path) -> None:
    create_scratch_repo(root)
    _metric_target(root, "base", {"latency/time": 100})
    _metric_target(root, "cand", {"latency/time": 110})
    _metric_target(root, "cand2", {"latency/time": 105})


def _build_zero_diff_compare(root: Path) -> None:
    create_scratch_repo(root)
    _metric_target(root, "base", {"latency/time": 100})
    _metric_target(root, "cand", {"latency/time": 100})


def _build_nan_delta_compare(root: Path) -> None:
    create_scratch_repo(root)
    _metric_target(root, "base", {"ratio/time": 0})
    _metric_target(root, "cand", {"ratio/time": 5})


def _build_one_sided_metric_compare(root: Path) -> None:
    create_scratch_repo(root)
    _metric_target(root, "base", {"shared/time": 10, "only_base/time": 5})
    _metric_target(root, "cand", {"shared/time": 12})


# Shared metrics for the two-ref fixtures: main and feature bench distinct values
# so a ref compare produces a real delta and a ref measure a stable single reading.
_REF_MAIN_METRICS: dict[str, float] = {"latency/time": 100}
_REF_FEATURE_METRICS: dict[str, float] = {"latency/time": 80}


def _build_ref_repo(root: Path) -> None:
    create_two_ref_repo(
        root,
        main_metrics=_REF_MAIN_METRICS,
        feature_metrics=_REF_FEATURE_METRICS,
    )


# Shared flags: JSON output on stdout, color off, and a small sample count keep
# every fixture fast and diffable.
_JSON_FLAGS = ("--format", "json", "--no-color")

# Shared argv for the plain base/cand compare fixtures that need no extra flags.
_STD_COMPARE_ARGV = (
    "compare",
    "base",
    "cand",
    "--bench",
    "sh bench.sh",
    "--samples",
    "3",
    *_JSON_FLAGS,
)

# Shared argv for the record fixtures: measure the target with --record so each
# side appends a baseline record to the open session log.
_RECORD_ARGV = (
    "measure",
    "target",
    "--record",
    "--bench",
    "sh bench.sh",
    "--samples",
    "3",
    *_JSON_FLAGS,
)


def fixture_matrix() -> tuple[Fixture, ...]:
    """Return the full set of deterministic parity fixtures.

    Returns:
        Every named fixture the harness exercises, each ready to build a fresh
        root directory and drive the reference CLI to a JSON document.
    """
    return (
        Fixture(
            name="metric_lines_compare_band",
            build=_build_metric_lines_compare_band,
            argv=_STD_COMPARE_ARGV,
            schema_version=2,
        ),
        Fixture(
            name="metric_lines_measure",
            build=_build_metric_lines_measure,
            argv=("measure", "target", "--bench", "sh bench.sh", "--samples", "3", *_JSON_FLAGS),
            schema_version=1,
        ),
        Fixture(
            name="mitata_compare",
            build=_build_mitata_compare,
            argv=(
                "compare",
                "base",
                "cand",
                "--adapter",
                "mitata",
                "--bench",
                "sh bench.sh",
                "--samples",
                "3",
                *_JSON_FLAGS,
            ),
            schema_version=2,
        ),
        Fixture(
            name="signed_rank_compare",
            build=_build_signed_rank_compare,
            argv=(
                "compare",
                "base",
                "cand",
                "--bench",
                "sh bench.sh",
                "--prepare",
                "sh prepare.sh",
                "--samples",
                "8",
                *_JSON_FLAGS,
            ),
            schema_version=2,
        ),
        Fixture(
            name="exact_compare",
            build=_build_exact_compare,
            argv=(
                "compare",
                "base",
                "cand",
                "--bench",
                "sh bench.sh",
                "--config",
                "gymrat.json",
                "--samples",
                "1",
                *_JSON_FLAGS,
            ),
            schema_version=2,
        ),
        Fixture(
            name="multi_candidate_compare",
            build=_build_multi_candidate_compare,
            argv=(
                "compare",
                "old=base",
                "new=cand",
                "exp=cand2",
                "--bench",
                "sh bench.sh",
                "--samples",
                "3",
                *_JSON_FLAGS,
            ),
            schema_version=2,
        ),
        Fixture(
            name="zero_diff_compare",
            build=_build_zero_diff_compare,
            argv=_STD_COMPARE_ARGV,
            schema_version=2,
        ),
        Fixture(
            name="nan_delta_compare",
            build=_build_nan_delta_compare,
            argv=_STD_COMPARE_ARGV,
            schema_version=2,
        ),
        Fixture(
            name="one_sided_metric_compare",
            build=_build_one_sided_metric_compare,
            argv=_STD_COMPARE_ARGV,
            schema_version=2,
        ),
        Fixture(
            name="ref_compare",
            build=_build_ref_repo,
            argv=(
                "compare",
                "main",
                "feature",
                "--bench",
                "sh bench.sh",
                "--samples",
                "3",
                *_JSON_FLAGS,
            ),
            schema_version=2,
        ),
        Fixture(
            name="ref_measure",
            build=_build_ref_repo,
            argv=(
                "measure",
                "feature",
                "--bench",
                "sh bench.sh",
                "--samples",
                "3",
                *_JSON_FLAGS,
            ),
            schema_version=1,
        ),
        # Record fixtures live at the end so the hermetic slice tests, which take
        # ``fixture_matrix()[:2]`` and ``[0]``, never select them.
        Fixture(
            name="measure_record",
            build=_build_measure_record,
            argv=_RECORD_ARGV,
            schema_version=1,
            records=True,
        ),
        Fixture(
            name="measure_record_missing_session",
            build=_build_metric_lines_measure,
            argv=_RECORD_ARGV,
            schema_version=1,
            oracle_exit=2,
        ),
    )
