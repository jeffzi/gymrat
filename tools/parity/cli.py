"""Local parity-harness command-line entry point.

Two commands prove the harness before the port CLI exists:

- ``self-diff`` runs the reference binary on *both* sides of each fixture and
  diffs the two parsed-JSON documents. The emitters are deterministic, so the
  two runs are identical and a green self-diff is the harness proving itself
  before any port output is available to compare against.
- ``compare`` drives the oracle-vs-port comparison through the shared
  :class:`~tools.parity.oracle.Runner` interface. The port side raises until
  v0.8, so today the command exists to prove that seam is wired, not to produce
  a diff.

All human-facing output — reports and errors alike — is written to stdout via a
single rich console so a caller can capture the whole run with one redirect.
"""

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from gymrat_py.errors import GymratError
from tools.parity.differ import DiffEntry, DiffReport, diff_json
from tools.parity.fixtures import Fixture, fixture_matrix
from tools.parity.oracle import (
    OracleRunner,
    PortRunner,
    Runner,
    assert_pinned_sha,
    ensure_built,
    ts_repo_path,
)

app = typer.Typer(add_completion=False, help="Local parity harness for the gymrat port.")
console = Console()


@dataclass(frozen=True, slots=True)
class FixtureDiff:
    """A single fixture's self-diff outcome.

    Attributes:
        name: The fixture's identifier.
        report: The comparison of the fixture's two reference-CLI runs.
    """

    name: str
    report: DiffReport


@dataclass(frozen=True, slots=True)
class SelfDiffOutcome:
    """Aggregate outcome of a self-diff over one or more fixtures."""

    results: tuple[FixtureDiff, ...]

    @property
    def ok(self) -> bool:
        """True iff every fixture's report is green (p-notes never fail it)."""
        return all(result.report.is_green for result in self.results)


def _build_oracle_runner() -> Runner:
    """Build (if needed) and return a runner for the pinned reference CLI.

    Isolated behind this module-level indirection so tests can substitute a
    scripted fake without a node toolchain or reference checkout.

    The pin is enforced on the resolved checkout before any build so a drifted
    reference tree fails loudly instead of silently producing off-commit output.
    """
    repo = ts_repo_path()
    assert_pinned_sha(repo)
    return OracleRunner(ensure_built(repo))


def _emit(text: str) -> None:
    """Print a literal line to stdout, immune to rich markup and wrapping.

    Diff paths contain ``[i]`` index segments that rich would otherwise parse as
    markup, so markup is disabled; ``soft_wrap`` keeps long dotted paths on one
    line for stable downstream matching.
    """
    console.print(text, markup=False, soft_wrap=True)


def _resolve_fixtures(names: list[str]) -> tuple[Fixture, ...]:
    """Resolve fixture names to fixtures, or exit non-zero naming the bad ones.

    An empty ``names`` selects the whole matrix. Any unknown name is reported
    before the caller touches a runner, so nothing is executed on a typo.
    """
    matrix = fixture_matrix()
    if not names:
        return matrix
    by_name = {fixture.name: fixture for fixture in matrix}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        _emit(f"unknown fixture(s): {', '.join(unknown)}")
        _emit(f"known fixtures: {', '.join(sorted(by_name))}")
        raise typer.Exit(code=2)
    return tuple(by_name[name] for name in names)


def _run_json(runner: Runner, fixture: Fixture, root: Path) -> object:
    """Run a fixture's argv through ``runner`` and parse its stdout as JSON."""
    result = runner.run(list(fixture.argv), cwd=root)
    if result.exit_code != 0:
        msg = f"fixture {fixture.name!r} exited {result.exit_code}: {result.stderr.strip()}"
        raise GymratError(msg)
    return json.loads(result.stdout)


def run_self_diff(runner: Runner, fixtures: Sequence[Fixture]) -> SelfDiffOutcome:
    """Diff the reference binary against itself for each fixture.

    Each fixture is built into a fresh temporary directory, then its argv is run
    twice through ``runner`` (cwd = that directory) and the two parsed-JSON
    documents are compared with :func:`~tools.parity.differ.diff_json`. Identical
    deterministic output yields a green report; volatile worktree fields are
    normalized by the differ.

    Args:
        runner: The runner used for both sides of every fixture.
        fixtures: The fixtures to run, in order.

    Returns:
        The per-fixture reports wrapped in a :class:`SelfDiffOutcome`.
    """
    results: list[FixtureDiff] = []
    for fixture in fixtures:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture.build(root)
            first = _run_json(runner, fixture, root)
            second = _run_json(runner, fixture, root)
            report = diff_json(first, second)
        results.append(FixtureDiff(name=fixture.name, report=report))
    return SelfDiffOutcome(results=tuple(results))


def _render_entries(header: str, entries: Sequence[DiffEntry]) -> None:
    """Print a labelled block of diff entries as ``path: left -> right`` lines."""
    if not entries:
        return
    _emit(header)
    for entry in entries:
        _emit(f"  {entry.path}: {entry.left!r} -> {entry.right!r}")


def _render_self_diff(outcome: SelfDiffOutcome) -> None:
    """Render the per-fixture summary table plus any differences and p-notes."""
    table = Table(title="parity self-diff")
    table.add_column("fixture")
    table.add_column("status")
    for result in outcome.results:
        status = (
            Text("GREEN", style="green") if result.report.is_green else Text("RED", style="red")
        )
        table.add_row(result.name, status)
    console.print(table)

    for result in outcome.results:
        _render_entries(f"{result.name} differences:", result.report.differences)
        _render_entries(f"{result.name} p-notes (informational):", result.report.p_notes)


@app.command("self-diff")
def self_diff_command(
    fixtures: Annotated[
        list[str] | None,
        typer.Argument(help="Fixture names to run; omit to run the whole matrix."),
    ] = None,
) -> None:
    """Diff the reference binary against itself over the selected fixtures."""
    selected = _resolve_fixtures(fixtures or [])
    runner = _build_oracle_runner()
    outcome = run_self_diff(runner, selected)
    _render_self_diff(outcome)
    if not outcome.ok:
        raise typer.Exit(code=1)


@app.command("compare")
def compare_command(
    fixture: Annotated[str, typer.Argument(help="Fixture name to compare oracle vs port.")],
) -> None:
    """Compare the oracle against the port for one fixture.

    The port CLI does not exist until v0.8, so :class:`PortRunner` raises and the
    command surfaces that. This proves the oracle-vs-port seam is wired ahead of
    the port landing.
    """
    (target,) = _resolve_fixtures([fixture])
    port = PortRunner()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target.build(root)
        try:
            port.run(list(target.argv), cwd=root)
        except GymratError as error:
            console.print("port side unavailable:", style="red")
            _emit(str(error))
            raise typer.Exit(code=1) from error

    _emit("port side returned unexpectedly; oracle-vs-port diff is not implemented yet")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
