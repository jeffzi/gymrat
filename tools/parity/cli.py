"""Local parity-harness command-line entry point.

Two commands prove the harness before the port CLI exists:

- ``self-diff`` runs the reference binary on *both* sides of each fixture and
  diffs the two parsed-JSON documents. The emitters are deterministic, so the
  two runs are identical and a green self-diff is the harness proving itself
  before any port output is available to compare against.
- ``compare`` drives the oracle-vs-port comparison through the shared
  :class:`~tools.parity.oracle.Runner` interface: it runs each fixture through
  both the reference binary and the port, then diffs the two JSON documents and
  checks their exit codes agree. A document difference or an exit-code mismatch
  fails the fixture; this is the release gate.

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
    RunResult,
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


@dataclass(frozen=True, slots=True)
class CompareResult:
    """One fixture's oracle-vs-port outcome.

    Attributes:
        name: The fixture's identifier.
        oracle_exit: The reference binary's exit code.
        port_exit: The port CLI's exit code.
        report: The oracle-vs-port document diff, or ``None`` when the two sides
            disagreed on their exit code (no document diff is attempted then) or
            errored identically with no JSON to compare.
    """

    name: str
    oracle_exit: int
    port_exit: int
    report: DiffReport | None

    @property
    def exit_match(self) -> bool:
        """True iff the oracle and port returned the same exit code."""
        return self.oracle_exit == self.port_exit

    @property
    def ok(self) -> bool:
        """True iff exit codes agree and any document diff is green."""
        if not self.exit_match:
            return False
        return self.report is None or self.report.is_green


@dataclass(frozen=True, slots=True)
class CompareOutcome:
    """Aggregate outcome of an oracle-vs-port compare over one or more fixtures."""

    results: tuple[CompareResult, ...]

    @property
    def ok(self) -> bool:
        """True iff every fixture agreed on exit code and diffed green."""
        return all(result.ok for result in self.results)


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


def _build_port_runner() -> Runner:
    """Return a runner for the Python port CLI.

    Isolated behind this module-level indirection so tests can substitute a
    scripted fake in place of spawning a real interpreter, mirroring
    :func:`_build_oracle_runner`.
    """
    return PortRunner()


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


def _parse_document(fixture: Fixture, side: str, result: RunResult) -> object:
    """Parse a runner's stdout as JSON, raising with context on failure."""
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        msg = (
            f"fixture {fixture.name!r} {side} side emitted non-JSON stdout "
            f"(exit {result.exit_code}): {result.stderr.strip()}"
        )
        raise GymratError(msg) from error


def run_oracle_vs_port(oracle: Runner, port: Runner, fixtures: Sequence[Fixture]) -> CompareOutcome:
    """Compare the oracle against the port for each fixture.

    Each fixture is built into a fresh temporary directory, then its argv is run
    through both runners (cwd = that directory). When both sides exit zero the
    two parsed-JSON documents are compared with
    :func:`~tools.parity.differ.diff_json`; volatile worktree fields are
    normalized by the differ. A fixture is green only when the exit codes agree
    and any document diff is green.

    Args:
        oracle: The reference-binary runner.
        port: The port-CLI runner.
        fixtures: The fixtures to run, in order.

    Returns:
        The per-fixture results wrapped in a :class:`CompareOutcome`.
    """
    results: list[CompareResult] = []
    for fixture in fixtures:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture.build(root)
            oracle_result = oracle.run(list(fixture.argv), cwd=root)
            port_result = port.run(list(fixture.argv), cwd=root)
            report: DiffReport | None = None
            if oracle_result.exit_code == 0 and port_result.exit_code == 0:
                report = diff_json(
                    _parse_document(fixture, "oracle", oracle_result),
                    _parse_document(fixture, "port", port_result),
                )
        results.append(
            CompareResult(
                name=fixture.name,
                oracle_exit=oracle_result.exit_code,
                port_exit=port_result.exit_code,
                report=report,
            )
        )
    return CompareOutcome(results=tuple(results))


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


def _render_compare(outcome: CompareOutcome) -> None:
    """Render the per-fixture oracle-vs-port table plus diffs and exit mismatches."""
    table = Table(title="parity oracle-vs-port")
    table.add_column("fixture")
    table.add_column("oracle exit")
    table.add_column("port exit")
    table.add_column("status")
    for result in outcome.results:
        status = Text("GREEN", style="green") if result.ok else Text("RED", style="red")
        table.add_row(result.name, str(result.oracle_exit), str(result.port_exit), status)
    console.print(table)

    for result in outcome.results:
        if not result.exit_match:
            _emit(
                f"{result.name} exit-code mismatch: "
                f"oracle exit {result.oracle_exit} vs port exit {result.port_exit}"
            )
        if result.report is not None:
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
    fixtures: Annotated[
        list[str] | None,
        typer.Argument(help="Fixture names to compare; omit to run the whole matrix."),
    ] = None,
) -> None:
    """Diff the oracle against the port over the selected fixtures.

    Each fixture runs through both the reference binary and the port; a document
    difference or an exit-code mismatch fails the run. This is the release gate.
    """
    selected = _resolve_fixtures(fixtures or [])
    oracle = _build_oracle_runner()
    port = _build_port_runner()
    outcome = run_oracle_vs_port(oracle, port, selected)
    _render_compare(outcome)
    if not outcome.ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
