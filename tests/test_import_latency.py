"""Guard: importing ``gymrat`` must not pull in the heavy statistics stack.

The verdict engine depends on ``scipy`` and ``numpy``, both of which cost
hundreds of milliseconds to import. Keeping them out of the
package's import path preserves fast startup for commands that never compute a
verdict, so they must be imported lazily at their point of use, never at package
import. The permutation test's ``numpy`` use lives behind such a lazy import.

Each check runs in a fresh interpreter subprocess and asserts on the modules
that subprocess holds: the test process has its own imports (pytest pulls in a
large dependency tree), so inspecting this process's ``sys.modules`` could never
prove the package itself stayed clean.

The guard extends to the CLI entry module: importing ``gymrat.__main__`` and
``gymrat.cli.app`` and rendering ``--help`` must stay just as cheap as importing
the package, so the command bodies (comparison, measurement, the telemetry
provider and replay) and the heavy stack they pull are imported lazily at call
time, never when the app is assembled. ``gymrat.scaffold`` loads in the same
probe and must stay off ``tomli_w``, which only the config writer needs.

The same discipline covers ``claude_agent_sdk``, the supervise driver's backend:
it drags in ``mcp``, ``starlette``, ``uvicorn``, and ``httpx``, so the Claude
driver imports it lazily inside ``start`` — importing the supervisor modules
(and the CLI) must never pull it in. The package ``__init__`` imports nothing,
so the package probe names, one by one, the supervisor modules the package used
to import (all but ``events``, which ``hooks`` imports). The modules those import
load with them, and the exit-sequence module loads through the CLI probes, which
import the ``supervise`` command.

``opentelemetry`` ships only with the ``otel`` extra, so modules that load
without it — including the traceparent helpers in ``gymrat.telemetry.provider``
— import it inside the functions that need it.
"""

import subprocess
import sys

from tests._imports import loaded_under, modules_imported_by


def _run_probe(probe: str) -> subprocess.CompletedProcess[str]:
    """Run ``probe`` as a fresh interpreter subprocess and return its result."""
    return subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )


def test_importing_package_when_loaded_does_not_import_heavy_dependencies():
    loaded = modules_imported_by(
        "gymrat",
        "gymrat.stats",
        "gymrat.model",
        "gymrat.verdict",
        "gymrat.adapters",
        "gymrat.exec",
        "gymrat.signals",
        "gymrat.sampling",
        "gymrat.targets",
        "gymrat.supervisor",
        "gymrat.supervisor.claude",
        "gymrat.supervisor.driver",
        "gymrat.supervisor.kickoff",
        "gymrat.supervisor.supervise",
        "gymrat.supervisor.tools",
        "gymrat.supervisor.hooks",
        "gymrat.telemetry",
        "gymrat.telemetry.provider",
        "gymrat.telemetry.run_spans",
    )

    heavy = loaded_under(loaded, "scipy", "numpy", "claude_agent_sdk", "opentelemetry")
    assert heavy == [], f"package import pulled in heavy modules: {heavy}"


def test_main_module_when_rendering_help_does_not_import_heavy_modules_or_command_bodies():
    probe = """
import sys
import gymrat.__main__  # noqa: F401 -- exercise the module's top-level imports
import gymrat.scaffold  # noqa: F401 -- init's writer must stay off tomli_w
from gymrat.cli.app import app
from typer.testing import CliRunner
result = CliRunner().invoke(app, ["--help"])
if result.exit_code != 0:
    print(f'--help failed: {result.output}', file=sys.stderr)
    sys.exit(1)
heavy_roots = ('scipy', 'numpy', 'claude_agent_sdk', 'opentelemetry', 'tomli_w')
heavy = sorted(
    name
    for name in sys.modules
    if name in heavy_roots or name.startswith(tuple(f'{root}.' for root in heavy_roots))
)
bodies = [
    name
    for name in (
        'gymrat.compare',
        'gymrat.measure',
        'gymrat.telemetry.provider',
        'gymrat.telemetry.replay',
    )
    if name in sys.modules
]
if heavy:
    print(f'module entry pulled heavy modules: {heavy}', file=sys.stderr)
    sys.exit(1)
if bodies:
    print(f'module entry pulled command bodies: {bodies}', file=sys.stderr)
    sys.exit(1)
"""

    result = _run_probe(probe)

    assert result.returncode == 0, result.stderr
