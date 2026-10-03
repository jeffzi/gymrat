"""Guard: importing ``gymrat`` must not pull in the heavy statistics stack.

The verdict engine depends on ``scipy`` and ``numpy``, both of which cost
hundreds of milliseconds to import. Keeping them out of the
package's import path preserves fast startup for commands that never compute a
verdict, so they must be imported lazily at their point of use, never at package
import. The permutation test's ``numpy`` use lives behind such a lazy import.

The check runs in a fresh interpreter subprocess and asserts *inside* that
subprocess: the test process has its own imports (pytest pulls in a large
dependency tree), so inspecting this process's ``sys.modules`` could never prove
the package itself stayed clean.

The guard extends to the CLI entry module: importing ``gymrat.cli.app`` and
rendering ``--help`` must stay just as cheap as importing the package, so the
command bodies (and the heavy statistics stack they pull) are imported lazily at
call time, never when the app is assembled.

The same discipline covers ``claude_agent_sdk``, the supervise driver's backend:
it drags in ``mcp``, ``starlette``, ``uvicorn``, and ``httpx``, so the Claude
driver imports it lazily inside ``start`` — importing the supervisor modules
(and the CLI) must never pull it in. The package ``__init__`` imports nothing,
so the package probe names, one by one, the supervisor modules the package used
to import (all but ``events``, which ``hooks`` imports). The modules those import
load with them, and the exit-sequence module loads through the CLI probes, which
import the ``supervise`` command.

``opentelemetry`` ships only with the ``otel`` extra, so modules that load
without it — including the traceparent helpers in ``gymrat.telemetry.ids`` and
``gymrat.telemetry.attributes`` — import it inside the functions that need it.
"""

import subprocess
import sys

from tests._cli import unoptimized_env


def _run_probe(probe: str) -> subprocess.CompletedProcess[str]:
    """Run ``probe`` as a fresh interpreter subprocess and return its result."""
    return subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        env=unoptimized_env(),
    )


def test_importing_package_when_loaded_does_not_import_scipy_or_numpy():
    probe = """
import sys
import gymrat
import gymrat.stats
import gymrat.model
import gymrat.verdict
import gymrat.adapters
import gymrat.exec
import gymrat.signals
import gymrat.sampling
import gymrat.targets
import gymrat.supervisor
import gymrat.supervisor.claude
import gymrat.supervisor.driver
import gymrat.supervisor.kickoff
import gymrat.supervisor.supervise
import gymrat.supervisor.tools
import gymrat.supervisor.hooks
import gymrat.telemetry
import gymrat.telemetry.attributes
import gymrat.telemetry.ids
import gymrat.telemetry.run_spans
heavy = sorted(
    name
    for name in sys.modules
    if name in {'scipy', 'numpy', 'claude_agent_sdk', 'opentelemetry'}
    or name.startswith(('scipy.', 'numpy.', 'claude_agent_sdk.', 'opentelemetry.'))
)
if heavy:
    print(f'package import pulled in heavy modules: {heavy}', file=sys.stderr)
    sys.exit(1)
"""

    result = _run_probe(probe)

    assert result.returncode == 0, result.stderr


def test_importing_cli_app_when_rendering_help_does_not_import_scipy_or_numpy():
    probe = """
import sys
from typer.testing import CliRunner
from gymrat.cli.app import app
result = CliRunner().invoke(app, ["--help"])
if result.exit_code != 0:
    print(f'--help failed: {result.output}', file=sys.stderr)
    sys.exit(1)
heavy = sorted(
    name
    for name in sys.modules
    if name in {'scipy', 'numpy', 'claude_agent_sdk', 'opentelemetry'}
    or name.startswith(('scipy.', 'numpy.', 'claude_agent_sdk.', 'opentelemetry.'))
)
bodies = [name for name in ('gymrat.compare', 'gymrat.measure') if name in sys.modules]
if heavy:
    print(f'cli app import pulled heavy modules: {heavy}', file=sys.stderr)
    sys.exit(1)
if bodies:
    print(f'cli app import pulled command bodies: {bodies}', file=sys.stderr)
    sys.exit(1)
"""

    result = _run_probe(probe)

    assert result.returncode == 0, result.stderr


def test_main_module_when_rendering_help_does_not_import_scipy_or_numpy():
    probe = """
import sys
import gymrat.__main__  # noqa: F401 -- exercise the module's top-level imports
from gymrat.cli.app import app
from typer.testing import CliRunner
result = CliRunner().invoke(app, ["--help"])
if result.exit_code != 0:
    print(f'--help failed: {result.output}', file=sys.stderr)
    sys.exit(1)
heavy = sorted(
    name
    for name in sys.modules
    if name in {'scipy', 'numpy', 'claude_agent_sdk', 'opentelemetry'}
    or name.startswith(('scipy.', 'numpy.', 'claude_agent_sdk.', 'opentelemetry.'))
)
bodies = [name for name in ('gymrat.compare', 'gymrat.measure') if name in sys.modules]
if heavy:
    print(f'module entry pulled heavy modules: {heavy}', file=sys.stderr)
    sys.exit(1)
if bodies:
    print(f'module entry pulled command bodies: {bodies}', file=sys.stderr)
    sys.exit(1)
"""

    result = _run_probe(probe)

    assert result.returncode == 0, result.stderr
