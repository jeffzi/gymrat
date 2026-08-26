"""Guard: importing ``gymrat_py`` must not pull in the heavy statistics stack.

The verdict engine depends on ``scipy`` and ``numpy``, both of which cost
hundreds of milliseconds to import. Keeping them out of the
package's import path preserves fast startup for commands that never compute a
verdict, so they must be imported lazily at their point of use, never at package
import. The permutation test's ``numpy`` use lives behind such a lazy import.

The check runs in a fresh interpreter subprocess and asserts *inside* that
subprocess: the test process has its own imports (pytest pulls in a large
dependency tree), so inspecting this process's ``sys.modules`` could never prove
the package itself stayed clean.

The guard extends to the CLI entry module: importing ``gymrat_py.cli.app`` and
rendering ``--help`` must stay just as cheap as importing the package, so the
command bodies (and the heavy statistics stack they pull) are imported lazily at
call time, never when the app is assembled.

The same discipline covers ``claude_agent_sdk``, the supervise driver's backend:
it drags in ``mcp``, ``starlette``, ``uvicorn``, and ``httpx``, so the Claude
driver imports it lazily inside ``start`` — importing ``gymrat_py.supervisor``
(and the CLI) must never pull it in.
"""

import os
import subprocess
import sys


def _child_env() -> dict[str, str]:
    """Child environment with the optimize flag cleared so the child exits explicitly."""
    env = dict(os.environ)
    env.pop("PYTHONOPTIMIZE", None)  # cspell:disable-line
    return env


def test_importing_package_does_not_import_scipy_or_numpy():
    probe = """
import sys
import gymrat_py
import gymrat_py.stats
import gymrat_py.model
import gymrat_py.adapters
import gymrat_py.exec
import gymrat_py.signals
import gymrat_py.sampling
import gymrat_py.targets
import gymrat_py.supervisor
heavy = sorted(
    name
    for name in sys.modules
    if name in {'scipy', 'numpy', 'claude_agent_sdk'}
    or name.startswith(('scipy.', 'numpy.', 'claude_agent_sdk.'))
)
if heavy:
    print(f'package import pulled in heavy modules: {heavy}', file=sys.stderr)
    sys.exit(1)
"""

    result = subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        env=_child_env(),
    )

    assert result.returncode == 0, result.stderr


def test_importing_cli_app_and_rendering_help_does_not_import_scipy_or_numpy():
    probe = """
import sys
from typer.testing import CliRunner
from gymrat_py.cli.app import app
result = CliRunner().invoke(app, ["--help"])
if result.exit_code != 0:
    print(f'--help failed: {result.output}', file=sys.stderr)
    sys.exit(1)
heavy = sorted(
    name
    for name in sys.modules
    if name in {'scipy', 'numpy', 'claude_agent_sdk'}
    or name.startswith(('scipy.', 'numpy.', 'claude_agent_sdk.'))
)
bodies = [name for name in ('gymrat_py.compare', 'gymrat_py.measure') if name in sys.modules]
if heavy:
    print(f'cli app import pulled heavy modules: {heavy}', file=sys.stderr)
    sys.exit(1)
if bodies:
    print(f'cli app import pulled command bodies: {bodies}', file=sys.stderr)
    sys.exit(1)
"""

    result = subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        env=_child_env(),
    )

    assert result.returncode == 0, result.stderr
