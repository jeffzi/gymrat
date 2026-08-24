"""Guard: importing ``gymrat_py`` must not pull in the heavy statistics stack.

The verdict engine will depend on ``scipy`` and ``statsmodels``, both of which
cost hundreds of milliseconds to import. Keeping them out of the package's
import path preserves fast startup for commands that never compute a verdict, so
they must be imported lazily at their point of use, never at package import.

The check runs in a fresh interpreter subprocess and asserts *inside* that
subprocess: the test process has its own imports (pytest pulls in a large
dependency tree), so inspecting this process's ``sys.modules`` could never prove
the package itself stayed clean.

The guard extends to the CLI entry module: importing ``gymrat_py.cli.app`` and
rendering ``--help`` must stay just as cheap as importing the package, so the
command bodies (and the heavy statistics stack they pull) are imported lazily at
call time, never when the app is assembled.
"""

import subprocess
import sys


def test_importing_package_does_not_import_scipy_or_statsmodels():
    # Assert inside the child so the test process's own dependency tree cannot
    # mask a violation, and so a non-zero exit surfaces the offending modules.
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
heavy = sorted(
    name
    for name in sys.modules
    if name in {'scipy', 'statsmodels'}
    or name.startswith(('scipy.', 'statsmodels.'))
)
assert not heavy, f'package import pulled in heavy modules: {heavy}'
"""

    result = subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_importing_cli_app_and_rendering_help_does_not_import_scipy_or_statsmodels():
    # Assemble the app and render its help through the runner, then assert inside
    # the child that neither the heavy stack nor the command bodies were loaded:
    # a cheap --help must never reach the statistics-bearing modules.
    probe = """
import sys
from typer.testing import CliRunner
from gymrat_py.cli.app import app
result = CliRunner().invoke(app, ["--help"])
assert result.exit_code == 0, result.output
heavy = sorted(
    name
    for name in sys.modules
    if name in {'scipy', 'statsmodels'} or name.startswith(('scipy.', 'statsmodels.'))
)
bodies = [name for name in ('gymrat_py.compare', 'gymrat_py.measure') if name in sys.modules]
assert not heavy, f'cli app import pulled heavy modules: {heavy}'
assert not bodies, f'cli app import pulled command bodies: {bodies}'
"""

    result = subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
