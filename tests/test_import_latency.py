"""Guard: importing ``gymrat_py`` must not pull in the heavy statistics stack.

The verdict engine will depend on ``scipy`` and ``statsmodels``, both of which
cost hundreds of milliseconds to import. Keeping them out of the package's
import path preserves fast startup for commands that never compute a verdict, so
they must be imported lazily at their point of use, never at package import.

The check runs in a fresh interpreter subprocess and asserts *inside* that
subprocess: the test process has its own imports (pytest pulls in a large
dependency tree), so inspecting this process's ``sys.modules`` could never prove
the package itself stayed clean.

Extension point: when v0.8 adds the CLI entry point, extend this guard to import
that entry module too — invoking ``--help`` must stay just as cheap as importing
the package.
"""

import subprocess
import sys


def test_importing_package_does_not_import_scipy_or_statsmodels():
    # Assert inside the child so the test process's own dependency tree cannot
    # mask a violation, and so a non-zero exit surfaces the offending modules.
    probe = (
        "import sys\n"
        "import gymrat_py\n"
        "heavy = sorted(\n"
        "    name\n"
        "    for name in sys.modules\n"
        "    if name in {'scipy', 'statsmodels'}\n"
        "    or name.startswith(('scipy.', 'statsmodels.'))\n"
        ")\n"
        "assert not heavy, f'package import pulled in heavy modules: {heavy}'\n"
    )

    result = subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
