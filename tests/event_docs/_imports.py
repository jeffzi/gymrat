"""Import-isolation probing for the event-doc renderers."""

from __future__ import annotations

import json
import subprocess
import sys

#: Import the module named on the command line, then report every loaded module.
_PROBE = "import json, sys; __import__(sys.argv[1]); json.dump(sorted(sys.modules), sys.stdout)"


def modules_imported_by(module: str) -> frozenset[str]:
    """The modules a fresh interpreter holds after importing ``module``.

    The probe runs in a subprocess because import isolation is a property of a
    clean interpreter. Inside the test session the module under test — and most
    of what it could leak — are already in ``sys.modules`` from collection, so
    an in-process ``sys.modules`` delta is always empty and can never fail.

    Args:
        module: Dotted name of the module to import.

    Returns:
        Every ``sys.modules`` key the subprocess holds once the import finished.

    Raises:
        CalledProcessError: When the subprocess cannot import ``module``.
    """
    # S603: the only caller-supplied value is a module name written in a test
    # file, and the interpreter and script are fixed.
    probe = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PROBE, module],
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(json.loads(probe.stdout))
