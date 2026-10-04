"""Import-isolation probing for the seam tests."""

from __future__ import annotations

import json
import subprocess
import sys

#: Import each module named on the command line, then report every loaded module.
_PROBE = (
    "import json, sys\n"
    "for name in sys.argv[1:]:\n"
    "    __import__(name)\n"
    "json.dump(sorted(sys.modules), sys.stdout)"
)


def modules_imported_by(*modules: str) -> frozenset[str]:
    """The modules a fresh interpreter holds after importing every one of ``modules``.

    The probe runs in a subprocess because import isolation is a property of a
    clean interpreter. Inside the test session the module under test — and most
    of what it could leak — are already in ``sys.modules`` from collection, so
    an in-process ``sys.modules`` delta is always empty and can never fail.

    Args:
        *modules: Dotted names of the modules to import, in order.

    Returns:
        Every ``sys.modules`` key the subprocess holds once the imports finished.

    Raises:
        CalledProcessError: When the subprocess cannot import one of ``modules``.
    """
    # S603: the only caller-supplied values are module names written in a test
    # file, and the interpreter and script are fixed.
    probe = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PROBE, *modules],
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(json.loads(probe.stdout))


def loaded_under(loaded: frozenset[str], *packages: str) -> list[str]:
    """The names in ``loaded`` that are one of ``packages`` or a submodule of one.

    Args:
        loaded: Module names, as ``modules_imported_by`` returns them.
        *packages: Dotted package or module names to look for.

    Returns:
        The matching names, sorted.
    """
    prefixes = tuple(f"{package}." for package in packages)
    return sorted(name for name in loaded if name in packages or name.startswith(prefixes))
