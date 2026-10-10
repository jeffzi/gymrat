"""Import-isolation probing for the seam tests."""

from __future__ import annotations

import json
import subprocess
import sys

#: Run the given statements, then report every loaded module.
_PROBE = (
    "{statements}\nimport json as _json, sys as _sys\n_json.dump(sorted(_sys.modules), _sys.stdout)"
)


def modules_loaded_after(statements: str) -> frozenset[str]:
    """The modules a fresh interpreter holds after running ``statements``.

    The probe runs in a subprocess because import isolation is a property of a
    clean interpreter. Inside the test session the module under test — and most
    of what it could leak — are already in ``sys.modules`` from collection, so
    an in-process ``sys.modules`` delta is always empty and can never fail.

    Args:
        statements: Python source to run first, written by the test itself.

    Returns:
        Every ``sys.modules`` key the subprocess holds once the statements ran.

    Raises:
        CalledProcessError: When ``statements`` raise in the subprocess. The
            subprocess's stderr is attached as a note, so the failing seam test
            shows the traceback that caused it.
    """
    try:
        probe = subprocess.run(  # noqa: S603 -- fixed interpreter; the script is test-written source
            [sys.executable, "-c", _PROBE.format(statements=statements)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        error.add_note(error.stderr)
        raise
    return frozenset(json.loads(probe.stdout))


def modules_imported_by(*modules: str) -> frozenset[str]:
    """The modules a fresh interpreter holds after importing every one of ``modules``.

    Args:
        *modules: Dotted names of the modules to import, in order.

    Returns:
        Every ``sys.modules`` key the subprocess holds once the imports finished.

    Raises:
        CalledProcessError: When the subprocess cannot import one of ``modules``.
    """
    return modules_loaded_after("".join(f"__import__({module!r})\n" for module in modules))


def loaded_under(loaded: frozenset[str], *packages: str) -> list[str]:
    """The names in ``loaded`` that are one of ``packages`` or a submodule of one.

    Args:
        loaded: Module names, as ``modules_loaded_after`` returns them.
        *packages: Dotted package or module names to look for.

    Returns:
        The matching names, sorted.
    """
    prefixes = tuple(f"{package}." for package in packages)
    return sorted(name for name in loaded if name in packages or name.startswith(prefixes))
