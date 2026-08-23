"""Termination-cleanup registry for graceful shutdown on POSIX signals.

Callers register a zero-argument cleanup with :func:`install_termination_cleanup`
and receive an uninstall callable. When the process receives ``SIGINT``,
``SIGTERM``, or ``SIGHUP``, every active cleanup runs in install order and the
process then exits with ``128 + signal_number`` — the shell convention for
"terminated by signal N".

The handler is installed once per signal for the lifetime of the process and is
deliberately never restored. Python allows exactly one handler per signal, so a
single module-level handler owns each termination signal and consults the live
registry every time it fires; installing and uninstalling cleanups only mutates
that registry, never the signal disposition.
"""

import os
import signal
import warnings
from collections.abc import Callable
from types import FrameType
from typing import NoReturn

# Termination signals gymrat installs cleanup for, in the order the handler is
# wired up. SIGHUP is POSIX-only and absent on win32, so each name is resolved
# defensively.
_TERMINATION_SIGNAL_NAMES = ("SIGINT", "SIGTERM", "SIGHUP")

# Live cleanups keyed by an opaque install token. A dict preserves insertion
# order, which the handler relies on to run cleanups in install order; a set
# would not.
_registry: dict[object, Callable[[], None]] = {}

# Signals whose handler is already wired up. The disposition is set once and
# reused, so re-installing must not re-register it.
_installed_signals: set[int] = set()

# True while the handler is draining the registry. A second signal arriving
# mid-drain must exit immediately rather than re-enter the cleanups.
handling = False


def _exit_process(code: int) -> NoReturn:
    """Terminate the process immediately with ``code``.

    Uses ``os._exit`` rather than ``sys.exit``: the cleanups have already run,
    and raising ``SystemExit`` from a signal handler could be swallowed by an
    application ``except`` block, leaving the process alive after a termination
    signal. Tests monkeypatch this seam to observe the code instead of exiting.
    """
    os._exit(code)


def _run_cleanups() -> None:
    """Run every registered cleanup in install order, warning on failures."""
    for cleanup in list(_registry.values()):
        try:
            cleanup()
        except Exception as exc:  # noqa: BLE001 - a failing cleanup must not stop the rest
            warnings.warn(f"termination cleanup failed: {exc}", stacklevel=2)


def _handler(signal_number: int, _frame: FrameType | None) -> None:
    global handling  # noqa: PLW0603 - module-level re-entry guard the handler owns

    if not handling:
        handling = True
        try:
            _run_cleanups()
        finally:
            handling = False

    _exit_process(128 + signal_number)


def _ensure_handlers_installed() -> None:
    for name in _TERMINATION_SIGNAL_NAMES:
        signal_number = getattr(signal, name, None)
        if signal_number is None or signal_number in _installed_signals:
            continue
        signal.signal(signal_number, _handler)
        _installed_signals.add(signal_number)


def install_termination_cleanup(cleanup: Callable[[], None]) -> Callable[[], None]:
    """Register a cleanup to run when the process is terminated by a signal.

    On the first call the module wires a handler onto every termination signal
    the platform defines (``SIGINT``, ``SIGTERM``, and ``SIGHUP`` where
    available). On ``SIGINT``, ``SIGTERM``, or ``SIGHUP`` every active cleanup
    runs in install order and the process exits with ``128 + signal_number``.

    Args:
        cleanup: A zero-argument callable invoked during shutdown. An exception
            it raises is reported as a warning and does not stop the remaining
            cleanups.

    Returns:
        An uninstall callable that removes this cleanup from the registry.
        Calling it more than once is harmless.
    """
    _ensure_handlers_installed()

    token = object()
    _registry[token] = cleanup

    def uninstall() -> None:
        _registry.pop(token, None)

    return uninstall
